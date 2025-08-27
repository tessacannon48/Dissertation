# main.py

# Import packages
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt
import wandb
import os
import json
from torch.utils.data import Subset
import time
import yaml
import random


# Import modules
from utils.argparse import parse_arguments
from data.dataset import LidarS2Dataset
from data.processing import compute_s2_mean_std_multi
from models.unet import ConditionalUNet
from diffusion.scheduler import LinearDiffusionScheduler, CosineDiffusionScheduler
from diffusion.sampling import p_sample_loop_ddpm, p_sample_loop_ddim, p_sample_loop_plms
from utils.metrics import compute_topographic_rmse, normalize_batch, masked_mse_loss

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(config):
    """Unified training function."""
    device = torch.device(config['system']['device'])
    
    # Set noise scheduler
    if config["training"]["noise_schedule"] == "linear":
        scheduler = LinearDiffusionScheduler(timesteps=config["training"]["timesteps"], device=device)
    else:
        scheduler = CosineDiffusionScheduler(timesteps=config["training"]["timesteps"], device=device)

    # Initialize wandb with run name
    if not config["logging"]["wandb_name"]:
        attention_flag = "att" if config["model"]["attention_variant"] != "none" else "noatt"
        debug_suffix = "debug" if config["system"]["debug"] else ""
        wandb_name = f"{config['logging']['run_name']}_k{config['training']['context_k']}_{attention_flag}{f'_{debug_suffix}' if debug_suffix else ''}"
        config["logging"]["wandb_name"] = wandb_name
    
    wandb.init(
        project=config["logging"]["wandb_project"],
        name=config["logging"]["wandb_name"],
        config=config,
    )

    # Load dataset statistics
    s2_stats_path = os.path.join(config["data"]["s2_dir"], "s2_stats_24.pt")
    if os.path.exists(s2_stats_path):
        stats = torch.load(s2_stats_path)
        s2_means = stats["mean"]
        s2_stds = stats["std"]
    else:
        s2_means, s2_stds = compute_s2_mean_std_multi(config["data"]["s2_dir"])
        torch.save({"mean": s2_means, "std": s2_stds}, s2_stats_path)

    # Ensure proper length of S2 means and stds for selected k 
    if s2_means.numel() == 24 and config["training"]["context_k"] < 6:
        s2_means = s2_means[:4 * config["training"]["context_k"]]
        s2_stds  = s2_stds[:4 * config["training"]["context_k"]]

    # Create full dataset (split later)
    dataset = LidarS2Dataset(
        lidar_dir=config["data"]["lidar_dir"],
        s2_dir=config["data"]["s2_dir"],
        s2_means=s2_means,
        s2_stds=s2_stds,
        context_k=config["training"]["context_k"],
        randomize_context=config["training"]["randomize_context"],
        augment=True,
        debug=config["system"]["debug"]
    )

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Set dataset split tags
    train_dataset.dataset.split = "train"
    val_dataset.dataset.split = "val"

    # Load fixed eval index if available
    eval_json = config["evaluation"]["eval_index_json"]
    if eval_json and os.path.exists(eval_json):
        with open(eval_json, "r") as f:
            val_dataset.dataset.eval_patch_order = json.load(f)

    torch.cuda.empty_cache()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"] // 2 if config["training"]["num_workers"] > 1 else 1
    )

    # Initialize model
    model = ConditionalUNet(
        in_channels=1,
        cond_channels=4 * config["training"]["context_k"],
        #attr_dim=8 * config["training"]["context_k"],
        attr_dim=0,
        base_channels=config["model"]["base_channels"],
        embed_dim=config["model"]["embed_dim"],
        unet_depth=config["model"]["unet_depth"],
        attention_variant=config["model"]["attention_variant"]
    ).to(device)


    #params = sum(p.numel() for p in model.parameters())
    #for name, p in model.named_parameters():
        #print(name, p.numel())

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    # Check model parameters
    # for name, param in model.named_parameters():
        # print(name, param.shape)

    train_losses = []
    best_val_loss = float('inf')
    
    # Create models directory
    os.makedirs(config["logging"]["save_dir"], exist_ok=True)
    
    # Add training time tracking
    wandb.define_metric("training_time_sec_per_epoch", summary="min")

    # Initialize a list to store epoch durations
    epoch_durations = []

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        
        model.train()
        epoch_start_time = time.perf_counter()
        epoch_loss = 0

        # Training metrics
        total_train_mse = 0
        total_train_weighted_mse = 0

        # Training step
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
            lidar = batch["lidar"].to(device)
            s2    = batch["s2"].to(device)
            attrs = batch["attrs"].to(device)
            mask  = batch["mask"].to(device)
            t = torch.randint(0, config["training"]["timesteps"], (lidar.size(0),), device=device).long()
            
            # Training step
            noisy = scheduler.q_sample(lidar, t)
            pred = model(noisy, s2, attrs, t)
            loss = masked_mse_loss(pred, lidar, mask)

            # Metrics for DDPM
            total_train_mse += ((pred - lidar) ** 2).mean().item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Evaluation step
        model.eval()
        total_val_mse = 0
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                lidar = batch["lidar"].to(device)
                s2    = batch["s2"].to(device)
                attrs = batch["attrs"].to(device)
                mask  = batch["mask"].to(device)
                t = torch.randint(0, config["training"]["timesteps"], (lidar.size(0),), device=device).long()

                # Evaluation
                noisy = scheduler.q_sample(lidar, t)
                pred  = model(noisy, s2, attrs, t)
                batch_val_loss = masked_mse_loss(pred, lidar, mask)
                val_loss += batch_val_loss.item()
                
                total_val_mse += ((pred - lidar) ** 2).mean().item()

        avg_val_loss = val_loss / len(val_loader)

        # End of epoch timing
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_durations.append(epoch_duration)

        # Log metrics to wandb
        wandb.log({
            "train_objective": avg_loss,
            "val_objective": avg_val_loss,
            "total_train_mse": total_train_mse / len(train_loader),
            "total_val_mse": total_val_mse / len(val_loader),
            "epoch": epoch
        })

        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

    average_time = sum(epoch_durations) / len(epoch_durations)
    if wandb.run:
        wandb.log({"average_training_time_sec_per_epoch": average_time})
    print(f"Average training time per epoch: {average_time:.2f} seconds")
    
    # Save model checkpoints
    save_dir = config["logging"]["save_dir"]
    wandb_name = config["logging"]["wandb_name"]


    # Always save the latest model
    #latest_path = os.path.join(save_dir, f"{wandb_name}_latest.pth")
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_loss,
        'val_loss': avg_val_loss,
        'config': config
    }
    #torch.save(checkpoint, latest_path)

    # Save best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_path = os.path.join(save_dir, f"{wandb_name}_best.pth")
        torch.save(checkpoint, best_path)
        print(f"New best model saved with val_loss: {best_val_loss:.4f}")

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Final model path for return
    final_model_path = os.path.join(save_dir, f"{wandb_name}_best.pth")

    # Run reconstruction evaluation if requested
    if config["evaluation"]["evaluate"]:
        print(f"Loading best model from {final_model_path} for evaluation...")
        best_checkpoint = torch.load(final_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Best model loaded (epoch {best_checkpoint['epoch']}, val_loss: {best_checkpoint['val_loss']:.4f})")
        
        run_reconstruction_evaluation(model, val_dataset, config, scheduler)

    return {model.__class__.__name__: {"losses": train_losses, "model_path": final_model_path}}

# =============================================================================
# RECONSTRUCTION EVALUATION
# =============================================================================

def run_reconstruction_evaluation(model, val_dataset, config, scheduler=None):
    """Run reconstruction evaluation with S2 selection and error map visualization (dynamic k)."""
    print("\n" + "="*60)
    print("RUNNING RECONSTRUCTION EVALUATION")
    print("="*60)

    if hasattr(val_dataset, "dataset"):
        val_dataset.dataset.split = "val"
        eval_json = config["evaluation"]["eval_index_json"]
        if eval_json and os.path.exists(eval_json):
            with open(eval_json, "r") as f:
                val_dataset.dataset.eval_patch_order = json.load(f)

    output_dir = config["logging"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    torch.manual_seed(42)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    batch = next(iter(val_loader))

    s2 = batch["s2"].to(config["system"]["device"])
    lidar = batch["lidar"].to(config["system"]["device"])
    attrs = batch["attrs"].to(config["system"]["device"])
    mask = batch["mask"].to(config["system"]["device"])
    chosen_ids_batch = batch["chosen_ids"] # Retrieve the chosen IDs

    B = lidar.size(0)
    context_k = config["training"]["context_k"]                     
    run_name = config["logging"]["run_name"]
    model_type = model.__class__.__name__

    wandb.define_metric("sampling_time_sec", summary="min")

    all_samplers = {
        "ddpm": lambda m, s, c, a, d: p_sample_loop_ddpm(m, scheduler, s, c, a, d) if scheduler else None,
        "ddim": lambda m, s, c, a, d: p_sample_loop_ddim(m, scheduler, s, c, a, d) if scheduler else None,
        "plms": lambda m, s, c, a, d: p_sample_loop_plms(m, scheduler, s, c, a, d) if scheduler else None,
    }
    requested_methods = config["evaluation"]["sampling_methods"]
    p_samplers = {m: all_samplers[m] for m in requested_methods if m in all_samplers}
    if not p_samplers:
        print("No valid samplers available")
        return

    # Try to retrieve which t{i} were used (optional; may not exist)
    used_patch_ids = chosen_ids_batch # Now it's a direct reference to the correct data

    for sampler_name, sampler_func in p_samplers.items():
        print(f"\nSampling method: {sampler_name}")

        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            pred = sampler_func(model, lidar.shape, s2, attrs, config["system"]["device"])
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            sampling_time = end_time - start_time

            if wandb.run:
                wandb.log({f"{sampler_name}_sampling_time_sec": sampling_time})

            gt = lidar.cpu()
            pred = pred.cpu()
            mask_cpu = mask.cpu()

            gt_norm = normalize_batch(gt)
            pred_norm = normalize_batch(pred)
            abs_error = (gt - pred).abs()

            err_max = abs_error.amax(dim=(1,2,3), keepdim=True)
            err_norm = abs_error / (err_max + 1e-8)
            s2_cpu = s2.cpu()

            mae_tile  = F.l1_loss(pred, gt, reduction='none').mean(dim=(1, 2, 3)).tolist()
            rmse_tile = ((pred - gt) ** 2).mean(dim=(1, 2, 3)).sqrt().tolist()
            ssim_tile = [ssim(pred_norm[i:i+1], gt_norm[i:i+1], data_range=1.0).item() for i in range(B)]
            topo_tile = [compute_topographic_rmse(gt[i:i+1], pred[i:i+1]).item() for i in range(B)]
            rough_tile = [abs(torch.std(gt[i]) - torch.std(pred[i])).item() for i in range(B)]

            mae = float(np.mean(mae_tile))
            rmse = float(np.mean(rmse_tile))
            ssim_avg = float(np.mean(ssim_tile))
            topo_avg = float(np.mean(topo_tile))
            rough_avg = float(np.mean(rough_tile))

            if wandb.run:
                wandb.log({
                    f"{sampler_name}_mae": mae,
                    f"{sampler_name}_rmse": rmse,
                    f"{sampler_name}_ssim": ssim_avg,
                    f"{sampler_name}_topographic_rmse": topo_avg,
                    f"{sampler_name}_roughness": rough_avg,  
                })
            
            # --- START VISUALIZATION LOGIC ---
            gt_rgb   = gt_norm.repeat(1, 3, 1, 1)
            pred_rgb = pred_norm.repeat(1, 3, 1, 1)
            err_rgb  = err_norm.repeat(1, 3, 1, 1)

            # Visualization batch size
            Bvis = min(config["training"]["batch_size"], B)

            # Rebuild the full S2 RGB stacks for visualization
            s2_full_stack = [val_dataset.dataset.data_cache[i]["s2_patches"] for i in range(Bvis)]
            
            s2_full_processed = []
            for sample_s2_stack in s2_full_stack:
                processed_sample = []
                for arr in sample_s2_stack:
                    # pick R,G,B from [base, base+1, base+2]
                    rgb = arr[[0, 1, 2], :, :]
                    rgb = normalize_batch(rgb.unsqueeze(0)).squeeze(0)
                    rgb = F.interpolate(rgb.unsqueeze(0), size=gt.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
                    processed_sample.append(rgb)
                s2_full_processed.append(processed_sample)

            Bvis = min(config["training"]["batch_size"], B)
            tiles = []
            for i in range(Bvis):
                # Stack all 6 S2 times, then GT, Pred, Error
                rows = [p for p in s2_full_processed[i]] + [gt_rgb[i], pred_rgb[i], err_rgb[i]]
                tile = torch.cat(rows, dim=1)
                tiles.append(tile)

            full_image = torch.cat(tiles, dim=2)
            img_np = full_image.permute(1, 2, 0).numpy()

            fig_h = 22 if val_dataset.dataset.max_s2 <= 3 else 26
            fig, ax = plt.subplots(figsize=(42, fig_h))
            ax.axis("off")
            ax.imshow(img_np)

            row_labels = [f"S2 t{t}" for t in range(val_dataset.dataset.max_s2)] + ["GT", "Pred", "|Error|"]
            row_height = img_np.shape[0] // len(row_labels)
            for r, label in enumerate(row_labels):
                y = r * row_height + row_height // 2
                ax.text(-10, y, label, ha="right", va="center", fontsize=13, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

            tile_width = img_np.shape[1] // Bvis
            metrics_y_offset = img_np.shape[0] + 40  
            footer_y_offset = img_np.shape[0] + 180
            for i in range(Bvis):
                x = i * tile_width + 5
                metrics = (f"MAE: {mae_tile[i]:.3f}\n"
                           f"RMSE: {rmse_tile[i]:.3f}\n"
                           f"SSIM: {ssim_tile[i]:.3f}\n"
                           f"Topo: {topo_tile[i]:.3f}\n"
                           f"Rough: {rough_tile[i]:.3f}")  
                ax.text(x, img_np.shape[0] + 10, metrics, ha='left', va='top',
                        fontsize=12, bbox=dict(facecolor='white', alpha=0.85, boxstyle='round'))

                # Highlight the specific t{i} rows that were used for this sample
                chosen = used_patch_ids[i].tolist()
                for r in chosen:
                    if 0 <= r < val_dataset.dataset.max_s2:
                        y0 = r * row_height
                        ax.add_patch(plt.Rectangle(
                            (x, y0), tile_width - 10, row_height,
                            edgecolor='red', linewidth=2, facecolor='none'
                        ))

            footer = (f"Mean → MAE: {mae:.3f} | RMSE: {rmse:.3f} | SSIM: {ssim_avg:.3f} | Topo RMSE: {topo_avg:.3f} | "
                      f"Roughness: {rough_avg:.3f}")  
            ax.text(img_np.shape[1] // 2, img_np.shape[0] + 120, footer,
                    ha='center', fontsize=16, weight='bold', color='darkblue')

            plt.tight_layout()
            out_name = config["logging"]["wandb_name"] or config["logging"]["run_name"]
            out_path = os.path.join(config["logging"]["output_dir"], f"{out_name}_{sampler_name}_vis.png")
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization to {out_path}")

# =============================================================================
# SET GLOBAL SEED
# =============================================================================

def set_seed(seed):
    """Sets the seed for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Set global seed
    seed = 42
    set_seed(seed)

    # Parse command line arguments
    args = parse_arguments()
    
    # Load the base configuration from the YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override YAML values with command-line arguments if provided
    # Note: `action='store_true'` arguments are handled differently
    if args.s2_dir: config['data']['s2_dir'] = args.s2_dir
    if args.lidar_dir: config['data']['lidar_dir'] = args.lidar_dir
    if args.batch_size: config['training']['batch_size'] = args.batch_size
    if args.epochs: config['training']['epochs'] = args.epochs
    if args.lr: config['training']['lr'] = args.lr
    if args.timesteps: config['training']['timesteps'] = args.timesteps
    if args.noise_schedule: config['training']['noise_schedule'] = args.noise_schedule
    if args.num_workers: config['training']['num_workers'] = args.num_workers
    if args.context_k: config['training']['context_k'] = args.context_k
    if args.randomize_context: config['training']['randomize_context'] = True
    if args.wandb_project: config['logging']['wandb_project'] = args.wandb_project
    if args.wandb_name: config['logging']['wandb_name'] = args.wandb_name
    if args.run_name: config['logging']['run_name'] = args.run_name
    if args.save_dir: config['logging']['save_dir'] = args.save_dir
    if args.output_dir: config['logging']['output_dir'] = args.output_dir
    if args.sampling_methods: config['evaluation']['sampling_methods'] = args.sampling_methods
    if args.evaluate: config['evaluation']['evaluate'] = True
    if args.eval_index_json: config['evaluation']['eval_index_json'] = args.eval_index_json
    if args.device: config['system']['device'] = args.device
    if args.debug: config['system']['debug'] = True
    if args.unet_depth: config['model']['unet_depth'] = args.unet_depth
    if args.base_channels: config['model']['base_channels'] = args.base_channels
    if args.embed_dim: config['model']['embed_dim'] = args.embed_dim
    if args.attention_variant: config['model']['attention_variant'] = args.attention_variant

    # Auto-detect device if not specified
    device = config['system']['device']
    if device == 'auto':
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"Found {device_count} CUDA device(s)")
            device = "cuda" if device_count > 0 else "cpu"
            print(f"Using device: {device}")
        else:
            device = "cpu"
            print("CUDA not available, using CPU")
    config['system']['device'] = device
    
    # Print configuration
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print(f"Debug Mode: {config['system']['debug']}")
    print("="*50)
    print(f"Model Type: Standard U-Net")
    print(f"Data Paths:")
    print(f"  S2 Directory: {config['data']['s2_dir']}")
    print(f"  LiDAR Directory: {config['data']['lidar_dir']}")
    print(f"Training Parameters:")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Learning Rate: {config['training']['lr']}")
    print(f"  Timesteps: {config['training']['timesteps']}")
    print(f"  Noise Schedule: {config['training']['noise_schedule']}")
    print(f"Model Parameters:")
    print(f"  Base Channels: {config['model']['base_channels']}")
    print(f"  Embed Dim: {config['model']['embed_dim']}")
    print(f"Context k: {config['training']['context_k']}")
    print(f"Randomize Context: {config['training']['randomize_context']}")
    print(f"Attention Variant: {config['model']['attention_variant']}")
    print(f"Eval Index File: {config['evaluation']['eval_index_json']}")
    print(f"System:")
    print(f"  Device: {config['system']['device']}")
    print(f"  Num Workers: {config['training']['num_workers']}")
    print(f"Logging:")
    print(f"  W&B Project: {config['logging']['wandb_project']}")
    print(f"  W&B Run Name: {config['logging']['wandb_name'] or config['logging']['run_name']}")
    print(f"  Run Label: {config['logging']['run_name']}")
    print(f"  Save Directory: {config['logging']['save_dir']}")
    print(f"  Output Directory: {config['logging']['output_dir']}")
    if config['evaluation']['evaluate']:
        print(f"Evaluation:")
        print(f"  Sampling Methods: {', '.join(config['evaluation']['sampling_methods'])}")
    print("="*50)
    # Empty cache
    torch.cuda.empty_cache()
    print("\nStarting training...")
    
    # Train model
    results = train_model(config)
    
    print("\nTraining complete!")
    print(f"Best model saved to: {results[list(results.keys())[0]]['model_path']}")