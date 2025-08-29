# main.py

# Import packages
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
import warnings
import glob
import rasterio
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import modules
from utils.argparse import parse_arguments
from data.dataset import LidarS2Dataset
from data.processing import compute_s2_mean_std_multi
from models.unet import ConditionalUNet
from diffusion.scheduler import LinearDiffusionScheduler, CosineDiffusionScheduler
from diffusion.sampling import p_sample_loop_ddpm, p_sample_loop_ddim, p_sample_loop_plms
from utils.metrics import compute_topographic_rmse, normalize_batch, masked_mse_loss, masked_mae_loss, masked_hybrid_mse_loss, masked_hybrid_mae_loss

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(config):
    """Unified training function with early stopping."""
    
    # Set device
    device = torch.device(config['system']['device'])

    # Set save directory
    save_dir = config["logging"]["save_dir"]

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

    # Map loss function names to their corresponding functions
    loss_functions = {
        'masked_mse_loss': masked_mse_loss,
        'masked_mae_loss': masked_mae_loss,
        'masked_hybrid_mse_loss': masked_hybrid_mse_loss,
        'masked_hybrid_mae_loss': masked_hybrid_mae_loss
    }
    
    # Get loss function and alpha from config
    loss_name = config['training']['loss']['name']
    loss_alpha = config['training']['loss']['alpha']
    
    # Select the loss function to use
    criterion = loss_functions.get(loss_name)
    
    # Load all patch IDs and their regions
    all_patch_ids = sorted([os.path.basename(p).split('_')[-1].split('.')[0] for p in glob.glob(os.path.join(config["data"]["s2_dir"], "s2_patch_*")) if os.path.isdir(p)])
    train_pids = []
    val_pids = []
    
    # Designate region 4 as validation, the rest as train
    print("Separating patches by region...")
    for pid in tqdm(all_patch_ids):
        region_path = os.path.join(config["data"]["s2_dir"], f"s2_patch_{pid}", "region.json")
        if os.path.exists(region_path):
            with open(region_path, 'r') as f:
                region_data = json.load(f)
                region_id = region_data.get("region_id", -1)
                
                if region_id == 4:
                    val_pids.append(pid)
                else:
                    train_pids.append(pid)

    print(f"Number of training patches (Regions 0-3, 5-9): {len(train_pids)}")
    print(f"Number of validation patches (Region 4): {len(val_pids)}")
    
    # Create a list of S2 patch directories for the training set only
    train_s2_dirs = [os.path.join(config["data"]["s2_dir"], f"s2_patch_{pid}") for pid in train_pids]

    # Load or calculate dataset statistics
    s2_stats_path = os.path.join(config["data"]["s2_dir"], "s2_stats_24.pt")

    if os.path.exists(s2_stats_path):
        stats = torch.load(s2_stats_path)
        s2_means = stats["mean"]
        s2_stds = stats["std"]
    else:
        # Calculate means and stds ONLY on the training regions to avoid data leakage
        print("Calculating S2 means and stds on training data...")
        s2_means, s2_stds = compute_s2_mean_std_multi(
            s2_root=config["data"]["s2_dir"],
            num_times=6,
            num_bands=4,
            patch_group_dirs=train_s2_dirs 
        )
        torch.save({"mean": s2_means, "std": s2_stds}, s2_stats_path)

    # Create training and validation datasets using the pre-defined patch IDs
    train_dataset = LidarS2Dataset(
        lidar_dir=config["data"]["lidar_dir"],
        s2_dir=config["data"]["s2_dir"],
        s2_means=s2_means,
        s2_stds=s2_stds,
        context_k=config["training"]["context_k"],
        randomize_context=config["training"]["randomize_context"],
        augment=True,
        debug=config["system"]["debug"],
        split_pids=train_pids,
        split="train"
    )

    val_dataset = LidarS2Dataset(
        lidar_dir=config["data"]["lidar_dir"],
        s2_dir=config["data"]["s2_dir"],
        s2_means=s2_means,
        s2_stds=s2_stds,
        context_k=config["training"]["context_k"],
        randomize_context=config["training"]["randomize_context"],
        augment=False, # No augmentation for validation set
        debug=config["system"]["debug"],
        split_pids=val_pids,
        split="val"
    )
    
    # Set dataset split tags 
    train_dataset.split = "train"
    val_dataset.split = "val"

    # Empty cache
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
        attr_dim=8 * config["training"]["context_k"],
        base_channels=config["model"]["base_channels"],
        embed_dim=config["model"]["embed_dim"],
        unet_depth=config["model"]["unet_depth"],
        attention_variant=config["model"]["attention_variant"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    train_losses = []
    
    # --- Early Stopping Variables ---
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10 # Wait for 10 epochs with no improvement
    max_epochs = config["training"]["epochs"]
    # --------------------------------

    # Create models directory
    os.makedirs(config["logging"]["save_dir"], exist_ok=True)

    # Initialize a list to store epoch durations
    epoch_durations = []

    # Training loop
    for epoch in range(max_epochs): 
        
        model.train()
        epoch_start_time = time.perf_counter()
        epoch_loss = 0

        # Training metrics
        total_train_loss = 0
        total_train_pixel_loss = 0
        total_train_gradient_loss = 0

        # Training step
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
            lidar = batch["lidar"].to(device)
            s2    = batch["s2"].to(device)
            attrs = batch["attrs"].to(device)
            mask  = batch["mask"].to(device)
            t = torch.randint(0, config["training"]["timesteps"], (lidar.size(0),), device=device).long()
            
            # Training step
            noisy = scheduler.q_sample(lidar, t)
            pred = model(noisy, s2, attrs, t)
            
            # Select loss function with or without alpha
            if 'hybrid' in loss_name:
                loss, pixel_loss_component, gradient_loss_component = criterion(pred, lidar, mask, alpha=loss_alpha)
                total_train_pixel_loss += pixel_loss_component.item()
                total_train_gradient_loss += gradient_loss_component.item()
            else:
                loss = criterion(pred, lidar, mask)
                # If not a hybrid loss, the other components are zero
                pixel_loss_component = loss
                gradient_loss_component = torch.tensor(0.0)

            total_train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Evaluation step
        model.eval()
        total_val_loss = 0
        total_val_pixel_loss = 0
        total_val_gradient_loss = 0

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
                
                # Select loss function with or without alpha
                if 'hybrid' in loss_name:
                    batch_val_loss = criterion(pred, lidar, mask, alpha=loss_alpha)
                    pixel_val_component = loss_functions[f"masked_{'mse' if 'mse' in loss_name else 'mae'}_loss"](pred, lidar, mask)
                    gradient_val_component = (batch_val_loss - pixel_val_component) / loss_alpha
                    total_val_pixel_loss += pixel_val_component.item()
                    total_val_gradient_loss += gradient_val_component.item()
                else:
                    batch_val_loss = criterion(pred, lidar, mask)
                
                total_val_loss += batch_val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # End of epoch timing
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_durations.append(epoch_duration)

        # Log metrics to wandb
        log_metrics = {
            f"train_{loss_name}": avg_loss,
            f"val_{loss_name}": avg_val_loss,
            "epoch": epoch
        }
        if 'hybrid' in loss_name:
            log_metrics.update({
                f"train_{'mse' if 'mse' in loss_name else 'mae'}": total_train_pixel_loss / len(train_loader),
                f"train_gradient": total_train_gradient_loss / len(train_loader),
                f"val_{'mse' if 'mse' in loss_name else 'mae'}": total_val_pixel_loss / len(val_loader),
                f"val_gradient": total_val_gradient_loss / len(val_loader)
            })
        wandb.log(log_metrics)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save the best model
            best_path = os.path.join(save_dir, f"{wandb_name}_best.pth")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }
            torch.save(checkpoint, best_path)
            print(f"New best model saved with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    average_time = sum(epoch_durations) / len(epoch_durations)
    if wandb.run:
        wandb.log({"average_training_time_sec_per_epoch": average_time})
    print(f"Average training time per epoch: {average_time:.2f} seconds")
    
    # Final model path for return.
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
    """
    Run reconstruction evaluation with S2 selection and error map visualization, including comprehensive metric logging and statistical data saving.
    """
    print("\n" + "="*60)
    print("RUNNING RECONSTRUCTION EVALUATION")
    print("="*60)

    # Set split for augmentation control
    val_dataset.split = "val"

    # Define output directories
    output_dir = config["logging"]["output_dir"]
    stats_dir = os.path.join(output_dir, "reconstruction_statistics")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Set model to eval
    model.eval()
    
    # Access hand-picked validation sets by ID
    eval_pids = ["01851", "01857", "01887", "01924", "01959", "02015", "02035", "02047"]

    # Find the indices of these patches in the validation dataset
    eval_indices = [i for i, sample in enumerate(val_dataset.samples) if sample['tile_id'] in eval_pids]

    # Handle the case where not all patches are found
    if len(eval_indices) != len(eval_pids):
        print(f"Warning: Found {len(eval_indices)} out of 8 requested evaluation patches. Continuing with found patches.")
    
    # Create a Subset of the validation dataset using the found indices
    eval_subset = Subset(val_dataset, eval_indices)

    # Create eval loader
    eval_loader = DataLoader(eval_subset, batch_size=len(eval_subset), shuffle=False)
    
    # Access the single batch from the evaluation loader
    batch = next(iter(eval_loader))
    s2 = batch["s2"].to(config["system"]["device"])
    lidar = batch["lidar"].to(config["system"]["device"])
    attrs = batch["attrs"].to(config["system"]["device"])
    mask = batch["mask"].to(config["system"]["device"])
    chosen_ids_batch = batch["chosen_ids"]
    tile_ids_batch = batch["tile_id"] # Retrieve tile IDs for naming saved files

    # Get batch dimensions and context information
    B = lidar.size(0)
    context_k = config["training"]["context_k"]
    run_name = config["logging"]["run_name"]

    # Get sampling methods
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

    # Determine which sentinel-2 patches were used to condition the model
    used_patch_ids = chosen_ids_batch

    # Iterate over each sampling method
    for sampler_name, sampler_func in p_samplers.items():
        print(f"\nSampling method: {sampler_name}")

        # Sample from the model
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            pred = sampler_func(model, lidar.shape, s2, attrs, config["system"]["device"])
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            sampling_time = end_time - start_time

            # Log sampling time
            if wandb.run:
                wandb.log({f"{sampler_name}_sampling_time_sec": sampling_time})

            # Move tensors to CPU
            gt = lidar.cpu()
            pred = pred.cpu()

            # Calculate and log core reconstruction metrics for the entire batch
            mae_tile = F.l1_loss(pred, gt, reduction='none').mean(dim=(1, 2, 3)).tolist()
            rmse_tile = ((pred - gt) ** 2).mean(dim=(1, 2, 3)).sqrt().tolist()
            ssim_tile = [ssim(normalize_batch(pred[i:i+1]), normalize_batch(gt[i:i+1]), data_range=1.0).item() for i in range(B)]
            topo_tile = [compute_topographic_rmse(gt[i:i+1], pred[i:i+1]).item() for i in range(B)]
            rough_tile = [abs(torch.std(gt[i]) - torch.std(pred[i])).item() for i in range(B)]

            # Calculate averages for logging
            mae = float(np.mean(mae_tile))
            rmse = float(np.mean(rmse_tile))
            ssim_avg = float(np.mean(ssim_tile))
            topo_avg = float(np.mean(topo_tile))
            rough_avg = float(np.mean(rough_tile))

            # Log metrics
            if wandb.run:
                wandb.log({
                    f"{sampler_name}_mae": mae,
                    f"{sampler_name}_rmse": rmse,
                    f"{sampler_name}_ssim": ssim_avg,
                    f"{sampler_name}_topographic_rmse": topo_avg,
                    f"{sampler_name}_roughness": rough_avg,
                })

            # Save reconstruction statistics for each sample
            for i in range(B):
                tile_id = tile_ids_batch[i]
                reco_stats = {
                    "tile_id": tile_id,
                    "mae": mae_tile[i],
                    "rmse": rmse_tile[i],
                    "ssim": ssim_tile[i],
                    "topographic_rmse": topo_tile[i],
                    "roughness": rough_tile[i],
                    "gt_min_val": float(gt[i].min()),
                    "gt_max_val": float(gt[i].max()),
                    "gt_mean_val": float(gt[i].mean()),
                    "gt_std_val": float(gt[i].std()),
                    "pred_min_val": float(pred[i].min()),
                    "pred_max_val": float(pred[i].max()),
                    "pred_mean_val": float(pred[i].mean()),
                    "pred_std_val": float(pred[i].std()),
                    "gt_mode_val": float(torch.mode(gt[i].flatten().to(torch.float32))[0]),
                    "pred_mode_val": float(torch.mode(pred[i].flatten().to(torch.float32))[0]),
                }
                
                debug_flag = "_debug_" if config["system"]["debug"] else ""
                stats_path = os.path.join(stats_dir, f"{tile_id}{debug_flag}{run_name}_{sampler_name}_stats.json")
                with open(stats_path, "w") as f:
                    json.dump(reco_stats, f, indent=4)
                print(f"Saved stats for tile {tile_id} to {stats_path}")

            # Visualization normalization + error calculations
            gt_norm = normalize_batch(gt)
            pred_norm = normalize_batch(pred)
            abs_error = (gt - pred).abs()
            err_max = abs_error.amax(dim=(1,2,3), keepdim=True)
            err_norm = abs_error / (err_max + 1e-8)

            # Repeat grayscale GT, Pred, and Error tensors to have 3 channels for visualization
            gt_rgb = gt_norm.repeat(1, 3, 1, 1)
            pred_rgb = pred_norm.repeat(1, 3, 1, 1)
            err_rgb = err_norm.repeat(1, 3, 1, 1)

            # Rebuild the full S2 RGB stacks for visualization
            s2_processed_selected = []
            for i in range(B):
                tile_id = tile_ids_batch[i]
                s2_group_dir = os.path.join(val_dataset.s2_dir, f"s2_patch_{tile_id}")
                
                # Retrieve the specific indices used for conditioning for this sample
                chosen_ids = chosen_ids_batch[i].tolist()
                
                processed_sample = []
                for t_id in chosen_ids:
                    s2_path = os.path.join(s2_group_dir, f"t{t_id}.tif")
                    with rasterio.open(s2_path) as src:
                        arr = torch.from_numpy(src.read()[:4].astype(np.float32))
                    
                    # Get RGB for visualization
                    rgb = arr[[0, 1, 2], :, :]
                    rgb = normalize_batch(rgb.unsqueeze(0)).squeeze(0)
                    rgb = F.interpolate(rgb.unsqueeze(0), size=gt.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
                    processed_sample.append(rgb)
                s2_processed_selected.append(processed_sample)
            
            # Create visualization plot
            tiles = []
            for i in range(B):
                # Stack only the USED S2 times, then GT, Pred, Error
                rows = [p for p in s2_processed_selected[i]] + [gt_rgb[i], pred_rgb[i], err_rgb[i]]
                tile = torch.cat(rows, dim=1)
                tiles.append(tile)

            full_image = torch.cat(tiles, dim=2)
            img_np = full_image.permute(1, 2, 0).numpy()

            # Adjust the figure height based on the number of S2 patches
            fig_h = (context_k * 4) + 12
            fig, ax = plt.subplots(figsize=(42, fig_h))
            ax.axis("off")
            ax.imshow(img_np)

            # Adjust the row labels to only show the chosen S2 patches
            row_labels = [f"S2 t{t}" for t in chosen_ids_batch[0].tolist()] + ["GT", "Pred", "|Error|"]
            row_height = img_np.shape[0] // len(row_labels)
            for r, label in enumerate(row_labels):
                y = r * row_height + row_height // 2
                ax.text(-10, y, label, ha="right", va="center", fontsize=13, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            
            tile_width = img_np.shape[1] // B
            for i in range(B):
                x = i * tile_width + 5
                metrics = (f"MAE: {mae_tile[i]:.3f}\n"
                           f"RMSE: {rmse_tile[i]:.3f}\n"
                           f"SSIM: {ssim_tile[i]:.3f}\n"
                           f"Topo: {topo_tile[i]:.3f}\n"
                           f"Rough: {rough_tile[i]:.3f}")
                ax.text(x, img_np.shape[0] + 10, metrics, ha='left', va='top',
                        fontsize=12, bbox=dict(facecolor='white', alpha=0.85, boxstyle='round'))

                # No need to draw red rectangles, since all S2 patches shown are "chosen"
            
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
    if args.loss_name: config['training']['loss']['name'] = args.loss_name
    if args.loss_alpha is not None: config['training']['loss']['alpha'] = args.loss_alpha

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
    print(f" Loss Function: {config['training']['loss']['name']} (alpha={config['training']['loss'].get('alpha', 'N/A')})")
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