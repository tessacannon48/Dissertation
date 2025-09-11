# test.py

# Import packages
import torch
import os
import yaml
import warnings
import json
import glob
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import wandb 

warnings.simplefilter(action='ignore', category=FutureWarning)

# Import modules from your library
from src.utils.argparse import parse_arguments
from src.data.dataset import LidarS2Dataset
from src.data.processing import compute_s2_mean_std_multi
from src.model.unet import ConditionalUNet
from src.diffusion.scheduler import LinearDiffusionScheduler, CosineDiffusionScheduler
from scripts.main import run_reconstruction_evaluation, set_seed, train_model

def run_sampling_experiment(config):
    """
    Runs an evaluation experiment with a pre-trained model using different samplers.
    """
    
    # Set device
    device = torch.device(config['system']['device'])
    
    # Set noise scheduler
    if config["training"]["noise_schedule"] == "linear":
        scheduler = LinearDiffusionScheduler(timesteps=config["training"]["timesteps"], device=device)
    else:
        scheduler = CosineDiffusionScheduler(timesteps=config["training"]["timesteps"], device=device)
        
    # Define model path
    model_path = config["evaluation"]["pretrained_model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
    # Load model and config from checkpoint
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Re-initialize the model using the config from the checkpoint
    model_config = checkpoint['config']['model']
    training_config = checkpoint['config']['training']
    
    model = ConditionalUNet(
        in_channels=1,
        cond_channels=4 * training_config["context_k"],
        attr_dim=8 * training_config["context_k"],
        base_channels=model_config["base_channels"],
        embed_dim=model_config["embed_dim"],
        unet_depth=model_config["unet_depth"],
        attention_variant=model_config["attention_variant"]
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f})")
    
    # Set model to evaluation mode
    model.eval()
    
    # Load or calculate dataset statistics
    s2_stats_path = os.path.join(config["data"]["s2_dir"], "s2_stats_24.pt")
    if os.path.exists(s2_stats_path):
        stats = torch.load(s2_stats_path)
        s2_means = stats["mean"]
        s2_stds = stats["std"]
    else:
        # Load all patch IDs and their regions
        all_patch_ids = sorted([os.path.basename(p).split('_')[-1].split('.')[0] for p in glob.glob(os.path.join(config["data"]["s2_dir"], "s2_patch_*")) if os.path.isdir(p)])
        train_pids = []
        for pid in all_patch_ids:
            region_path = os.path.join(config["data"]["s2_dir"], f"s2_patch_{pid}", "region.json")
            if os.path.exists(region_path):
                with open(region_path, 'r') as f:
                    region_data = json.load(f)
                    region_id = region_data.get("region_id", -1)
                    if region_id != 4:
                        train_pids.append(pid)
        train_s2_dirs = [os.path.join(config["data"]["s2_dir"], f"s2_patch_{pid}") for pid in train_pids]
        print("Calculating S2 means and stds on training data...")
        s2_means, s2_stds = compute_s2_mean_std_multi(
            s2_root=config["data"]["s2_dir"],
            num_times=6,
            num_bands=4,
            patch_group_dirs=train_s2_dirs 
        )
        torch.save({"mean": s2_means, "std": s2_stds}, s2_stats_path)
        
    # Recreate the validation dataset
    all_pids = sorted([os.path.basename(p).split('_')[-1].split('.')[0] for p in glob.glob(os.path.join(config["data"]["s2_dir"], "s2_patch_*")) if os.path.isdir(p)])
    val_pids = []
    for pid in all_pids:
        region_path = os.path.join(config["data"]["s2_dir"], f"s2_patch_{pid}", "region.json")
        if os.path.exists(region_path):
            with open(region_path, 'r') as f:
                region_data = json.load(f)
                region_id = region_data.get("region_id", -1)
                if region_id == 4:
                    val_pids.append(pid)
    
    val_dataset = LidarS2Dataset(
        lidar_dir=config["data"]["lidar_dir"],
        s2_dir=config["data"]["s2_dir"],
        s2_means=s2_means,
        s2_stds=s2_stds,
        context_k=training_config["context_k"],
        randomize_context=training_config["randomize_context"],
        augment=False, 
        debug=config["system"]["debug"],
        split_pids=val_pids,
        split="val"
    )

    # Run the reconstruction evaluation function from main.py
    run_reconstruction_evaluation(model, val_dataset, config, scheduler)

if __name__ == "__main__":
    # Set global seed for reproducibility
    seed = 42
    set_seed(seed)
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load the base configuration from the YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override YAML values with command-line arguments
    if args.s2_dir: config['data']['s2_dir'] = args.s2_dir
    if args.lidar_dir: config['data']['lidar_dir'] = args.lidar_dir
    if args.wandb_project: config['logging']['wandb_project'] = args.wandb_project
    if args.wandb_name: config['logging']['wandb_name'] = args.wandb_name
    if args.run_name: config['logging']['run_name'] = args.run_name
    if args.output_dir: config['logging']['output_dir'] = args.output_dir
    if args.sampling_methods: config['evaluation']['sampling_methods'] = args.sampling_methods
    if args.eval_index_json: config['evaluation']['eval_index_json'] = args.eval_index_json
    if args.device: config['system']['device'] = args.device
    if args.debug: config['system']['debug'] = True
    
    # Add a new argument for the pretrained model path
    if args.pretrained_model_path:
        config['evaluation']['pretrained_model_path'] = args.pretrained_model_path
    else:
        raise ValueError("Please provide the path to the pretrained model using --pretrained_model_path")
    
    # Set training parameters for evaluation consistency
    config['training']['context_k'] = 1 
    config['training']['randomize_context'] = False 
    config['training']['timesteps'] = 1000 
    config['training']['noise_schedule'] = "cosine"

    # Auto-detect device if not specified
    device = config['system']['device']
    if device == 'auto':
        if torch.cuda.is_available():
            device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
        else:
            device = "cpu"
    config['system']['device'] = device
    
    # Load original config from the model checkpoint to ensure consistency
    checkpoint = torch.load(config['evaluation']['pretrained_model_path'], map_location=device)
    original_config = checkpoint['config']
    
    # Update current config with original values for consistency
    config['training']['context_k'] = original_config['training']['context_k']
    config['training']['randomize_context'] = original_config['training']['randomize_context']
    config['training']['timesteps'] = original_config['training']['timesteps']
    config['training']['noise_schedule'] = original_config['training']['noise_schedule']
    config['model'] = original_config['model']

    print("\n" + "="*50)
    print("SAMPLING EXPERIMENT CONFIGURATION")
    print(f"Pre-trained Model Path: {config['evaluation']['pretrained_model_path']}")
    print(f"Sampling Methods to Test: {', '.join(config['evaluation']['sampling_methods'])}")
    print("="*50)
    
    # Initialize wandb with run name
    if not config["logging"]["wandb_name"]:
        attention_flag = "att" if config["model"]["attention_variant"] != "none" else "noatt"
        debug_suffix = "debug" if config["system"]["debug"] else ""
        wandb_name = f"{config['logging']['run_name']}_eval_k{config['training']['context_k']}_{attention_flag}{f'_{debug_suffix}' if debug_suffix else ''}"
        config["logging"]["wandb_name"] = wandb_name
    
    wandb.init(
        project=config["logging"]["wandb_project"],
        name=config["logging"]["wandb_name"],
        config=config,
    )

    wandb.log(config)
    
    run_sampling_experiment(config)
    
    wandb.finish()
    print("\nSampling experiment complete!")