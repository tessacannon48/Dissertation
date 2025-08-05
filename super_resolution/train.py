import glob
import random
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import rasterio
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt
import wandb
import os

# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train diffusion model for LiDAR generation')
    
    # Data paths
    parser.add_argument('--s2_dir', type=str, 
                       default='/cs/student/projects2/aisd/2024/tcannon/dissertation/s2_patches',
                       help='Path to S2 patches directory')
    parser.add_argument('--lidar_dir', type=str,
                       default='/cs/student/projects2/aisd/2024/tcannon/dissertation/lidar_patches', 
                       help='Path to LiDAR patches directory')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps (default: 1000)')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=128,
                       help='Base number of channels in U-Net (default: 128)')
    parser.add_argument('--embed_dim', type=int, default=256,
                       help='Embedding dimension for time conditioning (default: 256)')
    
    # Training options
    parser.add_argument('--edm', action='store_true', default=True,
                       help='Use EDM model instead of standard U-Net (default: True)')
    parser.add_argument('--no_edm', dest='edm', action='store_false',
                       help='Use standard U-Net instead of EDM model')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    
    # Logging and saving
    parser.add_argument('--wandb_project', type=str, default='diss',
                       help='Weights & Biases project name (default: diss)')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Weights & Biases run name (default: auto-generated)')
    parser.add_argument('--run_name', type=str, default='default_run',
                       help='Specific run name for labeling outputs (default: default_run)')
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save model checkpoints (default: ./models)')
    parser.add_argument('--output_dir', type=str, default='./reconstructions',
                       help='Directory to save reconstruction images (default: ./reconstructions)')
    
    # Sampling and evaluation
    parser.add_argument('--sampling_methods', type=str, nargs='+', 
                       choices=['ddpm', 'ddim', 'plms', 'edm_euler', 'edm_heun'], 
                       default=['edm_heun'],
                       help='Sampling methods to use for reconstruction (default: edm_heun)')
    parser.add_argument('--evaluate', action='store_true', default=False,
                       help='Run reconstruction evaluation after training')

    
    parser.add_argument('--sigma_data', type=float, default=0.5)
    parser.add_argument('--sigma_min', type=float, default=0.002)  
    parser.add_argument('--sigma_max', type=float, default=80.0)
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto) (default: auto)')
    
    return parser.parse_args()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_s2_mean_std(s2_dir):
    """Compute mean and standard deviation for S2 data."""
    s2_paths = sorted(glob.glob(os.path.join(s2_dir, "s2_patch_*.tif")))
    sum_vals, sum_sq_vals = torch.zeros(4), torch.zeros(4)
    pixel_count = 0
    
    for path in tqdm(s2_paths, desc="Computing S2 stats"):
        with rasterio.open(path) as src:
            arr = torch.from_numpy(src.read().astype(np.float32))
        arr = arr.view(4, -1)
        sum_vals += arr.sum(dim=1)
        sum_sq_vals += (arr ** 2).sum(dim=1)
        pixel_count += arr.shape[1]
    
    mean = sum_vals / pixel_count
    std = torch.sqrt(sum_sq_vals / pixel_count - mean ** 2)
    return mean, std


def normalize_batch(batch):
    """Normalize each image in batch to [0, 1] independently."""
    rescaled = []
    for i in range(batch.shape[0]):
        img = batch[i]
        img_rescaled = (img - img.min()) / (img.max() - img.min() + 1e-8)
        rescaled.append(img_rescaled)
    return torch.stack(rescaled)
"""
# Old denormalization function
def denormalize_patch(normalized, min_val, max_val):
    device = normalized.device  # Ensure all tensors are on the same device
    min_val = min_val.to(device)
    max_val = max_val.to(device)
    return normalized * (max_val - min_val) + min_val
"""

def compute_topographic_rmse(gt, pred):
    """Compute RMSE between gradients of GT and prediction (topographic RMSE)."""
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    dx_rmse = F.mse_loss(pred_dx, gt_dx).sqrt()
    dy_rmse = F.mse_loss(pred_dy, gt_dy).sqrt()
    return (dx_rmse + dy_rmse) / 2

def compute_nrmse(gt, pred):
    """Compute normalized RMSE as percentage."""
    mse = np.mean((gt - pred) ** 2)
    rmse = np.sqrt(mse)
    gt_range = np.max(gt) - np.min(gt)
    return (rmse / gt_range) * 100 if gt_range > 0 else 0

def timestep_embedding(timesteps, dim):
    """Create sinusoidal timestep embeddings."""
    device = timesteps.device
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return F.pad(emb, (0, 1, 0, 0)) if dim % 2 else emb


def masked_mse_loss(pred, target, mask):
    """Calculate MSE loss with mask weighting."""
    return ((pred - target) ** 2 * mask.unsqueeze(1)).sum() / mask.sum()


# =============================================================================
# SAMPLING METHODS
# =============================================================================

def p_sample_loop_ddpm(model, scheduler, shape, cond, device):
    """DDPM sampling method."""
    # Start from pure random noise x_t
    x = torch.randn(shape).to(device)

    # Loop through all of the timesteps in the scheduler from T to 1
    for t in reversed(range(scheduler.timesteps)):

        # Create a batch-sized tensor with each element set to the timestep t
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

        # Predict the clean input x_0 given the pure random noise and the timestep t 
        pred_x0 = model(x, cond, t_batch)

        # Get the alphas and betas for each timestep from the diffusion scheduler
        alpha_t = scheduler.alphas[t]
        alpha_cumprod_t = scheduler.alpha_cumprod[t]
        
        # Handle alpha_cumprod_prev
        if hasattr(scheduler, 'alpha_cumprod_prev'):
            alpha_cumprod_prev_t = scheduler.alpha_cumprod_prev[t]
        else:
            alpha_cumprod_prev_t = scheduler.alpha_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        
        beta_t = 1 - alpha_t

        # If we're not at the final time_step, add noise 
        if t > 0:
            noise = torch.randn_like(x)
        # At t=0, we stop adding noise and just return the predicted sample
        else:
            noise = torch.zeros_like(x)

        # Get coefficient for x_0
        coef1 = torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1 - alpha_cumprod_t)
        
        # Get coefficient for x_t
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)

        # Calculate posterior mean of x_t-1 given x_t
        mean = coef1 * pred_x0 + coef2 * x

        # Calculate the variance of x_t-1 given x_t
        var = beta_t * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)
        std = torch.sqrt(var)

        # Produce x_t-1 by sampling from the Gaussian 
        x = mean + std * noise

    # Return the final predicted sample
    return x


def p_sample_loop_ddim(model, scheduler, shape, cond, device, eta=0.0):
    """DDIM sampling method."""
    # Start from pure random noise x_t
    x = torch.randn(shape).to(device)

    # Loop through all of the timesteps in the scheduler from T to 1
    for i, t in enumerate(reversed(range(scheduler.timesteps))):

        # Create a batch-sized tensor with each element set to the timestep t
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

        # Predict the clean input x_0 given the current sample and the timestep t 
        pred_x0 = model(x, cond, t_batch)

        # Get the alphas for each timestep from the diffusion scheduler
        alpha_cumprod_t = scheduler.alpha_cumprod[t]
        
        # Get the previous timestep (or 0 if this is the last step)
        if i < scheduler.timesteps - 1:
            t_prev = list(reversed(range(scheduler.timesteps)))[i + 1]
            alpha_cumprod_prev_t = scheduler.alpha_cumprod[t_prev]
        else:
            alpha_cumprod_prev_t = torch.tensor(1.0)

        # Calculate the predicted noise (epsilon)
        pred_epsilon = (x - torch.sqrt(alpha_cumprod_t) * pred_x0) / torch.sqrt(1 - alpha_cumprod_t)

        # Calculate the direction pointing towards x_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_prev_t - eta**2 * (1 - alpha_cumprod_t / alpha_cumprod_prev_t)) * pred_epsilon

        # If we're not at the final timestep and eta > 0, add stochastic noise
        if i < scheduler.timesteps - 1 and eta > 0:
            noise = torch.randn_like(x)
            sigma_t = eta * torch.sqrt((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_prev_t)
        else:
            noise = torch.zeros_like(x)
            sigma_t = 0

        # Calculate the deterministic part: x_0 prediction scaled by alpha
        x0_part = torch.sqrt(alpha_cumprod_prev_t) * pred_x0

        # Produce x_t-1 using DDIM formula
        x = x0_part + dir_xt + sigma_t * noise

    # Return the final denoised sample
    return x


def p_sample_loop_plms(model, scheduler, shape, cond, device, order=4):
    """PLMS sampling method."""
    # Start from pure random noise x_t
    x = torch.randn(shape).to(device)
    
    # Store previous noise predictions for PLMS
    prev_eps = []

    # Loop through all of the timesteps in the scheduler from T to 1
    for i, t in enumerate(reversed(range(scheduler.timesteps))):

        # Create a batch-sized tensor with each element set to the timestep t
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

        # Predict the clean input x_0 given the current sample and the timestep t 
        pred_x0 = model(x, cond, t_batch)

        # Get the alphas for each timestep from the diffusion scheduler
        alpha_cumprod_t = scheduler.alpha_cumprod[t]
        
        # Get the previous timestep (or 0 if this is the last step)
        if i < scheduler.timesteps - 1:
            t_prev = list(reversed(range(scheduler.timesteps)))[i + 1]
            alpha_cumprod_prev_t = scheduler.alpha_cumprod[t_prev]
        else:
            alpha_cumprod_prev_t = torch.tensor(1.0)

        # Calculate the predicted noise (epsilon)
        pred_epsilon = (x - torch.sqrt(alpha_cumprod_t) * pred_x0) / torch.sqrt(1 - alpha_cumprod_t)
        
        # Add current epsilon to the history
        prev_eps.append(pred_epsilon)
        
        # Keep only the required number of previous predictions
        if len(prev_eps) > order:
            prev_eps = prev_eps[-order:]

        # Calculate PLMS coefficients based on the number of stored predictions
        if len(prev_eps) == 1:
            # First step: use Euler method
            eps = prev_eps[-1]
        elif len(prev_eps) == 2:
            # Second step: use 2nd order Adams-Bashforth
            eps = (3/2) * prev_eps[-1] - (1/2) * prev_eps[-2]
        elif len(prev_eps) == 3:
            # Third step: use 3rd order Adams-Bashforth
            eps = (23/12) * prev_eps[-1] - (16/12) * prev_eps[-2] + (5/12) * prev_eps[-3]
        else:
            # Fourth step and beyond: use 4th order Adams-Bashforth
            eps = (55/24) * prev_eps[-1] - (59/24) * prev_eps[-2] + (37/24) * prev_eps[-3] - (9/24) * prev_eps[-4]

        # Calculate the deterministic part: x_0 prediction scaled by alpha
        x0_part = torch.sqrt(alpha_cumprod_prev_t) * pred_x0

        # Calculate the direction pointing towards x_t using PLMS-corrected epsilon
        dir_xt = torch.sqrt(1 - alpha_cumprod_prev_t) * eps

        # At t=0, we stop adding noise and just return the predicted sample
        if i == scheduler.timesteps - 1:
            noise = torch.zeros_like(x)
        else:
            noise = torch.zeros_like(x)  # PLMS is typically deterministic

        # Produce x_t-1 using PLMS formula
        x = x0_part + dir_xt + noise

    # Return the final denoised sample
    return x


def edm_sample_euler(model, shape, cond, device, num_steps=50):
    """EDM-consistent Euler sampling method."""
    # Create noise schedule matching training
    sigma_max = model.sigma_max
    sigma_min = model.sigma_min
    
    # Generate decreasing sigma schedule
    sigmas = torch.exp(torch.linspace(
        math.log(sigma_max), math.log(sigma_min), num_steps + 1
    )).to(device)
    
    # Start with pure noise
    x = torch.randn(shape, device=device) * sigma_max
    
    for i in range(num_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # Get model scalings
        c_skip, c_out, c_in = model.get_scalings(sigma.view(-1, 1, 1, 1))
        
        # Convert sigma to timestep for model input
        t = ((torch.log(sigma) - math.log(model.sigma_min)) / 
             (math.log(model.sigma_max) - math.log(model.sigma_min)) * 999).long().clamp(0, 999)
        t_batch = t.repeat(x.size(0))
        
        # Get model prediction
        model_input = x * c_in
        model_output = model(model_input, cond, t_batch)
        
        # Convert to denoised prediction
        denoised = c_out * model_output + c_skip * x
        
        # Euler step
        if i < num_steps - 1:
            # Calculate derivative
            d = (x - denoised) / sigma
            # Take Euler step
            dt = sigma_next - sigma
            x = x + d * dt
        else:
            # Final step
            x = denoised
    
    return x


def edm_sample_heun(model, shape, cond, device, num_steps=50):
    """EDM-consistent Heun sampling method."""
    sigma_max = model.sigma_max
    sigma_min = model.sigma_min
    
    sigmas = torch.exp(torch.linspace(
        math.log(sigma_max), math.log(sigma_min), num_steps + 1
    )).to(device)
    
    x = torch.randn(shape, device=device) * sigma_max
    
    for i in range(num_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # First evaluation
        c_skip, c_out, c_in = model.get_scalings(sigma.view(-1, 1, 1, 1))
        t = ((torch.log(sigma) - math.log(model.sigma_min)) / 
             (math.log(model.sigma_max) - math.log(model.sigma_min)) * 999).long().clamp(0, 999)
        t_batch = t.repeat(x.size(0))
        
        model_input = x * c_in
        model_output = model(model_input, cond, t_batch)
        denoised = c_out * model_output + c_skip * x
        d = (x - denoised) / sigma
        
        if i < num_steps - 1:
            dt = sigma_next - sigma
            x_next = x + d * dt
            
            # Second evaluation (Heun's method)
            c_skip_next, c_out_next, c_in_next = model.get_scalings(sigma_next.view(-1, 1, 1, 1))
            t_next = ((torch.log(sigma_next) - math.log(model.sigma_min)) / 
                      (math.log(model.sigma_max) - math.log(model.sigma_min)) * 999).long().clamp(0, 999)
            t_batch_next = t_next.repeat(x.size(0))
            
            model_input_next = x_next * c_in_next
            model_output_next = model(model_input_next, cond, t_batch_next)
            denoised_next = c_out_next * model_output_next + c_skip_next * x_next
            d_next = (x_next - denoised_next) / sigma_next
            
            # Average the derivatives
            d_avg = (d + d_next) / 2
            x = x + d_avg * dt
        else:
            x = denoised
    
    return x

class LidarS2Dataset(Dataset):
    """Dataset for paired LiDAR and Sentinel-2 patches."""

    def __init__(
        self,
        lidar_dir: str,
        s2_dir: str,
        s2_means: torch.Tensor,
        s2_stds: torch.Tensor,
        augment: bool = True
    ):
        super().__init__()
        self.lidar_dir = lidar_dir
        self.s2_dir = s2_dir
        self.s2_means = s2_means
        self.s2_stds = s2_stds
        self.augment = augment

        # Get file pairs
        self.lidar_paths = sorted(
            glob.glob(os.path.join(lidar_dir, "lidar_patch_*.tif"))
        )
        self.s2_paths = sorted(glob.glob(os.path.join(s2_dir, "s2_patch_*.tif")))

        # Create pairs by matching IDs
        self.pairs = []
        for lidar_path in self.lidar_paths:
            patch_id = self._extract_id(lidar_path)
            s2_path = os.path.join(s2_dir, f"s2_patch_{patch_id}.tif")
            if os.path.exists(s2_path):
                self.pairs.append((lidar_path, s2_path))

        print(f"Created {len(self.pairs)} matched pairs")

    def _extract_id(self, path: str) -> str:
        """Extract patch ID from filename."""
        return os.path.basename(path).split("_")[-1].split(".")[0]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        lidar_path, s2_path = self.pairs[idx]

        # Read LiDAR (assuming it has 3 channels - mean, std, mask)
        with rasterio.open(lidar_path) as src:
            lidar_full = torch.from_numpy(src.read().astype(np.float32))  # (3, H, W)

        # Read S2 (take first 4 channels)
        with rasterio.open(s2_path) as src:
            s2 = torch.from_numpy(src.read()[:4].astype(np.float32))  # (4, H, W)

        # Normalize S2 using provided stats
        s2 = (s2 - self.s2_means.view(-1, 1, 1)) / self.s2_stds.view(-1, 1, 1)

        # Extract LiDAR components
        lidar = lidar_full[:2]  # (2, H, W) - mean and std
        mask = lidar_full[2]  # (H, W) - mask

        # Normalize based on range preference
        # Assume normalized data is roughly in [-4, 4] std devs, map to [-1, 1]
        s2 = torch.clamp(s2 / 4, -1, 1)

        # For LiDAR, we have to clip independently by channel
        lidar[:, 0] = torch.clamp(lidar[:, 0], -1.0, 1.0)  # mean channel
        lidar[:, 1] = torch.clamp(lidar[:, 1], 0.0, 1.0)   # std channel

        # Old normalization strategy
        #lidar = (lidar - lidar.mean()) / (lidar.std() + 1e-8)
        #lidar = torch.clamp(lidar / 4, -1, 1)
       
        # Apply augmentations
        if self.augment:
            if random.random() > 0.5:
                lidar = TF.hflip(lidar)
                s2 = TF.hflip(s2)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                lidar = TF.vflip(lidar)
                s2 = TF.vflip(s2)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                lidar = TF.rotate(lidar, angle)
                s2 = TF.rotate(s2, angle)
                # Add channel dimension for mask rotation, then remove it
                mask = TF.rotate(mask.unsqueeze(0), angle).squeeze(0)

        return {
            "s2": s2.float(),      # [4, H, W]
            "lidar": lidar.float(), # [2, H, W]
            "mask": mask.float(),   # [H, W]
        }


# =============================================================================
# ATTENTION AND CONVOLUTION BLOCKS
# =============================================================================

class SelfAttention2D(nn.Module):
    """2D Self-attention block for spatial feature processing."""
    
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).permute(0, 2, 1)    # B, HW, C
        k = self.k(x).flatten(2)                     # B, C, HW
        v = self.v(x).flatten(2).permute(0, 2, 1)    # B, HW, C

        attn = torch.bmm(q, k) * self.scale          # B, HW, HW
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)                     # B, HW, C
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return self.proj(out + x)


class DoubleConv(nn.Module):
    """Double convolution block with time embedding and optional attention."""
    
    def __init__(self, in_channels, out_channels, embed_dim, use_attention=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
        self.attn = SelfAttention2D(out_channels) if use_attention else None

    def forward(self, x, t):
        h = self.double_conv(x)
        t_emb = self.time_embed(t).view(t.shape[0], -1, 1, 1)
        h = h + t_emb
        if self.attn is not None:
            h = self.attn(h)
        return h


class Down(nn.Module):
    """Downsampling block with max pooling."""
    
    def __init__(self, in_channels, out_channels, embed_dim, use_attention=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, embed_dim, use_attention)

    def forward(self, x, t):
        return self.conv(self.pool(x), t)


class Up(nn.Module):
    """Upsampling block with transpose convolution."""
    
    def __init__(self, in_channels, out_channels, embed_dim, use_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, embed_dim, use_attention)

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1), t)


# =============================================================================
# EDM-SPECIFIC BLOCKS
# =============================================================================

class EDMResBlock(nn.Module):
    """Residual block optimized for EDM with FiLM conditioning."""
    
    def __init__(self, in_channels, out_channels, time_embed_dim, use_attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time conditioning via FiLM
        self.time_proj = nn.Linear(time_embed_dim, out_channels * 2)

        # Main convolution path
        self.norm1 = nn.GroupNorm(min(32, in_channels // 4), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(min(32, out_channels // 4), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

        # Attention
        if use_attention:
            self.attention = SelfAttention2D(out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t_emb):
        # Skip connection
        skip = self.skip_conv(x)

        # Main path
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # FiLM conditioning
        t_proj = self.time_proj(t_emb)
        scale, shift = t_proj.chunk(2, dim=1)
        h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        # Residual connection
        h = h + skip

        # Attention
        h = self.attention(h)

        return h


class EDMDoubleConv(nn.Module):
    """EDM version of DoubleConv with FiLM conditioning."""
    
    def __init__(self, in_channels, out_channels, embed_dim, use_attention=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.attn = SelfAttention2D(out_channels) if use_attention else nn.Identity()

    def forward(self, x, t):
        h = self.double_conv(x)
        t_emb = self.time_embed(t)
        # Reshape t_emb to be broadcastable with h
        t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)
        h = h + t_emb
        h = self.attn(h)
        return h


class EDMDown(nn.Module):
    """EDM version of Down with FiLM conditioning."""
    
    def __init__(self, in_channels, out_channels, embed_dim, use_attention=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = EDMDoubleConv(in_channels, out_channels, embed_dim, use_attention)

    def forward(self, x, t):
        return self.conv(self.pool(x), t)


class EDMUp(nn.Module):
    """EDM version of Up with FiLM conditioning."""
    
    def __init__(self, in_channels, out_channels, embed_dim, use_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = EDMDoubleConv(in_channels, out_channels, embed_dim, use_attention)

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1), t)


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class ConditionalUNet(nn.Module):
    """Standard conditional U-Net for diffusion models."""
    
    def __init__(self, in_channels=2, cond_channels=4, base_channels=128, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Encoder
        self.input_conv = DoubleConv(in_channels + cond_channels, base_channels, embed_dim)
        self.down1 = Down(base_channels, base_channels * 2, embed_dim, use_attention=True)
        self.down2 = Down(base_channels * 2, base_channels * 4, embed_dim, use_attention=True)
        self.down3 = Down(base_channels * 4, base_channels * 8, embed_dim, use_attention=False)

        # Bottleneck
        self.bottleneck_conv = DoubleConv(base_channels * 8, base_channels * 8, embed_dim, use_attention=True)

        # Decoder
        self.up3 = Up(base_channels * 8, base_channels * 4, embed_dim, use_attention=False)
        self.up2 = Up(base_channels * 4, base_channels * 2, embed_dim, use_attention=True)
        self.up1 = Up(base_channels * 2, base_channels, embed_dim, use_attention=True)
        self.output_conv = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, cond, t):
        t_emb = timestep_embedding(t, self.embed_dim)
        cond = F.interpolate(cond, size=(52, 52), mode='bilinear', align_corners=False)
        x = torch.cat([x, cond], dim=1)
        
        # Encoder
        x1 = self.input_conv(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        
        # Bottleneck
        x4 = self.bottleneck_conv(x4, t_emb)
        
        # Decoder
        x = self.up3(x4, x3, t_emb)
        x = self.up2(x, x2, t_emb)
        x = self.up1(x, x1, t_emb)
        
        return self.output_conv(x)


class EDMUNet(nn.Module):
    """EDM-optimized U-Net for noise prediction with proper scaling and conditioning."""
    
    def __init__(
        self,
        in_channels=2,
        cond_channels=4,
        base_channels=128,
        embed_dim=512,
        device="cuda",
        sigma_data=0.5,
        sigma_min=0.001,
        sigma_max=80.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
        )

        # Encoder
        self.input_conv = EDMDoubleConv(
            in_channels + cond_channels, base_channels, embed_dim * 2
        )
        self.down1 = EDMDown(
            base_channels, base_channels * 2, embed_dim * 2, use_attention=True
        )
        self.down2 = EDMDown(
            base_channels * 2, base_channels * 4, embed_dim * 2, use_attention=True
        )
        self.down3 = EDMDown(
            base_channels * 4, base_channels * 8, embed_dim * 2, use_attention=False
        )

        # Bottleneck
        self.bottleneck_conv = EDMDoubleConv(
            base_channels * 8, base_channels * 8, embed_dim * 2, use_attention=True
        )

        # Decoder
        self.up3 = EDMUp(
            base_channels * 8, base_channels * 4, embed_dim * 2, use_attention=False
        )
        self.up2 = EDMUp(
            base_channels * 4, base_channels * 2, embed_dim * 2, use_attention=True
        )
        self.up1 = EDMUp(
            base_channels * 2, base_channels, embed_dim * 2, use_attention=True
        )
        self.output_conv = nn.Conv2d(base_channels, in_channels, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize model weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, cond, t):
        t_emb = timestep_embedding(t, self.embed_dim)
        t_emb = self.time_embed(t_emb)

        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, cond], dim=1)
        
        # Encoder
        x1 = self.input_conv(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        
        # Bottleneck
        x4 = self.bottleneck_conv(x4, t_emb)
        
        # Decoder
        x = self.up3(x4, x3, t_emb)
        x = self.up2(x, x2, t_emb)
        x = self.up1(x, x1, t_emb)
        
        return self.output_conv(x)

    def sample_sigma(self, n):
        """Sample noise levels for EDM training."""
        log_sigma = torch.rand(n, device=self.device) * (
            math.log(self.sigma_max) - math.log(self.sigma_min)
        ) + math.log(self.sigma_min)
        return torch.exp(log_sigma)

    def get_scalings(self, sigma):
        """Get EDM scaling factors."""
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
        return c_skip, c_out, c_in

    def get_loss_weight(self, sigma):
        """Get EDM loss weighting."""
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


# =============================================================================
# DIFFUSION SCHEDULER
# =============================================================================

class DiffusionScheduler:
    """Standard DDPM diffusion scheduler."""
    
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process."""
        if noise is None: 
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(config, edm=False):
    """Unified training function for both standard and EDM models."""
    device = torch.device(config["device"])
    scheduler = DiffusionScheduler(timesteps=config["timesteps"], device=device)

    # Initialize wandb with run name
    wandb_name = config.get("wandb_name") or config.get("run_name", "progressive-comparison")
    wandb.init(
        project=config.get("wandb_project", "tessa-to-edm-progressive"),
        name=wandb_name,
        config=config,
    )

    # Load dataset statistics
    s2_mean = torch.tensor([10445.1582, 10331.0732, 10378.2646, 10169.2549], dtype=torch.float32)
    s2_std = torch.tensor([160.6985, 148.0270, 131.9091, 157.6071], dtype=torch.float32)

    # Create dataset
    dataset = LidarS2Dataset(
        config["lidar_dir"],
        config["s2_dir"],
        s2_mean,
        s2_std,
        augment=True,
    )

    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    """
    # Limit to 100 samples for debug
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, range(100))
    val_dataset = Subset(val_dataset, range(100))
    """

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config.get("num_workers", 4)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=config.get("num_workers", 2)
    )

    # Initialize model and optimizer
    if edm:
        model = EDMUNet(
            in_channels=2,
            cond_channels=4,
            base_channels=config["base_channels"],
            embed_dim=config["embed_dim"],
            device=device,
            sigma_data=config["sigma_data"],
            sigma_min=config["sigma_min"],
            sigma_max=config["sigma_max"]
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"] * 0.5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.1
        )
    else:
        model = ConditionalUNet(
            in_channels=2,
            cond_channels=4,
            base_channels=config["base_channels"],
            embed_dim=config["embed_dim"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_losses = []
    best_val_loss = float('inf')
    
    # Create models directory
    os.makedirs(config.get("save_dir", "./models"), exist_ok=True)

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0

        # Training metrics
        total_train_mse = 0
        mean_train_mse = 0
        std_train_mse = 0
        total_train_weighted_mse = 0

        # Training step
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            lidar = batch["lidar"].to(device)
            s2 = batch["s2"].to(device)
            mask = batch["mask"].to(device)
            t = torch.randint(0, config["timesteps"], (lidar.size(0),), device=device).long()

            if edm:
                # EDM training step
                sigma = model.sample_sigma(lidar.size(0)).view(-1, 1, 1, 1)
                noise = torch.randn_like(lidar)
                noisy_lidar = lidar + sigma * noise
                c_skip, c_out, c_in = model.get_scalings(sigma)
                loss_weight = model.get_loss_weight(sigma)
                t_scaled = (
                    (torch.log(sigma.view(-1)) - math.log(model.sigma_min)) / 
                    (math.log(model.sigma_max) - math.log(model.sigma_min)) * 999
                ).long().clamp(0, 999)

                model_input = noisy_lidar * c_in
                model_output = model(model_input, s2, t_scaled)
                denoised = c_out * model_output + c_skip * noisy_lidar

                loss = (
                    (denoised - lidar) ** 2 * loss_weight * mask.unsqueeze(1)
                ).sum() / mask.sum()

                # Metrics for EDM
                total_train_mse += ((denoised - lidar) ** 2).mean().item()
                mean_train_mse += ((denoised[:, 0:1] - lidar[:, 0:1]) ** 2).mean().item()
                std_train_mse += ((denoised[:, 1:2] - lidar[:, 1:2]) ** 2).mean().item()
                total_train_weighted_mse += loss.item()

            else:
                # Standard DDPM training step
                noisy = scheduler.q_sample(lidar, t)
                pred = model(noisy, s2, t)
                loss = masked_mse_loss(pred, lidar, mask)

                # Metrics for DDPM
                total_train_mse += ((pred - lidar) ** 2).mean().item()
                mean_train_mse += ((pred[:, 0:1] - lidar[:, 0:1]) ** 2).mean().item()
                std_train_mse += ((pred[:, 1:2] - lidar[:, 1:2]) ** 2).mean().item()

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
        mean_val_mse = 0
        std_val_mse = 0
        total_val_weighted_mse = 0
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                lidar = batch["lidar"].to(device)
                s2 = batch["s2"].to(device)
                mask = batch["mask"].to(device)
                t = torch.randint(0, config["timesteps"], (lidar.size(0),), device=device).long()

                if edm:
                    # EDM evaluation
                    sigma = model.sample_sigma(lidar.size(0)).view(-1, 1, 1, 1)
                    noise = torch.randn_like(lidar)
                    noisy_lidar = lidar + sigma * noise
                    c_skip, c_out, c_in = model.get_scalings(sigma)
                    loss_weight = model.get_loss_weight(sigma)
                    t_scaled = (
                        (torch.log(sigma.view(-1)) - math.log(model.sigma_min)) / 
                        (math.log(model.sigma_max) - math.log(model.sigma_min)) * 999
                    ).long().clamp(0, 999)

                    model_input = noisy_lidar * c_in
                    model_output = model(model_input, s2, t_scaled)
                    denoised = c_out * model_output + c_skip * noisy_lidar

                    batch_val_loss = (
                        (denoised - lidar) ** 2 * loss_weight * mask.unsqueeze(1)
                    ).sum() / mask.sum()
                    val_loss += batch_val_loss.item()

                    total_val_mse += ((denoised - lidar) ** 2).mean().item()
                    mean_val_mse += ((denoised[:, 0:1] - lidar[:, 0:1]) ** 2).mean().item()
                    std_val_mse += ((denoised[:, 1:2] - lidar[:, 1:2]) ** 2).mean().item()
                    total_val_weighted_mse += (
                        (denoised - lidar) ** 2 * loss_weight * mask.unsqueeze(1)
                    ).sum().item() / mask.sum().item()
                else:
                    # Standard DDPM evaluation
                    noisy = scheduler.q_sample(lidar, t)
                    pred = model(noisy, s2, t)
                    batch_val_loss = masked_mse_loss(pred, lidar, mask)
                    val_loss += batch_val_loss.item()
                    
                    total_val_mse += ((pred - lidar) ** 2).mean().item()
                    mean_val_mse += ((pred[:, 0:1] - lidar[:, 0:1]) ** 2).mean().item()
                    std_val_mse += ((pred[:, 1:2] - lidar[:, 1:2]) ** 2).mean().item()

        avg_val_loss = val_loss / len(val_loader)

        # Log metrics to wandb
        wandb.log({
            "train_objective": avg_loss,
            "val_objective": avg_val_loss,
            "total_train_mse": total_train_mse / len(train_loader),
            "total_val_mse": total_val_mse / len(val_loader),
            "ch1_train_mse": mean_train_mse / len(train_loader),
            "ch2_train_mse": std_train_mse / len(train_loader),
            "ch1_val_mse": mean_val_mse / len(val_loader),
            "ch2_val_mse": std_val_mse / len(val_loader),
            "epoch": epoch
        })

        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")
        
        if edm:
            lr_scheduler.step()

        # Save model checkpoints
        model_name = model.__class__.__name__
        save_dir = config.get("save_dir", "./models")
        
        # Always save the latest model
        latest_path = os.path.join(save_dir, f"{args.run_name}_{model_name}_latest.pth")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
            'config': config
        }
        torch.save(checkpoint, latest_path)
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(save_dir, f"{args.run_name}_{model_name}_best.pth")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with val_loss: {best_val_loss:.4f}")
        
        # Save periodic checkpoints every 50 epochs
        """
        if (epoch + 1) % 50 == 0:
            periodic_path = os.path.join(save_dir, f"{args.run_name}_{model_name}_epoch_{epoch+1}.pth")
            torch.save(checkpoint, periodic_path)
        """

    # Final model path for return
    final_model_path = os.path.join(config.get("save_dir", "./models"), f"{args.run_name}_{model.__class__.__name__}_best.pth")
    
    # Run reconstruction evaluation if requested
    if config.get("evaluate", False):
        
        # Load the best model for evaluation
        print(f"Loading best model from {final_model_path} for evaluation...")
        best_checkpoint = torch.load(final_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Best model loaded (epoch {best_checkpoint['epoch']}, val_loss: {best_checkpoint['val_loss']:.4f})")
        
        run_reconstruction_evaluation(model, val_dataset, config, scheduler if not edm else None)
    
    return {model.__class__.__name__: {"losses": train_losses, "model_path": final_model_path}}

# =============================================================================
# RECONSTRUCTION EVALUATION
# =============================================================================

def run_reconstruction_evaluation(model, val_dataset, config, scheduler=None):
    """Run reconstruction evaluation with specified sampling methods."""
    print("\n" + "="*60)
    print("RUNNING RECONSTRUCTION EVALUATION")
    print("="*60)
    
    # Create output directory
    output_dir = config.get("output_dir", "./reconstructions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to eval
    model.eval()
    
    # Extract sample batch from the validation set (consistent seed for reproducibility)
    torch.manual_seed(42)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    batch = next(iter(val_loader))
    s2 = batch["s2"].to(config["device"])
    lidar = batch["lidar"].to(config["device"])
    
    # Determine if we're using EDM or standard model
    is_edm = hasattr(model, 'get_scalings')
    
    # Define available sampling methods
    all_samplers = {
        "ddpm": lambda m, s, c, d: p_sample_loop_ddpm(m, scheduler, s, c, d) if scheduler else None,
        "ddim": lambda m, s, c, d: p_sample_loop_ddim(m, scheduler, s, c, d) if scheduler else None,
        "plms": lambda m, s, c, d: p_sample_loop_plms(m, scheduler, s, c, d) if scheduler else None,
        "edm_euler": lambda m, s, c, d: edm_sample_euler(m, s, c, d) if is_edm else None,
        "edm_heun": lambda m, s, c, d: edm_sample_heun(m, s, c, d) if is_edm else None,
    }
    
    # Filter samplers based on model type and user selection
    requested_methods = config.get("sampling_methods", ["edm_heun"])
    p_samplers = {}
    
    for method in requested_methods:
        if method in all_samplers and all_samplers[method] is not None:
            # Test if the method is compatible
            if method.startswith('edm') and not is_edm:
                print(f"Warning: {method} not available for non-EDM models, skipping...")
                continue
            elif not method.startswith('edm') and scheduler is None:
                print(f"Warning: {method} requires scheduler, skipping...")
                continue
            p_samplers[method] = all_samplers[method]
        else:
            print(f"Warning: {method} not available, skipping...")
    
    if not p_samplers:
        print("No valid sampling methods available!")
        return
    
    # Get run info
    run_name = config.get("run_name", "default_run")
    model_type = model.__class__.__name__
    
    # Perform evaluation for each sampling method
    for sampler_name, sampler_func in p_samplers.items():
        print(f"\nSampling method: {sampler_name}")
        
        with torch.no_grad():
            # Generate samples
            generated = sampler_func(model, lidar.shape, s2, config["device"])
            
            # Extract raw tensors
            gt_mean = lidar[:, 0:1].cpu()
            gt_std = lidar[:, 1:2].cpu()
            pred_mean = generated[:, 0:1].cpu()
            pred_std = generated[:, 1:2].cpu()

            """
            # Denormalize
            gt_lidar_denorm = denormalize_patch(lidar, lidar_min, lidar_max)
            pred_lidar_denorm = denormalize_patch(generated, lidar_min, lidar_max)

            gt_mean_m = gt_lidar_denorm[:, 0:1]
            gt_std_m = gt_lidar_denorm[:, 1:2]
            pred_mean_m = pred_lidar_denorm[:, 0:1]
            pred_std_m = pred_lidar_denorm[:, 1:2] 

            print(f"gt_mean_m: min={gt_mean_m.min()}, max={gt_mean_m.max()}, mean={gt_mean_m.mean()}")
            print(f"gt_std_m: min={gt_std_m.min()}, max={gt_std_m.max()}, mean={gt_std_m.mean()}")
            """

            # MAE & RMSE in meters
            mae_mean = F.l1_loss(pred_mean, gt_mean).item()
            rmse_mean = F.mse_loss(pred_mean, gt_mean).sqrt().item()
            mae_std = F.l1_loss(pred_std, gt_std).item()
            rmse_std = F.mse_loss(pred_std, gt_std).sqrt().item()

            # Normalized RMSE by mean elevation
            nrmse_mean = rmse_mean / gt_mean.mean().item()
            nrmse_std = rmse_std / gt_std.mean().item()

            # Topographic RMSE
            topo_rmse = compute_topographic_rmse(gt_mean, pred_mean).item()

            # Normalize to [0, 1] for SSIM and visualization
            gt_mean_norm = normalize_batch(gt_mean)
            gt_std_norm = normalize_batch(gt_std)
            pred_mean_norm = normalize_batch(pred_mean)
            pred_std_norm = normalize_batch(pred_std)

            # SSIM
            ssim_mean = ssim(pred_mean_norm, gt_mean_norm, data_range=1.0).item()
            ssim_std = ssim(pred_std_norm, gt_std_norm, data_range=1.0).item()

            # Print results
            print(f"{'Metric':<20} | {'Mean Elevation':^15} | {'Std Elevation':^15}")
            print("-" * 55)
            print(f"{'MAE':<20} | {mae_mean:^15.4f} | {mae_std:^15.4f}")
            print(f"{'RMSE':<20} | {rmse_mean:^15.4f} | {rmse_std:^15.4f}")
            print(f"{'Normalized RMSE (%)':<20} | {nrmse_mean:^15.2f} | {nrmse_std:^15.2f}")
            print(f"{'SSIM':<20} | {ssim_mean:^15.4f} | {ssim_std:^15.4f}")
            print(f"{'Topographic RMSE':<20} | {topo_rmse:^15.4f} | {'-':^15}")

            # Log metrics to wandb if available
            if wandb.run is not None:
                wandb.log({
                    f"{sampler_name}_mae_mean": mae_mean,
                    f"{sampler_name}_mae_std": mae_std,
                    f"{sampler_name}_rmse_mean": rmse_mean,
                    f"{sampler_name}_rmse_std": rmse_std,
                    f"{sampler_name}_nrmse_mean": nrmse_mean,
                    f"{sampler_name}_nrmse_std": nrmse_std,
                    f"{sampler_name}_ssim_mean": ssim_mean,
                    f"{sampler_name}_ssim_std": ssim_std,
                    f"{sampler_name}_topographic_rmse": topo_rmse,
                })

            # Reconstruction Visualization
            s2_vis = normalize_batch(s2[:8, :3].cpu())
            gt_mean_vis = gt_mean_norm[:8]
            gt_std_vis = gt_std_norm[:8]
            pred_mean_vis = pred_mean_norm[:8]
            pred_std_vis = pred_std_norm[:8]

            vis_list = []
            for i in range(8):
                stacked = torch.cat([
                    F.interpolate(s2_vis[i].unsqueeze(0), size=(52, 52), mode="bilinear", align_corners=False).squeeze(0),
                    gt_mean_vis[i].repeat(3, 1, 1),
                    gt_std_vis[i].repeat(3, 1, 1),
                    pred_mean_vis[i].repeat(3, 1, 1),
                    pred_std_vis[i].repeat(3, 1, 1)
                ], dim=1)
                vis_list.append(stacked)

            final_grid = torch.cat(vis_list, dim=2)
            img = final_grid.permute(1, 2, 0).numpy()

            # Add row titles
            row_labels = ["S2 RGB", "GT Mean", "GT Std", "Pred Mean", "Pred Std"]
            n_rows = len(row_labels)
            row_height = img.shape[0] // n_rows

            plt.figure(figsize=(40, 12))
            plt.imshow(img)
            plt.title(f"Reconstruction Samples Using {sampler_name.upper()}", fontsize=20, pad=20)
            plt.axis("off")

            # Add row labels on the left
            for idx, label in enumerate(row_labels):
                y = row_height * idx + row_height // 2
                plt.text(-5, y, label, va='center', ha='right', fontsize=14, 
                        fontweight='bold', color='black', backgroundcolor='white')

            plt.tight_layout()
            
            # Save the figure
            filename = f"{run_name}_{model_type}_{sampler_name}_reconstruction.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()  # Close to save memory
            
            print(f"Reconstruction saved to: {filepath}")
    
    print(f"\nAll reconstructions saved to: {output_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Auto-detect device if not specified
    if args.device == 'auto':
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"Found {device_count} CUDA device(s)")
            device = "cuda" if device_count > 0 else "cpu"
            print(f"Using device: {device}")
        else:
            device = "cpu"
            print("CUDA not available, using CPU")
    else:
        device = args.device
        print(f"Using specified device: {device}")
    
    # Create configuration dictionary from arguments
    config = {
        "s2_dir": args.s2_dir,
        "lidar_dir": args.lidar_dir,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "timesteps": args.timesteps,
        "base_channels": args.base_channels,
        "embed_dim": args.embed_dim,
        "sigma_data": args.sigma_data,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "num_workers": args.num_workers,
        "device": device,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
        "run_name": args.run_name,
        "save_dir": args.save_dir,
        "output_dir": args.output_dir,
        "sampling_methods": args.sampling_methods,
        "evaluate": args.evaluate,
    }
    
    # Print configuration
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Model Type: {'EDM U-Net' if args.edm else 'Standard U-Net'}")
    print(f"Data Paths:")
    print(f"  S2 Directory: {args.s2_dir}")
    print(f"  LiDAR Directory: {args.lidar_dir}")
    print(f"Training Parameters:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"Model Parameters:")
    print(f"  Base Channels: {args.base_channels}")
    print(f"  Embed Dim: {args.embed_dim}")
    print("Sigma Parameters:")
    print(f"  Sigma Data: {args.sigma_data}")
    print(f"  Sigma Min: {args.sigma_min}")
    print(f"  Sigma Max: {args.sigma_max}")
    print(f"System:")
    print(f"  Device: {device}")
    print(f"  Num Workers: {args.num_workers}")
    print(f"Logging:")
    print(f"  W&B Project: {args.wandb_project}")
    print(f"  W&B Run Name: {args.wandb_name or args.run_name}")
    print(f"  Run Label: {args.run_name}")
    print(f"  Save Directory: {args.save_dir}")
    print(f"  Output Directory: {args.output_dir}")
    if args.evaluate:
        print(f"Evaluation:")
        print(f"  Sampling Methods: {', '.join(args.sampling_methods)}")
    print("="*50)
    
    print("\nStarting training...")
    results = train_model(config, edm=args.edm)
    print("\nTraining complete!")
    print(f"Best model saved to: {results[list(results.keys())[0]]['model_path']}")