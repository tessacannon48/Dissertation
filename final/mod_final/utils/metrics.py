import glob # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from tqdm import tqdm
import os

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_batch(batch):
    """Normalize each image in batch to [0, 1] independently."""
    rescaled = []
    for i in range(batch.shape[0]):
        img = batch[i]
        img_rescaled = (img - img.min()) / (img.max() - img.min() + 1e-8)
        rescaled.append(img_rescaled)
    return torch.stack(rescaled)

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