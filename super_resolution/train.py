
import os
import glob
import random
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

# === CONFIG ===
config = {
    "s2_dir": "/Users/tessacannon/Documents/UCL/Dissertation/super_resolution/s2_patches",
    "lidar_dir": "/Users/tessacannon/Documents/UCL/Dissertation/super_resolution/lidar_patches",
    "epochs": 30,
    "batch_size": 8,
    "lr": 1e-4,
    "timesteps": 1000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb_project": "dissertation",
    "run_notes": "V0 (batch size 8, lr=1e-4)"
}

# === UTILS ===
def compute_s2_mean_std(s2_dir):
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

def normalize(t): return (t - t.min()) / (t.max() - t.min() + 1e-8)

# === DATASET ===
class LidarS2Dataset(Dataset):
    def __init__(self, lidar_dir, s2_dir, s2_means, s2_stds, augment=True):
        self.lidar_paths = sorted(glob.glob(os.path.join(lidar_dir, "lidar_patch_*.tif")))
        self.s2_paths = sorted(glob.glob(os.path.join(s2_dir, "s2_patch_*.tif")))
        self.augment = augment
        self.s2_mean = s2_means
        self.s2_std = s2_stds
        assert len(self.lidar_paths) == len(self.s2_paths)

    def __len__(self): return len(self.lidar_paths)

    def read_tif(self, path):
        with rasterio.open(path) as src:
            return torch.from_numpy(src.read().astype(np.float32))

    def __getitem__(self, idx):
        lidar = self.read_tif(self.lidar_paths[idx])
        s2 = self.read_tif(self.s2_paths[idx])
        s2 = (s2 - self.s2_mean[:, None, None]) / self.s2_std[:, None, None]
        if self.augment:
            if random.random() > 0.5:
                lidar, s2 = TF.hflip(lidar), TF.hflip(s2)
            if random.random() > 0.5:
                lidar, s2 = TF.vflip(lidar), TF.vflip(s2)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                lidar, s2 = TF.rotate(lidar, angle), TF.rotate(s2, angle)
        return {
            "s2": s2.float(),
            "lidar": lidar[:2].float(),
            "mask": lidar[2].float()
        }

# === EMBEDDING ===
def timestep_embedding(timesteps, dim):
    device = timesteps.device
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return F.pad(emb, (0, 1, 0, 0)) if dim % 2 else emb

# === MODEL ===
class SelfAttention2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q, self.k, self.v = nn.Conv2d(channels, channels, 1), nn.Conv2d(channels, channels, 1), nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).permute(0, 2, 1)
        k = self.k(x).flatten(2)
        v = self.v(x).flatten(2).permute(0, 2, 1)
        attn = torch.bmm(q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        return self.proj(out + x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, attn=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch), nn.GELU()
        )
        self.time_embed = nn.Sequential(
            nn.Linear(emb_dim, out_ch), nn.GELU(),
            nn.Linear(out_ch, out_ch)
        )
        self.attn = SelfAttention2D(out_ch) if attn else None

    def forward(self, x, t):
        h = self.double_conv(x)
        t_emb = self.time_embed(t).view(t.shape[0], -1, 1, 1)
        h = h + t_emb
        return self.attn(h) if self.attn else h

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, attn=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, emb_dim, attn)

    def forward(self, x, t): return self.conv(self.pool(x), t)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, attn=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, emb_dim, attn)

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        diffY, diffX = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1), t)

class ConditionalUNet(nn.Module):
    def __init__(self, in_ch=2, cond_ch=4, base_ch=128, emb_dim=256):
        super().__init__()
        self.embed_dim = emb_dim
        self.input_conv = DoubleConv(in_ch + cond_ch, base_ch, emb_dim)
        self.down1 = Down(base_ch, base_ch*2, emb_dim, True)
        self.down2 = Down(base_ch*2, base_ch*4, emb_dim, True)
        self.down3 = Down(base_ch*4, base_ch*8, emb_dim, False)
        self.bottleneck = DoubleConv(base_ch*8, base_ch*8, emb_dim, True)
        self.up3 = Up(base_ch*8, base_ch*4, emb_dim)
        self.up2 = Up(base_ch*4, base_ch*2, emb_dim, True)
        self.up1 = Up(base_ch*2, base_ch, emb_dim, True)
        self.output = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, cond, t):
        t_emb = timestep_embedding(t, self.embed_dim)
        cond = F.interpolate(cond, size=(52, 52), mode='bilinear', align_corners=False)
        x = torch.cat([x, cond], dim=1)
        x1 = self.input_conv(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x4 = self.bottleneck(x4, t_emb)
        x = self.up3(x4, x3, t_emb)
        x = self.up2(x, x2, t_emb)
        x = self.up1(x, x1, t_emb)
        return self.output(x)

class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

def masked_mse_loss(pred, target, mask):
    return ((pred - target) ** 2 * mask.unsqueeze(1)).sum() / mask.sum()

# === TRAINING ===
def main(config):
    wandb.init(project=config["wandb_project"], config=config, notes=config["run_notes"])
    config = wandb.config

    mean, std = compute_s2_mean_std(config["s2_dir"])
    dataset = LidarS2Dataset(config["lidar_dir"], config["s2_dir"], mean, std, augment=True)
    train_size = int(0.9 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = ConditionalUNet().to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = DiffusionScheduler(timesteps=config["timesteps"], device=config["device"])

    for epoch in range(config["epochs"]):
        # === Training ===
        model.train()
        train_mse, train_mae = 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            lidar = batch["lidar"].to(config["device"])
            s2 = batch["s2"].to(config["device"])
            mask = batch["mask"].to(config["device"])

            t = torch.randint(0, config["timesteps"], (lidar.size(0),), device=config["device"]).long()
            noise = torch.randn_like(lidar)
            noisy_lidar = scheduler.q_sample(lidar, t, noise)
            pred_x0 = model(noisy_lidar, s2, t)

            loss = masked_mse_loss(pred_x0, lidar, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse += F.mse_loss(pred_x0, lidar, reduction="sum").item()
            train_mae += F.l1_loss(pred_x0, lidar, reduction="sum").item()

        total_train_pixels = len(train_loader.dataset) * lidar.shape[2] * lidar.shape[3]
        wandb.log({
            "Train MSE": train_mse / total_train_pixels,
            "Train MAE": train_mae / total_train_pixels,
        }, step=epoch)

        # === Evaluation ===
        model.eval()
        test_mse, test_mae = 0.0, 0.0
        lidar_mean_mse = 0.0
        lidar_std_mse = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1} [Eval]"):
                lidar = batch["lidar"].to(config["device"])
                s2 = batch["s2"].to(config["device"])
                mask = batch["mask"].to(config["device"])

                t = torch.randint(0, config["timesteps"], (lidar.size(0),), device=config["device"]).long()
                noise = torch.randn_like(lidar)
                noisy_lidar = scheduler.q_sample(lidar, t, noise)
                pred_x0 = model(noisy_lidar, s2, t)

                test_mse += F.mse_loss(pred_x0, lidar, reduction="sum").item()
                test_mae += F.l1_loss(pred_x0, lidar, reduction="sum").item()

                # Channel-specific MSEs
                lidar_mean_mse += F.mse_loss(pred_x0[:, 0:1], lidar[:, 0:1], reduction="sum").item()
                lidar_std_mse += F.mse_loss(pred_x0[:, 1:2], lidar[:, 1:2], reduction="sum").item()

        total_test_pixels = len(test_loader.dataset) * lidar.shape[2] * lidar.shape[3]
        wandb.log({
            "Test MSE": test_mse / total_test_pixels,
            "Test MAE": test_mae / total_test_pixels,
            "LiDAR Mean MSE": lidar_mean_mse / total_test_pixels,
            "LiDAR STD MSE": lidar_std_mse / total_test_pixels,
        }, step=epoch)

        print(f"[Epoch {epoch+1}] Train MSE: {train_mse / total_train_pixels:.4f}, Test MSE: {test_mse / total_test_pixels:.4f}")
        print(f"Train MAE: {train_mae / total_train_pixels:.4f}, Test MAE: {test_mae / total_test_pixels:.4f}")


if __name__ == "__main__":
    main(config)
