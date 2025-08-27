import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# MODEL BLOCKS
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
        q = self.q(x).flatten(2).permute(0, 2, 1)  # B, HW, C
        k = self.k(x).flatten(2)  # B, C, HW
        v = self.v(x).flatten(2).permute(0, 2, 1)  # B, HW, C
        attn = torch.bmm(q, k) * self.scale  # B, HW, HW
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)  # B, HW, C
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return self.proj(out + x)

class DoubleConv(nn.Module):
    """Double conv with *conditioning* embedding (time + attrs combined)."""
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
        self.cond_embed = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
        if use_attention:
            self.attn = SelfAttention2D(out_channels)
        else:
            self.attn = None

    def forward(self, x, cond_vec):
        h = self.double_conv(x)
        c = self.cond_embed(cond_vec).view(cond_vec.size(0), -1, 1, 1)
        h = h + c
        if self.attn is not None:
            h = self.attn(h)
        return h

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, use_attention=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, embed_dim, use_attention)

    def forward(self, x, cond_vec):
        return self.conv(self.pool(x), cond_vec)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, embed_dim, use_attention=False):
        super().__init__()
        # The ConvTranspose2d should halve the input channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # The DoubleConv should take the concatenated channels
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels, embed_dim, use_attention)

    def forward(self, x1, x2, cond_vec):
        x1 = self.up(x1)  # [B, in_channels // 2, H*2, W*2]
        # Match skip connection size
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # [B, (in_channels // 2) + skip_channels, H, W]
        return self.conv(x, cond_vec)