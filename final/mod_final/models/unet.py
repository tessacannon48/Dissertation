import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import DoubleConv, Down, Up, SelfAttention2D
from diffusion.utils import timestep_embedding

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class ConditionalUNet(nn.Module):
    """ U-Net with dynamic depth and configurable attention. """
    def __init__(self, in_channels=1, cond_channels=24, attr_dim=48, base_channels=128, embed_dim=256, unet_depth=4, attention_variant='default'):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = unet_depth
        self.attention_variant = attention_variant
        self.base_channels = base_channels

        # Time + attributes -> single conditioning vector
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        if attr_dim > 0:
            self.attr_mlp = nn.Sequential(
                nn.Linear(attr_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.attr_mlp = None

        # ========== Build U-Net dynamically ==========

        # Input convolution block
        self.input_conv = DoubleConv(in_channels + cond_channels, base_channels, embed_dim)

        # Encoder path with downsampling blocks
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i in range(unet_depth):
            max_channels = base_channels * 8
            out_ch = min(in_ch * 2, max_channels)
            use_attn = self._use_attention(i, stage='down', depth=unet_depth)
            self.downs.append(Down(in_ch, out_ch, embed_dim, use_attention=use_attn))
            in_ch = out_ch

        # Bottleneck convolution block
        self.bottleneck_conv = DoubleConv(in_ch, in_ch, embed_dim, use_attention=True)

        # Decoder path with upsampling blocks
        self.ups = nn.ModuleList()
        for i in range(unet_depth):
            in_ch_prev = in_ch
            # The input channels to the Up block are the output of the previous layer
            # plus the channels from the corresponding skip connection from the encoder
            skip_ch = self._get_skip_channels(unet_depth, i)
            out_ch = skip_ch
            use_attn = self._use_attention(i, stage='up', depth=unet_depth)
            self.ups.append(Up(in_channels=in_ch_prev, skip_channels=skip_ch, out_channels=out_ch, embed_dim=embed_dim, use_attention=use_attn))
            in_ch = out_ch

        # Final output convolution
        self.output_conv = nn.Conv2d(in_ch, in_channels, 1)

    def _get_skip_channels(self, depth, current_up_idx):
        """Helper to compute skip connection channels dynamically."""
        channels = [self.base_channels]
        for i in range(depth):
            max_channels = self.base_channels * 8
            out_ch = min(channels[-1] * 2, max_channels)
            channels.append(out_ch)
        return channels[-(current_up_idx + 2)]

    def _use_attention(self, idx, stage, depth):
        """Determine whether to use attention based on variant and layer idx."""
        if self.attention_variant == 'none':
            return False
        elif self.attention_variant == 'all':
            return True
        elif self.attention_variant == 'mid':
            # Applies attention to the two innermost encoder/decoder layers
            if stage == 'down' and idx == depth - 1:
                return True
            if stage == 'up' and idx == 0:
                return True
            return False
        elif self.attention_variant == 'default':
            return False
        
    def forward(self, x, cond_img, attrs, t):
        """
        The forward pass of the U-Net.

        Args:
            x (torch.Tensor): The input noisy image (lidar RANSAC residuals). Shape: [B, 1, H, W]
            cond_img (torch.Tensor): The conditional Sentinel-2 image patches. Shape: [B, k*4, Hc, Wc]
            attrs (torch.Tensor): The attributes (metadata) for the S2 patches. Shape: [B, k*8]
            t (torch.Tensor): The timestep for the diffusion process. Shape: [B]
        """
        # 1. Embed time and attributes into a conditioning vector 
        # The timestep and attribute information are processed by MLPs to create a dense
        # vector that the U-Net can use as a conditioning signal at each layer.
        t_emb = self.time_mlp(timestep_embedding(t, self.embed_dim))
        if self.attr_mlp is not None:
            a_emb = self.attr_mlp(attrs)
            cond_vec = t_emb + a_emb
        else:
            cond_vec = t_emb

        # 2. Combine the noisy image with the conditional images
        # The S2 images are concatenated directly with the noisy input.
        # This provides the UNet with pixel-level conditioning information.
        x = torch.cat([x, cond_img], dim=1)

        # 3. Encoder Path: Downsample and store skip connections
        # The network processes the input through a series of Down blocks. Each Down
        # block halves the spatial resolution and doubles the channels. A copy
        # of the feature map is saved at each level for the decoder.
        skips = []
        x = self.input_conv(x, cond_vec)
        for down in self.downs:
            skips.append(x)
            x = down(x, cond_vec)

        # 4. Bottleneck
        # The innermost layer processes the most compressed representation of the data.
        x = self.bottleneck_conv(x, cond_vec)

        # 5. Decoder Path: Upsample and concatenate skip connections
        # The network reverses the process, upsampling the feature maps and
        # combining them with the saved skip connections from the encoder.
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, cond_vec)

        # 6. Final Output
        # The final convolution maps the feature maps back to the original channel size (1).
        return self.output_conv(x)