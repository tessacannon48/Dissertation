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

        # time + attributes → single conditioning vector
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

        # Input conv
        self.input_conv = DoubleConv(in_channels + cond_channels, base_channels, embed_dim)

        # Track channel sizes for skip connections
        self.skip_channels = [base_channels]

        # Encoder
        self.skip_channels = [base_channels]
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i in range(unet_depth):
            max_channels = base_channels * 8
            out_ch = min(in_ch * 2, max_channels)
            use_attn = self._use_attention(i, stage='down', depth=unet_depth)
            self.downs.append(Down(in_ch, out_ch, embed_dim, use_attention=use_attn))
            self.skip_channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck
        self.bottleneck_conv = DoubleConv(in_ch, in_ch, embed_dim, use_attention=True)

        # Decoder path
        self.ups = nn.ModuleList()
        for i in range(unet_depth):
            in_ch_prev = in_ch
            # The input channels to the Up block are the output of the previous layer
            skip_ch = self.skip_channels[-(i + 2)]  # Match encoder skip
            out_ch = skip_ch  # Output channels are the same as the skip channels
            use_attn = self._use_attention(i, stage='up', depth=unet_depth)
            self.ups.append(Up(in_channels=in_ch_prev, skip_channels=skip_ch, out_channels=out_ch, embed_dim=embed_dim, use_attention=use_attn))
            in_ch = out_ch

        # Final output
        self.output_conv = nn.Conv2d(in_ch, in_channels, 1)

    def _use_attention(self, idx, stage, depth):
        """Determine whether to use attention based on variant and layer idx."""
        if self.attention_variant == 'none':
            return False
        elif self.attention_variant == 'all':
            return True
        elif self.attention_variant == 'mid':
            return idx == (depth // 2)
        elif self.attention_variant == 'default':
            return False
            # mimic original setup (attn in down1/down2/up1/up2)
            #if stage == 'down' and idx in [0, 1]:
                #return True
            #if stage == 'up' and idx in [depth - 2, depth - 1]:
                #return True
            #return False
        else:
            return False

    def forward(self, x, cond_img, attrs, t):
        # Ensure conditioning S2 patch matches LiDAR size
        if cond_img.shape[-2:] != x.shape[-2:]:
            cond_img = F.interpolate(cond_img, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # Time and attribute embedding
        t_emb = self.time_mlp(timestep_embedding(t, self.embed_dim))
        if self.attr_mlp is not None:
            a_emb = self.attr_mlp(attrs)
            cond_vec = t_emb + a_emb
        else:
            cond_vec = t_emb

        # Combine input and condition
        x = torch.cat([x, cond_img], dim=1)

        # Encoder
        skips = []
        x = self.input_conv(x, cond_vec)
        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x, cond_vec)

        # Bottleneck
        x = self.bottleneck_conv(x, cond_vec)

        # Decoder
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, cond_vec)

        return self.output_conv(x)