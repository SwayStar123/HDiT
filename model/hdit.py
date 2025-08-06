import torch
import torch.nn as nn
import math
from model.utils import TimestepEmbedder, LabelEmbedder, TokenMerge, TokenSplit
from model.layers import FeedForward, Attention, RMSNorm, FinalLayer
from einops import rearrange
from typing import List

#################################################################################
#                                 Core HDiT Model                                #
#################################################################################

class HDiTBlock(nn.Module):
    """ A single block of the Hourglass Diffusion Transformer """
    def __init__(self, hidden_size, num_heads, mlp_ratio, cond_size, attention_type, kernel_size):
        super().__init__()
        mlp_dim = int(hidden_size * mlp_ratio)
        self.attn = Attention(hidden_size, num_heads, cond_size, attention_type, kernel_size)
        self.ff = FeedForward(hidden_size, mlp_dim, cond_size)

    def forward(self, x, c, l=None):
        x, l = self.attn(x, c, l=l)
        x = self.ff(x, c)
        if l is not None:
            l = self.ff(l, c)
        return x, l

class HDiT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        latent_channels=32,
        patch_size=2,
        num_classes=1000,
        level_widths: List[int]       = [256, 512, 1024],
        level_depths: List[int]       = [2, 8], # [depth_per_level, depth_bottleneck]
        head_dim: int                 = 64,
        attention_kernel_size: int    = 7,
        mlp_ratio: float              = 4.0,
        mapping_network_depth: int    = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        num_levels = len(level_widths)
        depth_per_level, depth_bottleneck = level_depths

        # --- Input and Conditioning ---
        self.patch_embed = TokenMerge(in_channels, level_widths[0], patch_size)
        bottleneck_width = level_widths[-1]
        self.latent_proj = nn.Linear(latent_channels, bottleneck_width)

        map_width = bottleneck_width # Use the widest dimension for mapping network
        self.t_embedder = TimestepEmbedder(map_width)
        self.y_embedder = LabelEmbedder(num_classes, map_width, 0.05)
        self.mapping_network = nn.Sequential(
            RMSNorm(map_width),
            *[nn.Sequential(nn.Linear(map_width, map_width), nn.SiLU()) for _ in range(mapping_network_depth)]
        )

        # --- Hourglass Backbone ---
        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        self.merges, self.splits = nn.ModuleList(), nn.ModuleList()

        # Build the downsampling and upsampling levels
        for i in range(num_levels - 1):
            width, next_width = level_widths[i], level_widths[i+1]
            assert width % head_dim == 0, f"Width {width} at level {i} must be divisible by head_dim {head_dim}"
            num_heads = width // head_dim

            self.down_levels.append(nn.ModuleList([
                HDiTBlock(width, num_heads, mlp_ratio, map_width, "neighborhood", attention_kernel_size)
                for _ in range(depth_per_level)
            ]))
            self.merges.append(TokenMerge(width, next_width))

            self.up_levels.insert(0, nn.ModuleList([
                HDiTBlock(width, num_heads, mlp_ratio, map_width, "neighborhood", attention_kernel_size)
                for _ in range(depth_per_level)
            ]))
            self.splits.insert(0, TokenSplit(next_width, width))

        # Build the bottleneck level with global attention
        assert bottleneck_width % head_dim == 0, f"Bottleneck width {bottleneck_width} must be divisible by head_dim {head_dim}"
        num_heads = bottleneck_width // head_dim
        self.mid_level = nn.ModuleList([
            HDiTBlock(bottleneck_width, num_heads, mlp_ratio, map_width, "global", attention_kernel_size)
            for _ in range(depth_bottleneck)
        ])

        # --- Output ---
        self.final_norm = RMSNorm(level_widths[0])
        self.unpatch = FinalLayer(level_widths[0], in_channels, patch_size)

    def forward(self, x, t, y):
        # 1. Input processing and conditioning
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.patch_embed(x)

        t_embed = self.t_embedder(t.float())
        y_embed, _ = self.y_embedder(y, self.training)
        c = self.mapping_network(t_embed + y_embed)

        # 2. Hourglass Encoder
        skips = []
        for level, merge in zip(self.down_levels, self.merges):
            for block in level:
                x = block(x, c)
            skips.append(x)
            x = merge(x)

        # 3. Bottleneck (Mid-level) with latent conditioning
        for block in self.mid_level:
            x = block(x, c)

        # 4. Hourglass Decoder
        for level, split in zip(self.up_levels, self.splits):
            x = split(x, skips.pop())
            for block in level:
                x = block(x, c)

        # 5. Output processing
        x = self.final_norm(x)
        x = self.unpatch(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x

#################################################################################
#                                 HDiT Configs                                  #
#################################################################################
def HDiT_XL(patch_size, **kwargs):
    """ Extra-Large HDiT """
    return HDiT(
        patch_size=patch_size,
        level_widths=[384, 768, 1152],
        level_depths=[3, 24], # [depth_per_level, depth_bottleneck]
        **kwargs
    )

def HDiT_L(patch_size, **kwargs):
    """ Large HDiT """
    return HDiT(
        patch_size=patch_size,
        level_widths=[384, 640, 1024],
        level_depths=[3, 20],
        **kwargs
    )

def HDiT_B(patch_size, **kwargs):
    """ Base HDiT """
    return HDiT(
        patch_size=patch_size,
        level_widths=[256, 384, 768],
        level_depths=[2, 10],
        **kwargs
    )

def HDiT_S(patch_size, **kwargs):
    """ Small HDiT """
    return HDiT(
        patch_size=patch_size,
        level_widths=[192, 384],
        level_depths=[4, 10],
        **kwargs
    )

# --- Factory functions for specific patch sizes ---

# XL models
def HDiT_XL_2(**kwargs): return HDiT_XL(patch_size=2, **kwargs)
def HDiT_XL_4(**kwargs): return HDiT_XL(patch_size=4, **kwargs)
def HDiT_XL_8(**kwargs): return HDiT_XL(patch_size=8, **kwargs)

# L models
def HDiT_L_2(**kwargs): return HDiT_L(patch_size=2, **kwargs)
def HDiT_L_4(**kwargs): return HDiT_L(patch_size=4, **kwargs)
def HDiT_L_8(**kwargs): return HDiT_L(patch_size=8, **kwargs)

# B models
def HDiT_B_2(**kwargs): return HDiT_B(patch_size=2, **kwargs)
def HDiT_B_4(**kwargs): return HDiT_B(patch_size=4, **kwargs)
def HDiT_B_8(**kwargs): return HDiT_B(patch_size=8, **kwargs)

# S models
def HDiT_S_2(**kwargs): return HDiT_S(patch_size=2, **kwargs)
def HDiT_S_4(**kwargs): return HDiT_S(patch_size=4, **kwargs)
def HDiT_S_8(**kwargs): return HDiT_S(patch_size=8, **kwargs)


# --- Model registry dictionary ---

HDiT_models = {
    'HDiT-XL/2': HDiT_XL_2, 'HDiT-XL/4': HDiT_XL_4, 'HDiT-XL/8': HDiT_XL_8,
    'HDiT-L/2':  HDiT_L_2,  'HDiT-L/4':  HDiT_L_4,  'HDiT-L/8':  HDiT_L_8,
    'HDiT-B/2':  HDiT_B_2,  'HDiT-B/4':  HDiT_B_4,  'HDiT-B/8':  HDiT_B_8,
    'HDiT-S/2':  HDiT_S_2,  'HDiT-S/4':  HDiT_S_4,  'HDiT-S/8':  HDiT_S_8,
}

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = HDiT_B_8().to(device)
    print(f"Model created. Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    B, C, H, W = 4, 3, 256, 256
    C_l, H_l, W_l = 32, 4, 4
    x = torch.randn(B, C, H, W).to(device)
    t = torch.rand((B,)).to(device)
    y = torch.randint(0, 1000, (B,)).to(device)
    l = torch.randn(B, C_l, H_l, W_l).to(device) # Latent conditioning map

    output = model(x, t, y, l)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape must match input shape!"
    print("Forward pass successful!")


