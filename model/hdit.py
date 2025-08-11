import torch
import torch.nn as nn
import math
from model.utils import TimestepEmbedder, LabelEmbedder, TokenMerge, TokenSplit
from model.layers import FeedForward, Attention, RMSNorm, FinalLayer
from einops import rearrange
from typing import List, Optional, Tuple

#################################################################################
#                                 Core HDiT Model                                #
#################################################################################

class HDiTBlock(nn.Module):
    """ A single block of the Hourglass Diffusion Transformer """
    def __init__(self, hidden_size, num_heads, mlp_ratio, cond_size,
                 attention_type, kernel_size,
                 static_hw: Optional[Tuple[int, int]] = None):  # NEW
        super().__init__()
        mlp_dim = int(hidden_size * mlp_ratio)
        self.attn = Attention(hidden_size, num_heads, cond_size,
                              attention_type, kernel_size,
                              static_hw=static_hw)  # pass static HW
        self.ff = FeedForward(hidden_size, mlp_dim, cond_size)

    def forward(self, x, c):
        x = self.attn(x, c)
        x = self.ff(x, c)
        return x

class HDiT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        patch_size=2,
        num_classes=1000,
        level_widths: List[int]       = [256, 512, 1024],
        level_depths: List[int]       = [2, 8], # [depth_per_level, depth_bottleneck]
        head_dim: int                 = 64,
        attention_kernel_size: int    = 7,
        mlp_ratio: float              = 4.0,
        mapping_network_depth: int    = 2,
        input_spatial_size: Optional[Tuple[int,int]] = None  # NEW: (H_in, W_in)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_levels = len(level_widths)
        depth_per_level, depth_bottleneck = level_depths

        # --- Decide default input size if not given (based on your note)
        if input_spatial_size is None:
            if in_channels == 3:
                input_spatial_size = (256, 256)
            elif in_channels == 4:
                input_spatial_size = (32, 32)
            else:
                raise ValueError("Please provide input_spatial_size=(H,W) for in_channels not in {3,4}.")
        H_in, W_in = input_spatial_size

        # --- Precompute the spatial schedule per level ---
        # After initial patch_embed (TokenMerge with patch_size), spatial drops once
        def div_check(x): 
            assert x % patch_size == 0, f"Spatial size must be divisible by patch_size={patch_size}"
            return x // patch_size

        H0, W0 = div_check(H_in), div_check(W_in)  # resolution entering level 0 blocks
        down_hw = []  # shapes BEFORE each down-merge
        Hc, Wc = H0, W0
        for _ in range(self.num_levels - 1):
            down_hw.append((Hc, Wc))
            Hc, Wc = div_check(Hc), div_check(Wc)  # after merge for next level
        mid_hw = (Hc, Wc)  # bottleneck resolution
        up_hw = list(reversed(down_hw))  # decoder resolutions

        # --- Input and Conditioning ---
        self.patch_embed = TokenMerge(in_channels, level_widths[0], patch_size)
        bottleneck_width = level_widths[-1]

        map_width = bottleneck_width
        self.t_embedder = TimestepEmbedder(map_width)
        self.y_embedder = LabelEmbedder(num_classes, map_width)
        self.mapping_network = nn.Sequential(
            RMSNorm(map_width),
            *[nn.Sequential(nn.Linear(map_width, map_width), nn.SiLU()) for _ in range(mapping_network_depth)]
        )

        # --- Hourglass Backbone ---
        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        self.merges, self.splits = nn.ModuleList(), nn.ModuleList()

        # Build down levels (encoder)
        for i in range(self.num_levels - 1):
            width, next_width = level_widths[i], level_widths[i+1]
            assert width % head_dim == 0, f"Width {width} at level {i} must be divisible by head_dim {head_dim}"
            num_heads = width // head_dim

            self.down_levels.append(nn.ModuleList([
                HDiTBlock(width, num_heads, mlp_ratio, map_width,
                          "neighborhood", attention_kernel_size,
                          static_hw=down_hw[i])  # <<< pass fixed HW
                for _ in range(depth_per_level)
            ]))
            self.merges.append(TokenMerge(width, next_width))

        # Build the bottleneck level (global attention) at mid_hw
        bottleneck_width = level_widths[-1]
        assert bottleneck_width % head_dim == 0, f"Bottleneck width {bottleneck_width} must be divisible by head_dim {head_dim}"
        num_heads = bottleneck_width // head_dim
        self.mid_level = nn.ModuleList([
            HDiTBlock(bottleneck_width, num_heads, mlp_ratio, map_width,
                      "global", attention_kernel_size,
                      static_hw=mid_hw)  # <<< pass fixed HW
            for _ in range(depth_bottleneck)
        ])

        # Build the up levels (decoder)
        for i in range(self.num_levels - 1):
            width = level_widths[-(i+2)]  # widths: ... L2, L1, L0
            assert width % head_dim == 0, f"Width {width} at up level {i} must be divisible by head_dim {head_dim}"
            num_heads = width // head_dim

            self.up_levels.append(nn.ModuleList([
                HDiTBlock(width, num_heads, mlp_ratio, map_width,
                          "neighborhood", attention_kernel_size,
                          static_hw=up_hw[i])  # <<< pass fixed HW
                for _ in range(depth_per_level)
            ]))
            prev_width = level_widths[-(i+1)]
            self.splits.append(TokenSplit(prev_width, width))

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

        # 2. Encoder
        skips = []
        for level, merge in zip(self.down_levels, self.merges):
            for block in level:
                x = block(x, c)
            skips.append(x)
            x = merge(x)

        # 3. Bottleneck
        for block in self.mid_level:
            x = block(x, c)

        # 4. Decoder
        for level, split in zip(self.up_levels, self.splits):
            x = split(x, skips.pop())
            for block in level:
                x = block(x, c)

        # 5. Output
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
    x = torch.randn(B, C, H, W).to(device)
    t = torch.rand((B,)).to(device)
    y = torch.randint(0, 1000, (B,)).to(device)

    output = model(x, t, y)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape must match input shape!"
    print("Forward pass successful!")


