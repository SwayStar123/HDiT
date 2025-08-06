import torch
import torch.nn as nn
import torch.nn.functional as F
import natten
from einops import rearrange
import math

class FinalLayer(nn.Module):
    """ The final layer that unpatches tokens back to an image. """
    def __init__(self, in_features, out_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_features, out_channels * patch_size * patch_size)
        # Initialize the final projection to zero.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = self.proj(x)
        # Rearrange tokens back into spatial patches (PixelShuffle)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size)
        return x
    
class RMSNorm(nn.Module):
    """ Root Mean Square Layer Normalization """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate the root mean square of the input tensor.
        # The input is expected to be in (B, ..., D) format, and normalization is over the last dimension D.
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Scale the input by the RMS value and the learned weight.
        return x * rms * self.weight

class AdaRMSNorm(nn.Module):
    """ Adaptive Root Mean Square Layer Normalization """
    def __init__(self, hidden_size, cond_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        # A linear layer to project the conditioning signal to the feature dimension.
        # Initialized to zero to make the initial block an identity function.
        self.linear = nn.Linear(cond_size, hidden_size, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, c):
        # Project the conditioning signal and add 1 for initial scaling.
        scale = self.linear(c).unsqueeze(1).unsqueeze(1) + 1
        # Calculate RMS of the input tensor.
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Apply the adaptive scale.
        return x * rms * scale

class GEGLU(nn.Module):
    """ Gated GELU Activation """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)

class FeedForward(nn.Module):
    """ The Feed-Forward block for the transformer """
    def __init__(self, hidden_size, mlp_dim, cond_size):
        super().__init__()
        self.norm = AdaRMSNorm(hidden_size, cond_size)
        self.up_proj = GEGLU(hidden_size, mlp_dim)
        self.down_proj = nn.Linear(mlp_dim, hidden_size)
        # Initialize the down projection to zero.
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

    def forward(self, x, c):
        skip = x
        x = self.norm(x, c)
        x = self.up_proj(x)
        x = self.down_proj(x)
        return x + skip

class AxialRoPE(nn.Module):
    """ 2D Rotary Positional Embeddings for Image (h, w) """
    def __init__(self, dim_head):
        super().__init__()
        # Ensure head dimension is divisible by 4 for 2 axes (x and y)
        assert dim_head % 4 == 0, "Head dimension must be divisible by 4 for 2D RoPE."
        self.dim_head = dim_head

        # Generate separate frequency bands for each of the 2 axes
        freqs = torch.exp(torch.linspace(
            math.log(math.pi), 
            math.log(10 * math.pi), 
            self.dim_head // 4
        ))
        self.register_buffer("freqs", freqs)

    def forward(self, h, w, device):
        # Create coordinates for the image stream
        y_pos = torch.arange(h, device=device).float()
        x_pos = torch.arange(w, device=device).float()
        # Calculate theta for image coordinates
        theta_y = torch.outer(y_pos, self.freqs).reshape(h, 1, -1)
        theta_x = torch.outer(x_pos, self.freqs).reshape(1, w, -1)
        # Flatten and concatenate
        return torch.cat([
            theta_y.reshape(h * w, -1),
            theta_x.reshape(h * w, -1)
        ], dim=-1)

def apply_rotary_pos_emb(q, k, theta):
    """ Apply RoPE"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    # The part of the head dimension that will be rotated
    rope_dim = theta.shape[-1]
    q_rope, q_pass = q[..., :rope_dim], q[..., rope_dim:]
    k_rope, k_pass = k[..., :rope_dim], k[..., rope_dim:]

    sin, cos = theta.sin(), theta.cos()
    q_rotated = (q_rope * cos) + (rotate_half(q_rope) * sin)
    k_rotated = (k_rope * cos) + (rotate_half(k_rope) * sin)

    return torch.cat([q_rotated, q_pass], dim=-1), torch.cat([k_rotated, k_pass], dim=-1)

class Attention(nn.Module):
    """
    The core Attention block, supporting both global (with optional latent conditioning)
    and neighborhood attention.
    """
    def __init__(self, hidden_size, num_heads, cond_size, attention_type="global", kernel_size=7):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads
        self.attention_type = attention_type
        self.kernel_size = kernel_size

        if attention_type == "neighborhood":
            if natten is None:
                raise ImportError("`natten` is required. Please run `pip install natten`.")

        self.norm = AdaRMSNorm(hidden_size, cond_size)
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.pos_emb = AxialRoPE(self.dim_head)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, c):
        B, H, W, C = x.shape
        x_skip = x
        x_norm = self.norm(x, c)
        qkv = self.qkv_proj(x_norm)
        q, k, v = rearrange(qkv, 'b h w (qkv heads d) -> qkv b heads (h w) d', qkv=3, heads=self.num_heads)

        theta_img = self.pos_emb(H, W, x.device)
        theta = theta_img.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, theta)

        if self.attention_type == "neighborhood":
            q, k, v = map(lambda t: rearrange(t, 'b heads (h w) d -> b h w heads d', h=H, w=W), (q, k, v))
            out = natten.functional.na2d(q, k, v, kernel_size=self.kernel_size)
            out = rearrange(out, 'b h w heads d -> b heads (h w) d')
        else:
            out = F.scaled_dot_product_attention(q, k, v)

        # Reshape and project output
        out = rearrange(out, 'b heads (h w) d -> b h w (heads d)', h=H, w=W)
        out = self.out_proj(out)

        return out + x_skip
