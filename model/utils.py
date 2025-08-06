import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import math

class TokenMerge(nn.Module):
    """ Downsamples by merging tokens, like Pixel-Unshuffle """
    def __init__(self, in_features, out_features, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_features * patch_size * patch_size, out_features)

    def forward(self, x):
        x = rearrange(x, 'b (h p1) (w p2) c -> b h w (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        return self.proj(x)

class TokenSplit(nn.Module):
    """ Upsamples by splitting tokens, like Pixel-Shuffle, with a learned skip connection """
    def __init__(self, in_features, out_features, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_features, out_features * patch_size * patch_size)
        # Learnable interpolation factor for the skip connection, as per the paper.
        self.lerp_factor = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size)
        # Interpolate between the upsampled path and the skip connection.
        # return torch.lerp(skip, x, self.lerp_factor) # Idk why accelerate doesnt cast the dtype. Must manually cast.
        lerp_factor = self.lerp_factor.to(dtype=x.dtype)
        return torch.lerp(skip, x, lerp_factor)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


    

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        # + 1 for unconditional (no class)
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels, train):
        embeddings = self.embedding_table(labels)
        return embeddings, labels
