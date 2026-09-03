
import math
import torch
import torch.nn as nn
from constants import *
import torch.nn.functional as F


def build_rope_cache(seq_len, head_dim, base=10000.0, device=None, dtype=torch.float32):
    '''precompute cos and sin tables for rotary embeddings, one row per position'''
    assert head_dim % 2 == 0, 'head_dim must be even for rope'
    '''inverse frequencies, one per pair of dimensions'''
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    '''outer product gives angle per (position, freq pair)'''
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)



def apply_rope(x, cos, sin):
    '''x is [B, heads, T, head_dim], rotate each adjacent pair of dims by the position angle'''
    B, H, T, D = x.shape
    cos = cos[:T].view(1, 1, T, D // 2)
    sin = sin[:T].view(1, 1, T, D // 2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    '''standard 2d rotation applied pairwise'''
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    out = torch.stack((r1, r2), dim=-1).flatten(-2)
    return out