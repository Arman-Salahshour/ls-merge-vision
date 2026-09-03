
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




class RoPESelfAttention(nn.Module):
    def __init__(self, dim, n_heads=4, rope_base=10000.0, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0, 'dim must divide evenly by n_heads'
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope_base = rope_base
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        '''cache is rebuilt if sequence length or device changes'''
        self._cache_len = -1
        self.register_buffer('cos', torch.zeros(0), persistent=False)
        self.register_buffer('sin', torch.zeros(0), persistent=False)

    def _rope(self, T, device, dtype):
        if self._cache_len != T or self.cos.device != device or self.cos.dtype != dtype:
            cos, sin = build_rope_cache(T, self.head_dim, self.rope_base, device, dtype)
            self.cos, self.sin = cos, sin
            self._cache_len = T
        return self.cos, self.sin

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        cos, sin = self._rope(T, x.device, x.dtype)
        '''rope goes on q and k only, never on v'''
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        '''no causal mask, weight chunks are not autoregressive'''
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)




class MLP(nn.Module):
    def __init__(self, dim, ratio=4, dropout=0.0):
        super().__init__()
        hidden = dim * ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))