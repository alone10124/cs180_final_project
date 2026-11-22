"""The RTLViT-style network with modernised transformer blocks, as the ViT-5 ablation.

ViT-5 ("Vision Transformers for the Mid-2020s", arXiv 2602.08071) modernises the plain
ViT block. Its exact recipe was not checked for this file; the block below uses the
components such modernisations commonly combine: RMSNorm, QK-normalisation, 2D rotary
position embeddings, a SwiGLU MLP, LayerScale and register tokens. Compare it against
rtlvit_style.py at matched parameter count; cite it as "ViT-5-style", not as ViT-5.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sodacan.restore.models.rtlvit_style import RTLViTStyle


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_2d(x, hw, base=100.0):
    """Axial rotary embedding on (B, heads, N, d): half the channels rotate with the row,
    half with the column. Register tokens (beyond h*w) are left unrotated."""
    B, Hh, N, d = x.shape
    h, w = hw
    n = h * w
    q = d // 4
    freqs = 1.0 / base ** (torch.arange(q, device=x.device, dtype=torch.float32) / q)
    rows = torch.arange(h, device=x.device, dtype=torch.float32).repeat_interleave(w)
    cols = torch.arange(w, device=x.device, dtype=torch.float32).repeat(h)
    ang = torch.cat([rows[:, None] * freqs, cols[:, None] * freqs], -1)  # (n, d/2)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    grid, rest = x[:, :, :n], x[:, :, n:]
    a, b = grid[..., : d // 2], grid[..., d // 2 :]
    rot = torch.cat([a * cos - b * sin, a * sin + b * cos], -1)
    return torch.cat([rot, rest], 2)


class Vit5Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, layerscale=1e-2):
        super().__init__()
        self.heads = heads
        self.n1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.qn = RMSNorm(dim // heads)
        self.kn = RMSNorm(dim // heads)
        self.proj = nn.Linear(dim, dim)
        self.n2 = RMSNorm(dim)
        hidden = int(dim * mlp_ratio * 2 / 3) // 8 * 8 or 8
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.g1 = nn.Parameter(torch.full((dim,), layerscale))
        self.g2 = nn.Parameter(torch.full((dim,), layerscale))

    def forward(self, t, hw):
        B, N, C = t.shape
        q, k, v = self.qkv(self.n1(t)).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rope_2d(self.qn(q), hw), rope_2d(self.kn(k), hw)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        t = t + self.g1 * self.proj(a)
        x1, x2 = self.w12(self.n2(t)).chunk(2, -1)
        return t + self.g2 * self.w3(F.silu(x1) * x2)


class ViT5Style(RTLViTStyle):
    def __init__(self, dim=128, depth=6, heads=4, patch=8, stem=32, mlp_ratio=2.0, registers=4):
        super().__init__(dim, depth, heads, patch, stem, mlp_ratio, block=Vit5Block)
        self.norm = RMSNorm(dim)
        self.registers = nn.Parameter(torch.zeros(1, registers, dim))
        nn.init.trunc_normal_(self.registers, std=0.02)

    def tokens(self, t, hw):
        B, n = t.shape[0], t.shape[1]
        t = torch.cat([t, self.registers.expand(B, -1, -1)], 1)
        for blk in self.blocks:
            t = blk(t, hw)
        return t[:, :n]
