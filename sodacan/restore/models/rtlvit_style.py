"""A lightweight vision transformer for reconstruction, in the spirit of RTLViT.

Only RTLViT's abstract was read (about 1 M parameters, real time on laptops and phones),
so this is not a reproduction. Design: a convolutional stem at full resolution,
8 x 8 patch tokens, plain pre-norm transformer blocks with a depthwise-convolution
position encoding (works at any panorama size), and a pixel-shuffle head fused with the
stem features. About 1.4 M parameters at the defaults.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sodacan.restore.common import IN_CHANNELS, PanoConv


class ConvPosition(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = PanoConv(dim, dim, 3, groups=dim)

    def forward(self, t, hw):
        B, N, C = t.shape
        g = t.transpose(1, 2).reshape(B, C, *hw)
        return t + self.conv(g).flatten(2).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0):
        super().__init__()
        self.pos = ConvPosition(dim)
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.heads = heads

    def forward(self, t, hw):
        t = self.pos(t, hw)
        B, N, C = t.shape
        q, k, v = self.qkv(self.n1(t)).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        t = t + self.proj(a)
        return t + self.mlp(self.n2(t))


class RTLViTStyle(nn.Module):
    def __init__(self, dim=128, depth=6, heads=4, patch=8, stem=32, mlp_ratio=2.0, block=Block):
        super().__init__()
        self.multiple = patch
        self.patch = patch
        self.stem = nn.Sequential(PanoConv(IN_CHANNELS, stem), nn.GELU(), PanoConv(stem, stem), nn.GELU())
        self.embed = nn.Conv2d(stem, dim, patch, stride=patch)
        self.blocks = nn.ModuleList([block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.unembed = nn.Linear(dim, stem * patch * patch)
        self.fuse = nn.Sequential(PanoConv(2 * stem, stem), nn.GELU(), nn.Conv2d(stem, 3, 1))
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def tokens(self, t, hw):
        for blk in self.blocks:
            t = blk(t, hw)
        return t

    def forward(self, x):
        s = self.stem(x)
        g = self.embed(s)
        B, C, h, w = g.shape
        t = self.tokens(g.flatten(2).transpose(1, 2), (h, w))
        t = self.unembed(self.norm(t)).transpose(1, 2).reshape(B, -1, h, w)
        up = F.pixel_shuffle(t, self.patch)
        return self.fuse(torch.cat([up, s], 1))
