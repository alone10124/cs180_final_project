"""Baseline: a small U-Net with panorama-aware padding (wraps in azimuth)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sodacan.restore.common import IN_CHANNELS, PanoConv


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        g = math.gcd(cout, 8)
        self.body = nn.Sequential(PanoConv(cin, cout), nn.GroupNorm(g, cout), nn.GELU(), PanoConv(cout, cout), nn.GroupNorm(g, cout), nn.GELU())
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)


class UNet(nn.Module):
    multiple = 16

    def __init__(self, base=24, depth=4):
        super().__init__()
        chans = [base * 2**i for i in range(depth)]
        self.stem = PanoConv(IN_CHANNELS, base)
        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()
        c = base
        for i, ch in enumerate(chans):
            self.enc.append(Block(c, ch))
            c = ch
            if i < depth - 1:
                self.down.append(PanoConv(ch, ch, 3, stride=2))
        self.mid = Block(c, c)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for ch in reversed(chans[:-1]):
            self.up.append(nn.Conv2d(c, ch, 1))
            self.dec.append(Block(2 * ch, ch))
            c = ch
        self.head = nn.Conv2d(c, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        x = self.mid(x)
        for up, dec in zip(self.up, self.dec):
            s = skips.pop()
            x = up(F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False))
            x = dec(torch.cat([x, s], 1))
        return self.head(x)
