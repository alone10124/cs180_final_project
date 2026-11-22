"""Weighted bilinear splatting of samples into an equirectangular panorama."""
from dataclasses import dataclass

import torch

from sodacan.geometry.equirect import dir_to_uv


class Accumulator:
    """Running weighted sum of samples per panorama texel. Wraps in azimuth, clamps at the poles."""

    def __init__(self, H, W, C=3, device="cpu"):
        self.H, self.W, self.C = H, W, C
        self.sum = torch.zeros(C, H * W, device=device)
        self.weight = torch.zeros(H * W, device=device)

    def add(self, values, dirs, weights):
        """values (N, C), dirs (N, 3) unit, weights (N,)."""
        if values.numel() == 0:
            return
        u, v = dir_to_uv(dirs, self.H, self.W)
        x, y = u - 0.5, v - 0.5
        x0, y0 = torch.floor(x), torch.floor(y)
        fx, fy = x - x0, y - y0
        x0 = x0.long()
        y0 = y0.long()
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi = torch.remainder(x0 + dx, self.W)
            yi = (y0 + dy).clamp(0, self.H - 1)
            idx = yi * self.W + xi
            ww = w * weights
            self.weight.index_add_(0, idx, ww)
            self.sum.index_add_(1, idx, (values * ww[:, None]).T)

    def finalize(self, min_weight=1e-3):
        w = self.weight.view(self.H, self.W)
        pano = self.sum.view(self.C, self.H, self.W) / w.clamp_min(1e-12)
        valid = w > min_weight
        pano = torch.where(valid[None], pano, torch.zeros_like(pano)).clamp_min(0)
        return Panorama(pano, w.clone(), valid)


@dataclass
class Panorama:
    pano: torch.Tensor    # (C, H, W) linear
    weight: torch.Tensor  # (H, W) total splat weight (the coverage map)
    valid: torch.Tensor   # (H, W) bool, weight above threshold


def splat(values, dirs, weights, H, W, min_weight=1e-3):
    acc = Accumulator(H, W, values.shape[-1], values.device)
    acc.add(values, dirs, weights)
    return acc.finalize(min_weight)
