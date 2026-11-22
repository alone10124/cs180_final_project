"""Per-sample weights for accumulation: Fresnel strength, pixel footprint, viewing angle."""
import math

import torch


def footprint(reflected, hit):
    """Solid angle (sr) each pixel's reflection covers, from finite differences of the
    reflection directions (B, H, W, 3). One-sided differences at the silhouette."""
    def diff(dim):
        fwd = torch.roll(reflected, -1, dims=dim) - reflected
        bwd = reflected - torch.roll(reflected, 1, dims=dim)
        ok_f = torch.roll(hit, -1, dims=dim)  # hit (B, H, W) shares dims 0..2 with reflected
        return torch.where(ok_f[..., None], fwd, bwd)

    dx = diff(2)
    dy = diff(1)
    return torch.cross(dx, dy, dim=-1).norm(dim=-1)


def texel_solid_angle(dirs, H, W):
    sin_t = torch.sqrt((1 - dirs[..., 2] ** 2).clamp_min(0)).clamp_min(0.05)
    return (math.pi / H) * (2 * math.pi / W) * sin_t


def sample_weights(geom, cfg, pano_hw):
    """Weights (B, H, W) for the pixels of a PixelGeometry; zero outside geom.mask."""
    wc = cfg.get("weights", {})
    w = geom.fresnel.clamp_min(1e-6) ** float(wc.get("fresnel_power", 2.0))
    w = w * geom.cos_i.clamp_min(0) ** float(wc.get("cos_power", 1.0))
    if wc.get("footprint", True):
        omega = footprint(geom.reflected, geom.hit)
        w = w / (1 + omega / texel_solid_angle(geom.reflected, *pano_hw))
    return torch.where(geom.mask, w, torch.zeros_like(w))
