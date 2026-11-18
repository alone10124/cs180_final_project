"""Dent map: a small per-surface-point normal correction, optimised so all frames agree.

A dent tilts the normal, so the same room point appears at a slightly different
reflection direction in different frames and the accumulated panorama smears.
Alternate: (1) splat the panorama with the current dent map, (2) move the dent map
(a coarse grid, upsampled smoothly) by gradient descent so that every sample matches
the panorama looked up along its corrected reflection direction. The samples are
differentiable through the ray-cylinder geometry and the bilinear panorama lookup.
"""
import torch
import torch.nn.functional as F

from sodacan.accumulate.weights import sample_weights
from sodacan.geometry import cylinder, equirect
from sodacan.geometry.reflect import can_geometry
from sodacan.pipeline import evenly, reconstruct

DEG2 = (torch.pi / 180) ** 2


def upsample(coarse, atlas_hw):
    """Coarse (2, h, w) -> (2, Ha, Wa), bicubic, wrapping in the angle around the can."""
    padded = torch.cat([coarse[..., -2:], coarse, coarse[..., :2]], -1)[None]
    Ha, Wa = atlas_hw
    w = coarse.shape[-1]
    up = F.interpolate(padded, size=(Ha, Wa * (w + 4) // w), mode="bicubic", align_corners=False)[0]
    off = (up.shape[-1] - Wa) // 2
    return up[..., off : off + Wa]


def fit_dent_map(frames, c2w, K, can, cfg, label=None, grid=(12, 48), iters=150, outer=5, lr=2e-3,
                 smooth=1e-2, max_frames=24, verbose=False):
    """Returns (dent atlas (2, Ha, Wa) in radians, refined Result)."""
    dev = frames.device
    coarse = torch.zeros(2, *grid, device=dev, requires_grad=True)
    opt = torch.optim.Adam([coarse], lr=lr)
    idx = evenly(frames.shape[0], max_frames).to(dev)
    H, W = cfg["pano"]
    result = reconstruct(frames, c2w, K, can, cfg, label=label)
    label = result.label if label is None else label
    with torch.no_grad():
        g0 = can_geometry(c2w[idx], K, can)
        m = g0.mask
        vals = frames[idx].permute(0, 2, 3, 1)[m]
        if label is not None:
            vals = vals - cylinder.atlas_sample(label, g0.s[m], g0.h[m], can.height)
        vals = vals / g0.fresnel[m].clamp_min(1e-4)[:, None]
        w = sample_weights(g0, cfg, (H, W))[m]
    for o in range(outer):
        pano = torch.log1p(result.pano.detach() * 10)
        valid = result.valid
        for it in range(iters):
            dent = upsample(coarse, can.atlas)
            g = can_geometry(c2w[idx], K, can, dent=dent)
            r = g.reflected[m]
            pred = equirect.sample(pano, r)
            ok = equirect.sample(valid[None].float(), r)[:, 0] > 0.99
            err = (torch.log1p(vals.clamp_min(0) * 10) - pred).abs().mean(1)
            data = (w * err * ok).sum() / (w * ok).sum().clamp_min(1e-9)
            tv = (coarse - torch.roll(coarse, 1, -1)).pow(2).mean() + (coarse[:, 1:] - coarse[:, :-1]).pow(2).mean()
            loss = data + smooth * tv / DEG2  # smoothness in degrees^2
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            dent = upsample(coarse, can.atlas)
            result = reconstruct(frames, c2w, K, can, cfg, dent=dent, label=label)
        if verbose:
            print(f"outer {o}: data loss {float(data):.4f}, dent rms {float(dent.pow(2).mean().sqrt()) * 57.3:.3f} deg")
    return upsample(coarse.detach(), can.atlas), result
