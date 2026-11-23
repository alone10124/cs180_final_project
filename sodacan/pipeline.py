"""The classical pipeline, stages 3 to 5: reflection geometry, label removal, accumulation.

Input: linear frames, camera-to-world poses in the can frame, intrinsics, the can.
Output: an equirectangular panorama of the room (up to an unknown gain) with a weight
map that doubles as the coverage mask. `reconstruct_steps` yields the panorama as it
builds up, for the live demo.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from sodacan.accumulate.splat import Accumulator
from sodacan.accumulate.weights import sample_weights
from sodacan.capture.uv_warp import warp_to_surface
from sodacan.geometry import cylinder
from sodacan.geometry.reflect import can_geometry
from sodacan.separate.label_median import estimate_label


@dataclass
class Result:
    pano: torch.Tensor                    # (3, H, W) linear, unobserved texels 0
    weight: torch.Tensor                  # (H, W) accumulated weight (coverage map)
    valid: torch.Tensor                   # (H, W) bool coverage mask
    label: Optional[torch.Tensor] = None  # (3, Ha, Wa) label estimate
    label_count: Optional[torch.Tensor] = None
    frames_done: int = 0


def evenly(n, k):
    if k <= 0 or k >= n:
        return torch.arange(n)
    return torch.linspace(0, n - 1, k).round().long()


@torch.no_grad()
def estimate_label_from_frames(frames, c2w, K, can, cfg, pixel_mask=None):
    lc = cfg.get("label", {})
    idx = evenly(frames.shape[0], int(lc.get("max_frames", 64))).to(frames.device)
    pm = pixel_mask[idx] if pixel_mask is not None else None
    obs, valid = warp_to_surface(frames[idx], c2w[idx], K, can, pixel_mask=pm)
    return estimate_label(obs, valid, lc.get("method", "percentile"), float(lc.get("percentile", 0.2)))


def photographer_weight(g, cfg):
    """Down-weight reflections of the person holding the camera.

    Light reaching the camera off the can near normal incidence comes from right behind the
    camera, where the photographer stands. The person's reflection moves with the camera, so
    other views see those room directions unobstructed. A sample is down-weighted when its
    reflected direction is within az_deg of the direction back to the camera (in azimuth)
    and between el_below_deg below and el_above_deg above it (the body is below the phone).
    cfg["photographer"] = {az_deg, el_below_deg, el_above_deg, weight, up}; absent = off.
    up: the world's up direction in the can frame (default the can axis, z; a can lying on
    its side has it across the axis)."""
    pc = cfg.get("photographer")
    if not pc:
        return None
    r, v = g.reflected, -g.view  # v: from the surface point back to the camera
    up = torch.tensor(pc.get("up", (0.0, 0.0, 1.0)), dtype=r.dtype, device=r.device)
    up = up / up.norm()
    e1 = torch.tensor((1.0, 0.0, 0.0) if abs(float(up[0])) < 0.9 else (0.0, 1.0, 0.0), dtype=r.dtype, device=r.device)
    e1 = e1 - (e1 @ up) * up
    e1 = e1 / e1.norm()
    e2 = torch.linalg.cross(up, e1)
    az = lambda d: torch.atan2(d @ e2, d @ e1)
    daz = az(r) - az(v)
    daz = torch.remainder(daz + torch.pi, 2 * torch.pi) - torch.pi
    dz = torch.asin((r @ up).clamp(-1, 1)) - torch.asin((v @ up).clamp(-1, 1))
    rad = torch.pi / 180
    inside = ((daz.abs() < float(pc.get("az_deg", 25)) * rad)
              & (dz > -float(pc.get("el_below_deg", 60)) * rad) & (dz < float(pc.get("el_above_deg", 15)) * rad))
    return torch.where(inside, torch.full_like(daz, float(pc.get("weight", 0.02))), torch.ones_like(daz))


@torch.no_grad()
def reconstruct_steps(frames, c2w, K, can, cfg, dent=None, label=None, chunk=4, yield_every=1, pixel_mask=None):
    """Generator over partial Results; the last one is final. frames (B, 3, H, W) linear.
    pixel_mask (B, H, W) bool, optional: pixels to use (e.g. not hidden behind another can)."""
    H, W = cfg["pano"]
    method = cfg.get("label", {}).get("method", "percentile")
    count = None
    if label is None and method != "none":
        label, count = estimate_label_from_frames(frames, c2w, K, can, cfg, pixel_mask=pixel_mask)
    acc = Accumulator(H, W, 3, frames.device)
    min_w = float(cfg.get("min_weight", 1e-3))
    B = frames.shape[0]
    for n, i in enumerate(range(0, B, chunk)):
        g = can_geometry(c2w[i : i + chunk], K, can, dent=dent)
        m = g.mask
        if pixel_mask is not None:
            m = m & pixel_mask[i : i + chunk]
        vals = frames[i : i + chunk].permute(0, 2, 3, 1)[m]
        if label is not None:
            vals = vals - cylinder.atlas_sample(label, g.s[m], g.h[m], can.height)
        vals = vals / g.fresnel[m].clamp_min(1e-4)[:, None]
        w = sample_weights(g, cfg, (H, W))
        pw = photographer_weight(g, cfg)
        if pw is not None:
            w = w * pw
        w = w[m]
        acc.add(vals, g.reflected[m], w)
        last = i + chunk >= B
        if last or (n + 1) % yield_every == 0:
            p = acc.finalize(min_w)
            yield Result(p.pano, p.weight, p.valid, label, count, min(i + chunk, B))


def reconstruct(frames, c2w, K, can, cfg, dent=None, label=None, chunk=4, pixel_mask=None):
    result = None
    for result in reconstruct_steps(frames, c2w, K, can, cfg, dent, label, chunk, yield_every=10**9, pixel_mask=pixel_mask):
        pass
    return result
