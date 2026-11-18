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
def estimate_label_from_frames(frames, c2w, K, can, cfg):
    lc = cfg.get("label", {})
    idx = evenly(frames.shape[0], int(lc.get("max_frames", 64))).to(frames.device)
    obs, valid = warp_to_surface(frames[idx], c2w[idx], K, can)
    return estimate_label(obs, valid, lc.get("method", "percentile"), float(lc.get("percentile", 0.2)))


@torch.no_grad()
def reconstruct_steps(frames, c2w, K, can, cfg, dent=None, label=None, chunk=4, yield_every=1):
    """Generator over partial Results; the last one is final. frames (B, 3, H, W) linear."""
    H, W = cfg["pano"]
    method = cfg.get("label", {}).get("method", "percentile")
    count = None
    if label is None and method != "none":
        label, count = estimate_label_from_frames(frames, c2w, K, can, cfg)
    acc = Accumulator(H, W, 3, frames.device)
    min_w = float(cfg.get("min_weight", 1e-3))
    B = frames.shape[0]
    for n, i in enumerate(range(0, B, chunk)):
        g = can_geometry(c2w[i : i + chunk], K, can, dent=dent)
        m = g.mask
        vals = frames[i : i + chunk].permute(0, 2, 3, 1)[m]
        if label is not None:
            vals = vals - cylinder.atlas_sample(label, g.s[m], g.h[m], can.height)
        vals = vals / g.fresnel[m].clamp_min(1e-4)[:, None]
        w = sample_weights(g, cfg, (H, W))[m]
        acc.add(vals, g.reflected[m], w)
        last = i + chunk >= B
        if last or (n + 1) % yield_every == 0:
            p = acc.finalize(min_w)
            yield Result(p.pano, p.weight, p.valid, label, count, min(i + chunk, B))


def reconstruct(frames, c2w, K, can, cfg, dent=None, label=None, chunk=4):
    result = None
    for result in reconstruct_steps(frames, c2w, K, can, cfg, dent, label, chunk, yield_every=10**9):
        pass
    return result
