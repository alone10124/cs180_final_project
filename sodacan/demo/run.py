"""The end-to-end run used by the demo app and the timing benchmark.

`process` is a generator of Update objects: decoding and posing, the panorama building
up frame by frame, then the enhanced panorama. Everything except the restoration
network is the classical pipeline.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from sodacan.capture.load import load_capture
from sodacan.geometry.equirect import solid_angle
from sodacan.pipeline import reconstruct_steps


@dataclass
class Update:
    stage: str
    message: str
    pano: Optional[torch.Tensor] = None
    weight: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None
    enhanced: Optional[torch.Tensor] = None
    hint: str = ""
    timing: dict = field(default_factory=dict)
    frames_done: int = 0
    frames_total: int = 0


def coverage_hint(valid, c2w):
    """Where to move the phone next: toward the least-covered azimuth near the horizon, and
    lower or higher if one side of the horizon is empty. (A simple arrow, not RL.)"""
    H, W = valid.shape
    band = valid[H // 2 - H // 8 : H // 2 + H // 8].float().mean(0)  # coverage per column near the horizon
    k = max(W // 12, 1)  # 30 degree sectors
    sectors = band[: (W // k) * k].reshape(-1, k).mean(1)
    gap = int(torch.argmin(sectors))
    gap_az = (gap + 0.5) * k / W * 360 - 180
    cam = c2w[-1, :3, 3]
    cam_az = math.degrees(math.atan2(float(cam[1]), float(cam[0])))
    turn = (gap_az - cam_az + 180) % 360 - 180
    covered_up = valid[: H // 2].float().mean().item()
    covered_down = valid[H // 2 :].float().mean().item()
    parts = []
    if sectors[gap] < 0.5:
        parts.append(f"walk {'counter-clockwise' if turn > 0 else 'clockwise'} around the can by about {abs(turn):.0f} deg (seen from above)")
    if covered_up < 0.5 * covered_down:
        parts.append("hold the phone lower (near table height) to see higher in the room")
    return "; ".join(parts) or "coverage looks even; keep sweeping to average out noise"


def process(video, phone, board, can, cfg, model=None, device="cpu", frames_rgb=None, yield_every=2):
    t0 = time.time()
    timing = {}
    yield Update("decode", "decoding video and finding the board ...")
    cap = load_capture(video, phone, board, cfg, device=device, frames_rgb=frames_rgb)
    timing["decode+pose"] = time.time() - t0
    n = len(cap.frames)
    yield Update("pose", f"board found in {cap.stats['posed']} of {cap.stats['decoded']} frames "
                 f"(median reprojection {cap.stats['median_reproj_px']:.2f} px)", timing=dict(timing), frames_total=n)
    t1 = time.time()
    result = None
    for result in reconstruct_steps(cap.frames, cap.c2w, cap.K, can, cfg, chunk=4, yield_every=yield_every):
        w = solid_angle(*result.valid.shape, device=result.valid.device)
        cov = float((w * result.valid).sum() / w.sum())
        yield Update("accumulate", f"frame {result.frames_done}/{n}: {cov:.0%} of the sphere observed",
                     result.pano, result.weight, result.valid, frames_done=result.frames_done, frames_total=n,
                     timing=dict(timing))
    timing["classical"] = time.time() - t1
    hint = coverage_hint(result.valid, cap.c2w)
    enhanced = None
    if model is not None:
        from sodacan.restore.infer import enhance

        t2 = time.time()
        enhanced = enhance(model, result.pano, result.weight, result.valid)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        timing["restore"] = time.time() - t2
    timing["total"] = time.time() - t0
    yield Update("done", f"done in {timing['total']:.1f} s", result.pano, result.weight, result.valid, enhanced, hint,
                 dict(timing), n, n)
