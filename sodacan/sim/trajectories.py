"""Camera paths around the can: a hand-held arc with up/down tilt, as in the capture protocol."""
import math

import numpy as np
import torch

from sodacan.geometry.camera import look_at


def arc_with_tilt(count, distance=0.3, arc_deg=45, elevation_deg=(-15, 25), cycles=2, target_z=0.06,
                  start_deg=0.0, jitter_m=0.0, seed=0, device="cpu", min_z=None):
    """Cameras sweeping arc_deg of azimuth around the can axis while the elevation (seen
    from the can's middle) oscillates cycles times between the two elevation limits.
    All look at the axis at target_z. min_z keeps the phone above the table (a can on a
    table cannot be filmed from below). Returns camera-to-world (count, 4, 4)."""
    i = np.arange(count) / max(count - 1, 1)
    az = np.radians(start_deg + arc_deg * (i - 0.5))
    lo, hi = np.radians(elevation_deg)
    el = lo + (hi - lo) * 0.5 * (1 - np.cos(2 * math.pi * cycles * i + math.pi * 0.0)) if cycles else np.full_like(i, (lo + hi) / 2)
    eye = np.stack([distance * np.cos(el) * np.cos(az), distance * np.cos(el) * np.sin(az), target_z + distance * np.sin(el)], axis=1)
    if jitter_m > 0:
        eye = eye + np.random.default_rng(seed).normal(0, jitter_m, eye.shape)
    if min_z is not None:
        eye[:, 2] = np.maximum(eye[:, 2], min_z)
    eye = torch.tensor(eye, dtype=torch.float32, device=device)
    target = torch.tensor([0.0, 0.0, target_z], device=device)
    return look_at(eye, target)


def random_path(rng, count, sim, can, device="cpu"):
    """A random hand-held path drawn from the ranges in configs/sim.yaml."""
    u = lambda key: float(rng.uniform(*sim[key]))  # noqa: E731
    lo, hi = sim["elevation_deg"]
    e0 = rng.uniform(lo, (lo + hi) / 2)
    e1 = rng.uniform((lo + hi) / 2, hi)
    return arc_with_tilt(
        count,
        distance=u("distance"),
        arc_deg=u("arc_deg"),
        elevation_deg=(e0, e1),
        cycles=int(rng.integers(sim["tilt_cycles"][0], sim["tilt_cycles"][1] + 1)),
        target_z=can.height / 2 + rng.uniform(-0.01, 0.01),
        start_deg=rng.uniform(0, 360),
        jitter_m=sim.get("jitter_mm", 0) / 1000,
        seed=int(rng.integers(2**31)),
        device=device,
        min_z=sim.get("min_camera_z"),
    )
