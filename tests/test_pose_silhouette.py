"""The no-board fallback: poses from the can's outline (five numbers) and its label (azimuth).

A simulated printed can on a plain table, filmed along the capture protocol's arc. The
outline fixes everything but the camera's angle around the can, the label fixes that, and
the first frame sets the heading. Slow: about 6 s per frame on a laptop CPU.
"""
import math

import numpy as np
import pytest
import torch

from sodacan.capture.pose_silhouette import (canonicalise, estimate_poses_silhouette, merge_collinear,
                                             outline_points, params_to_c2w, c2w_to_params, rot_z)
from sodacan.color import linear_to_srgb
from sodacan.config import load_can, load_pipeline
from sodacan.geometry.camera import Intrinsics, look_at
from sodacan.metrics import psnr
from sodacan.pipeline import reconstruct
from sodacan.sim.dataset import random_label
from sodacan.sim.fast_render import RenderParams, render
from sodacan.sim.rooms import procedural_room
from sodacan.sim.trajectories import arc_with_tilt

K = Intrinsics(960, 720, 760.0, 760.0, 479.5, 359.5)


def rot_deg(a, b):
    dR = a[:3, :3].T @ b[:3, :3]
    return math.degrees(math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))


def test_params_round_trip_and_symmetry():
    can = load_can()
    c2w = look_at(torch.tensor([[0.2, 0.25, 0.15]]), torch.tensor([0.0, 0.0, 0.06]))[0].numpy().astype(np.float64)
    canon, a = canonicalise(c2w)
    assert abs(canon[1, 3]) < 1e-6 and canon[0, 3] > 0
    assert np.allclose(rot_z(a) @ canon, c2w, atol=1e-6)
    assert np.allclose(params_to_c2w(c2w_to_params(c2w)), canon, atol=1e-6)
    # the limbs seen from the canonical pose are the rotated limbs (the azimuth is unobservable);
    # the rings are sampled on a fixed angle grid, so only the limb points match one to one
    P0, _ = outline_points(canon[:3, 3], can)
    P1, _ = outline_points(c2w[:3, 3], can)
    assert np.allclose((rot_z(a)[:3, :3] @ P0[:80].T).T, P1[:80], atol=1e-6)


def test_merge_collinear_joins_a_broken_wall():
    segs = np.array([[100.0, 10, 101, 60], [101.2, 70, 102, 120], [300, 10, 300, 120]])
    out = merge_collinear(segs)
    assert len(out) == 2
    long = out[np.argmax(np.hypot(out[:, 2] - out[:, 0], out[:, 3] - out[:, 1]))]
    assert long[1] < 12 and long[3] > 118


@pytest.mark.slow
def test_silhouette_poses_on_a_printed_can():
    can = load_can()
    env = procedural_room(512, 1024, seed=3)
    c2w = arc_with_tilt(16, distance=0.33, arc_deg=40, elevation_deg=(5, 30), cycles=1, target_z=can.height / 2,
                        start_deg=30)
    label = random_label(np.random.default_rng(1), *can.atlas)
    frames = render(env, c2w, K, can, RenderParams(label=label, ss=2, exposure=0.8, roughness_deg=1.0, strength=0.6))
    rgb = [(linear_to_srgb(f.clamp(0, 1)).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8) for f in frames]
    poses = estimate_poses_silhouette(rgb, frames, K, can)
    ok = [i for i, p in enumerate(poses) if p.c2w is not None]
    assert len(ok) >= 0.8 * len(rgb)

    true = c2w.numpy().astype(np.float64)
    _, a0 = canonicalise(true[0])  # the estimate's heading is that of frame 0
    rot, trans, az = [], [], []
    for i in ok:
        est = rot_z(a0) @ poses[i].c2w
        rot.append(rot_deg(est, true[i]))
        trans.append(np.linalg.norm(est[:3, 3] - true[i, :3, 3]))
        az.append(abs(math.degrees(canonicalise(est)[1] - canonicalise(true[i])[1])))
    print(f"silhouette poses: {len(ok)}/{len(rgb)}, median rotation error {np.median(rot):.2f} deg, "
          f"translation {np.median(trans) * 1000:.1f} mm, azimuth {np.median(az):.2f} deg")
    assert np.median(az) < 1.0          # the label match
    assert np.median(rot) < 2.5 and np.median(trans) < 0.015

    cfg = load_pipeline(pano=[128, 256])
    est = torch.tensor(np.stack([rot_z(a0) @ poses[i].c2w for i in ok]), dtype=torch.float32)
    r_true = reconstruct(frames[ok], c2w[ok], K, can, cfg)
    r_est = reconstruct(frames[ok], est, K, can, cfg)
    both = r_true.valid & r_est.valid
    print(f"panorama from silhouette poses vs from true poses: {psnr(r_est.pano, r_true.pano, both):.2f} dB, "
          f"coverage {r_est.valid.float().mean():.3f} vs {r_true.valid.float().mean():.3f}")
    assert both.float().sum() > 0.8 * r_true.valid.float().sum()
