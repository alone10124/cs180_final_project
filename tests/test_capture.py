"""Phase 2 code on synthetic captures: board convention, ChArUco poses, video round trip.

The real-capture acceptance test (window and ceiling lights in the right places against
a phone panorama) needs a real video; see README "Phase 2 acceptance".
"""
import math

import numpy as np
import pytest
import torch

from sodacan.capture.board import board_to_world, load_board, render_texture
from sodacan.capture.load import load_capture
from sodacan.capture.pose_charuco import estimate_poses
from sodacan.capture.segment_can import silhouette_score
from sodacan.capture.video import write_video
from sodacan.color import linear_to_srgb
from sodacan.config import load_can, load_pipeline
from sodacan.geometry.camera import Intrinsics
from sodacan.geometry.equirect import resize_pano
from sodacan.metrics import psnr
from sodacan.pipeline import reconstruct
from sodacan.sim.fast_render import RenderParams, render
from sodacan.sim.rooms import procedural_room
from sodacan.sim.trajectories import arc_with_tilt

K = Intrinsics(960, 720, 760.0, 760.0, 479.5, 359.5)


def to_rgb8(frames):
    return [(linear_to_srgb(f.clamp(0, 1)).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8) for f in frames]


@pytest.fixture(scope="module")
def synthetic_capture():
    can = load_can(material={"model": "mirror"})
    _, bcfg = load_board()
    env = procedural_room(512, 1024, seed=3)
    c2w = arc_with_tilt(24, distance=0.33, arc_deg=50, elevation_deg=(5, 35), cycles=1, target_z=can.height / 2)
    params = RenderParams(table=render_texture(bcfg, ppm=3000), ss=2, exposure=0.8)
    frames = render(env, c2w, K, can, params)
    return can, env, c2w, frames, to_rgb8(frames)


def test_board_to_world_convention():
    _, cfg = load_board()
    T = board_to_world(cfg)
    cx, cy = cfg["can_centre"]
    pts = np.array([[cx, cy, 0, 1], [cx + 0.01, cy, 0, 1], [cx, cy + 0.01, 0, 1], [cx, cy, -0.01, 1]]).T
    w = (T @ pts)[:3].T
    assert np.allclose(w, [[0, 0, 0], [0.01, 0, 0], [0, -0.01, 0], [0, 0, 0.01]], atol=1e-9)


def test_charuco_poses_match_truth(synthetic_capture):
    can, _, c2w, _, rgb = synthetic_capture
    board, cfg = load_board()
    poses = estimate_poses(rgb, np.array(K.matrix()), np.zeros(5), board, cfg)
    found = [i for i, p in enumerate(poses) if p.c2w is not None]
    assert len(found) >= 0.8 * len(rgb)
    rot, trans = [], []
    for i in found:
        est, true = poses[i].c2w, c2w[i].numpy()
        dR = est[:3, :3].T @ true[:3, :3]
        rot.append(math.degrees(math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        trans.append(np.linalg.norm(est[:3, 3] - true[:3, 3]))
    print(f"poses: {len(found)}/{len(rgb)}, median rotation error {np.median(rot):.3f} deg, translation {np.median(trans) * 1000:.2f} mm")
    assert np.median(rot) < 0.3 and np.median(trans) < 0.002
    assert silhouette_score(rgb[found[0]], torch.tensor(poses[found[0]].c2w, dtype=torch.float32), K, can) > 1.5


def test_video_file_to_panorama(synthetic_capture, tmp_path):
    can, env, _, _, rgb = synthetic_capture
    path = tmp_path / "can.mp4"
    write_video(path, rgb, fps=10)
    cfg = load_pipeline(pano=[128, 256], label={"method": "none"})
    phone = {"width": K.width, "height": K.height, "fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy, "dist": [0] * 5}
    cap = load_capture(path, phone, "board", cfg, stride=1, max_frames=100, long_side=10000)
    assert cap.stats["posed"] >= 0.8 * len(rgb)
    result = reconstruct(cap.frames, cap.c2w, cap.K, can, cfg)
    ref = resize_pano(env, 128, 256) * 0.8
    score = psnr(result.pano, ref, result.valid)
    print(f"video -> panorama: PSNR {score:.2f} dB from {cap.stats['posed']} posed frames (mp4 compression included)")
    assert score > 22
