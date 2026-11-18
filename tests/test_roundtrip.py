"""Phase 0 acceptance test, written before the code it tests.

Render a perfect-mirror can from a known panorama along a 60-frame arc with up/down
tilt, unwrap the frames with the classical pipeline, and require PSNR > 30 dB inside
the coverage mask. Every later change must keep this passing.
"""
import torch

from sodacan.config import load_can, load_pipeline
from sodacan.geometry.camera import Intrinsics
from sodacan.geometry.equirect import resize_pano
from sodacan.metrics import psnr
from sodacan.pipeline import reconstruct
from sodacan.sim.fast_render import RenderParams, render
from sodacan.sim.rooms import procedural_room
from sodacan.sim.trajectories import arc_with_tilt

PANO = (128, 256)


def roundtrip(device="cpu", frames=60, seed=0):
    can = load_can(material={"model": "mirror"})
    cfg = load_pipeline(pano=list(PANO), label={"method": "none"})
    env = procedural_room(512, 1024, seed=seed, device=device)
    K = Intrinsics.from_fov(480, 640, fov_y_deg=24.0)
    c2w = arc_with_tilt(frames, distance=0.30, arc_deg=60, elevation_deg=(-25, 30), cycles=2, target_z=can.height / 2, device=device)
    images = render(env, c2w, K, can, RenderParams(ss=2))
    result = reconstruct(images, c2w, K, can, cfg)
    reference = resize_pano(env, *PANO)
    return result, reference


def test_roundtrip_psnr_above_30db():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result, reference = roundtrip(device)
    coverage = result.valid.float().mean().item()
    score = psnr(result.pano, reference, result.valid)
    print(f"round trip: PSNR {score:.2f} dB inside coverage, coverage {coverage:.1%} of texels")
    assert coverage > 0.10, "the arc should observe a sizeable part of the panorama"
    assert score > 30.0
