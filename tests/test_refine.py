"""Dent map refinement and the relighting extension."""
import math

import numpy as np
import pytest
import torch

from sodacan.config import load_can, load_pipeline
from sodacan.extensions.relight import render_sphere
from sodacan.geometry.camera import Intrinsics
from sodacan.geometry.equirect import resize_pano
from sodacan.metrics import psnr
from sodacan.pipeline import reconstruct
from sodacan.refine.dent_map import fit_dent_map
from sodacan.sim.dataset import random_dent
from sodacan.sim.fast_render import RenderParams, render
from sodacan.sim.rooms import procedural_room
from sodacan.sim.trajectories import arc_with_tilt


@pytest.mark.slow
def test_dent_map_sharpens_a_dented_can():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    can = load_can(material={"model": "mirror"})
    cfg = load_pipeline(pano=[128, 256], label={"method": "none"})
    env = procedural_room(512, 1024, seed=5, device=dev)
    K = Intrinsics.from_fov(480, 640, 24.0)
    c2w = arc_with_tilt(32, distance=0.3, arc_deg=60, elevation_deg=(-20, 25), cycles=2, target_z=can.height / 2, device=dev)
    dent = random_dent(np.random.default_rng(1), *can.atlas, 1.0, dev)
    frames = render(env, c2w, K, can, RenderParams(dent=dent, ss=2))
    ref = resize_pano(env, 128, 256)
    plain = reconstruct(frames, c2w, K, can, cfg)
    fitted_dent, fitted = fit_dent_map(frames, c2w, K, can, cfg, iters=60, outer=3, max_frames=16)
    m = plain.valid & fitted.valid
    before, after = psnr(plain.pano, ref, m), psnr(fitted.pano, ref, m)
    corr = float(torch.corrcoef(torch.stack([fitted_dent.flatten(), dent.flatten()]))[0, 1])
    print(f"dent map: {before:.2f} -> {after:.2f} dB, correlation with the true dents {corr:.2f}")
    assert after > before + 0.5 and corr > 0.2


def test_relight_uniform_room_gives_albedo_times_radiance():
    # a white furnace: uniform radiance L lights a diffuse sphere to albedo * L everywhere
    L = 0.8
    pano = torch.full((3, 64, 128), L)
    img, inside = render_sphere(pano, size=64, albedo=(0.5, 0.5, 0.5))
    left = inside.clone()
    left[:, 32:] = False
    assert torch.allclose(img[:, left], torch.full_like(img[:, left], 0.5 * L), atol=0.02)
    right = inside & ~left
    assert torch.allclose(img[:, right], torch.full_like(img[:, right], L), atol=0.02)  # perfect gloss sees L


def test_relight_light_from_above_brightens_the_top():
    pano = torch.full((3, 64, 128), 0.1)
    pano[:, :8] = 10.0  # bright ceiling
    img, inside = render_sphere(pano, size=64)
    lum = img.mean(0)
    assert lum[8, 20] > 2 * lum[56, 20]
