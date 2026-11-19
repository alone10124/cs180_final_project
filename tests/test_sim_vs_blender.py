"""Phase 1: Blender frames pass through the Phase 0 code with recognisable output.

Skipped until scenes are exported and rendered:
  python -m sodacan.sim.blender_export --out data/blender --scenes 20
  blender -b -P sodacan/sim/blender_scene.py -- data/blender/scene_000   (each scene)

Scored above the horizon. Below it, Cycles' can reflects the table it stands on (a near,
lit plane), while the fast renderer looks up the distant room's floor: a real difference
between the simulator and the world, not a failure of the Phase 0 code.
"""
from pathlib import Path

import pytest
import torch

from sodacan.config import REPO_ROOT, load_can, load_pipeline
from sodacan.geometry.camera import Intrinsics
from sodacan.geometry.equirect import resize_pano
from sodacan.metrics import psnr
from sodacan.pipeline import reconstruct
from sodacan.sim.blender_export import read_renders
from sodacan.sim.rooms import read_hdr

SCENES = sorted((REPO_ROOT / "data" / "blender").glob("scene_*")) if (REPO_ROOT / "data" / "blender").is_dir() else []
def _complete(scene):
    """Every pose has a rendered frame (a scene still rendering is skipped)."""
    import json

    if not (scene / "blender").is_dir():
        return False
    poses = json.loads((scene / "scene.json").read_text())["poses"]
    return len(list((scene / "blender").glob("frame_*.*"))) == len(poses)


RENDERED = [s for s in SCENES if _complete(s)]


@pytest.mark.skipif(not RENDERED, reason="no Blender renders in data/blender/*/blender")
@pytest.mark.parametrize("folder", RENDERED, ids=[s.name for s in RENDERED])
def test_blender_frames_reconstruct(folder):
    frames_b, spec = read_renders(folder, "blender")
    frames_f, _ = read_renders(folder, "fast")
    can = load_can(material=spec["material"])
    cfg = load_pipeline()
    H, W = cfg["pano"]
    K = Intrinsics(spec["width"], spec["height"], spec["fx"], spec["fy"], spec["cx"], spec["cy"])
    c2w = torch.tensor(spec["poses"], dtype=torch.float32)
    ref = resize_pano(read_hdr(Path(folder) / "env.hdr"), H, W)
    rb = reconstruct(frames_b, c2w, K, can, cfg)
    rf = reconstruct(frames_f, c2w, K, can, cfg)
    both = rb.valid & rf.valid
    above = both.clone()
    above[H // 2 :] = False  # row 0 is straight up; the horizon is the middle row
    pb, pf = psnr(rb.pano, ref, above), psnr(rf.pano, ref, above)
    agree = psnr(rb.pano, rf.pano, above)
    full = psnr(rb.pano, rf.pano, both)
    print(f"{folder.name} ({spec['material']['model']}), above the horizon: Blender {pb:.2f} dB, fast renderer {pf:.2f} dB "
          f"vs truth; Blender vs fast {agree:.2f} dB ({full:.2f} dB including the table below the horizon)")
    frac = float(above.sum() / both.sum().clamp_min(1))
    if frac < 0.15:
        pytest.skip(f"{folder.name}: only {frac:.0%} of the observed texels are above the horizon (the camera looks "
                    "down, so the can reflects mostly the table)")
    assert pb > pf - 6, "the fast renderer is much more idealised than Cycles"
    assert agree > 15
