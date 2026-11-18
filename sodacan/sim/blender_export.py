"""Export simulated scenes for Blender Cycles (the photoreal validation set), and read renders back.

  python -m sodacan.sim.blender_export --out data/blender --scenes 20
  blender -b -P sodacan/sim/blender_scene.py -- data/blender/scene_000      # for each scene
  pytest tests/test_sim_vs_blender.py

Each scene folder holds scene.json (camera, poses, can, material), env.hdr (this
package's panorama layout), label.png (surface atlas, sRGB) and fast/ (the fast
renderer's frames of the same scene, linear .hdr) for comparison.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sodacan.color import to_uint8
from sodacan.config import load_can, load_yaml
from sodacan.sim.dataset import sample_scene
from sodacan.sim.rooms import read_hdr, write_hdr


def export_scene(folder, seed, sim, can, frames=16):
    folder = Path(folder)
    rng = np.random.default_rng(seed)
    scene = sample_scene(rng, sim, can, frames=frames)
    p = scene.params
    write_hdr(folder / "env.hdr", scene.env)
    if p.label is not None:
        cv2.imwrite(str(folder / "label.png"), to_uint8(p.label)[..., ::-1])
    for i, f in enumerate(scene.frames):
        write_hdr(folder / "fast" / f"frame_{i:04d}.hdr", f)
    spec = {
        "width": scene.K.width, "height": scene.K.height, "fx": scene.K.fx, "fy": scene.K.fy, "cx": scene.K.cx, "cy": scene.K.cy,
        "poses": scene.c2w.cpu().numpy().tolist(),
        "can": {"radius": can.radius, "height": can.height},
        "material": scene.can.material,
        "roughness_deg": p.roughness_deg, "strength": p.strength, "exposure": p.exposure, "ambient": p.ambient,
        "label": "label.png" if p.label is not None else None,
        "seed": seed,
    }
    (folder / "scene.json").write_text(json.dumps(spec, indent=2))
    return spec


def read_renders(folder, which="blender"):
    """Frames (B, 3, H, W) linear from folder/<which>/frame_*.hdr|exr, and the scene spec."""
    folder = Path(folder)
    spec = json.loads((folder / "scene.json").read_text())
    files = sorted((folder / which).glob("frame_*.*"))
    frames = torch.stack([read_hdr(f) for f in files]) if files else None
    return frames, spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/blender")
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    sim, can = load_yaml("sim"), load_can()
    sim = dict(sim, noise_read=[0, 0], noise_shot=[0, 0], motion_blur=[0, 0], dent_prob=0)  # Blender adds its own noise
    for k in range(args.scenes):
        export_scene(Path(args.out) / f"scene_{k:03d}", args.seed + k, sim, can, args.frames)
        print(f"scene_{k:03d}")


if __name__ == "__main__":
    main()
