"""Training-free environment reconstruction: average specular residuals by reflection direction.

  SCENE=two python splat.py --cans a --views 32
  SCENE=two python splat.py --cans both --views 32 --keep 0.5   # two cans, pixel count of one

Writes results/<out>/: pano.npy (linear), valid.npy, pano.png (estimate over reference), metrics.json.
"""
import argparse
import json

import numpy as np

from common import CACHE, RESULTS, load_poses, select_views
from data import Scene, pick_device
from envmap import accumulate, compare, finalize, load_reference, save_png, side_by_side


def reconstruct(scene, cans="both", views=0, view_select="fps", diffuse="texture", keep=1.0, seed=0):
    """Returns (pano, valid, frames used)."""
    centre = np.load(CACHE / "split.npz")["centre"]
    frames = select_views(load_poses(), centre, scene.train, views, view_select, seed)
    pano, valid = finalize(*accumulate(scene, frames, cans, diffuse, keep, seed))
    return pano, valid, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cans", choices=["all", "a", "b", "both"], default="both")
    ap.add_argument("--views", type=int, default=0, help="number of training views; 0 uses all")
    ap.add_argument("--view-select", choices=["fps", "random", "arc"], default="fps")
    ap.add_argument("--diffuse", choices=["texture", "none"], default="texture", help="subtract the diffuse layer or not")
    ap.add_argument("--buffer", choices=["diffuse", "diffuse_est"], default="diffuse", help="which cached diffuse layer")
    ap.add_argument("--keep", type=float, default=1.0, help="fraction of pixels used")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", default=None, help="reference panorama; default is the scene's ground truth")
    ap.add_argument("--align", action="store_true", help="search the azimuth offset of the reference")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scene = Scene(pick_device(), diffuse=args.buffer)
    pano, valid, frames = reconstruct(scene, args.cans, args.views, args.view_select, args.diffuse, args.keep, args.seed)
    metrics, estimate, reference = compare(pano, valid, load_reference(args.reference), align=args.align)
    metrics["views"] = len(frames)

    out = RESULTS / (args.out or f"splat_{args.cans}_{len(frames)}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pano.npy", pano)
    np.save(out / "valid.npy", valid)
    save_png(out / "pano.png", side_by_side(estimate, reference, valid))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
