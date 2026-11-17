"""Evaluate the environment recovered by a trained SRM run.

  SCENE=two python eval_env.py --run srm_a_32

The run's own settings (cans, views) are read from runs/<run>/args.json. The panorama
scored is the basis the selected cans use most; texels count as observed if the same
frames and cans cover them geometrically. Writes results/<run>/ like splat.py.
"""
import argparse
import json

import numpy as np

from common import CACHE, RESULTS, RUNS, load_poses, select_views
from data import Scene, pick_device
from envmap import accumulate, compare, finalize, load_reference, save_png, side_by_side, srm_environment
from train import load_srm


def evaluate_run(scene, run, reference, align=False):
    """Returns (metrics, pano, valid, estimate_display, reference_display)."""
    saved = json.loads((RUNS / run / "args.json").read_text())
    model = load_srm(scene.device, name=run)
    centre = np.load(CACHE / "split.npz")["centre"]
    frames = select_views(load_poses(), centre, scene.train, saved["views"], saved["view_select"], saved["seed"])
    _, valid = finalize(*accumulate(scene, frames, saved["cans"]))
    pano, share = srm_environment(model, scene, frames, saved["cans"])
    metrics, estimate, ref = compare(pano, valid, reference, align=align)
    metrics.update(views=len(frames), cans=saved["cans"], basis_share=[float(s) for s in share])
    return metrics, pano, valid, estimate, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="srm")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--align", action="store_true")
    args = ap.parse_args()

    scene = Scene(pick_device())
    metrics, pano, valid, estimate, reference = evaluate_run(scene, args.run, load_reference(args.reference), args.align)
    out = RESULTS / args.run
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pano.npy", pano)
    np.save(out / "valid.npy", valid)
    save_png(out / "pano.png", side_by_side(estimate, reference, valid))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
