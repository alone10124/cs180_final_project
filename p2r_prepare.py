"""Turn a recovered environment (pano.npy + valid.npy) into Pano2Room's input panorama.

  HF_HOME=/overflow/xiang/hf_cache python p2r_prepare.py --input splat_both --out results/p2r_cans

Steps: display encoding (envmap.compare), a gamma lift for the dark room (--tone-gamma),
upsampling to 2048 x 1024, Stable Diffusion inpainting of the texels no can ever reflected
(inpaint.diffusion, Pano2Room's own SD2-inpainting model), and a horizontal flip, because
the repo's panoramas are mirrored relative to a camera panorama (README, Conventions).

Writes <out>/input_panorama.png, <out>/valid_mask.png (white = observed, same layout),
<out>/prepare.png (observed / filled, for the record) and <out>/prepare.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import RESULTS
from envmap import compare, save_png, side_by_side
from inpaint import PROMPT, diffusion, telea

SIZE = (2048, 1024)  # Pano2Room resizes its input to 2048 wide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory under results/ with pano.npy and valid.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tone-gamma", type=float, default=0.7, help="display ** g; < 1 lifts dark texels")
    ap.add_argument("--method", choices=["diffusion", "telea"], default="diffusion")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-2-inpainting")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    source = RESULTS / args.input
    pano, valid = np.load(source / "pano.npy"), np.load(source / "valid.npy")
    _, estimate, _ = compare(pano, valid, None)
    estimate = np.clip(estimate, 0, 1) ** args.tone_gamma

    big = np.clip(cv2.resize(estimate, SIZE, interpolation=cv2.INTER_CUBIC), 0, 1)
    big_valid = cv2.resize(valid.astype(np.uint8), SIZE, interpolation=cv2.INTER_NEAREST) > 0
    if args.method == "diffusion":
        filled = diffusion(big, ~big_valid, args.model, PROMPT, args.steps, args.seed)
    else:
        filled = telea(big, ~big_valid)
    filled = np.where(big_valid[..., None], big, filled)

    # Mirror into camera-panorama layout for Pano2Room.
    save_png(out / "input_panorama.png", np.ascontiguousarray(filled[:, ::-1]))
    cv2.imwrite(str(out / "valid_mask.png"), np.ascontiguousarray(big_valid[:, ::-1]).astype(np.uint8) * 255)
    small = cv2.resize(filled, valid.shape[::-1], interpolation=cv2.INTER_AREA)
    save_png(out / "prepare.png", np.vstack([side_by_side(estimate, None, valid), small]))
    info = {"input": args.input, "tone_gamma": args.tone_gamma, "fill": args.method, "model": args.model,
            "observed_pixel_fraction": float(big_valid.mean()), "size": SIZE}
    (out / "prepare.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
