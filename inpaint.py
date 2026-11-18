"""Fill the unobserved parts of a recovered panorama.

  SCENE=two python inpaint.py --input both --method telea       # classical, no model needed
  SCENE=two python inpaint.py --input both --method diffusion   # pretrained inpainting model (GPU)

--input is a results/ directory written by splat.py, eval_env.py or experiments.py
(pano.npy and valid.npy). Observed texels are kept exactly; only the rest is generated.
With a reference environment, PSNR is reported separately over observed texels
(fidelity of the reconstruction) and filled texels (how close the guess came), since
a plausible fill is not evidence about the real room.

Writes results/<input>_<method>/: filled.png (the panorama, input for walkthrough.py),
compare.png (input, filled, reference), metrics.json. --out-width W fills at W x W/2
(observed texels upsampled), e.g. 2048 for Pano2Room; figures and metrics stay at the
panorama's own size.
"""
import argparse
import json

import cv2
import numpy as np

from common import GAMMA, RESULTS, solid_angle
from envmap import compare, load_reference, save_png, side_by_side

PROMPT = "equirectangular 360 degree panorama photo of an indoor room, walls, ceiling, floor, natural light"


def telea(image, hole):
    """OpenCV inpainting with wrap-around padding so the azimuth seam is continuous."""
    pad = image.shape[1] // 4
    wide = np.concatenate([image[:, -pad:], image, image[:, :pad]], axis=1)
    wide_hole = np.concatenate([hole[:, -pad:], hole, hole[:, :pad]], axis=1)
    filled = cv2.inpaint((wide * 255).astype(np.uint8), wide_hole.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    return filled[:, pad:-pad] / 255.0


def diffusion(image, hole, model, prompt, steps, seed):
    """Stable Diffusion inpainting at 1024x512, then a second pass over the azimuth seam."""
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model, torch_dtype=dtype).to(device)
    generator = torch.Generator(device).manual_seed(seed)
    size = (1024, 512)

    def run(img, mask):
        out = pipe(
            prompt=prompt,
            image=Image.fromarray((img * 255).astype(np.uint8)).resize(size, Image.BICUBIC),
            mask_image=Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST),
            width=size[0],
            height=size[1],
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        out = np.asarray(out.resize(img.shape[1::-1], Image.BICUBIC)) / 255.0
        return np.where(mask[..., None], out, img)

    start = telea(image, hole)  # a neutral initial fill instead of the mask colour
    filled = run(start, hole)
    half = image.shape[1] // 2
    band = max(16, image.shape[1] // 32)
    strip = np.zeros_like(hole)
    strip[:, half - band : half + band] = True
    rolled = run(np.roll(filled, half, axis=1), strip & np.roll(hole, half, axis=1))
    return np.roll(rolled, -half, axis=1)


def weighted_psnr(a, b, mask):
    w = solid_angle(*mask.shape) * mask
    if not w.any():
        return None
    return float(-10 * np.log10(max((w[..., None] * (a - b) ** 2).sum() / (3 * w.sum()), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory under results/ holding pano.npy and valid.npy")
    ap.add_argument("--method", choices=["telea", "diffusion"], default="telea")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-2-inpainting")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--out-width", type=int, default=0, help="fill at this width (height = width / 2); 0 keeps the panorama size")
    args = ap.parse_args()

    source = RESULTS / args.input
    pano, valid = np.load(source / "pano.npy"), np.load(source / "valid.npy")
    metrics, estimate, reference = compare(pano, valid, load_reference(args.reference), align=args.align)
    hole = ~valid
    # Fill at the output size, so generated texels keep the detail of the 1024 x 512 diffusion pass.
    big, big_valid = estimate, valid
    if args.out_width:
        size = (args.out_width, args.out_width // 2)
        big = np.clip(cv2.resize(estimate, size, interpolation=cv2.INTER_CUBIC), 0, 1)
        big_valid = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    if args.method == "telea":
        filled = telea(big, ~big_valid)
    else:
        filled = diffusion(big, ~big_valid, args.model, args.prompt, args.steps, args.seed)
    filled = np.where(big_valid[..., None], big, filled)
    # Figures and metrics at the panorama's own size.
    small = filled if not args.out_width else cv2.resize(filled, hole.shape[::-1], interpolation=cv2.INTER_AREA)

    out = RESULTS / f"{args.input}_{args.method}"
    out.mkdir(parents=True, exist_ok=True)
    save_png(out / "filled.png", filled)
    np.save(out / "filled.npy", (filled**GAMMA).astype(np.float32))
    panels = [side_by_side(estimate, None, valid), small] + ([reference] if reference is not None else [])
    save_png(out / "compare.png", np.vstack(panels))
    result = {"method": args.method, "filled_fraction": float(solid_angle(*hole.shape)[hole].sum())}
    if reference is not None:
        result["psnr_observed"] = weighted_psnr(small, reference, valid)
        result["psnr_filled"] = weighted_psnr(small, reference, hole)
        result["blur_deg"] = metrics["blur_deg"]
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
