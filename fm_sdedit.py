"""Detail-filling after render-only optimisation (SDEdit): no render gradient during sampling.

1. Take the render-only panorama (fm_renderopt.py output, camera layout, 2048 x 1024).
2. Encode it with the FLUX VAE -> clean latent z.
3. Noise it to sigma_start: x = (1 - sigma_start) z + sigma_start eps.
4. Run the unchanged DiT360 sampler (FLUX.1-dev + panorama LoRA) from sigma_start to 0, unguided.

--strength s re-runs the last s fraction of the 28-step schedule (s=1 would ignore the input).
Scoring is fm_guide.py's (re-render train/test PSNR, GT PSNR/SSIM on observed texels and on all GT).

  SCENE=corncho1 python fm_sdedit.py --input results/renderopt_corncho1/I_renderopt_pixel.png \
      --select above --diffuse texture --sat 0.98 --gt ... --gt-rows 0.68 --out results/sdedit_corncho1
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import RESULTS
from data import Scene, pick_device
from fm_guide import (DEFAULT_PROMPT, Pixels, Renderer, camera_to_lookup, gt_scores, label, load_gt, load_pipe,
                      load_samples, splat_min)

sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party" / "DiT360"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="render-only panorama PNG (camera layout)")
    ap.add_argument("--strength", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=2.8)
    ap.add_argument("--prompt", default="This is a panorama. The image shows an indoor room.")
    ap.add_argument("--lora", default="Insta360-Research/DiT360-Panorama-Image-Generation")
    ap.add_argument("--max-samples", type=int, default=8_000_000)
    ap.add_argument("--sat", type=float, default=0.98)
    ap.add_argument("--select", choices=["cans", "above"], default="cans")
    ap.add_argument("--z-min", type=float, default=0.008)
    ap.add_argument("--diffuse", choices=["min", "texture"], default="texture")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--gt-rows", type=float, default=1.0)
    ap.add_argument("--out", default=str(RESULTS / "sdedit"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    scene = Scene(device)
    pixels = Pixels(scene, args.select, args.diffuse, args.z_min)
    data = load_samples(scene, pixels, device, args.max_samples)
    train, test = Renderer(data["train"], args.sat), Renderer(data["test"])
    gts, gt_rows = load_gt(args.gt, args.gt_rows) if args.gt else (None, None)
    sm, sm_valid = splat_min(data["train"])

    def score(pano_lin, valid=None):
        _, g = train.loss(pano_lin)
        r = {"train_psnr": train.psnr(pano_lin, g), "test_psnr": test.psnr(pano_lin, g), "gain": g.tolist()}
        if gts is not None:
            o = gt_scores(pano_lin, sm_valid, gts, gt_rows)
            r["gt_observed"] = o[0] if o else None
            if valid is None:
                a = gt_scores(pano_lin, torch.ones_like(sm_valid), gts, gt_rows)
                r["gt_all"] = a[0] if a else None
        return r

    src = cv2.imread(args.input)[..., ::-1].copy()
    src = cv2.resize(src, (2048, 1024), interpolation=cv2.INTER_AREA)
    src_t = torch.from_numpy(src).permute(2, 0, 1).float().div(255).to(device)
    metrics = {"args": vars(args), "runs": {"splat_min_diffuse": score(sm, sm_valid), "input_renderopt": score(camera_to_lookup(src_t))}}
    print("input_renderopt", json.dumps(metrics["runs"]["input_renderopt"]), flush=True)

    pipe = load_pipe(device, args.lora)
    with torch.no_grad():
        pe, ppe, _ = pipe.encode_prompt(prompt=args.prompt, prompt_2=None, device=device, max_sequence_length=512)
        z = pipe.vae.encode((src_t[None] * 2 - 1).to(pipe.vae.dtype)).latent_dist.mean
        z = (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z = pipe._pack_latents(z, 1, z.shape[1], z.shape[2], z.shape[3]).float()  # (1, 128*256/4, 64)
    pipe.text_encoder, pipe.text_encoder_2 = None, None
    torch.cuda.empty_cache()

    from pipeline import calculate_shift  # DiT360 pipeline (third_party/DiT360/pipeline.py)

    n_h, n_w = 1024 // 16, 2048 // 16
    seq_len = n_h * (n_w + 2)  # the pipeline pads two circular columns before computing the shift
    mu = calculate_shift(seq_len, pipe.scheduler.config.get("base_image_seq_len", 256), pipe.scheduler.config.get("max_image_seq_len", 4096),
                         pipe.scheduler.config.get("base_shift", 0.5), pipe.scheduler.config.get("max_shift", 1.15))
    full = np.linspace(1.0, 1 / args.steps, args.steps)

    rows = []
    for s in args.strength:
        k = int(round((1 - s) * args.steps))
        sub = full[k:]
        sch = copy.deepcopy(pipe.scheduler)
        sch.set_timesteps(sigmas=sub, device=device, mu=mu)
        sigma0 = float(sch.sigmas[0])
        row = []
        for seed in args.seeds:
            name = f"str{s:g}_s{seed}"
            gen = torch.Generator(device=device).manual_seed(seed)
            eps = torch.randn(z.shape, generator=gen, device=device, dtype=torch.float32)
            x0 = ((1 - sigma0) * z + sigma0 * eps).to(torch.bfloat16)
            t = time.time()
            image = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe, width=2048, height=1024, latents=x0,
                         sigmas=list(sub), guidance_scale=args.cfg).images[0]
            image.save(out / f"I_{name}.png")
            img = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255).to(device)
            r = score(camera_to_lookup(img))
            r.update(strength=s, sigma_start=sigma0, steps_run=len(sub), seconds=round(time.time() - t, 1))
            metrics["runs"][name] = r
            print(name, json.dumps(r), flush=True)
            o = (r.get("gt_all") or {})
            row.append(label(cv2.resize(np.asarray(image), (1024, 512)),
                             f"strength {s:g} (sigma0 {sigma0:.2f}) seed {seed}" + (f"  GT {o['psnr']:.2f}/{o['ssim']:.2f}" if o else f"  test {r['test_psnr']:.2f}")))
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        rows.append(np.hstack(row))
    top = label(cv2.resize(src, (1024, 512)), "input: render-only panorama")
    rows.insert(0, np.hstack([top] + [np.zeros_like(top)] * (len(args.seeds) - 1)))
    cv2.imwrite(str(out / "grid.png"), np.vstack(rows)[..., ::-1])


if __name__ == "__main__":
    main()
