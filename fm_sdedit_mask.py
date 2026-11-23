"""Mask-aware SDEdit in IMAGE space (pixel-space flow model, PixelDiT): strength 0.5 on the part of the
render-only panorama covered by data, strength 1.0 on the grey part no data reached. No render gradient.

With t = sigma (1 = pure noise, 0 = clean) and x the pixel-space sample (values in [-1, 1]):
  start: x = eps (pure noise everywhere, t = 1)
  while t_i >= switch (0.5):  x[covered] <- (1 - t_i) ref + t_i eps_i     (fresh eps_i every step)
                              x <- x + (t_{i+1} - t_i) v_theta(x)          (flow step on the whole image)
  once  t_i <  switch:        x <- x + (t_{i+1} - t_i) v_theta(x)          (no more overwriting)
so the covered region is effectively SDEdit from t = 0.5 and the grey region is generated from t = 1.

The grey area is defined on the reference image itself: pixels that render-only optimisation never moved
from its mid-grey start (all channels exactly 128/255), cleaned with a small morphological opening.
A --feather (px) blur softens the mask boundary.

  SCENE=corncho1 python fm_sdedit_mask.py --input results/renderopt_corncho1/I_renderopt_pixel.png \
      --select above --diffuse texture --sat 0.98 --gt ... --gt-rows 0.68 --out results/sdedit_mask_corncho1
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import RESULTS
from data import Scene, pick_device
from fm_guide import Pixels, Renderer, camera_to_lookup, gt_scores, label, load_gt, load_samples, splat_min
from fm_guide_pixel import NEGATIVE, load_pixeldit


def grey_mask(img_u8, open_px=3):
    """True where the render-only panorama is still exactly its mid-grey initialisation."""
    grey = np.all(img_u8 == 128, axis=2).astype(np.uint8)
    k = np.ones((open_px, open_px), np.uint8)
    return cv2.morphologyEx(grey, cv2.MORPH_OPEN, k) > 0


@torch.no_grad()
def sample(pipe, embeds, ref, covered, steps, cfg, switch, seed, roll, device, snaps=None):
    """ref: (1,3,H,W) in [-1,1]; covered: (1,1,H,W) in [0,1] (1 = keep reference while t >= switch)."""
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=4.0)
    sched.set_timesteps(steps, device=device)
    _, _, H, W = ref.shape
    gen = torch.Generator(device=device).manual_seed(seed)
    roll_gen = torch.Generator().manual_seed(20_000 + seed)
    x = torch.randn((1, 3, H, W), generator=gen, device=device, dtype=torch.float32)
    tr = pipe.transformer
    replaced = 0
    to_img = lambda z: ((z[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()  # noqa: E731
    last_rep = max(k for k in range(len(sched.timesteps)) if float(sched.sigmas[k]) >= switch)
    for i, t in enumerate(sched.timesteps):
        sigma = float(sched.sigmas[i])
        if sigma >= switch:  # covered region := reference noised to the current level
            eps = torch.randn(x.shape, generator=gen, device=device, dtype=torch.float32)
            x = covered * ((1 - sigma) * ref + sigma * eps) + (1 - covered) * x
            replaced += 1
        shift = int(torch.randint(0, W, (1,), generator=roll_gen)) if roll else 0
        xin = torch.roll(x, shift, dims=3).to(tr.dtype).repeat(2, 1, 1, 1)
        out = tr(xin, timestep=t.expand(2).to(tr.dtype), encoder_hidden_states=embeds, encoder_attention_mask=None).sample.float()
        v_u, v_c = out.chunk(2)
        v = torch.roll(v_u + cfg * (v_c - v_u), -shift, dims=3)
        if snaps is not None and i in (last_rep, last_rep + 1):  # around the switch: noisy input and clean prediction
            tag = "last_overwrite" if i == last_rep else "first_joint"
            snaps[tag] = {"t": sigma, "x_t": to_img(x), "x0_hat": to_img(x - sigma * v)}
        x = sched.step(v, t, x, return_dict=False)[0]
    img = ((x[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
    return (img * 255 + 0.5).astype(np.uint8), replaced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="render-only panorama PNG (camera layout)")
    ap.add_argument("--switch", type=float, default=0.5, help="noise level t where the covered region stops being overwritten")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=2.75)
    ap.add_argument("--width", type=int, default=1408)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--roll", type=int, default=1)
    ap.add_argument("--mask", choices=["grey", "none"], default="grey",
                    help="none: plain image-space SDEdit, one strength (= --switch) for the whole image")
    ap.add_argument("--feather", type=float, default=4.0, help="Gaussian sigma (px) softening the mask edge")
    ap.add_argument("--prompt", default="This is a panorama. The image shows an indoor room.")
    ap.add_argument("--max-samples", type=int, default=8_000_000)
    ap.add_argument("--sat", type=float, default=0.98)
    ap.add_argument("--select", choices=["cans", "above"], default="cans")
    ap.add_argument("--z-min", type=float, default=0.008)
    ap.add_argument("--diffuse", choices=["min", "texture"], default="texture")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--gt-rows", type=float, default=1.0)
    ap.add_argument("--out", default=str(RESULTS / "sdedit_mask"))
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

    src = cv2.imread(args.input)[..., ::-1].copy()  # RGB uint8, camera layout
    grey = grey_mask(src)
    size = (args.width, args.height)
    ref_u8 = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
    cov = cv2.resize((~grey).astype(np.float32), size, interpolation=cv2.INTER_AREA)
    cov = (cov > 0.5).astype(np.float32)
    if args.feather > 0:
        cov = cv2.GaussianBlur(cov, (0, 0), args.feather)
    if args.mask == "none":  # every pixel is "covered": the last overwrite at t >= switch is the SDEdit start
        cov = np.ones_like(cov)
    cv2.imwrite(str(out / "mask_covered.png"), (cov * 255).astype(np.uint8))
    ref = torch.from_numpy(ref_u8).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)
    covered = torch.from_numpy(cov)[None, None].to(device)

    src_t = torch.from_numpy(src).permute(2, 0, 1).float().div(255).to(device)
    metrics = {"args": vars(args), "grey_fraction": float(grey.mean()),
               "runs": {"splat_min_diffuse": score(sm, sm_valid), "input_renderopt": score(camera_to_lookup(src_t))}}
    print("grey fraction", round(float(grey.mean()), 3), "| input", json.dumps(metrics["runs"]["input_renderopt"]), flush=True)

    pipe = load_pixeldit(device)
    embeds, _ = pipe.encode_prompt(args.prompt, negative_prompt=NEGATIVE, device=device, do_classifier_free_guidance=True)
    embeds = embeds.clone()
    pipe.text_encoder = None
    torch.cuda.empty_cache()

    tiles = [label(cv2.resize(src, (1024, 512)), "input: render-only panorama"),
             label(cv2.resize(np.repeat((cov[..., None] * 255).astype(np.uint8), 3, 2), (1024, 512)), "mask: white = covered (strength 0.5)")]
    for seed in args.seeds:
        name = f"mask_s{seed}"
        t = time.time()
        snaps = {}
        img, replaced = sample(pipe, embeds, ref, covered, args.steps, args.cfg, args.switch, seed, args.roll, device, snaps)
        u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)  # noqa: E731
        lo, fj = snaps["last_overwrite"], snaps["first_joint"]
        strip = [label(cv2.resize(u8(lo["x_t"]), (1024, 512)), f"x_t at last overwrite (t={lo['t']:.2f})"),
                 label(cv2.resize(u8(lo["x0_hat"]), (1024, 512)), f"x0_hat at last overwrite (t={lo['t']:.2f})"),
                 label(cv2.resize(u8(fj["x0_hat"]), (1024, 512)), f"x0_hat at first joint step (t={fj['t']:.2f})"),
                 label(cv2.resize(img, (1024, 512)), f"final, seed {seed}")]
        cv2.imwrite(str(out / f"switch_snapshots_s{seed}.png"), np.vstack([np.hstack(strip[:2]), np.hstack(strip[2:])])[..., ::-1])
        for tag, d in snaps.items():
            for k in ("x_t", "x0_hat"):
                cv2.imwrite(str(out / f"snap_s{seed}_{tag}_{k}.png"), u8(d[k])[..., ::-1])
        cv2.imwrite(str(out / f"I_{name}.png"), img[..., ::-1])
        pano = camera_to_lookup(torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(device))
        r = score(pano)
        r.update(steps_overwritten=replaced, seconds=round(time.time() - t, 1))
        metrics["runs"][name] = r
        print(name, json.dumps(r), flush=True)
        a = r.get("gt_all") or {}
        tiles.append(label(cv2.resize(img, (1024, 512), interpolation=cv2.INTER_CUBIC),
                           f"seed {seed}" + (f"  GT all {a['psnr']:.2f}/{a['ssim']:.2f}" if a else f"  test {r['test_psnr']:.2f}")))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)])
    cv2.imwrite(str(out / "grid.png"), grid[..., ::-1])


if __name__ == "__main__":
    main()
