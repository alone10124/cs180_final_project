"""Baseline without any flow-matching velocity: optimise the panorama on the render loss alone.

Same pixel selection, diffuse, renderer, saturation mask, per-channel gain and scoring as fm_guide.py;
the generator's velocity u_theta is simply absent. Two parameterisations:

  --space pixel   the 2048 x 1024 sRGB panorama itself (clamped to [0, 1]), initialised mid-grey
  --space latent  the FLUX.1-dev latent (16 x 128 x 256), decoded by the frozen FLUX VAE, initialised
                  with the latent of a mid-grey image; the decoder is the only "prior"

  SCENE=corncho1 python fm_renderopt.py --space pixel --select above --diffuse texture --sat 0.98 --gt ... --out ...
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

H, W = 1024, 2048


def load_vae(device):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    vae.enable_gradient_checkpointing()
    vae.requires_grad_(False)
    return vae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=["pixel", "latent"], required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=None, help="Adam step; default 0.01 (pixel) / 0.05 (latent)")
    ap.add_argument("--max-samples", type=int, default=8_000_000)
    ap.add_argument("--sat", type=float, default=0.98)
    ap.add_argument("--select", choices=["cans", "above"], default="cans")
    ap.add_argument("--z-min", type=float, default=0.008)
    ap.add_argument("--diffuse", choices=["min", "texture"], default="texture")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--gt-rows", type=float, default=1.0)
    ap.add_argument("--out", default=str(RESULTS / "renderopt"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    lr = args.lr or (0.01 if args.space == "pixel" else 0.05)

    scene = Scene(device)
    pixels = Pixels(scene, args.select, args.diffuse, args.z_min)
    data = load_samples(scene, pixels, device, args.max_samples)
    train, test = Renderer(data["train"], args.sat), Renderer(data["test"])
    gts, gt_rows = load_gt(args.gt, args.gt_rows) if args.gt else (None, None)
    sm, sm_valid = splat_min(data["train"])

    def score(pano_lin, valid=None):
        _, g = train.loss(pano_lin)
        r = {"train_mse": float(train.loss(pano_lin, g)[0]), "train_psnr": train.psnr(pano_lin, g),
             "test_psnr": test.psnr(pano_lin, g), "gain": g.tolist()}
        fig = None
        if gts is not None:
            o = gt_scores(pano_lin, sm_valid, gts, gt_rows)
            r["gt_observed"] = o[0] if o else None
            if valid is None:
                a = gt_scores(pano_lin, torch.ones_like(sm_valid), gts, gt_rows)
                r["gt_all"] = a[0] if a else None
                fig = a
        return r, fig

    metrics = {"args": vars(args), "lr": lr, "splat_min_diffuse": score(sm, sm_valid)[0]}

    if args.space == "pixel":
        param = torch.full((1, 3, H, W), 0.5, device=device, requires_grad=True)
        decode = lambda p: p[0]  # noqa: E731
    else:
        vae = load_vae(device)
        with torch.no_grad():
            grey = torch.zeros((1, 3, H, W), device=device, dtype=torch.bfloat16)  # 0.5 grey in [-1, 1]
            z0 = vae.encode(grey).latent_dist.mean.float()
        param = z0.clone().requires_grad_(True)
        decode = lambda z: vae.decode(z.to(vae.dtype), return_dict=False)[0][0].float() * 0.5 + 0.5  # noqa: E731

    opt = torch.optim.Adam([param], lr=lr)
    curve, t = [], time.time()
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        loss, _ = train.loss(camera_to_lookup(decode(param)))
        loss.backward()
        opt.step()
        if args.space == "pixel":
            with torch.no_grad():
                param.clamp_(0, 1)
        curve.append(float(loss))
        if it % max(1, args.iters // 20) == 0 or it == args.iters - 1:
            print(f"iter {it} loss {float(loss):.6f} ({time.time() - t:.0f}s)", flush=True)

    with torch.no_grad():
        image = decode(param).clamp(0, 1)
    img8 = (image.permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(out / f"I_renderopt_{args.space}.png"), img8[..., ::-1])
    run, fig = score(camera_to_lookup(image))
    run.update(seconds=round(time.time() - t, 1), loss_first=curve[0], loss_last=curve[-1])
    metrics[f"renderopt_{args.space}"] = run
    metrics[f"curve_{args.space}"] = curve[:: max(1, len(curve) // 200)]
    print(f"renderopt_{args.space}", json.dumps(run), flush=True)
    if fig is not None:
        m_, est, refd = fig
        cv2.imwrite(str(out / f"gt_compare_{args.space}.png"),
                    label((np.vstack([est, refd])[:, ::-1] * 255).astype(np.uint8).copy(),
                          f"render-only {args.space}: PSNR {m_['psnr']:.2f} SSIM {m_['ssim']:.3f}")[..., ::-1])
    path = out / "metrics.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(metrics)
    path.write_text(json.dumps(old, indent=2))


if __name__ == "__main__":
    main()
