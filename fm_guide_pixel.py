"""Render-guided flow matching in PIXEL space (no VAE): same method as fm_guide.py, with the
latent-space FLUX.1-dev + DiT360 generator replaced by NVIDIA PixelDiT-1300M-1024px, a text-to-image
flow-matching DiT that predicts the velocity directly on RGB pixels.

Everything else is fm_guide.py's: pixel selection, fixed diffuse, mirror render, loss, per-channel gain,
x_t / x0_hat guidance, saturation mask, per-element clip, self-recurrence (re-noise) and evaluation.
The only change in the guidance is that "decode" is the identity: the render gradient is taken directly
with respect to the pixels of x_t, so it lands exactly on the texels the object reflects.

Differences forced by the model (not by choice):
  - PixelDiT is not panorama-trained (there is no DiT360-style panorama LoRA for any pixel-space model);
    only the prompt asks for an equirectangular image.
  - It has no circular padding; --roll (on by default) rolls the image by a random azimuth before every
    transformer call and rolls the velocity back, the usual way to make a non-panoramic model seamless.
  - It uses true classifier-free guidance (2 forward passes, scale 2.75) and a Gemma-2 text encoder.
  - Resolution 1408 x 704 (its 2:1 bucket at 1024px) instead of 2048 x 1024.

  SCENE=corncho1 python fm_guide_pixel.py --select above --diffuse texture --gt ... --recur 3 --lam 0 0.333
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import RESULTS, SCENE
from data import Scene, pick_device
from fm_guide import (DEFAULT_PROMPT, Guide, Pixels, Renderer, camera_to_lookup, gt_scores, load_gt,
                      load_samples, report, splat_min)

PIXELDIT = Path("/overflow/xiang/hf_cache/hub/models--BiliSakura--PixelDiT-diffusers/snapshots/"
                "fbad4502849d7159bdc51f8d6ebbfcf0a87d9b87/PixelDiT-T2I-1024")
NEGATIVE = "low quality, worst quality, over-saturated, blurry, deformed, watermark"  # PixelDiT default


class PixelGuide(Guide):
    """fm_guide.Guide on a (1, 3, H, W) pixel tensor in [-1, 1]: decode(x) = x * 0.5 + 0.5."""

    def renoise(self, x, sigma, sigma_next, *_):
        sigma, sigma_next = float(sigma), float(sigma_next)
        a = (1 - sigma) / (1 - sigma_next)
        std = max(sigma ** 2 - (a * sigma_next) ** 2, 0.0) ** 0.5
        if self.gen is None:
            self.gen = torch.Generator(device=x.device).manual_seed(10_000 + self.seed)
        eps = torch.randn(x.shape, generator=self.gen, device=x.device, dtype=torch.float32)
        return (a * x.float() + std * eps).to(x.dtype)

    def __call__(self, i, sigma, x, v):
        if self.lam == 0 and i % 4:
            return v
        with torch.enable_grad():
            src = x.float() - float(sigma) * v.float() if self.on == "x0" else x.float()
            src = src.detach().requires_grad_(True)
            image = src[0] * 0.5 + 0.5  # identity "decoder"
            loss, _ = self.r.loss(camera_to_lookup(image))
            grad = torch.autograd.grad(loss, src)[0] if self.lam_at(i) > 0 else None
        self.losses.append((i, float(sigma), float(loss)))
        if grad is None:
            return v
        if not (torch.isfinite(loss) and torch.isfinite(grad).all()):
            self.skipped += 1
            return v
        vf = v.float()
        step = self.lam_at(i) * vf.norm() / grad.norm().clamp(min=1e-12) * grad
        if self.clip > 0:
            cap = self.clip * vf.pow(2).mean().sqrt()
            step = step.clamp(-cap, cap)
        out = vf + step
        if not torch.isfinite(out).all():
            self.skipped += 1
            return v
        return out.to(v.dtype)


def load_pixeldit(device):
    """Assemble the pipeline from its parts. The port's own loaders resolve symlinks (the HF cache stores
    files as blobs/<hash>), which breaks its sibling-module import, so the snapshot folder goes on sys.path."""
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(PIXELDIT / "transformer"))
    import transformer_pixeldit_t2i  # noqa: E402  (read before use: model definition only)

    spec = importlib.util.spec_from_file_location("pixeldit_t2i_pipeline", PIXELDIT / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transformer = transformer_pixeldit_t2i.PixelDiTT2ITransformer2DModel.from_pretrained(
        str(PIXELDIT / "transformer"), torch_dtype=torch.bfloat16)
    text_encoder = AutoModelForCausalLM.from_pretrained(str(PIXELDIT / "text_encoder"), dtype=torch.bfloat16).get_decoder()
    tokenizer = AutoTokenizer.from_pretrained(str(PIXELDIT / "tokenizer"))
    tokenizer.padding_side = "right"
    pipe = module.PixelDiTT2IPipeline(transformer=transformer, scheduler=FlowMatchEulerDiscreteScheduler(shift=4.0),
                                      text_encoder=text_encoder, tokenizer=tokenizer)
    pipe.to(device)
    return pipe


@torch.no_grad()
def sample(pipe, embeds, guide, steps, cfg, width, height, seed, roll, device):
    """Euler flow sampling (FlowMatchEulerDiscreteScheduler, shift 4) with render guidance and
    self-recurrence, the same loop as the DiT360 pipeline hook."""
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=4.0)
    sched.set_timesteps(steps, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    roll_gen = torch.Generator().manual_seed(20_000 + seed)
    x = torch.randn((1, 3, height, width), generator=gen, device=device, dtype=torch.float32)
    tr = pipe.transformer
    for i, t in enumerate(sched.timesteps):
        for r in range(guide.repeats(i)):
            shift = int(torch.randint(0, width, (1,), generator=roll_gen)) if roll else 0
            xin = torch.roll(x, shift, dims=3).to(tr.dtype).repeat(2, 1, 1, 1)
            out = tr(xin, timestep=t.expand(2).to(tr.dtype), encoder_hidden_states=embeds,
                     encoder_attention_mask=None).sample.float()
            v_u, v_c = out.chunk(2)
            v = torch.roll(v_u + cfg * (v_c - v_u), -shift, dims=3)
            v = guide(i, sched.sigmas[sched.step_index or 0], x, v)
            x = sched.step(v, t, x, return_dict=False)[0]
            if r < guide.repeats(i) - 1:
                x = guide.renoise(x, sched.sigmas[sched.step_index - 1], sched.sigmas[sched.step_index])
                sched._step_index -= 1
    img = ((x[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
    return (img * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, nargs="+", default=[0, 0.333])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=2.75, help="PixelDiT classifier-free guidance scale (its default)")
    ap.add_argument("--width", type=int, default=1408)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--roll", type=int, default=1, help="random azimuth roll per transformer call (seamless wrap)")
    ap.add_argument("--stop-frac", type=float, default=0.85)
    ap.add_argument("--max-samples", type=int, default=8_000_000)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out", default=str(RESULTS / "fm_pixel"))
    ap.add_argument("--sat", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--guide-on", choices=["x0", "xt"], default="x0")
    ap.add_argument("--recur", type=int, default=1)
    ap.add_argument("--lam-decay", choices=["none", "exp"], default="none")
    ap.add_argument("--decay-end", type=float, default=0.5)
    ap.add_argument("--decay-floor", type=float, default=0.01)
    ap.add_argument("--select", choices=["cans", "above"], default="cans")
    ap.add_argument("--z-min", type=float, default=0.008)
    ap.add_argument("--diffuse", choices=["min", "texture"], default="min")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--gt-rows", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    scene = Scene(device)
    pixels = Pixels(scene, args.select, args.diffuse, args.z_min)
    data = load_samples(scene, pixels, device, args.max_samples)
    gts, gt_rows = load_gt(args.gt, args.gt_rows) if args.gt else (None, None)
    train, test = Renderer(data["train"], args.sat), Renderer(data["test"])
    metrics = {"model": "nvidia/PixelDiT-1300M-1024px (pixel space)", "samples": {s: len(d["dir"]) for s, d in data.items()},
               "args": vars(args), "runs": {}}
    gt_figs, gains = {}, {}
    sm, sm_valid = splat_min(data["train"])

    def score(name, pano_lin, valid=None):
        _, g = train.loss(pano_lin)
        metrics["runs"][name] = {"train_mse": float(train.loss(pano_lin, g)[0]), "train_psnr": train.psnr(pano_lin, g),
                                 "test_psnr": test.psnr(pano_lin, g), "gain": g.tolist()}
        if gts is not None:
            r = gt_scores(pano_lin, sm_valid, gts, gt_rows)
            metrics["runs"][name]["gt_observed"] = r[0] if r else None
            if valid is None:
                r_all = gt_scores(pano_lin, torch.ones_like(sm_valid), gts, gt_rows)
                metrics["runs"][name]["gt_all"] = r_all[0] if r_all else None
                gt_figs[name] = r_all
        print(name, json.dumps(metrics["runs"][name]), flush=True)
        return g

    gains["splat_min_diffuse"] = score("splat_min_diffuse", sm, valid=sm_valid)

    pipe = load_pixeldit(device)
    embeds, _ = pipe.encode_prompt(args.prompt, negative_prompt=NEGATIVE, device=device, do_classifier_free_guidance=True)
    embeds = embeds.clone()  # (2, L, D): [negative, positive]; leave inference mode
    pipe.text_encoder = None
    torch.cuda.empty_cache()

    curves, images = {}, {}
    from PIL import Image
    for lam in args.lam:
        for seed in args.seeds:
            name = f"lam{lam:g}_s{seed}"
            guide = PixelGuide(train, lam, args.stop_frac, args.steps, args.guide_on, args.clip, args.recur, seed,
                               None if args.lam_decay == "none" else args.lam_decay, args.decay_end, args.decay_floor)
            t = time.time()
            img = sample(pipe, embeds, guide, args.steps, args.cfg, args.width, args.height, seed, args.roll, device)
            Image.fromarray(img).save(out / f"I_{name}.png")
            pano = camera_to_lookup(torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(device))
            gains[name] = score(name, pano)
            metrics["runs"][name]["seconds"] = round(time.time() - t, 1)
            metrics["runs"][name]["skipped_steps"] = guide.skipped
            curves[name] = guide.losses
            import cv2
            images[name] = (cv2.resize(img, (2048, 1024), interpolation=cv2.INTER_CUBIC), pano)
            (out / "metrics.json").write_text(json.dumps(dict(metrics, curves=curves), indent=2))

    report(out, args, metrics, curves, images, gains, sm, sm_valid, scene, pixels, device, gt_figs)


if __name__ == "__main__":
    sys.exit(main())
