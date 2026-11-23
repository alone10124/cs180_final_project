"""Render-guided flow matching: generate the environment panorama I with a frozen FLUX.1-dev +
DiT360 panorama model, steering every sampling step with the gradient of a plain reflection
render loss on the Cans photos. Nothing is trained.

  HF_HOME=/overflow/xiang/hf_cache HF_HUB_OFFLINE=1 python fm_guide.py --lam 0 0.1 0.3 1 --seeds 0 1

Render model (linear in I):  photo_lin(x) = diffuse_min(x) + g * lin(I)[r(x)]
  diffuse_min  minimum colour of each can point across training views (label.estimate), fixed
  r(x)         mirror reflection direction of pixel x (data.Scene)
  g            per-channel gain, closed-form least squares each step (unknown exposure)
Guidance: x0_hat = x_t - sigma * v ;  v <- v + lam * |v|/|grad| * grad_{x0_hat} L(decode(x0_hat)).

Writes results/fm_guide/: I_lam*_s*.png (camera layout), grid.png, metrics.json,
loss_curves.png, test_views.png.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import RESULTS, SCENE
from data import GAMMA, Scene, pick_device, to_linear
from envmap import chunks, compare, observed, save_png
from models import SRM_H, SRM_W
from label import atlas_index, estimate

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party" / "DiT360"))

LOOK_W, LOOK_H = 512, 256  # lookup resolution, ~ the paper's 500 x 250
SRGB = 2.2
CACHE_DIR = Path("/overflow/xiang/fm_cache")
DEFAULT_PROMPT = ("This is a panorama. The image shows a dim indoor room with large bright windows, "
                  "a dark ceiling, walls and a carpeted floor.")


# ----------------------------------------------------------------------------- data
class Pixels:
    """Which pixels carry the specular signal, and their fixed diffuse layer.

    select "cans":  the two fitted cans (Cans scene).   "above": any surface above z_min (e.g. a bag on paper).
    diffuse "min":  per-point minimum colour across training views (label.estimate, cans only).
            "texture": the dataset's texture, which the paper builds as the robust (IRLS) minimum
                       of each surface point's intensity across views (supplement, Sec. G).
    """

    def __init__(self, scene, select, diffuse, z_min):
        self.scene, self.select, self.diffuse, self.z_min = scene, select, diffuse, z_min
        self.atlas = estimate(scene, scene.train).reshape(-1, 3) if diffuse == "min" else None

    def __call__(self, b):
        """(mask B,H,W bool; display-encoded diffuse B,H,W,3)."""
        if self.select == "cans":
            m = observed(b, "both")
        else:
            m = observed(b, "all") & (b["pos"][:, 2] > self.z_min)
        if self.diffuse == "min":
            index, _ = atlas_index(b, self.scene.cans)
            d = self.atlas[index]
            m = m & (d.sum(-1) > 0)
        else:
            d = b["diffuse"].permute(0, 2, 3, 1)
        return m, d


def build_samples(scene, frames, pixels, max_n, seed=0):
    """Selected pixels of `frames`: reflection dir, linear diffuse, linear photo (fp16, GPU)."""
    dirs, diff, rgb = [], [], []
    for idx in chunks(frames):
        b = scene.batch(idx)
        m, d = pixels(b)
        dirs.append(b["refl"].permute(0, 2, 3, 1)[m].half())
        diff.append(to_linear(d[m]).half())
        rgb.append(to_linear(b["rgb"].permute(0, 2, 3, 1)[m]).half())
    out = {k: torch.cat(v) for k, v in (("dir", dirs), ("diff", diff), ("rgb", rgb))}
    n = len(out["dir"])
    if n > max_n:
        keep = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:max_n].to(out["dir"].device)
        out = {k: v[keep] for k, v in out.items()}
    return out


def load_samples(scene, pixels, device, max_n):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = "cans" if SCENE == "" else SCENE  # keeps the existing Cans cache file name
    if (pixels.select, pixels.diffuse) != ("cans", "min"):
        tag += f"_{pixels.select}{pixels.z_min:g}_{pixels.diffuse}"
    path = CACHE_DIR / f"{tag}_samples_{max_n}.pt"
    if path.exists():
        return {s: {k: v.to(device) for k, v in d.items()} for s, d in torch.load(path).items()}
    t = time.time()
    data = {"train": build_samples(scene, scene.train, pixels, max_n),
            "test": build_samples(scene, scene.test, pixels, max_n)}
    torch.save({s: {k: v.cpu() for k, v in d.items()} for s, d in data.items()}, path)
    path.chmod(0o664)
    print(f"samples in {time.time() - t:.0f}s: train {len(data['train']['dir'])}, test {len(data['test']['dir'])}")
    return data


def load_gt(path, rows_frac):
    """Ground-truth photo panorama -> linear lookup layout at the SRM size, plus its row mask.

    `rows_frac`: fraction of the sphere's height the image covers from the zenith (a figure crop).
    Returns (camera-layout version, mirrored version) so the handedness can be tested both ways."""
    if str(path).endswith(".hdr"):  # synthetic ground truth: linear radiance, already in lookup layout, full sphere
        env = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32)
        env = cv2.resize(env, (SRM_W, SRM_H), interpolation=cv2.INTER_AREA)
        rows = np.ones((SRM_H, SRM_W), bool)
        return {"as_is": env, "mirrored": np.ascontiguousarray(env[:, ::-1])}, rows
    img = cv2.imread(str(path))[..., ::-1].astype(np.float32) / 255
    h_full = int(round(img.shape[0] / rows_frac))
    full = np.zeros((h_full, img.shape[1], 3), np.float32)
    full[: img.shape[0]] = img ** SRGB
    full = cv2.resize(full, (SRM_W, SRM_H), interpolation=cv2.INTER_AREA)
    rows = np.arange(SRM_H)[:, None] < int(SRM_H * rows_frac) - 1
    return {"mirrored": np.ascontiguousarray(full[:, ::-1]), "as_is": full}, np.broadcast_to(rows, (SRM_H, SRM_W))


def gt_scores(pano_lookup, valid, gts, rows):
    """envmap.compare (gain + blur matching, azimuth search) against the GT in both handedness."""
    pano = cv2.resize(pano_lookup.permute(1, 2, 0).float().cpu().numpy(), (SRM_W, SRM_H), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(valid.float().cpu().numpy(), (SRM_W, SRM_H), interpolation=cv2.INTER_NEAREST) > 0.5
    best = None
    for hand, ref in gts.items():
        m, est, refd = compare(pano, valid, ref, region=rows, align=True)
        if "psnr" in m and (best is None or m["psnr"] > best[0]["psnr"]):
            best = (dict(m, handedness=hand), est, refd)
    return best


# ----------------------------------------------------------------------------- render
def grid_coords(dirs):
    """grid_sample coordinates of reflection directions in a lookup-layout panorama padded by one
    column on each side (azimuth wrap). Same mapping as SRMModel.lookup / common.equirect_sample."""
    d = dirs.float()
    u = (torch.atan2(d[:, 1], d[:, 0]) / (2 * math.pi) + 0.5) * LOOK_W - 0.5  # texel index space
    v = torch.acos(d[:, 2].clamp(-1, 1)) / math.pi * LOOK_H - 0.5
    x = (2 * (u + 1) + 1) / (LOOK_W + 2) - 1
    y = (2 * v + 1) / LOOK_H - 1
    return torch.stack([x, y], -1)[None, None]  # 1, 1, N, 2


def lookup(pano_lin, grid):
    """pano_lin (3, LOOK_H, LOOK_W) lookup layout -> (N, 3) bilinear samples."""
    p = torch.cat([pano_lin[:, :, -1:], pano_lin, pano_lin[:, :, :1]], dim=2)[None]
    return F.grid_sample(p, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T


def camera_to_lookup(image):
    """(3, h, w) display image in camera-panorama layout -> linear (3, LOOK_H, LOOK_W) lookup layout."""
    small = F.adaptive_avg_pool2d(image[None].clamp(0, 1), (LOOK_H, LOOK_W))[0]
    return small.flip(-1) ** SRGB


def fit_gain(spec, target):
    return ((spec * target).sum(0) / (spec * spec).sum(0).clamp(min=1e-12)).clamp(min=0).detach()


class Renderer:
    def __init__(self, samples, sat=1.0):
        """sat < 1 drops photo pixels whose display value reaches `sat` in any channel (clipped
        highlights: the true radiance is unknown, and an 8-bit panorama cannot exceed 1)."""
        self.s = samples
        self.grid = grid_coords(samples["dir"])
        self.target = (samples["rgb"] - samples["diff"]).float()
        self.keep = (samples["rgb"].float().max(1).values < sat ** GAMMA) if sat < 1 else None

    def loss(self, pano_lin, gain=None):
        spec, target = lookup(pano_lin, self.grid), self.target
        if self.keep is not None:
            spec, target = spec[self.keep], target[self.keep]
        g = fit_gain(spec, target) if gain is None else gain
        return ((g * spec - target) ** 2).mean(), g

    @torch.no_grad()
    def psnr(self, pano_lin, gain):
        """PSNR of the re-rendered can pixels, display space (pixel = linear ** (1/GAMMA))."""
        pred = (self.s["diff"].float() + gain * lookup(pano_lin, self.grid)).clamp(1e-6, 1) ** (1 / GAMMA)
        true = self.s["rgb"].float().clamp(1e-6, 1) ** (1 / GAMMA)
        return float(-10 * torch.log10(((pred - true) ** 2).mean()))


def splat_min(samples):
    """Data-only least-squares panorama of the same loss: mean target per lookup texel."""
    tgt = (samples["rgb"] - samples["diff"]).float()
    d = samples["dir"].float()
    u = ((torch.atan2(d[:, 1], d[:, 0]) / (2 * math.pi) + 0.5) * LOOK_W).long().clamp(0, LOOK_W - 1)
    v = (torch.acos(d[:, 2].clamp(-1, 1)) / math.pi * LOOK_H).long().clamp(0, LOOK_H - 1)
    flat = v * LOOK_W + u
    tot = torch.zeros(LOOK_H * LOOK_W, 3, device=d.device).index_add_(0, flat, tgt)
    cnt = torch.zeros(LOOK_H * LOOK_W, device=d.device).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
    pano = (tot / cnt.clamp(min=1)[:, None]).clamp(min=0)
    return pano.T.reshape(3, LOOK_H, LOOK_W), (cnt > 0).reshape(LOOK_H, LOOK_W)


# ----------------------------------------------------------------------------- guidance
class Guide:
    """Called by the pipeline right before scheduler.step; returns the modified velocity."""

    def __init__(self, renderer, lam, stop_frac, steps, on="x0", clip=0.0, recur=1, seed=0,
                 decay=None, decay_end=0.5, decay_floor=0.01):
        self.r, self.lam, self.stop, self.on = renderer, lam, int(round(stop_frac * steps)), on
        # decay="exp": lam_i = lam * floor ** (f / decay_end) for f = i / steps < decay_end, then 0
        self.steps, self.decay, self.decay_end, self.decay_floor = steps, decay, decay_end, decay_floor
        if decay == "exp":
            self.stop = min(self.stop, int(np.ceil(decay_end * steps)))
        self.clip = clip  # > 0: cap each element of the guidance step at clip * rms(v)
        self.recur = recur  # self-recurrence: each guided step is taken this many times
        self.seed = seed
        self.gen = None
        self.losses, self.skipped = [], 0

    def lam_at(self, i):
        """Guidance weight at sampling step i (constant, or exponentially decaying to ~0 by decay_end)."""
        if i >= self.stop:
            return 0.0
        if self.decay == "exp":
            f = i / self.steps
            return self.lam * self.decay_floor ** (f / self.decay_end) if f < self.decay_end else 0.0
        return self.lam

    def repeats(self, i):
        return self.recur if self.lam_at(i) > 0 else 1

    def renoise(self, x, sigma, sigma_next, n_h, n_w):
        """Exact forward kernel of the flow from sigma_next back up to sigma (> sigma_next):
        x_sigma = a x_next + sqrt(sigma^2 - (a sigma_next)^2) eps,  a = (1 - sigma) / (1 - sigma_next).
        The fresh noise wraps across the azimuth seam like DiT360's circular pad columns."""
        sigma, sigma_next = float(sigma), float(sigma_next)
        a = (1 - sigma) / (1 - sigma_next)
        std = max(sigma ** 2 - (a * sigma_next) ** 2, 0.0) ** 0.5
        if self.gen is None:
            self.gen = torch.Generator(device=x.device).manual_seed(10_000 + self.seed)
        b, _, c = x.shape
        eps = torch.randn((b, n_h, n_w, c), generator=self.gen, device=x.device, dtype=torch.float32)
        eps = torch.cat([eps[:, :, -1:], eps, eps[:, :, :1]], dim=2).reshape(b, -1, c)
        return (a * x.float() + std * eps).to(x.dtype)

    def __call__(self, pipe, i, sigma, latents, v, n_h, n_w):
        if self.lam == 0 and i % 4:  # prior-only run: log the loss every 4 steps only
            return v
        b, _, c = latents.shape
        with torch.enable_grad():
            # x0: the one-jump clean estimate x_t - sigma v; xt: the noisy in-progress latent itself
            src = latents.float() - float(sigma) * v.float() if self.on == "x0" else latents.float()
            x0 = src.reshape(b, n_h, n_w + 2, c)[:, :, 1:-1]
            x0 = x0.detach().requires_grad_(True)
            lat = pipe._unpack_latents(x0.reshape(b, n_h * n_w, c), n_h * 16, n_w * 16, pipe.vae_scale_factor)
            lat = lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            image = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0][0].float() * 0.5 + 0.5
            loss, _ = self.r.loss(camera_to_lookup(image))
            grad = torch.autograd.grad(loss, x0)[0] if self.lam_at(i) > 0 else None
        self.losses.append((i, float(sigma), float(loss)))
        if grad is None:
            return v
        if not (torch.isfinite(loss) and torch.isfinite(grad).all()):
            self.skipped += 1  # never let a bad step poison the latents
            return v
        grad = torch.cat([grad[:, :, -1:], grad, grad[:, :, :1]], dim=2).reshape(b, -1, c)  # pad columns
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


# ----------------------------------------------------------------------------- main
def load_pipe(device, lora):
    from pipeline import DiT360Pipeline

    pipe = DiT360Pipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(lora, weight_name="adapter_model.safetensors")
    pipe.to(device)
    pipe.vae.enable_gradient_checkpointing()
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, nargs="+", default=[0, 0.1, 0.3, 1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=2.8, help="FLUX distilled guidance scale (DiT360 default)")
    ap.add_argument("--stop-frac", type=float, default=0.85, help="no render guidance after this fraction of steps")
    ap.add_argument("--max-samples", type=int, default=8_000_000)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--lora", default="Insta360-Research/DiT360-Panorama-Image-Generation")
    ap.add_argument("--out", default=str(RESULTS / "fm_guide"))
    ap.add_argument("--samples-only", action="store_true")
    ap.add_argument("--sat", type=float, default=1.0, help="drop photo pixels at/above this display value from the loss (fix 1)")
    ap.add_argument("--clip", type=float, default=0.0, help="cap each guidance-step element at clip * rms(v) (fix 2)")
    ap.add_argument("--guide-on", choices=["x0", "xt"], default="x0", help="render the jump-to-end estimate or the noisy latent")
    ap.add_argument("--recur", type=int, default=1, help="self-recurrence: re-noise and repeat each guided step this many times")
    ap.add_argument("--lam-decay", choices=["none", "exp"], default="none", help="exp: lambda decays exponentially to ~0 by --decay-end")
    ap.add_argument("--decay-end", type=float, default=0.5, help="fraction of sampling steps where the decayed lambda reaches floor, then 0")
    ap.add_argument("--decay-floor", type=float, default=0.01, help="lambda(decay_end) / lambda_0")
    ap.add_argument("--select", choices=["cans", "above"], default="cans", help="specular pixels: fitted cans, or surfaces above --z-min")
    ap.add_argument("--z-min", type=float, default=0.008, help="metres above the ground plane for --select above")
    ap.add_argument("--diffuse", choices=["min", "texture"], default="min", help="min-colour atlas (cans) or the dataset texture (paper's robust min)")
    ap.add_argument("--gt", default=None, help="ground-truth panorama image (camera layout) for panorama PSNR/SSIM")
    ap.add_argument("--gt-rows", type=float, default=1.0, help="fraction of the sphere height the GT image covers from the zenith")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    scene = Scene(device)
    pixels = Pixels(scene, args.select, args.diffuse, args.z_min)
    data = load_samples(scene, pixels, device, args.max_samples)
    gts, gt_rows = load_gt(args.gt, args.gt_rows) if args.gt else (None, None)
    train, test = Renderer(data["train"], args.sat), Renderer(data["test"])
    metrics = {"samples": {s: len(d["dir"]) for s, d in data.items()}, "args": vars(args), "runs": {}}

    gt_figs = {}

    def score(name, pano_lin, valid=None):
        _, g = train.loss(pano_lin)
        metrics["runs"][name] = {"train_mse": float(train.loss(pano_lin, g)[0]), "train_psnr": train.psnr(pano_lin, g),
                                 "test_psnr": test.psnr(pano_lin, g), "gain": g.tolist()}
        if gts is not None:  # panorama vs ground truth: on texels the object observed, and (if defined) everywhere
            r = gt_scores(pano_lin, sm_valid, gts, gt_rows)
            metrics["runs"][name]["gt_observed"] = r[0] if r else None
            if valid is None:
                r_all = gt_scores(pano_lin, torch.ones_like(sm_valid), gts, gt_rows)
                metrics["runs"][name]["gt_all"] = r_all[0] if r_all else None
                gt_figs[name] = r_all
        print(name, json.dumps(metrics["runs"][name]))
        return g

    # Baselines: the data-only least-squares panorama of this exact loss, and the repo's splat panorama.
    gains = {}
    sm, sm_valid = splat_min(data["train"])
    gains["splat_min_diffuse"] = score("splat_min_diffuse", sm, valid=sm_valid)
    if SCENE == "":
        splat = torch.from_numpy(np.load(RESULTS / "splat_both" / "pano.npy")).permute(2, 0, 1).to(device)
        score("splat_both", F.interpolate(splat[None], (LOOK_H, LOOK_W), mode="bilinear", align_corners=False)[0])
    if args.samples_only:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return

    pipe = load_pipe(device, args.lora)
    with torch.no_grad():
        pe, ppe, _ = pipe.encode_prompt(prompt=args.prompt, prompt_2=None, device=device, max_sequence_length=512)
    pipe.text_encoder, pipe.text_encoder_2 = None, None
    torch.cuda.empty_cache()

    curves, images = {}, {}
    for lam in args.lam:
        for seed in args.seeds:
            name = f"lam{lam:g}_s{seed}"
            guide = Guide(train, lam, args.stop_frac, args.steps, args.guide_on, args.clip, args.recur, seed,
                          None if args.lam_decay == "none" else args.lam_decay, args.decay_end, args.decay_floor)
            t = time.time()
            image = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe, width=2048, height=1024,
                         num_inference_steps=args.steps, guidance_scale=args.cfg,
                         generator=torch.Generator(device=device).manual_seed(seed), guide=guide).images[0]
            image.save(out / f"I_{name}.png")
            img = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float().div(255).to(device)
            pano = camera_to_lookup(img)
            gains[name] = score(name, pano)
            metrics["runs"][name]["seconds"] = round(time.time() - t, 1)
            metrics["runs"][name]["skipped_steps"] = guide.skipped
            curves[name] = guide.losses
            images[name] = (np.asarray(image), pano)
            (out / "metrics.json").write_text(json.dumps(dict(metrics, curves=curves), indent=2))

    report(out, args, metrics, curves, images, gains, sm, sm_valid, scene, pixels, device, gt_figs)


# ----------------------------------------------------------------------------- figures
def label(img, text):
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    return img


def report(out, args, metrics, curves, images, gains, sm, sm_valid, scene, pixels, device, gt_figs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def disp(lin_lookup):  # lookup-layout linear -> camera-layout uint8, exposure-normalised
        x = lin_lookup.flip(-1).permute(1, 2, 0).cpu().numpy()
        x = x / max(np.percentile(x, 99.5), 1e-6)
        return cv2.resize((np.clip(x, 0, 1) ** (1 / SRGB) * 255).astype(np.uint8), (1024, 512))

    rows = []
    for lam in args.lam:
        rows.append(np.hstack([label(cv2.resize(images[f"lam{lam:g}_s{s}"][0], (1024, 512)),
                                     f"lam={lam:g} seed={s} test={metrics['runs'][f'lam{lam:g}_s{s}']['test_psnr']:.2f}dB")
                               for s in args.seeds]))
    base = [label(disp(sm * sm_valid), f"splat (min diffuse) test={metrics['runs']['splat_min_diffuse']['test_psnr']:.2f}dB")]
    if args.gt:
        gt_img = cv2.imread(str(args.gt))[..., ::-1]
        full = np.zeros((int(round(gt_img.shape[0] / args.gt_rows)), gt_img.shape[1], 3), np.uint8)
        full[: gt_img.shape[0]] = gt_img
        base.append(label(cv2.resize(full, (1024, 512), interpolation=cv2.INTER_AREA), "ground truth"))
    elif SCENE == "":
        paper = ROOT / "results" / "p2r_paper" / "input_panorama.png"
        base.append(label(cv2.resize(cv2.imread(str(paper))[..., ::-1], (1024, 512)), "paper Fig. 6(d) (other scene)"))
    while len(base) < len(args.seeds):
        base.append(np.zeros_like(base[0]))
    rows.append(np.hstack(base[: len(args.seeds)]))
    cv2.imwrite(str(out / "grid.png"), np.vstack(rows)[..., ::-1])

    plt.figure(figsize=(7, 4))
    for name, c in curves.items():
        c = np.array(c)
        plt.plot(c[:, 0], c[:, 2], label=name, ls="-" if "lam0_" not in name else "--")
    plt.yscale("log"); plt.xlabel("sampling step"); plt.ylabel("render MSE of x0_hat (train)")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out / "loss_curves.png", dpi=120)

    # Test views: photo | splat(min diffuse) re-render | best guided re-render, cropped to the cans.
    guided = [n for n in metrics["runs"] if n.startswith("lam") and not n.startswith("lam0_")]
    best = max(guided or [n for n in metrics["runs"] if n.startswith("lam")], key=lambda n: metrics["runs"][n]["test_psnr"])
    metrics["best"] = best
    frames = scene.test[np.linspace(0, len(scene.test) - 1, 4).astype(int)]
    b = scene.batch(frames)
    m, d = pixels(b)
    d_lin = to_linear(d)
    refl = b["refl"].permute(0, 2, 3, 1)
    photo = b["rgb"].permute(0, 2, 3, 1)
    tiles = []
    for pano, g in ((sm, gains["splat_min_diffuse"]), (images[best][1], gains[best])):
        spec = lookup(pano, grid_coords(refl[m]))
        img = photo.clone()
        img[m] = ((d_lin[m] + torch.as_tensor(g, device=device) * spec).clamp(1e-6, 1)) ** (1 / GAMMA)
        tiles.append(img)
    ys, xs = torch.nonzero(m.any(0), as_tuple=True)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    rows = [np.hstack([(t[k, y0:y1, x0:x1].cpu().numpy() * 255).astype(np.uint8) for t in (photo, *tiles)])
            for k in range(len(frames))]
    sheet = np.vstack(rows)
    cv2.imwrite(str(out / "test_views.png"), label(sheet, f"photo | splat min-diffuse | {best}")[..., ::-1])
    # Ground truth vs estimates after gain/blur/azimuth matching (envmap.compare's display images).
    if gt_figs:
        rows = []
        for name in ["lam0_s0", best]:
            if gt_figs.get(name):
                m_, est, refd = gt_figs[name]
                rows.append(label((np.vstack([est, refd])[:, ::-1] * 255).astype(np.uint8).copy(),
                                  f"{name}: PSNR {m_['psnr']:.2f} SSIM {m_['ssim']:.3f} ({m_['handedness']})"))
        if rows:
            cv2.imwrite(str(out / "gt_compare.png"), np.hstack(rows)[..., ::-1])
    (out / "metrics.json").write_text(json.dumps(dict(metrics, curves=curves), indent=2))
    print("best", best, json.dumps(metrics["runs"][best]))


if __name__ == "__main__":
    main()
