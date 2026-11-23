"""Minimal renderer-guided rectified-flow refinement for a trained SRM.

First run the paper reconstruction::

    python train.py srm --run srm --max-steps 8000 --lambda-g 0

Then refine its panorama against the same captured frames::

    python flow_refine.py --srm-run srm --run srm_flow

The tiny flow is deliberately scene-specific.  It learns the straight transport
from noisy copies of the reconstructed panorama back to that panorama.  During
ODE sampling, a differentiable rendering loss steers the trajectory toward SRMs
that reproduce the original RGB frames.  This is the smallest useful version of
"flow prior + measurement guidance"; a dataset-trained panorama flow can later
replace TinyPanoramaFlow without changing refine_srm_with_flow().
"""
import argparse
import json
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RUNS
from data import Scene, pick_device, select, to_display, to_linear
from models import SRM_H, SRM_W, masked_l1
from train import gate, load_srm


class CircularConv(nn.Module):
    """3x3 convolution with panorama wraparound only in longitude."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=(1, 0))

    def forward(self, x):
        return self.conv(F.pad(x, (1, 1, 0, 0), mode="circular"))


class TinyPanoramaFlow(nn.Module):
    """A small conditional velocity field v(x_t, t, initial_srm)."""

    def __init__(self, channels, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            CircularConv(channels * 2 + 1, hidden),
            nn.SiLU(),
            CircularConv(hidden, hidden),
            nn.SiLU(),
            CircularConv(hidden, channels),
        )

    def forward(self, x_t, t, initial):
        time = torch.ones_like(x_t[:, :1]) * t
        return self.net(torch.cat([x_t, initial, time], dim=1))


def lookup(srm, refl):
    """Differentiably sample M panorama bases at B reflection maps."""
    x, y, z = refl[:, 0], refl[:, 1], refl[:, 2]
    u = (torch.atan2(y, x) / (2 * math.pi) + 0.5) * SRM_W
    v = torch.acos(z.clamp(-1, 1)) / math.pi * SRM_H
    grid = torch.stack([2 * (u + 1) / (SRM_W + 2) - 1, 2 * v / SRM_H - 1], dim=-1)
    padded = torch.cat([srm[..., -1:], srm, srm[..., :1]], dim=-1)
    batch, materials = len(refl), len(srm)
    sampled = F.grid_sample(
        padded[None].expand(batch, -1, -1, -1, -1).reshape(batch * materials, 3, SRM_H, SRM_W + 2),
        grid.repeat_interleave(materials, dim=0),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled.reshape(batch, materials, 3, *refl.shape[-2:])


def frame_loss(srm, model, batch):
    """Paper renderer loss, with gradients only to ``srm``."""
    batch = gate(batch, model.cans)
    with torch.no_grad():
        diffuse_linear = to_linear(batch["diffuse"])
        residual = model.refine(batch["diffuse"]) if model.refine is not None else 0
        diffuse = (diffuse_linear + residual).clamp(min=0) * batch["mask"]
        weights = model.material(batch["diffuse"])

    specular = (weights[:, :, None] * lookup(srm, batch["refl"])).sum(dim=1)
    specular = specular * batch["vis"] * batch["mask"]
    if "spec_mask" in batch:
        specular = specular * batch["spec_mask"]
    prediction = (diffuse + specular).clamp(0, 1)
    return masked_l1(prediction, to_linear(batch["rgb"]), batch["mask"])


def train_single_scene_flow(initial, steps=150, sigma=0.15, lr=2e-4):
    """Fit a minimal rectified-flow prior to noisy copies of one reconstructed SRM."""
    channels = initial.shape[0] * initial.shape[1]
    target = initial.reshape(1, channels, SRM_H, SRM_W)
    flow = TinyPanoramaFlow(channels).to(initial.device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    for step in range(steps):
        source = (target + sigma * torch.randn_like(target)).clamp(min=0)
        t = torch.rand((), device=initial.device)
        x_t = (1 - t) * source + t * target
        desired_velocity = target - source
        loss = F.mse_loss(flow(x_t, t, target), desired_velocity)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0 or step + 1 == steps:
            print(f"flow prior {step + 1}/{steps}: {loss.item():.6f}", flush=True)
    return flow.eval(), target


def refine_srm_with_flow(
    model,
    scene,
    *,
    prior_steps=150,
    ode_steps=30,
    frame_batch=2,
    sigma=0.15,
    guidance=0.03,
    seed=0,
):
    """Return SRM bases refined by rectified flow and captured-frame guidance.

    ``model`` is the frozen result of the paper's stage-one optimization and
    ``scene`` supplies the original captured frames plus geometry buffers.
    """
    torch.manual_seed(seed)
    initial = model.srm.detach().clone()
    flow, condition = train_single_scene_flow(initial, prior_steps, sigma)
    source = (condition + sigma * torch.randn_like(condition)).clamp(min=0)
    x = source
    rng = np.random.default_rng(seed)

    for step in range(ode_steps):
        t = torch.tensor(step / max(ode_steps - 1, 1), device=initial.device)
        x = x.detach().requires_grad_(True)
        count = min(frame_batch, len(scene.train))
        indices = rng.choice(scene.train, size=count, replace=False).tolist()
        measurement = frame_loss(x.reshape_as(initial), model, scene.batch(indices))
        grad = torch.autograd.grad(measurement, x)[0]
        grad = grad / (grad.square().mean().sqrt() + 1e-8)
        with torch.no_grad():
            velocity = flow(x, t, condition) - guidance * grad
            x = (x + velocity / ode_steps).clamp(min=0)
        if (step + 1) % 5 == 0 or step + 1 == ode_steps:
            print(f"guided ODE {step + 1}/{ode_steps}: frame L1 {measurement.item():.6f}", flush=True)
    return x.reshape_as(initial).detach()


def save_srm(path, srm):
    image = to_display(srm).detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    cv2.imwrite(str(path), (image[..., ::-1] * 255 + 0.5).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srm-run", default="srm", help="paper SRM checkpoint under runs/")
    parser.add_argument("--run", default="srm_flow", help="output checkpoint under runs/")
    parser.add_argument("--prior-steps", type=int, default=150)
    parser.add_argument("--ode-steps", type=int, default=30)
    parser.add_argument("--frame-batch", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--guidance", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()
    scene = Scene(device)
    model = load_srm(device, name=args.srm_run)
    refined = refine_srm_with_flow(
        model,
        scene,
        prior_steps=args.prior_steps,
        ode_steps=args.ode_steps,
        frame_batch=args.frame_batch,
        sigma=args.sigma,
        guidance=args.guidance,
        seed=args.seed,
    )

    out = RUNS / args.run
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(RUNS / args.srm_run / "last.pt", map_location=device)
    state["model"]["srm"] = refined.cpu()
    torch.save(state, out / "last.pt")
    source_args = {}
    source_path = RUNS / args.srm_run / "args.json"
    if source_path.exists():
        source_args = json.loads(source_path.read_text())
    source_args.update(flow_refined_from=args.srm_run, flow_refine=vars(args))
    (out / "args.json").write_text(json.dumps(source_args, indent=2))
    for i, panorama in enumerate(refined):
        save_srm(out / f"srm{i}_flow.png", panorama)
    print(f"saved refined checkpoint to {out}")


if __name__ == "__main__":
    main()
