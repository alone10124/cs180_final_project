"""Train the two stages of the paper.

  python train.py srm       # Sec. 3.2: SRMs, material weights, diffuse refinement (Eq. 7)
  python train.py renderer  # Sec. 3.3: neural renderer on physical inputs (Eq. 8)

Experiment controls for the srm stage:
  --cans a|b|both   only those cans may reflect the environment (default all: every surface)
  --views N         train on N of the training views (see common.select_views)
  --max-steps S     fixed step budget; without --epochs the epoch count grows to reach it
"""
import argparse
import contextlib
import json
import math
import time

import cv2
import numpy as np
import torch
from torch.amp import GradScaler, autocast

from common import CACHE, RUNS, load_poses, select_views
from data import Scene, augment, cross_project, pick_device, select, to_display, to_linear
from models import (
    Discriminator,
    EncoderDecoder,
    Perceptual,
    SRMModel,
    gan_d_loss,
    gan_g_loss,
    masked_l1,
)

# Main paper Eqs. 6 and 7: use the published coefficients directly.
LAMBDA_1 = 0.01
LAMBDA_P = 1.0
LAMBDA_G = 0.05
LAMBDA_S = 1e-4
LAMBDA_V = 1e-4
LAMBDA_T = 1e-4
LR_NET = 2e-4
LR_SRM = 1e-3


def amp(device):
    return autocast("cuda") if device.type == "cuda" else contextlib.nullcontext()


def save_image(path, tensor):
    """Save a 3xHxW display-encoded image."""
    array = (tensor.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    cv2.imwrite(str(path), array[..., ::-1])


def epochs_of(scene, batch_size, epochs, start, rng):
    """Yield (epoch, iteration, frame indices), one pass over training frames per epoch."""
    n = len(scene.train) - len(scene.train) % batch_size
    for epoch in range(start, epochs):
        order = rng.permutation(scene.train)[:n]
        for i in range(0, n, batch_size):
            yield epoch, i // batch_size, order[i : i + batch_size].tolist()


def load_srm(device, materials=None, name="srm"):
    """Load a trained SRM run with the settings it was trained with (runs/<name>/args.json)."""
    saved = {}
    if (RUNS / name / "args.json").exists():
        saved = json.loads((RUNS / name / "args.json").read_text())
    model = SRMModel(saved.get("materials", materials or 2), refine=not saved.get("no_refine", False)).to(device)
    model.load_state_dict(torch.load(RUNS / name / "last.pt", map_location=device)["model"])
    model.cans = saved.get("cans", "all")
    return model.eval().requires_grad_(False)


def gate(b, cans):
    """Restrict specular reflection to the selected cans."""
    return b if cans == "all" else dict(b, spec_mask=select(b, cans))


def physical_inputs(model, b):
    """The 13-channel renderer input C_P = (D, S, FBI, R_P, FCI) of Sec. 3.3."""
    b = gate(b, model.cans)
    diffuse, specular, weights = model(b, to_linear(b["diffuse"]), b["diffuse"])
    mask = b["mask"]
    c = torch.cat([to_display(diffuse) * mask, to_display(specular) * mask, b["fbi"], b["refl"], b["fci"]], dim=1)
    return c, diffuse, specular, weights


def train_srm(args, scene, device):
    out = RUNS / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    model = SRMModel(args.materials, refine=not args.no_refine).to(device)
    disc = Discriminator().to(device) if args.lambda_g else None
    perceptual = Perceptual().to(device) if args.lambda_p else None
    net_params = list(model.material.parameters())
    if model.refine is not None:
        net_params += list(model.refine.parameters())
    # The supplement specifies PyTorch Adam, but no non-default betas.
    opt_nets = torch.optim.Adam(net_params, LR_NET)
    opt_srm = torch.optim.Adam([model.srm], args.lr_srm)
    opt_disc = torch.optim.Adam(disc.parameters(), LR_NET) if disc else None
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    start = 0
    if args.resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        opt_nets.load_state_dict(state["opt_nets"])
        opt_srm.load_state_dict(state["opt_srm"])
        if disc:
            disc.load_state_dict(state["disc"])
            opt_disc.load_state_dict(state["opt_disc"])
        start = state["epoch"] + 1

    def checkpoint(epoch):
        state = {
            "model": model.state_dict(),
            "opt_nets": opt_nets.state_dict(),
            "opt_srm": opt_srm.state_dict(),
            "epoch": epoch,
        }
        if disc:
            state.update(disc=disc.state_dict(), opt_disc=opt_disc.state_dict())
        torch.save(state, out / "last.pt")
        for i, srm in enumerate(model.srm):
            save_image(out / f"srm{i}_{epoch:03d}.png", to_display(srm))
        print(f"saved epoch {epoch}", flush=True)

    rng = np.random.default_rng(args.seed + start)
    t0 = time.time()
    steps = saved = 0
    current = start
    zero = torch.zeros((), device=device)
    for epoch, iteration, idx in epochs_of(scene, args.batch, args.epochs, start, rng):
        if epoch != current:
            if steps - saved >= args.save_every:
                checkpoint(current)
                saved = steps
            current = epoch
        b = gate(augment(scene.batch(idx)), args.cans)
        mask = b["mask"]
        pairs = torch.roll(torch.arange(len(idx), device=device), 1)
        everyone = torch.arange(len(idx), device=device)
        with amp(device):
            diffuse, specular, weights = model(b, to_linear(b["diffuse"]), b["diffuse"])
            pred_linear = (diffuse + specular).clamp(0, 1)
            real_linear = to_linear(b["rgb"])
            pred_display = to_display(pred_linear) * mask
            real_display = b["rgb"] * mask
            l1 = masked_l1(pred_linear, real_linear, mask)
            lp = perceptual(pred_display, real_display) if perceptual else zero
            lg = gan_g_loss(disc, pred_display) if disc else zero
            sparsity = masked_l1(specular, torch.zeros_like(specular), mask)
            warped_w, valid = cross_project(b, weights, everyone, pairs)
            consistency_w = masked_l1(weights, warped_w, valid)
            warped_d, valid = cross_project(b, diffuse, everyone, pairs)
            consistency_d = masked_l1(diffuse, warped_d, valid)
            loss = (
                LAMBDA_1 * l1
                + args.lambda_p * lp
                + args.lambda_g * lg
                + LAMBDA_S * sparsity
                + LAMBDA_V * consistency_w
                + LAMBDA_T * consistency_d
            )
        opt_nets.zero_grad()
        opt_srm.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt_nets)
        scaler.step(opt_srm)
        model.clamp_srm()

        d_loss = zero
        if disc:
            opt_disc.zero_grad()
            with amp(device):
                d_loss = gan_d_loss(disc, real_display, pred_display)
            scaler.scale(d_loss).backward()
            scaler.step(opt_disc)
        scaler.update()

        steps += 1
        if steps % args.log == 0:
            rate = steps / (time.time() - t0)
            print(
                f"epoch {epoch} iter {iteration} l1 {l1.item():.4f} perc {lp.item():.3f} "
                f"gan {lg.item():.3f} d {d_loss.item():.3f} spec {sparsity.item():.4f} "
                f"V {consistency_w.item():.4f} T {consistency_d.item():.4f} {rate:.2f} it/s",
                flush=True,
            )
        if steps % args.preview == 0:
            save_image(out / f"preview_{epoch:03d}.png", torch.cat([real_display[0], pred_display[0]], dim=2))
        if args.max_steps and steps >= args.max_steps:
            break
    checkpoint(current)


def train_renderer(args, scene, device):
    out = RUNS / args.run
    out.mkdir(parents=True, exist_ok=True)
    srm = load_srm(device, args.materials, args.srm_run)
    generator = EncoderDecoder(13).to(device)
    disc = Discriminator().to(device)
    perceptual = Perceptual().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), LR_NET)
    opt_d = torch.optim.Adam(disc.parameters(), LR_NET)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    start = 0
    if args.resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device)
        generator.load_state_dict(state["generator"])
        disc.load_state_dict(state["disc"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        start = state["epoch"] + 1

    def checkpoint(epoch):
        torch.save(
            {
                "generator": generator.state_dict(),
                "disc": disc.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "epoch": epoch,
            },
            out / "last.pt",
        )
        print(f"saved epoch {epoch}", flush=True)

    rng = np.random.default_rng(args.seed + start)
    t0 = time.time()
    steps = saved = 0
    current = start
    for epoch, iteration, idx in epochs_of(scene, args.batch, args.epochs, start, rng):
        if epoch != current:
            if steps - saved >= args.save_every:
                checkpoint(current)
                saved = steps
            current = epoch
        b = augment(scene.batch(idx), rescale=True)
        mask = b["mask"]
        with torch.no_grad(), amp(device):
            inputs = physical_inputs(srm, b)[0]
        with amp(device):
            pred = generator(inputs).clamp(0, 1) * mask
            real = b["rgb"] * mask
            l1 = masked_l1(pred, real, mask)
            lp = perceptual(pred, real)
            lg = gan_g_loss(disc, pred)
            loss = LAMBDA_1 * l1 + LAMBDA_P * lp + LAMBDA_G * lg
        opt_g.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt_g)

        opt_d.zero_grad()
        with amp(device):
            d_loss = gan_d_loss(disc, real, pred)
        scaler.scale(d_loss).backward()
        scaler.step(opt_d)
        scaler.update()

        steps += 1
        if steps % args.log == 0:
            rate = steps / (time.time() - t0)
            print(
                f"epoch {epoch} iter {iteration} l1 {l1.item():.4f} perc {lp.item():.3f} "
                f"gan {lg.item():.3f} d {d_loss.item():.3f} {rate:.2f} it/s",
                flush=True,
            )
        if steps % args.preview == 0:
            save_image(out / f"preview_{epoch:03d}.png", torch.cat([real[0], pred[0]], dim=2))
        if args.max_steps and steps >= args.max_steps:
            break
    checkpoint(current)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["srm", "renderer"])
    ap.add_argument("--run", default=None, help="run directory name under runs/")
    ap.add_argument("--srm-run", default="srm", help="SRM run used by the renderer stage")
    ap.add_argument("--epochs", type=int, default=None, help="paper: 40 (srm), 75 (renderer)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--materials", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=int, default=200)
    ap.add_argument("--preview", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=500, help="minimum steps between checkpoints")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cans", choices=["all", "a", "b", "both"], default="all")
    ap.add_argument("--views", type=int, default=0, help="number of training views; 0 uses all")
    ap.add_argument("--view-select", choices=["fps", "random", "arc"], default="fps")
    ap.add_argument("--diffuse", choices=["diffuse", "diffuse_est"], default="diffuse")
    ap.add_argument("--no-refine", action="store_true", help="ablation: no diffuse refinement network")
    ap.add_argument("--lr-srm", type=float, default=LR_SRM, help="panorama learning rate; Adam moves a texel at most this far per step")
    ap.add_argument("--lambda-p", type=float, default=LAMBDA_P, help="perceptual weight; 0 disables")
    ap.add_argument("--lambda-g", type=float, default=LAMBDA_G, help="GAN weight; 0 disables")
    args = ap.parse_args()
    args.run = args.run or args.stage

    torch.manual_seed(args.seed)
    device = pick_device()
    scene = Scene(device, diffuse=args.diffuse)
    if args.cans == "all":
        scene.cans = None
    if args.views:
        centre = np.load(CACHE / "split.npz")["centre"]
        scene.train = select_views(load_poses(), centre, scene.train, args.views, args.view_select, args.seed)
    args.batch = min(args.batch, len(scene.train))
    if args.epochs is None:
        args.epochs = {"srm": 40, "renderer": 75}[args.stage]
        if args.max_steps:
            args.epochs = math.ceil(args.max_steps / (len(scene.train) // args.batch))
    {"srm": train_srm, "renderer": train_renderer}[args.stage](args, scene, device)


if __name__ == "__main__":
    main()
