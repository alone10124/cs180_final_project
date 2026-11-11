"""Train the two stages of the paper.

  python train.py srm       # Sec. 3.2: SRMs, material weights, diffuse refinement (Eq. 7)
  python train.py renderer  # Sec. 3.3: neural renderer on physical inputs (Eq. 8)
"""
import argparse
import time

import cv2
import numpy as np
import torch
from torch.amp import GradScaler, autocast

from common import RUNS
from data import Scene, augment, cross_project, to_display, to_linear
from models import (
    Discriminator,
    EncoderDecoder,
    Perceptual,
    SRMModel,
    gan_d_loss,
    gan_g_loss,
    masked_l1,
)

# Eq. 6 and 7. The paper does not state its intensity range; L1-type terms are
# expressed for 0-255 images, which keeps them comparable to the perceptual loss.
PIXEL_SCALE = 255
LAMBDA_1 = 0.01 * PIXEL_SCALE
LAMBDA_P = 1.0
LAMBDA_G = 0.05
LAMBDA_S = 1e-4 * PIXEL_SCALE
LAMBDA_V = 1e-4 * PIXEL_SCALE
LAMBDA_T = 1e-4 * PIXEL_SCALE
LR_NET = 2e-4
LR_SRM = 1e-3
BETAS = (0.5, 0.999)  # pix2pix default; the paper only says ADAM


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


def load_srm(device, materials, name="srm"):
    model = SRMModel(materials).to(device)
    model.load_state_dict(torch.load(RUNS / name / "last.pt", map_location=device)["model"])
    return model.eval().requires_grad_(False)


def physical_inputs(model, b):
    """The 13-channel renderer input C_P = (D, S, FBI, R_P, FCI) of Sec. 3.3."""
    diffuse, specular, weights = model(b, to_linear(b["diffuse"]), b["diffuse"])
    mask = b["mask"]
    c = torch.cat([to_display(diffuse) * mask, to_display(specular) * mask, b["fbi"], b["refl"], b["fci"]], dim=1)
    return c, diffuse, specular, weights


def train_srm(args, scene, device):
    out = RUNS / args.run
    out.mkdir(parents=True, exist_ok=True)
    model = SRMModel(args.materials).to(device)
    disc = Discriminator().to(device)
    perceptual = Perceptual().to(device)
    opt_nets = torch.optim.Adam(
        list(model.material.parameters()) + list(model.refine.parameters()), LR_NET, betas=BETAS
    )
    opt_srm = torch.optim.Adam([model.srm], LR_SRM)
    opt_disc = torch.optim.Adam(disc.parameters(), LR_NET, betas=BETAS)
    scaler = GradScaler("cuda")
    start = 0
    if args.resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        disc.load_state_dict(state["disc"])
        opt_nets.load_state_dict(state["opt_nets"])
        opt_srm.load_state_dict(state["opt_srm"])
        opt_disc.load_state_dict(state["opt_disc"])
        start = state["epoch"] + 1

    rng = np.random.default_rng(args.seed + start)
    t0 = time.time()
    steps = 0
    current = start
    for epoch, iteration, idx in epochs_of(scene, args.batch, args.epochs, start, rng):
        if epoch != current:
            _save_srm(out, model, disc, opt_nets, opt_srm, opt_disc, current)
            current = epoch
        b = augment(scene.batch(idx))
        mask = b["mask"]
        pairs = torch.roll(torch.arange(len(idx), device=device), 1)
        everyone = torch.arange(len(idx), device=device)
        with autocast("cuda"):
            diffuse, specular, weights = model(b, to_linear(b["diffuse"]), b["diffuse"])
            pred_linear = (diffuse + specular).clamp(0, 1)
            real_linear = to_linear(b["rgb"])
            pred_display = to_display(pred_linear) * mask
            real_display = b["rgb"] * mask
            l1 = masked_l1(pred_linear, real_linear, mask)
            lp = perceptual(pred_display, real_display)
            lg = gan_g_loss(disc, pred_display)
            sparsity = masked_l1(specular, torch.zeros_like(specular), mask)
            warped_w, valid = cross_project(b, weights, everyone, pairs)
            consistency_w = masked_l1(weights, warped_w, valid)
            warped_d, valid = cross_project(b, diffuse, everyone, pairs)
            consistency_d = masked_l1(diffuse, warped_d, valid)
            loss = (
                LAMBDA_1 * l1
                + LAMBDA_P * lp
                + LAMBDA_G * lg
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

        opt_disc.zero_grad()
        with autocast("cuda"):
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
    _save_srm(out, model, disc, opt_nets, opt_srm, opt_disc, current)


def _save_srm(out, model, disc, opt_nets, opt_srm, opt_disc, epoch):
    torch.save(
        {
            "model": model.state_dict(),
            "disc": disc.state_dict(),
            "opt_nets": opt_nets.state_dict(),
            "opt_srm": opt_srm.state_dict(),
            "opt_disc": opt_disc.state_dict(),
            "epoch": epoch,
        },
        out / "last.pt",
    )
    for i, srm in enumerate(model.srm):
        save_image(out / f"srm{i}_{epoch:03d}.png", to_display(srm))
    print(f"saved epoch {epoch}", flush=True)


def train_renderer(args, scene, device):
    out = RUNS / args.run
    out.mkdir(parents=True, exist_ok=True)
    srm = load_srm(device, args.materials, args.srm_run)
    generator = EncoderDecoder(13).to(device)
    disc = Discriminator().to(device)
    perceptual = Perceptual().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), LR_NET, betas=BETAS)
    opt_d = torch.optim.Adam(disc.parameters(), LR_NET, betas=BETAS)
    scaler = GradScaler("cuda")
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
    steps = 0
    current = start
    for epoch, iteration, idx in epochs_of(scene, args.batch, args.epochs, start, rng):
        if epoch != current:
            checkpoint(current)
            current = epoch
        b = augment(scene.batch(idx), rescale=True)
        mask = b["mask"]
        with torch.no_grad(), autocast("cuda"):
            inputs = physical_inputs(srm, b)[0]
        with autocast("cuda"):
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
        with autocast("cuda"):
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
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.run = args.run or args.stage
    args.epochs = args.epochs or {"srm": 40, "renderer": 75}[args.stage]

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    scene = Scene(device)
    {"srm": train_srm, "renderer": train_renderer}[args.stage](args, scene, device)


if __name__ == "__main__":
    main()
