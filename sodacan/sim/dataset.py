"""Simulated can videos and restoration training pairs.

A pair is (classical panorama + coverage, sharp panorama): a random room is rendered
through a random can (label, coat, roughness, dents) along a random hand-held path with
noise, motion blur and 8-bit quantisation; the classical pipeline then reconstructs it
from slightly wrong poses, as it would a real video. The target is the true room at the
panorama resolution, scaled by the same reflection gain.

  python -m sodacan.sim.dataset --count 10000 --out data/pairs     # write shards
  python -m sodacan.sim.dataset --benchmark 50                     # pairs per second
"""
import argparse
import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sodacan.config import load_can, load_pipeline, load_yaml
from sodacan.geometry.camera import Intrinsics, perturb_poses
from sodacan.geometry.equirect import resize_pano
from sodacan.pipeline import reconstruct
from sodacan.sim.fast_render import RenderParams, render
from sodacan.sim.rooms import list_hdris, random_environment
from sodacan.sim.trajectories import random_path


@dataclass
class SimScene:
    frames: torch.Tensor   # (B, 3, H, W) linear
    c2w: torch.Tensor      # true poses
    c2w_est: torch.Tensor  # poses with simulated estimation error
    K: Intrinsics
    env: torch.Tensor      # (3, He, We)
    can: object            # Can actually rendered (material may differ from the config)
    params: RenderParams


def random_label(rng, rows, cols, device="cpu"):
    """A procedural printed label (3, rows, cols): background, bands, logo blobs, text blocks."""
    s = torch.linspace(0, 1, cols, device=device)[None].expand(rows, cols)
    h = torch.linspace(0, 1, rows, device=device)[:, None].expand(rows, cols)
    col = lambda: torch.tensor(rng.uniform(0.03, 0.85, 3), dtype=torch.float32, device=device)[:, None, None]  # noqa: E731
    label = col().expand(3, rows, cols).clone()
    for _ in range(rng.integers(0, 4)):
        h0, dh = rng.uniform(0, 1), rng.uniform(0.02, 0.25)
        label = torch.where(((h > h0) & (h < h0 + dh))[None], col(), label)
    for _ in range(rng.integers(1, 4)):
        cs, ch, rs, rh = rng.uniform(0, 1), rng.uniform(0.2, 0.8), rng.uniform(0.05, 0.25), rng.uniform(0.05, 0.3)
        ds = torch.remainder(s - cs + 0.5, 1) - 0.5
        label = torch.where(((ds / rs) ** 2 + ((h - ch) / rh) ** 2 < 1)[None], col(), label)
    for _ in range(rng.integers(0, 6)):
        cs, ch = rng.uniform(0, 1), rng.uniform(0.1, 0.9)
        ds = torch.remainder(s - cs + 0.5, 1) - 0.5
        box = (ds.abs() < rng.uniform(0.02, 0.1)) & ((h - ch).abs() < rng.uniform(0.01, 0.05))
        fine = torch.remainder(torch.floor(s * cols / rng.integers(2, 6)), 2) > 0
        label = torch.where((box & fine)[None], col(), label)
    return label


def random_dent(rng, rows, cols, deg, device="cpu"):
    """Smooth random normal tilts (2, rows, cols), std about `deg` degrees."""
    lr = torch.tensor(rng.normal(0, 1, (1, 2, rng.integers(3, 10), rng.integers(8, 40))), dtype=torch.float32)
    lr = torch.cat([lr, lr[..., :1]], -1)  # wrap
    up = F.interpolate(lr, size=(rows, cols + cols // max(lr.shape[-1] - 1, 1)), mode="bicubic", align_corners=True)[0, :, :, :cols]
    up = up / up.std().clamp_min(1e-6) * math.radians(deg)
    return up.to(device)


def u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def sample_scene(rng, sim, can, device="cpu", mirror=False, frames=None, env=None, hdris=(), env_hw=(512, 1024), cache=None):
    """Draw and render one simulated can video."""
    W, H = sim["image"]
    f = u(rng, sim["focal"])
    K = Intrinsics(W, H, f, f, (W - 1) / 2 + rng.uniform(-5, 5), (H - 1) / 2 + rng.uniform(-5, 5))
    n = frames or int(sim["frames"])
    c2w = random_path(rng, n, sim, can, device)
    if env is None:
        env = random_environment(*env_hw, rng, hdris, sim.get("hdri_prob", 0.5), device, cache)
    can = copy.deepcopy(can)
    if mirror:
        can.material = {"model": "mirror"}
        params = RenderParams(ss=2)
    else:
        labelled = rng.random() < sim["label_prob"]
        if labelled:
            can.material = {"model": "coated", "ior": 1.5}
        else:
            can.material = {"model": "metal", "f0": u(rng, sim["metal_f0"])}
        params = RenderParams(
            roughness_deg=u(rng, sim["roughness_deg"]),
            strength=u(rng, sim["strength"]),
            label=random_label(rng, *can.atlas, device) if labelled else None,
            ambient=u(rng, sim["ambient"]),
            dent=random_dent(rng, *can.atlas, u(rng, sim["dent_deg"]), device) if rng.random() < sim["dent_prob"] else None,
            exposure=u(rng, sim["exposure"]),
            noise_read=u(rng, sim["noise_read"]),
            noise_shot=u(rng, sim["noise_shot"]),
            quantize=True,
            motion_blur=u(rng, sim["motion_blur"]),
            motion_steps=3,
        )
    gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
    images = render(env, c2w, K, can, params, generator=gen)
    if mirror:
        c2w_est = c2w
    else:
        c2w_est = perturb_poses(c2w, u(rng, sim["pose_error_deg"]), u(rng, sim["pose_error_mm"]) / 1000, gen)
    return SimScene(images, c2w, c2w_est, K, env, can, params)


def target_of(scene, H, W):
    """The panorama the classical pipeline estimates: room radiance times reflection gain."""
    return resize_pano(scene.env, H, W) * scene.params.strength * scene.params.exposure


def band(valid, min_rows=64, margin=4):
    """Row range [r0, r1) holding all observed texels plus a margin, at least min_rows tall."""
    H = valid.shape[0]
    rows = valid.any(-1).nonzero().squeeze(1)
    if rows.numel() == 0:
        c = H // 2
        r0, r1 = c - min_rows // 2, c + min_rows // 2
    else:
        r0, r1 = int(rows.min()) - margin, int(rows.max()) + 1 + margin
    if r1 - r0 < min_rows:
        extra = min_rows - (r1 - r0)
        r0 -= extra // 2
        r1 += extra - extra // 2
    r0 = max(r0, 0)
    r1 = min(max(r1, r0 + min_rows), H)
    r0 = max(min(r0, r1 - min_rows), 0)
    return r0, r1


def make_pair(rng, sim, can, cfg, device="cpu", hdris=(), cache=None, crop_band=True):
    """One training pair, cropped to the band of rows the can observed (most of a
    panorama is unobserved; storing it would waste space)."""
    H, W = cfg["pano"]
    scene = sample_scene(rng, sim, can, device, hdris=hdris, env_hw=(2 * H, 2 * W), cache=cache)
    # the pipeline assumes the material model of the can being filmed
    pipe_can = copy.deepcopy(can)
    pipe_can.material = scene.can.material
    res = reconstruct(scene.frames, scene.c2w_est, scene.K, pipe_can, cfg)
    target = target_of(scene, H, W)
    r0, r1 = band(res.valid) if crop_band else (0, H)
    return {
        "input": res.pano[:, r0:r1].half().cpu(),
        "weight": res.weight[r0:r1].half().cpu(),
        "valid": res.valid[r0:r1].cpu(),
        "target": target[:, r0:r1].half().cpu(),
        "row0": r0,
        "H": H,
    }


class PairStream:
    """Endless on-the-fly pairs for training (GPU when available)."""

    def __init__(self, seed=0, sim="sim", can="can", pipeline="pipeline", device="cpu", pano=None):
        self.rng = np.random.default_rng(seed)
        self.sim = load_yaml(sim) if isinstance(sim, str) else sim
        self.can = load_can(can) if isinstance(can, str) else can
        self.cfg = load_pipeline(pipeline) if isinstance(pipeline, str) else pipeline
        if pano:
            self.cfg["pano"] = list(pano)
        self.device = device
        self.hdris = list_hdris(self.sim.get("hdri_dir", "data/hdri"))
        self.cache = {}

    def __iter__(self):
        return self

    def __next__(self):
        return make_pair(self.rng, self.sim, self.can, self.cfg, self.device, self.hdris, self.cache)

    def take(self, n):
        return [next(self) for _ in range(n)]


def write_shards(out, count, shard_size=500, seed=0, device="cpu", **kw):
    """Shards of pairs; each shard has its own seed, so a cut-off run resumes where it stopped."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stream = PairStream(seed=seed, device=device, **kw)
    t0, made = time.time(), 0
    for s in range(math.ceil(count / shard_size)):
        path = out / f"pairs_{seed:04d}_{s:04d}.pt"
        if path.exists():
            continue
        stream.rng = np.random.default_rng([seed, s])
        n = min(shard_size, count - s * shard_size)
        tmp = path.with_suffix(".tmp")
        torch.save(stream.take(n), tmp)
        tmp.replace(path)
        made += n
        print(f"shard {s}: {s * shard_size + n}/{count} pairs, {made / (time.time() - t0):.2f} pairs/s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/pairs")
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", type=int, default=0, help="time this many pairs and exit")
    ap.add_argument("--sim", default="sim")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.benchmark:
        stream = PairStream(seed=args.seed, device=device, sim=args.sim)
        next(stream)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.benchmark):
            next(stream)
        if device == "cuda":
            torch.cuda.synchronize()
        rate = args.benchmark / (time.time() - t0)
        print(f"{rate:.2f} pairs/s on {device}; 10,000 pairs take {10000 / rate / 3600:.2f} h")
        return
    write_shards(args.out, args.count, args.shard_size, args.seed, device, sim=args.sim)


if __name__ == "__main__":
    main()
