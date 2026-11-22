"""Train a restoration model on simulated pairs, resumably.

  python -m sodacan.restore.train --config configs/train_unet.yaml --out runs/restore/unet
  python -m sodacan.restore.train --config configs/train_unet.yaml --out runs/restore/unet --resume
  python -m sodacan.restore.train ... --max-minutes 540          # stop cleanly before a Kaggle session ends

Data: shards from `python -m sodacan.sim.dataset` (data.shards), or pairs simulated on
the fly into a replay pool (data.on_the_fly: true). Training uses random crops of the
observed band; validation runs on whole bands of held-out simulated rooms (a fixed seed
range never used for training) and reports PSNR/SSIM inside the coverage mask for the
classical input and the model. Checkpoints: last.pt every ckpt_minutes and at exit,
best.pt on the best validation PSNR. --resume continues from last.pt.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from sodacan.config import load_yaml
from sodacan.metrics import psnr, ssim
from sodacan.restore.losses import restoration_loss
from sodacan.restore.models import build_model, count_parameters
from sodacan.sim.dataset import PairStream

VAL_SEED = 987_654


def pick_device():
    for _ in range(3):  # CUDA detection is occasionally flaky on laptops
        if torch.cuda.is_available():
            return torch.device("cuda")
        time.sleep(1)
    return torch.device("cpu")


def crop_pair(p, ch, cw, rng):
    """Random crop (ch rows, cw columns, wrapping in azimuth) of a band pair, with flips."""
    x, w, v, t = p["input"], p["weight"], p["valid"], p["target"]
    h, W = v.shape
    if h < ch:
        pad = ch - h
        x = torch.cat([x, x.new_zeros(3, pad, W)], 1)
        t = torch.cat([t, t[:, -1:].expand(3, pad, W)], 1)
        w = torch.cat([w, w.new_zeros(pad, W)], 0)
        v = torch.cat([v, v.new_zeros(pad, W)], 0)
        h = ch
    r0 = rng.randint(0, h - ch)
    shift = rng.randint(0, W - 1)
    sl = lambda a: torch.roll(a[..., r0 : r0 + ch, :], shift, -1)[..., :cw]  # noqa: E731
    out = {"input": sl(x), "weight": sl(w), "valid": sl(v), "target": sl(t)}
    if rng.random() < 0.5:
        out = {k: a.flip(-1) for k, a in out.items()}
    return out


def collate(pairs, device):
    out = {}
    for k in ("input", "weight", "valid", "target"):
        t = torch.stack([p[k] for p in pairs]).to(device)
        out[k] = t if k == "valid" else t.float()
    return out


class ShardSource:
    """Pairs from shard files; keeps a pool of up to `pool_shards` shards in memory and rotates."""

    def __init__(self, folder, rng, pool_shards=2, swap_every=200):
        self.files = sorted(Path(folder).glob("pairs_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no pairs_*.pt in {folder}; run python -m sodacan.sim.dataset")
        self.rng, self.pool_shards, self.swap_every, self.calls = rng, pool_shards, swap_every, 0
        self.pool = []
        for _ in range(min(pool_shards, len(self.files))):
            self._load()

    def _load(self):
        f = self.rng.choice(self.files)
        self.pool.append(torch.load(f, weights_only=False))
        if len(self.pool) > self.pool_shards:
            self.pool.pop(0)

    def sample(self, n):
        self.calls += 1
        if self.calls % self.swap_every == 0 and len(self.files) > self.pool_shards:
            self._load()
        flat = [p for shard in self.pool for p in shard]
        return [self.rng.choice(flat) for _ in range(n)]


class StreamSource:
    """Pairs simulated on the fly into a replay pool (fresh_per_step new pairs each step)."""

    def __init__(self, stream, rng, pool=128, fresh_per_step=1):
        self.stream, self.rng, self.fresh = stream, rng, fresh_per_step
        self.pool = stream.take(min(pool, max(8, pool)))
        self.size = pool

    def sample(self, n):
        for _ in range(self.fresh):
            new = next(self.stream)
            if len(self.pool) < self.size:
                self.pool.append(new)
            else:
                self.pool[self.rng.randrange(self.size)] = new
        return [self.rng.choice(self.pool) for _ in range(n)]


def validation_pairs(out, cfg, device, n):
    path = Path(out) / "val.pt"
    if path.exists():
        pairs = torch.load(path, weights_only=False)
        if len(pairs) >= n:
            return pairs[:n]
    print(f"simulating {n} held-out validation pairs ...", flush=True)
    stream = PairStream(seed=VAL_SEED + cfg.get("val", {}).get("seed", 0), device=device, sim=cfg.get("sim", "sim"),
                        pano=cfg.get("pano"))
    pairs = stream.take(n)
    torch.save(pairs, path)
    return pairs


@torch.no_grad()
def evaluate(model, pairs, device):
    model.eval()
    rows = []
    for p in pairs:
        b = collate([p], device)
        pred = model(b["input"], b["weight"], b["valid"])[0].float()
        v, tgt, inp = b["valid"][0], b["target"][0], b["input"][0]
        if not v.any():
            continue
        rows.append({"psnr_in": psnr(inp, tgt, v), "psnr_out": psnr(pred, tgt, v), "ssim_in": ssim(inp, tgt, v), "ssim_out": ssim(pred, tgt, v)})
    model.train()
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def save(path, **state):
    tmp = Path(path).with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop and checkpoint after this long (0: no limit)")
    ap.add_argument("--steps", type=int, help="override the config's step count")
    ap.add_argument("--shards", help="override data.shards")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.steps:
        cfg["steps"] = args.steps
    if args.shards:
        cfg.setdefault("data", {})["shards"] = args.shards
        cfg["data"]["on_the_fly"] = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    torch.manual_seed(cfg.get("seed", 0))
    rng = random.Random(cfg.get("seed", 0))

    mcfg = dict(cfg["model"])
    model = build_model(mcfg.pop("name"), **mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    steps, warmup = int(cfg["steps"]), int(cfg.get("warmup", 500))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    use_amp = device.type == "cuda" and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, best = 0, -math.inf
    last = out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        step, best = ck["step"], ck["best"]
        rng.setstate(ck["py_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        print(f"resumed from step {step}", flush=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{cfg['model']['name']}: {count_parameters(model) / 1e6:.2f} M parameters on {device}", flush=True)

    dcfg = cfg.get("data", {})
    no_shards = not dcfg.get("shards") or not list(Path(dcfg["shards"]).glob("pairs_*.pt"))
    if no_shards and not dcfg.get("on_the_fly", False):
        print(f"no shards in {dcfg.get('shards')}; simulating pairs on the fly", flush=True)
    if dcfg.get("on_the_fly", False) or no_shards:
        stream = PairStream(seed=cfg.get("seed", 0) * 7919 + step, device=device, sim=cfg.get("sim", "sim"), pano=cfg.get("pano"))
        source = StreamSource(stream, rng, dcfg.get("pool", 128), dcfg.get("fresh_per_step", 1))
    else:
        source = ShardSource(dcfg["shards"], rng)
    vcfg = cfg.get("val", {})
    val = validation_pairs(out, cfg, device, vcfg.get("pairs", 32))
    ch, cw = cfg.get("crop", [64, 256])
    weights = cfg.get("loss", {})

    def checkpoint(path):
        save(path, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
             step=step, best=best, py_rng=rng.getstate(), torch_rng=torch.get_rng_state(), config=cfg)

    t_start = t_ckpt = time.time()
    log = open(out / "history.jsonl", "a")
    model.train()
    while step < steps:
        batch = collate([crop_pair(p, ch, cw, rng) for p in source.sample(cfg["batch"])], device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            loss, terms = restoration_loss(model, batch, weights)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % 50 == 0:
            rec = {"step": step, "lr": sched.get_last_lr()[0], **terms, "minutes": (time.time() - t_start) / 60}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
        if step % vcfg.get("every", 1000) == 0 or step == steps:
            m = evaluate(model, val, device)
            log.write(json.dumps({"step": step, "val": m}) + "\n")
            log.flush()
            print(f"val step {step}: classical {m['psnr_in']:.2f} dB -> model {m['psnr_out']:.2f} dB, "
                  f"SSIM {m['ssim_in']:.3f} -> {m['ssim_out']:.3f}", flush=True)
            if m["psnr_out"] > best:
                best = m["psnr_out"]
                save(out / "best.pt", model=model.state_dict(), config=cfg, step=step, val=m)
        timed_out = args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes
        if time.time() - t_ckpt > 60 * cfg.get("ckpt_minutes", 30) or timed_out:
            checkpoint(last)
            t_ckpt = time.time()
        if timed_out:
            print(f"stopping at step {step} after {args.max_minutes} minutes; continue with --resume", flush=True)
            break
    checkpoint(last)
    if not (out / "best.pt").exists():
        save(out / "best.pt", model=model.state_dict(), config=cfg, step=step)
    log.close()


if __name__ == "__main__":
    main()
