"""Phase 3 acceptance on simulation: classical vs each restoration model on held-out rooms.

  python scripts/evaluate_restore.py --models runs/restore/unet/best.pt runs/restore/rtlvit/best.pt --pairs 64

Held-out rooms come from a seed range used by neither training nor validation (and
held-out HDRIs if data/hdri_test/ exists). Metrics are inside the coverage mask only.
Writes figures/restore_table.{json,md}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sodacan.metrics import all_metrics  # noqa: E402
from sodacan.restore.infer import load_model  # noqa: E402
from sodacan.sim.dataset import PairStream  # noqa: E402

TEST_SEED = 31_415_926


def evaluate(model_paths, n=64, device="cpu", sim="sim", hdri_dir=None):
    stream = PairStream(seed=TEST_SEED, device=device, sim=sim)
    if hdri_dir:
        from sodacan.sim.rooms import list_hdris

        stream.hdris = list_hdris(hdri_dir)
    models = {Path(p).parent.name: load_model(p, device) for p in model_paths}
    rows = {name: [] for name in ["classical", *models]}
    for _ in range(n):
        p = next(stream)
        x, w, v, t = (p[k].to(device) for k in ("input", "weight", "valid", "target"))
        x, w, t = x.float(), w.float(), t.float()
        if v.sum() < 50:
            continue
        rows["classical"].append(all_metrics(x, t, v))
        for name, m in models.items():
            with torch.no_grad():
                y = m(x[None], w[None], v[None])[0].float()
            rows[name].append(all_metrics(y, t, v))
    # pairs are cropped to their observed band, so a coverage fraction here would be of the band, not the sphere
    keys = ("psnr", "ssim", "lpips")
    return {name: {k: float(np.mean([r[k] for r in rs])) for k in keys if k in rs[0]} | {"pairs": len(rs)}
            for name, rs in rows.items() if rs}


def to_markdown(table):
    keys = ["psnr", "ssim", "lpips"]
    lines = ["| method | PSNR dB (higher better) | SSIM (higher better) | LPIPS (lower better) |", "|---|---|---|---|"]
    for name, m in table.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[k]:.3f}" if k in m else "n/a" for k in keys) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--hdri-dir")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = evaluate(args.models, args.pairs, device, hdri_dir=args.hdri_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "restore_table.json").write_text(json.dumps(table, indent=2))
    (out / "restore_table.md").write_text(to_markdown(table) + "\n", encoding="utf-8")
    print(to_markdown(table))


if __name__ == "__main__":
    main()
