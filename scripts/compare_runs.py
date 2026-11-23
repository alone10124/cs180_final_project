"""One comparison chart over several run_noboard.py / run_photos.py output folders:
PSNR and SSIM of each panorama against its ground-truth photo of the room, the coverage
of the sphere, and each run's 05_reference.png (panorama vs ground truth) stacked below.

python scripts/compare_runs.py --runs out/one_can_0 out/three_cans_0 ... --out out/comparison.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    names, psnr, ssim, cov, cans = [], [], [], [], []
    for r in args.runs:
        p = Path(r) / "metrics.json"
        if not p.exists():
            print(f"{r}: no metrics.json, skipped")
            continue
        m = json.loads(p.read_text())
        ref = (m.get("reference") or {}).get("measured") or {}
        names.append(Path(r).name)
        psnr.append(ref.get("psnr", np.nan))
        ssim.append(ref.get("ssim", np.nan))
        cov.append((m.get("coverage") or {}).get("sphere_fraction", np.nan))
        cans.append(len(m.get("cans", [])))
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, v, title, fmt in ((ax[0], psnr, "PSNR vs ground truth (dB)", "{:.1f}"),
                             (ax[1], ssim, "SSIM vs ground truth", "{:.3f}"),
                             (ax[2], cov, "sphere covered", "{:.2f}")):
        bars = a.bar(x, v, color="#4a78b5")
        for b, val in zip(bars, v):
            if np.isfinite(val):
                a.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(val), ha="center", va="bottom", fontsize=8)
        a.set_xticks(x, [f"{n}\n({c} can{'s' if c > 1 else ''})" for n, c in zip(names, cans)], fontsize=8)
        a.set_title(title)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Board-free runs: COLMAP SfM poses, no restoration model")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    # the per-run panorama-vs-ground-truth images, stacked
    imgs = [plt.imread(str(Path(r) / "05_reference.png")) for r in args.runs if (Path(r) / "05_reference.png").exists()]
    if imgs:
        w = max(i.shape[1] for i in imgs)
        rows = []
        for r, i in zip([r for r in args.runs if (Path(r) / "05_reference.png").exists()], imgs):
            pad = np.ones((i.shape[0], w, i.shape[2]), i.dtype)
            pad[:, : i.shape[1]] = i
            rows.append(pad)
        plt.imsave(str(out.with_name(out.stem + "_references.png")), np.concatenate(rows))
    (out.with_suffix(".json")).write_text(json.dumps(
        [{"run": n, "psnr": p, "ssim": s, "coverage": c, "cans": k} for n, p, s, c, k in zip(names, psnr, ssim, cov, cans)], indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
