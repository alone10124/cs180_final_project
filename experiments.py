"""Sweeps over cans and view counts, the label ablation, and their figures.

  SCENE=two python experiments.py sweep --views 4 8 16 32 64 0
  SCENE=two python experiments.py sweep --method srm --views 16 64 --steps 4000
  SCENE=label python experiments.py ablation
  python experiments.py merge --scenes one two     # one-can scene against two-can scene

Results go to results/sweep_<method>/ and results/ablation/ of the scene, and for
merge to results/merge/ at the repo root. Every figure has a CSV with the same numbers.
"""
import argparse
import csv
import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, ROOT

# setting -> (label, colour, marker); colours are fixed per setting, never by rank
STYLE = {
    "a": ("can A", "#2a78d6", "o"),
    "b": ("can B", "#eb6834", "s"),
    "both": ("both cans", "#1baf7a", "^"),
    "both_half": ("both cans, half the pixels", "#eda100", "D"),
}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"


def line_chart(path, rows, key, ylabel, title, series="setting", styles=None):
    """One measure against view count, one line per series, labelled at the line end."""
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    names = [s for s in dict.fromkeys(r[series] for r in rows)]
    ends = []
    for i, name in enumerate(names):
        points = sorted((r["views"], r[key]) for r in rows if r[series] == name and r.get(key) is not None)
        if not points:
            continue
        label, colour, marker = (styles or STYLE)[name]
        x, y = zip(*points)
        ax.plot(x, y, color=colour, marker=marker, markersize=6, linewidth=2, label=label, markeredgecolor="white", markeredgewidth=1)
        ends.append((y[-1], x[-1], label))
    if len(ends) <= 4:
        gap = 0.055 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        placed = -np.inf
        for y, x, label in sorted(ends):
            placed = max(y, placed + gap)
            ax.annotate(label, (x, y), xytext=(x * 1.06, placed), va="center", fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ticks = sorted({r["views"] for r in rows})
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.margins(x=0.25)
    ax.set_xlabel("training views", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_table(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flat(metrics):
    return {k: v for k, v in metrics.items() if not isinstance(v, list)}


def sweep(args):
    from data import Scene, pick_device
    from envmap import compare, load_reference, save_png, side_by_side
    from splat import reconstruct

    scene = Scene(pick_device(), diffuse=args.buffer)
    reference = load_reference(args.reference)
    out = RESULTS / f"sweep_{args.method}"
    out.mkdir(parents=True, exist_ok=True)
    settings = ["a"] if len(scene.cans) < 2 else ["a", "b", "both", "both_half"]
    if args.method == "srm":
        settings = [s for s in settings if s != "both_half"]
    rows, strips = [], []
    for views in args.views:
        results = {}
        for setting in settings:
            cans, keep = ("both", 0.5) if setting == "both_half" else (setting, 1.0)
            if args.method == "splat":
                pano, valid, frames = reconstruct(scene, cans, views, args.view_select, keep=keep, seed=args.seed)
            else:
                from eval_env import evaluate_run

                run = f"sweep_{setting}_{views}"
                command = [sys.executable, str(ROOT / "train.py"), "srm", "--run", run, "--cans", cans]
                command += ["--views", str(views), "--view-select", args.view_select, "--seed", str(args.seed)]
                command += ["--max-steps", str(args.steps), "--diffuse", args.buffer, *args.train_args.split()]
                subprocess.run(command, check=True)
                _, pano, valid, _, _ = evaluate_run(scene, run, reference)
                frames = range(views or len(scene.train))
            results[setting] = (pano, valid, len(frames))
        shared = np.logical_and.reduce([valid for _, valid, _ in results.values()])
        for setting, (pano, valid, count) in results.items():
            metrics, estimate, ref = compare(pano, valid, reference, align=args.align)
            row = {"setting": setting, "views": count, **flat(metrics)}
            if reference is not None and shared.any():
                # same texels for every setting: does a second can improve what one can already sees?
                row["psnr_shared"] = compare(pano, valid, reference, region=shared, align=args.align)[0]["psnr"]
            rows.append(row)
            print(row, flush=True)
            strips.append(side_by_side(estimate, None, valid))
            save_png(out / f"pano_{setting}_{count}.png", side_by_side(estimate, ref, valid))
    (out / "sweep.json").write_text(json.dumps(rows, indent=2))
    write_table(out / "sweep.csv", rows)
    grid = [np.hstack(strips[i : i + len(settings)]) for i in range(0, len(strips), len(settings))]
    save_png(out / "panoramas.png", np.vstack(grid))
    line_chart(out / "coverage.png", rows, "coverage", "fraction of the sphere observed", "Environment coverage")
    line_chart(out / "coverage_upper.png", rows, "coverage_upper", "fraction of the upper hemisphere observed", "Coverage above the horizon")
    if reference is not None:
        line_chart(out / "psnr.png", rows, "psnr", "PSNR over observed texels (dB)", "Fidelity where each setting has observations")
        line_chart(out / "psnr_shared.png", rows, "psnr_shared", "PSNR over shared texels (dB)", "Fidelity on texels every setting observes")
    print(f"wrote {out}")


def ablation(args):
    """Label separation: does subtracting the diffuse layer matter, and how good is an estimated one?"""
    from data import Scene, pick_device
    from envmap import compare, load_reference, save_png, side_by_side
    from label import estimate, write
    from splat import reconstruct

    device = pick_device()
    scene = Scene(device)
    write(scene, estimate(scene, scene.train), len(scene.arrays["rgb"]))
    reference = load_reference(args.reference)
    out = RESULTS / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    variants = {
        "no subtraction": (scene, "none"),
        "given diffuse layer": (scene, "texture"),
        "estimated label (min over views)": (Scene(device, diffuse="diffuse_est"), "texture"),
    }
    rows, strips = [], []
    for name, (source, mode) in variants.items():
        for cans in ["a"] if len(scene.cans) < 2 else ["a", "both"]:
            pano, valid, frames = reconstruct(source, cans, args.views[0], args.view_select, diffuse=mode, seed=args.seed)
            metrics, estimate_d, ref = compare(pano, valid, reference, align=args.align)
            rows.append({"variant": name, "setting": cans, "views": len(frames), **flat(metrics)})
            print(rows[-1], flush=True)
            if cans == "a":
                strips.append(side_by_side(estimate_d, None, valid))
    if reference is not None:
        strips.append(ref)
    save_png(out / "panoramas.png", np.vstack(strips))
    (out / "ablation.json").write_text(json.dumps(rows, indent=2))
    write_table(out / "ablation.csv", rows)
    print(f"wrote {out}; panoramas.png rows: {', '.join(variants)}, reference")


def merge(args):
    """Put sweeps from separately modelled scenes on one chart (e.g. a one-can and a two-can scene)."""
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    markers = ["o", "s", "^", "D"]
    rows, styles = [], {}
    for scene in args.scenes:
        path = ROOT / "scenes" / scene / "results" / f"sweep_{args.method}" / "sweep.json"
        for row in json.loads(path.read_text()):
            if row["setting"] in args.settings:
                key = f"{scene}/{row['setting']}"
                rows.append({**row, "series": key})
                styles.setdefault(key, None)
    assert len(styles) <= len(palette), "at most four series per chart; narrow --settings"
    for i, key in enumerate(styles):
        scene, setting = key.split("/")
        styles[key] = (f"{STYLE[setting][0]}, scene {scene}", palette[i], markers[i])
    out = ROOT / "results" / "merge"
    out.mkdir(parents=True, exist_ok=True)
    write_table(out / "merge.csv", rows)
    line_chart(out / "coverage.png", rows, "coverage", "fraction of the sphere observed", "Environment coverage", "series", styles)
    if any(r.get("psnr") is not None for r in rows):
        line_chart(out / "psnr.png", rows, "psnr", "PSNR over observed texels (dB)", "Fidelity where each setting has observations", "series", styles)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sweep", "ablation", "merge"])
    ap.add_argument("--method", choices=["splat", "srm"], default="splat")
    ap.add_argument("--views", type=int, nargs="+", default=[4, 8, 16, 32, 64, 0], help="0 means all training views")
    ap.add_argument("--view-select", choices=["fps", "random", "arc"], default="fps")
    ap.add_argument("--buffer", choices=["diffuse", "diffuse_est"], default="diffuse")
    ap.add_argument("--steps", type=int, default=4000, help="srm: training steps per setting")
    ap.add_argument("--train-args", default="--lambda-g 0", help="srm: extra train.py arguments")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--scenes", nargs="+", default=[], help="merge: scene names")
    ap.add_argument("--settings", nargs="+", default=["a", "both"], help="merge: settings to keep")
    args = ap.parse_args()
    {"sweep": sweep, "ablation": ablation, "merge": merge}[args.command](args)


if __name__ == "__main__":
    main()
