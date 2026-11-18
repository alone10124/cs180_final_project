"""Regenerate every report figure from one script.

  python scripts/make_figures.py [--models runs/restore/*/best.pt] [--slow] [--out figures]

Writes into figures/ (each chart has a CSV with the same numbers):
  roundtrip.png              Phase 0: mirror can -> panorama, coverage, reference
  frames_psnr.png/.csv       single frame vs video: PSNR and coverage against frame count
  frames_coverage.png
  label_ablation.png/.csv    label removal: none / min / median / percentile / true label
  sim_example.png            a printed can: frame, label estimate, measured, enhanced, reference
  relight.png                extension: a sphere lit by the measured panorama and by the room
  restore_table.md/.json     with --models: classical vs each model on held-out rooms
  real_<name>.png            for every runs/real/<name>/panorama.npz from scripts/run_pipeline.py
  dent_map.png               with --slow: dent-map refinement on a dented can
"""
import argparse
import copy
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sodacan import debug  # noqa: E402
from sodacan.color import auto_exposure, to_uint8  # noqa: E402
from sodacan.config import REPO_ROOT, load_can, load_pipeline  # noqa: E402
from sodacan.extensions.relight import fill_unobserved, render_sphere  # noqa: E402
from sodacan.geometry.camera import Intrinsics  # noqa: E402
from sodacan.geometry.equirect import resize_pano  # noqa: E402
from sodacan.metrics import all_metrics, psnr  # noqa: E402
from sodacan.pipeline import reconstruct  # noqa: E402
from sodacan.sim.dataset import random_label  # noqa: E402
from sodacan.sim.fast_render import RenderParams, render  # noqa: E402
from sodacan.sim.rooms import procedural_room  # noqa: E402
from sodacan.sim.trajectories import arc_with_tilt  # noqa: E402

# validated categorical palette (dataviz reference instance), fixed order, never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
PANO = (256, 512)


def style(ax, title, ylabel, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_ylabel(ylabel, color=INK2)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def line_chart(path, xs, series, title, ylabel, xlabel, logx=False):
    """series: list of (name, ys) in fixed order; legend plus direct end labels."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    for k, (name, ys) in enumerate(series):
        ax.plot(xs, ys, color=SERIES[k], linewidth=2, marker="o", markersize=5, label=name,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
    style(ax, title, ylabel, xlabel)
    ax.legend(frameon=False, labelcolor=INK2, loc="lower right" if not logx else "best")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def bar_chart(path, names, values, title, ylabel):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    bars = ax.bar(names, values, color=SERIES[0], width=0.6, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", color=INK2, fontsize=9)
    style(ax, title, ylabel)
    ax.set_ylim(0, max(values) * 1.12)  # bars start at zero
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def scene(device, labelled, frames=64, seed=4):
    can = load_can(material={"model": "coated" if labelled else "mirror"})
    env = procedural_room(2 * PANO[0], 2 * PANO[1], seed=seed, device=device, person=True)
    K = Intrinsics.from_fov(480, 640, 24.0)
    c2w = arc_with_tilt(frames, distance=0.3, arc_deg=60, elevation_deg=(-20, 28), cycles=2, target_z=can.height / 2, device=device)
    rng = np.random.default_rng(seed)
    if labelled:
        label = random_label(rng, *can.atlas, device)
        p = RenderParams(roughness_deg=0.8, label=label, ambient=0.6, noise_read=0.002, noise_shot=0.001, quantize=True)
    else:
        label = None
        p = RenderParams(ss=2)
    frames_ = render(env, c2w, K, can, p, generator=torch.Generator().manual_seed(seed))
    return can, env, K, c2w, frames_, label


def fig_roundtrip(out, device):
    can, env, K, c2w, frames, _ = scene(device, False, 60)
    cfg = load_pipeline(pano=list(PANO), label={"method": "none"})
    r = reconstruct(frames, c2w, K, can, cfg)
    ref = resize_pano(env, *PANO)
    m = all_metrics(r.pano, ref, r.valid, with_lpips=False)
    e = auto_exposure(ref, r.valid)
    debug.save_rgb(out / "roundtrip.png", debug.stack(
        [debug.display(r.pano, r.valid, e), debug.heat(r.weight, r.valid), debug.display(ref, None, e)],
        labels=[f"measured (mirror can, 60 frames): {m['psnr']:.1f} dB inside coverage", f"coverage {m['coverage']:.0%} of the sphere", "true room"]))
    return m


def fig_frames(out, device):
    counts = [1, 2, 4, 8, 16, 32, 64]
    rows, psnrs, covs = [], {}, {}
    for labelled, name in ((False, "mirror can"), (True, "printed can")):
        can, env, K, c2w, frames, _ = scene(device, labelled, 64)
        cfg = load_pipeline(pano=list(PANO), **({} if labelled else {"label": {"method": "none"}}))
        ref = resize_pano(env, *PANO)
        psnrs[name], covs[name] = [], []
        for n in counts:
            idx = torch.linspace(0, 63, n).round().long() if n > 1 else torch.tensor([32])
            r = reconstruct(frames[idx], c2w[idx], K, can, cfg)
            m = all_metrics(r.pano, ref, r.valid, with_lpips=False)
            psnrs[name].append(m["psnr"])
            covs[name].append(100 * m["coverage"])
            rows.append([name, n, m["psnr"], m["ssim"], m["coverage"]])
    write_csv(out / "frames_ablation.csv", ["can", "frames", "psnr", "ssim", "coverage"], rows)
    line_chart(out / "frames_psnr.png", counts, list(psnrs.items()), "Single frame vs video: fidelity (each inside its own coverage)", "PSNR (dB)", "frames", logx=True)
    line_chart(out / "frames_coverage.png", counts, list(covs.items()), "Single frame vs video: coverage", "sphere observed (%)", "frames", logx=True)


def fig_label(out, device, seeds=(4, 5, 6)):
    methods = [("none", {"method": "none"}), ("min", {"method": "min"}), ("median", {"method": "median"}),
               ("20th pct", {"method": "percentile", "percentile": 0.2}), ("true label", None)]
    scores = {m: [] for m, _ in methods}
    for s in seeds:
        can, env, K, c2w, frames, label = scene(device, True, 64, seed=s)
        ref = resize_pano(env, *PANO)
        for name, lab in methods:
            if lab is None:
                # the ink as the camera sees it: albedo * ambient (the renderer's diffuse term at F ~ 0)
                from sodacan.sim.fast_render import mean_radiance

                given = label * (mean_radiance(env) * 0.6)[:, None, None]
                r = reconstruct(frames, c2w, K, can, load_pipeline(pano=list(PANO)), label=given)
            else:
                r = reconstruct(frames, c2w, K, can, load_pipeline(pano=list(PANO), label=lab))
            scores[name].append(psnr(r.pano, ref, r.valid))
    names = list(scores)
    vals = [float(np.mean(scores[n])) for n in names]
    write_csv(out / "label_ablation.csv", ["method", "psnr_mean", *[f"seed_{s}" for s in seeds]],
              [[n, v, *scores[n]] for n, v in zip(names, vals)])
    bar_chart(out / "label_ablation.png", names, vals, "Label removal on a printed can (3 rooms)", "PSNR inside coverage (dB)")


def fig_example(out, device, models):
    can, env, K, c2w, frames, _ = scene(device, True, 64)
    cfg = load_pipeline(pano=list(PANO))
    r = reconstruct(frames, c2w, K, can, cfg)
    ref = resize_pano(env, *PANO)
    e = auto_exposure(ref, r.valid)
    panels = [debug.display(r.pano, r.valid, e)]
    labels = [f"measured: {psnr(r.pano, ref, r.valid):.1f} dB"]
    enhanced = None
    for path in models:
        from sodacan.restore.infer import enhance, load_model

        enhanced = enhance(load_model(path, device), r.pano, r.weight, r.valid)
        panels.append(debug.display(enhanced, r.valid, e))
        labels.append(f"prior-enhanced ({Path(path).parent.name}): {psnr(enhanced, ref, r.valid):.1f} dB")
    panels.append(debug.display(ref, r.valid, e))
    labels.append("true room (same mask)")
    frame = to_uint8(frames[32], auto_exposure(frames[32]))
    lab = debug.display(r.label, r.label_count > 0)
    top = debug.montage([frame, np.ascontiguousarray(np.pad(lab, ((0, frame.shape[0] - lab.shape[0]), (0, 0), (0, 0)), constant_values=255))],
                        cols=2, labels=["input frame", "label estimate (surface space)"])
    debug.save_rgb(out / "sim_example_inputs.png", top)
    debug.save_rgb(out / "sim_example.png", debug.stack(panels, labels))
    # relighting extension
    imgs = [render_sphere(fill_unobserved(r.pano, r.valid))[0]]
    names = ["lit by measured (gaps: row means)"]
    if enhanced is not None:
        imgs.append(render_sphere(fill_unobserved(enhanced, r.valid))[0])
        names.append("lit by enhanced")
    imgs.append(render_sphere(ref)[0])
    names.append("lit by the true room")
    gain = float((ref * r.valid).sum() / (r.pano * r.valid).sum().clamp_min(1e-9))
    imgs[:-1] = [i * gain for i in imgs[:-1]]
    ee = auto_exposure(imgs[-1], imgs[-1].mean(0) > 0)
    debug.save_rgb(out / "relight.png", debug.montage([debug.display(i, None, ee) for i in imgs], cols=len(imgs), labels=names))


def fig_real(out):
    for f in sorted(glob.glob(str(REPO_ROOT / "runs" / "real" / "*" / "panorama.npz"))):
        d = np.load(f)
        valid = torch.from_numpy(d["valid"])
        pano = torch.from_numpy(d["pano"])
        e = auto_exposure(pano, valid)
        panels, labels = [debug.display(pano, valid, e)], ["measured"]
        if "enhanced" in d:
            panels.append(debug.display(torch.from_numpy(d["enhanced"]), valid, e))
            labels.append("prior-enhanced (may add plausible, unmeasured detail)")
        if "reference" in d:
            ref = torch.from_numpy(d["reference"])
            panels.append(to_uint8(ref, auto_exposure(ref, valid)))
            labels.append("phone panorama (aligned)")
        debug.save_rgb(out / f"real_{Path(f).parent.name}.png", debug.stack(panels, labels))


def fig_dent(out, device):
    from sodacan.refine.dent_map import fit_dent_map
    from sodacan.sim.dataset import random_dent

    can = load_can(material={"model": "mirror"})
    cfg = load_pipeline(pano=list(PANO), label={"method": "none"})
    env = procedural_room(512, 1024, seed=5, device=device)
    K = Intrinsics.from_fov(480, 640, 24.0)
    c2w = arc_with_tilt(48, distance=0.3, arc_deg=60, elevation_deg=(-20, 25), cycles=2, target_z=can.height / 2, device=device)
    dent = random_dent(np.random.default_rng(1), *can.atlas, 1.0, device)
    frames = render(env, c2w, K, can, RenderParams(dent=dent, ss=2))
    ref = resize_pano(env, *PANO)
    plain = reconstruct(frames, c2w, K, can, cfg)
    fitted_dent, fitted = fit_dent_map(frames, c2w, K, can, cfg)
    m = plain.valid & fitted.valid
    e = auto_exposure(ref, m)
    mag = lambda d: debug.heat(d.norm(dim=0) * 57.3, None, log=False)  # noqa: E731
    debug.save_rgb(out / "dent_map.png", debug.stack(
        [debug.display(plain.pano, m, e), debug.display(fitted.pano, m, e), debug.display(ref, m, e)],
        [f"no dent map: {psnr(plain.pano, ref, m):.1f} dB", f"fitted dent map: {psnr(fitted.pano, ref, m):.1f} dB", "true room"]))
    debug.save_rgb(out / "dent_map_atlas.png", debug.montage([mag(dent), mag(fitted_dent)], cols=1, labels=["true dents", "fitted"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--slow", action="store_true")
    ap.add_argument("--pairs", type=int, default=64, help="held-out pairs for the restoration table")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"roundtrip": fig_roundtrip(out, device)}
    print("roundtrip", summary["roundtrip"], flush=True)
    fig_frames(out, device)
    print("frames ablation done", flush=True)
    fig_label(out, device)
    print("label ablation done", flush=True)
    fig_example(out, device, args.models)
    print("example and relight done", flush=True)
    fig_real(out)
    if args.models:
        from evaluate_restore import evaluate, to_markdown

        table = evaluate(args.models, args.pairs, device)
        (out / "restore_table.json").write_text(json.dumps(table, indent=2))
        (out / "restore_table.md").write_text(to_markdown(table) + "\n", encoding="utf-8")
        print(to_markdown(table))
    if args.slow:
        fig_dent(out, device)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
