"""Fit the simulator to a real capture: roughness blur, reflection strength, noise. Also
export the real can's label atlas so the simulator can use it.

  python -m sodacan.refine.roughness_fit --video can.mp4 --phone configs/phone_pixel.yaml \
      [--reference room_pano.jpg] --out runs/fit_kitchen

Writes fit.json and label.png (into --out and, with --save-label, data/labels/).
Centre the ranges in configs/sim.yaml on these numbers, then randomise around them.

  noise          temporal differences of consecutive frames in surface space, fitted as
                 variance = shot * signal + read^2 (adjacent frames see nearly the same
                 reflection, so their difference is mostly sensor noise)
  strength       mean |frame - label| over the mean label: how strong the reflection is
                 relative to the print. The same statistic is computed on simulated
                 captures (--compare-sim) so the ranges can be matched directly
  roughness_deg  with a reference panorama: the Gaussian blur (degrees) of the aligned
                 reference that best matches the measured panorama. Without one, a crude
                 upper bound from the width of the brightest highlight
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sodacan.capture.uv_warp import warp_to_surface
from sodacan.color import to_uint8
from sodacan.config import load_can, load_pipeline, load_yaml
from sodacan.geometry import equirect
from sodacan.metrics import align_azimuth, psnr
from sodacan.pipeline import evenly, reconstruct
from sodacan.separate.label_median import estimate_label


def capture_statistics(frames, c2w, K, can, cfg):
    idx = evenly(frames.shape[0], 48).to(frames.device)
    obs, valid = warp_to_surface(frames[idx], c2w[idx], K, can)
    label, count = estimate_label(obs, valid, "percentile", cfg["label"].get("percentile", 0.2))
    both = valid[1:] & valid[:-1]
    diff = (obs[1:] - obs[:-1])[both[:, None].expand(-1, 3, -1, -1)]
    level = (0.5 * (obs[1:] + obs[:-1]))[both[:, None].expand(-1, 3, -1, -1)]
    # variance in intensity bins -> var = shot * level + read^2
    bins = torch.quantile(level, torch.linspace(0.05, 0.95, 10, device=level.device))
    xs, ys = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (level >= a) & (level < b)
        if sel.sum() > 100:
            d = diff[sel]
            mad = (d - d.median()).abs().median() * 1.4826
            xs.append(float(level[sel].mean()))
            ys.append(float(mad**2 / 2))
    A = np.stack([xs, np.ones(len(xs))], 1)
    (shot, read2), *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    lab = label[:, count > 3]
    resid = (obs - label[None]).abs()[valid[:, None].expand(-1, 3, -1, -1)]
    return {
        "noise_shot": float(max(shot, 0)),
        "noise_read": float(np.sqrt(max(read2, 0))),
        "reflection_to_label": float(resid.mean() / lab.mean().clamp_min(1e-6)),
        "label_mean": float(lab.mean()),
    }, label


def fit_roughness(pano, valid, reference, sigmas=(0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4)):
    ref, _, _ = align_azimuth(pano, reference, valid)
    scores = {s: psnr(pano, equirect.blur(ref, s), valid) for s in sigmas}
    best = max(scores, key=scores.get)
    return best, scores


def highlight_width(pano, valid):
    lum = pano.mean(0) * valid
    i, j = np.unravel_index(int(torch.argmax(lum)), lum.shape)
    row = lum[i].cpu().numpy()
    peak = row[j]
    half = np.where(np.roll(row, -j) < peak / 2)[0]
    if half.size == 0:
        return float("nan")
    right = half.min()
    left = len(row) - half.max()
    fwhm_cols = right + left
    return float(fwhm_cols / len(row) * 360 / 2.355)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--phone", default="phone_example")
    ap.add_argument("--board", default="board")
    ap.add_argument("--reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-label", action="store_true", help="also copy the label atlas into data/labels/")
    ap.add_argument("--compare-sim", type=int, default=0, help="also report the statistics of N simulated captures")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    can, cfg = load_can(), load_pipeline()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    if args.video:
        from sodacan.capture.load import load_capture, load_reference

        cap = load_capture(args.video, args.phone, args.board, cfg, device=device)
        stats, label = capture_statistics(cap.frames, cap.c2w, cap.K, can, cfg)
        report["capture"] = stats
        img = to_uint8(label / label.max().clamp_min(1e-6) * 0.85)
        cv2.imwrite(str(out / "label.png"), img[..., ::-1])
        if args.save_label:
            Path("data/labels").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path("data/labels") / f"{out.name}.png"), img[..., ::-1])
        res = reconstruct(cap.frames, cap.c2w, cap.K, can, cfg, label=label)
        if args.reference:
            ref = load_reference(args.reference, *cfg["pano"]).to(device)
            best, scores = fit_roughness(res.pano, res.valid, ref)
            report["roughness_deg"] = best
            report["roughness_scores"] = scores
        else:
            report["roughness_deg_upper_bound"] = highlight_width(res.pano, res.valid)
    if args.compare_sim:
        from sodacan.sim.dataset import sample_scene

        sim = load_yaml("sim")
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(args.compare_sim):
            sc = sample_scene(rng, sim, can, device=device)
            s, _ = capture_statistics(sc.frames, sc.c2w, sc.K, sc.can, cfg)
            rows.append(s)
        report["sim_percentiles_10_50_90"] = {k: [float(np.percentile([r[k] for r in rows], q)) for q in (10, 50, 90)] for k in rows[0]}
    (out / "fit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
