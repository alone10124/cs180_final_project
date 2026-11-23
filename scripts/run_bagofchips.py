"""Run the soda-can pipeline on the Cans scene of "Seeing the World in a Bag of Chips"
(Park et al., CVPR 2020): a Pepsi and a Diet Coke lying on their sides on a rug, filmed
with known poses (2,265 frames, 640 x 480).

  python scripts/run_bagofchips.py --stride 8 --out runs/bagofchips

Data: cans_data/ as described in INPUTS.md (rgb/, xy_poses.txt, xy_mesh.ply). Per can:
  1. fit a cylinder with a free axis to the scanned mesh (the cans are horizontal),
  2. re-express every camera pose in that can's frame (z along the axis, base at z = 0),
  3. drop frames in which the other can overlaps this one in the image,
  4. run the unchanged pipeline (label removal, accumulation) in the can's frame,
  5. turn the panorama into the world frame (z up), and merge the two cans.
There is no ground-truth room panorama for this scene, so the output is coverage and
pictures, not PSNR.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sodacan import debug  # noqa: E402
from sodacan.color import auto_exposure  # noqa: E402
from sodacan.config import load_can, load_pipeline  # noqa: E402
from sodacan.geometry import equirect  # noqa: E402
from sodacan.geometry.camera import Intrinsics  # noqa: E402
from sodacan.geometry.reflect import can_geometry  # noqa: E402
from sodacan.pipeline import reconstruct  # noqa: E402

DATA = ROOT / "cans_data"
K = Intrinsics(640, 480, 541.961, 541.961, 320.0, 240.0)  # from the project page (common.py)
GAMMA = 1.6  # the dataset's linearisation: linear = pixel ** 1.6 (common.py)
NAMES = ("diet_coke", "pepsi")  # sorted by world x, checked on outline_check.png


def read_mesh_vertices():
    path = DATA / "xy_mesh.ply"
    with open(path, encoding="ascii") as f:
        header = 0
        for line in f:
            header += 1
            if line.startswith("element vertex"):
                num_v = int(line.split()[-1])
            elif line.startswith("end_header"):
                break
    return np.loadtxt(path, skiprows=header, max_rows=num_v, dtype=np.float64, usecols=(0, 1, 2))


def circle_fit(xy):
    A = np.concatenate([2 * xy, np.ones((len(xy), 1))], axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, (xy**2).sum(axis=1), rcond=None)
    return np.array([cx, cy]), float(np.sqrt(c + cx**2 + cy**2))


def fit_cylinders(v, count=2, z_min=0.02, extent=0.25, iterations=50):
    """Free-axis cylinders: k-means on the raised vertices, then per cluster the axis from
    PCA (a can is longer than it is wide), a circle fit across the axis, and the ends from
    the extent along it. Returns can-to-world 4x4 transforms (z = axis, origin = one end)
    and lengths."""
    raised = v[(v[:, 2] > z_min) & (np.abs(v[:, :2]).max(axis=1) < extent)]
    xy = raised[:, :2]
    centres = [xy[np.argmin(xy[:, 0])]]
    while len(centres) < count:
        centres.append(xy[np.argmax(np.min([np.linalg.norm(xy - c, axis=1) for c in centres], axis=0))])
    centres = np.array(centres)
    for _ in range(iterations):
        lab = np.argmin(np.linalg.norm(xy[:, None] - centres[None], axis=2), axis=1)
        centres = np.array([xy[lab == k].mean(axis=0) for k in range(count)])
    out = []
    for k in range(count):
        p = raised[lab == k]
        c = p.mean(0)
        axis = np.linalg.svd(p - c, full_matrices=False)[2][0]
        x = np.cross([0.0, 0.0, 1.0], axis)
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
        y = np.cross(axis, x)
        q = p - c
        centre2, radius = circle_fit(np.stack([q @ x, q @ y], 1))
        t = q @ axis
        lo, hi = np.quantile(t, [0.005, 0.995])
        origin = c + centre2[0] * x + centre2[1] * y + lo * axis
        T = np.eye(4)
        T[:3, :3] = np.stack([x, y, axis], 1)
        T[:3, 3] = origin
        out.append({"T": T, "radius": radius, "length": float(hi - lo), "points": len(p)})
    return sorted(out, key=lambda c: c["T"][0, 3])


def load_poses():
    rows = np.loadtxt(DATA / "xy_poses.txt", skiprows=1, ndmin=2)
    c2w = np.tile(np.eye(4), (len(rows), 1, 1))
    c2w[:, :3, :3] = rows[:, 4:].reshape(-1, 3, 3)
    c2w[:, :3, 3] = rows[:, 1:4]
    return c2w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=8, help="use every n-th frame")
    ap.add_argument("--out", default="runs/bagofchips")
    ap.add_argument("--pano", type=int, nargs=2, default=[256, 512])
    ap.add_argument("--max-overlap", type=float, default=0.01, help="drop frames where the other can covers more than this fraction")
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--radius", type=float, default=None, help="override the fitted can radius (m)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cans = fit_cylinders(read_mesh_vertices())
    for name, c in zip(NAMES, cans):
        tilt = np.degrees(np.arcsin(abs(c["T"][2, 2])))
        print(f"{name}: radius {c['radius'] * 1000:.1f} mm, length {c['length'] * 1000:.1f} mm, "
              f"axis {tilt:.1f} deg from horizontal, {c['points']} mesh vertices", flush=True)
    (out / "cans.json").write_text(json.dumps([{"name": n, "T": c["T"].tolist(), "radius": c["radius"], "length": c["length"]}
                                               for n, c in zip(NAMES, cans)], indent=2))
    c2w_world = load_poses()
    idx = np.arange(0, len(c2w_world), args.stride)
    models = []
    for c in cans:
        can = load_can(radius=args.radius or c["radius"], height=c["length"],
                       usable_z=[0.015, c["length"] - 0.015])  # skip both ends: either may be the neck
        poses = torch.tensor(np.linalg.inv(c["T"])[None] @ c2w_world[idx], dtype=torch.float32)
        models.append((can, poses))

    def read(i):
        rgb = cv2.cvtColor(cv2.imread(str(DATA / "rgb" / f"{i + 1:04d}.png")), cv2.COLOR_BGR2RGB)
        return rgb

    # outline check on a few frames
    checks = []
    for j in np.linspace(0, len(idx) - 1, 6).astype(int):
        img = read(idx[j])
        for (can, poses), colour in zip(models, [(0, 255, 0), (255, 0, 255)]):
            img = debug.can_outline(img, poses[j], K, can, colour)
        checks.append(img)
    debug.save_rgb(out / "outline_check.png", debug.montage(checks, cols=3))
    if args.fit_only:
        return

    # frames in which the other can sits in front of this one (they lie side by side)
    keep = [[], []]
    for j in range(len(idx)):
        g = [can_geometry(poses[j : j + 1], K, can) for can, poses in models]
        for k in (0, 1):
            other = g[1 - k]
            hidden = g[k].mask & other.hit & (other.t < g[k].t)
            n = int(g[k].mask.sum())
            if n > 500 and int(hidden.sum()) <= args.max_overlap * n:
                keep[k].append(j)
    frames = torch.stack([torch.from_numpy(read(i)) for i in idx]).permute(0, 3, 1, 2).float().div(255).pow(GAMMA)

    cfg = load_pipeline(pano=list(args.pano))
    H, W = args.pano
    dirs_w = equirect.pano_dirs(H, W, device)
    num, den, stats = torch.zeros(3, H, W, device=device), torch.zeros(H, W, device=device), []
    panels, labels = [], []
    for k, (name, c, (can, poses)) in enumerate(zip(NAMES, cans, models)):
        j = torch.tensor(keep[k])
        r = reconstruct(frames[j].to(device), poses[j].to(device), K, can, cfg)
        # can frame -> world frame (z up): look up each world direction in the can's panorama
        R = torch.tensor(c["T"][:3, :3], dtype=torch.float32, device=device)
        d_can = dirs_w @ R  # rows are R^T d
        pano_w = equirect.sample(r.pano, d_can).permute(2, 0, 1)
        w_w = equirect.sample(r.weight[None], d_can)[..., 0]
        v_w = equirect.sample(r.valid[None].float(), d_can)[..., 0] > 0.5
        w_w = w_w * v_w
        num += pano_w * w_w
        den += w_w
        cov = float((equirect.solid_angle(H, W, device) * v_w).sum() / (4 * np.pi))
        stats.append({"can": name, "frames": len(keep[k]), "of": len(idx), "coverage": round(cov, 3)})
        print(f"{name}: {len(keep[k])}/{len(idx)} frames unoccluded, coverage {cov:.1%} of the sphere", flush=True)
        panels.append(debug.display(pano_w, v_w, auto_exposure(pano_w, v_w)))
        labels.append(f"{name}: {len(keep[k])} frames, {cov:.0%} of the sphere")
        if r.label is not None:
            debug.save_rgb(out / f"label_{name}.png", debug.display(r.label))
    valid = den > 0
    merged = num / den.clamp_min(1e-12)
    cov = float((equirect.solid_angle(H, W, device) * valid).sum() / (4 * np.pi))
    stats.append({"can": "both", "coverage": round(cov, 3)})
    print(f"both cans: coverage {cov:.1%} of the sphere", flush=True)
    panels.append(debug.display(merged, valid, auto_exposure(merged, valid)))
    labels.append(f"both cans: {cov:.0%} of the sphere (world frame, row 0 = up)")
    debug.save_rgb(out / "panoramas.png", debug.stack(panels, labels))
    np.savez_compressed(out / "panorama.npz", pano=merged.cpu().numpy(), weight=den.cpu().numpy(), valid=valid.cpu().numpy())
    (out / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
