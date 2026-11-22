"""Import your own photos of cans as a scene, from a COLMAP reconstruction.

  colmap automatic_reconstructor --workspace_path ws --image_path photos --single_camera 1
  colmap model_converter --input_path ws/sparse/0 --output_path ws/txt --output_type TXT
  python capture.py --name mine --images photos --model ws/txt --cans 2 --reference room_pano.jpg

  SCENE=mine python label.py                                    # estimate the printed label
  SCENE=mine python splat.py --buffer diffuse_est --align       # recover the environment

Steps: fit the table plane to the sparse points (RANSAC) and make it z = 0 with z up;
find the lids as the densest height above the table, split them into one cluster per
can, and take each can's axis from its cluster centre; rescale so the lids sit at the
real can height; then build the geometry buffers from the fitted cylinders. The mirror
walls give COLMAP almost no points, so the fit relies on the lids being visible. Lens
distortion is ignored, so undistort first if the lens is wide. Check data/check_*.jpg:
the drawn outlines must sit on the cans, otherwise every reflection direction is off.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from common import ROOT
from synth import write_scene


def quat_to_matrix(w, x, y, z):
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_colmap(model):
    """Returns (camera dict, {image name: 3x4 camera-to-world}, points (N, 3))."""
    model = Path(model)
    fields = [l.split() for l in (model / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")][0]
    kind, width, height, params = fields[1], int(fields[2]), int(fields[3]), [float(p) for p in fields[4:]]
    if kind in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        focal, cx, cy = (params[0] + params[1]) / 2, params[2], params[3]
    else:
        focal, cx, cy = params[0], params[1], params[2]
    if kind not in ("PINHOLE", "SIMPLE_PINHOLE"):
        print(f"warning: camera model {kind} has lens distortion, which is ignored")
    lines = [l for l in (model / "images.txt").read_text().splitlines() if not l.startswith("#")]
    poses = {}
    for line in lines[::2]:
        f = line.split()
        if len(f) < 10:
            continue
        R = quat_to_matrix(*map(float, f[1:5])).T  # COLMAP stores world-to-camera
        t = -R @ np.array(f[5:8], float)
        poses[f[9]] = np.concatenate([R, t[:, None]], axis=1)
    rows = [l.split() for l in (model / "points3D.txt").read_text().splitlines() if l and not l.startswith("#")]
    points = np.array([[float(v) for v in r[1:4]] for r in rows])
    return {"width": width, "height": height, "focal": focal, "cx": cx, "cy": cy}, poses, points


def fit_plane(points, tolerance, rng, iterations=2000):
    """RANSAC plane; returns (unit normal, point on plane, inlier mask)."""
    best = None
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-12:
            continue
        normal /= np.linalg.norm(normal)
        inliers = np.abs((points - a) @ normal) < tolerance
        if best is None or inliers.sum() > best.sum():
            best = inliers
    centre = points[best].mean(axis=0)
    normal = np.linalg.svd(points[best] - centre)[2][2]
    return normal, centre, best


def fit_lids(local, count, extent, focus, reach, iterations=20):
    """Lid height and one (cx, cy, r) per can, in model units.

    The lids are the densest height above the table near where the cameras look.
    Points in that band are kept where they are dense, split into one cluster per can
    (farthest-point start, then k-means), and r is the radius of a filled disc with
    the cluster's spread.
    """
    raised = local[(local[:, 2] > 0.03 * extent) & (np.hypot(*(local[:, :2] - focus[:2]).T) < reach)]
    hist, edges = np.histogram(raised[:, 2], bins=100)
    top = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    xy = raised[np.abs(raised[:, 2] - top) < 0.05 * top][:, :2]
    cell = np.floor(xy / (0.02 * reach)).astype(int)
    _, inverse, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    xy = xy[counts[inverse.ravel()] >= 0.2 * counts.max()]
    assert len(xy) >= 10 * count, "no lids found; the can tops must be visible in the photos"
    centres = [xy[np.argmax(np.hypot(*(xy - np.median(xy, axis=0)).T))]]
    while len(centres) < count:
        d = np.min([np.hypot(*(xy - c).T) for c in centres], axis=0)
        centres.append(xy[np.argmax(d)])
    centres = np.array(centres, float)
    for _ in range(iterations):
        label = np.argmin([np.hypot(*(xy - c).T) for c in centres], axis=0)
        centres = np.array([xy[label == k].mean(axis=0) for k in range(count)])
    lids = []
    for k in range(count):
        spread = np.mean(np.sum((xy[label == k] - centres[k]) ** 2, axis=1))
        lids.append((centres[k][0], centres[k][1], np.sqrt(2 * spread)))
    return top, lids


def align(poses, points, count, height, seed=0):
    """Similarity transform to a metric frame with the table at z = 0 and the cans at the origin.

    Returns (poses (N, 3, 4), cans [(x, y)], lid radii in metres).
    """
    rng = np.random.default_rng(seed)
    centres = poses[:, :, 3]
    extent = np.median(np.linalg.norm(points - np.median(points, axis=0), axis=1))
    normal, origin, _ = fit_plane(points, 0.01 * extent, rng)
    if (centres - origin).mean(axis=0) @ normal < 0:
        normal = -normal
    x = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(normal, x), normal])
    local = (points - origin) @ R.T
    cams = (centres - origin) @ R.T
    # where the cameras look: the point nearest to every optical axis
    axes = np.einsum("ij,nj->ni", R, poses[:, :, 2])
    A = np.eye(3)[None] - axes[:, :, None] * axes[:, None, :]
    focus = np.linalg.solve(A.sum(axis=0), np.einsum("nij,nj->i", A, cams))
    reach = np.median(np.hypot(*(cams[:, :2] - focus[:2]).T))
    top, lids = fit_lids(local, count, extent, focus, reach)
    scale = height / top
    middle = np.array([*np.mean([l[:2] for l in lids], axis=0), 0])
    new = np.concatenate([np.einsum("ij,njk->nik", R, poses[:, :, :3]), (scale * (cams - middle))[:, :, None]], axis=2)
    order = np.lexsort(np.array([l[:2] for l in lids]).T[::-1])
    cans = [(scale * (lids[i][0] - middle[0]), scale * (lids[i][1] - middle[1])) for i in order]
    return new, cans, [scale * lids[i][2] for i in order]


def outline(image, pose, spec):
    """Draw each can's bottom and top circle to check the fit."""
    angles = np.linspace(0, 2 * np.pi, 90)
    for can in spec["cans"]:
        for z in (0, can["height"]):
            ring = np.stack([can["x"] + can["radius"] * np.cos(angles), can["y"] + can["radius"] * np.sin(angles), np.full(90, z)], axis=1)
            cam = (ring - pose[:, 3]) @ pose[:, :3]
            uv = spec["focal"] * cam[:, :2] / cam[:, 2:3] + (spec["cx"], spec["cy"])
            cv2.polylines(image, [uv.astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", required=True, help="directory of the photos given to COLMAP")
    ap.add_argument("--model", required=True, help="COLMAP model exported as TXT")
    ap.add_argument("--cans", type=int, default=1)
    ap.add_argument("--radius", type=float, default=0.033, help="real can radius in metres")
    ap.add_argument("--height", type=float, default=0.122, help="real can height in metres")
    ap.add_argument("--width", type=int, default=640, help="photos are resized to this width")
    ap.add_argument("--gamma", type=float, default=2.2)
    ap.add_argument("--reference", default=None, help="equirectangular photo of the room, for evaluation")
    args = ap.parse_args()

    camera, by_name, points = read_colmap(args.model)
    names = sorted(by_name)
    poses, cans, lid_radii = align(np.stack([by_name[n] for n in names]), points, args.cans, args.height)
    scale = args.width / camera["width"]
    spec = {
        "width": args.width,
        "height": round(camera["height"] * scale),
        "focal": camera["focal"] * scale,
        "cx": camera["cx"] * scale,
        "cy": camera["cy"] * scale,
        "gamma": args.gamma,
        "cans": [{"x": float(x), "y": float(y), "radius": args.radius, "height": args.height} for x, y in cans],
        "floor_half": 0.5,
        "reflectance": 1.0,
        "label": False,
    }
    base = ROOT / "scenes" / args.name
    (base / "data").mkdir(parents=True, exist_ok=True)
    if args.reference:
        spec["env"] = "reference" + Path(args.reference).suffix
        shutil.copy(args.reference, base / "data" / spec["env"])

    def photo(i):
        image = cv2.imread(str(Path(args.images) / names[i]))
        assert image is not None, f"cannot read {names[i]}"
        return cv2.resize(image, (spec["width"], spec["height"]), interpolation=cv2.INTER_AREA)[..., ::-1]

    write_scene(base, spec, poses, photo, floor_from_photo=True)
    for i in np.linspace(0, len(names) - 1, min(6, len(names))).astype(int):
        cv2.imwrite(str(base / "data" / f"check_{i + 1:04d}.jpg"), outline(photo(i)[..., ::-1].copy(), poses[i], spec))
    for name, (x, y), r in zip("ab", cans, lid_radii):
        print(f"can {name} at ({x:.3f}, {y:.3f}) m, lid radius {r:.4f} m")
    if not all(0.6 * args.radius < r < 1.0 * args.radius for r in lid_radii):
        print(f"warning: lid radii should be a little under --radius {args.radius}; the lid fit or --height is off")
    print(f"wrote {base}; inspect data/check_*.jpg before using the scene")


if __name__ == "__main__":
    main()
