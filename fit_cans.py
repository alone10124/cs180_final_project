"""Locate the cans in the scanned mesh and write cache/cans.json (run once, after prepare.py).

Each can is modelled as a vertical cylinder: x, y of the axis, radius, height.
The ids follow the sorted x coordinate: can "a" has the smaller x.
"""
import argparse
import json

import numpy as np

from common import CACHE


def circle_fit(xy):
    """Algebraic least-squares circle through 2D points; returns (cx, cy, radius)."""
    A = np.concatenate([2 * xy, np.ones((len(xy), 1))], axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, (xy**2).sum(axis=1), rcond=None)
    return cx, cy, np.sqrt(c + cx**2 + cy**2)


def fit_cans(vertices, count=2, z_min=0.025, extent=0.2, iterations=50):
    """Cluster raised vertices in XY with k-means, then fit a circle to each cluster's wall."""
    raised = vertices[(vertices[:, 2] > z_min) & (np.abs(vertices[:, :2]).max(axis=1) < extent)]
    xy = raised[:, :2]
    centres = [xy[np.argmin(xy[:, 0])]]
    while len(centres) < count:
        dist = np.min([np.linalg.norm(xy - c, axis=1) for c in centres], axis=0)
        centres.append(xy[np.argmax(dist)])
    centres = np.array(centres)
    for _ in range(iterations):
        label = np.argmin(np.linalg.norm(xy[:, None] - centres[None], axis=2), axis=1)
        centres = np.array([xy[label == k].mean(axis=0) for k in range(count)])
    cans = []
    for k in range(count):
        points = raised[label == k]
        height = np.quantile(points[:, 2], 0.99)
        wall = points[(points[:, 2] > 0.2 * height) & (points[:, 2] < 0.8 * height)]
        cx, cy, radius = circle_fit(wall[:, :2])
        cans.append({"x": float(cx), "y": float(cy), "radius": float(radius), "height": float(height)})
    return sorted(cans, key=lambda c: c["x"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2)
    args = ap.parse_args()
    from prepare import read_mesh

    cans = fit_cans(read_mesh()[0], args.count)
    CACHE.mkdir(exist_ok=True)
    (CACHE / "cans.json").write_text(json.dumps(cans, indent=2))
    for name, can in zip("ab", cans):
        print(f"can {name}: centre ({can['x']:.3f}, {can['y']:.3f}) radius {can['radius']:.3f} height {can['height']:.3f}")
    print("a standard can is 0.033 m in radius and 0.122 m tall; check the fit against that")


if __name__ == "__main__":
    main()
