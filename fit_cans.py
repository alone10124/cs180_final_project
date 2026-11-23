"""Locate the two horizontal cans in the released Cans mesh.

The public sequence has cans lying on the carpet, not upright cylinders.  Each
can is represented by a 3D centre, unit axis, radius and half-length.  The old
vertical-cylinder fit silently selected the wrong surface regions.
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


def fit_cans(vertices, count=2, z_min=0.015, extent=0.2, iterations=50):
    """Cluster raised vertices in XY, then fit an arbitrarily oriented cylinder."""
    finite = np.isfinite(vertices).all(axis=1)
    raised = vertices[finite & (vertices[:, 2] > z_min) & (np.abs(vertices[:, :2]).max(axis=1) < extent)]
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
        points = raised[label == k].astype(np.float64)
        centre = points.mean(axis=0)
        centred = points - centre
        # A can's long principal component is its cylinder axis.
        _, vectors = np.linalg.eigh(centred.T @ centred / len(centred))
        axis = vectors[:, -1]
        if axis[0] < 0:
            axis = -axis
        axial = np.einsum("ij,j->i", centred, axis)
        radial = np.linalg.norm(centred - axial[:, None] * axis, axis=1)
        lo, hi = np.quantile(axial, [0.01, 0.99])
        centre = centre + 0.5 * (lo + hi) * axis
        # The mesh contains end caps and some interior samples; an upper radial
        # quantile is more stable than the median for the outer wall.
        radius = np.quantile(radial, 0.78)
        cans.append(
            {
                "center": [float(v) for v in centre],
                "axis": [float(v) for v in axis],
                "radius": float(radius),
                "half_length": float(0.5 * (hi - lo)),
            }
        )
    return sorted(cans, key=lambda c: c["center"][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2)
    args = ap.parse_args()
    from prepare import read_mesh

    cans = fit_cans(read_mesh()[0], args.count)
    CACHE.mkdir(exist_ok=True)
    (CACHE / "cans.json").write_text(json.dumps(cans, indent=2))
    for name, can in zip("ab", cans):
        centre = ", ".join(f"{v:.3f}" for v in can["center"])
        axis = ", ".join(f"{v:.3f}" for v in can["axis"])
        print(
            f"can {name}: centre ({centre}) axis ({axis}) "
            f"radius {can['radius']:.3f} length {2 * can['half_length']:.3f}"
        )
    print("a standard can is about 0.033 m in radius and 0.122 m long; check the fit against that")


if __name__ == "__main__":
    main()
