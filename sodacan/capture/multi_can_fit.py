"""Several identical cans on one surface, fitted together to the photos' edges.

Chrome cans defeat point-based structure (SfM, stereo): their surface shows the room, not
itself, so the points SfM triangulates on them are reflections at the wrong depth. What a
mirror cannot fake is its outline: where the can ends, the image changes from can to room.
This is how inspection systems handle shiny parts (model-based contour / shape matching):
a known model, here cylinders of the can's size, is posed so its projected outline lies on
image edges, in many calibrated views at once (the camera poses from SfM are fine: they come
from the matte room).

The model: an upright can (its base on the surface, axis along the surface normal) and n
cans lying on the surface (axis one radius above it, parallel to it); one scale for all.
The cost, per view: each can's outline (the two limb lines and the front halves of its rim
and base, sodacan.capture.pose_silhouette.outline_points) projected, distance to the nearest
edge in units of that can's projected radius, clipped; outline points hidden behind another
can are not scored (they would land on the other can's label) but cost a constant, so a can
cannot lower its cost by hiding. Cans may not overlap.

Search: each lying can by a grid over centre and heading on the surface (coarse images),
the best few refined; then everything (surface tilt, upright position, scale, every lying
can) refined jointly with a tightening clip.
"""
import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from sodacan.capture.can_fit import lying_pose
from sodacan.capture.pose_silhouette import outline_points, project_np, sample_dist

HIDDEN = 0.6  # cost of a hidden outline point, as a fraction of clip^2


def _seg_dist2d(p1, q1, p2, q2):
    def pt_seg(p, a, b):
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
        return float(np.linalg.norm(p - (a + t * ab)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    if cross(p2, q2, p1) * cross(p2, q2, q1) < 0 and cross(p1, q1, p2) * cross(p1, q1, q2) < 0:
        return 0.0
    return min(pt_seg(p1, p2, q2), pt_seg(q1, p2, q2), pt_seg(p2, p1, q1), pt_seg(q2, p1, q1))


def _blocked(o, q, radius, height, eps=1e-3):
    """Segments from o (3,) to q (N, 3), in a can's metric frame: does the can (wall or
    lids) cut them before they reach q?"""
    dv = q - o
    a = dv[:, 0] ** 2 + dv[:, 1] ** 2
    b = 2 * (o[0] * dv[:, 0] + o[1] * dv[:, 1])
    c = o[0] ** 2 + o[1] ** 2 - radius ** 2
    disc = b * b - 4 * a * c
    ok = (disc > 0) & (a > 1e-12)
    sq = np.sqrt(np.maximum(disc, 0))
    out = np.zeros(len(q), bool)
    for t in ((-b - sq) / (2 * np.maximum(a, 1e-12)), (-b + sq) / (2 * np.maximum(a, 1e-12))):
        z = o[2] + t * dv[:, 2]
        out |= ok & (t > eps) & (t < 1 - eps) & (z >= 0) & (z <= height)
    for zl in (0.0, height):
        dz = dv[:, 2]
        t = (zl - o[2]) / np.where(np.abs(dz) > 1e-12, dz, 1e-12)
        x, y = o[0] + t * dv[:, 0], o[1] + t * dv[:, 1]
        out |= (t > eps) & (t < 1 - eps) & (x * x + y * y <= radius ** 2)
    return out


def _can_terms(data, can, poses, s, k, clip):
    """Clipped squared relative residuals of can k's outline over the views, hidden points
    at HIDDEN * clip^2. poses: [(b, R)] in the SfM frame. Returns (sum, count)."""
    b, R = poses[k]
    tot, cnt = 0.0, 0
    for dist, c2w, K in data:
        Cw = c2w[:3, 3]
        C = R.T @ (Cw - b) / s                        # camera centre, metric can frame
        P, w = outline_points(C, can)
        if len(P) == 0:
            tot += clip ** 2 * 60
            cnt += 60
            continue
        Pw = b + s * (P @ R.T)
        uv, z = project_np(Pw, c2w, K)
        mid = c2w[:3, :3].T @ (b + s * R[:, 2] * can.height / 2 - Cw)
        if mid[2] <= 0:
            tot += clip ** 2 * len(P)
            cnt += len(P)
            continue
        r = sample_dist(dist, uv) * w / (K.fx * s * can.radius / mid[2])
        r = np.where((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= K.width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= K.height - 1),
                     np.minimum(r, clip), clip)
        hidden = np.zeros(len(P), bool)
        for j, (bj, Rj) in enumerate(poses):
            if j != k:
                hidden |= _blocked(Rj.T @ (Cw - bj) / s, (Pw - bj) @ Rj / s, can.radius, can.height)
        e = np.where(hidden, HIDDEN * clip ** 2, r ** 2)
        tot += float(e.sum())
        cnt += len(e)
    return tot, cnt


def group_cost(data, can, poses, s, clip):
    tot, cnt = 0.0, 0
    for k in range(len(poses)):
        t, c = _can_terms(data, can, poses, s, k, clip)
        tot, cnt = tot + t, cnt + c
    return tot / max(cnt, 1) / clip ** 2


class Scene:
    """Parameters relative to a start (b0, R0, s0) of the upright can: x[0:2] base shift on
    the surface (m), x[2:4] tilt of the axis (rad), x[4] log scale, then (x, y, yaw) per
    lying can (m, m, rad) on the surface."""

    def __init__(self, can, b0, R0, s0):
        self.can, self.b0, self.R0, self.s0 = can, b0, R0, s0

    def upright(self, x):
        s = self.s0 * math.exp(x[4])
        R = Rotation.from_rotvec(self.R0 @ np.array([x[2], x[3], 0.0])).as_matrix() @ self.R0
        b = self.b0 + self.s0 * (self.R0 @ np.array([x[0], x[1], 0.0]))
        return b, R, s

    def poses(self, x):
        b, R, s = self.upright(x)
        out = [(b, R)]
        for i in range(5, len(x), 3):
            out.append(lying_pose(b, R, s, self.can, x[i], x[i + 1], x[i + 2], False))
        return out, s

    def overlap(self, x):
        """True if two cans overlap (in the surface plane; all stand on the same surface)."""
        r, half = self.can.radius, self.can.height / 2
        segs = [(np.zeros(2), np.zeros(2))]
        for i in range(5, len(x), 3):
            c = np.array([x[i], x[i + 1]])
            u = np.array([math.cos(x[i + 2]), math.sin(x[i + 2])])
            segs.append((c - half * u, c + half * u))
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                if _seg_dist2d(*segs[a], *segs[b]) < 1.9 * r:
                    return True
        return False

    def cost(self, data, x, clip):
        if self.overlap(x):
            return 10.0
        poses, s = self.poses(x)
        return group_cost(data, self.can, poses, s, clip)


def _place_one(scene, coarse, fine, x, log, extent=0.2, step=0.012, yaws=15, keep=6):
    """Adds one lying can to the parameter vector x: grid, then the best few refined."""
    can = scene.can
    cands = []
    for cx in np.arange(-extent, extent + 1e-9, step):
        for cy in np.arange(-extent, extent + 1e-9, step):
            if math.hypot(cx, cy) < 1.9 * can.radius:
                continue
            for yaw in np.linspace(0, math.pi, yaws, endpoint=False):
                xx = np.concatenate([x, [cx, cy, yaw]])
                if scene.overlap(xx):
                    continue
                poses, s = scene.poses(xx)
                t, c = _can_terms(coarse, can, poses, s, len(poses) - 1, 0.2)
                cands.append((t / max(c, 1) / 0.04, cx, cy, yaw))
    if not cands:
        return None
    cands.sort()
    picks = []
    for c in cands:
        if all(math.hypot(c[1] - p[1], c[2] - p[2]) > 0.03 or abs(math.remainder(c[3] - p[3], math.pi)) > 0.4 for p in picks):
            picks.append(c)
        if len(picks) >= keep:
            break
    best = None
    for c0, cx, cy, yaw in picks:
        def f(p, clip):
            return scene.cost(fine, np.concatenate([x, p]), clip)

        p = np.array([cx, cy, yaw])
        for clip in (0.15, 0.08):
            p = minimize(lambda q: f(q, clip), p, method="Powell",
                         bounds=[(cx - 0.03, cx + 0.03), (cy - 0.03, cy + 0.03), (yaw - 0.35, yaw + 0.35)],
                         options={"xtol": 1e-4, "ftol": 1e-5, "maxfev": 400}).x
        cost = f(p, 0.08)
        log(f"  lying candidate {c0:.3f} at ({cx:.3f}, {cy:.3f}, {math.degrees(yaw):.0f} deg) -> {cost:.3f} at "
            f"({p[0]:.3f}, {p[1]:.3f}, {math.degrees(p[2]):.0f} deg)")
        if best is None or cost < best[0]:
            best = (cost, p)
    return np.concatenate([x, best[1]]), best[0]


def fit_cans(coarse, fine, can, b0, R0, s0, n_lying, log=print, max_tilt=0.12, max_shift=0.03, lying0=None,
             extra=None, extra_weight=1.0):
    """coarse, fine: can_fit.edge_data lists. (b0, R0, s0): a start for the upright can and
    the scale (e.g. the SfM-point group fit snapped to the outlines). lying0: (n, 3) starts
    (x, y, yaw) of the lying cans (else each is searched on a grid). extra(poses, s): a cost
    added to the outline cost times extra_weight (e.g. 1 - silhouette IoU with can masks),
    which keeps the refinement near a global start. Returns (cans [dict(kind, b, R)], s, info)."""
    scene = Scene(can, b0, R0, s0)
    if extra is not None:
        base_cost = scene.cost

        def cost(data, x, clip):
            c = base_cost(data, x, clip)
            if c >= 10.0:
                return c
            poses, s = scene.poses(x)
            return c + extra_weight * extra(poses, s)

        scene.cost = cost
    x = np.zeros(5) if lying0 is None else np.concatenate([np.zeros(5), np.asarray(lying0, float).ravel()])
    info = {"cost_start": round(scene.cost(fine, x, 0.08), 4)}
    for k in range(n_lying if lying0 is None else 0):
        log(f"outline fit: lying can {k + 1}")
        res = _place_one(scene, coarse, fine, x, log)
        if res is None:
            log("outline fit: no place left for another lying can")
            break
        x = res[0]
    info["cost_placed"] = round(scene.cost(fine, x, 0.08), 4)
    # everything together
    for i in range(5, len(x), 3):
        x[i + 2] = x[i + 2] % math.pi  # heading: a lying can looks the same turned by 180 deg
    lim = [(-max_shift, max_shift)] * 2 + [(-max_tilt, max_tilt)] * 2 + [(-0.25, 0.25)]
    for i in range(5, len(x), 3):
        lim += [(x[i] - 0.03, x[i] + 0.03), (x[i + 1] - 0.03, x[i + 1] + 0.03), (x[i + 2] - 0.3, x[i + 2] + 0.3)]
    x = np.clip(x, [l for l, _ in lim], [h for _, h in lim])
    for clip in (0.12, 0.08, 0.05):
        x0 = x
        x = minimize(lambda q: scene.cost(fine, q, clip), x, method="Powell", bounds=lim,
                     options={"xtol": 1e-4, "ftol": 1e-6, "maxfev": 3000}).x
        if scene.cost(fine, x, clip) > scene.cost(fine, x0, clip):  # never end worse (e.g. on overlapping cans)
            x = x0
    poses, s = scene.poses(x)
    info.update(cost_final=round(scene.cost(fine, x, 0.08), 4), scale_change=round(math.exp(x[4]), 4),
                upright_moved_m=round(math.hypot(x[0], x[1]), 4), axis_tilted_deg=round(math.degrees(math.hypot(x[2], x[3])), 2),
                lying=[{"xy_m": [round(x[i], 3), round(x[i + 1], 3)], "heading_deg": round(math.degrees(x[i + 2]) % 180, 1)}
                       for i in range(5, len(x), 3)])
    log(f"outline fit: {info}")
    cans = [{"kind": "upright" if k == 0 else "lying", "b": b, "R": R} for k, (b, R) in enumerate(poses)]
    return cans, s, info
