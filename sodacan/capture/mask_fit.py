"""Several identical cans posed by their silhouettes in many calibrated views.

Why: a chrome can mirrors the room, so the 3D points SfM finds on it are reflections at the
wrong depth and edges inside it are reflected edges. Its silhouette is neither. Known-model
pose from multi-view masks is what recent work uses for shiny objects (Accidental Light
Probes, Yu et al., CVPR 2023, mask + photometric loss on a known can; multi-view mask
alignment with one shared object pose, Shugurov et al. 2022; CAD alignment with SAM masks,
2025); a cylinder's silhouette has a closed form, so no renderer is needed: it is the convex
hull of its two end circles' projections (exact for a pinhole camera seeing both circles in
front of it).

The masks come from an open-vocabulary detector plus SAM (sodacan.capture.can_masks). Every
can in one photo is in one union mask: which can is which is never needed.

Parameters (SfM frame, its scale unknown): the surface's up direction (2 angles around the
cameras' mean up), log scale (SfM units per metre, from a 10-80 cm camera distance), the
upright can's base (3, around the point the cameras aim at), then (x, y, heading) per lying
can on the surface, in metres from the upright can (the convention of multi_can_fit.Scene).
Cost: 1 - mean over views of the IoU of the rendered union with the detected union, plus a
penalty when cans overlap; the worst views (a wrong mask, a camera SfM placed badly) are left
out of the mean (trimmed). Searched globally (differential evolution: the cost is flat
outside the masks and full of local minima from swapping cans), then polished.

Then, as multi-view silhouette methods reject inconsistent segmentations: every detected can
mask (one per instance) that the fitted scene does not explain in its view (the noodle cup
under the board, a tablet SAM spread into) is dropped, and the scene is searched again from
the first answer on the cleaned masks. Last, the views the scene still fits badly get their
own camera pose refined against their masks (a few degrees and centimetres: chrome and
reflections misplace some SfM cameras), and the scene is polished once more.
"""
import math

import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation

from sodacan.capture.can_fit import basis, look_at_point
from sodacan.capture.multi_can_fit import Scene

RING = 24


def small_masks(masks, K, long_side):
    """Downsized float masks and matching intrinsics."""
    h, w = masks[0].shape
    f = long_side / max(h, w)
    ws, hs = round(w * f), round(h * f)
    # area-averaged and thresholded at one half: "> 0" would fatten every mask by a pixel,
    # which at a few hundred pixels is a large part of a can
    out = np.stack([cv2.resize(m.astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA) >= 0.5 for m in masks])
    return out, K.scaled(ws, hs)


def _ring_pts(can):
    a = np.linspace(0, 2 * math.pi, RING, endpoint=False)
    c = np.stack([can.radius * np.cos(a), can.radius * np.sin(a)], 1)
    bot = np.concatenate([c, np.zeros((RING, 1))], 1)
    top = np.concatenate([c, np.full((RING, 1), can.height)], 1)
    return np.concatenate([bot, top])  # (2 RING, 3) metric, can frame


class MaskScene:
    def __init__(self, can, c2w, K, masks, n_lying, cams_up, centre, dist, trim=0.0):
        self.can, self.K, self.n, self.trim = can, K, n_lying, trim
        self.c2w = np.asarray(c2w, float)
        self.masks = masks  # (V, h, w) bool
        self.area = masks.reshape(len(masks), -1).sum(1)
        self.up0 = cams_up / np.linalg.norm(cams_up)
        self.e = basis(self.up0)[:, :2]  # two directions across the cameras' up
        self.centre, self.dist = centre, dist
        self.ring = _ring_pts(can)

    # x: [t1, t2, log s, bx, by, bz, (x, y, yaw) * n]
    def bounds(self, max_tilt_deg=35.0, near=0.10, far=0.80):
        t = math.radians(max_tilt_deg)
        ls = (math.log(self.dist / far), math.log(self.dist / near))
        b = [(-t, t), (-t, t), ls] + [(-0.35 * self.dist, 0.35 * self.dist)] * 3
        return b + [(-0.25, 0.25), (-0.25, 0.25), (0.0, math.pi)] * self.n

    def start(self, x):
        u = Rotation.from_rotvec(self.e @ np.array([x[0], x[1]])).apply(self.up0)
        R0 = basis(u, x_hint=self.e[:, 0])
        s = math.exp(x[2])
        b0 = self.centre + np.asarray(x[3:6])
        return b0, R0, s

    def scene(self, x):
        b0, R0, s = self.start(x)
        return Scene(self.can, b0, R0, s), np.concatenate([np.zeros(5), x[6:]])

    def poses(self, x):
        sc, xs = self.scene(x)
        return sc.poses(xs)

    def world(self, poses, s):
        return [b + s * (self.ring @ R.T) for b, R in poses]

    def render_one(self, world, M, s, each=False):
        """Silhouettes of the cans (world points from world()) seen by camera M: their union,
        or with each=True one mask per can."""
        h, w = self.masks.shape[1:]
        out = np.zeros((len(world) if each else 1, h, w), np.uint8)
        for k, P in enumerate(world):
            pc = (P - M[:3, 3]) @ M[:3, :3]
            if np.any(pc[:, 2] < 1e-3 * s):
                continue
            uv = np.stack([self.K.fx * pc[:, 0] / pc[:, 2] + self.K.cx, self.K.fy * pc[:, 1] / pc[:, 2] + self.K.cy], 1)
            if np.abs(uv).max() > 20 * max(h, w):
                continue
            hull = cv2.convexHull(np.round(uv * 4).astype(np.int32))
            cv2.fillConvexPoly(out[k if each else 0], hull, 1, lineType=cv2.LINE_8, shift=2)
        return out.astype(bool) if each else out[0].astype(bool)

    def render(self, poses, s, views=None):
        V = range(len(self.c2w)) if views is None else views
        world = self.world(poses, s)
        return np.stack([self.render_one(world, self.c2w[i], s) for i in V])

    def iou(self, x):
        poses, s = self.poses(x)
        r = self.render(poses, s)
        inter = (r & self.masks).reshape(len(r), -1).sum(1)
        union = (r | self.masks).reshape(len(r), -1).sum(1)
        return inter / np.maximum(union, 1)

    def __call__(self, x):
        sc, xs = self.scene(x)
        pen = 0.5 if sc.overlap(xs) else 0.0
        iou = self.iou(x)
        if self.trim > 0:  # the worst views left out: a wrong mask or a misplaced camera
            iou = np.sort(iou)[int(self.trim * len(iou)):]
        return 1.0 - float(iou.mean()) + pen


class _Recorder:
    """A map for differential_evolution's `workers`: evaluates in a process pool (or here)
    and keeps the best points with the scores they got."""

    def __init__(self, workers, keep=200):
        import multiprocessing as mp
        import os

        n = (os.cpu_count() or 1) if workers == -1 else int(workers)
        self.pool = mp.Pool(n) if n > 1 else None
        self.keep, self.xs, self.fs = keep, np.zeros((0, 0)), np.zeros(0)

    def __call__(self, func, iterable):
        xs = [np.asarray(x, float) for x in iterable]
        fs = list(self.pool.map(func, xs)) if self.pool else [func(x) for x in xs]
        X, F = np.array(xs), np.array(fs, float)
        if len(self.fs):
            X, F = np.vstack([self.xs, X]), np.concatenate([self.fs, F])
        o = np.argsort(F)[: self.keep]
        self.xs, self.fs = X[o], F[o]
        return fs

    def top(self):
        return self.xs

    def best_cost(self):
        return float(self.fs[0]) if len(self.fs) else float("nan")

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()


def _rng_kw(seed):
    import inspect

    return {"rng": seed} if "rng" in inspect.signature(differential_evolution).parameters else {"seed": seed}


def _search(ms, bnds, seed, workers, maxiter, popsize, log, init=None, tag=""):
    """Differential evolution; returns (reported x, the best points it scored)."""
    rec = _Recorder(workers)
    try:
        res = differential_evolution(ms, bnds, **_rng_kw(seed), popsize=popsize, maxiter=maxiter, tol=1e-5,
                                     mutation=(0.5, 1.0), recombination=0.7, polish=False, workers=rec,
                                     updating="deferred", init="latinhypercube" if init is None else init)
    finally:
        rec.close()
    log(f"mask fit{tag}: global search cost {res.fun:.4f} after {res.nfev} evaluations; best recorded {rec.best_cost():.4f}")
    return res.x, rec.top()


def _polish(mb, bnds, cands, log, n=3, tag=""):
    """The best few candidates (scored on mb) polished by Nelder-Mead from a small simplex
    (2 % of each range: Powell's long line searches jump across the overlap penalty); a
    polish that ends worse is not kept."""
    costs = np.array([mb(p) for p in cands])
    order = np.argsort(costs)
    width = np.array([hi - lo for lo, hi in bnds])
    best = None
    for i in order[:n]:
        x0 = np.asarray(cands[i], float)
        simplex = np.vstack([x0] + [x0 + 0.02 * width[k] * np.eye(len(x0))[k] for k in range(len(x0))])
        xi = minimize(mb, x0, method="Nelder-Mead", bounds=bnds,
                      options={"initial_simplex": simplex, "xatol": 1e-5, "fatol": 1e-5, "maxfev": 1500}).x
        ci = mb(xi)
        log(f"mask fit{tag}: candidate {costs[i]:.4f} -> polished {ci:.4f}")
        if ci > costs[i]:
            xi, ci = cands[i], costs[i]
        if best is None or ci < best[0]:
            best = (ci, xi)
    return best[1]


def _consistent(mb, x, inst, min_explained=0.4):
    """Per view, the union of the instance masks the fitted scene explains (at least
    min_explained of the mask inside the rendered cans, a little widened); a view where it
    explains none keeps them all. Returns (masks, kept, views of the dropped ones)."""
    poses, s = mb.poses(x)
    world = mb.world(poses, s)
    ker = np.ones((5, 5), np.uint8)
    out, kept, dropped = [], 0, []
    for i, iv in enumerate(inst):
        r = cv2.dilate(mb.render_one(world, mb.c2w[i], s).astype(np.uint8), ker).astype(bool)
        ok = [m.sum() > 0 and (m & r).sum() >= min_explained * m.sum() for m in iv]
        u = np.zeros(mb.masks.shape[1:], bool)
        for m, o in zip(iv, ok):
            # dropped only beside a mask the scene does explain: where it explains none, the
            # camera (or the first fit) is off in that view, not the masks
            if o or not any(ok):
                u |= m
                kept += 1
            else:
                dropped.append(i)
        out.append(u)
    return np.stack(out), kept, dropped


def refine_camera(mb, x, i, max_rot_deg=12.0, max_shift=0.2, seed=0):
    """Camera i's pose refined against its own mask with the scene fixed: a rotation about
    the camera centre (rotation vector, +- max_rot_deg) and a shift (+- max_shift of the
    camera distance). Returns (c2w, iou before, iou after)."""
    poses, s = mb.poses(x)
    world = mb.world(poses, s)
    M0 = mb.c2w[i].copy()
    target = mb.masks[i]

    def cam(p):
        M = M0.copy()
        M[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix() @ M0[:3, :3]
        M[:3, 3] = M0[:3, 3] + np.asarray(p[3:]) * mb.dist
        return M

    def cost(p):
        r = mb.render_one(world, cam(p), s)
        return 1.0 - (r & target).sum() / max(1, (r | target).sum())

    t, d = math.radians(max_rot_deg), max_shift
    bnds = [(-t, t)] * 3 + [(-d, d)] * 3
    c0 = cost(np.zeros(6))
    res = differential_evolution(cost, bnds, popsize=12, maxiter=60, tol=1e-6, polish=False, **_rng_kw(seed))
    p = minimize(cost, res.x, method="Nelder-Mead", bounds=bnds, options={"xatol": 1e-6, "fatol": 1e-6, "maxfev": 600}).x
    c1 = cost(p)
    if c1 >= c0:
        return M0, 1 - c0, 1 - c0
    return cam(p), 1 - c0, 1 - c1


def fit_masks(masks, c2w, K, can, n_lying, cams_up, log=print, long_side=320, seed=0, workers=1, maxiter=400, popsize=25,
              instances=None, trim=0.15, refine_below=0.75, min_gain=0.05):
    """masks: per view HxW bool union masks of the cans, at the size K describes; instances:
    optional per view (N, H, W) masks of each detected can (for dropping the ones the scene
    does not explain). Returns dict(b0, R0, s, lying (n, 3) [x, y, yaw], info, x, scene,
    views, c2w (every input camera; those refined against their masks replaced)) or None."""
    small, Ks = small_masks(masks, K, long_side)
    have = small.reshape(len(small), -1).any(1)
    if have.sum() < 4:
        log(f"mask fit: cans found in only {int(have.sum())} photos")
        return None
    c2w_all = np.asarray(c2w, float)
    c2w = c2w_all[have]
    small = small[have]
    centre = look_at_point(c2w)
    dist = float(np.median(np.linalg.norm(c2w[:, :3, 3] - centre, axis=1)))
    ms = MaskScene(can, c2w, Ks, small, n_lying, cams_up, centre, dist, trim=trim)
    bnds = ms.bounds()
    log(f"mask fit: {len(small)} views with cans, {len(bnds)} parameters, cameras {dist:.3f} SfM units from their aim point")
    x_rep, pop = _search(ms, bnds, seed, workers, maxiter, popsize, log)
    big, Kb = small_masks(masks, K, 2 * long_side)
    mb = MaskScene(can, c2w.copy(), Kb, big[have], n_lying, cams_up, centre, dist, trim=trim)
    x = _polish(mb, bnds, np.vstack([x_rep[None], pop]), log)
    info = {"round1_iou_mean": round(float(mb.iou(x).mean()), 4)}

    # 2. drop the detected masks the scene does not explain; search again from this answer
    if instances is not None:
        inst = [iv for iv, h_ in zip(instances, have) if h_]
        inst_b = [small_masks(iv, K, 2 * long_side)[0] if len(iv) else np.zeros((0,) + mb.masks.shape[1:], bool) for iv in inst]
        new_b, kept, dropped = _consistent(mb, x, inst_b)
        log(f"mask fit: {kept} can masks explained by the fitted scene, {len(dropped)} dropped (views {sorted(set(dropped))})")
        info["masks_dropped"] = len(dropped)
        if dropped:
            hs, ws = ms.masks.shape[1:]
            new_s = np.stack([cv2.resize(m.astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA) >= 0.5 for m in new_b])
            ms.masks, ms.area = new_s, new_s.reshape(len(new_s), -1).sum(1)
            mb.masks, mb.area = new_b, new_b.reshape(len(new_b), -1).sum(1)
            rng = np.random.default_rng(seed + 1)
            width = np.array([hi - lo for lo, hi in bnds])
            lo, hi = np.array([b[0] for b in bnds]), np.array([b[1] for b in bnds])
            n_pop = popsize * len(bnds)
            init = np.clip(x + rng.normal(0, 0.05, (n_pop, len(x))) * width, lo, hi)
            init[0] = x
            init[1:n_pop // 4] = lo + rng.random((n_pop // 4 - 1, len(x))) * width  # some of it global still
            x_rep, pop = _search(ms, bnds, seed + 1, workers, max(60, maxiter // 3), popsize, log, init=init, tag=" (cleaned masks)")
            x = _polish(mb, bnds, np.vstack([x[None], x_rep[None], pop]), log, tag=" (cleaned masks)")

    # 3. the views still fitted badly: their cameras refined against their own masks
    iou = mb.iou(x)
    refined = []
    for i in np.flatnonzero(iou < refine_below):
        M, a, b_ = refine_camera(mb, x, i, seed=seed)
        if b_ >= a + min_gain:
            mb.c2w[i] = M
            refined.append({"view": int(np.flatnonzero(have)[i]), "iou": [round(float(a), 3), round(float(b_), 3)]})
    log(f"mask fit: cameras refined against their masks: {refined}")
    if refined:
        x = _polish(mb, bnds, x[None], log, n=1, tag=" (refined cameras)")
    iou = mb.iou(x)
    b0, R0, s = mb.start(x)
    tilt = math.degrees(math.acos(min(1.0, float(R0[:, 2] @ ms.up0))))
    info.update({"iou_mean": round(float(iou.mean()), 4), "iou_min": round(float(iou.min()), 4),
                 "iou_per_view": [round(float(v), 3) for v in iou], "cameras_refined": refined,
                 "views": int(len(iou)), "camera_distance_m": round(dist / s, 3), "up_vs_cameras_deg": round(tilt, 1),
                 "lying": [{"xy_m": [round(x[i], 3), round(x[i + 1], 3)], "heading_deg": round(math.degrees(x[i + 2]) % 180, 1)}
                           for i in range(6, len(x), 3)]})
    log(f"mask fit: {info}")
    c2w_out = c2w_all.copy()
    c2w_out[np.flatnonzero(have)] = mb.c2w
    mb.trim = 0.0
    return {"b0": b0, "R0": R0, "s": s, "lying": np.asarray(x[6:]).reshape(-1, 3), "info": info, "x": x,
            "scene": mb, "views": np.flatnonzero(have), "c2w": c2w_out}


def mask_cost(scene, poses, s):
    """1 - mean IoU of given poses (for the joint edge + mask refinement)."""
    r = scene.render(poses, s)
    inter = (r & scene.masks).reshape(len(r), -1).sum(1)
    union = (r | scene.masks).reshape(len(r), -1).sum(1)
    return 1.0 - float((inter / np.maximum(union, 1)).mean())
