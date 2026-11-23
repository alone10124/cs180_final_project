"""One upright can and n cans lying on their side, all the same can, on one surface: fitted
together to the SfM points, with reflections in mind.

A chrome can is a convex mirror. Whatever it reflects (the room, and above all a neighbouring
can) triangulates to a virtual point *inside* it, between its surface and about half its
radius in. So the SfM cloud of a can is a filled disk, not a ring, and a neighbour's
reflection makes the disk fuller. What never happens is a point just *outside* the can: in
the air around it there is nothing to see. So each can is scored by

    (points on its surface shell) - alpha * (points in a band just outside it)

and points inside it are neither for nor against. That outside band is what fixes the
radius (the scale of the scene) and the centre, which the points inside would otherwise pull
in. Points explained by one can (on or inside it) are not counted for or against another.

The scene: an "up" direction (the camera's up, or the normal of a plane of points, or near
either), the upright can's centre, radius (scale) and height of its base along up from a grid
search over the label band; for its few best, the lying cans (centre, heading) on the surface
at that base height, with their axes one radius above it, from a grid search that never lets
two cans overlap. The best total wins. Returns poses in the can_fit conventions (b: centre of
the bottom, R: z column the axis).
"""
import math

import numpy as np

from sodacan.capture.can_fit import _cam_up, basis, look_at_point, lying_pose, plane_ransac

SHELL = 0.07     # surface shell, +- fraction of the radius
BAND = (1.10, 1.45)  # the empty band outside, in radii
ALPHA = 1.0      # weight of a point in the empty band against one on the shell


def _upright_scores(X, r_list, centres, z_bins, dz, lab, top, band_from, alpha=ALPHA, chunk=1500):
    """For each radius and centre (in the up frame X: x, y across, z along up), and every
    height of the can's base: points on the shell over the label band lab = (lo, hi), minus
    alpha * points in the empty band outside from band_from to top (all in radii above the
    base). The band starts above the lying cans (2 radii high), which touch the upright can
    and would otherwise count against its true size. Returns arrays (score, base_z) of shape
    (len(r_list), len(centres))."""
    nz = len(z_bins) - 1
    zi = np.clip(np.digitize(X[:, 2], z_bins) - 1, 0, nz - 1)
    zin = (X[:, 2] >= z_bins[0]) & (X[:, 2] < z_bins[-1])
    score = np.full((len(r_list), len(centres)), -np.inf)
    zbase = np.zeros_like(score)
    for ri, r in enumerate(r_list):
        o1, o2 = int(round(lab[0] * r / dz)), int(round(lab[1] * r / dz))
        b1, b2 = int(round(band_from * r / dz)), int(round(top * r / dz))
        nb = nz - b2  # base bins for which every window fits
        if nb <= 0:
            continue
        for c0 in range(0, len(centres), chunk):
            C = centres[c0:c0 + chunk]
            rho = np.hypot(X[None, :, 0] - C[:, None, 0], X[None, :, 1] - C[:, None, 1])
            idx = (np.arange(len(C))[:, None] * nz + zi[None]).ravel()

            def cum(w):
                h = np.bincount(idx, weights=(w * zin[None]).ravel(), minlength=len(C) * nz).reshape(len(C), nz)
                return np.concatenate([np.zeros((len(C), 1)), np.cumsum(h, 1)], 1)

            cs = cum((np.abs(rho - r) < SHELL * r).astype(np.float32))
            cb = cum(((rho > BAND[0] * r) & (rho < BAND[1] * r)).astype(np.float32))
            k = np.arange(nb)
            win = (cs[:, k + o2] - cs[:, k + o1]) - alpha * (cb[:, k + b2] - cb[:, k + b1])
            j = win.argmax(1)
            score[ri, c0:c0 + chunk] = win[np.arange(len(C)), j]
            zbase[ri, c0:c0 + chunk] = z_bins[j]
    return score, zbase


def _seg_dist2d(p1, q1, p2, q2):
    """Shortest distance between 2D segments p1q1 and p2q2."""
    def pt_seg(p, a, b):
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        return np.linalg.norm(p - (a + t * ab))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(p2, q2, p1), cross(p2, q2, q1)
    d3, d4 = cross(p1, q1, p2), cross(p1, q1, q2)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return 0.0
    return min(pt_seg(p1, p2, q2), pt_seg(q1, p2, q2), pt_seg(p2, p1, q1), pt_seg(q2, p1, q1))


def _lying_search(X, r, half, extent, step, taken, used, alpha=ALPHA, yaw_step_deg=4.0):
    """The best lying can (centre (x, y), yaw) on the surface z = 0 (X in the up frame with
    the surface at z = 0), axis at height r, length 2 half; not overlapping the cans in taken
    (list of ('up', centre) or ('lying', p, q) in 2D) and not counting points in used.
    Returns (score, x, y, yaw, shell_mask) or None."""
    keep = ~used & (X[:, 2] > 0.15 * r) & (X[:, 2] < 2.6 * r)  # not the surface itself
    P = X[keep]
    zc = P[:, 2] - r
    grid = np.arange(-extent, extent + 1e-9, step)
    best = None
    for yaw in np.radians(np.arange(0, 180, yaw_step_deg)):
        e = np.array([math.cos(yaw), math.sin(yaw)])
        f = np.array([-e[1], e[0]])
        a, b = P[:, :2] @ e, P[:, :2] @ f
        a_bins = np.arange(-extent - half - step, extent + half + 2 * step, step)
        ai = np.clip(np.digitize(a, a_bins) - 1, 0, len(a_bins) - 2)
        L = max(1, int(round(2 * half / step)))
        for cb in grid:
            rho = np.hypot(b - cb, zc)
            w = (np.abs(rho - r) < SHELL * r).astype(np.float32) - alpha * ((rho > BAND[0] * r) & (rho < BAND[1] * r))
            if not w.any():
                continue
            hist = np.bincount(ai, weights=w, minlength=len(a_bins) - 1)
            cs = np.concatenate([[0.0], np.cumsum(hist)])
            win = cs[L:] - cs[:-L]          # window starting at bin k covers a in [a_bins[k], a_bins[k + L])
            centres_a = a_bins[:len(win)] + half
            ok = np.abs(centres_a) <= extent
            if not ok.any():
                continue
            win = np.where(ok, win, -np.inf)
            for k in np.argsort(win)[::-1][:3]:  # a few best along the axis, the first that fits
                if not np.isfinite(win[k]) or (best is not None and win[k] <= best[0]):
                    break
                c = centres_a[k] * e + cb * f
                p, q = c - half * e, c + half * e
                clash = False
                for t in taken:
                    if t[0] == "up":
                        clash |= _seg_dist2d(p, q, t[1], t[1]) < 1.95 * r
                    else:
                        clash |= _seg_dist2d(p, q, t[1], t[2]) < 1.95 * r
                if not clash:
                    best = (float(win[k]), float(c[0]), float(c[1]), float(yaw))
                    break
    if best is None:
        return None
    _, x, y, yaw = best
    e = np.array([math.cos(yaw), math.sin(yaw)])
    f = np.array([-e[1], e[0]])
    a = (X[:, :2] - [x, y]) @ e
    rho = np.hypot((X[:, :2] - [x, y]) @ f, X[:, 2] - r)
    inside = (np.abs(a) < half * 1.05) & (rho < (1 + SHELL) * r)
    shell = inside & (rho > (1 - SHELL) * r) & ~used
    return best + (inside, shell)


def _directions(points_near, up0, d, log):
    """Candidate up directions: the cameras' up and the normals of the planes of points near
    the aim point (within 35 deg of it)."""
    dirs = [("camera up", up0)]
    pool = points_near
    for k in range(3):
        pl = plane_ransac(pool, up0, 35.0, 0.004 * d, iters=2000, seed=k)
        if pl is None or pl[2].sum() < 40:
            break
        n, q, m = pl
        if all(math.degrees(math.acos(min(1.0, abs(float(n @ v))))) > 2.0 for _, v in dirs):
            dirs.append((f"plane {k} ({int(m.sum())} points)", n))
        pool = pool[~m]
    return dirs


def _cone(n, deg, step):
    out = [n]
    R = basis(n)
    for t in np.arange(step, deg + 1e-9, step):
        for az in np.linspace(0, 2 * math.pi, max(4, int(round(2 * math.pi * t / step))), endpoint=False):
            tt = math.radians(t)
            out.append(R @ np.array([math.sin(tt) * math.cos(az), math.sin(tt) * math.sin(az), math.cos(tt)]))
    return out


def fit_group(points, c2w_all, can, n_lying, log=print, max_points=4000, seed=0):
    """Returns dict(cans=[dict(kind, b, R, info)], s, up, info) or None."""
    rng = np.random.default_rng(seed)
    up0 = _cam_up(c2w_all)
    p0 = look_at_point(c2w_all)
    C = np.array([M[:3, 3] for M in c2w_all])
    d = float(np.median(np.linalg.norm(C - p0, axis=1)))
    near = points[np.linalg.norm(points - p0, axis=1) < 0.6 * d]
    if len(near) < 50:
        return None
    if len(near) > max_points:
        near = near[rng.choice(len(near), max_points, replace=False)]
    hr = can.height / can.radius                       # can length in radii
    lab0, lab1 = can.usable_z[0] / can.radius, can.usable_z[1] / can.radius
    dz = 0.006 * d
    z_bins = np.arange(-0.6 * d, 0.6 * d + dz, dz)
    sub = near[rng.choice(len(near), min(len(near), 1500), replace=False)]
    grids = {}
    for coarse, st, nr in ((True, 0.02 * d, 10), (False, 0.01 * d, 18)):
        g = np.arange(-0.3 * d, 0.3 * d + 1e-9, st)
        grids[coarse] = (st, np.stack(np.meshgrid(g, g), -1).reshape(-1, 2), d * np.geomspace(0.045, 0.22, nr))

    def evaluate(n, coarse, r_list=None):
        """Best arrangement for the up direction n: per radius (scale), the best upright can,
        then the lying cans around it; scored together, the upright's empty band leaving
        out the points the lying cans explain."""
        R = basis(n)
        pts = sub if coarse else near
        X = (pts - p0) @ R
        step, centres, r_def = grids[coarse]
        r_list = r_def if r_list is None else r_list
        sc, zs = _upright_scores(X, r_list, centres, z_bins, dz, (lab0, lab1), hr, 2.15)
        best = None
        for ri, r in enumerate(r_list):
            ci = int(np.argmax(sc[ri]))
            if not np.isfinite(sc[ri, ci]):
                continue
            c, z0 = centres[ci], float(zs[ri, ci])
            Xs = X - np.array([c[0], c[1], z0])        # surface frame, upright axis at the origin
            rho = np.hypot(Xs[:, 0], Xs[:, 1])
            used = (rho < (1 + SHELL) * r) & (Xs[:, 2] > -0.2 * r) & (Xs[:, 2] < hr * r * 1.05)
            taken = [("up", np.zeros(2))]
            lying, lying_inside = [], np.zeros(len(X), bool)
            for _ in range(n_lying):
                lc = _lying_search(Xs, r, hr * r / 2, 0.35 * d, step, taken, used, yaw_step_deg=6.0 if coarse else 3.0)
                if lc is None:
                    break
                sc_l, x, y, yaw, inside, shell = lc
                e = np.array([math.cos(yaw), math.sin(yaw)])
                taken.append(("lying", np.array([x, y]) - hr * r / 2 * e, np.array([x, y]) + hr * r / 2 * e))
                used = used | inside
                lying_inside |= inside
                lying.append({"score": sc_l, "x": x, "y": y, "yaw": yaw, "shell_points": int(shell.sum())})
            zr = Xs[:, 2] / r
            shell_up = int(((np.abs(rho - r) < SHELL * r) & (zr > lab0) & (zr < lab1)).sum())
            band_up = int(((rho > BAND[0] * r) & (rho < BAND[1] * r) & (zr > lab0) & (zr < hr) & ~lying_inside).sum())
            up_score = shell_up - ALPHA * band_up
            total = up_score + sum(lc["score"] for lc in lying)
            if best is None or total > best["total"]:
                best = {"total": total, "upright_score": up_score, "upright_shell": shell_up, "upright_band": band_up,
                        "n": n, "R": R, "c": c, "r": float(r), "z0": z0, "lying": lying}
        return best

    dirs = _directions(near, up0, d, log)
    trials = []
    for name, n in dirs:  # each candidate up direction and a cone around it, coarse
        for m in _cone(n, 6.0, 3.0):
            res = evaluate(m, coarse=True)
            if res is not None:
                res["from"] = name
                trials.append(res)
    if not trials:
        return None
    trials.sort(key=lambda t: -t["total"])
    # the best coarse direction, fine, over radii around its coarse scale
    t = trials[0]
    log(f"group fit coarse: best of {len(trials)}: up from {t['from']}, total {t['total']:.0f}, radius {t['r'] / d:.3f} d, "
        f"{len(t['lying'])} lying; next {[round(x['total']) for x in trials[1:5]]}")
    best = evaluate(t["n"], coarse=False, r_list=t["r"] * np.geomspace(0.8, 1.25, 9))
    best["from"] = t["from"]
    if len(best["lying"]) < n_lying:
        log(f"group fit: found {len(best['lying'])} of {n_lying} lying cans")

    # poses in the SfM frame
    R = best["R"]
    r = best["r"]
    s = r / can.radius
    up = R[:, 2]
    b_up = p0 + R @ np.array([best["c"][0], best["c"][1], best["z0"]])
    R_up = basis(up, x_hint=R[:, 0])
    tilt_cam = math.degrees(math.acos(min(1.0, float(up @ up0))))
    info = {"method": "group fit (shell minus empty band)", "up_from": best["from"], "up_vs_camera_up_deg": round(tilt_cam, 2),
            "cam_dist_m": round(d / s, 3), "total_score": round(best["total"], 1), "points_used": int(len(near))}
    cans = [{"kind": "upright", "b": b_up, "R": R_up,
             "info": {"score": round(best["upright_score"], 1), "shell_points": best["upright_shell"],
                      "band_points": best["upright_band"]}}]
    for lc in best["lying"]:
        # (x, y) in metres in R_up's frame (same x axis as R)
        bl, Rl = lying_pose(b_up, R_up, s, can, lc["x"] / s, lc["y"] / s, lc["yaw"], False)
        cans.append({"kind": "lying", "b": bl, "R": Rl,
                     "info": {"score": round(lc["score"], 1), "shell_points": lc["shell_points"],
                              "xy_m": [round(lc["x"] / s, 3), round(lc["y"] / s, 3)], "heading_deg": round(math.degrees(lc["yaw"]), 1)}})
    log(f"group fit: {info}; " + "; ".join(f"{c['kind']} {c['info']}" for c in cans))
    return {"cans": cans, "s": s, "up": up, "info": info}


def place_lying(points, c2w_all, can, b_up, R_up, s, n_lying, log=print, max_points=4000, seed=0):
    """The lying cans again, around an upright can whose base b_up, axes R_up and scale s are
    settled (e.g. by the outline snap): the same reflection-aware search at that scale.
    Returns [dict(kind, b, R, info)]."""
    rng = np.random.default_rng(seed)
    p0 = look_at_point(c2w_all)
    C = np.array([M[:3, 3] for M in c2w_all])
    d = float(np.median(np.linalg.norm(C - p0, axis=1)))
    near = points[np.linalg.norm(points - p0, axis=1) < 0.6 * d]
    if len(near) > max_points:
        near = near[rng.choice(len(near), max_points, replace=False)]
    r = s * can.radius
    hr = can.height / can.radius
    Xs = (near - b_up) @ R_up
    rho = np.hypot(Xs[:, 0], Xs[:, 1])
    used = (rho < (1 + SHELL) * r) & (Xs[:, 2] > -0.2 * r) & (Xs[:, 2] < hr * r * 1.05)
    taken = [("up", np.zeros(2))]
    out = []
    for _ in range(n_lying):
        lc = _lying_search(Xs, r, hr * r / 2, 0.35 * d, 0.008 * d, taken, used, yaw_step_deg=2.0)
        if lc is None:
            break
        sc_l, x, y, yaw, inside, shell = lc
        e = np.array([math.cos(yaw), math.sin(yaw)])
        taken.append(("lying", np.array([x, y]) - hr * r / 2 * e, np.array([x, y]) + hr * r / 2 * e))
        used = used | inside
        bl, Rl = lying_pose(b_up, R_up, s, can, x / s, y / s, yaw, False)
        out.append({"kind": "lying", "b": bl, "R": Rl,
                    "info": {"score": round(sc_l, 1), "shell_points": int(shell.sum()), "xy_m": [round(x / s, 3), round(y / s, 3)],
                             "heading_deg": round(math.degrees(yaw), 1), "placed_at": "snapped scale"}})
    log("group fit, lying cans at the snapped scale: " + "; ".join(str(c["info"]) for c in out))
    return out
