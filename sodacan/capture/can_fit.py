"""Cans located in a structure-from-motion frame, with no board at all.

SfM gives every photo's pose up to an unknown similarity: no metric scale and no "up".
A can of known size fixes both. Its outline (two limb lines plus the front of the rim and
base rings), projected into many views, must land on image edges; only the right 3D
position, axis direction and scale do that in every view at once.

A can pose in the SfM frame is (b, R, s): base centre b, rotation R whose z column is the
can axis, and s = SfM units per metre. A point P of the can (metres, can frame) sits at
b + s R P. The camera seen from the can is then a metric pose in the can frame, so the
rest of the pipeline (pose_silhouette.outline_points, reflect.can_geometry) is unchanged.

Upright can first: it gives the scale, "up" (its axis) and the height of the surface the
cans rest on. Cans lying on their side are then found on that surface: centre one can
radius above it, axis horizontal, position and heading by grid search, then refined freely.
"""
import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from sodacan.capture.pose_silhouette import edge_map, outline_points, project_np, sample_dist


def basis(a, x_hint=None):
    """Rotation whose z column is the unit vector a; x column along x_hint (made orthogonal)."""
    a = a / np.linalg.norm(a)
    h = np.array([1.0, 0, 0]) if x_hint is None else np.asarray(x_hint, float)
    if abs(h @ a) > 0.9:
        h = np.array([0.0, 1, 0])
    x = h - (h @ a) * a
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(a, x), a], 1)


def camera_in_can(c2w, b, R, s):
    """Metric camera pose in the can frame, from an SfM-frame camera pose."""
    out = np.eye(4)
    out[:3, :3] = R.T @ c2w[:3, :3]
    out[:3, 3] = R.T @ (c2w[:3, 3] - b) / s
    return out


def edge_data(frames_rgb, c2w, K, count=12, long_side=None):
    """[(edge distance map, SfM c2w, intrinsics)] for count evenly spread photos, optionally
    downsized (a coarse search on small images has a wider basin per pixel of clip)."""
    import cv2

    idx = np.linspace(0, len(frames_rgb) - 1, min(count, len(frames_rgb))).round().astype(int)
    out = []
    for i in idx:
        im, Ki = frames_rgb[i], K
        if long_side and max(im.shape[:2]) > long_side:
            f = long_side / max(im.shape[:2])
            w, h = round(im.shape[1] * f), round(im.shape[0] * f)
            im, Ki = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), K.scaled(w, h)
        out.append((edge_map(im)[1], np.asarray(c2w[i], float), Ki))
    return out


def view_residuals(dist, cam, K, can, relative=False):
    """Outline-to-edge distances (ring points down-weighted) for a metric camera pose in the
    can frame: in px, or (relative) in units of the can's projected radius, so a can shrunk
    to a few pixels (all its outline on one edge) cannot score well. Points off the image or
    behind the camera get inf (clipped by the caller)."""
    P, w = outline_points(cam[:3, 3], can)
    if len(P) == 0:
        return None
    uv, z = project_np(P, cam, K)
    r = sample_dist(dist, uv) * w
    inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= K.width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= K.height - 1)
    r[~inside] = np.inf
    if relative:
        mid = cam[:3, :3].T @ (np.array([0.0, 0.0, can.height / 2]) - cam[:3, 3])
        if mid[2] <= 0:
            return np.full(len(r), np.inf)
        r = r / (K.fx * can.radius / mid[2])
    return r


def fit_cost(data, can, b, R, s, clip=0.1):
    """Mean squared clipped outline residual over the views, in units of the projected can radius."""
    res = []
    for dist, c2w, K in data:
        r = view_residuals(dist, camera_in_can(c2w, b, R, s), K, can, relative=True)
        res.append(np.full(60, clip) if r is None else np.minimum(r, clip))
    return float(np.mean(np.concatenate(res) ** 2)) / clip ** 2


def median_outline_px(data, can, b, R, s):
    out = []
    for dist, c2w, K in data:
        r = view_residuals(dist, camera_in_can(c2w, b, R, s), K, can)
        if r is not None:
            out.append(float(np.median(np.minimum(r, 25.0))))
    return out


def look_at_point(c2w):
    """Least-squares meeting point of the cameras' optical axes (where the photographer aimed)."""
    A, rhs = np.zeros((3, 3)), np.zeros(3)
    for M in c2w:
        z = M[:3, 2] / np.linalg.norm(M[:3, 2])
        P = np.eye(3) - np.outer(z, z)
        A += P
        rhs += P @ M[:3, 3]
    return np.linalg.solve(A, rhs)


def ground_up(points, c2w_all, iters=2000, seed=0):
    """"Up" from the SfM points: the normal of the largest plane near where the cameras aim
    (table, board), within 40 deg of the cameras' mean -y axis, pointing the same way.
    Falls back to the cameras' mean -y. Returns (unit up, inlier count)."""
    cam_up = -np.mean([M[:3, 1] for M in c2w_all], 0)
    cam_up /= np.linalg.norm(cam_up)
    if points is None or len(points) < 50:
        return cam_up, 0
    p0 = look_at_point(c2w_all)
    d = float(np.median(np.linalg.norm(np.array([M[:3, 3] for M in c2w_all]) - p0, axis=1)))
    P = points[np.linalg.norm(points - p0, axis=1) < 2.0 * d]
    if len(P) < 50:
        return cam_up, 0
    rng = np.random.default_rng(seed)
    thr = 0.01 * d
    best_n, best_in = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if n @ cam_up < 0:
            n = -n
        if n @ cam_up < math.cos(math.radians(40)):
            continue
        cnt = int((np.abs((P - a) @ n) < thr).sum())
        if cnt > best_in:
            best_n, best_in, best_pt = n, cnt, a
    if best_n is None:
        return cam_up, 0
    # least-squares normal of the inliers
    Q = P[np.abs((P - best_pt) @ best_n) < thr]
    n = np.linalg.svd(Q - Q.mean(0))[2][2]
    return (n if n @ best_n > 0 else -n), best_in


def _refine(data, can, b, R, s, free_scale=True, clips=(0.1, 0.05)):
    """Powell over base centre (metres along R's axes, within 10 cm), axis tilt (within 17 deg)
    and log scale (within x0.74 .. x1.35) around the start."""
    def unpack(x):
        Rn = Rotation.from_rotvec(R @ np.array([x[3], x[4], 0.0])).as_matrix() @ R
        sn = s * math.exp(x[5]) if free_scale else s
        return b + s * (R @ x[:3]), Rn, sn

    bounds = [(-0.1, 0.1)] * 3 + [(-0.3, 0.3)] * 2 + [(-0.3, 0.3) if free_scale else (-1e-9, 1e-9)]
    x = np.zeros(6)
    for clip in clips:
        f = lambda x: fit_cost(data, can, *unpack(x), clip=clip)
        x = minimize(f, x, method="Powell", bounds=bounds, options={"xtol": 1e-4, "ftol": 1e-5, "maxfev": 1500}).x
    return unpack(x)


def _distinct(cands, n, min_sep=0.03):
    """The n best (cost, b, s, ...) candidates, skipping ones within min_sep metres of a
    kept one at a similar scale."""
    kept = []
    for c in cands:
        if all(np.linalg.norm(c[1] - k[1]) / c[2] > min_sep or abs(math.log(c[2] / k[2])) > 0.1 for k in kept):
            kept.append(c)
        if len(kept) == n:
            break
    return kept


def find_upright(coarse, fine, c2w_all, can, up, log=print, scales=25, lateral=0.08, vertical=0.12, step=0.015):
    """The upright can: scale, base centre and axis in the SfM frame. Grid (on the coarse,
    downsized edge maps) over scale and position around the cameras' aim point, axis along
    up; the best few are refined on the full-size maps with the axis free.
    Returns dict(b, R, s, cost, tilt_deg)."""
    C = np.array([M[:3, 3] for M in c2w_all])
    R0 = basis(up)
    p0 = look_at_point(c2w_all)
    d_sfm = float(np.median(np.linalg.norm(C - p0, axis=1)))
    s0 = d_sfm / 0.45  # a guess of 45 cm from camera to can; the grid spans x0.25 .. x4 of it
    offs = lambda r: np.arange(-r, r + 1e-9, step)
    cands = []
    for s in s0 * 2.0 ** np.linspace(-2, 2, scales):
        for dz in offs(vertical):
            for dx in offs(lateral):
                for dy in offs(lateral):
                    b = p0 + s * (R0 @ np.array([dx, dy, dz - can.height / 2]))
                    cands.append((fit_cost(coarse, can, b, R0, s, clip=0.2), b, s))
    cands.sort(key=lambda c: c[0])
    log(f"upright can: {len(cands)} grid candidates; best cost {cands[0][0]:.2f} at {cands[0][2] / s0:.2f} x the guessed scale")
    best = None
    for c, b, s in _distinct(cands, 4):
        b1, R1, s1 = _refine(coarse, can, b, R0, s, clips=(0.2,))
        bn, Rn, sn = _refine(fine, can, b1, R1, s1)
        cost = fit_cost(fine, can, bn, Rn, sn, clip=0.05)
        tilt = math.degrees(math.acos(min(1.0, abs(float(Rn[:, 2] @ up)))))
        log(f"  start {c:.2f} -> refined {cost:.3f} (scale {sn / s0:.3f} x guess, axis {tilt:.1f} deg from up)")
        if best is None or cost < best["cost"]:
            best = {"b": bn, "R": Rn, "s": sn, "cost": cost, "tilt_deg": tilt}
    return best


def lying_pose(b_up, R_up, s, can, x, y, yaw, flip):
    """Base centre and rotation of a can lying on the surface the upright can stands on;
    (x, y) is its centre on that surface in metres from the upright can's base."""
    up = R_up[:, 2]
    e1, e2 = R_up[:, 0], R_up[:, 1]
    d = math.cos(yaw) * e1 + math.sin(yaw) * e2
    d = -d if flip else d
    centre = b_up + s * (x * e1 + y * e2 + can.radius * up)
    R = basis(d, x_hint=up)
    return centre - s * (can.height / 2) * d, R


def surface_xy(b, R, b_up, R_up, s, can):
    """(x, y, height above the surface, axis angle to the surface in deg) of a can, metres."""
    c = (b + s * (can.height / 2) * R[:, 2] - b_up) / s
    x, y, z = R_up.T @ c
    tilt = math.degrees(math.asin(min(1.0, abs(float(R[:, 2] @ R_up[:, 2])))))
    return float(x), float(y), float(z), tilt


def find_lying(coarse, fine, can, b_up, R_up, s, avoid, log=print, extent=0.20, step=0.015, yaws=12):
    """One can lying on its side: grid (coarse maps) over its centre on the surface, heading
    and which end is which, away from the (x, y) centres in avoid; the best few refined on
    the full-size maps with all of position and axis free (scale fixed), so its height above
    the surface and its axis tilt are a check. Returns dict(b, R, cost, x, y, z, tilt_deg)."""
    cands = []
    for x in np.arange(-extent, extent + 1e-9, step):
        for y in np.arange(-extent, extent + 1e-9, step):
            if any(math.hypot(x - ax, y - ay) < 0.06 for ax, ay in avoid):
                continue
            for yaw in np.linspace(0, math.pi, yaws, endpoint=False):
                for flip in (False, True):
                    b, R = lying_pose(b_up, R_up, s, can, x, y, yaw, flip)
                    cands.append((fit_cost(coarse, can, b, R, s, clip=0.2), b, s, R, (x, y)))
    cands.sort(key=lambda c: c[0])
    log(f"lying can: {len(cands)} grid candidates; best cost {cands[0][0]:.2f} at "
        f"({cands[0][4][0]:.3f}, {cands[0][4][1]:.3f}) m")
    best = None
    for c, b, _, R, xy in _distinct(cands, 4):
        b1, R1, _ = _refine(coarse, can, b, R, s, free_scale=False, clips=(0.2,))
        bn, Rn, _ = _refine(fine, can, b1, R1, s, free_scale=False)
        cost = fit_cost(fine, can, bn, Rn, s, clip=0.05)
        x, y, z, tilt = surface_xy(bn, Rn, b_up, R_up, s, can)
        log(f"  start {c:.2f} at ({xy[0]:.3f}, {xy[1]:.3f}) -> refined {cost:.3f} at ({x:.3f}, {y:.3f}), "
            f"centre {z * 1000:.0f} mm above the surface (expected {can.radius * 1000:.0f}), axis {tilt:.1f} deg off it")
        if any(math.hypot(x - ax, y - ay) < 0.05 for ax, ay in avoid):
            continue  # slid onto a can already found
        if best is None or cost < best["cost"]:
            best = {"b": bn, "R": Rn, "cost": cost, "x": x, "y": y, "z": z, "tilt_deg": tilt}
    return best


# ----------------------------------------------------------------------------- from SfM points
# The printed label is texture fixed to the can, so SfM triangulates points on the can wall
# (reflections move with the viewpoint and mostly do not triangulate, or land inside the
# can). A cylinder through those points gives the scale (its radius is the can's), the axis
# and the position directly; the surface the cans stand on (a tablet, the table) is a plane
# of SfM points. Far more constraining than outlines in a room full of straight edges.

def _cam_up(c2w_all):
    u = -np.mean([M[:3, 1] for M in c2w_all], 0)
    return u / np.linalg.norm(u)


def plane_ransac(P, hint, max_angle_deg, thr, iters=3000, seed=0):
    """Plane with the most points within thr, normal within max_angle_deg of hint (oriented
    along hint). Returns (normal, a point on it, inlier mask) or None."""
    if len(P) < 20:
        return None
    rng = np.random.default_rng(seed)
    best = None
    cos_max = math.cos(math.radians(max_angle_deg))
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if n @ hint < 0:
            n = -n
        if n @ hint < cos_max:
            continue
        m = np.abs((P - a) @ n) < thr
        if best is None or m.sum() > best[2].sum():
            best = (n, a, m)
    if best is None:
        return None
    n, a, m = best
    Q = P[m]
    n2 = np.linalg.svd(Q - Q.mean(0))[2][2]
    n2 = n2 if n2 @ n > 0 else -n2
    q = Q.mean(0)
    return n2, q, np.abs((P - q) @ n2) < thr


def _circles(X, idx):
    """Circumcircles of point triples X[idx] (k, 3) in 2D: centres (k, 2), radii (k,)."""
    A, B, C = X[idx[:, 0]], X[idx[:, 1]], X[idx[:, 2]]
    d = 2 * (A[:, 0] * (B[:, 1] - C[:, 1]) + B[:, 0] * (C[:, 1] - A[:, 1]) + C[:, 0] * (A[:, 1] - B[:, 1]))
    d = np.where(np.abs(d) < 1e-12, np.nan, d)
    a2, b2, c2 = (A ** 2).sum(1), (B ** 2).sum(1), (C ** 2).sum(1)
    ux = (a2 * (B[:, 1] - C[:, 1]) + b2 * (C[:, 1] - A[:, 1]) + c2 * (A[:, 1] - B[:, 1])) / d
    uy = (a2 * (C[:, 0] - B[:, 0]) + b2 * (A[:, 0] - C[:, 0]) + c2 * (B[:, 0] - A[:, 0])) / d
    cen = np.stack([ux, uy], 1)
    return cen, np.linalg.norm(A - cen, axis=1)


def _max_gap_deg(X, c):
    a = np.sort(np.arctan2(X[:, 1] - c[1], X[:, 0] - c[0]))
    if len(a) < 2:
        return 360.0
    gaps = np.diff(np.concatenate([a, a[:1] + 2 * math.pi]))
    return math.degrees(gaps.max())


def _axis_frame(a, ref):
    """Rotation with z = a and x as close as possible to ref (continuous in a)."""
    return basis(a, x_hint=ref)


def upright_on_plane(points, c2w_all, can, log=print, seed=0, max_plane_deg=25.0, max_tilt_deg=8.0):
    """Surface first: the largest plane of SfM points near the aim point, normal within
    max_plane_deg of the cameras' mean up and below the aim point (a tablet, the table).
    Then the upright can standing on it: axis within max_tilt_deg of the normal, scored by
    the points on the ring within the label's height band above the plane (so a lying can,
    or the stand under the surface, cannot pass for it), refined by least squares.
    Points triangulated from reflections inside the can are left out by the ring tolerance.
    Returns dict(b, R, s, info, inliers, plane) or None."""
    rng = np.random.default_rng(seed)
    up = _cam_up(c2w_all)
    p0 = look_at_point(c2w_all)
    C = np.array([M[:3, 3] for M in c2w_all])
    d = float(np.median(np.linalg.norm(C - p0, axis=1)))
    near = points[np.linalg.norm(points - p0, axis=1) < 0.6 * d]
    if len(near) < 40:
        return None
    # the largest plane only: trying the planes below it as well (in case the largest is
    # the tops of lying cans) picked worse surfaces on noisy chrome-can point clouds
    tol = 0.004 * d
    pool, results = near, []
    for k in range(1):
        pl = plane_ransac(pool, up, max_plane_deg, tol, iters=3000, seed=seed + k)
        if pl is None:
            break
        n, q, _ = pl
        pm = np.abs((near - q) @ n) < tol
        res = _upright_on_given_plane(near, n, q, pm, up, p0, d, can, rng, log, max_tilt_deg)
        if res is not None:
            results.append(res)
        pool = near[(near - q) @ n < -3 * tol]
        if len(pool) < 30:
            break
    if not results:
        return None
    best = max(results, key=lambda r: r["info"]["cylinder_points"])
    if len(results) > 1:
        log(f"plane: {len(results)} candidate surfaces, kept the one with {best['info']['cylinder_points']} label points")
    log(f"points: upright can on the plane {best['info']}")
    return best


def _upright_on_given_plane(near, n, q, pm, up, p0, d, can, rng, log, max_tilt_deg):
    from scipy.optimize import least_squares

    info = {"points_near": int(len(near)), "plane_points": int(pm.sum()),
            "plane_vs_camera_up_deg": round(math.degrees(math.acos(min(1.0, float(n @ up)))), 2)}
    if pm.sum() < 30 or (p0 - q) @ n <= 0:
        log(f"plane: rejected {info}")
        return None
    r_lo, r_hi = can.radius / 0.70 * d, can.radius / 0.15 * d  # camera 15-70 cm from the can
    z_lo, z_hi = can.usable_z[0] / can.radius, can.usable_z[1] / can.radius  # label band, in radii
    h = (near - q) @ n
    above = near[(h > 0) & (h < 0.6 * d)]
    ha = (above - q) @ n
    R_n = basis(n)
    best = None
    for tilt in np.arange(0, max_tilt_deg + 1e-9, 2.0):
        azs = np.linspace(0, 2 * math.pi, max(1, int(round(2 * math.pi * tilt / 2.0))), endpoint=False) if tilt else [0.0]
        for az in azs:
            t = math.radians(tilt)
            ax = R_n @ np.array([math.sin(t) * math.cos(az), math.sin(t) * math.sin(az), math.cos(t)])
            Rb = basis(ax, x_hint=R_n[:, 0])
            X = above @ Rb[:, :2]
            idx = rng.integers(0, len(X), (1500, 3))
            cen, r = _circles(X, idx)
            ok = np.isfinite(r) & (r > r_lo) & (r < r_hi)
            if not ok.any():
                continue
            cen, r = cen[ok], r[ok]
            band = (ha[None] > z_lo * r[:, None]) & (ha[None] < z_hi * r[:, None])
            cnt = ((np.abs(np.linalg.norm(X[None] - cen[:, None], axis=2) - r[:, None]) < 0.04 * r[:, None]) & band).sum(1)
            k = int(cnt.argmax())
            if best is None or cnt[k] > best[0]:
                best = (int(cnt[k]), ax, cen[k], float(r[k]))
    if best is None or best[0] < 15:
        log(f"plane: no can standing on it ({best[0] if best else 0} points)")
        return None
    _, ax, cen, r = best
    R1 = basis(ax, x_hint=R_n[:, 0])
    lim = math.tan(math.radians(max_tilt_deg))

    def frame(x):
        a = R1 @ np.array([x[0], x[1], 1.0])
        return basis(a / np.linalg.norm(a), x_hint=R1[:, 0])

    def resid(x, P):
        return np.linalg.norm(P @ frame(x)[:, :2] - x[2:4], axis=1) - x[4]

    x0 = np.array([0.0, 0.0, cen[0], cen[1], r])
    lb, ub = [-lim, -lim, -np.inf, -np.inf, r_lo], [lim, lim, np.inf, np.inf, r_hi]
    for tol in (0.08, 0.05, 0.04):
        band = (ha > z_lo * x0[4]) & (ha < z_hi * x0[4])
        inl = band & (np.abs(resid(x0, above)) < tol * x0[4])
        x0 = least_squares(resid, np.clip(x0, lb, ub), args=(above[inl],), loss="soft_l1", f_scale=0.02 * x0[4],
                           bounds=(lb, ub)).x
    band = (ha > z_lo * x0[4]) & (ha < z_hi * x0[4])
    inl = band & (np.abs(resid(x0, above)) < 0.04 * x0[4])
    Rb = frame(x0)
    a = Rb[:, 2]
    r = float(x0[4])
    s = r / can.radius
    p_axis = Rb[:, 0] * x0[2] + Rb[:, 1] * x0[3]
    b = p_axis + a * (((q - p_axis) @ n) / (a @ n))  # where the axis meets the surface
    zb = (above[inl] - b) @ a / s
    info.update(cylinder_points=int(inl.sum()), cam_dist_m=round(d / s, 3),
                axis_vs_plane_deg=round(math.degrees(math.acos(min(1.0, float(a @ n)))), 2),
                axis_vs_camera_up_deg=round(math.degrees(math.acos(min(1.0, float(a @ up)))), 2),
                label_height_m=[round(float(np.percentile(zb, 2)), 4), round(float(np.percentile(zb, 98)), 4)],
                largest_angle_gap_deg=round(_max_gap_deg(above[inl] @ Rb[:, :2], x0[2:4]), 1),
                rms_mm=round(float(np.sqrt(np.mean(resid(x0, above[inl]) ** 2))) / s * 1000, 2))
    return {"b": b, "R": Rb, "s": s, "info": info, "inliers": above[inl], "plane": near[pm]}


def snap_to_outlines(data, can, b, R, s, log=print, scales=np.linspace(0.6, 1.2, 13), reach=1.5, steps=9):
    """The point fit's scale and centre checked against the image outlines. SfM points on
    a chrome can mix label points with reflections (virtual images inside the surface), and
    a partial arc of them can fit a circle too large with its centre off. The base stays on
    the surface (moved only within the plane z = 0 of R) and the axis stays R's z; a grid over
    scale x (x, y) within reach can radii, then Powell on (x, y, log scale). Returns
    (b, s, info)."""
    r0 = can.radius * s

    def pose(x):
        sn = s * math.exp(x[2])
        return b + R[:, 0] * x[0] * r0 + R[:, 1] * x[1] * r0, sn

    grid = np.linspace(-reach, reach, steps)
    best = None
    for f in scales:
        for gx in grid:
            for gy in grid:
                x = np.array([gx, gy, math.log(f)])
                bn, sn = pose(x)
                c = fit_cost(data, can, bn, R, sn, clip=0.15)
                if best is None or c < best[0]:
                    best = (c, x)
    c0 = fit_cost(data, can, b, R, s, clip=0.15)
    x = best[1]
    for clip in (0.1, 0.05):
        x = minimize(lambda x: fit_cost(data, can, pose(x)[0], R, pose(x)[1], clip=clip), x, method="Powell",
                     options={"xtol": 1e-3, "ftol": 1e-5, "maxfev": 600}).x
    bn, sn = pose(x)
    info = {"scale_factor": round(float(sn / s), 3), "moved_m": round(float(np.linalg.norm(bn - b) / sn), 4),
            "cost_before": round(c0, 3), "cost_after": round(fit_cost(data, can, bn, R, sn, clip=0.15), 3)}
    log(f"outlines: {info}")
    return bn, sn, info


def snap_lying(data, can, b_up, R_up, s, b, R, log=print, reach=0.01, yaw_reach_deg=10.0):
    """A lying can from the points, checked against the image outlines: it stays on the
    surface (centre one radius above it, axis parallel to it); a grid over its centre
    (within reach metres) and heading (within yaw_reach_deg), then Powell. Returns (b, R, info)."""
    x0, y0, _, _ = surface_xy(b, R, b_up, R_up, s, can)
    d = R[:, 2]
    yaw0 = math.atan2(float(d @ R_up[:, 1]), float(d @ R_up[:, 0]))

    def pose(p):
        return lying_pose(b_up, R_up, s, can, x0 + p[0], y0 + p[1], yaw0 + p[2], False)

    def cost(p, clip):
        # it may not pass through the upright can (axis at the origin): distance from the
        # origin to its axis segment in the surface plane at least ~2 radii
        x, y, yaw = x0 + p[0], y0 + p[1], yaw0 + p[2]
        u = np.array([math.cos(yaw), math.sin(yaw)])
        t = float(np.clip(-(np.array([x, y]) @ u), -can.height / 2, can.height / 2))
        if np.linalg.norm(np.array([x, y]) + t * u) < 1.8 * can.radius:
            return 10.0
        bn, Rn = pose(p)
        return fit_cost(data, can, bn, Rn, s, clip=clip)

    best = None
    for dx in np.linspace(-reach, reach, 5):
        for dy in np.linspace(-reach, reach, 5):
            for dyaw in np.radians(np.linspace(-yaw_reach_deg, yaw_reach_deg, 9)):
                p = np.array([dx, dy, dyaw])
                c = cost(p, 0.15)
                if best is None or c < best[0]:
                    best = (c, p)
    c0 = cost(np.zeros(3), 0.15)
    p = best[1]
    lim = [(-2 * reach, 2 * reach)] * 2 + [(-math.radians(2 * yaw_reach_deg), math.radians(2 * yaw_reach_deg))]
    for clip in (0.1, 0.05):
        p = minimize(lambda p: cost(p, clip), p, method="Powell", bounds=lim,
                     options={"xtol": 1e-4, "ftol": 1e-5, "maxfev": 600}).x
    bn, Rn = pose(p)
    info = {"moved_m": round(float(math.hypot(p[0], p[1])), 4), "turned_deg": round(math.degrees(p[2]), 1),
            "cost_before": round(c0, 3), "cost_after": round(cost(p, 0.15), 3)}
    log(f"outlines, lying can: {info}")
    return bn, Rn, info


def upright_from_points(points, c2w_all, can, log=print, seed=0, max_tilt_deg=35.0):
    """The upright can and the surface it stands on, from the SfM points near where the
    cameras aim. Tries upright_on_plane first (surface, then a can standing on it); if that
    finds nothing, can first: a cylinder with its axis within max_tilt_deg of the cameras'
    mean up, a radius that puts the cameras 15-70 cm away, scored by the points within a
    can-length window along the axis (the label). Then the surface: a plane of points just
    below the label, normal within 12 deg of the axis. Returns dict(b, R, s, info, inliers,
    plane) or None."""
    on_plane = upright_on_plane(points, c2w_all, can, log=log, seed=seed)
    if on_plane is not None and on_plane["info"]["axis_vs_plane_deg"] <= 5.0:
        on_plane["info"]["method"] = "plane first"
        return on_plane
    if on_plane is not None and on_plane["info"]["axis_vs_plane_deg"] > 12.0:
        log("points: the can found on the plane is not standing on it, trying can first")
        on_plane = None
    first = _upright_can_first(points, c2w_all, can, log=log, seed=seed, max_tilt_deg=max_tilt_deg)
    if on_plane is None:
        return first
    # leaning 5-12 deg on the plane: keep it unless can first finds a can closer to upright
    up_p = on_plane["info"].get("axis_vs_camera_up_deg", 90.0)
    up_c = first["info"].get("axis_vs_camera_up_deg", 90.0) if first is not None else 90.0
    if up_c + 5.0 < up_p:
        log(f"points: can first ({up_c:.1f} deg from up) beats the plane fit ({up_p:.1f} deg)")
        return first
    log(f"points: keeping the plane fit ({up_p:.1f} deg from up) over can first ({up_c:.1f} deg)")
    on_plane["info"]["method"] = "plane first"
    return on_plane


def _upright_can_first(points, c2w_all, can, log=print, seed=0, max_tilt_deg=35.0):
    """The can-first search of upright_from_points."""
    from scipy.optimize import least_squares

    rng = np.random.default_rng(seed)
    up = _cam_up(c2w_all)
    p0 = look_at_point(c2w_all)
    C = np.array([M[:3, 3] for M in c2w_all])
    d = float(np.median(np.linalg.norm(C - p0, axis=1)))
    near = points[np.linalg.norm(points - p0, axis=1) < 0.6 * d]
    info = {"points_near": int(len(near))}
    if len(near) < 20:
        return None
    r_lo, r_hi = can.radius / 0.70 * d, can.radius / 0.15 * d  # camera 15-70 cm from the can
    hr = can.height / can.radius
    R_up = basis(up)
    dirs = [up]
    for tilt in np.arange(2.5, max_tilt_deg + 1e-9, 2.5):
        k = max(6, int(round(2 * math.pi * tilt / 2.5)))
        for az in np.linspace(0, 2 * math.pi, k, endpoint=False):
            t = math.radians(tilt)
            dirs.append(R_up @ np.array([math.sin(t) * math.cos(az), math.sin(t) * math.sin(az), math.cos(t)]))
    best = None
    for ax in dirs:
        Rb = basis(ax, x_hint=R_up[:, 0])
        X, Z = near @ Rb[:, :2], near @ ax
        idx = rng.integers(0, len(X), (400, 3))
        cen, r = _circles(X, idx)
        ok = np.isfinite(r) & (r > r_lo) & (r < r_hi)
        if not ok.any():
            continue
        cen, r = cen[ok], r[ok]
        on = np.abs(np.linalg.norm(X[None] - cen[:, None], axis=2) - r[:, None]) < 0.06 * r[:, None]
        for k in np.argsort(on.sum(1))[::-1][:3]:
            zs = np.sort(Z[on[k]])
            if len(zs) < 8:
                continue
            cnt = np.searchsorted(zs, zs + hr * r[k]) - np.arange(len(zs))
            i = int(cnt.argmax())
            if best is None or cnt[i] > best[0]:
                best = (int(cnt[i]), ax, cen[k], float(r[k]), zs[i])
    if best is None or best[0] < 12:
        log(f"points: no can-sized cylinder ({best[0] if best else 0} points)")
        return None
    _, ax, cen, r, z0 = best
    R1 = basis(ax, x_hint=R_up[:, 0])
    lim = math.tan(math.radians(6))

    def frame(x):
        a = R1 @ np.array([x[0], x[1], 1.0])
        return basis(a / np.linalg.norm(a), x_hint=R1[:, 0])

    def resid(x, P):
        return np.linalg.norm(P @ frame(x)[:, :2] - x[2:4], axis=1) - x[4]

    zwin = (near @ ax >= z0 - 0.1 * r) & (near @ ax <= z0 + (hr + 0.1) * r)
    x0 = np.array([0.0, 0.0, cen[0], cen[1], r])
    inl = zwin & (np.abs(resid(x0, near)) < 0.06 * r)
    lb = [-lim, -lim, -np.inf, -np.inf, r_lo]
    ub = [lim, lim, np.inf, np.inf, r_hi]
    for _ in range(3):
        x0 = least_squares(resid, np.clip(x0, lb, ub), args=(near[inl],), loss="soft_l1", f_scale=0.03 * x0[4],
                           bounds=(lb, ub)).x
        inl = zwin & (np.abs(resid(x0, near)) < 0.05 * x0[4])
    Rb = frame(x0)
    a = Rb[:, 2]
    r = float(x0[4])
    s = r / can.radius
    p_axis = Rb[:, 0] * x0[2] + Rb[:, 1] * x0[3]
    z = (near[inl] - p_axis) @ a
    zlo = float(np.percentile(z, 2))
    # the surface: a plane of points just below the label, normal near the axis
    h = (near - p_axis) @ a - zlo
    lat = np.linalg.norm((near - p_axis) - np.outer((near - p_axis) @ a, a), axis=1)
    cand = near[(h > -0.03 * s) & (h < 0.01 * s) & (lat > r * 1.05) & (lat < 0.25 * s)]
    pl = plane_ransac(cand, a, 12, 0.003 * s, iters=1500, seed=seed) if len(cand) >= 20 else None
    if pl is not None and pl[2].sum() >= 15:
        n, q, pm = pl
        b = p_axis + a * (((q - p_axis) @ n) / (a @ n))
        info["plane_points"] = int(pm.sum())
        info["plane_vs_axis_deg"] = round(math.degrees(math.acos(min(1.0, float(a @ n)))), 2)
        plane = cand[pm]
    else:
        b = p_axis + a * (zlo - 0.005 * s)  # the label starts ~5 mm above the base
        info["plane_points"] = 0
        plane = np.zeros((0, 3))
    zb = (near[inl] - b) @ a / s
    info.update(cylinder_points=int(inl.sum()), cam_dist_m=round(d / s, 3),
                axis_vs_camera_up_deg=round(math.degrees(math.acos(min(1.0, float(a @ up)))), 2),
                label_height_m=[round(float(np.percentile(zb, 2)), 4), round(float(np.percentile(zb, 98)), 4)],
                largest_angle_gap_deg=round(_max_gap_deg(near[inl] @ Rb[:, :2], x0[2:4]), 1),
                rms_mm=round(float(np.sqrt(np.mean(resid(x0, near[inl]) ** 2))) / s * 1000, 2))
    log(f"points: upright can {info}")
    return {"b": b, "R": Rb, "s": s, "info": info, "inliers": near[inl], "plane": plane}


def lying_from_points(points, can, b0, R0, s, count, log=print):
    """Cans lying on the surface: cylinders of the known radius, axis parallel to the surface
    and one radius above it. A vote over (heading, offset across the axis), then the best
    can-length window along the axis. Returns a list of dict(b, R, info)."""
    n = R0[:, 2]
    r = can.radius * s
    rel = points - b0
    y = rel @ n
    lat = np.linalg.norm(rel - np.outer(y, n), axis=1)
    keep = (y > 0.003 * s) & (y < (2 * can.radius + 0.01) * s) & (lat > (can.radius + 0.012) * s) & (lat < 0.3 * s)
    P = rel[keep]
    used = np.zeros(len(P), bool)
    log(f"points: {len(P)} candidates for lying cans")
    out = []
    L = can.height * s
    for k in range(count):
        best = None
        Q = P[~used]
        if len(Q) < 10:
            break
        for yaw in np.radians(np.arange(0, 180, 2)):
            a = math.cos(yaw) * R0[:, 0] + math.sin(yaw) * R0[:, 1]
            e = np.cross(n, a)
            x, yy, t = Q @ e, Q @ n, Q @ a
            dy = yy - r
            ok = np.abs(dy) <= r
            if not ok.any():
                continue
            dx = np.sqrt(np.clip(r ** 2 - dy[ok] ** 2, 0, None))
            votes = np.concatenate([x[ok] - dx, x[ok] + dx])
            bins = np.arange(votes.min() - 0.002 * s, votes.max() + 0.004 * s, 0.002 * s)
            if len(bins) < 2:
                continue
            hist, edges = np.histogram(votes, bins)
            for j in np.argsort(hist)[::-1][:5]:
                xc = 0.5 * (edges[j] + edges[j + 1])
                on = np.abs(np.hypot(x - xc, dy) - r) < 0.003 * s
                if on.sum() < 8:
                    continue
                ts = np.sort(t[on])
                cnt = np.searchsorted(ts, ts + L) - np.arange(len(ts))
                # a lying can cannot pass through the upright one (its axis at the origin):
                # the segment's distance from it, in the surface plane, is at least ~2 radii
                along = np.where((ts <= 0) & (ts + L >= 0), 0.0, np.minimum(np.abs(ts), np.abs(ts + L)))
                cnt = np.where(np.hypot(xc, along) < 1.8 * r, 0, cnt)
                i = int(cnt.argmax())
                if best is None or cnt[i] > best[0]:
                    best = (int(cnt[i]), a, e, xc, ts[i])
        if best is None or best[0] < 10:
            log(f"points: lying can {k + 1} not found ({best[0] if best else 0} points)")
            break
        cnt, a, e, xc, t0 = best
        x, yy, t = Q @ e, Q @ n, Q @ a
        on = (np.abs(np.hypot(x - xc, yy - r) - r) < 0.003 * s) & (t >= t0) & (t <= t0 + L)
        tm = 0.5 * (np.percentile(t[on], 2) + np.percentile(t[on], 98))
        centre = xc * e + r * n + tm * a  # relative to b0
        R = basis(a, x_hint=n)
        info = {"points": int(on.sum()),
                "xy_m": [round(float(centre @ R0[:, 0]) / s, 3), round(float(centre @ R0[:, 1]) / s, 3)],
                "heading_deg": round(math.degrees(math.atan2(a @ R0[:, 1], a @ R0[:, 0])), 1),
                "label_length_m": round(float(np.percentile(t[on], 98) - np.percentile(t[on], 2)) / s, 3),
                "largest_angle_gap_deg": round(_max_gap_deg(np.stack([x[on] - xc, yy[on] - r], 1), np.zeros(2)), 1)}
        log(f"points: lying can {k + 1} {info}")
        out.append({"b": b0 + centre - (can.height / 2) * s * a, "R": R, "info": info})
        used |= np.linalg.norm(P - centre, axis=1) < (can.height / 2 + 0.01) * s
    return out
