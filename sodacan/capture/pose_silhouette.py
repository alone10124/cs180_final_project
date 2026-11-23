"""Camera poses without a board: from the can's outline, then the label.

The plan's fallback: "fit the can's two silhouette edges and top ellipse". The outline of
a cylinder of known radius and height fixes five of the six pose numbers: distance,
height, and the three camera angles. It cannot fix the sixth, the camera's angle around
the can axis, because the outline looks the same from every side. That one comes from the
printed label: in surface space (capture/uv_warp.py) the label of frame i is the label of
frame 0 shifted by the change in that angle, so a circular cross-correlation finds it.

Per frame:
  1. Edges (Canny) and their distance transform.
  2. First frame: two long near-parallel line segments are the side walls. Their width gives
     the distance; the elevation is tried at a few values. Later frames start from the
     previous frame's pose (and fall back to detection when that fit is poor).
  3. Refinement: project the two limb lines (over the straight wall, configs/can.yaml
     usable_z), the visible part of the rim (the "top ellipse") and the front of the base,
     and minimise the distance of those points to the nearest image edge (robust least
     squares over five parameters). The rim and base ellipses pin the camera's height
     and tilt, which the near-parallel walls alone leave loose.
  4. Azimuth: warp the frame into surface space with the five-parameter pose and register
     it against the label atlas built so far.

World = can frame as everywhere else (geometry/README.md). The heading of the result is
that of the first frame (camera on the +x axis); a reference panorama is aligned with
metrics.align_azimuth anyway.

A blank (unprinted) mirror can has no label, so its azimuth is unobservable: use the board.
Real cans narrow at the neck and the base, so the rim and the foot are smaller than the
body: set `rim_radius` and `base_radius` in configs/can.yaml (about 0.027 m and 0.025 m for
a 12 oz can; measure yours). Without them both equal the body radius, as in the simulator.
"""
import math
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from sodacan.capture.uv_warp import warp_to_surface


@dataclass
class SilhouettePose:
    c2w: np.ndarray            # (4, 4) or None when the fit failed
    residual_px: float         # median distance of the model outline to the nearest edge
    azimuth_deg: float = 0.0   # camera angle around the can relative to frame 0
    azimuth_score: float = float("nan")  # normalised correlation of the label match
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- model outline

def rim_radius(can):
    return float(can.extra.get("rim_radius") or can.radius)


def base_radius(can):
    return float(can.extra.get("base_radius") or can.radius)


def outline_points(C, can, n_line=40, n_ring=96):
    """World points on the can's outline as seen from camera centre C (3,).

    The two limb generators over the straight wall and the front halves of the rim
    (z = height) and of the base (z = 0, where the can meets the table). Returns (N, 3)
    points and (N,) weights; the rings count a little less than the walls, since a real
    can's rim is rolled and its base is domed."""
    R = can.radius
    rho = math.hypot(C[0], C[1])
    if rho <= R * 1.01:
        return np.zeros((0, 3)), np.zeros(0)
    a0 = math.atan2(C[1], C[0])
    da = math.acos(R / rho)
    z = np.linspace(can.usable_z[0], can.usable_z[1], n_line)
    pts, w = [], []
    for s in (a0 - da, a0 + da):
        pts.append(np.stack([np.full_like(z, R * math.cos(s)), np.full_like(z, R * math.sin(s)), z], 1))
        w.append(np.ones(n_line))
    s = np.linspace(-math.pi, math.pi, n_ring, endpoint=False)
    n = np.stack([np.cos(s), np.sin(s)], 1)
    # Only the front half of each ring: it is an edge from any height (wall against lid or
    # background, wall against table). The back of the rim shows only from above the lid and
    # often has no contrast; counting it only sometimes would let the fit lower its cost by
    # moving the camera below the lid.
    for radius, height in ((rim_radius(can), can.height), (base_radius(can), 0.0)):
        ring = np.stack([radius * np.cos(s), radius * np.sin(s), np.full_like(s, height)], 1)
        keep = ((C[None, :2] - ring[:, :2]) * n).sum(1) > 0
        pts.append(ring[keep])
        w.append(np.full(int(keep.sum()), 0.7))
    return np.concatenate(pts), np.concatenate(w)


N_RES = 2 * 40 + 2 * 96  # fixed residual length for least_squares


def project_np(P, c2w, K):
    Rm, t = c2w[:3, :3], c2w[:3, 3]
    pc = (P - t) @ Rm
    z = pc[:, 2]
    zs = np.maximum(z, 1e-6)
    return np.stack([K.fx * pc[:, 0] / zs + K.cx, K.fy * pc[:, 1] / zs + K.cy], 1), z


# The five free parameters: camera centre (rho, 0, z) on the +x side of the can, and a
# rotation vector. Rotating everything about the can axis changes nothing in the image.

def params_to_c2w(p):
    c2w = np.eye(4)
    c2w[:3, :3] = Rotation.from_rotvec(p[2:5]).as_matrix()
    c2w[:3, 3] = [p[0], 0.0, p[1]]
    return c2w


def canonicalise(c2w):
    """Rotate a pose about the can axis so the camera sits on +x. Returns pose, angle (rad)."""
    a = math.atan2(c2w[1, 3], c2w[0, 3])
    Rz = rot_z(-a)
    return Rz @ c2w, a


def c2w_to_params(c2w):
    c, _ = canonicalise(c2w)
    return np.concatenate([[c[0, 3], c[2, 3]], Rotation.from_matrix(c[:3, :3]).as_rotvec()])


def rot_z(a):
    T = np.eye(4)
    T[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return T


# ----------------------------------------------------------------------------- image side

def edge_map(rgb, roi=None):
    """Canny edges and their distance transform."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    # thresholds from the strong gradients present, so faint walls (a can against a table of
    # similar brightness) still give edges; Canny's own gradient is the L1 Sobel magnitude
    grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    high = float(np.clip(0.2 * np.percentile(grad, 99.5), 12, 150))
    edges = cv2.Canny(gray, 0.4 * high, high, L2gradient=False)
    if roi is not None:
        x0, y0, x1, y1 = roi
        m = np.zeros_like(edges)
        m[max(y0, 0):y1, max(x0, 0):x1] = 1
        edges = edges * m
    dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return edges, dist


def sample_dist(dist, uv, outside=25.0):
    H, W = dist.shape
    x, y = uv[:, 0], uv[:, 1]
    inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    xc, yc = np.clip(x, 0, W - 1.001), np.clip(y, 0, H - 1.001)
    x0, y0 = np.floor(xc).astype(int), np.floor(yc).astype(int)
    fx, fy = xc - x0, yc - y0
    v = (dist[y0, x0] * (1 - fx) * (1 - fy) + dist[y0, x0 + 1] * fx * (1 - fy)
         + dist[y0 + 1, x0] * (1 - fx) * fy + dist[y0 + 1, x0 + 1] * fx * fy)
    return np.where(inside, v, outside)


def merge_collinear(segs, max_angle_deg=2.0, max_offset_px=3.0, max_gap_frac=0.6):
    """Join Hough segments on the same line (a wall broken up by label edges) into one, top end first."""
    segs = np.array([s if s[1] <= s[3] else s[[2, 3, 0, 1]] for s in segs])
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    used = np.zeros(len(segs), bool)
    out = []
    for i in np.argsort(-length):
        if used[i]:
            continue
        d = (segs[i, 2:] - segs[i, :2]) / max(length[i], 1e-9)
        nrm = np.array([d[1], -d[0]])
        p0 = segs[i, :2]
        members = []
        for j in np.nonzero(~used)[0]:
            dj = (segs[j, 2:] - segs[j, :2]) / max(length[j], 1e-9)
            if abs(dj @ nrm) > math.sin(math.radians(max_angle_deg)):
                continue
            if max(abs((segs[j, :2] - p0) @ nrm), abs((segs[j, 2:] - p0) @ nrm)) > max_offset_px:
                continue
            members.append(j)
        t = np.concatenate([[(segs[j, :2] - p0) @ d, (segs[j, 2:] - p0) @ d] for j in members])
        covered = sum(length[j] for j in members)
        lo, hi = t.min(), t.max()
        used[members] = True
        if covered < (1 - max_gap_frac) * (hi - lo):
            lo, hi = 0.0, length[i]  # too gappy: keep the seed alone
        a, b = p0 + lo * d, p0 + hi * d
        out.append(np.concatenate([a, b]) if a[1] <= b[1] else np.concatenate([b, a]))
    return np.array(out)


def wall_segments(edges, min_len_frac=0.15):
    """Long, merged, roughly vertical edge segments (x1, y1, x2, y2), top end first."""
    H, W = edges.shape
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=20, minLineLength=int(0.04 * min(H, W)), maxLineGap=6)
    if segs is None:
        return np.zeros((0, 4))
    segs = merge_collinear(segs.reshape(-1, 4).astype(float))
    ang = np.degrees(np.arctan2(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))  # 0 = vertical
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs = segs[(length >= min_len_frac * min(H, W)) & (np.abs(ang) < 45)]
    ys, xs = np.nonzero(edges)
    pix = np.stack([xs, ys], 1).astype(np.float64)
    return np.array([refit_segment(sg, pix) for sg in segs]).reshape(-1, 4)


def refit_segment(seg, pix, band_px=2.5, iters=2):
    """Least-squares line through the edge pixels along a Hough segment (whose angle is
    quantised to 0.5 degrees, about 2 px over a wall); endpoints from the inliers' extent."""
    a, b = seg[:2], seg[2:]
    for _ in range(iters):
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([d[1], -d[0]])
        rel = pix - a
        t = rel @ d
        L = np.linalg.norm(b - a)
        sel = (np.abs(rel @ n) < band_px) & (t > -2) & (t < L + 2)
        if sel.sum() < 10:
            return seg
        q = pix[sel]
        c = q.mean(0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        d = vt[0] if vt[0][1] >= 0 else -vt[0]
        t = (q - c) @ d
        a, b = c + t.min() * d, c + t.max() * d
    return np.concatenate([a, b]) if a[1] <= b[1] else np.concatenate([b, a])


def detect_walls(edges, K, max_angle_deg=12, top_k=4):
    """Candidate side-wall pairs: two long near-parallel segments that overlap along their
    length, best first. Each is (left, right) with segments (x1, y1, x2, y2), top end first.

    Several candidates are returned because a door or window frame also makes a long
    parallel pair; the outline fit (rim and base included) decides between them."""
    H, W = edges.shape
    segs = wall_segments(edges)
    ang = np.degrees(np.arctan2(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    cands = []
    order = np.argsort(-length)[:40]
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = order[a], order[b]
            if abs(ang[i] - ang[j]) > max_angle_deg:
                continue
            mi, mj = segs[i].reshape(2, 2).mean(0), segs[j].reshape(2, 2).mean(0)
            d = np.array([math.sin(math.radians(ang[i])), math.cos(math.radians(ang[i]))])
            nrm = np.array([d[1], -d[0]])
            sep = abs((mj - mi) @ nrm)
            if sep < 0.04 * W or sep > 0.9 * W:
                continue
            ti = sorted([segs[i, :2] @ d, segs[i, 2:] @ d])
            tj = sorted([segs[j, :2] @ d, segs[j, 2:] @ d])
            overlap = min(ti[1], tj[1]) - max(ti[0], tj[0])
            if overlap < 0.6 * min(length[i], length[j]):
                continue
            # prefer long pairs near the image centre whose walls end level (rim and base)
            centre = (mi + mj) / 2
            off = np.hypot(centre[0] - K.cx, centre[1] - K.cy) / max(W, H)
            level = 1 - min(abs(ti[0] - tj[0]) + abs(ti[1] - tj[1]), sep) / max(sep, 1e-9)
            score = min(length[i], length[j]) * (1 - off) * (0.5 + 0.5 * level)
            left, right = (segs[i], segs[j]) if (segs[i, 0] + segs[i, 2]) <= (segs[j, 0] + segs[j, 2]) else (segs[j], segs[i])
            cands.append((score, (left, right)))
    cands.sort(key=lambda c: -c[0])
    return [c[1] for c in cands[:top_k]]


def track_walls(edges, c2w, K, can, max_px=15.0, max_angle_deg=5.0):
    """The wall pair near where the previous pose projects the limbs, or None."""
    segs = wall_segments(edges, min_len_frac=0.08)
    if len(segs) == 0:
        return None
    pair = []
    for line in limb_lines(c2w, K, can):
        best, best_d = None, max_px
        for sg in segs:
            d = sg[2:] - sg[:2]
            dn = d / max(np.linalg.norm(d), 1e-9)
            if abs(dn @ line[:2]) > math.sin(math.radians(max_angle_deg)):
                continue
            dist = 0.5 * (abs(sg[:2] @ line[:2] + line[2]) + abs(sg[2:] @ line[:2] + line[2]))
            if dist < best_d:
                best, best_d = sg, dist
        if best is None:
            return None
        pair.append(best)
    return tuple(pair)


def initial_pose(walls, elev, K, can):
    """Pose from the wall pair for an assumed viewing elevation elev (radians, seen from the
    can's middle). The walls' top end is taken as the rim and their bottom end as the base."""
    left, right = walls
    mid_top = (left[:2] + right[:2]) / 2
    mid_bot = (left[2:] + right[2:]) / 2
    f = 0.5 * (K.fx + K.fy)
    width = math.hypot(*(((left[:2] + left[2:]) - (right[:2] + right[2:])) / 2))
    alpha = math.atan(0.5 * width / f)            # apparent half width of the body
    rho_axis = can.radius / math.sin(alpha)       # distance from the camera to the axis
    C = np.array([rho_axis, 0.0, can.height / 2 + rho_axis * math.tan(elev)])
    b = []
    for px in (mid_top, mid_bot):
        v = np.array([(px[0] - K.cx) / K.fx, (px[1] - K.cy) / K.fy, 1.0])
        b.append(v / np.linalg.norm(v))
    t = []
    for P in (np.array([0.0, 0.0, can.height]), np.array([0.0, 0.0, 0.0])):
        v = P - C
        t.append(v / np.linalg.norm(v))
    # Wahba: the rotation taking the camera bearings onto the world directions
    b, t = np.array(b), np.array(t)
    b = np.vstack([b, np.cross(b[0], b[1]) / max(np.linalg.norm(np.cross(b[0], b[1])), 1e-9)])
    t = np.vstack([t, np.cross(t[0], t[1]) / max(np.linalg.norm(np.cross(t[0], t[1])), 1e-9)])
    U, _, Vt = np.linalg.svd(t.T @ b)
    Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = Rm, C
    return c2w


N_WALL = 12  # samples along each detected wall
LOWER = [0.05, -1.0, -np.inf, -np.inf, -np.inf]  # camera 5 cm to 3 m from the axis, below 2 m
UPPER = [3.0, 2.0, np.inf, np.inf, np.inf]


def limbs(c2w, K, can):
    """The two projected limbs, left first: (line (a, b, c) with a^2 + b^2 = 1, image point
    of the limb at the base z = 0, image point at the rim z = height)."""
    C = c2w[:3, 3]
    rho = math.hypot(C[0], C[1])
    a0, da = math.atan2(C[1], C[0]), math.acos(min(can.radius / max(rho, 1e-9), 1.0))
    out = []
    for s_ in (a0 - da, a0 + da):
        P = np.array([[can.radius * math.cos(s_), can.radius * math.sin(s_), z] for z in (0.0, can.height)])
        uv, _ = project_np(P, c2w, K)
        d = uv[1] - uv[0]
        n = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-9)
        out.append((np.array([n[0], n[1], -n @ uv[0]]), uv[0], uv[1]))
    return out if out[0][1][0] <= out[1][1][0] else out[::-1]


def limb_lines(c2w, K, can):
    return [l[0] for l in limbs(c2w, K, can)]


END_WEIGHT = 0.5  # wall ends are less certain than wall lines (rolled rim, domed foot, gaps)
END_CAP = 8.0     # px; end mismatches beyond this stop counting more


def outline_residual(p, dist, K, can, walls=None):
    """Model outline -> nearest edge distances, plus the detected walls against the projected limbs.

    The wall terms tie the fit to the walls actually found in this frame: their points lie on
    the limb lines, and their top and bottom ends sit where the rim and base meet the limbs.
    Without them the outline can shrink onto unrelated edges, or slide its rim onto a printed
    band (a band edge is an ellipse just like the rim)."""
    c2w = params_to_c2w(p)
    P, w = outline_points(c2w[:3, 3], can)
    out = np.zeros(N_RES + 2 * (N_WALL + 2))  # fixed length; unused entries stay 0
    if len(P) == 0:
        return out + 25.0
    uv, z = project_np(P, c2w, K)
    r = sample_dist(dist, uv) * w
    r[z <= 0] = 25.0
    out[: len(r)] = r[:N_RES]
    if walls is not None:
        t = np.linspace(0, 1, N_WALL)[:, None]
        k = N_RES
        for seg, (line, base_px, rim_px) in zip(walls, limbs(c2w, K, can)):
            pts = seg[:2] * (1 - t) + seg[2:] * t
            out[k : k + N_WALL] = pts @ line[:2] + line[2]
            along = rim_px - base_px
            along = along / max(np.linalg.norm(along), 1e-9)
            top, bottom = (seg[:2], seg[2:]) if seg[1] <= seg[3] else (seg[2:], seg[:2])
            # squashed: a wall broken by a label edge, or merged with a frame behind it, has a
            # wrong end, and should then not pull
            out[k + N_WALL] = END_WEIGHT * END_CAP * math.tanh(((top - rim_px) @ along) / END_CAP)
            out[k + N_WALL + 1] = END_WEIGHT * END_CAP * math.tanh(((bottom - base_px) @ along) / END_CAP)
            k += N_WALL + 2
    return out


def refine(p0, dist, K, can, walls=None, max_nfev=150, coarse=True):
    """Robust least squares, coarse (wide basin) then fine; tracking from the previous
    frame skips the coarse stage. Returns params and the median distance of the outline
    (and the walls) to the image edges (pixels)."""
    p = np.clip(np.asarray(p0, dtype=float), np.array(LOWER) + 1e-6, np.array(UPPER) - 1e-6)
    for f_scale in ((6.0, 1.5) if coarse else (1.5,)):
        # Cauchy: an outline part with no image edge behind it (the back of the rim against a
        # background of the lid's brightness) must not drag the fit; soft_l1 grows too fast
        res = least_squares(outline_residual, p, args=(dist, K, can, walls), loss="cauchy", f_scale=f_scale,
                            x_scale=[0.01, 0.01, 0.02, 0.02, 0.02], bounds=(LOWER, UPPER), max_nfev=max_nfev,
                            diff_step=1e-3)
        p = res.x
    c2w = params_to_c2w(p)
    P, _ = outline_points(c2w[:3, 3], can)
    if len(P) == 0:
        return p, float("inf")
    uv, _ = project_np(P, c2w, K)
    r = sample_dist(dist, uv)
    if walls is not None:
        r = np.concatenate([r, np.abs(outline_residual(p, dist, K, can, walls)[N_RES:])])
    return p, float(np.median(r))


def fit_cost(p, dist, K, can, walls, f_scale=1.5):
    """The robust (Cauchy) cost the fine stage minimises; lower is better. Unlike the median
    residual it separates the right basin from a nearby wrong one (rim on a printed band)."""
    r = outline_residual(p, dist, K, can, walls) / f_scale
    return float(np.sum(np.log1p(r * r)))


def fit_frame(edges, dist, K, can, prev=None, detect=True, elevations_deg=(-5, 5, 15, 25, 35, 50),
              orbit_deg=(-6, -3, -1.5, 1.5, 3, 6), track_ok_px=1.5, wall_pairs=None):
    """Fit one frame and keep the lowest robust cost over several starts: the previous pose
    (walls tracked near its projected limbs), and, when detect is set or tracking fails,
    each candidate wall pair at several assumed elevations (wall_pairs: these instead of
    detect_walls); then orbited restarts of the best. Returns params, median residual (px), walls."""
    starts = []
    if prev is not None:
        walls = track_walls(edges, params_to_c2w(prev), K, can)
        if walls is not None:
            starts.append((prev, walls, False))
            if not detect:
                _, r = refine(prev, dist, K, can, walls, max_nfev=40, coarse=False)
                detect = r > track_ok_px
    if detect or not starts:
        for walls in (detect_walls(edges, K) if wall_pairs is None else wall_pairs):
            for e in elevations_deg:
                starts.append((c2w_to_params(initial_pose(walls, math.radians(e), K, can)), walls, True))
    best = None
    for p0, walls, coarse in starts:
        p, r = refine(p0, dist, K, can, walls, max_nfev=80, coarse=coarse)
        c = fit_cost(p, dist, K, can, walls)
        if best is None or c < best[0]:
            best = (c, p, r, walls)
    if best is None:
        return None, float("inf"), None
    # The remaining local minima lie along the camera's orbit up or down around the can (the
    # rim snapping to a neighbouring edge): restart from orbited copies of the best fit.
    for _ in range(2):
        c0, p_best, _, walls = best
        improved = False
        for delta in orbit_deg:
            p0 = orbit(p_best, math.radians(delta), can)
            p, r = refine(p0, dist, K, can, walls, max_nfev=80, coarse=False)
            c = fit_cost(p, dist, K, can, walls)
            if c < best[0] - 1e-6:
                best, improved = (c, p, r, walls), True
        if not improved:
            break
    _, p, _, walls = best
    p, r = refine(p, dist, K, can, walls, coarse=False)
    return p, r, walls


def orbit(p, angle, can):
    """Params after orbiting the camera by angle (radians) up around the horizontal axis
    through the can's middle, perpendicular to the viewing plane."""
    c2w = params_to_c2w(p)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.0, -angle, 0.0]).as_matrix()  # camera on +x: about -y lifts it
    centre = np.array([0.0, 0.0, can.height / 2])
    T[:3, 3] = centre - T[:3, :3] @ centre
    return c2w_to_params(T @ c2w)


# ----------------------------------------------------------------------------- azimuth

def label_strip(frame_lin, c2w, K, can, cols):
    """Grey surface-space strip of one frame (rows, cols) and its validity mask, high-passed in s."""
    fr = frame_lin[None]
    pose = torch.tensor(c2w, dtype=torch.float32, device=fr.device)[None]
    rows = max(16, cols // 8)
    obs, valid = warp_to_surface(fr, pose, K, can, rows=rows, cols=cols)
    lo, hi = can.usable_z
    hgt = can.height * (1 - (torch.arange(rows, device=fr.device) + 0.5) / rows)
    band = ((hgt > lo) & (hgt < hi))[:, None]
    g = torch.log(obs[0].mean(0).clamp_min(1e-4))
    m = (valid[0] & band).float()
    g = (g - (g * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)) * m
    return g.cpu().numpy(), m.cpu().numpy()


def register_shift(g, m, ref, ref_m, search_cols, centre=0):
    """Circular shift k (columns) maximising the masked NCC of g(s) with ref(s + k)."""
    cols = g.shape[1]
    best, best_k, scores = -2.0, 0, {}
    for k in range(centre - search_cols, centre + search_cols + 1):
        r, rm = np.roll(ref, -k, axis=1), np.roll(ref_m, -k, axis=1)
        w = m * rm
        n = w.sum()
        if n < 0.05 * m.size:
            continue
        a, b = g * w, r * w
        a = a - a.sum() / n * w
        b = b - b.sum() / n * w
        den = math.sqrt((a * a).sum() * (b * b).sum()) + 1e-9
        s = float((a * b).sum() / den)
        scores[k] = s
        if s > best:
            best, best_k = s, k
    if best_k - 1 in scores and best_k + 1 in scores:  # parabolic sub-column peak
        sm, s0, sp = scores[best_k - 1], scores[best_k], scores[best_k + 1]
        den = sm - 2 * s0 + sp
        sub = 0.5 * (sm - sp) / den if abs(den) > 1e-9 else 0.0
    else:
        sub = 0.0
    return best_k + float(np.clip(sub, -0.5, 0.5)), best


# ----------------------------------------------------------------------------- driver

def smooth_track(values, ok, window=5):
    """Robust temporal smoothing of per-frame parameters (N, D): a running median over the
    frames that have a pose, then a running mean. A hand-held sweep is smooth over a few
    frames, while single-frame outline fits jitter."""
    idx = np.nonzero(ok)[0]
    out = values.copy()
    if len(idx) < 3:
        return out
    v = values[idx]
    h = window // 2
    med = np.stack([np.median(v[max(0, k - h): k + h + 1], 0) for k in range(len(v))])
    mean = np.stack([med[max(0, k - 1): k + 2].mean(0) for k in range(len(v))])
    out[idx] = mean
    return out


def track_outline(frames_rgb, K, can, roi=None, max_residual_px=3.0, redetect_every=10):
    """Pass 1: the five outline parameters of every frame (None where the outline does
    not fit). Returns a list of (params or None, residual_px, reason)."""
    out, prev = [], None
    for i, rgb in enumerate(frames_rgb):
        edges, dist = edge_map(rgb, roi if prev is None else None)
        p, resid, walls = fit_frame(edges, dist, K, can, prev, detect=prev is None or i % redetect_every == 0)
        if p is None:
            out.append((None, float("inf"), "no wall pair"))
        elif not resid <= max_residual_px:
            out.append((None, resid, "outline does not fit"))
        else:
            prev = p
            out.append((p, resid, ""))
    return out


def view_locked_background(strips, masks, window):
    """Per frame, the masked running median of the strips over +-window frames, in the
    canonical (camera on +x) surface frame. Anything locked to the view rather than to the
    label stays put there: limb Fresnel brightening, highlights, the coverage edge, and much
    of the room's reflection, which slides at half the label's speed. The label slides
    through at the camera's angular speed and drops out of the median."""
    G = np.stack(strips)
    M = np.stack(masks) > 0
    out = []
    for i in range(len(strips)):
        lo, hi = max(0, i - window), min(len(strips), i + window + 1)
        g = np.where(M[lo:hi], G[lo:hi], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns (never observed)
            med = np.nanmedian(g, 0)
        out.append(np.nan_to_num(med) * masks[i])
    return out


def track_azimuth(frames_lin, params, K, can, azimuth_cols=1440, max_step_deg=12.0, background_window=20,
                  refine=True):
    """Pass 2: the camera's angle around the can (degrees, frame 0 = 0) from the label,
    with the view-locked background removed first (background_window = 0 keeps it).
    Returns lists of azimuths and match scores (None where params is None)."""
    strips, masks, idx = [], [], []
    for i, p in enumerate(params):
        if p is None:
            continue
        g, m = label_strip(frames_lin[i], params_to_c2w(p), K, can, azimuth_cols)
        strips.append(g)
        masks.append(m)
        idx.append(i)
    az_out, score_out = [None] * len(params), [None] * len(params)
    if idx:
        az, sc = azimuth_from_strips(strips, masks, azimuth_cols, max_step_deg, background_window, refine)
        for j, i in enumerate(idx):
            az_out[i], score_out[i] = az[j], sc[j]
    return az_out, score_out


def azimuth_from_strips(strips, masks, azimuth_cols, max_step_deg=12.0, background_window=20, refine=True):
    """Azimuths (degrees, first = 0) and match scores of a sequence of label strips:
    frame-to-frame tracking against a growing reference after removing the view-locked
    background, then the joint label + background fit (refine_azimuth)."""
    col_deg = 360.0 / azimuth_cols
    raw = strips
    if background_window and len(strips) > 2:
        bg = view_locked_background(strips, masks, background_window)
        strips = [g - b for g, b in zip(strips, bg)]
    az_out, score_out = [], []
    ref = ref_m = ref_n = None
    az_prev = 0.0
    for g, m in zip(strips, masks):
        if ref is None:
            az, score = 0.0, 1.0
            ref, ref_m, ref_n = g.copy(), m.copy(), m.copy()
        else:
            centre = int(round(az_prev / col_deg))
            k, score = register_shift(g, m, ref, ref_m, int(max_step_deg / col_deg), centre)
            az = k * col_deg
            # add this frame to the reference, shifted into frame-0 coordinates
            kk = int(round(k))
            gs, ms = np.roll(g, kk, axis=1), np.roll(m, kk, axis=1)
            ref = (ref * ref_n + gs * ms) / np.maximum(ref_n + ms, 1)
            ref_n = ref_n + ms
            ref_m = (ref_n > 0).astype(np.float32)
        az_prev = az
        az_out.append(az)
        score_out.append(score)
    if refine and len(raw) > 2:
        az_out, score_out = refine_azimuth(raw, masks, az_out, azimuth_cols)
    return az_out, score_out


def refine_azimuth(strips, masks, az_deg, azimuth_cols, iters=6, search_deg=4.0):
    """Joint fit of strip_i(s) = label(s + az_i) + background(s): the label is fixed to the
    can, the background to the view (limb Fresnel brightening, highlights, most of the
    room's reflection). Alternates (1) the label as the mean of the background-free strips
    shifted into frame-0 coordinates, (2) the background as the mean, in the canonical
    frame, of what the label does not explain, (3) each azimuth by matching its
    background-free strip to the label within +-search_deg of the last estimate. Frame 0
    stays at 0. Returns azimuths (degrees) and match scores."""
    col_deg = 360.0 / azimuth_cols
    G, M = np.stack(strips), np.stack(masks)
    k = np.array(az_deg, dtype=float) / col_deg
    B = np.zeros_like(G[0])
    scores = np.ones(len(G))
    for _ in range(iters):
        ki = np.round(k).astype(int)
        num, den = np.zeros_like(G[0]), np.zeros_like(G[0])
        for g, m, kk in zip(G, M, ki):
            num += np.roll((g - B) * m, kk, axis=1)
            den += np.roll(m, kk, axis=1)
        L = num / np.maximum(den, 1)
        Lm = (den > 0).astype(np.float32)
        num, den = np.zeros_like(G[0]), np.zeros_like(G[0])
        for g, m, kk in zip(G, M, ki):
            w = m * np.roll(Lm, -kk, axis=1)
            num += (g - np.roll(L, -kk, axis=1)) * w
            den += w
        B = num / np.maximum(den, 1)
        new_k = k.copy()
        for i in range(1, len(G)):
            new_k[i], scores[i] = register_shift((G[i] - B) * M[i], M[i], L, Lm, int(search_deg / col_deg), int(round(k[i])))
        new_k -= new_k[0]
        done = np.max(np.abs(new_k - k)) < 0.25
        k = new_k
        if done:
            break
    return list(k * col_deg), list(scores)


def estimate_poses_silhouette(frames_rgb, frames_lin, K, can, roi=None, azimuth_cols=1440, max_step_deg=12.0,
                              max_residual_px=3.0, smooth=0, redetect_every=10, background_window=20, log=None):
    """Poses (list of SilhouettePose) for a video of a printed can, no board needed.

    frames_rgb: list of uint8 RGB frames (undistorted); frames_lin: (B, 3, H, W) linear
    tensor of the same frames (for the label match). roi = (x0, y0, x1, y1) limits the
    first frame's search to a box around the can. smooth = window (frames) of the temporal
    smoothing of the five outline parameters and the azimuth; 0 turns it off. Frames must
    be in capture order and close together (stride of a few frames at 30 fps). Every
    redetect_every-th frame also searches for the walls from scratch, so tracking cannot
    stay stuck on a wrong edge. background_window: frames on each side for the
    view-locked background removed before the label match (see view_locked_background)."""
    fits = track_outline(frames_rgb, K, can, roi, max_residual_px, redetect_every)
    params = [f[0] for f in fits]
    azs, scores = track_azimuth(frames_lin, params, K, can, azimuth_cols, max_step_deg, background_window)
    out = []
    for i, (p, resid, reason) in enumerate(fits):
        if p is None:
            out.append(SilhouettePose(None, resid, info={"reason": reason}))
            continue
        # obs_i(s) = ref(s + az): the camera sits at +az relative to frame 0
        c2w = rot_z(math.radians(azs[i])) @ params_to_c2w(p)
        out.append(SilhouettePose(c2w, resid, azs[i], scores[i], info={"params": p}))
        if log:
            log(f"frame {i}: outline residual {resid:.2f} px, azimuth {azs[i]:+.2f} deg (match {scores[i]:.2f})")
    if smooth and smooth > 1:
        ok = np.array([o.c2w is not None for o in out])
        vals = np.zeros((len(out), 6))
        for k, o in enumerate(out):
            if o.c2w is not None:
                vals[k, :5], vals[k, 5] = o.info["params"], o.azimuth_deg
        vals = smooth_track(vals, ok, smooth)
        for k, o in enumerate(out):
            if o.c2w is not None:
                o.c2w = rot_z(math.radians(vals[k, 5])) @ params_to_c2w(vals[k, :5])
                o.azimuth_deg = float(vals[k, 5])
    return out


# ----------------------------------------------------------------------------- can on a board

def fit_can_offset(frames_rgb, c2w, K, can, max_frames=24, search_m=0.03, step_m=0.0025, log=None):
    """Where the can really stands on the board. Board poses place the can at the board's
    printed circle (configs/board*.yaml can_centre); a can put down by hand is off by a
    centimetre or so. Slide the can in the table plane (dx, dy) until its walls and rim
    (not its foot, which stands among the board's squares) lie on image edges, summed over
    frames. Returns (dx, dy) in metres and the median outline distance before and after."""
    idx = np.linspace(0, len(frames_rgb) - 1, min(max_frames, len(frames_rgb))).round().astype(int)
    data = []
    for i in idx:
        _, dist = edge_map(frames_rgb[i])
        M = np.asarray(c2w[i], dtype=np.float64)
        data.append((dist, M, M[:3, 3]))

    def cost(d, robust=True):
        off = np.array([d[0], d[1], 0.0])
        res = []
        for dist, M, C in data:
            P, w = outline_points(C - off, can)
            keep = P[:, 2] > 0.004  # drop the base ring
            uv, z = project_np(P[keep] + off, M, K)
            r = sample_dist(dist, uv) * w[keep]
            res.append(r[z > 0])
        r = np.concatenate(res)
        return float(np.mean(np.minimum(r, 6.0) ** 2)) if robust else float(np.median(r))

    g = np.arange(-search_m, search_m + 1e-9, step_m)
    grid = np.array([[cost((x, y)) for x in g] for y in g])
    iy, ix = np.unravel_index(np.argmin(grid), grid.shape)
    best = np.array([g[ix], g[iy]])
    for s in (step_m / 2, step_m / 4, step_m / 8):  # local refinement
        cands = [best + np.array([a, b]) * s for a in (-1, 0, 1) for b in (-1, 0, 1)]
        best = min(cands, key=cost)
    before, after = cost((0, 0), robust=False), cost(best, robust=False)
    if log:
        log(f"can offset on the board: dx {best[0] * 1000:+.1f} mm, dy {best[1] * 1000:+.1f} mm; "
            f"median outline distance {before:.2f} -> {after:.2f} px")
    return best, before, after


def shift_world(c2w, offset):
    """Camera poses in the frame of a can moved by offset = (dx, dy) in the table plane."""
    T = np.eye(4)
    T[:2, 3] = -np.asarray(offset)[:2]
    return T @ np.asarray(c2w)


def _edge_data(frames_rgb, c2w, max_frames):
    idx = np.linspace(0, len(frames_rgb) - 1, min(max_frames, len(frames_rgb))).round().astype(int)
    data = []
    for i in idx:
        _, dist = edge_map(frames_rgb[i])
        M = np.asarray(c2w[i], dtype=np.float64)
        data.append((dist, M, M[:3, 3]))
    return data


def _outline_cost(data, K, can, d, robust=True):
    """Mean clipped squared distance (or median distance) of the walls and rim of a can
    standing at d = (dx, dy) to the nearest image edge, over the frames in data."""
    off = np.array([d[0], d[1], 0.0])
    res = []
    for dist, M, C in data:
        P, w = outline_points(C - off, can)
        keep = P[:, 2] > 0.004  # drop the base ring
        uv, z = project_np(P[keep] + off, M, K)
        r = sample_dist(dist, uv) * w[keep]
        res.append(r[z > 0])
    r = np.concatenate(res)
    return float(np.mean(np.minimum(r, 6.0) ** 2)) if robust else float(np.median(r))


def locate_cans(frames_rgb, c2w, K, can, count=1, x_range=(-0.13, 0.13), y_range=(-0.09, 0.09), step=0.005,
                max_frames=24, log=None):
    """Where `count` cans stand on the board, when they were not put on the printed circle.
    Grid search over the table plane (world frame of the board poses) for positions whose
    projected walls and rim lie on image edges in all frames, then local refinement. Later
    cans must stand at least a can diameter away from earlier ones. Returns a list of
    (dx, dy) offsets in metres and their median outline distance (px)."""
    data = _edge_data(frames_rgb, c2w, max_frames)
    xs = np.arange(x_range[0], x_range[1] + 1e-9, step)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, step)
    grid = np.array([[_outline_cost(data, K, can, (x, y)) for x in xs] for y in ys])
    found = []
    for _ in range(count):
        g = grid.copy()
        for f in found:
            X, Y = np.meshgrid(xs, ys)
            g[np.hypot(X - f[0], Y - f[1]) < 2 * can.radius + 0.005] = np.inf
        iy, ix = np.unravel_index(np.argmin(g), g.shape)
        best = np.array([xs[ix], ys[iy]])
        for s in (step / 2, step / 4, step / 8, step / 16):
            cands = [best + np.array([a, b]) * s for a in (-1, 0, 1) for b in (-1, 0, 1)]
            best = min(cands, key=lambda d: _outline_cost(data, K, can, d))
        found.append(best)
    out = [(f, _outline_cost(data, K, can, f, robust=False)) for f in found]
    if log:
        for k, (f, med) in enumerate(out):
            log(f"can {k}: at dx {f[0] * 1000:+.1f} mm, dy {f[1] * 1000:+.1f} mm from the printed circle; "
                f"median outline distance {med:.2f} px")
    return out


# ----------------------------------------------------------------------------- photos the board pose missed

def colour_strip(frame_lin, c2w, K, can, cols=720):
    """Surface-space strip of one frame for the label match: log R, G, B, each with its
    per-row mean removed, stacked (3 * rows, cols), and the mask. Colour matters: a window
    or door frame (the usual false outline) is grey, the label is not."""
    fr = frame_lin[None]
    pose = torch.tensor(np.asarray(c2w), dtype=torch.float32, device=fr.device)[None]
    rows = max(16, cols // 8)
    obs, valid = warp_to_surface(fr, pose, K, can, rows=rows, cols=cols)
    lo, hi = can.usable_z
    hgt = can.height * (1 - (torch.arange(rows, device=fr.device) + 0.5) / rows)
    m = (valid[0] & ((hgt > lo) & (hgt < hi))[:, None]).float()
    g = torch.log(obs[0].clamp_min(1e-4))
    g = (g - (g * m).sum(2, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)) * m
    return g.reshape(3 * rows, cols).cpu().numpy(), m.repeat(3, 1).cpu().numpy()


def label_reference(strips, masks):
    """Label in the can frame from strips of frames with known poses: the per-texel median,
    which keeps the print and drops most reflections (they move with the view)."""
    G, M = np.stack(strips), np.stack(masks) > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(np.where(M, G, np.nan), 0)
    ref_m = M.any(0).astype(np.float32)
    return np.nan_to_num(ref) * ref_m, ref_m


def pose_from_outline_and_label(rgb, frame_lin, K, can, ref, ref_m, dist_range=(0.05, 3.0), max_residual_px=3.0,
                                top_k=6):
    """Pose of one photo with no board in view, in the can frame of the label reference.

    Each candidate wall pair is fitted on its own (the outline gives five parameters: all but
    the angle around the can); fits off the outline or at a camera distance outside
    dist_range are dropped; the label match over the full circle gives the remaining angle,
    and the candidate whose label matches best wins. Window and door frames make parallel
    edge pairs that fit an outline as well as the can does, so the outline alone cannot
    pick. Returns (c2w or None, info dict with the match score; the caller thresholds it)."""
    edges, dist = edge_map(rgb)
    pairs = detect_walls(edges, K, top_k=top_k)
    cols = ref.shape[1]
    best, tried = None, []
    for walls in pairs:
        p, resid, _ = fit_frame(edges, dist, K, can, wall_pairs=[walls])
        if p is None:
            continue
        rho = float(math.hypot(p[0], 0.0))
        ok = resid <= max_residual_px and dist_range[0] <= rho <= dist_range[1]
        tried.append((round(float(resid), 2), round(rho, 2)))
        if not ok:
            continue
        g, m = colour_strip(frame_lin, params_to_c2w(p), K, can, cols)
        k, score = register_shift(g, m, ref, ref_m, cols // 2, 0)
        if best is None or score > best[0]:
            best = (score, k, p, resid)
    info = {"candidates": len(pairs), "fits(px,m)": tried}
    if best is None:
        info["reason"] = "no outline fits"
        return None, info
    score, k, p, resid = best
    az = k * 360.0 / cols
    info.update(outline_px=round(float(resid), 2), azimuth_deg=round(float(az), 2), label_match=round(float(score), 3),
                camera_height_m=round(float(p[1]), 3), camera_dist_m=round(float(p[0]), 3))
    # g(s) = ref(s + az): the camera sits at +az in the can frame
    return rot_z(math.radians(az)) @ params_to_c2w(p), info
