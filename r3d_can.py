"""Record3D (.r3d) capture of a soda can -> room panorama, using the sodacan pipeline.

Poses come from ARKit (in the .r3d metadata); the can's position comes from the LiDAR depth:
the table is the dominant horizontal plane near the image centre, and the can is a cylinder of
known size (configs/can.yaml) standing on it (--pose lying for a can on its side).
Seen from above, the can's centre comes from the lid; from low views, from the wall.
Panoramas are saved in the room frame (gravity up); 05_panorama_can_frame.png is the raw one.

  python r3d_can.py --r3d capture.r3d --out runs/r3d --inspect     # metadata, frames, depth only
  python r3d_can.py --r3d capture.r3d --out runs/r3d               # full run
  python r3d_can.py --r3d capture.r3d --out runs/r3d --model best.pt --reference room.jpg

Writes into --out:
  00_meta.json  00_frames.png  00_depth.png
  01_fit.png    can-frame fit, top view (points, fitted circle, cameras)
  01_poses.png  can outline drawn with the final poses: must sit on the can
  02_reflection.png  04_label.png  04_residual.png
  05_panorama.png  05_panorama_viewer.png  05_coverage.png  [05_reference.png]
  [06_enhanced.png]  panorama.npz  metrics.json
"""
import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sodacan import debug  # noqa: E402
from sodacan.color import auto_exposure, linearize, to_uint8  # noqa: E402
from sodacan.config import load_can, load_pipeline  # noqa: E402
from sodacan.geometry import cylinder  # noqa: E402
from sodacan.geometry.camera import Intrinsics  # noqa: E402
from sodacan.geometry.equirect import pano_dirs, sample, to_viewer  # noqa: E402
from sodacan.geometry.reflect import can_geometry  # noqa: E402
import sodacan.pipeline as sodacan_pipeline  # noqa: E402
from sodacan.pipeline import reconstruct  # noqa: E402

GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def quat_to_rot(qx, qy, qz, qw):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_cv(p):
    qx, qy, qz, qw, tx, ty, tz = p
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(qx, qy, qz, qw) @ GL_TO_CV
    T[:3, 3] = [tx, ty, tz]
    return T


class R3D:
    def __init__(self, path):
        import liblzfse

        self._lzfse = liblzfse
        self.z = zipfile.ZipFile(path)
        self.names = set(self.z.namelist())
        self.meta = json.loads(self.z.read("metadata"))
        self.ids = sorted(int(n[5:-4]) for n in self.names if n.startswith("rgbd/") and n.endswith(".jpg"))
        self.poses = self.meta["poses"]
        if len(self.poses) < self.ids[-1] + 1:
            raise RuntimeError(f"{len(self.poses)} poses for frame ids up to {self.ids[-1]}")
        self.K = np.array(self.meta["K"], dtype=np.float64).reshape(3, 3).T
        self.kw, self.kh = int(self.meta["w"]), int(self.meta["h"])

    def rgb(self, i):
        bgr = cv2.imdecode(np.frombuffer(self.z.read(f"rgbd/{i}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def depth(self, i, w, h):
        d = np.frombuffer(self._lzfse.decompress(self.z.read(f"rgbd/{i}.depth")), np.float32)
        dh, dw = self.meta.get("dh"), self.meta.get("dw")
        if not dh or dh * dw != d.size:
            dh = int(round(np.sqrt(d.size * h / w)))
            dw = d.size // dh
        d = np.nan_to_num(d.reshape(dh, dw).copy(), nan=0.0, posinf=0.0)
        name = f"rgbd/{i}.conf"
        if name in self.names:
            c = np.frombuffer(self._lzfse.decompress(self.z.read(name)), np.uint8)
            if c.size == d.size:
                d[c.reshape(dh, dw) < 1] = 0
        return d

    def intrinsics(self, w, h):
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        kw, kh = self.kw, self.kh
        if (kw > kh) != (w > h):
            kw, kh, fx, fy, cx, cy = kh, kw, fy, fx, cy, cx
        return Intrinsics(kw, kh, fx, fy, cx, cy).scaled(w, h)


def backproject(depth, K, c2w, crop=1.0, step=1):
    dh, dw = depth.shape
    Kd = K.scaled(dw, dh)
    y0, y1 = int(dh * (1 - crop) / 2), int(dh * (1 + crop) / 2)
    x0, x1 = int(dw * (1 - crop) / 2), int(dw * (1 + crop) / 2)
    v, u = np.mgrid[y0:y1:step, x0:x1:step]
    d = depth[y0:y1:step, x0:x1:step]
    ok = (d > 0.05) & (d < 4.0)
    u, v, d = u[ok], v[ok], d[ok]
    pc = np.stack([(u - Kd.cx) / Kd.fx * d, (v - Kd.cy) / Kd.fy * d, d], -1)
    return pc @ c2w[:3, :3].T + c2w[:3, 3]


def fit_circle_fixed(p, R, c0, iters=60):
    c = c0.copy()
    for gate in np.linspace(0.04, 0.008, iters):
        r = p - c
        dist = np.linalg.norm(r, axis=1)
        inl = np.abs(dist - R) < gate
        if inl.sum() < 20:
            break
        c = (p[inl] - R * r[inl] / dist[inl, None]).mean(0)
    dist = np.linalg.norm(p - c, axis=1)
    inl = np.abs(dist - R) < 0.008
    return c, inl


def fit_circle_free(p):
    A = np.c_[p, np.ones(len(p))]
    a, b, e = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    c = np.array([a / 2, b / 2])
    return c, float(np.sqrt(max(e + c @ c, 0)))


def frame_from_axes(x, y, z, origin):
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z])
    T[:3, 3] = -T[:3, :3] @ origin
    return T


ROOM_FROM_WORLD = frame_from_axes(np.array([1.0, 0, 0]), np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), np.zeros(3))


def table_height(pts, cam_y):
    below = pts[pts[:, 1] < cam_y - 0.05]
    bins = np.arange(below[:, 1].min(), below[:, 1].max() + 0.005, 0.005)
    hist, edges = np.histogram(below[:, 1], bins)
    k = int(hist.argmax())
    y = 0.5 * (edges[k] + edges[k + 1])
    return float(np.median(below[np.abs(below[:, 1] - y) < 0.01, 1]))


def ray_guess(c2ws):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for T in c2ws:
        o, d = T[:3, 3], T[:3, 2] / np.linalg.norm(T[:3, 2])
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.lstsq(A, b, rcond=None)[0][[0, 2]]


def fit_upright(pts, table_y, can, cams, guess, center):
    h = pts[:, 1] - table_y
    xz_all = pts[:, [0, 2]]
    g = np.asarray(center, dtype=np.float64) if center is not None else guess
    lid = (h > can.height - 0.02) & (h < can.height + 0.02) & (np.linalg.norm(xz_all - g, axis=1) < 0.10)
    report = {}
    q = xz_all[lid]
    if len(q) >= 100 and center is None:
        c = np.median(q, 0)
        for gate in (0.06, can.radius + 0.01, can.radius):
            q = q[np.linalg.norm(q - c, axis=1) < gate]
            c = np.median(q, 0)
        ev = np.linalg.eigvalsh(np.cov((q - c).T)) if len(q) > 10 else np.array([0.0, 1.0])
        if np.sqrt(ev[0] / max(ev[1], 1e-12)) < 0.6:
            q = q[:0]
    if len(q) >= 100 and center is None:
        lh = h[lid][np.linalg.norm(xz_all[lid] - c, axis=1) < 0.7 * can.radius]
        report.update({"center_from": "lid", "lid_points": int(len(q)), "lid_height_m": float(np.median(lh)),
                       "lid_extent_m": float(np.percentile(np.linalg.norm(q - c, axis=1), 95))})
    xz = pts[(h > 0.015) & (h < can.height - 0.015)][:, [0, 2]]
    if "center_from" in report:
        xz_fit = xz[np.linalg.norm(xz - c, axis=1) < can.radius + 0.04]
        inl = np.abs(np.linalg.norm(xz_fit - c, axis=1) - can.radius) < 0.008
        cf, rf = fit_circle_free(xz_fit[inl]) if inl.sum() > 20 else (c, float("nan"))
    elif center is None:
        near = xz[np.linalg.norm(xz - g, axis=1) < 0.10]
        if len(near) < 50:
            raise RuntimeError(f"only {len(near)} depth points near the can; pass --center x,z (see 01_fit.png)")
        arc = np.median(near, 0)
        away = arc - cams.mean(0)
        c0 = arc + can.radius * away / max(np.linalg.norm(away), 1e-9)
        xz_fit = xz[np.linalg.norm(xz - c0, axis=1) < can.radius + 0.04]
        c, inl = fit_circle_fixed(xz_fit, can.radius, c0)
        cf, rf = fit_circle_free(xz_fit[inl]) if inl.sum() > 20 else (c, float("nan"))
        report["center_from"] = "wall"
    else:
        c = g
        xz_fit = xz[np.linalg.norm(xz - c, axis=1) < can.radius + 0.04]
        inl = np.abs(np.linalg.norm(xz_fit - c, axis=1) - can.radius) < 0.008
        cf, rf = c, float("nan")
        report["center_from"] = "--center"
    T = frame_from_axes(np.array([1.0, 0, 0]), np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), np.array([c[0], table_y, c[1]]))
    report.update({"center_xz": c.tolist(), "wall_points": int(len(xz_fit)), "wall_inliers": int(inl.sum()),
                   "free_fit_radius": rf, "free_fit_center_offset_mm": float(np.linalg.norm(cf - c) * 1000)})
    if "lid_height_m" in report and abs(report["lid_height_m"] - can.height) > 0.015:
        print(f"warning: lid seen at {report['lid_height_m'] * 100:.1f} cm, configs/can.yaml says {can.height * 100:.1f} cm")
    return T, report, (xz, xz_fit[inl], c, None)


def fit_lying(pts, table_y, can, guess, center):
    R = can.radius
    h = pts[:, 1] - table_y
    body = pts[(h > 0.008) & (h < 2 * R + 0.01)]
    xz = body[:, [0, 2]]
    hb = body[:, 1] - table_y
    g = np.asarray(center, dtype=np.float64) if center is not None else guess
    near = np.linalg.norm(xz - g, axis=1) < 0.15
    top = near & (hb > 1.2 * R)
    if top.sum() < 50:
        raise RuntimeError(f"only {int(top.sum())} depth points on top of the can; pass --center x,z (see 01_fit.png)")
    sub = np.flatnonzero(near)
    sub = sub[np.random.default_rng(0).permutation(len(sub))[:4000]]
    best = (-1, None)
    for ang in np.radians(np.arange(0, 180, 1.0)):
        a = np.array([np.cos(ang), np.sin(ang)])
        b = np.array([-a[1], a[0]])
        cs = np.stack([(xz[sub] - g) @ b, hb[sub]], 1)
        topk = cs[:, 1] > 1.2 * R
        c2, inl = fit_circle_fixed(cs, R, np.array([np.median(cs[topk, 0]), R]), iters=20)
        score = int(inl.sum()) - 3 * abs(c2[1] - R) / 0.001
        if score > best[0]:
            best = (score, a)
    a = best[1]
    a0 = np.arctan2(a[1], a[0])
    sub = np.flatnonzero(near)
    for ang in a0 + np.radians(np.arange(-3, 3.01, 0.1)):
        a_ = np.array([np.cos(ang), np.sin(ang)])
        b_ = np.array([-a_[1], a_[0]])
        cs = np.stack([(xz[sub] - g) @ b_, hb[sub]], 1)
        c2, inl = fit_circle_fixed(cs, R, np.array([np.median(cs[cs[:, 1] > 1.2 * R, 0]), R]), iters=20)
        res = np.abs(np.linalg.norm(cs[inl] - c2, axis=1) - R).mean() if inl.sum() > 50 else np.inf
        score = -res + 1e-3 * inl.mean()
        if ang == a0 + np.radians(-3) or score > best_fine[0]:
            best_fine = (score, a_)
    a = best_fine[1]
    b = np.array([-a[1], a[0]])
    sv = (xz - g) @ a
    lo, hi = np.percentile(sv[top], [1, 99])
    s_mid = 0.5 * (lo + hi)
    seg = near & (np.abs(sv - s_mid) < 0.5 * can.height - 0.01)
    cs = np.stack([(xz[seg] - g) @ b, hb[seg]], 1)
    c2, inl = fit_circle_fixed(cs, R, np.array([np.median(cs[cs[:, 1] > 1.2 * R, 0]), R]))
    cf, rf = fit_circle_free(cs[inl]) if inl.sum() > 20 else (c2, float("nan"))
    cxz = g + s_mid * a + c2[0] * b
    axis = np.array([a[0], 0.0, a[1]])
    up = np.array([0.0, 1.0, 0.0])
    origin = np.array([cxz[0], table_y + c2[1], cxz[1]]) - 0.5 * can.height * axis
    T = frame_from_axes(up, np.cross(axis, up), axis, origin)
    report = {"center_xz": cxz.tolist(), "axis_xz": a.tolist(), "axis_height_m": float(c2[1]),
              "visible_length_m": float(hi - lo), "wall_points": int(seg.sum()), "wall_inliers": int(inl.sum()),
              "free_fit_radius": rf, "free_fit_center_offset_mm": float(np.linalg.norm(cf - c2) * 1000)}
    if hi - lo < 0.7 * can.height:
        print(f"warning: only {100 * (hi - lo):.1f} cm of the can's length seen in depth; its centre along the axis may be off")
    ends = np.array([cxz - 0.5 * can.height * a, cxz + 0.5 * can.height * a])
    rect = np.array([ends[0] + R * b, ends[1] + R * b, ends[1] - R * b, ends[0] - R * b])
    return T, report, (xz[near & ~top], xz[top], cxz, rect)


def fit_can_frame(depths, c2ws, K, can, mode="upright", center=None, table_y=None):
    full = np.concatenate([backproject(d, K, T, 1.0, 2) for d, T in zip(depths, c2ws)])
    if table_y is None:
        table_y = table_height(full, np.median(c2ws[:, 1, 3]))
    table = full[np.abs(full[:, 1] - table_y) < 0.006]
    tc = table - table.mean(0)
    normal = np.linalg.svd(tc[:: max(1, len(tc) // 20000)], full_matrices=False)[2][-1]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[1])))))
    cams = c2ws[:, [0, 2], 3]
    guess = ray_guess(c2ws)
    g = np.asarray(center, dtype=np.float64) if center is not None else guess
    xz = full[:, [0, 2]]
    hh = full[:, 1] - table_y
    near = (np.linalg.norm(xz - g, axis=1) < 0.10) & (hh > 0.008) & (hh < 0.3)
    h99 = float(np.percentile(hh[near], 99)) if near.sum() > 50 else float("nan")
    if mode == "auto":
        mode = "upright" if h99 > 2 * can.radius + 0.02 else "lying"
    if mode == "upright":
        T, report, viz = fit_upright(full, table_y, can, cams, guess, center)
    else:
        T, report, viz = fit_lying(full, table_y, can, guess, center)
    report = {"mode": mode, "object_top_height_m": h99, "table_y": table_y, "table_tilt_deg": tilt, **report,
              "can_radius": can.radius}
    return T, report, (*viz, cams)


def fit_panel(xz, inl, c, rect, cams, R, lo, hi, title, size=600):
    scale = (size - 40) / float((hi - lo).max())

    def px(p):
        return ((np.atleast_2d(p) - lo) * scale + 20).astype(np.int32)

    img = np.full((size, size, 3), 255, np.uint8)
    for pts, col in ((xz[:: max(1, len(xz) // 30000)], (150, 150, 150)), (inl, (220, 30, 30))):
        q = px(pts)
        ok = (q >= 0).all(1) & (q < size).all(1)
        img[q[ok, 1], q[ok, 0]] = col
    if rect is None:
        cv2.circle(img, tuple(int(v) for v in px(c)[0]), max(2, int(R * scale)), (0, 160, 0), 2, cv2.LINE_AA)
    else:
        cv2.polylines(img, [px(rect)], True, (0, 160, 0), 2, cv2.LINE_AA)
    cv2.polylines(img, [px(cams)], False, (40, 40, 220), 2, cv2.LINE_AA)
    cv2.putText(img, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def fit_figure(xz, inl, c, rect, cams, R):
    pts = np.concatenate([cams, c[None]])
    lo, hi = pts.min(0) - 0.1, pts.max(0) + 0.1
    wide = fit_panel(xz, inl, c, rect, cams, R, lo, hi, "top view, x right, z down: blue cameras, green can")
    close = fit_panel(xz, inl, c, rect, cams, R, c - 0.1, c + 0.1, "20 cm around the can: grey depth, red points used for the fit")
    return np.concatenate([wide, np.full((600, 6, 3), 255, np.uint8), close], 1)


def exclude_self(deg):
    base = can_geometry
    c = math.cos(math.radians(deg))

    def geometry(c2w, K, can, dent=None, ss=1):
        g = base(c2w, K, can, dent, ss)
        rh = torch.nn.functional.normalize(g.reflected[..., :2], dim=-1)
        bh = torch.nn.functional.normalize(-g.view[..., :2], dim=-1)
        g.mask = g.mask & ((rh * bh).sum(-1) < c)
        return g

    sodacan_pipeline.can_geometry = geometry
    return geometry


def to_room(pano, weight, valid, can_from_room):
    if torch.allclose(can_from_room, torch.eye(3, device=can_from_room.device), atol=1e-6):
        return pano, weight, valid
    H, W = valid.shape
    d = pano_dirs(H, W, pano.device) @ can_from_room.T
    stacked = torch.cat([pano, weight[None], valid[None].float()], 0)
    out = sample(stacked, d).permute(2, 0, 1)
    v = out[4] > 0.5
    return out[:3] * v, out[3] * v, v


def depth_rgb(d):
    m = d > 0
    v = np.zeros_like(d)
    if m.any():
        lo, hi = np.percentile(d[m], [1, 99])
        v = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    img[~m] = 0
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r3d", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inspect", action="store_true", help="only read metadata, frames and depth")
    ap.add_argument("--max-frames", type=int, default=90, help="frames used, evenly spaced")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--long-side", type=int, default=960)
    ap.add_argument("--fit-frames", type=int, default=16, help="frames whose depth locates the can")
    ap.add_argument("--pose", choices=["upright", "lying", "auto"], default="upright", help="can standing (default) or on its side")
    ap.add_argument("--center", help="override the can centre: x,z in ARKit world metres (read off 01_fit.png)")
    ap.add_argument("--table-y", type=float, help="override the table height in ARKit world metres")
    ap.add_argument("--can", default="can")
    ap.add_argument("--pipeline", default="pipeline")
    ap.add_argument("--label", choices=["median", "min", "percentile", "none"])
    ap.add_argument("--exclude-self", type=float, default=30.0,
                    help="drop reflections within this many degrees (azimuth) of the camera, where the person filming appears; 0 keeps all")
    ap.add_argument("--model", help="restoration checkpoint (best.pt)")
    ap.add_argument("--reference", help="phone/360 panorama of the room from the can's position")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rec = R3D(args.r3d)
    ids = [i for i in rec.ids if i >= args.start and (args.end is None or i <= args.end)]
    sel = [ids[k] for k in np.linspace(0, len(ids) - 1, min(args.max_frames, len(ids))).round().astype(int)]
    sel = sorted(set(sel))

    first = rec.rgb(sel[0])
    h0, w0 = first.shape[:2]
    s = min(1.0, args.long_side / max(h0, w0))
    w, h = int(round(w0 * s)), int(round(h0 * s))
    K = rec.intrinsics(w, h)
    d0 = rec.depth(sel[0], w0, h0)
    per_frame = rec.meta.get("perFrameIntrinsicCoeffs")
    info = {
        "metadata_keys": sorted(rec.meta.keys()),
        "frames_in_file": len(rec.ids),
        "poses_in_file": len(rec.poses),
        "frames_used": len(sel),
        "rgb_size": [w0, h0],
        "metadata_wh": [rec.kw, rec.kh],
        "depth_size": [int(d0.shape[1]), int(d0.shape[0])],
        "depth_m_range": [float(d0[d0 > 0].min()), float(d0[d0 > 0].max())] if (d0 > 0).any() else None,
        "K_used": {"w": K.width, "h": K.height, "fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy},
        "fps": rec.meta.get("fps"),
    }
    if per_frame:
        arr = np.array(per_frame, dtype=np.float64)
        info["per_frame_intrinsics_spread"] = (arr.max(0) - arr.min(0)).tolist()
    (out / "00_meta.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

    idx = [sel[k] for k in np.linspace(0, len(sel) - 1, 6).round().astype(int)]
    shots = [cv2.resize(rec.rgb(i), (w, h), interpolation=cv2.INTER_AREA) for i in idx]
    debug.save_rgb(out / "00_frames.png", debug.montage(shots, cols=3, labels=[f"frame {i}" for i in idx]))
    debug.save_rgb(out / "00_depth.png", debug.montage(
        [cv2.resize(depth_rgb(rec.depth(i, w0, h0)), (w // 2, h // 2), interpolation=cv2.INTER_NEAREST) for i in idx], cols=3))
    if args.inspect:
        return

    can = load_can(args.can)
    cfg = load_pipeline(args.pipeline)
    if args.label:
        cfg["label"]["method"] = args.label

    c2w_world = np.stack([pose_cv(rec.poses[i]) for i in sel])
    fit_ids = [sel[k] for k in np.linspace(0, len(sel) - 1, min(args.fit_frames, len(sel))).round().astype(int)]
    fit_c2w = np.stack([pose_cv(rec.poses[i]) for i in fit_ids])
    center = [float(v) for v in args.center.split(",")] if args.center else None
    can_from_world, report, viz = fit_can_frame([rec.depth(i, w0, h0) for i in fit_ids], fit_c2w, K, can,
                                                args.pose, center, args.table_y)
    debug.save_rgb(out / "01_fit.png", fit_figure(*viz, can.radius))
    print(json.dumps(report, indent=2))
    if report["table_tilt_deg"] > 5:
        print(f"warning: the fitted table is {report['table_tilt_deg']:.1f} deg from horizontal; check 01_fit.png")
    print(f"can pose: {report['mode']} (object top {report['object_top_height_m']*100:.1f} cm above the table)")
    if report["wall_inliers"] < 100:
        print("warning: few depth points on the can wall; check 01_fit.png and consider --center")

    c2w = torch.tensor(np.stack([can_from_world @ T for T in c2w_world]), dtype=torch.float32, device=device)
    rgb = np.stack([cv2.resize(rec.rgb(i), (w, h), interpolation=cv2.INTER_AREA) for i in sel])
    frames = linearize(torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float() / 255, cfg.get("gamma", "srgb"))
    lum = frames.mean((1, 2, 3))
    report["frame_brightness_ratio"] = float(lum.max() / lum.min().clamp_min(1e-6))
    if report["frame_brightness_ratio"] > 3:
        print(f"note: mean frame brightness varies {report['frame_brightness_ratio']:.2f}x; if the table's brightness changes in 00_frames.png, exposure was not locked")

    k6 = np.linspace(0, len(sel) - 1, 6).round().astype(int)
    exp = auto_exposure(frames[k6[0]])
    shots = [to_uint8(frames[k], exp) for k in k6]
    debug.save_rgb(out / "01_poses.png", debug.montage(
        [debug.can_outline(s_, c2w[k], K, can) for s_, k in zip(shots, k6)], cols=3, labels=[f"frame {sel[k]}" for k in k6]))

    geom = exclude_self(args.exclude_self) if args.exclude_self > 0 and args.pose == "upright" else can_geometry
    g = geom(c2w[k6[:3]], K, can)
    debug.save_rgb(out / "02_reflection.png", debug.montage([debug.direction_rgb(g.reflected[k], g.mask[k]) for k in range(3)], cols=3))
    report["can_pixels_per_frame"] = float(can_geometry(c2w[k6], K, can).mask.float().sum((1, 2)).mean())

    result = reconstruct(frames, c2w, K, can, cfg)
    if result.label is not None:
        cnt = result.label_count > 0
        debug.save_rgb(out / "04_label.png", debug.display(result.label, cnt, exp))
        gk = can_geometry(c2w[k6[:1]], K, can)
        res = frames[k6[0]].permute(1, 2, 0).clone()
        m = gk.mask[0]
        res[m] = res[m] - cylinder.atlas_sample(result.label, gk.s[0][m], gk.h[0][m], can.height)
        res[~m] = 0
        debug.save_rgb(out / "04_residual.png", to_uint8(res.permute(2, 0, 1).clamp_min(0), exp * 4))
    can_from_room = torch.tensor((can_from_world @ np.linalg.inv(ROOM_FROM_WORLD))[:3, :3], dtype=torch.float32, device=device)
    pano, weight, valid = to_room(result.pano, result.weight, result.valid, can_from_room)
    debug.save_rgb(out / "05_panorama.png", debug.display(pano, valid))
    debug.save_rgb(out / "05_panorama_viewer.png", debug.display(to_viewer(pano), to_viewer(valid)))
    debug.save_rgb(out / "05_coverage.png", debug.heat(weight, valid))
    debug.save_rgb(out / "05_panorama_can_frame.png", debug.display(result.pano, result.valid))
    saved = {"pano": pano.cpu().numpy(), "weight": weight.cpu().numpy(), "valid": valid.cpu().numpy(),
             "pano_can_frame": result.pano.cpu().numpy(), "valid_can_frame": result.valid.cpu().numpy(),
             "c2w_can": c2w.cpu().numpy(), "frame_ids": np.array(sel)}
    from sodacan.geometry.equirect import solid_angle
    sa = solid_angle(*result.valid.shape, device=result.valid.device)
    report["coverage"] = float((sa * valid).sum() / sa.sum())
    report["coverage_upper"] = float((sa * valid)[: valid.shape[0] // 2].sum() / sa[: valid.shape[0] // 2].sum())
    metrics = {"fit": report}

    if args.reference:
        from sodacan.capture.load import load_reference
        from sodacan.metrics import align_azimuth, all_metrics

        ref = load_reference(args.reference, *result.valid.shape).to(device)
        ref, shift, flip = align_azimuth(pano, ref, valid)
        metrics["reference_alignment"] = {"shift_columns": shift, "mirrored": flip}
        debug.save_rgb(out / "05_reference.png", debug.display(ref, None, auto_exposure(ref, valid)))
        metrics["measured"] = all_metrics(pano, ref, valid)
        saved["reference"] = ref.cpu().numpy()

    if args.model:
        from sodacan.restore.infer import enhance, load_model

        enhanced = enhance(load_model(args.model, device), pano, weight, valid)
        debug.save_rgb(out / "06_enhanced.png", debug.display(enhanced, valid))
        saved["enhanced"] = enhanced.cpu().numpy()
        if args.reference:
            metrics["enhanced"] = all_metrics(enhanced, ref, valid)

    np.savez_compressed(out / "panorama.npz", **saved)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
