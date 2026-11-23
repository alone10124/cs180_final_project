"""Camera poses for photos that miss the board: structure-from-motion on the room.

The board is only seen from above, but the room (table, walls, windows, furniture) is seen
from every viewpoint. COLMAP (pycolmap) recovers every photo's pose from the room alone, up
to an unknown similarity (scale, rotation, translation). The photos that do see the board
tie that frame to the board/can frame: a robust similarity fit (Umeyama, outliers dropped)
on their camera centres and viewing axes. Each pose can then be nudged so the projected can
outline lands on the image edges (pose_silhouette.refine), which removes the small
alignment error that a mirror would double.

Accuracy is measured, not assumed: leave-one-out over the board-posed photos (fit the
similarity without photo j, predict its pose, compare with its board pose).
"""
import math
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from sodacan.capture.pose_silhouette import (c2w_to_params, canonicalise, edge_map, outline_points, params_to_c2w,
                                             project_np, refine, rot_z, sample_dist)


def sfm_poses(images_rgb, names, K, workdir=None, max_features=8192, log=print, refine_focal=False, with_points=False):
    """COLMAP incremental SfM with the intrinsics held fixed (or, refine_focal, one focal
    length refined from K's, principal point fixed). images_rgb: uint8 RGB at the resolution
    of K. Returns ({name: c2w (4x4, OpenCV camera axes, SfM frame)}, stats), stats["focal_px"]
    the focal length used; with_points also the (N, 3) sparse points."""
    import pycolmap

    tmp = Path(workdir or tempfile.mkdtemp(prefix="sfm_"))
    if tmp.exists():
        shutil.rmtree(tmp)
    img_dir = tmp / "images"
    img_dir.mkdir(parents=True)
    for im, n in zip(images_rgb, names):
        cv2.imwrite(str(img_dir / n), cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    db = tmp / "database.db"

    reader = pycolmap.ImageReaderOptions()
    if refine_focal:
        reader.camera_model = "SIMPLE_PINHOLE"
        reader.camera_params = f"{0.5 * (K.fx + K.fy)},{K.cx},{K.cy}"
    else:
        reader.camera_model = "PINHOLE"
        reader.camera_params = f"{K.fx},{K.fy},{K.cx},{K.cy}"
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = int(max(K.width, K.height))
    ext.use_gpu = False
    ext.sift.max_num_features = max_features
    ext.sift.domain_size_pooling = True  # more robust descriptors (CPU only)
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext)
    mo = pycolmap.FeatureMatchingOptions()
    mo.use_gpu = False
    mo.guided_matching = True
    pycolmap.match_exhaustive(db, matching_options=mo)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = bool(refine_focal)
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    maps = pycolmap.incremental_mapping(db, img_dir, tmp / "sparse", options=opts)
    if not maps:
        return ({}, {"models": 0}, None) if with_points else ({}, {"models": 0})
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    poses = {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        w2c = np.eye(4)
        w2c[:3, :] = cfw.matrix()
        poses[im.name] = np.linalg.inv(w2c)
    stats = {"models": len(maps), "registered": len(poses), "of": len(names), "points3D": int(rec.num_points3D())}
    cam = next(iter(rec.cameras.values()))
    stats["focal_px"] = round(float(cam.params[0]), 2)
    pts = np.array([p.xyz for p in rec.points3D.values()]) if with_points else None
    try:
        stats["mean_reprojection_px"] = round(float(rec.compute_mean_reprojection_error()), 3)
    except Exception:  # noqa: BLE001 (name differs between versions)
        pass
    shutil.rmtree(tmp, ignore_errors=True)
    return (poses, stats, pts) if with_points else (poses, stats)


# ----------------------------------------------------------------------------- similarity

def umeyama(src, dst):
    """s, R, t minimising |dst - (s R src + t)|^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / ((xs ** 2).sum() / len(src)))
    return s, R, mu_d - s * R @ mu_s


def align_sim3(sfm_c2w, board_c2w, axis_len_m=0.1, iters=30):
    """Similarity taking SfM poses onto board poses, from camera centres plus points one
    axis_len_m along each camera's x, y and z axes (so orientation counts, not just
    position). Photos whose fit residual stays above max(3 x median, 5 mm) are dropped.
    Returns (s, R, t) and the inlier mask."""
    Cs = np.array([c[:3, 3] for c in sfm_c2w])
    Cb = np.array([c[:3, 3] for c in board_c2w])
    s, R, t = umeyama(Cs, Cb)
    keep = np.ones(len(Cs), bool)
    for _ in range(iters):
        s_prev = s
        src, dst, owner = [], [], []
        for k, (a, b) in enumerate(zip(sfm_c2w, board_c2w)):
            if not keep[k]:
                continue
            for ax in range(4):
                da = a[:3, ax] * (axis_len_m / s) if ax < 3 else 0
                db = b[:3, ax] * axis_len_m if ax < 3 else 0
                src.append(a[:3, 3] + da)
                dst.append(b[:3, 3] + db)
                owner.append(k)
        src, dst, owner = np.array(src), np.array(dst), np.array(owner)
        s, R, t = umeyama(src, dst)
        res = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        per = np.array([res[owner == k].max() if keep[k] else np.inf for k in range(len(Cs))])
        new = per <= max(3 * np.median(per[keep]), 0.005)
        if new.sum() < 3:
            break
        if (new == keep).all() and abs(s - s_prev) < 1e-9 * s:  # the axis offsets depend on s
            break
        keep = new
    return (s, R, t), keep


def apply_sim3(T, c2w):
    s, R, t = T
    out = np.eye(4)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = s * R @ c2w[:3, 3] + t
    return out


def pose_error(a, b):
    """(camera centre distance in m, rotation angle between them in degrees)."""
    dR = a[:3, :3].T @ b[:3, :3]
    ang = math.degrees(math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), ang


# ----------------------------------------------------------------------------- outline polish

def outline_median_px(c2w, dist, K, can):
    P, _ = outline_points(c2w[:3, 3], can)
    if len(P) == 0:
        return float("inf")
    uv, z = project_np(P, c2w, K)
    r = sample_dist(dist, uv)
    r[z <= 0] = 25.0
    return float(np.median(r))


def refine_with_outline(rgb, c2w, K, can, max_move_m=0.02, max_turn_deg=2.0):
    """Nudge a pose (all but the angle around the can axis, which the outline cannot see)
    so the projected can outline sits on the image edges. Kept only if the outline fits
    better and the pose moved less than max_move_m / max_turn_deg. Returns (c2w, info)."""
    _, dist = edge_map(rgb)
    _, a = canonicalise(c2w)
    before = outline_median_px(c2w, dist, K, can)
    p, _ = refine(c2w_to_params(c2w), dist, K, can, walls=None, coarse=False)
    new = rot_z(a) @ params_to_c2w(p)
    after = outline_median_px(new, dist, K, can)
    move, turn = pose_error(new, c2w)
    ok = after < before and move <= max_move_m and turn <= max_turn_deg
    info = {"outline_px_before": round(before, 2), "outline_px_after": round(after, 2),
            "moved_mm": round(move * 1000, 1), "turned_deg": round(turn, 2), "kept": bool(ok)}
    return (new if ok else c2w), info


def can_in_view(c2w, K, can, margin=0.0):
    """Whether the can's axis midpoint projects inside the image, in front of the camera."""
    uv, z = project_np(np.array([[0.0, 0.0, can.height / 2]]), c2w, K)
    u, v = uv[0]
    return bool(z[0] > 0 and -margin <= u <= K.width - 1 + margin and -margin <= v <= K.height - 1 + margin)
