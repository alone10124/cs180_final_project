"""Reconstruct the room from still photos of one or more cans standing on the ChArUco board.

  python scripts/run_photos.py --photos data/one_can --cans 1 --out runs/photos/one_can
  python scripts/run_photos.py --photos data/two_cans --cans 2 --out runs/photos/two_cans

For photo sets taken with a phone that has no calibration yet (configs/phone_*.yaml), and
cans put down anywhere on the board:
  1. calibrate the phone from the board in the photos themselves (pinhole, centred
     principal point: a few dozen photos cannot pin down lens distortion),
  2. pose every photo from the board at full resolution,
  3. find where each can stands on the board (outline search, capture/pose_silhouette.py),
     then (one can only) pose the photos that miss the board -- the low ones, which see
     the room above the horizon -- from the can's outline (5 DOF) and a match of their
     label against the label seen in the board-posed photos (the angle around the can),
  4. per can: hide pixels where another can is in front, down-weight the photographer's
     reflection (pipeline.photographer_weight), remove the label, accumulate,
  5. merge the cans into one panorama (same orientation; the cans are a few cm apart, so
     parallax is ignored) and run every restoration checkpoint given with --models.
Writes into --out: phone.yaml, 00_photos.png, 01_poses.png, 04_label_can*.png,
05_panorama.png (per can, merged, merged without the photographer weighting),
05_coverage.png, 06_enhanced_<model>.png, panorama.npz, metrics.json; with a reference
panorama (--reference, or a jpg in <photos>/Ground Truth/): 05_reference.png and metrics.
"""
import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sodacan import debug  # noqa: E402
from sodacan.capture.board import board_size_m, load_board  # noqa: E402
from sodacan.capture.calibrate import calibrate  # noqa: E402
from sodacan.capture.pose_charuco import estimate_pose, make_detector  # noqa: E402
from sodacan.capture.pose_silhouette import (colour_strip, label_reference, locate_cans,  # noqa: E402
                                             pose_from_outline_and_label, register_shift, shift_world)
from sodacan.color import auto_exposure, linearize  # noqa: E402
from sodacan.config import load_can, load_pipeline  # noqa: E402
from sodacan.geometry import equirect  # noqa: E402
from sodacan.geometry.camera import Intrinsics  # noqa: E402
from sodacan.geometry.reflect import can_geometry  # noqa: E402
from sodacan.pipeline import reconstruct  # noqa: E402

COLOURS = [(0, 255, 0), (255, 0, 255), (0, 200, 255), (255, 128, 0)]


def log(*a):
    print(*a, flush=True)


def read_photos(folder):
    """Full-resolution RGB photos (EXIF orientation applied), all turned to the same
    orientation (portrait if most are)."""
    files = sorted(f for f in glob.glob(str(Path(folder) / "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
    portrait = sum(i.shape[0] > i.shape[1] for i in imgs) * 2 >= len(imgs)
    imgs = [np.ascontiguousarray(np.rot90(i)) if (i.shape[0] > i.shape[1]) != portrait else i for i in imgs]
    return files, imgs


def coverage_stats(valid):
    H, W = valid.shape
    v = valid.cpu().numpy()
    el = 90 - (np.arange(H) + 0.5) * 180 / H
    sa = np.cos(np.radians(el))[:, None] * np.ones((1, W))
    rows = np.where(v.any(1))[0]
    return {"sphere_fraction": round(float((sa * v).sum() / sa.sum()), 3),
            "elevation_deg": [round(float(el[rows].min()), 1), round(float(el[rows].max()), 1)] if len(rows) else None}


def pose_with_sfm(args, files, rgb_all, kept, free, c2w_np, K, can, metrics, out):
    """Poses for the photos without a board pose from structure-from-motion on the room,
    tied to the can frame by the board-posed photos (sodacan/capture/pose_sfm.py). Checked
    leave-one-out on the board-posed photos first; used only if that check passes."""
    from sodacan.capture.pose_sfm import (align_sim3, apply_sim3, can_in_view, pose_error, refine_with_outline,
                                          sfm_poses)

    names = [Path(f).name for f in files]
    t1 = time.time()
    try:
        sfm, st = sfm_poses(rgb_all, names, K, log=log)
    except Exception as ex:  # noqa: BLE001 keep the board-only result
        log(f"sfm failed: {ex}")
        metrics["sfm"] = {"error": str(ex)}
        return c2w_np, []
    st["seconds"] = round(time.time() - t1, 1)
    metrics["sfm"] = st
    log(f"sfm: {st}")
    board = {names[i]: c2w_np[0][j] for j, i in enumerate(kept)}
    common = [n for n in board if n in sfm]
    if len(common) < 4:
        log(f"sfm: only {len(common)} board-posed photos registered; not used")
        return c2w_np, []
    T, inl = align_sim3([sfm[n] for n in common], [board[n] for n in common])
    st["alignment_inliers"] = f"{int(inl.sum())} of {len(common)}"
    st["scale_m_per_unit"] = round(float(T[0]), 4)

    # leave-one-out check against the board poses, with and without the outline polish
    rows = []
    for n in common:
        others = [m for m in common if m != n]
        Tn, _ = align_sim3([sfm[m] for m in others], [board[m] for m in others])
        pred = apply_sim3(Tn, sfm[n])
        pol, info = refine_with_outline(rgb_all[names.index(n)], pred, K, can)
        (p0, r0), (p1, r1) = pose_error(pred, board[n]), pose_error(pol, board[n])
        rows.append({"photo": n, "pos_mm": round(p0 * 1000, 1), "rot_deg": round(r0, 2),
                     "polished_pos_mm": round(p1 * 1000, 1), "polished_rot_deg": round(r1, 2), **info})
        log(f"sfm check {n}: {rows[-1]}")
    med = lambda k: float(np.median([r[k] for r in rows]))
    polish = med("polished_rot_deg") < med("rot_deg") and med("polished_pos_mm") <= med("pos_mm")
    rot, pos = (med("polished_rot_deg"), med("polished_pos_mm")) if polish else (med("rot_deg"), med("pos_mm"))
    ok = rot <= args.sfm_max_rot and pos / 1000 <= args.sfm_max_pos
    metrics["sfm_check"] = {"photos": rows, "median_pos_mm": med("pos_mm"), "median_rot_deg": med("rot_deg"),
                            "median_polished_pos_mm": med("polished_pos_mm"),
                            "median_polished_rot_deg": med("polished_rot_deg"), "outline_polish_used": polish,
                            "passed": ok}
    log(f"sfm check: median {med('pos_mm'):.1f} mm / {med('rot_deg'):.2f} deg; with outline polish "
        f"{med('polished_pos_mm'):.1f} mm / {med('polished_rot_deg'):.2f} deg -> "
        f"{'passed' if ok else 'FAILED'} (limits {args.sfm_max_pos * 1000:.0f} mm / {args.sfm_max_rot} deg)")
    if not ok:
        return c2w_np, []

    added, extra, checks = [], [], []
    metrics["board_free"] = {}
    for i in free:
        n = names[i]
        if n not in sfm:
            metrics["board_free"][n] = {"reason": "not registered by sfm"}
            continue
        pose = apply_sim3(T, sfm[n])
        if not can_in_view(pose, K, can):
            metrics["board_free"][n] = {"reason": "can not in view"}
            continue
        info = {}
        if polish:
            pose, info = refine_with_outline(rgb_all[i], pose, K, can)
        C = pose[:3, 3]
        info.update(camera_height_m=round(float(C[2]), 3), camera_dist_m=round(float(np.hypot(C[0], C[1])), 3),
                    azimuth_deg=round(math.degrees(math.atan2(C[1], C[0])), 1))
        metrics["board_free"][n] = info
        added.append(i)
        extra.append(pose)
        if len(checks) < 12:
            checks.append(debug.can_outline(rgb_all[i], torch.tensor(pose, dtype=torch.float32), K, can, COLOURS[0]))
    if checks:
        debug.save_rgb(out / "01b_poses_board_free.png", debug.montage(checks, cols=4))
    log(f"sfm poses added for {len(added)} of {len(free)} photos without the board")
    metrics["posed"]["board_free_kept"] = len(added)
    if added:
        c2w_np = [np.concatenate([c2w_np[0], np.stack(extra)])]
    return c2w_np, added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cans", type=int, default=1)
    ap.add_argument("--board", default="board_printed")
    ap.add_argument("--can", default="can_pepsi")
    ap.add_argument("--phone", default=None, help="calibration yaml; default: calibrate from the photos")
    ap.add_argument("--long-side", type=int, default=2000, help="working resolution")
    ap.add_argument("--pano", type=int, nargs=2, default=[256, 512])
    ap.add_argument("--max-reproj", type=float, default=3.0, help="drop board poses above this RMS (full-res px)")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--no-photographer", action="store_true")
    ap.add_argument("--board-free", choices=["sfm", "label", "none"], default="sfm",
                    help="how to pose photos that miss the board: structure-from-motion on the room (sfm), "
                         "can outline + label match (label), or not at all")
    ap.add_argument("--sfm-max-rot", type=float, default=1.5,
                    help="sfm: use the poses only if the leave-one-out rotation error (median, deg) is below this")
    ap.add_argument("--sfm-max-pos", type=float, default=0.02,
                    help="sfm: ... and the camera centre error (median, m) is below this")
    ap.add_argument("--max-outline", type=float, default=3.0, help="board-free poses: max outline residual (working px)")
    ap.add_argument("--min-match", type=float, default=None,
                    help="board-free poses: min label match (NCC); default from the self-check on board-posed photos")
    ap.add_argument("--calib-from", nargs="*", default=[],
                    help="more photo folders from the same phone to calibrate with (photos of --photos are always used)")
    ap.add_argument("--reference", default=None, help="room panorama to compare with (default: <photos>/Ground Truth/*.jpg)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    metrics = {"photos": args.photos}

    files, full = read_photos(args.photos)
    h, w = full[0].shape[:2]
    log(f"{len(full)} photos, {w} x {h}")
    board, bcfg = load_board(args.board)
    can = load_can(args.can)

    # 1. phone calibration
    if args.phone:
        phone = yaml.safe_load(open(args.phone))
    else:
        extra = [im for d in args.calib_from if Path(d).resolve() != Path(args.photos).resolve() for im in read_photos(d)[1]]
        extra = [im for im in extra if im.shape[:2] == (h, w)]
        phone = calibrate(full + extra, board, min_corners=12, max_views=80, model="pinhole", fix_principal_point=True,
                          square_pixels=True)
        del extra
        (out / "phone.yaml").write_text(yaml.safe_dump(phone, sort_keys=False))
    if (phone["width"], phone["height"]) != (w, h):
        if abs(phone["width"] / phone["height"] - w / h) > 0.01:
            raise SystemExit(f"calibration is for {phone['width']} x {phone['height']}, photos are {w} x {h}")
        # same lens, downsized photos: scale the intrinsics (pixel centres at +0.5)
        sc = w / phone["width"]
        phone = dict(phone, width=w, height=h, fx=phone["fx"] * sc, fy=phone["fy"] * sc,
                     cx=(phone["cx"] + 0.5) * sc - 0.5, cy=(phone["cy"] + 0.5) * sc - 0.5)
        log(f"calibration scaled by {sc:.3f} to {w} x {h}")
    log(f"calibration: fx {phone['fx']:.1f} fy {phone['fy']:.1f} cx {phone['cx']:.1f} cy {phone['cy']:.1f}, "
        f"RMS {phone.get('rms_px', float('nan')):.2f} px over {phone.get('views')} views")
    metrics["calibration"] = {k: phone[k] for k in ("fx", "fy", "cx", "cy", "rms_px", "views") if k in phone}
    Kf = np.array([[phone["fx"], 0, phone["cx"]], [0, phone["fy"], phone["cy"]], [0, 0, 1]], dtype=np.float64)
    dist = np.array(phone.get("dist", [0] * 5), dtype=np.float64)

    # 2. board poses at full resolution
    det = make_detector(board)
    poses = [estimate_pose(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), Kf, dist, board, bcfg, det) for f in full]
    kept = [i for i, p in enumerate(poses) if p.c2w is not None and p.reproj_px <= args.max_reproj]
    errs = [poses[i].reproj_px for i in kept]
    log(f"board pose in {len(kept)} of {len(full)} photos, median reprojection {np.median(errs):.2f} px")
    metrics["posed"] = {"kept": len(kept), "of": len(full), "median_reproj_px": round(float(np.median(errs)), 2),
                        "dropped": [Path(files[i]).name for i in range(len(full)) if i not in kept]}
    if len(kept) < 3:
        raise SystemExit("too few posed photos")

    # working resolution
    s = args.long_side / max(w, h)
    ww, hh = round(w * s), round(h * s)
    K = Intrinsics(ww, hh, phone["fx"] * s, phone["fy"] * s, (phone["cx"] + 0.5) * s - 0.5, (phone["cy"] + 0.5) * s - 0.5)
    Kw = np.array([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1]], dtype=np.float64)
    rgb_all = []
    for f in full:
        if np.any(dist != 0):
            f = cv2.undistort(f, Kf, dist)
        rgb_all.append(cv2.resize(f, (ww, hh), interpolation=cv2.INTER_AREA))
    del full
    rgb = [rgb_all[i] for i in kept]
    c2w = np.stack([poses[i].c2w for i in kept])
    debug.save_rgb(out / "00_photos.png", debug.montage([rgb[j] for j in np.linspace(0, len(rgb) - 1, 6).astype(int)], cols=6))

    # 3. where the cans stand
    bw, bh = board_size_m(bcfg)
    c0 = bcfg["can_centre"]
    xr = max(c0[0], bw - c0[0]) - can.radius
    yr = max(c0[1], bh - c0[1]) - can.radius
    found = locate_cans(rgb, c2w, K, can, count=args.cans, x_range=(-xr, xr), y_range=(-yr, yr), log=log)
    metrics["cans"] = [{"offset_mm": [round(float(v) * 1000, 1) for v in f], "median_outline_px": round(med, 2)}
                       for f, med in found]
    c2w_np = [shift_world(c2w, f) for f, _ in found]

    # 3b. photos without a board pose (one can): outline + label match
    free = [i for i in range(len(rgb_all)) if i not in kept]
    metrics["board_free"] = {}
    if free and args.cans == 1 and args.board_free == "sfm":
        c2w_np, added = pose_with_sfm(args, files, rgb_all, kept, free, c2w_np, K, can, metrics, out)
        if added:
            rgb = rgb + [rgb_all[i] for i in added]
            kept = kept + added
    if free and args.cans == 1 and args.board_free == "label":
        lin = lambda im: linearize(torch.from_numpy(im).permute(2, 0, 1).float().div(255), "srgb").to(device)
        strips = [colour_strip(lin(im), c, K, can) for im, c in zip(rgb, c2w_np[0])]
        ref, ref_m = label_reference([g for g, _ in strips], [m for _, m in strips])
        rho = np.hypot(c2w_np[0][:, 0, 3], c2w_np[0][:, 1, 3])
        dist_range = (0.5 * float(rho.min()), 2.0 * float(rho.max()))
        log(f"board-posed cameras {rho.min():.2f}-{rho.max():.2f} m from the can; board-free fits kept within "
            f"{dist_range[0]:.2f}-{dist_range[1]:.2f} m")

        # self-check: pose board-posed photos without the board (label from the others) and compare
        check, true_scores = [], []
        for j in np.linspace(0, len(rgb) - 1, min(10, len(rgb))).astype(int):
            others = [k for k in range(len(rgb)) if k != j]
            r_j, rm_j = label_reference([strips[k][0] for k in others], [strips[k][1] for k in others])
            true_scores.append(register_shift(strips[j][0], strips[j][1], r_j, rm_j, 0, 0)[1])
            pose, info = pose_from_outline_and_label(rgb[j], lin(rgb[j]), K, can, r_j, rm_j, dist_range, args.max_outline)
            truth = c2w_np[0][j]
            if pose is not None:
                a_t, a_e = math.atan2(truth[1, 3], truth[0, 3]), math.atan2(pose[1, 3], pose[0, 3])
                info["azimuth_error_deg"] = round(math.degrees((a_e - a_t + math.pi) % (2 * math.pi) - math.pi), 1)
                info["position_error_m"] = round(float(np.linalg.norm(pose[:3, 3] - truth[:3, 3])), 3)
            info["true_pose_match"] = round(float(true_scores[-1]), 3)
            check.append(info)
            log(f"self-check {Path(files[kept[j]]).name}: {info}")
        thr = args.min_match if args.min_match is not None else max(0.3, 0.8 * float(np.median(true_scores)))
        good = [c for c in check if abs(c.get("azimuth_error_deg", 999)) < 10 and c.get("position_error_m", 9) < 0.05]
        metrics["board_free_self_check"] = {"photos": check, "threshold": round(thr, 3),
                                            "correct_within_10deg_5cm": f"{len(good)} of {len(check)}",
                                            "correct_and_above_threshold": sum(c["label_match"] >= thr for c in good),
                                            "wrong_but_above_threshold": sum(c.get("label_match", -1) >= thr for c in check if c not in good)}
        log(f"self-check: {metrics['board_free_self_check']['correct_within_10deg_5cm']} right; label-match threshold {thr:.3f}")

        added, extra_c2w, checks_free = [], [], []
        if len(good) < 0.6 * len(check):
            # the method could not re-find known poses: its poses for the other photos are not trusted
            log(f"self-check failed ({len(good)} of {len(check)} right): board-free poses not used")
            metrics["board_free_self_check"]["used"] = False
            free = []
        for i in free:
            t1 = time.time()
            pose, info = pose_from_outline_and_label(rgb_all[i], lin(rgb_all[i]), K, can, ref, ref_m, dist_range, args.max_outline)
            if pose is not None and info["label_match"] < thr:
                info["reason"], pose = "label does not match", None
            info["seconds"] = round(time.time() - t1, 1)
            metrics["board_free"][Path(files[i]).name] = info
            log(f"board-free {Path(files[i]).name}: {info}")
            if pose is not None:
                added.append(i)
                extra_c2w.append(pose)
                if len(checks_free) < 12:
                    checks_free.append(debug.can_outline(rgb_all[i], torch.tensor(pose, dtype=torch.float32), K, can, COLOURS[0]))
        if checks_free:
            debug.save_rgb(out / "01b_poses_board_free.png", debug.montage(checks_free, cols=4))
        log(f"board-free poses for {len(added)} of {len(free)} photos")
        metrics["posed"]["board_free_kept"] = len(added)
        if added:
            rgb = rgb + [rgb_all[i] for i in added]
            c2w_np = [np.concatenate([c2w_np[0], np.stack(extra_c2w)])]
            kept = kept + added
    del rgb_all
    c2w_cans = [torch.tensor(c, dtype=torch.float32, device=device) for c in c2w_np]
    checks = []
    for j in np.linspace(0, len(rgb) - 1, 6).astype(int):
        img = rgb[j]
        for k, cw in enumerate(c2w_cans):
            img = debug.can_outline(img, cw[j], K, can, COLOURS[k % len(COLOURS)])
        checks.append(img)
    debug.save_rgb(out / "01_poses.png", debug.montage(checks, cols=3))

    # 4. per-can reconstruction
    frames = linearize(torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).float().div(255), "srgb").to(device)
    photographer = None if args.no_photographer else {"az_deg": 25, "el_below_deg": 60, "el_above_deg": 15, "weight": 0.02}
    cfg = load_pipeline(pano=list(args.pano), photographer=photographer)
    cfg_plain = load_pipeline(pano=list(args.pano))
    H, W = args.pano
    B = frames.shape[0]
    results, plain = [], []
    for k, cw in enumerate(c2w_cans):
        mask = torch.ones(B, hh, ww, dtype=torch.bool, device=device)
        for j, other in enumerate(c2w_cans):
            if j == k:
                continue
            for i in range(B):
                gk = can_geometry(cw[i : i + 1], K, can)
                gj = can_geometry(other[i : i + 1], K, can)
                mask[i] &= ~(gk.hit & gj.hit & (gj.t < gk.t))[0]
        hidden = 1 - float(mask.float().mean())
        r = reconstruct(frames, cw, K, can, cfg, pixel_mask=mask if args.cans > 1 else None)
        rp = reconstruct(frames, cw, K, can, cfg_plain, label=r.label, pixel_mask=mask if args.cans > 1 else None)
        results.append(r)
        plain.append(rp)
        if r.label is not None:
            debug.save_rgb(out / f"04_label_can{k}.png", debug.display(r.label))
        log(f"can {k}: coverage {coverage_stats(r.valid)}, {hidden:.1%} of pixels hidden by another can")

    def merge(rs):
        num = sum(r.pano * r.weight[None] for r in rs)
        den = sum(r.weight for r in rs)
        valid = torch.stack([r.valid for r in rs]).any(0)
        return num / den.clamp_min(1e-12)[None] * valid[None], den, valid

    pano, weight, valid = merge(results)
    pano0, _, valid0 = merge(plain)
    exp = auto_exposure(pano, valid)
    panels, labels = [], []
    for k, r in enumerate(results):
        panels.append(debug.display(r.pano, r.valid, exp))
        labels.append(f"can {k}")
    if len(results) > 1:
        panels.append(debug.display(pano, valid, exp))
        labels.append("all cans merged")
    panels.append(debug.display(pano0, valid0, exp))
    labels.append("merged, photographer NOT down-weighted (for comparison)")
    debug.save_rgb(out / "05_panorama.png", debug.stack(panels, labels))
    debug.save_rgb(out / "05_coverage.png", debug.heat(weight, valid))
    metrics["coverage"] = coverage_stats(valid)
    log(f"merged coverage {metrics['coverage']}")

    # reference panorama of the room, if there is one (heading unknown: aligned in azimuth)
    ref_path = args.reference
    if ref_path is None:
        cands = sorted(q for d in Path(args.photos).iterdir() if d.is_dir() and "truth" in d.name.lower().replace("grouth", "truth")
                       for q in glob.glob(str(d / "*.jpg")))  # Ground Truth, ground_truth, Grouth Truth
        ref_path = cands[0] if cands else None
    reference = None
    if ref_path:
        from sodacan.capture.load import load_reference
        from sodacan.metrics import align_azimuth, all_metrics

        reference = load_reference(ref_path, H, W).to(device)
        reference, shift, flip = align_azimuth(pano, reference, valid)
        metrics["reference"] = {"file": Path(ref_path).name, "shift_columns": shift, "mirrored": flip,
                                "measured": all_metrics(pano, reference, valid, with_lpips=False)}
        debug.save_rgb(out / "05_reference.png", debug.stack(
            [debug.display(reference, None, auto_exposure(reference, valid)), debug.display(reference, valid, auto_exposure(reference, valid)),
             debug.display(pano, valid, exp)],
            ["reference (aligned)", "reference where the can saw", "measured"]))
        log(f"reference: {metrics['reference']}")

    # 5. restoration
    saved = {"pano": pano.cpu().numpy(), "weight": weight.cpu().numpy(), "valid": valid.cpu().numpy()}
    if args.models:
        from sodacan.restore.infer import enhance, load_model

        for path in args.models:
            name = Path(path).parent.name
            try:
                e = enhance(load_model(path, device), pano, weight, valid)
                debug.save_rgb(out / f"06_enhanced_{name}.png", debug.display(e, valid))
                saved[f"enhanced_{name}"] = e.cpu().numpy()
                if reference is not None:
                    metrics["reference"][name] = all_metrics(e, reference, valid, with_lpips=False)
                log(f"restoration {name}: ok")
            except Exception as ex:  # keep the other models going
                log(f"restoration {name}: failed: {ex}")
    if reference is not None:
        saved["reference"] = reference.cpu().numpy()
    np.savez_compressed(out / "panorama.npz", **saved)
    metrics["seconds"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
