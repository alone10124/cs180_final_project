"""Reconstruct the room from still photos of cans with no board in view at all.

  python scripts/run_noboard.py --photos data/one_can_4 --phone configs/phone_s25.yaml --refine-focal --out runs/one_can_4
  python scripts/run_noboard.py --photos data/three_cans --phone configs/phone_s25.yaml --lying 2 --out runs/three_cans

  1. camera poses from structure-from-motion on the room (capture/pose_sfm.py): known
     intrinsics (a phone calibration, scaled to the photo size; --refine-focal lets COLMAP
     refine the focal length), but no metric scale and no "up",
  2. the surface the cans stand on: the largest plane of SfM points near where the
     cameras aim (a tablet, the table),
  3. the upright can: a cylinder through the SfM points on its printed label, axis near
     that plane's normal; its radius is the can's, which gives the metric scale, and its
     axis is "up" (can_fit.upright_from_points),
  4. --lying cans lying on their side: cylinders of the known radius, axis parallel to the
     surface and one radius above it (can_fit.lying_from_points). If the points show no can,
     both fall back to fitting the can outlines to image edges (--fit outlines),
  5. per can: cameras in that can's frame, pixels hidden by another can dropped, the
     photographer's reflection down-weighted (with that can's "up"), label removed,
     accumulated; each can's panorama turned into the shared world frame (z = the upright
     can's axis) and merged. No restoration step.
Writes into --out: 00_photos.png, 01_cans.png, 05_panorama.png, 05_coverage.png,
05_reference.png (with a reference: --reference or <photos>/Ground Truth/*.jpg),
panorama.npz, metrics.json.
"""
import argparse
import glob
import json
import math
import os
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
from sodacan.capture import can_fit  # noqa: E402
from sodacan.capture.can_masks import detect_can_masks, sam_box_prompter  # noqa: E402
from sodacan.capture.group_fit import fit_group  # noqa: E402
from sodacan.capture.mask_fit import fit_masks, mask_cost  # noqa: E402
from sodacan.capture.multi_can_fit import fit_cans  # noqa: E402
from sodacan.capture.pose_sfm import can_in_view, sfm_poses  # noqa: E402
from sodacan.color import auto_exposure, linearize  # noqa: E402
from sodacan.config import load_can, load_pipeline  # noqa: E402
from sodacan.geometry import equirect  # noqa: E402
from sodacan.geometry.camera import Intrinsics  # noqa: E402
from sodacan.geometry.reflect import can_geometry, reflects_other_can  # noqa: E402
from sodacan.pipeline import reconstruct  # noqa: E402
from sodacan.separate.label_video import fit_label_atlas  # noqa: E402

COLOURS = [(0, 255, 0), (255, 0, 255), (0, 200, 255), (255, 128, 0)]


def log(*a):
    print(*a, flush=True)


def read_photos(folder, long_side):
    """Photos (EXIF orientation applied) turned to the majority orientation and resized so
    the long side is long_side. Returns (files, images, full-resolution (w, h))."""
    files = sorted(f for f in glob.glob(str(Path(folder) / "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    imgs, size = [], None
    for f in files:
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        if size is None:
            size = (im.shape[1], im.shape[0])
        elif (im.shape[1], im.shape[0]) not in (size, size[::-1]):
            raise SystemExit(f"{f}: {im.shape[1]} x {im.shape[0]}, the others are {size[0]} x {size[1]}")
        imgs.append(im)
    portrait = sum(i.shape[0] > i.shape[1] for i in imgs) * 2 >= len(imgs)
    out = []
    for im in imgs:
        if (im.shape[0] > im.shape[1]) != portrait:
            im = np.ascontiguousarray(np.rot90(im))
        s = long_side / max(im.shape[:2])
        out.append(cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA))
    w, h = (min(size), max(size)) if portrait else (max(size), min(size))
    return files, out, (w, h)


def phone_intrinsics(path, w, h):
    """The calibration in path, scaled to w x h (same aspect ratio: the same lens, binned)."""
    K = Intrinsics.from_dict(yaml.safe_load(open(path)))
    if abs(K.width / K.height - w / h) > 0.01:
        raise SystemExit(f"calibration is {K.width} x {K.height}, photos are {w} x {h}: different aspect ratio")
    return K.scaled(w, h)


def coverage_stats(valid):
    H, W = valid.shape
    v = valid.cpu().numpy()
    el = 90 - (np.arange(H) + 0.5) * 180 / H
    sa = np.cos(np.radians(el))[:, None] * np.ones((1, W))
    rows = np.where(v.any(1))[0]
    return {"sphere_fraction": round(float((sa * v).sum() / sa.sum()), 3),
            "elevation_deg": [round(float(el[rows].min()), 1), round(float(el[rows].max()), 1)] if len(rows) else None}


def to_world(r, M, H, W):
    """A can-frame Result turned into the world frame; M maps world directions to can-frame
    ones. Returns (pano, weight, valid)."""
    dev = r.pano.device
    d = equirect.pano_dirs(H, W, dev) @ torch.tensor(M, dtype=torch.float32, device=dev).T
    v = r.valid.float()
    vs = equirect.sample(v[None], d)[..., 0]
    pano = equirect.sample(r.pano * v[None], d).permute(2, 0, 1) / vs.clamp_min(1e-6)[None]
    weight = equirect.sample(r.weight[None], d)[..., 0]
    valid = vs > 0.5
    return pano * valid[None], weight * valid, valid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phone", required=True, help="calibration yaml of the phone (any resolution, same aspect ratio)")
    ap.add_argument("--refine-focal", action="store_true", help="let SfM refine the focal length")
    ap.add_argument("--lying", type=int, default=0, help="number of cans lying on their side (besides one upright can)")
    ap.add_argument("--can", default="can_pepsi")
    ap.add_argument("--long-side", type=int, default=2000, help="working resolution")
    ap.add_argument("--pano", type=int, nargs=2, default=[256, 512])
    ap.add_argument("--fit", choices=["auto", "points", "outlines"], default="auto",
                    help="locate the cans from the SfM points on their labels (auto: outlines if that fails) or outlines only")
    ap.add_argument("--fit-views", type=int, default=12, help="photos used to fit the can outlines")
    ap.add_argument("--no-photographer", action="store_true")
    ap.add_argument("--label-atlas", nargs="+", default=None,
                    help="label_atlas.npy files (label_from_video.py, label_from_render.py): per can, the one that matches "
                         "the scene's own label estimate best is used instead of that estimate")
    ap.add_argument("--label-min-corr", type=float, default=0.3, help="use the video label only if it correlates at least this well")
    ap.add_argument("--keep-can-reflections", action="store_true",
                    help="keep pixels where a can mirrors another can (default: left out of label and room)")
    ap.add_argument("--fit-group", choices=["auto", "on", "off"], default="auto",
                    help="fit all cans together to the SfM points, reflection-aware (auto: when --lying > 0)")
    ap.add_argument("--mask-fit", choices=["auto", "on", "off"], default="auto",
                    help="pose all cans by their silhouettes (OWLv2 + SAM masks, multi-view) before the edge fit "
                         "(auto: when --lying > 0)")
    ap.add_argument("--label-flip-margin", type=float, default=0.05,
                    help="with --label-atlas: turn a lying can end over end if its label then matches the video label better by this")
    ap.add_argument("--no-snap", action="store_true", help="keep the SfM-point fit as is (no outline check of scale and centre)")
    ap.add_argument("--reference", default=None, help="room panorama to compare with (default: <photos>/Ground Truth/*.jpg)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    metrics = {"photos": args.photos}
    can = load_can(args.can)

    files, rgb_all, (w, h) = read_photos(args.photos, args.long_side)
    hh, ww = rgb_all[0].shape[:2]
    K = phone_intrinsics(args.phone, w, h).scaled(ww, hh)
    log(f"{len(files)} photos, {w} x {h}; working {ww} x {hh}, f {K.fx:.1f} px")
    names = [Path(f).name for f in files]

    # 1. structure from motion
    t1 = time.time()
    sfm, st, points = sfm_poses(rgb_all, names, K, log=log, refine_focal=args.refine_focal, with_points=True)
    st["seconds"] = round(time.time() - t1, 1)
    metrics["sfm"] = st
    log(f"sfm: {st}")
    if len(sfm) < 5:
        raise SystemExit("sfm registered too few photos")
    if args.refine_focal:
        f0 = K.fx
        K = Intrinsics(K.width, K.height, st["focal_px"], st["focal_px"], K.cx, K.cy)
        log(f"focal length {f0:.1f} -> {K.fx:.1f} px (working resolution)")
    reg = [i for i, n in enumerate(names) if n in sfm]
    rgb = [rgb_all[i] for i in reg]
    c2w_sfm = [sfm[names[i]] for i in reg]
    del rgb_all
    debug.save_rgb(out / "00_photos.png", debug.montage([rgb[j] for j in np.linspace(0, len(rgb) - 1, 6).astype(int)], cols=6))

    # 2.-4. the surface, the upright can and the lying cans: cylinders through the SfM points
    # on the cans' printed labels (can_fit.upright_from_points); the outline search is the
    # fallback when the points do not show a can
    t1 = time.time()
    fine = can_fit.edge_data(rgb, c2w_sfm, K, count=args.fit_views)
    cams_up = can_fit._cam_up(c2w_sfm)
    group = args.fit_group == "on" or (args.fit_group == "auto" and args.lying > 0)
    mf = None
    if args.mask_fit == "on" or (args.mask_fit == "auto" and args.lying > 0):
        det = detect_can_masks(rgb, device=device, log=log)
        if det is not None:
            masks = det[0]
            views = []
            for j in np.linspace(0, len(rgb) - 1, min(9, len(rgb))).astype(int):
                img = rgb[j].copy()
                img[masks[j]] = (0.5 * img[masks[j]] + [0, 127, 0]).astype(np.uint8)
                for x0, y0, x1, y1 in det[1][j]:
                    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 3)
                views.append(img)
            debug.save_rgb(out / "01_masks.png", debug.montage(views, cols=3))
            sm = [cv2.resize(m.astype(np.float32), (round(ww * 400 / max(ww, hh)), round(hh * 400 / max(ww, hh))),
                             interpolation=cv2.INTER_AREA) >= 0.5 for m in masks]
            np.savez_compressed(out / "masks.npz", masks=np.stack(sm))  # 400 px, for checking fits offline
            prompt = sam_box_prompter(device=device, log=log)
            mf = fit_masks(masks, c2w_sfm, K, can, args.lying, cams_up, log=log, workers=-1 if (os.cpu_count() or 1) > 1 else 1,
                           instances=det[2], redetect=None if prompt is None else (lambda i, boxes, ims=rgb: prompt(ims[i], boxes)))
            if mf is not None:
                # the masks kept (explained by the fitted scene) and the cameras refined against them
                msc = mf["scene"]
                views = []
                vi = list(mf["views"])
                for j in np.linspace(0, len(rgb) - 1, min(9, len(rgb))).astype(int):
                    img = rgb[j].copy()
                    if j in vi:
                        mk = cv2.resize(msc.masks[vi.index(j)].astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST).astype(bool)
                        img[mk] = (0.5 * img[mk] + [0, 127, 0]).astype(np.uint8)
                        dropped = masks[j] & ~mk
                        img[dropped] = (0.5 * img[dropped] + [127, 0, 0]).astype(np.uint8)
                    views.append(img)
                debug.save_rgb(out / "01_masks_kept.png", debug.montage(views, cols=3))
                if mf["info"].get("cameras_refined"):
                    c2w_sfm = [np.asarray(M) for M in mf["c2w"]]
                    fine = can_fit.edge_data(rgb, c2w_sfm, K, count=args.fit_views)
    gf = None
    if mf is None and group and args.fit in ("points", "auto"):
        gf = fit_group(points, c2w_sfm, can, args.lying, log=log)
    pf = None
    if mf is None and gf is None and args.fit in ("points", "auto"):
        pf = can_fit.upright_from_points(points, c2w_sfm, can, log=log)
    if mf is not None:
        # silhouettes give the global pose (up, scale, every can); the edges then pin the rims
        # and limbs, with the silhouette term kept so the refinement stays in that basin
        sdata = can_fit.edge_data(rgb, c2w_sfm, K, count=10, long_side=1000)
        msc = mf["scene"]
        fit_cans_list, s, ofit = fit_cans(None, sdata, can, mf["b0"], mf["R0"], mf["s"], len(mf["lying"]), log=log,
                                          lying0=mf["lying"], extra=lambda poses, sc: mask_cost(msc, poses, sc),
                                          max_tilt=0.08, max_shift=0.02)
        b0, R0 = fit_cans_list[0]["b"], fit_cans_list[0]["R"]
        cans = [{"kind": c["kind"], "b": c["b"], "R": c["R"], "cost": can_fit.fit_cost(fine, can, c["b"], c["R"], s, clip=0.05)}
                for c in fit_cans_list]
        ofit["mask_iou_after"] = round(1 - mask_cost(msc, [(c["b"], c["R"]) for c in cans], s), 4)
        metrics["fit"] = "silhouettes of all cans (OWLv2 + SAM masks, global search), then edges + silhouettes"
        metrics["mask_fit"] = {"silhouettes": mf["info"], "edges_and_silhouettes": ofit}
        pf = {"plane": np.zeros((0, 3)), "inliers": np.zeros((0, 3))}
    elif gf is not None:
        # SfM points on chrome are reflections at the wrong depth, so they only start the fit:
        # the up direction and where the upright can stands. Its size and centre come from the
        # outlines (snap), then every can is fitted to the image edges together
        # (multi_can_fit: known cylinders, outlines hidden by another can not scored)
        s = gf["s"]
        b0, R0 = gf["cans"][0]["b"], gf["cans"][0]["R"]
        sdata = can_fit.edge_data(rgb, c2w_sfm, K, count=10, long_side=1000)
        b0, s, snap = can_fit.snap_to_outlines(sdata, can, b0, R0, s, log=log)
        coarse = can_fit.edge_data(rgb, c2w_sfm, K, count=8, long_side=500)
        fit_cans_list, s, ofit = fit_cans(coarse, sdata, can, b0, R0, s, args.lying, log=log)
        b0, R0 = fit_cans_list[0]["b"], fit_cans_list[0]["R"]
        cans = [{"kind": c["kind"], "b": c["b"], "R": c["R"], "cost": can_fit.fit_cost(fine, can, c["b"], c["R"], s, clip=0.05)}
                for c in fit_cans_list]
        cans[0]["outline_snap"] = snap
        metrics["fit"] = "outlines of all cans together (start: SfM points, reflection-aware)"
        metrics["group_fit"] = {"points_start": gf["info"], "outline_fit": ofit}
        pf = {"plane": np.zeros((0, 3)), "inliers": np.zeros((0, 3))}
    elif pf is not None:
        b0, R0, s = pf["b"], pf["R"], pf["s"]
        snap = {}
        if not args.no_snap:  # scale and centre checked against the outlines (reflections bias the points)
            sdata = can_fit.edge_data(rgb, c2w_sfm, K, count=10, long_side=1000)
            b0, s, snap = can_fit.snap_to_outlines(sdata, can, b0, R0, s, log=log)
        cans = [{"kind": "upright", "b": b0, "R": R0, "cost": can_fit.fit_cost(fine, can, b0, R0, s, clip=0.05),
                 "points": pf["info"], "outline_snap": snap}]
        metrics["fit"] = "sfm points" + ("" if args.no_snap else " + outline snap")
        for lc in can_fit.lying_from_points(points, can, b0, R0, s, args.lying, log=log):
            bl, Rl, lsnap = (lc["b"], lc["R"], {}) if args.no_snap else can_fit.snap_lying(sdata, can, b0, R0, s, lc["b"], lc["R"], log=log)
            cans.append({"kind": "lying", "b": bl, "R": Rl, "cost": can_fit.fit_cost(fine, can, bl, Rl, s, clip=0.05),
                         "points": lc["info"], "outline_snap": lsnap})
    else:
        metrics["fit"] = "outlines"
        up, n_plane = can_fit.ground_up(points, c2w_sfm)
        coarse = can_fit.edge_data(rgb, c2w_sfm, K, count=min(8, args.fit_views), long_side=500)
        upright = can_fit.find_upright(coarse, fine, c2w_sfm, can, up, log=log)
        b0, R0, s = upright["b"], upright["R"], upright["s"]
        cans = [{"kind": "upright", "b": b0, "R": R0, "cost": upright["cost"]}]
        avoid = [(0.0, 0.0)]
        for k in range(args.lying):
            lc = can_fit.find_lying(coarse, fine, can, b0, R0, s, avoid, log=log)
            if lc is None:
                log(f"lying can {k + 1}: not found")
                break
            avoid.append((lc["x"], lc["y"]))
            cans.append({"kind": "lying", **lc})
    np.savez_compressed(out / "sfm.npz", names=np.array([names[i] for i in reg]), c2w=np.stack(c2w_sfm), points=points,
                        K=np.array([K.width, K.height, K.fx, K.fy, K.cx, K.cy]), up=cams_up,
                        cans_b=np.stack([c["b"] for c in cans]), cans_R=np.stack([c["R"] for c in cans]), scale=s)
    if pf is not None:  # which SfM points made the fit: can label green, surface red, the rest grey
        from sodacan.capture.pose_silhouette import project_np

        views = []
        for j in np.linspace(0, len(rgb) - 1, min(6, len(rgb))).astype(int):
            img = rgb[j].copy()
            for P, col, rad in ((points, (160, 160, 160), 2), (pf["plane"], (255, 40, 40), 4), (pf["inliers"], (40, 255, 40), 4)):
                if len(P) == 0:
                    continue
                uv, zc = project_np(P, c2w_sfm[j], K)
                for (u, v), zz in zip(uv, zc):
                    if zz > 0 and 0 <= u < ww and 0 <= v < hh:
                        cv2.circle(img, (int(u), int(v)), rad, col, -1)
            views.append(img)
        debug.save_rgb(out / "01_points.png", debug.montage(views, cols=3))
    tilt_plane = math.degrees(math.acos(min(1.0, abs(float(R0[:, 2] @ cams_up)))))
    log(f"cans located by {metrics['fit']} ({time.time() - t1:.0f} s): scale {s:.4f} SfM units per m, "
        f"upright axis {tilt_plane:.1f} deg from the cameras' mean up, {len(cans) - 1} lying of {args.lying}")

    # per can: metric camera poses in its own frame
    for c in cans:
        c["c2w"] = np.stack([can_fit.camera_in_can(M, c["b"], c["R"], s) for M in c2w_sfm])
        c["outline_px"] = can_fit.median_outline_px(fine, can, c["b"], c["R"], s)
        c["in_view"] = int(sum(can_in_view(M, K, can) for M in c["c2w"]))
    cams = np.stack([can_fit.camera_in_can(M, b0, R0, s) for M in c2w_sfm])
    rho = np.hypot(cams[:, 0, 3], cams[:, 1, 3])
    metrics["scale_sfm_units_per_m"] = round(float(s), 5)
    metrics["upright_axis_vs_camera_up_deg"] = round(tilt_plane, 2)
    metrics["cameras"] = {"registered": len(reg), "of": len(files),
                          "distance_m": [round(float(rho.min()), 3), round(float(np.median(rho)), 3), round(float(rho.max()), 3)],
                          "height_m": [round(float(cams[:, 2, 3].min()), 3), round(float(cams[:, 2, 3].max()), 3)]}
    metrics["cans"] = []
    for c in cans:
        m = {"kind": c["kind"], "outline_cost": round(float(c["cost"]), 3),
             "median_outline_px": round(float(np.median(c["outline_px"])), 2) if c["outline_px"] else None,
             "in_view": f"{c['in_view']} of {len(reg)}"}
        if "points" in c:
            m["sfm_points"] = c["points"]
        if c.get("outline_snap"):
            m["outline_snap"] = c["outline_snap"]
        if c["kind"] == "lying" and "x" in c:
            m.update(xy_m=[round(c["x"], 3), round(c["y"], 3)], centre_height_mm=round(c["z"] * 1000, 1),
                     expected_height_mm=round(can.radius * 1000, 1), axis_off_surface_deg=round(c["tilt_deg"], 2))
        metrics["cans"].append(m)
    log(f"cans: {metrics['cans']}; cameras: {metrics['cameras']}")

    c2w_cans = [torch.tensor(c["c2w"], dtype=torch.float32, device=device) for c in cans]
    checks = []
    for j in np.linspace(0, len(rgb) - 1, min(9, len(rgb))).astype(int):
        img = rgb[j]
        for k, cw in enumerate(c2w_cans):
            img = debug.can_outline(img, cw[j], K, can, COLOURS[k % len(COLOURS)])
        checks.append(img)
    debug.save_rgb(out / "01_cans.png", debug.montage(checks, cols=3))

    # 5. per-can reconstruction, turned into the world frame (axes of the upright can)
    frames = linearize(torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).float().div(255), "srgb").to(device)
    del rgb
    atlases = [torch.from_numpy(np.load(p)).float().to(device) for p in (args.label_atlas or [])]
    video_label = atlases[0] if atlases else None

    def fit_any(est, count, rows):
        """The atlas (of those given) that matches this estimate best, fitted to it."""
        best = (None, {"correlation": -1.0})
        for j, a in enumerate(atlases):
            f, inf = fit_label_atlas(a, est, count, rows=rows)
            inf["atlas"] = Path(args.label_atlas[j]).parent.name
            if inf.get("correlation", -1.0) > best[1].get("correlation", -1.0):
                best = (f, inf)
        return best
    H, W = args.pano
    B = frames.shape[0]
    up_world = R0[:, 2]
    results, plain = [], []
    for k, (c, cw) in enumerate(zip(cans, c2w_cans)):
        mask = torch.ones(B, hh, ww, dtype=torch.bool, device=device)
        others = [o for j, o in enumerate(c2w_cans) if j != k]
        refl_px = can_px = 0
        for i in range(B):
            gk = can_geometry(cw[i : i + 1], K, can)
            for other in others:  # hidden behind another can
                gj = can_geometry(other[i : i + 1], K, can)
                mask[i] &= ~(gk.hit & gj.hit & (gj.t < gk.t))[0]
            if others and not args.keep_can_reflections:  # showing another can in its mirror
                rf = reflects_other_can(gk, cw[i : i + 1], [o[i : i + 1] for o in others], can)[0]
                refl_px += int((rf & mask[i]).sum())
                can_px += int((gk.hit[0] & mask[i]).sum())
                mask[i] &= ~rf
        hidden = 1 - float(mask.float().mean())
        if can_px:
            metrics["cans"][k]["reflects_other_cans"] = round(refl_px / can_px, 4)
            log(f"can {k}: {refl_px / can_px:.1%} of its visible pixels reflect another can (left out)")
        up_k = (c["R"].T @ up_world).tolist()
        photographer = None if args.no_photographer else {"az_deg": 25, "el_below_deg": 60, "el_above_deg": 15,
                                                          "weight": 0.02, "up": up_k}
        cfg = load_pipeline(pano=[H, W], photographer=photographer)
        pm = mask if len(cans) > 1 else None
        r = reconstruct(frames, cw, K, can, cfg, pixel_mask=pm)
        label, est = r.label, r.label
        if video_label is not None and r.label is not None:
            z_rows = (round((1 - can.usable_z[1] / can.height) * r.label.shape[1]),
                      round((1 - can.usable_z[0] / can.height) * r.label.shape[1]))
        if video_label is not None and r.label is not None and c["kind"] == "lying":
            # a lying can's silhouette and outline barely say which end is the lid: its label
            # does. Turned end over end, the can's label estimate is the same one flipped
            # upside down and with its angle reversed (turning it 180 deg about an axis across
            # it flips the can's own y and z: theta -> -theta).
            _, i1 = fit_any(r.label, r.label_count, z_rows)
            _, i2 = fit_any(r.label.flip(1).flip(2), r.label_count.flip(0).flip(1),
                            (r.label.shape[1] - z_rows[1], r.label.shape[1] - z_rows[0]))
            c1, c2 = i1.get("correlation", -1.0), i2.get("correlation", -1.0)
            flip = c2 > c1 + args.label_flip_margin
            metrics["cans"][k]["end_over_end"] = {"label_corr_as_fitted": round(c1, 3), "label_corr_turned": round(c2, 3),
                                                  "turned": bool(flip)}
            log(f"can {k}: label correlation as fitted {c1:.3f}, turned end over end {c2:.3f}"
                + (" -> turned" if flip else ""))
            if flip:
                c["b"] = c["b"] + s * can.height * c["R"][:, 2]
                c["R"] = c["R"] @ np.diag([1.0, -1.0, -1.0])
                c["c2w"] = np.stack([can_fit.camera_in_can(M, c["b"], c["R"], s) for M in c2w_sfm])
                cw = torch.tensor(c["c2w"], dtype=torch.float32, device=device)
                c2w_cans[k] = cw
                r = reconstruct(frames, cw, K, can, cfg, pixel_mask=pm)
                label, est = r.label, r.label
        if video_label is not None and r.label is not None:
            # the label from the video, turned and scaled to this scene's own estimate
            fitted, linfo = fit_any(r.label, r.label_count, z_rows)
            metrics["cans"][k]["video_label"] = linfo
            log(f"can {k}: video label {linfo}")
            if fitted is not None and linfo["correlation"] >= args.label_min_corr:
                label = fitted
                r = reconstruct(frames, cw, K, can, cfg, label=label, pixel_mask=pm)
        rp = reconstruct(frames, cw, K, can, load_pipeline(pano=[H, W]), label=label, pixel_mask=pm)
        M = c["R"].T @ R0  # world (upright-can axes) direction -> this can's frame
        results.append(to_world(r, M, H, W))
        plain.append(to_world(rp, M, H, W))
        if est is not None:
            if label is not est:
                debug.save_rgb(out / f"04_label_can{k}.png", debug.stack(
                    [debug.display(est), debug.display(label)], ["label estimated from these photos", "label from the video, fitted (used)"]))
            else:
                debug.save_rgb(out / f"04_label_can{k}.png", debug.display(est))
        cov = coverage_stats(results[-1][2])
        metrics["cans"][k]["coverage"] = cov
        log(f"can {k} ({c['kind']}): coverage {cov}, {hidden:.1%} of pixels hidden by another can")

    np.savez_compressed(out / "cans_final.npz", cans_b=np.stack([c["b"] for c in cans]),
                        cans_R=np.stack([c["R"] for c in cans]), scale=s, kinds=np.array([c["kind"] for c in cans]))

    def merge(rs):
        num = sum(p * wt[None] for p, wt, _ in rs)
        den = sum(wt for _, wt, _ in rs)
        valid = torch.stack([v for _, _, v in rs]).any(0)
        return num / den.clamp_min(1e-12)[None] * valid[None], den, valid

    pano, weight, valid = merge(results)
    pano0, _, valid0 = merge(plain)
    exp = auto_exposure(pano, valid)
    panels, labels = [], []
    for k, (p, _, v) in enumerate(results):
        panels.append(debug.display(p, v, exp))
        labels.append(f"can {k} ({cans[k]['kind']})")
    if len(results) > 1:
        panels.append(debug.display(pano, valid, exp))
        labels.append("all cans merged")
    panels.append(debug.display(pano0, valid0, exp))
    labels.append("merged, photographer NOT down-weighted (for comparison)")
    debug.save_rgb(out / "05_panorama.png", debug.stack(panels, labels))
    debug.save_rgb(out / "05_coverage.png", debug.heat(weight, valid))
    metrics["coverage"] = coverage_stats(valid)
    log(f"merged coverage {metrics['coverage']}")

    ref_path = args.reference
    if ref_path is None:
        cands = sorted(q for d in Path(args.photos).iterdir() if d.is_dir() and "truth" in d.name.lower().replace("grouth", "truth")
                       for q in glob.glob(str(d / "*.jpg")))  # Ground Truth, ground_truth, Grouth Truth
        ref_path = cands[0] if cands else None
    saved = {"pano": pano.cpu().numpy(), "weight": weight.cpu().numpy(), "valid": valid.cpu().numpy()}
    if ref_path:
        from sodacan.capture.load import load_reference
        from sodacan.metrics import align_azimuth, all_metrics

        reference = load_reference(ref_path, H, W).to(device)
        reference, shift, flip = align_azimuth(pano, reference, valid)
        metrics["reference"] = {"file": Path(ref_path).name, "shift_columns": shift, "mirrored": flip,
                                "measured": all_metrics(pano, reference, valid, with_lpips=False)}
        ra = auto_exposure(reference, valid)
        debug.save_rgb(out / "05_reference.png", debug.stack(
            [debug.display(reference, None, ra), debug.display(reference, valid, ra), debug.display(pano, valid, exp)],
            ["reference (aligned)", "reference where the cans saw", "measured"]))
        saved["reference"] = reference.cpu().numpy()
        log(f"reference: {metrics['reference']}")
    np.savez_compressed(out / "panorama.npz", **saved)
    metrics["seconds"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
