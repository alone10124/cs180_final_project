"""The can's printed label, unwrapped from a video of the can turning in front of the camera.

The camera is still (hand-held) and a hand turns the can on the table. Per frame:
  1. the can's position: the lid is template-matched against the first frame (the lid looks
     the same at any rotation; the label does not), giving the can's shift in the image;
     --box gives the can in the first frame (left, top, right, bottom of the body in px at
     --work-width),
  2. the central +-theta_max of the can is unwrapped onto (angle, height): x = cx + R sin(theta),
     with the small sag of the circles seen from slightly above,
  3. the rotation: SIFT matches between unwrapped strips 1-6 frames apart give the label's
     shift in angle (matches that do not move are reflections, fixed to the view, and are
     dropped); all frame angles by weighted least squares, as the can turns in bursts and
     blurred frames match poorly,
  4. the turn is closed: a late frame is matched against the first and tied to --turns
     whole turns. Frames where nothing matched (the can turned fast and blurred) are not
     assumed still: the solve prefers a steady turning speed there, so the turn missing from
     the matches goes into those gaps rather than being spread over the whole video (which
     smeared the logo twice over),
  5. every frame is then re-aligned against the mosaic of all the others (correlation of the
     print, a few columns either way), twice, and every texel of the label atlas takes a low
     percentile of the frames that saw it near face-on: the label plus the darkest reflection,
     as the pipeline's own label estimate (sodacan.separate.label_median) is, but over the
     full 360 deg.
Writes into --out: label_atlas.npy (3, rows, cols) linear RGB in the pipeline's atlas layout
(row 0 = the top of the can, column 0 = angle -pi), label_atlas.png (sRGB), and
00_track.png / 01_strips.png / 02_rotation.png for checking.

Use it in a run with run_noboard.py --label-atlas <out>/label_atlas.npy: per can the atlas
is turned (and mirrored if needed) and its brightness fitted to that scene's own label
estimate before it is subtracted.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def read_video(path, step, width):
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            h = round(f.shape[0] * width / f.shape[1])
            frames.append(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA))
        i += 1
    return np.stack(frames)


def edges(f):
    g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    return cv2.GaussianBlur(cv2.Canny(g, 40, 120).astype(np.float32), (7, 7), 2)


def track_lid(frames, box, lid_rows=55, search=120):
    """Image shift (dx, dy) of the can in every frame vs the first, from its lid."""
    l, t, r, b = box
    E0 = edges(frames[0])
    a0, a1 = t - 10, t + lid_rows - 10
    x0, x1 = l - 15, r + 15
    T = E0[a0:a1, x0:x1]
    out = []
    for f in frames:
        E = edges(f)
        ya, yb = max(0, a0 - search), min(E.shape[0], a1 + search)
        res = cv2.matchTemplate(E[ya:yb], T, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        out.append((ml[0] - x0, ml[1] + ya - a0, mv))
    out = np.array(out, float)
    from scipy.ndimage import median_filter

    out[:, :2] = median_filter(out[:, :2], size=(5, 1), mode="nearest")
    return out


def unwrap(frame, cx, R, y_top, y_bot, sag, thetas, rows):
    """frame (H, W, 3) -> strip (rows, len(thetas), 3): row 0 = the top of the can."""
    th, hh = np.meshgrid(thetas, (np.arange(rows) + 0.5) / rows, indexing="xy")
    x = cx + R * np.sin(th)
    y = y_top + hh * (y_bot - y_top) + sag * (1 - np.cos(th))  # the circles bow towards the camera
    return cv2.remap(frame, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)


def sift_shift(a, b, sift, matcher, max_dy=3.0, min_shift=0.8):
    """Median column shift s with b[:, j] ~ a[:, j + s], from SIFT matches between two strips
    (printed text gives sharp features, the smooth reflections few). Returns (s, matches)."""
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(ka) < 5 or len(kb) < 5:
        return None, 0
    pairs = matcher.knnMatch(db, da, k=2)
    d = []
    for p in pairs:
        if len(p) == 2 and p[0].distance < 0.75 * p[1].distance:
            pa, pb = ka[p[0].trainIdx].pt, kb[p[0].queryIdx].pt
            if abs(pa[1] - pb[1]) < max_dy:
                d.append(pa[0] - pb[0])
    if len(d) < 4:
        return None, len(d)
    d = np.array(d)
    d = d[np.abs(d) >= min_shift]  # reflections are fixed to the view: they match at ~0
    if len(d) < 4:
        return None, len(d)
    hist, edges = np.histogram(d, np.arange(d.min() - 0.5, d.max() + 1.5, 1.0))
    m = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    d = d[np.abs(d - m) < 1.5]
    if len(d) < 4:
        return None, len(d)
    return float(np.median(d)), len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", type=float, nargs=4, required=True,
                    help="the can body in the first frame at --work-width: left top right bottom (px); "
                         "top = the top of the lid, bottom = the front of the base")
    ap.add_argument("--box-width", type=int, default=540, help="frame width the --box is given at")
    ap.add_argument("--work-width", type=int, default=1080)
    ap.add_argument("--match-scale", type=int, default=4, help="strips for SIFT matching at this many times the atlas resolution")
    ap.add_argument("--match-rows", type=float, default=0.4, help="match on the strip below this fraction of the can height (from the top)")
    ap.add_argument("--step", type=int, default=2, help="use every step-th frame")
    ap.add_argument("--sag", type=float, default=0.09, help="half height of the circles seen from above, in can radii")
    ap.add_argument("--theta-max", type=float, default=50.0, help="unwrap +- this many degrees around the front")
    ap.add_argument("--atlas", type=int, nargs=2, default=[192, 512], help="rows (height) cols (angle)")
    ap.add_argument("--percentile", type=float, default=25.0)
    ap.add_argument("--turns", type=int, default=1, help="whole turns the can makes between the first frame and the closing one")
    ap.add_argument("--smooth", type=float, default=0.3, help="weight of the steady-speed prior (per frame, in columns)")
    ap.add_argument("--refine-iters", type=int, default=4, help="passes re-aligning every frame against the mosaic")
    ap.add_argument("--refine-search", type=int, default=20, help="columns either way a frame may move per pass")
    ap.add_argument("--turn-scale", type=float, nargs=2, default=[0.85, 1.4], help="range searched for the turn's total length (x the matched one)")
    ap.add_argument("--turn-scale-search", action="store_true", help="search the turn's length even when the turn was closed")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames = read_video(args.video, args.step, args.work_width)
    l, t, r, b = (v * args.work_width / args.box_width for v in args.box)
    R = (r - l) / 2
    k = args.work_width / args.box_width  # track the lid at the box's size (its constants are in those px)
    small = [cv2.resize(f, (args.box_width, round(f.shape[0] / k)), interpolation=cv2.INTER_AREA) for f in frames]
    track = track_lid(small, tuple(int(v) for v in args.box))
    track[:, :2] *= k
    del small
    rows, cols = args.atlas
    tm = math.radians(args.theta_max)
    dth = 2 * math.pi / cols  # one atlas column
    thetas = np.arange(-tm, tm + 1e-9, dth)
    sag = args.sag * R
    strips = [unwrap(f, (l + r) / 2 + dx, R, t + dy + sag, b + dy - sag, sag, thetas, rows)
              for f, (dx, dy, _) in zip(frames, track)]
    ms = args.match_scale  # finer strips for matching: the label's small print gives SIFT features
    thetas_m = np.arange(-tm, tm + 1e-9, dth / ms)
    mstrips = [unwrap(f, (l + r) / 2 + dx, R, t + dy + sag, b + dy - sag, sag, thetas_m, rows * ms)
               for f, (dx, dy, _) in zip(frames, track)]
    # match on the lower part of the can only: the hand turning it, and its reflection, are at the top
    r0 = int(args.match_rows * rows * ms)
    mstrips = [np.ascontiguousarray(s[r0:]) for s in mstrips]

    # 3. rotation between consecutive frames (columns of the atlas)
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    # pairs over several frame gaps, all frame angles by weighted least squares (with a weak
    # pull towards no motion, for stretches where nothing matched)
    n = len(strips)
    rows_eq, rhs, wts = [], [], []
    scores = np.zeros(n)
    for gap in (1, 2, 3, 4, 6):
        for i in range(n - gap):
            s, m = sift_shift(mstrips[i], mstrips[i + gap], sift, matcher, max_dy=3.0 * ms, min_shift=0.8 * ms)
            if s is None:
                continue
            s /= ms
            rows_eq.append((i, i + gap)); rhs.append(s); wts.append(math.sqrt(m))
            if gap == 1:
                scores[i + 1] = m
    # the can turns one way: equations against the majority direction are reflections
    rhs_a = np.array(rhs)
    sign = np.sign(np.sum(np.sign(rhs_a) * np.array(wts)))
    keep_eq = np.sign(rhs_a) == sign
    rows_eq = [e for e, k_ in zip(rows_eq, keep_eq) if k_]
    rhs = list(rhs_a[keep_eq])
    wts = list(np.array(wts)[keep_eq])
    n_dir = int(keep_eq.sum())
    # 4. the closing frame: of the last quarter of the frames, the one matching the first best
    turn = None
    for i in range(3 * n // 4, n):
        sh, m = sift_shift(mstrips[0], mstrips[i], sift, matcher, max_dy=3.0 * ms, min_shift=0.0)
        if sh is not None and (turn is None or m > turn[0]):
            turn = (m, i, sh / ms)
    eq = [(("d", i_, j_), r_, w_) for (i_, j_), r_, w_ in zip(rows_eq, rhs, wts)]
    for i_ in range(1, n - 1):  # steady speed where nothing else says otherwise
        eq.append((("dd", i_, 0), 0.0, args.smooth))
    eq.append((("abs", 0, 0), 0.0, 100.0))  # phi_0 = 0
    if turn is not None:
        m, ic, sh = turn
        eq.append((("abs", ic, 0), sign * args.turns * cols + sh, 20.0))  # phi_i - phi_0 = sh + whole turns
    A = np.zeros((len(eq), n))
    y = np.zeros(len(eq))
    for k_, ((kind, a_, b_), r_, w_) in enumerate(eq):
        if kind == "d":
            A[k_, a_], A[k_, b_] = -w_, w_
        elif kind == "dd":
            A[k_, a_ - 1], A[k_, a_], A[k_, a_ + 1] = w_, -2 * w_, w_
        else:
            A[k_, a_] = w_
        y[k_] = r_ * w_
    wv = np.array([e[2] for e in eq])
    is_data = np.array([e[0][0] == "d" for e in eq])
    keep = np.ones(len(y), bool)
    for _ in range(3):  # robust: refit without the worst matches
        phi = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        res = np.abs(A @ phi - y) / wv
        keep = ~is_data | (res < max(2.0, 3 * np.median(res[is_data]) if is_data.any() else 2.0))
    info = {"frames": n, "lid_match": [round(float(np.percentile(track[:, 2], 10)), 3), round(float(np.median(track[:, 2])), 3)],
            "equations": int(keep[is_data].sum()), "equations_in_turn_direction": n_dir,
            "turn_deg": round(float(phi[-1] * dth * 180 / math.pi), 1)}
    if turn is not None:
        info.update(closure_frame=int(turn[1]), closure_matches=int(turn[0]), turns=args.turns,
                    closure_residual_columns=round(float(phi[turn[1]] - (sign * args.turns * cols + turn[2])), 2))

    # 5. every frame re-aligned against the mosaic of all the OTHER frames (its own
    # reflections, fixed to the view, would otherwise pull it to where it already is), on the
    # print's edges and its colour (the logo's red and blue). First the turn's total length:
    # the angles scaled until the frames agree best (SIFT misses fast turns, so the chain
    # comes out short), then each frame within +-search columns, smoothed by the same
    # steady-speed prior.
    j0 = cols // 2 - len(thetas) // 2  # frame 0's front faces angle 0 (column cols/2)
    face = np.cos(thetas) > math.cos(math.radians(args.theta_max - 5))
    jf = np.nonzero(face)[0]
    rlo = int(args.match_rows * rows)

    def feats(st):
        f = st[rlo:, jf].astype(np.float32) / 255
        g = cv2.cvtColor(st, cv2.COLOR_BGR2GRAY).astype(np.float32)[rlo:, jf] / 255
        g = g - cv2.blur(g, (9, 9))
        sm = f.sum(2) + 0.1 * f.sum(2).mean() + 1e-4
        ch = [(f[..., 2] - f[..., 1]) / sm, (f[..., 0] - f[..., 1]) / sm]  # BGR: r - g, b - g
        out = np.stack([g] + ch)
        out = out - out.mean((1, 2), keepdims=True)
        return out / (out.std((1, 2), keepdims=True) + 1e-6)

    F_ = np.stack([feats(st) for st in strips])  # (n, 3, rows', len(jf))

    def mosaic(ph):
        acc = np.zeros(F_.shape[1:3] + (cols,), np.float32)
        cnt = np.zeros(cols, np.float32)
        cs = (np.round(j0 + jf[None] + ph[:, None]).astype(int)) % cols
        for i_ in range(n):
            np.add.at(acc.transpose(2, 0, 1), cs[i_], F_[i_].transpose(2, 0, 1))
            np.add.at(cnt, cs[i_], 1)
        return acc, cnt, cs

    def corr_with_others(i_, acc, cnt, cs, ph_i):
        """Correlation of frame i with the mosaic of the others, frame placed at ph_i."""
        own = np.zeros_like(acc)
        own_c = np.zeros(cols, np.float32)
        np.add.at(own.transpose(2, 0, 1), cs[i_], F_[i_].transpose(2, 0, 1))
        np.add.at(own_c, cs[i_], 1)
        c_ = (np.round(j0 + jf + ph_i).astype(int)) % cols
        cnt_o = (cnt - own_c)[c_]
        ok = cnt_o > 0
        if ok.sum() < 0.3 * len(jf):
            return None
        mo = ((acc - own)[..., c_] / np.maximum(cnt_o, 1))[..., ok]
        a_ = F_[i_][..., ok]
        a_, mo = a_ - a_.mean(), mo - mo.mean()
        return float((a_ * mo).sum() / (np.linalg.norm(a_) * np.linalg.norm(mo) + 1e-9))

    def agreement(ph):
        acc, cnt, cs = mosaic(ph)
        v = [corr_with_others(i_, acc, cnt, cs, ph[i_]) for i_ in range(0, n, 2)]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else -1.0

    if turn is None or args.turn_scale_search:
        scales = np.linspace(args.turn_scale[0], args.turn_scale[1], 31)
        sc_scores = [agreement(phi * k) for k in scales]
        kbest = float(scales[int(np.argmax(sc_scores))])
        phi = phi * kbest
        info["turn_scale"] = round(kbest, 3)
        info["turn_deg_scaled"] = round(float(phi[-1] * dth * 180 / math.pi), 1)
        info["turn_scale_agreement"] = [round(min(sc_scores), 3), round(max(sc_scores), 3)]
    moves, agree = [], [round(agreement(phi), 3)]
    for it in range(args.refine_iters):
        acc, cnt, cs = mosaic(phi)
        meas = phi.copy()
        conf = np.zeros(n)
        for i_ in range(n):
            best = (-2.0, 0)
            for d in range(-args.refine_search, args.refine_search + 1):
                c = corr_with_others(i_, acc, cnt, cs, phi[i_] + d)
                if c is not None and c > best[0]:
                    best = (c, d)
            meas[i_], conf[i_] = phi[i_] + best[1], max(0.0, best[0])
        # the re-aligned angles, smoothed by the same steady-speed prior (phi_0 kept)
        wv_ = conf * 10 + 1e-3
        A2 = [np.diag(wv_)]
        y2 = [meas * wv_]
        for i_ in range(1, n - 1):
            r_ = np.zeros(n)
            r_[i_ - 1], r_[i_], r_[i_ + 1] = args.smooth, -2 * args.smooth, args.smooth
            A2.append(r_[None]); y2.append([0.0])
        new = np.linalg.lstsq(np.vstack(A2), np.concatenate(y2), rcond=None)[0]
        moves.append(round(float(np.median(np.abs(new - phi))), 2))
        phi = new - new[0]
        agree.append(round(agreement(phi), 3))
    info["refine_median_move_columns"] = moves
    info["frame_agreement"] = agree
    info["turn_deg_final"] = round(float(phi[-1] * dth * 180 / math.pi), 1)

    # 6. composite: frame i column j shows the label at atlas column (j0 + j + phi_i)
    stack = [[] for _ in range(cols)]
    lin = [srgb_to_linear(cv2.cvtColor(s, cv2.COLOR_BGR2RGB).astype(np.float32) / 255) for s in strips]
    for i, S in enumerate(lin):
        for j in np.nonzero(face)[0]:
            c = int(round(j0 + j + phi[i])) % cols
            stack[c].append(S[:, j])
    atlas = np.zeros((rows, cols, 3), np.float32)
    count = np.zeros(cols, int)
    for c in range(cols):
        if stack[c]:
            atlas[:, c] = np.percentile(np.stack(stack[c]), args.percentile, axis=0)
            count[c] = len(stack[c])
    info["columns_seen"] = int((count > 0).sum())
    info["samples_per_column"] = [int(count.min()), int(np.median(count)), int(count.max())]
    np.save(out / "label_atlas.npy", atlas.transpose(2, 0, 1))
    png = (linear_to_srgb(atlas) * 255).round().astype(np.uint8)
    cv2.imwrite(str(out / "label_atlas.png"), cv2.cvtColor(png, cv2.COLOR_RGB2BGR))
    # checks
    vis = []
    for i in np.linspace(0, len(frames) - 1, 6).astype(int):
        im = frames[i].copy()
        dx, dy, _ = track[i]
        cv2.rectangle(im, (int(l + dx), int(t + dy)), (int(r + dx), int(b + dy)), (0, 255, 0), 2)
        vis.append(im)
    cv2.imwrite(str(out / "00_track.png"), np.hstack(vis))
    cv2.imwrite(str(out / "01_strips.png"), np.hstack([strips[i] for i in np.linspace(0, len(strips) - 1, 8).astype(int)]))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(np.degrees(phi * dth)); ax[0].set_xlabel("frame"); ax[0].set_ylabel("label rotation (deg)")
    ax[1].plot(scores); ax[1].set_xlabel("frame"); ax[1].set_ylabel("SIFT matches")
    fig.tight_layout()
    fig.savefig(out / "02_rotation.png", dpi=80)
    (out / "label_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
