"""The can's label, all the way round, from one picture of several identical cans turned
different ways (a 3D render, or a photo of cans on a table).

Per can: the straight part of its wall is given as four image points (the left and right
edge at the top and at the bottom of the straight wall). A homography takes that quadrilateral
to an upright rectangle (it removes the can's lean and the keystone of the perspective), then
the rectangle is unwrapped as a cylinder: a point at angle theta from the front and height h
is at x = cx + R sin(theta), y = top + h (bottom - top) + sag(h) (1 - cos(theta)), the sag being
how far the front of the circles bows down, seen from above (half the height of the ellipse,
in can radii, at the top and at the bottom). Each can gives about +-theta_max around its front.

The strips are then placed around the can by matching where they overlap (the print's edges
and its colour), and every atlas texel takes the weighted median of the strips that see it,
weighted towards each strip's front (the sides are foreshortened and catch the highlights).

Writes label_atlas.npy (3, rows, cols) linear RGB in the pipeline's layout (row 0 = the top of
the can, column 0 = angle -pi), label_atlas.png and 00_strips.png.
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


def unwrap_can(img, quad, rows, cols, theta_max, sag_top, sag_bot, rect_w=600):
    """quad: [[x, y] top-left, top-right, bottom-right, bottom-left] of the straight wall.
    Returns (strip (rows, n, 3) float sRGB 0..1, thetas (n,))."""
    q = np.asarray(quad, np.float32)
    top_w = np.linalg.norm(q[1] - q[0])
    bot_w = np.linalg.norm(q[2] - q[3])
    hgt = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    rect_h = round(rect_w * hgt / (0.5 * (top_w + bot_w)))
    dst = np.float32([[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]])
    Hm = cv2.getPerspectiveTransform(q, dst)
    rect = cv2.warpPerspective(img, Hm, (rect_w, rect_h + round(0.4 * rect_w)), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    R = rect_w / 2
    dth = 2 * math.pi / cols
    tm = math.radians(theta_max)
    thetas = np.arange(-tm, tm + 1e-9, dth)
    th, hh = np.meshgrid(thetas, (np.arange(rows) + 0.5) / rows, indexing="xy")
    sag = (sag_top + (sag_bot - sag_top) * hh) * R
    x = R + R * np.sin(th)
    y = hh * rect_h + sag * (1 - np.cos(th))
    strip = cv2.remap(rect, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)
    return strip.astype(np.float32) / 255, thetas, rect


def feats(strip):
    """(3, rows, n): high-passed gray and the two chromaticities, normalized (BGR input)."""
    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    g = g - cv2.blur(g, (9, 9))
    sm = strip.sum(2) + 0.1 * strip.sum(2).mean() + 1e-4
    ch = [(strip[..., 2] - strip[..., 1]) / sm, (strip[..., 0] - strip[..., 1]) / sm]
    # all high-passed: the smooth shading and the wall behind would otherwise match anywhere
    f = np.stack([g] + [c - cv2.blur(c, (15, 15)) for c in ch])
    f = f - f.mean((1, 2), keepdims=True)
    return f / (f.std((1, 2), keepdims=True) + 1e-6)


def overlap_corr(fa, oa, fb, ob, cols, min_overlap):
    """Correlation of strips a (at column offset oa) and b (at ob) where both see the atlas."""
    na, nb = fa.shape[2], fb.shape[2]
    ca = (oa + np.arange(na)) % cols
    cb = (ob + np.arange(nb)) % cols
    common, ia, ib = np.intersect1d(ca, cb, return_indices=True)
    if len(common) < min_overlap:
        return None, len(common)
    a, b = fa[..., ia], fb[..., ib]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), len(common)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--quads", required=True, help="JSON list of quads [[x,y] tl, tr, br, bl], one per can")
    ap.add_argument("--out", required=True)
    ap.add_argument("--atlas", type=int, nargs=2, default=[192, 512])
    ap.add_argument("--theta-max", type=float, default=70.0)
    ap.add_argument("--sag", type=float, nargs=2, default=[0.2, 0.26], help="ellipse half-height / radius at the top and bottom")
    ap.add_argument("--min-matches", type=int, default=8, help="SIFT matches that link two strips")
    ap.add_argument("--link", type=float, nargs=3, action="append", metavar=("A", "B", "COLUMNS"),
                    help="strip B's front is COLUMNS after strip A's (measured by hand, for strips that share little print)")
    ap.add_argument("--sharpness", type=float, default=60.0, help="power on the face-on weight (higher: less blending)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, cols = args.atlas
    img = cv2.imread(args.image)
    quads = json.loads(Path(args.quads).read_text()) if Path(args.quads).exists() else json.loads(args.quads)
    # an entry is a quad, or {"quad": quad, "theta": [lo, hi]}: the angles (deg from the can's
    # front) this can's strip may give to the atlas (a can seen at a steep angle gives only the
    # side it shows best; it still helps place the others)
    use = [np.radians(q["theta"]) if isinstance(q, dict) and "theta" in q else None for q in quads]
    quads = [q["quad"] if isinstance(q, dict) else q for q in quads]
    strips, thetas = [], None
    for q in quads:
        s, thetas, _ = unwrap_can(img, q, rows, cols, args.theta_max, *args.sag)
        strips.append(s)
    n = len(thetas)
    w_front = np.cos(thetas) ** 2  # weight towards each strip's front

    # placement: SIFT matches between every pair of strips (strips at 3x for the small print)
    # give the column offset and the small vertical shift between them (the quads are clicked
    # by hand); pairs that share too little print can be linked by hand (--link a b columns,
    # read off 00_strips.png). The strips are joined along the strongest links (a maximum
    # spanning tree), the first strip's front at angle 0 (column cols/2). No mirroring: a label
    # is never mirrored on a real can.
    up = 3
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    kd = []
    for q in quads:
        s3, _, _ = unwrap_can(img, q, rows * up, cols * up, args.theta_max, *args.sag)
        kd.append(sift.detectAndCompute((s3 * 255).astype(np.uint8), None))
    K = len(strips)
    links = {}
    for a in range(K):
        for b in range(a + 1, K):
            (ka, da), (kb, db) = kd[a], kd[b]
            if da is None or db is None:
                continue
            d = []
            for pr in bf.knnMatch(da, db, k=2):
                if len(pr) == 2 and pr[0].distance < 0.7 * pr[1].distance:
                    pa, pb = ka[pr[0].queryIdx].pt, kb[pr[0].trainIdx].pt
                    d.append((pa[0] - pb[0], pa[1] - pb[1], pa[1], pb[1]))
            if len(d) < 4:
                continue
            d = np.array(d) / up
            h, e = np.histogram(d[:, 0], np.arange(-cols, cols, 2.0))
            m = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
            inl = d[np.abs(d[:, 0] - m) < 3]
            if len(inl) >= args.min_matches:
                # feature at column xa in a and xb in b: off_b = off_a + (xa - xb); rows
                # yb = alpha ya + beta (the strips' heights differ a little)
                # a shift only: a fitted height scale follows the foreshortened sides, where
                # the hand-clicked quads are least accurate
                alpha = 1.0
                ya, yb = inl[:, 2], inl[:, 3]
                beta = float(np.median(yb - alpha * ya))
                links[(a, b)] = (float(len(inl)), float(np.median(inl[:, 0])), (float(alpha), float(beta)), "sift")
    for a, b, dc in args.link or []:
        a, b = int(a), int(b)
        links[(a, b)] = (float(args.min_matches), float(dc), (1.0, 0.0), "by hand")
    # each strip's rows in the first strip's rows: y0 = S y + T
    offs, vmap = {0: float(cols // 2 - n // 2)}, {0: (1.0, 0.0)}
    info = {"cans": K, "links": {f"{a}-{b}": [v[0], round(v[1], 1), [round(v[2][0], 3), round(v[2][1], 1)], v[3]]
                                 for (a, b), v in links.items()}, "tree": []}
    while len(offs) < K:
        cand = [(w, a, b, dc, ab) for (a, b), (w, dc, ab, _) in links.items() if (a in offs) != (b in offs)]
        if not cand:
            break
        w, a, b, dc, (al, be) = max(cand)
        if a in offs:  # yb = al ya + be  ->  y0 = Sa (yb - be) / al + Ta
            Sa, Ta = vmap[a]
            offs[b], vmap[b] = offs[a] + dc, (Sa / al, Ta - Sa * be / al)
        else:  # ya = (yb - be) / al, and y0 = Sb yb + Tb  ->  Sb = Sa al, Tb = Ta - Sb be ... from a's side
            Sb, Tb = vmap[b]
            offs[a], vmap[a] = offs[b] - dc, (Sb * al, Tb + Sb * be)
        info["tree"].append([a, b])
    offs = {k: int(round(v)) % cols for k, v in offs.items()}
    info["offsets"] = {str(k): [offs[k], [round(vmap[k][0], 3), round(vmap[k][1], 1)]] for k in offs}

    def to_root_rows(img_, k):
        """strip k resampled so its rows are the first strip's rows."""
        S_, T_ = vmap[k]
        y = ((np.arange(rows) - T_) / S_).astype(np.float32)
        mx = np.tile(np.arange(img_.shape[1], dtype=np.float32), (rows, 1))
        my = np.tile(y[:, None], (1, img_.shape[1]))
        return cv2.remap(img_, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # composite: each texel mostly from the strip that saw it nearest face-on (weights raised
    # to a high power: a blend of two strips a column off ghosts the print), highlights down-weighted
    lin = [to_root_rows(srgb_to_linear(cv2.cvtColor(s, cv2.COLOR_BGR2RGB)).astype(np.float32), k) for k, s in enumerate(strips)]
    acc = np.zeros((rows, cols, 3))
    wsum = np.zeros((rows, cols))
    cover = np.zeros(cols, int)
    for k, o in offs.items():
        c_ = (o + np.arange(n)) % cols
        L = lin[k].mean(2)
        med = cv2.medianBlur((np.clip(L, 0, 1) * 255).astype(np.uint8), 21).astype(np.float32) / 255
        hl = np.exp(-np.maximum(0, L - med - 0.05) / 0.05)
        wk = (w_front[None].astype(np.float64) ** args.sharpness) * hl + 1e-200
        if use[k] is not None:
            wk = wk * ((thetas >= use[k][0]) & (thetas <= use[k][1]))[None]
        np.add.at(acc.transpose(1, 0, 2), c_, (lin[k] * wk[..., None]).transpose(1, 0, 2))
        np.add.at(wsum.T, c_, wk.T)
        np.add.at(cover, c_, (wk.max(0) > 0).astype(int))
    atlas = (acc / np.maximum(wsum, 1e-300)[..., None]).astype(np.float32)
    seen = cover > 0
    info["columns_seen"] = int(seen.sum())
    # columns no can showed: blended linearly between the seen columns either side (around
    # the circle), so a label made from this atlas is never black there
    if (~seen).any() and seen.any():
        idx = np.flatnonzero(seen)
        for c in np.flatnonzero(~seen):
            r = idx[np.searchsorted(idx, c) % len(idx)]
            l = idx[np.searchsorted(idx, c) - 1]
            dl, dr = (c - l) % cols, (r - c) % cols
            atlas[:, c] = (atlas[:, l] * dr + atlas[:, r] * dl) / max(1, dl + dr)
    info["columns_filled"] = int((~seen).sum())
    np.save(out / "label_atlas.npy", atlas.transpose(2, 0, 1).astype(np.float32))
    np.save(out / "label_count.npy", (wsum > 0.05).astype(np.float32) * 5)
    png = (linear_to_srgb(atlas) * 255).round().astype(np.uint8)
    png[:, ~seen] = (40, 30, 90)
    cv2.imwrite(str(out / "label_atlas.png"), cv2.cvtColor(png, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "00_strips.png"), np.hstack([(s * 255).astype(np.uint8) for s in strips]))
    (out / "label_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
