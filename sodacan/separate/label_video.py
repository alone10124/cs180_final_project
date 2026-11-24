"""A label atlas made once from a video of the can turning (scripts/label_from_video.py),
fitted to one scene.

The scene's own label estimate (a low percentile of what each texel showed, sodacan.
separate.label_median) is noisy and has gaps where the photos did not see the can face-on,
and where other cans hid it. The video atlas covers the whole turn, but its angle origin,
its handedness and its brightness (exposure, room light) are the video's. Here: the circular
shift in angle (and mirroring) that best correlates it with the scene's estimate, over the
texels the scene saw often enough (with a stretch and shift in height: the two bands are cut
at slightly different heights), then a per-channel gain and offset by least squares.
"""
import torch
import torch.nn.functional as F


def _gray(x):
    return x.mean(0)


def _vwarp(video, scale, offset):
    """video (C, H, W) resampled in height: out row y shows video row c + (y - c) * scale + offset
    (c the middle row), so the print can be shifted and stretched against the estimate."""
    C, H, W = video.shape
    y = torch.arange(H, dtype=video.dtype, device=video.device)
    c = (H - 1) / 2
    src = c + (y - c) * scale + offset
    gy = (2 * src + 1) / H - 1  # align_corners=False
    grid = torch.stack(torch.broadcast_tensors(torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=video.device, dtype=video.dtype)[None, :],
                                               gy[:, None]), -1)
    return F.grid_sample(video[None], grid[None], mode="bilinear", padding_mode="border", align_corners=False)[0]


def _hp(x):
    return x - F.avg_pool2d(x[None, None], (9, 9), stride=1, padding=4, count_include_pad=False)[0, 0]


def _features(x):
    """(F, H, W): the high-passed gray (print edges) and the two chromaticities (the logo's red
    and blue; the room's reflections on chrome are mostly gray), each scaled to unit spread."""
    s = x.sum(0) + 0.1 * x.sum(0).mean() + 1e-4  # dark texels: their chromaticity is noise
    feats = [_hp(_gray(x)), (x[0] - x[1]) / s, (x[2] - x[1]) / s]
    return torch.stack([f / (f.std() + 1e-6) for f in feats])


def _best_shift_multi(E, m, V):
    """Mean over feature channels of the normalized correlation for every circular shift."""
    tot = 0
    for e, v in zip(E, V):
        e = (e - e[m].mean()) * m
        tot = tot + _corr_curve(e, m, v)
    tot = tot / len(E)
    k = int(tot.argmax())
    return float(tot[k]), k


def _corr_curve(e, m, v):
    Wa = e.shape[1]
    Ef, Vf, Mf = torch.fft.rfft(e, dim=1), torch.fft.rfft(v, dim=1), torch.fft.rfft(m.to(v.dtype), dim=1)
    num = torch.fft.irfft(Ef * Vf.conj(), n=Wa, dim=1).sum(0)
    vs = torch.fft.irfft(Mf * Vf.conj(), n=Wa, dim=1).sum(0)
    vv = torch.fft.irfft(Mf * torch.fft.rfft(v * v, dim=1).conj(), n=Wa, dim=1).sum(0)
    n = m.sum()
    var = (vv - vs ** 2 / n).clamp_min(1e-12)
    return num / (var.sqrt() * e.norm() + 1e-12)


def fit_label_atlas(video, est, count, min_count=3, rows=None, scales=None, offsets=None, allow_mirror=False):
    """video, est (3, Ha, Wa) linear; count (Ha, Wa) views per texel of est.
    rows: optional (lo, hi) atlas rows to compare (e.g. the printed band).
    The video atlas is also stretched and shifted in height (scales, offsets in rows): the
    video's unwrapped band and the scene's are cut at slightly different heights. Mirroring is
    off by default: every atlas here is in the pipeline's own handedness (checked: the render,
    board-video and can_label atlases all match unmirrored), and a free mirror lets a can
    turned end over end match a mirrored label instead of being turned.
    Returns (label (3, Ha, Wa), info dict)."""
    video = video.to(est.device, est.dtype)
    if video.shape[1:] != est.shape[1:]:
        video = F.interpolate(video[None], size=est.shape[1:], mode="bilinear", align_corners=False)[0]
    Ha, Wa = est.shape[1:]
    m = count >= min_count
    if rows is not None:
        band = torch.zeros_like(m)
        band[rows[0]:rows[1]] = True
        m = m & band
    info = {"texels_compared": int(m.sum())}
    if m.sum() < 200:
        info["used"] = False
        return None, info
    # normalized correlation of the high-passed gray and the chromaticities over the seen texels, for every
    # circular shift (FFT along the angle, summed over rows), both handednesses and a grid of
    # vertical stretches and shifts, then a finer grid around the best
    E = _features(est)
    if scales is None:
        scales = torch.linspace(0.7, 1.4, 15).tolist()
    if offsets is None:
        offsets = torch.arange(-0.2 * Ha, 0.2 * Ha + 1e-6, max(1.0, Ha / 64)).tolist()
    g = {mirror: (video.flip(-1) if mirror else video) for mirror in (False, True)}

    def search(sc_list, of_list, best):
        for mirror in ((False, True) if allow_mirror else (False,)):
            for sc in sc_list:
                for of in of_list:
                    c, k = _best_shift_multi(E, m, _features(_vwarp(g[mirror], sc, of)))
                    if best is None or c > best[0]:
                        best = (c, k, mirror, sc, of)
        return best

    best = search(scales, offsets, None)
    ds = (scales[1] - scales[0]) if len(scales) > 1 else 0.05
    do = (offsets[1] - offsets[0]) if len(offsets) > 1 else 1.0
    _, _, _, sc0, of0 = best
    best = search([sc0 + ds * t for t in (-0.5, -0.25, 0, 0.25, 0.5)], [of0 + do * t for t in (-0.5, -0.25, 0, 0.25, 0.5)], best)
    c, k, mirror, sc, of = best
    vid = video.flip(-1) if mirror else video
    vid = _vwarp(vid, sc, of)
    vid = torch.roll(vid, shifts=k, dims=-1)  # vid(y) = video(y - k): aligned with est
    # per-channel gain and offset over the seen texels
    out = torch.empty_like(vid)
    gains = []
    for ch in range(3):
        x, y = vid[ch][m], est[ch][m]
        A = torch.stack([x, torch.ones_like(x)], 1)
        sol = torch.linalg.lstsq(A, y[:, None]).solution[:, 0]
        a, b = float(sol[0]), float(sol[1])
        if a <= 0:
            a, b = float(y.mean() / x.mean().clamp_min(1e-6)), 0.0
        out[ch] = (a * vid[ch] + b).clamp_min(0)
        gains.append([round(a, 3), round(b, 4)])
    resid = (out - est)[:, m]
    info.update(used=True, correlation=round(c, 3), shift_columns=k, mirrored=bool(mirror),
                height_scale=round(float(sc), 3), height_offset_rows=round(float(of), 2), gain_offset=gains,
                rms_vs_estimate=round(float(resid.pow(2).mean().sqrt()), 4))
    return out, info
