"""A label atlas made once from a video of the can turning (scripts/label_from_video.py),
fitted to one scene.

The scene's own label estimate (a low percentile of what each texel showed, sodacan.
separate.label_median) is noisy and has gaps where the photos did not see the can face-on,
and where other cans hid it. The video atlas covers the whole turn, but its angle origin,
its handedness and its brightness (exposure, room light) are the video's. Here: the circular
shift in angle (and mirroring) that best correlates it with the scene's estimate, over the
texels the scene saw often enough, then a per-channel gain and offset by least squares.
"""
import torch
import torch.nn.functional as F


def _gray(x):
    return x.mean(0)


def fit_label_atlas(video, est, count, min_count=3, rows=None):
    """video, est (3, Ha, Wa) linear; count (Ha, Wa) views per texel of est.
    rows: optional (lo, hi) atlas rows to compare (e.g. the printed band).
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
    # normalized correlation of the gray, high-passed images over the seen texels, for every
    # circular shift (FFT along the angle, summed over rows) and both handednesses
    def hp(x):
        return x - F.avg_pool2d(x[None, None], (9, 9), stride=1, padding=4, count_include_pad=False)[0, 0]

    e = hp(_gray(est)) * m
    e = e - e[m].mean() * m
    best = None
    for mirror in (False, True):
        v = _gray(video.flip(-1) if mirror else video)
        v = hp(v)
        # irfft(A conj(B))[k] = sum_x a(x + k) b(x): for every k, sum_y e(y) v(y - k) and the
        # masked sums of v(y - k), v(y - k)^2, i.e. v rolled by +k against e
        Ef, Vf, Mf = torch.fft.rfft(e, dim=1), torch.fft.rfft(v, dim=1), torch.fft.rfft(m.to(v.dtype), dim=1)
        num = torch.fft.irfft(Ef * Vf.conj(), n=Wa, dim=1).sum(0)
        vs = torch.fft.irfft(Mf * torch.fft.rfft(v, dim=1).conj(), n=Wa, dim=1).sum(0)
        vv = torch.fft.irfft(Mf * torch.fft.rfft(v * v, dim=1).conj(), n=Wa, dim=1).sum(0)
        n = m.sum()
        var = (vv - vs ** 2 / n).clamp_min(1e-12)
        corr = num / (var.sqrt() * e.norm() + 1e-12)
        k = int(corr.argmax())
        if best is None or float(corr[k]) > best[0]:
            best = (float(corr[k]), k, mirror)
    c, k, mirror = best
    vid = video.flip(-1) if mirror else video
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
    info.update(used=True, correlation=round(c, 3), shift_columns=k, mirrored=bool(mirror), gain_offset=gains,
                rms_vs_estimate=round(float(resid.pow(2).mean().sqrt()), 4))
    return out, info
