"""Panorama metrics, reported only inside the coverage mask.

Panoramas are HDR and the real reflection strength is unknown, so every metric first
matches the estimate's gain to the reference (least squares over the mask), then maps
both to display values with one exposure (reference median -> 0.18) and the sRGB curve.
Averages are weighted by texel solid angle.
"""
import math

import torch
import torch.nn.functional as F

from sodacan.color import linear_to_srgb
from sodacan.geometry.equirect import solid_angle

_LPIPS = {}


def gain_match(est, ref, mask):
    m = mask[None].expand_as(est)
    num = (est[m] * ref[m]).sum()
    den = (est[m] ** 2).sum().clamp_min(1e-12)
    return est * (num / den)


def exposure_of(ref, mask, key=0.18):
    lum = ref.mean(0)[mask]
    lum = lum[lum > 0]
    return key / float(lum.median()) if lum.numel() else 1.0


def to_display(est, ref, mask, gain=True):
    if gain:
        est = gain_match(est, ref, mask)
    e = exposure_of(ref, mask)
    return linear_to_srgb(est * e), linear_to_srgb(ref * e)


def psnr(est, ref, mask, gain=True):
    a, b = to_display(est.float(), ref.float(), mask, gain)
    w = solid_angle(*mask.shape, device=mask.device) * mask
    mse = (((a - b) ** 2).mean(0) * w).sum() / w.sum().clamp_min(1e-12)
    return float(10 * torch.log10(1 / mse.clamp_min(1e-12)))


def ssim(est, ref, mask, gain=True, sigma=1.5):
    a, b = to_display(est.float(), ref.float(), mask, gain)
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=a.device, dtype=a.dtype) / sigma) ** 2)
    k = (k / k.sum())
    k2 = (k[:, None] * k[None]).expand(3, 1, -1, -1)

    def filt(x):
        x = F.pad(x[None], (r, r, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, r, r), mode="replicate")
        return F.conv2d(x, k2, groups=3)[0]

    mu_a, mu_b = filt(a), filt(b)
    saa = filt(a * a) - mu_a**2
    sbb = filt(b * b) - mu_b**2
    sab = filt(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2))
    w = solid_angle(*mask.shape, device=mask.device) * mask
    return float((s.mean(0) * w).sum() / w.sum().clamp_min(1e-12))


def lpips(est, ref, mask, gain=True):
    """LPIPS (VGG) on display images with the unobserved texels zeroed in both. Needs `lpips`."""
    import lpips as lp

    dev = est.device
    if dev not in _LPIPS:
        _LPIPS[dev] = lp.LPIPS(net="vgg", verbose=False).to(dev).eval()
    a, b = to_display(est.float(), ref.float(), mask, gain)
    a, b = a * mask, b * mask
    with torch.no_grad():
        return float(_LPIPS[dev](a[None] * 2 - 1, b[None] * 2 - 1).mean())


def all_metrics(est, ref, mask, with_lpips=True):
    out = {"psnr": psnr(est, ref, mask), "ssim": ssim(est, ref, mask)}
    if with_lpips:
        try:
            out["lpips"] = lpips(est, ref, mask)
        except ImportError:
            pass
    w = solid_angle(*mask.shape, device=mask.device)
    out["coverage"] = float((w * mask).sum() / w.sum())
    return out


def align_azimuth(est, ref, mask, flips=(False, True)):
    """Rotate (and optionally mirror) a reference panorama to best match est inside mask.

    For real captures, whose phone panorama has an unknown heading. Returns the aligned
    reference, the shift in columns and whether it was mirrored."""
    best = (-math.inf, 0, False)
    for flip in flips:
        r = ref.flip(-1) if flip else ref
        a = torch.log1p(est.mean(0) * 10) * mask
        for shift in range(ref.shape[-1]):
            b = torch.log1p(torch.roll(r, shift, -1).mean(0) * 10) * mask
            am, bm = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
            c = float((am * bm).sum() / (am.norm() * bm.norm()).clamp_min(1e-12))
            if c > best[0]:
                best = (c, shift, flip)
    _, shift, flip = best
    r = ref.flip(-1) if flip else ref
    return torch.roll(r, shift, -1), shift, flip
