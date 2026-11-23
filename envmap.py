"""Environment panoramas: splatting observations, coverage, and comparison with a reference.

Panoramas are (SRM_H, SRM_W, 3) linear RGB numpy arrays in the layout of SRMModel.lookup.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import CONFIG, DATA, GAMMA, solid_angle
from data import select, to_linear
from models import SRM_H, SRM_W

MIN_WEIGHT = 0.5  # splat weight a texel needs to count as observed


def chunks(frames, size=8):
    frames = [int(f) for f in frames]
    return [frames[i : i + size] for i in range(0, len(frames), size)]


def observed(b, cans, keep=1.0, generator=None):
    """Pixels whose mirror reflection of the environment is usable: selected surface,
    unoccluded reflection, and not on a silhouette (1-pixel erosion)."""
    sel = select(b, cans) * b["vis"]
    sel = -F.max_pool2d(-sel, 3, stride=1, padding=1)
    if keep < 1:
        sel = sel * (torch.rand(sel.shape, generator=generator, device=sel.device) < keep)
    return sel[:, 0] > 0


def accumulate(scene, frames, cans="both", diffuse="texture", keep=1.0, seed=0):
    """Splat specular residuals (photo - diffuse, linear) into a panorama along reflection directions.

    Each pixel is spread over its four nearest texels with bilinear weights, the adjoint
    of the lookup in SRMModel. `keep` < 1 drops pixels at random (pixel-count control).
    Returns (total (H, W, 3), weight (H, W)); the mean radiance is total / weight.
    """
    device = scene.device
    total = torch.zeros(SRM_H * SRM_W, 3, device=device, dtype=torch.float64)
    weight = torch.zeros(SRM_H * SRM_W, device=device, dtype=torch.float64)
    generator = torch.Generator(device).manual_seed(seed)
    for idx in chunks(frames):
        b = scene.batch(idx)
        m = observed(b, cans, keep, generator)
        residual = to_linear(b["rgb"])
        if diffuse == "texture":
            residual = residual - to_linear(b["diffuse"])
        value = residual.permute(0, 2, 3, 1)[m].double()
        refl = b["refl"].permute(0, 2, 3, 1)[m]
        u = (torch.atan2(refl[:, 1], refl[:, 0]) / (2 * math.pi) + 0.5) * SRM_W - 0.5
        v = (torch.acos(refl[:, 2].clamp(-1, 1)) / math.pi * SRM_H - 0.5).clamp(0, SRM_H - 1)
        u0, v0 = torch.floor(u), torch.floor(v)
        fu, fv = (u - u0).double(), (v - v0).double()
        for du, dv, w in [(0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)]:
            col = (u0.long() + du) % SRM_W
            row = (v0.long() + dv).clamp(max=SRM_H - 1)
            flat = row * SRM_W + col
            weight.index_add_(0, flat, w)
            total.index_add_(0, flat, w[:, None] * value)
    return total.reshape(SRM_H, SRM_W, 3).cpu().numpy(), weight.reshape(SRM_H, SRM_W).cpu().numpy()


def finalize(total, weight):
    """Mean panorama and the mask of observed texels."""
    valid = weight >= MIN_WEIGHT
    pano = np.where(valid[..., None], total / np.maximum(weight, 1e-8)[..., None], 0)
    return np.clip(pano, 0, None).astype(np.float32), valid


def srm_environment(model, scene, frames, cans, share_frames=16):
    """The trained basis panorama that the selected surface uses most, and each basis's share.

    Basis selection does not need all ~2,000 full-resolution frames.  Sampling
    evenly across the capture makes CPU evaluation tens of times faster while
    preserving the material-share estimate used only to choose a panorama.
    """
    frames = np.asarray(frames)
    if share_frames and len(frames) > share_frames:
        sample = np.linspace(0, len(frames) - 1, share_frames).round().astype(int)
        frames = frames[sample]
    share = torch.zeros(len(model.srm), device=scene.device)
    with torch.no_grad():
        for batch_no, idx in enumerate(chunks(frames), 1):
            b = scene.batch(idx)
            m = observed(b, cans)[:, None].float()
            share += (model.material(b["diffuse"]) * m).sum(dim=(0, 2, 3))
            print(f"selecting SRM basis: {min(batch_no * 8, len(frames))}/{len(frames)} frames", flush=True)
    share = (share / share.sum().clamp(min=1)).cpu().numpy()
    best = int(share.argmax())
    return model.srm[best].detach().permute(1, 2, 0).cpu().numpy(), share


def load_reference(path=None):
    """Reference environment resampled to panorama size, linear RGB; None if the scene has none.

    8-bit images are treated as photographs: gamma 2.2, and mirrored into the lookup layout.
    """
    if path is None:
        if "env" not in CONFIG:
            return None
        path = DATA / CONFIG["env"]
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None, f"cannot read {path}"
    if image.dtype == np.uint8:
        # A camera panorama runs left to right as you turn right; the lookup layout runs
        # the other way (column grows with atan2(y, x), counter-clockwise from above).
        image = (image[:, ::-1] / 255.0) ** 2.2
    image = image[..., 2::-1].astype(np.float32)
    return cv2.resize(image, (SRM_W, SRM_H), interpolation=cv2.INTER_AREA)


def display(linear):
    return np.clip(linear, 0, 1) ** (1 / GAMMA)


def blur_wrap(image, sigma):
    """Gaussian blur in texels, continuous across the azimuth seam."""
    if sigma <= 0:
        return image
    pad = int(3 * sigma) + 1
    wide = np.concatenate([image[:, -pad:], image, image[:, :pad]], axis=1)
    return cv2.GaussianBlur(wide, (0, 0), sigma)[:, pad:-pad]


def ssim_map(a, b):
    """Per-texel SSIM of two display-range images, averaged over channels (11x11 Gaussian window)."""
    def mean(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    a, b = a.astype(np.float64), b.astype(np.float64)
    mu_a, mu_b = mean(a), mean(b)
    var_a, var_b = mean(a * a) - mu_a**2, mean(b * b) - mu_b**2
    cov = mean(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return ssim.mean(axis=2)


def fit_scale(pano, reference, w):
    """Per-channel gain that best maps the estimate onto the reference (weighted least squares)."""
    w = w[..., None] * (reference.max(axis=2, keepdims=True) < 0.95)
    return (w * pano * reference).sum(axis=(0, 1)) / np.maximum((w * pano * pano).sum(axis=(0, 1)), 1e-12)


def align_azimuth(pano, valid, reference):
    """Column shift of the reference that best correlates with the estimate (unknown heading)."""
    w = (solid_angle(SRM_H, SRM_W) * valid)[..., None]
    a = display(pano / max(np.percentile(pano[valid], 99), 1e-6))
    r = display(blur_wrap(reference, 2))
    a0 = a - (w * a).sum() / (3 * w.sum())
    best, best_score = 0, -np.inf
    for shift in range(SRM_W):
        rs = np.roll(r, shift, axis=1)
        r0 = rs - (w * rs).sum() / (3 * w.sum())
        score = (w * a0 * r0).sum() / np.sqrt((w * r0 * r0).sum() + 1e-12)
        if score > best_score:
            best, best_score = shift, score
    return best


def compare(pano, valid, reference=None, region=None, sigmas=(0, 0.5, 1, 2, 3, 5, 8), align=False):
    """Coverage of a panorama and, given a reference, PSNR/SSIM over observed texels.

    Every texel is weighted by its solid angle. The estimate is gain-matched per channel
    and the reference is blurred to the Gaussian width that maximises PSNR (the reflection
    lobe); both are clipped to [0, 1] and display-encoded before comparison.
    `region` restricts PSNR/SSIM to a sub-mask, e.g. texels shared by two settings.
    Returns (metrics, estimate_display, reference_display).
    """
    sa = solid_angle(SRM_H, SRM_W)
    upper = np.arange(SRM_H)[:, None] < SRM_H // 2
    metrics = {
        "coverage": float(sa[valid].sum()),
        "coverage_upper": float(sa[valid & upper].sum() / sa[np.broadcast_to(upper, valid.shape)].sum()),
    }
    if reference is None or not valid.any():
        return metrics, display(pano / max(np.percentile(pano, 99.5), 1e-6)), None
    if align:
        shift = align_azimuth(pano, valid, reference)
        reference = np.roll(reference, shift, axis=1)
        metrics["azimuth_shift_deg"] = shift * 360 / SRM_W
    region = valid if region is None else region & valid
    w = sa * region
    best = None
    for sigma in sigmas:
        ref = blur_wrap(reference, sigma)
        gain = fit_scale(pano, ref, w)
        est_d, ref_d = display(pano * gain), display(ref)
        mse = (w[..., None] * (est_d - ref_d) ** 2).sum() / (3 * w.sum())
        if best is None or mse < best[0]:
            best = (mse, sigma, gain, est_d, ref_d)
    mse, sigma, gain, est_d, ref_d = best
    inner = cv2.erode(region.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    metrics.update(
        psnr=float(-10 * np.log10(max(mse, 1e-12))),
        ssim=float((sa * inner * ssim_map(est_d, ref_d)).sum() / (sa * inner).sum()) if inner.any() else None,
        blur_deg=float(sigma * 360 / SRM_W),
        gain=[float(g) for g in gain],
    )
    return metrics, est_d * valid[..., None], ref_d


def save_png(path, image):
    """Write a display-range RGB float image."""
    cv2.imwrite(str(path), (np.clip(image, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))


def side_by_side(estimate, reference, valid):
    """Estimate over reference, with unobserved texels of the estimate in dark blue."""
    shown = np.where(valid[..., None], estimate, np.array([0.05, 0.05, 0.25]))
    return shown if reference is None else np.vstack([shown, reference])
