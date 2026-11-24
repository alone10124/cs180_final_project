"""Debug images: every stage writes one, so a wrong pose or a flipped axis shows at once."""
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from sodacan.color import auto_exposure, to_uint8
from sodacan.geometry import cylinder
from sodacan.geometry.camera import project

UNOBSERVED = np.array([40, 30, 90], dtype=np.uint8)  # dark blue, RGB


def save_rgb(path, rgb_uint8):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(rgb_uint8), cv2.COLOR_RGB2BGR))


def display(linear, mask=None, exposure=None):
    """(3, H, W) linear -> (H, W, 3) uint8, auto-exposed, unobserved texels dark blue."""
    e = exposure if exposure is not None else auto_exposure(linear, mask)
    img = to_uint8(linear, e)
    if mask is not None:
        img[~mask.cpu().numpy()] = UNOBSERVED
    return img


def heat(x, mask=None, log=True):
    """Scalar map -> colour image (viridis-like via OpenCV)."""
    v = x.detach().float().cpu().numpy()
    if log:
        v = np.log10(np.maximum(v, 1e-6))
    lo, hi = np.percentile(v[np.isfinite(v)], [2, 99.5]) if np.isfinite(v).any() else (0, 1)
    v = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
    if mask is not None:
        img[~mask.cpu().numpy()] = UNOBSERVED
    return img


def montage(images, cols=3, pad=4, labels=None):
    """Grid of images; each label sits on a white strip above its image."""
    strip = 24 if labels else 0
    h = max(i.shape[0] for i in images) + strip
    w = max(i.shape[1] for i in images)
    rows = math.ceil(len(images) / cols)
    out = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 255, np.uint8)
    for k, img in enumerate(images):
        r, c = divmod(k, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        out[y + strip : y + strip + img.shape[0], x : x + img.shape[1]] = img
        if labels:
            cv2.putText(out, labels[k], (x + 2, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def stack(images, labels=None):
    """Vertical stack of equal-width images with labels."""
    return montage(images, cols=1, labels=labels)


def can_outline(frame_rgb, c2w, K, can, colour=(0, 255, 0)):
    """Draw the projected can wall (rim, base, usable band, silhouette lines) on a frame."""
    img = frame_rgb.copy()
    s = torch.linspace(-math.pi, math.pi, 181, device=c2w.device)
    for z, col in ((0.0, colour), (can.height, colour), (can.z_lo, (255, 200, 0)), (can.z_hi, (255, 200, 0))):
        p = cylinder.surface_point(s, torch.full_like(s, z), can.radius)
        uv, depth = project(p[None], c2w[None], K)
        pts = uv[0].cpu().numpy().astype(np.int32)
        if (depth > 0).all():
            cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
    # world axes at the can base: x red, y green, z blue
    axes = torch.tensor([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]], dtype=torch.float32, device=c2w.device)
    uv, depth = project(axes[None], c2w[None], K)
    if (depth > 0).all():
        o = tuple(int(v) for v in uv[0, 0])
        for k, col in ((1, (255, 0, 0)), (2, (0, 255, 0)), (3, (0, 0, 255))):
            cv2.line(img, o, tuple(int(v) for v in uv[0, k]), col, 2, cv2.LINE_AA)
    return img


def can_silhouette(frame_rgb, c2w, K, can, colour=(0, 255, 0)):
    """Draw the projected can as a whole: its silhouette (convex hull of rim and base, lightly
    tinted, outlined) and the rim, so a misplaced can is obvious at a glance."""
    img = frame_rgb.copy()
    s = torch.linspace(-math.pi, math.pi, 97, device=c2w.device)[:-1]
    pts, ok = [], True
    for z in (0.0, can.height):
        p = cylinder.surface_point(s, torch.full_like(s, z), can.radius)
        uv, depth = project(p[None], c2w[None], K)
        ok &= bool((depth > 0).all())
        pts.append(uv[0].cpu().numpy())
    if not ok:
        return img
    hull = cv2.convexHull(np.vstack(pts).astype(np.float32)).astype(np.int32)
    tint = img.copy()
    cv2.fillConvexPoly(tint, hull, colour)
    img = cv2.addWeighted(img, 0.75, tint, 0.25, 0)
    cv2.polylines(img, [hull], True, colour, 3, cv2.LINE_AA)
    cv2.polylines(img, [pts[1].astype(np.int32)], True, colour, 2, cv2.LINE_AA)  # the top rim
    return img


def direction_rgb(dirs, mask):
    """Unit directions (H, W, 3) shown as RGB = (d + 1) / 2, black outside mask."""
    img = ((dirs.detach().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    img[~mask.cpu().numpy()] = 0
    return img
