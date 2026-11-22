"""Can segmentation. With a known pose and a known cylinder the mask is the projected
wall; `silhouette_score` checks that the projected silhouette sits on real image edges,
which catches a wrong pose, a wrong can size or a misplaced can."""
import cv2
import numpy as np
import torch

from sodacan.geometry.reflect import can_geometry


def can_mask(c2w, K, can, erode_px=2):
    """(B, H, W) bool mask of usable can-wall pixels, eroded to keep clear of the silhouette."""
    g = can_geometry(c2w, K, can)
    m = g.mask
    if erode_px > 0:
        k = 2 * erode_px + 1
        m = -torch.nn.functional.max_pool2d(-m[:, None].float(), k, 1, erode_px)[:, 0] > 0.5
    return m


def silhouette_score(frame_rgb, c2w, K, can, band_px=3):
    """Mean edge strength on the projected silhouette divided by that a few pixels away.

    Well above 1 means the outline sits on the can's real edges; near 1 means it does not."""
    g = can_geometry(c2w[None], K, can)
    hit = g.hit[0].cpu().numpy().astype(np.uint8)
    edge = cv2.morphologyEx(hit, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    near = cv2.dilate(edge.astype(np.uint8), np.ones((4 * band_px + 1,) * 2, np.uint8)) > 0
    near &= ~(cv2.dilate(edge.astype(np.uint8), np.ones((2 * band_px + 1,) * 2, np.uint8)) > 0)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    on = cv2.dilate(grad, np.ones((2 * band_px + 1,) * 2, np.uint8))[edge]
    off = grad[near]
    if on.size == 0 or off.size == 0:
        return float("nan")
    return float(on.mean() / max(off.mean(), 1e-6))
