"""Equirectangular panoramas, (C, H, W) tensors in linear light.

Direction d = (x, y, z) with z up. Column from the azimuth phi = atan2(y, x):
u = (phi / 2pi + 0.5) * W. Row from the polar angle theta = acos(z): v = theta / pi * H,
so row 0 is straight up. Texel (i, j) has its centre at (u, v) = (j + 0.5, i + 0.5).

This is the layout seen from outside the sphere: +y (left of someone facing +x) is to
the right of +x in the image. `to_viewer` flips it into the usual photo panorama.
"""
import math

import torch
import torch.nn.functional as F


def dir_to_uv(d, H, W):
    phi = torch.atan2(d[..., 1], d[..., 0])
    theta = torch.acos(d[..., 2].clamp(-1, 1))
    return (phi / (2 * math.pi) + 0.5) * W, theta / math.pi * H


def uv_to_dir(u, v, H, W):
    phi = (u / W - 0.5) * 2 * math.pi
    theta = v / H * math.pi
    st = torch.sin(theta)
    return torch.stack([st * torch.cos(phi), st * torch.sin(phi), torch.cos(theta)], dim=-1)


def pano_dirs(H, W, device="cpu"):
    v = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    u = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return uv_to_dir(uu, vv, H, W)


def sample(pano, dirs):
    """Bilinear lookup of pano (C, H, W) along directions (..., 3); wraps in azimuth. Returns (..., C)."""
    C, H, W = pano.shape
    shape = dirs.shape[:-1]
    u, v = dir_to_uv(dirs.reshape(-1, 3), H, W)
    x, y = u - 0.5, v - 0.5
    x = torch.remainder(x, W)
    padded = torch.cat([pano[..., -1:], pano, pano[..., :2]], dim=-1)  # columns -1 .. W+1
    gx = (x + 1) / (W + 2) * 2 - 1
    gy = y / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)[None, None]
    out = F.grid_sample(padded[None], grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out[0, :, 0].T.reshape(*shape, C)


def solid_angle(H, W, device="cpu"):
    """Solid angle of each texel in steradians, (H, W); sums to 4 pi."""
    theta0 = torch.arange(H, device=device, dtype=torch.float32) / H * math.pi
    theta1 = theta0 + math.pi / H
    rows = (torch.cos(theta0) - torch.cos(theta1)) * 2 * math.pi / W
    return rows[:, None].expand(H, W)


def resize_pano(pano, H, W):
    """Area-average resampling to H x W (exact when the size divides, which is the common case)."""
    C, h, w = pano.shape
    if (h, w) == (H, W):
        return pano
    if h % H == 0 and w % W == 0:
        return F.avg_pool2d(pano[None], (h // H, w // W))[0]
    return F.interpolate(pano[None], size=(H, W), mode="area" if h > H else "bilinear", align_corners=None if h > H else False)[0]


def blur(pano, sigma_deg):
    """Approximately isotropic Gaussian blur on the sphere: vertical blur, then a horizontal
    blur whose width grows as 1/sin(theta), in bands of rows. Wraps in azimuth."""
    if sigma_deg <= 0:
        return pano
    C, H, W = pano.shape
    sy = sigma_deg / 180 * H
    out = _blur1d(pano, sy, dim=1, wrap=False)
    theta = (torch.arange(H, dtype=torch.float32) + 0.5) / H * math.pi
    sx_rows = sigma_deg / 360 * W / torch.sin(theta).clamp_min(0.05)
    sx_rows = sx_rows.clamp(max=W / 4)
    bands = torch.unique(torch.round(torch.log2(sx_rows) * 4))
    result = torch.empty_like(out)
    key = torch.round(torch.log2(sx_rows) * 4)
    for b in bands:
        rows = (key == b).nonzero().squeeze(1).to(pano.device)
        sx = float(2 ** (b / 4))
        result[:, rows] = _blur1d(out[:, rows], sx, dim=2, wrap=True)
    return result


def _blur1d(x, sigma, dim, wrap):
    if sigma < 0.3:
        return x
    radius = int(math.ceil(3 * sigma))
    k = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[0]
    if dim == 2:
        y = x[None]
        n = y.shape[-1]
        if wrap:
            reps = radius // n + 1
            y = torch.cat([y] * (2 * reps + 1), dim=-1)[..., reps * n - radius : (reps + 1) * n + radius]
        else:
            y = F.pad(y, (radius, radius), mode="replicate")
        return F.conv2d(y, k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)[0]
    y = x[None]
    y = F.pad(y, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(y, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)[0]


def to_viewer(pano):
    """Flip into the photo convention (turning right moves the view right) for 360 viewers."""
    return pano.flip(-1)
