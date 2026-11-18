"""The can as a vertical cylinder: x^2 + y^2 = R^2, 0 <= z <= height, axis on world z.

Surface coordinates (s, h): s = atan2(y, x) in [-pi, pi), the angle around the can,
and h = z, the height. The surface atlas is an image over (s, h): column j holds
s = -pi + (j + 0.5) / W * 2 pi (increasing s reads left to right when the can is seen
from outside, so a label is not mirrored) and row i holds h = height * (1 - (i + 0.5) / H)
(row 0 at the rim).
"""
import math

import torch
import torch.nn.functional as F


def intersect(origins, dirs, radius, z0, z1):
    """First hit of rays from outside with the side wall between z0 and z1.

    origins broadcastable to dirs (..., 3). Returns distance t (inf on a miss) and hit mask.
    """
    ox, oy, oz = origins[..., 0], origins[..., 1], origins[..., 2]
    dx, dy, dz = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4 * a * c
    ok = (disc > 0) & (a > 1e-12)
    t = (-b - torch.sqrt(disc.clamp_min(0))) / (2 * a.clamp_min(1e-12))
    z = oz + t * dz
    hit = ok & (t > 0) & (z >= z0) & (z <= z1)
    return torch.where(hit, t, torch.full_like(t, math.inf)), hit


def intersect_disk(origins, dirs, radius, z):
    """Hit of rays with the horizontal disk of the given radius at height z (the lid)."""
    dz = dirs[..., 2]
    t = (z - origins[..., 2]) / torch.where(dz.abs() > 1e-12, dz, torch.full_like(dz, 1e-12))
    p = origins + t[..., None] * dirs
    hit = (t > 0) & (p[..., 0] ** 2 + p[..., 1] ** 2 <= radius * radius)
    return torch.where(hit, t, torch.full_like(t, math.inf)), hit


def normal(points):
    """Outward unit normal of the side wall at points (..., 3)."""
    n = torch.stack([points[..., 0], points[..., 1], torch.zeros_like(points[..., 0])], dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def tangent(s):
    """Unit tangent along increasing s."""
    return torch.stack([-torch.sin(s), torch.cos(s), torch.zeros_like(s)], dim=-1)


def surface_coords(points):
    return torch.atan2(points[..., 1], points[..., 0]), points[..., 2]


def surface_point(s, h, radius):
    return torch.stack([radius * torch.cos(s), radius * torch.sin(s), h], dim=-1)


def atlas_coords(rows, cols, height, device="cpu"):
    """(s, h) at the centres of an atlas of rows x cols texels, each (rows, cols)."""
    s = -math.pi + (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) / cols * 2 * math.pi
    h = height * (1 - (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) / rows)
    hh, ss = torch.meshgrid(h, s, indexing="ij")
    return ss, hh


def atlas_sample(atlas, s, h, height):
    """Bilinear lookup of an atlas (C, Ha, Wa) at surface coords s, h of any shape (...).

    Wraps around in s, clamps in h. Returns (..., C).
    """
    C, Ha, Wa = atlas.shape
    shape = s.shape
    x = (s.reshape(-1) + math.pi) / (2 * math.pi) * Wa - 0.5  # texel index space
    y = (1 - h.reshape(-1) / height) * Ha - 0.5
    padded = torch.cat([atlas[..., -1:], atlas, atlas[..., :1]], dim=-1)
    gx = (x + 1) / (Wa + 1) * 2 - 1
    gy = y / max(Ha - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)[None, None]
    out = F.grid_sample(padded[None], grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out[0, :, 0].T.reshape(*shape, C)


def perturbed_normal(n, s, h, dent, height):
    """Normal tilted by a dent map (2, Ha, Wa) of small angles (radians):
    channel 0 tilts toward increasing s, channel 1 toward +z."""
    if dent is None:
        return n
    offset = atlas_sample(dent, s, h, height)
    m = n + offset[..., :1] * tangent(s) + offset[..., 1:2] * torch.tensor([0.0, 0.0, 1.0], device=n.device)
    return m / m.norm(dim=-1, keepdim=True)
