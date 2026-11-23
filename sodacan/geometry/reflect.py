"""Per-pixel reflection geometry: which pixels see the can wall, and what direction each reflects."""
from dataclasses import dataclass

import torch

from sodacan.geometry import cylinder, fresnel
from sodacan.geometry.camera import world_rays


def reflect(d, n):
    """Mirror reflection r = d - 2 (d . n) n."""
    return d - 2 * (d * n).sum(-1, keepdim=True) * n


@dataclass
class PixelGeometry:
    mask: torch.Tensor      # (B, H, W) bool: ray hits the usable wall at a non-grazing angle
    hit: torch.Tensor       # (B, H, W) bool: ray hits the wall anywhere between 0 and height
    t: torch.Tensor         # (B, H, W) ray distance to the wall (inf on a miss)
    s: torch.Tensor         # (B, H, W) angle around the can
    h: torch.Tensor         # (B, H, W) height on the can
    view: torch.Tensor      # (B, H, W, 3) unit camera ray d
    normal: torch.Tensor    # (B, H, W, 3) unit normal (with dents, if given)
    reflected: torch.Tensor # (B, H, W, 3) unit reflection direction r
    cos_i: torch.Tensor     # (B, H, W) cosine of incidence, -d . n
    fresnel: torch.Tensor   # (B, H, W)


def reflects_other_can(geom, c2w_self, c2w_others, can):
    """(B, H, W) bool: pixels of this can whose mirror reflection hits another can (its wall
    or a lid) instead of the room. Close together, chrome cans show each other: those pixels
    carry a neighbour's label, not the environment, and are wrong for both the label estimate
    and the room. geom: can_geometry of this can for the B cameras c2w_self (B, 4, 4) in its
    frame; c2w_others: list of (B, 4, 4) for the same cameras in each other can's frame."""
    t = torch.where(geom.hit, geom.t, torch.zeros_like(geom.t))
    o = c2w_self[:, None, None, :3, 3]
    p = o + t[..., None] * geom.view          # hit points, this can's frame
    out = torch.zeros_like(geom.hit)
    for other in c2w_others:
        T = other @ torch.linalg.inv(c2w_self)  # this can's frame -> the other's, per camera
        Rm, tv = T[:, None, None, :3, :3], T[:, None, None, :3, 3]
        q = (Rm @ p[..., None])[..., 0] + tv
        r = (Rm @ geom.reflected[..., None])[..., 0]
        q = q + 1e-4 * r
        _, hw = cylinder.intersect(q, r, can.radius, 0.0, can.height)
        _, ht = cylinder.intersect_disk(q, r, can.radius, can.height)
        _, hb = cylinder.intersect_disk(q, r, can.radius, 0.0)
        out |= hw | ht | hb
    return out & geom.hit


def can_geometry(c2w, K, can, dent=None, ss=1):
    """Ray-cylinder intersection, normals and reflections for every pixel of B cameras."""
    o, d = world_rays(c2w, K, ss)
    t, hit = cylinder.intersect(o, d, can.radius, 0.0, can.height)
    tt = torch.where(hit, t, torch.zeros_like(t))
    p = o + tt[..., None] * d
    s, h = cylinder.surface_coords(p)
    n = cylinder.normal(p)
    n = cylinder.perturbed_normal(n, s, h, dent, can.height)
    cos_i = -(d * n).sum(-1)
    r = reflect(d, n)
    Fr = fresnel.reflectance(cos_i, can.material)
    mask = hit & (h >= can.z_lo) & (h <= can.z_hi) & (cos_i > can.min_cos)
    return PixelGeometry(mask, hit, t, s, h, d, n, r, cos_i, Fr)
