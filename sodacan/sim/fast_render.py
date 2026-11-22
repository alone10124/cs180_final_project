"""Fast renderer of a glossy can under a distant environment map, in PyTorch.

Each can pixel is one environment lookup along its reflection direction, so no ray
tracer is needed. The model, per pixel of the side wall:

    I = exposure * ( strength * F(cos_i) * env_rough(r)          specular (clear coat or metal)
                     + (1 - F(cos_i)) * label(s, h) * ambient )    diffuse ink under the coat
        + noise

env_rough is the environment blurred by the surface roughness; r is the reflection of
the camera ray about the (optionally dented) normal. Other pixels show the lid (grey),
the table or ChArUco board at z = 0, or the environment straight behind. The reverse
of this model is the classical pipeline (stages 3 to 5).
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch

from sodacan.color import linear_to_srgb, srgb_to_linear
from sodacan.geometry import cylinder, equirect
from sodacan.geometry.camera import interpolate_poses
from sodacan.geometry.reflect import can_geometry


@dataclass
class RenderParams:
    roughness_deg: float = 0.0      # std of the specular lobe, degrees
    strength: float = 1.0           # specular scale (1 = perfect mirror)
    label: Optional[torch.Tensor] = None  # (3, Ha, Wa) ink albedo in surface coordinates
    ambient: float = 1.0            # label irradiance relative to the environment's mean radiance
    dent: Optional[torch.Tensor] = None   # (2, Ha, Wa) normal tilts in radians
    exposure: float = 1.0
    noise_read: float = 0.0         # std of additive noise (linear units, after exposure)
    noise_shot: float = 0.0         # variance per unit signal
    quantize: bool = False          # 8-bit sRGB round trip, like a video frame
    motion_blur: float = 0.0        # fraction of the interval to the next frame the shutter stays open
    motion_steps: int = 4
    table: Optional[dict] = None    # {"albedo": (3,)} or {"texture": (3, Hb, Wb), "ppm": px/m, "centre": (cx, cy)}
    ss: int = 1                     # supersampling per axis
    chunk: int = 8                  # frames per batch (memory)


def mean_radiance(env):
    w = equirect.solid_angle(env.shape[1], env.shape[2], env.device)
    return (env * w).sum((1, 2)) / w.sum()


def _table_radiance(points, table, ambient_rgb):
    """Table/board albedo at world points (..., 3) on z = 0, times ambient."""
    if table is None:
        table = {"albedo": (0.35, 0.33, 0.30)}
    if "texture" in table:
        tex = table["texture"]
        ppm, (cx, cy) = float(table["ppm"]), table["centre"]
        bx = points[..., 0] + cx  # board coords: world = diag(1, -1, -1) (board - centre)
        by = cy - points[..., 1]
        Hb, Wb = tex.shape[1:]
        gx = (bx * ppm - 0.5) / (Wb - 1) * 2 - 1
        gy = (by * ppm - 0.5) / (Hb - 1) * 2 - 1
        grid = torch.stack([gx, gy], -1).reshape(1, 1, -1, 2)
        base = torch.tensor(table.get("albedo", (0.35, 0.33, 0.30)), device=points.device)
        val = torch.nn.functional.grid_sample(tex[None], grid, align_corners=True, padding_mode="zeros")[0, :, 0].T
        inside = (gx.abs() <= 1) & (gy.abs() <= 1)
        alb = torch.where(inside.reshape(-1, 1), val, base.expand_as(val)).reshape(*points.shape[:-1], 3)
    else:
        alb = torch.tensor(table["albedo"], device=points.device).expand(*points.shape[:-1], 3)
    return alb * ambient_rgb


def _render_static(env_sharp, env_rough, c2w, K, can, p, ambient_rgb):
    g = can_geometry(c2w, K, can, dent=p.dent, ss=p.ss)
    o = c2w[:, None, None, :3, 3].expand_as(g.view)
    d = g.view
    out = equirect.sample(env_sharp, d)  # background: the room straight behind
    # table plane z = 0
    t_tab = torch.where(d[..., 2] < -1e-6, -o[..., 2] / d[..., 2].clamp_max(-1e-6), torch.full_like(d[..., 2], math.inf))
    t_lid, lid = cylinder.intersect_disk(o, d, can.radius, can.height)
    t_wall = g.t
    nearest = torch.minimum(torch.minimum(t_tab, t_lid), t_wall)
    on_table = torch.isfinite(t_tab) & (t_tab <= nearest)
    if on_table.any():
        pts = o[on_table] + t_tab[on_table][:, None] * d[on_table]
        out[on_table] = _table_radiance(pts, p.table, ambient_rgb)
    on_lid = lid & (t_lid <= nearest)
    out[on_lid] = 0.35 * ambient_rgb
    on_wall = g.hit & (t_wall <= nearest)
    if on_wall.any():
        Fr = g.fresnel[on_wall][:, None]
        spec = p.strength * Fr * equirect.sample(env_rough, g.reflected[on_wall])
        val = spec
        if p.label is not None:
            alb = cylinder.atlas_sample(p.label, g.s[on_wall], g.h[on_wall], can.height)
            val = val + (1 - Fr) * alb * ambient_rgb
        out[on_wall] = val
    out = out.permute(0, 3, 1, 2)
    if p.ss > 1:
        B, C, H, W = out.shape
        out = out.reshape(B, C, H // p.ss, p.ss, W // p.ss, p.ss).mean((3, 5))
    return out


@torch.no_grad()
def render(env, c2w, K, can, params=None, generator=None):
    """Render B frames (B, 3, H, W) in linear light. env (3, He, We); c2w (B, 4, 4)."""
    p = params or RenderParams()
    env = env.to(c2w.device)
    env_rough = equirect.blur(env, p.roughness_deg)
    ambient_rgb = mean_radiance(env) * p.ambient
    frames = []
    B = c2w.shape[0]
    for i in range(0, B, p.chunk):
        idx = torch.arange(i, min(i + p.chunk, B), device=c2w.device)
        poses = c2w[idx]
        if p.motion_blur > 0 and B > 1:
            nxt = c2w[torch.clamp(idx + 1, max=B - 1)]
            prev = c2w[torch.clamp(idx - 1, min=0)]
            nxt = torch.where((idx == B - 1)[:, None, None], 2 * poses - prev, nxt)  # extrapolate at the end
            acc = 0
            for a in torch.linspace(0, p.motion_blur, p.motion_steps):
                acc = acc + _render_static(env, env_rough, interpolate_poses(poses, nxt, float(a)), K, can, p, ambient_rgb)
            img = acc / p.motion_steps
        else:
            img = _render_static(env, env_rough, poses, K, can, p, ambient_rgb)
        frames.append(img)
    img = torch.cat(frames) * p.exposure
    if p.noise_read > 0 or p.noise_shot > 0:
        std = torch.sqrt(p.noise_shot * img.clamp_min(0) + p.noise_read**2)
        noise = torch.randn(img.shape, generator=generator, device="cpu" if generator is not None else img.device)
        img = img + std * noise.to(img.device)
    if p.quantize:
        img = srgb_to_linear(torch.round(linear_to_srgb(img) * 255) / 255)
    return img
