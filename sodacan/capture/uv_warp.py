"""Warp frames into the can's own surface coordinates (angle around the can, height).

In that space the printed label stays still while reflections slide as the camera moves,
which is what label removal (separate/label_median.py) relies on.
"""
import torch
import torch.nn.functional as F

from sodacan.geometry import cylinder
from sodacan.geometry.camera import project


@torch.no_grad()
def warp_to_surface(frames, c2w, K, can, rows=None, cols=None, chunk=8):
    """Sample frames (B, 3, H, W) at every atlas texel of the can wall.

    Returns obs (B, 3, rows, cols) and valid (B, rows, cols): the texel faces the camera
    at better than can.min_cos and projects inside the image.
    """
    rows = rows or can.atlas[0]
    cols = cols or can.atlas[1]
    device = frames.device
    s, h = cylinder.atlas_coords(rows, cols, can.height, device)
    p = cylinder.surface_point(s, h, can.radius)
    n = cylinder.normal(p)
    obs, valid = [], []
    for i in range(0, frames.shape[0], chunk):
        fr, pose = frames[i : i + chunk], c2w[i : i + chunk]
        B = fr.shape[0]
        cam = pose[:, :3, 3]
        v = cam[:, None, None] - p[None]
        cos = (v * n[None]).sum(-1) / v.norm(dim=-1)
        uv, z = project(p[None].expand(B, -1, -1, -1), pose, K)
        inside = (uv[..., 0] >= 0) & (uv[..., 0] <= K.width - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= K.height - 1)
        ok = (cos > can.min_cos) & inside & (z > 0)
        grid = torch.stack([uv[..., 0] / (K.width - 1) * 2 - 1, uv[..., 1] / (K.height - 1) * 2 - 1], -1)
        sampled = F.grid_sample(fr, grid, mode="bilinear", padding_mode="border", align_corners=True)
        obs.append(sampled * ok[:, None])
        valid.append(ok)
    return torch.cat(obs), torch.cat(valid)
