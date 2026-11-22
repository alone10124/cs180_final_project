"""Pinhole camera. OpenCV axes: x right, y down, z forward. Poses are camera-to-world.

Pixel (col j, row i) has its centre at image coordinate (j, i), as in OpenCV.
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width, height, fov_y_deg):
        f = 0.5 * height / math.tan(math.radians(fov_y_deg) / 2)
        return cls(width, height, f, f, (width - 1) / 2, (height - 1) / 2)

    @classmethod
    def from_dict(cls, d):
        return cls(int(d["width"]), int(d["height"]), float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]))

    def scaled(self, width, height):
        """Intrinsics for the same camera after resizing the image to width x height."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5)

    def matrix(self):
        return [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]


def pixel_grid(K, ss=1, device="cpu"):
    """Image coordinates (H*ss, W*ss, 2) of sub-pixel sample centres, x then y."""
    xs = (torch.arange(K.width * ss, device=device, dtype=torch.float32) + 0.5) / ss - 0.5
    ys = (torch.arange(K.height * ss, device=device, dtype=torch.float32) + 0.5) / ss - 0.5
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([x, y], dim=-1)


def camera_rays(K, ss=1, device="cpu"):
    """Unit ray directions in camera coordinates, (H*ss, W*ss, 3)."""
    xy = pixel_grid(K, ss, device)
    d = torch.stack([(xy[..., 0] - K.cx) / K.fx, (xy[..., 1] - K.cy) / K.fy, torch.ones_like(xy[..., 0])], dim=-1)
    return d / d.norm(dim=-1, keepdim=True)


def world_rays(c2w, K, ss=1):
    """Origins (B, 1, 1, 3) and unit directions (B, H, W, 3) in world coordinates."""
    d = camera_rays(K, ss, c2w.device)
    dirs = torch.einsum("bij,hwj->bhwi", c2w[:, :3, :3], d)
    return c2w[:, None, None, :3, 3], dirs


def project(points, c2w, K):
    """World points (B, ..., 3) -> image coordinates (B, ..., 2) and camera depth (B, ...)."""
    R, t = c2w[:, :3, :3], c2w[:, :3, 3]
    shape = points.shape
    p = points.reshape(shape[0], -1, 3) - t[:, None]
    pc = torch.einsum("bji,bnj->bni", R, p)  # R^T (p - t)
    z = pc[..., 2]
    zs = z.clamp_min(1e-9)
    uv = torch.stack([K.fx * pc[..., 0] / zs + K.cx, K.fy * pc[..., 1] / zs + K.cy], dim=-1)
    return uv.reshape(*shape[:-1], 2), z.reshape(shape[:-1])


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera-to-world (B, 4, 4) for cameras at eye (B, 3) looking at target (B, 3) or (3,)."""
    eye = torch.as_tensor(eye, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32, device=eye.device).expand_as(eye)
    up = torch.as_tensor(up, dtype=torch.float32, device=eye.device).expand_as(eye)
    forward = target - eye
    forward = forward / forward.norm(dim=-1, keepdim=True)
    right = torch.cross(forward, up, dim=-1)
    right = right / right.norm(dim=-1, keepdim=True)
    down = torch.cross(forward, right, dim=-1)
    c2w = torch.eye(4, device=eye.device).repeat(eye.shape[0], 1, 1)
    c2w[:, :3, 0], c2w[:, :3, 1], c2w[:, :3, 2], c2w[:, :3, 3] = right, down, forward, eye
    return c2w


def rotation_from_axis_angle(axis_angle):
    """Rodrigues: (B, 3) axis * angle -> (B, 3, 3)."""
    theta = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = axis_angle / theta
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    Kx = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1).reshape(-1, 3, 3)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    eye = torch.eye(3, device=axis_angle.device).expand_as(Kx)
    return eye + s * Kx + (1 - c) * (Kx @ Kx)


def perturb_poses(c2w, rot_deg, trans_m, generator=None):
    """Random pose error: rotation of std rot_deg about a random axis, translation of std trans_m."""
    B = c2w.shape[0]
    aa = torch.randn(B, 3, generator=generator, device="cpu").to(c2w.device) * math.radians(rot_deg) / math.sqrt(3)
    dt = torch.randn(B, 3, generator=generator, device="cpu").to(c2w.device) * trans_m / math.sqrt(3)
    out = c2w.clone()
    out[:, :3, :3] = rotation_from_axis_angle(aa) @ c2w[:, :3, :3]
    out[:, :3, 3] += dt
    return out


def interpolate_poses(a, b, alpha):
    """Pose between a and b (B, 4, 4) at fraction alpha: linear translation, re-orthonormalised rotation."""
    out = a.clone()
    out[:, :3, 3] = (1 - alpha) * a[:, :3, 3] + alpha * b[:, :3, 3]
    U, _, Vh = torch.linalg.svd((1 - alpha) * a[:, :3, :3] + alpha * b[:, :3, :3])
    out[:, :3, :3] = U @ Vh
    return out
