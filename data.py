"""Loads cached frames to the GPU, derives per-pixel geometry, and augments batches."""
import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, CAN_Z_MIN, CX, CY, FOCAL, GAMMA, H, W, load_cans, load_poses, pixel_rays

CAN_CHOICES = {"a": [1], "b": [2], "both": [1, 2]}


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_linear(x):
    return x.clamp(0, 1) ** GAMMA


def to_display(x):
    return x.clamp(1e-4, 1) ** (1 / GAMMA)


def can_ids(pos, mask, cans, margin=1.2):
    """Per-pixel can id from world position: 0 for floor and background, k + 1 for can k."""
    ids = torch.zeros_like(mask)
    if cans is None:
        return ids
    z = pos[:, 2:3]
    for k, (x, y, radius, height) in enumerate(cans):
        near = torch.hypot(pos[:, 0:1] - x, pos[:, 1:2] - y) < radius * margin
        ids = torch.where(near & (z > CAN_Z_MIN) & (z < height * margin), float(k + 1), ids)
    return ids * mask


def select(b, which):
    """Pixels allowed to carry specular reflection. "all" keeps every surface, floor included."""
    if which == "all":
        return b["mask"]
    return sum((b["can"] == k).float() for k in CAN_CHOICES[which])


class Scene:
    def __init__(self, device, diffuse="diffuse"):
        """`diffuse` names the cached diffuse buffer: "diffuse" or "diffuse_est" (label.py)."""
        self.device = device
        names = ["rgb", "fbi", "vis", "depth", "normal"]
        self.arrays = {n: np.load(CACHE / f"{n}.npy", mmap_mode="r") for n in names}
        self.arrays["diffuse"] = np.load(CACHE / f"{diffuse}.npy", mmap_mode="r")
        cans = load_cans()
        self.cans = None if cans is None else cans.tolist()
        split = np.load(CACHE / "split.npz")
        self.train, self.test, self.test_angle = split["train"], split["test"], split["angle"]
        self.poses = torch.tensor(load_poses(), dtype=torch.float32, device=device)
        rays = torch.tensor(pixel_rays(), dtype=torch.float32, device=device)
        self.rays = rays.permute(2, 0, 1)[None]  # 1, 3, H, W with z = 1

    def _tensor(self, name, idx):
        array = np.stack([self.arrays[name][i] for i in idx])
        return torch.from_numpy(array).to(self.device).permute(0, 3, 1, 2)

    def batch(self, idx):
        """Return un-augmented buffers for frame indices `idx`."""
        depth = self._tensor("depth", idx).float() / 1e4
        normal = self._tensor("normal", idx).float() / 32767
        R = self.poses[idx, :3, :3]
        t = self.poses[idx, :3, 3]
        ray = torch.einsum("bij,bjhw->bihw", R, self.rays.expand(len(idx), -1, -1, -1))
        pos = t[:, :, None, None] + depth * ray
        unit = F.normalize(ray, dim=1)
        cos = (-(unit * normal).sum(1, keepdim=True)).clamp(0, 1)
        mask = (depth > 0).float()
        return {
            "rgb": self._tensor("rgb", idx).float() / 255,
            "diffuse": self._tensor("diffuse", idx).float() / 255,
            "fbi": self._tensor("fbi", idx).float() / 255,
            "vis": self._tensor("vis", idx).float(),
            "mask": mask,
            "depth": depth,
            "pos": pos,
            "can": can_ids(pos, mask, self.cans),
            "refl": (unit + 2 * cos * normal) * mask,
            "fci": ((1 - cos) ** 5) * mask,
            "pose": self.poses[idx],
        }


def _identity_affine(n, device):
    m = torch.zeros(n, 2, 3, device=device)
    m[:, 0, 0] = m[:, 1, 1] = 1
    return m


def sample_affine(n, device, rescale):
    """Random augmentation from the supplement (Sec. H).

    Returns (to_source, to_augmented): 2x3 pixel-space affines that map augmented
    coordinates to original coordinates and back.
    """
    angle = torch.rand(n, device=device) * 2 * torch.pi - torch.pi  # up to 180 degrees
    shift = (torch.rand(n, 2, device=device) * 2 - 1) * 100 * W / 640  # up to 100 pixels at 640 wide
    flip_x = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    flip_y = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    scale = torch.ones(n, device=device)
    if rescale:
        scale = torch.exp(torch.rand(n, device=device) * 2 * np.log(1.25) - np.log(1.25))

    c, s = torch.cos(angle), torch.sin(angle)
    forward = torch.zeros(n, 2, 2, device=device)
    forward[:, 0, 0] = scale * c * flip_x
    forward[:, 0, 1] = -scale * s * flip_y
    forward[:, 1, 0] = scale * s * flip_x
    forward[:, 1, 1] = scale * c * flip_y
    inverse = torch.linalg.inv(forward)

    centre = torch.tensor([W / 2, H / 2], device=device, dtype=torch.float32)
    to_aug = torch.cat([forward, (centre + shift - forward @ centre)[:, :, None]], dim=2)
    to_src = torch.cat([inverse, centre[None, :, None] - inverse @ (centre + shift)[:, :, None]], dim=2)
    return to_src, to_aug


def pixel_grid(n, device):
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32) + 0.5,
        torch.arange(W, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    return torch.stack([x, y], dim=-1)[None].expand(n, -1, -1, -1)  # B, H, W, 2


def apply_affine(m, xy):
    return torch.einsum("bij,bhwj->bhwi", m[:, :2, :2], xy) + m[:, None, None, :2, 2]


def normalise(xy):
    return torch.stack([2 * xy[..., 0] / W - 1, 2 * xy[..., 1] / H - 1], dim=-1)


def augment(batch, rescale=False, enabled=True):
    """Warp every image buffer by one random affine per sample."""
    n = len(batch["rgb"])
    device = batch["rgb"].device
    if enabled:
        to_src, to_aug = sample_affine(n, device, rescale)
    else:
        to_src = to_aug = _identity_affine(n, device)

    grid = normalise(apply_affine(to_src, pixel_grid(n, device)))
    out = dict(batch, to_aug=to_aug)
    for key in ["rgb", "diffuse", "fbi", "refl", "fci"]:
        out[key] = F.grid_sample(batch[key], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    for key in ["vis", "mask", "depth", "pos", "can"]:
        out[key] = F.grid_sample(batch[key], grid, mode="nearest", padding_mode="zeros", align_corners=False)
    out["refl"] = F.normalize(out["refl"], dim=1) * out["mask"]
    return out


def cross_project(batch, images, src, dst, occlusion_tol=0.01):
    """Warp `images[dst]` into the augmented view `src` using known geometry.

    Returns (warped, valid) for the view-consistency term V of Eq. 7.
    """
    pose = batch["pose"][dst]
    R, t = pose[:, :3, :3], pose[:, :3, 3]
    delta = batch["pos"][src] - t[:, :, None, None]
    cam = torch.einsum("bij,bihw->bjhw", R, delta)  # R transpose * (x - t)
    z = cam[:, 2:3].clamp(min=1e-3)
    u = FOCAL * cam[:, 0:1] / z + CX
    v = FOCAL * cam[:, 1:2] / z + CY
    xy = torch.cat([u, v], dim=1).permute(0, 2, 3, 1)
    grid = normalise(apply_affine(batch["to_aug"][dst], xy))

    warped = F.grid_sample(images[dst], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    dst_depth = F.grid_sample(batch["depth"][dst], grid, mode="nearest", padding_mode="zeros", align_corners=False)
    visible = (dst_depth > 0) & ((dst_depth - cam[:, 2:3]).abs() < occlusion_tol)
    valid = visible.float() * batch["mask"][src]
    return warped, valid
