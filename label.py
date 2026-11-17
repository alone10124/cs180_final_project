"""Estimate each can's printed label from the photos, without a texture scan.

  SCENE=label python label.py

A surface point shows its label colour plus whatever it reflects, and the reflection
changes with the viewpoint while the label does not. The estimate is the darkest
colour each point takes across views (after a 3x3 blur), stored in a cylindrical
atlas per can (azimuth x height). Reflections never vanish entirely, so this is
biased bright by the dimmest reflection seen.

Writes cache/diffuse_est.npy: the cached diffuse buffer with can pixels replaced by
the atlas. Use it with `--buffer diffuse_est` (splat.py) or `--diffuse diffuse_est` (train.py).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE
from data import Scene, pick_device
from envmap import chunks

ATLAS_W, ATLAS_H = 360, 64


def atlas_index(b, cans):
    """Flat atlas texel of every pixel, and the mask of pixels on a can."""
    pos, can = b["pos"], b["can"][:, 0]
    index = torch.zeros_like(can, dtype=torch.long)
    for k, (x, y, _, height) in enumerate(cans):
        phi = torch.atan2(pos[:, 1] - y, pos[:, 0] - x)
        col = ((phi / (2 * math.pi) + 0.5) * ATLAS_W).long().clamp(0, ATLAS_W - 1)
        row = (pos[:, 2] / height * ATLAS_H).long().clamp(0, ATLAS_H - 1)
        index = torch.where(can == k + 1, (k * ATLAS_H + row) * ATLAS_W + col, index)
    return index, can > 0


def estimate(scene, frames):
    """(K * ATLAS_H * ATLAS_W, 3) display-encoded minimum colour per atlas texel."""
    size = len(scene.cans) * ATLAS_H * ATLAS_W
    atlas = torch.full((size, 3), 2.0, device=scene.device)
    for idx in chunks(frames):
        b = scene.batch(idx)
        index, on_can = atlas_index(b, scene.cans)
        inner = -F.max_pool2d(-on_can[:, None].float(), 3, stride=1, padding=1)[:, 0] > 0
        rgb = F.avg_pool2d(b["rgb"], 3, stride=1, padding=1).permute(0, 2, 3, 1)[inner]
        atlas.scatter_reduce_(0, index[inner][:, None].expand(-1, 3), rgb, reduce="amin")
    return torch.where(atlas > 1.5, torch.zeros_like(atlas), atlas)


def write(scene, atlas, count):
    source = np.load(CACHE / "diffuse.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(CACHE / "diffuse_est.npy", "w+", source.dtype, source.shape)
    for idx in chunks(range(count)):
        b = scene.batch(idx)
        index, on_can = atlas_index(b, scene.cans)
        frame = torch.from_numpy(np.stack([source[i] for i in idx])).to(scene.device)
        est = (atlas[index] * 255 + 0.5).to(torch.uint8)
        out[idx[0] : idx[-1] + 1] = torch.where(on_can[..., None], est, frame).cpu().numpy()
    out.flush()


def main():
    scene = Scene(pick_device())
    assert scene.cans, "this scene has no can geometry on record (run fit_cans.py for Cans)"
    atlas = estimate(scene, scene.train)
    write(scene, atlas, len(scene.arrays["rgb"]))
    print(f"wrote {CACHE / 'diffuse_est.npy'}")


if __name__ == "__main__":
    main()
