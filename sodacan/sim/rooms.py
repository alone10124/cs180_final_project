"""Environment maps: random procedural box rooms, and HDR panoramas loaded from disk.

A procedural room is a box around the can with painted rectangles on its six faces:
windows (bright, with mullions and a sky gradient), ceiling lights, a door, pictures,
furniture, shelves of books, posters with high-frequency text-like stripes, and
sometimes a person-like silhouette. Radiance is linear; lights and windows exceed 1.
"""
import math
import os
from pathlib import Path

import numpy as np
import torch

from sodacan.geometry.equirect import pano_dirs, resize_pano

# face ids: 0 +x, 1 -x, 2 +y, 3 -y, 4 ceiling (+z), 5 floor (-z)
WALLS = (0, 1, 2, 3)


def _face_hits(d, lo, hi):
    """Hit point on the box [lo, hi] (containing the origin) along directions d (H, W, 3)."""
    big = torch.full_like(d, math.inf)
    t = torch.where(d > 1e-9, hi / d.clamp_min(1e-9), torch.where(d < -1e-9, lo / d.clamp_max(-1e-9), big))
    tmin, axis = t.min(dim=-1)
    p = d * tmin[..., None]
    positive = torch.gather(d, -1, axis[..., None])[..., 0] > 0
    face = axis * 2 + (~positive).long()
    return p, face


def _wall_coords(p, face):
    """(a, b) coordinates on each face: a horizontal along the wall (or x on floor/ceiling), b up (or y)."""
    x, y, z = p.unbind(-1)
    a = torch.where(face < 2, y, torch.where(face < 4, x, x))
    b = torch.where(face < 4, z, y)
    return a, b


def procedural_room(H=512, W=1024, seed=0, device="cpu", person=None):
    """A random room as a (3, H, W) linear panorama, seen from the can."""
    rng = np.random.default_rng(seed)
    size = rng.uniform([2.5, 2.5, 2.3], [6.0, 6.0, 3.2])
    centre = rng.uniform(0.2, 0.8, size=2) * size[:2]
    eye = rng.uniform(0.6, 1.1)  # can height above the floor
    lo = torch.tensor([-centre[0], -centre[1], -eye], dtype=torch.float32, device=device)
    hi = torch.tensor([size[0] - centre[0], size[1] - centre[1], size[2] - eye], dtype=torch.float32, device=device)
    d = pano_dirs(H, W, device)
    p, face = _face_hits(d, lo, hi)
    a, b = _wall_coords(p, face)
    env = torch.zeros(3, H, W, device=device)

    def colour(lo_=0.15, hi_=0.8):
        return torch.tensor(rng.uniform(lo_, hi_, 3), dtype=torch.float32, device=device)

    def paint(mask, value):
        env[:, mask] = value.reshape(3, 1)

    wall_base = colour(0.3, 0.75)
    for f in WALLS:
        paint(face == f, wall_base * float(rng.uniform(0.8, 1.15)))
    paint(face == 4, colour(0.6, 0.85))
    floor_col = colour(0.08, 0.4)
    plank = 0.85 + 0.15 * torch.remainder(torch.floor(b / rng.uniform(0.1, 0.3)), 2)
    m = face == 5
    env[:, m] = floor_col[:, None] * plank[m][None]
    skirting = (face < 4) & (b < lo[2] + 0.08)
    paint(skirting, colour(0.05, 0.3))

    def wall_extent(f):
        if f in (0, 1):
            return float(lo[1]), float(hi[1])
        return float(lo[0]), float(hi[0])

    def rect(f, a0, a1, b0, b1):
        return (face == f) & (a > a0) & (a < a1) & (b > b0) & (b < b1)

    def place(f, width, b0, b1):
        e0, e1 = wall_extent(f)
        width = min(width, 0.8 * (e1 - e0))
        a0 = rng.uniform(e0 + 0.05, e1 - width - 0.05)
        return rect(f, a0, a0 + width, b0, b1), a0

    walls = list(rng.permutation(WALLS))
    floor_z, ceil_z = float(lo[2]), float(hi[2])
    # windows: sky gradient, mullions
    for f in walls[: rng.integers(1, 3)]:
        bottom = floor_z + rng.uniform(0.8, 1.1)
        win, a0 = place(f, rng.uniform(0.8, 2.0), bottom, min(ceil_z - 0.2, bottom + rng.uniform(1.0, 1.6)))
        sky = torch.tensor(rng.uniform([1.5, 2.0, 2.8], [3.5, 4.5, 6.5]), dtype=torch.float32, device=device)
        grad = (0.7 + 0.3 * (b - bottom).clamp(0, 2) / 2)
        env[:, win] = sky[:, None] * grad[win][None]
        mull = win & ((torch.remainder(a - a0, rng.uniform(0.4, 0.7)) < 0.04) | (torch.abs(b - (bottom + 0.6)) < 0.03))
        paint(mull, colour(0.05, 0.3))
    # ceiling lights or a lamp
    for _ in range(rng.integers(1, 5)):
        cx, cy = rng.uniform(float(lo[0]) + 0.3, float(hi[0]) - 0.3), rng.uniform(float(lo[1]) + 0.3, float(hi[1]) - 0.3)
        r = rng.uniform(0.1, 0.35)
        shape = (face == 4) & (torch.abs(a - cx) < r) & (torch.abs(b - cy) < r * rng.uniform(0.3, 1.5))
        paint(shape, torch.tensor([1.0, 0.95, 0.85], device=device) * float(rng.uniform(6, 20)))
    # door
    f = walls[-1]
    door, _ = place(f, 0.9, floor_z, floor_z + 2.05)
    paint(door, colour(0.1, 0.5))
    # pictures and posters with text-like stripes
    for _ in range(rng.integers(1, 5)):
        f = int(rng.choice(WALLS))
        b0 = floor_z + rng.uniform(1.2, 1.7)
        pic, a0 = place(f, rng.uniform(0.3, 1.0), b0, b0 + rng.uniform(0.3, 0.8))
        paint(pic, colour(0.05, 0.9))
        if rng.random() < 0.5:
            lines = pic & (torch.remainder(b - b0, 0.08) < 0.035) & (torch.remainder(a - a0, rng.uniform(0.05, 0.15)) < 0.04)
            paint(lines, colour(0.0, 0.15))
    # furniture blocks and shelves
    for _ in range(rng.integers(1, 4)):
        f = int(rng.choice(WALLS))
        top = floor_z + rng.uniform(0.4, 1.9)
        block, a0 = place(f, rng.uniform(0.6, 2.0), floor_z, top)
        if rng.random() < 0.4:
            book = torch.floor((a - a0) / rng.uniform(0.03, 0.08)).long()
            palette = torch.tensor(rng.uniform(0.05, 0.8, (6, 3)), dtype=torch.float32, device=device)
            env[:, block] = palette[torch.remainder(book[block], 6)].T
            gaps = block & (torch.remainder(b - floor_z, rng.uniform(0.3, 0.45)) < 0.03)
            paint(gaps, colour(0.03, 0.15))
        else:
            paint(block, colour(0.05, 0.6))
    # a standing person (the live-demo volunteer)
    if person is None:
        person = rng.random() < 0.4
    if person:
        f = int(rng.choice(WALLS))
        e0, e1 = wall_extent(f)
        ac = rng.uniform(e0 + 0.4, e1 - 0.4)
        torso = (face == f) & (torch.abs(a - ac) < 0.22) & (b > floor_z + 0.9) & (b < floor_z + 1.5)
        legs = (face == f) & (torch.abs(a - ac) < 0.15) & (b > floor_z) & (b < floor_z + 0.9)
        head = (face == f) & ((a - ac) ** 2 + (b - floor_z - 1.62) ** 2 < 0.11**2)
        paint(torso, colour(0.05, 0.7))
        paint(legs, colour(0.03, 0.3))
        paint(head, torch.tensor([0.55, 0.38, 0.28], device=device))
    # gentle shading: walls darker toward the floor
    shade = torch.where(face < 4, 0.8 + 0.2 * ((b - floor_z) / (ceil_z - floor_z)).clamp(0, 1), torch.ones_like(b))
    return (env * shade[None]).contiguous()


def read_hdr(path):
    """Load an equirectangular .hdr/.exr as a (3, H, W) float32 tensor, linear RGB."""
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return torch.from_numpy(np.ascontiguousarray(img[..., ::-1].astype(np.float32))).permute(2, 0, 1)


def write_hdr(path, pano):
    import cv2

    img = pano.detach().float().cpu().permute(1, 2, 0).numpy()[..., ::-1]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.ascontiguousarray(img))


def photo_to_pano(img):
    """A photo-convention equirect (as from Poly Haven or a 360 camera) into this package's layout.

    Photo panoramas put azimuth increasing to the left when z is up and the image is
    viewed from inside; ours is mirrored (see geometry/equirect.py), so flip columns.
    """
    return img.flip(-1)


def list_hdris(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".hdr", ".exr"))


def random_environment(H, W, rng, hdris=(), hdri_prob=0.5, device="cpu", cache=None):
    """A procedural room or a randomly rotated, randomly exposed HDRI, at H x W."""
    if hdris and rng.random() < hdri_prob:
        path = hdris[rng.integers(len(hdris))]
        if cache is not None and path in cache:
            env = cache[path]
        else:
            env = resize_pano(photo_to_pano(read_hdr(path)).to(device), H, W)
            if cache is not None:
                cache[path] = env
        env = torch.roll(env, int(rng.integers(W)), dims=-1)
        if rng.random() < 0.5:
            env = env.flip(-1)
        return env / env.mean().clamp_min(1e-6) * 0.5
    return procedural_room(H, W, seed=int(rng.integers(2**31)), device=device)
