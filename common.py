"""Shared constants, camera model, and dataset split.

The default scene is Cans (cans_data/, cache/, runs/, results/ at the repo root).
Setting the environment variable SCENE=<name> switches every path to
scenes/<name>/, and scenes/<name>/data/scene.json overrides the camera.
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE = os.environ.get("SCENE", "")
BASE = ROOT / "scenes" / SCENE if SCENE else ROOT
DATA = BASE / ("data" if SCENE else "cans_data")
CACHE = BASE / "cache"
RUNS = BASE / "runs"
RESULTS = BASE / "results"

W, H = 640, 480
FOCAL = 541.961  # same camera for every dataset, from the project page
CX, CY = 320.0, 240.0
GAMMA = 1.6  # dataset linearization: linear = pixel ** 1.6

# Object region in world coordinates (floor is the XY plane). Used only to
# report metrics on the cans, as the paper does in Table 1 and Fig. 8.
CAN_Z_MIN = 0.012
TEST_FRACTION = 0.12

CONFIG = {}
if (DATA / "scene.json").exists():
    CONFIG = json.loads((DATA / "scene.json").read_text())
    W, H = CONFIG["width"], CONFIG["height"]
    FOCAL, CX, CY = CONFIG["focal"], CONFIG["cx"], CONFIG["cy"]
    GAMMA = CONFIG.get("gamma", GAMMA)


def load_poses():
    """Return (N, 3, 4) camera-to-world matrices; frame i is file {i+1:04d}.png."""
    rows = np.loadtxt(DATA / "xy_poses.txt", skiprows=1, ndmin=2)
    assert (rows[:, 0] == np.arange(1, len(rows) + 1)).all()
    t = rows[:, 1:4]
    R = rows[:, 4:].reshape(-1, 3, 3)
    return np.concatenate([R, t[:, :, None]], axis=2)


def save_poses(path, poses):
    rows = [[i + 1, *p[:, 3], *p[:, :3].reshape(-1)] for i, p in enumerate(poses)]
    np.savetxt(path, rows, fmt=["%d"] + ["%.8f"] * 12, header="index tx ty tz r11 r12 r13 r21 r22 r23 r31 r32 r33")


def load_cans():
    """Can definitions in legacy vertical or oriented-cylinder format.

    Can k (0-based row) has id k + 1 in the per-pixel can buffer; "a" is id 1, "b" is id 2.
    """
    if "cans" in CONFIG:
        rows = CONFIG["cans"]
    elif (CACHE / "cans.json").exists():
        rows = json.loads((CACHE / "cans.json").read_text())
    else:
        return None
    if rows and "center" in rows[0]:
        return rows
    return np.array([[c["x"], c["y"], c["radius"], c["height"]] for c in rows], np.float32)


def pixel_rays():
    """Camera-space ray directions (z = 1) for every pixel centre, shape (H, W, 3)."""
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    return np.stack([(u - CX) / FOCAL, (v - CY) / FOCAL, np.ones_like(u)], axis=-1)


def object_centre(vertices):
    """Centroid of mesh geometry raised above the carpet, near the middle of the scene."""
    keep = (vertices[:, 2] > 0.025) & (np.abs(vertices[:, :2]).max(axis=1) < 0.2)
    return vertices[keep].mean(axis=0)


def view_directions(poses, centre):
    """Unit vectors from the object centre toward each camera."""
    d = poses[:, :3, 3] - centre
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def make_split(poses, centre, fraction=TEST_FRACTION):
    """Hold out a contiguous cap of view directions to test extrapolation.

    The paper evaluates on test views far from the training views, but the
    released data has no split. The cap is centred on the lowest-elevation view
    (the edge of the capture), and grows until it holds `fraction` of the frames.

    Returns (train_idx, test_idx, angle_deg), where angle_deg[i] is the angle from
    test frame test_idx[i] to its nearest training view.
    """
    dirs = view_directions(poses, centre)
    seed = np.argmin(dirs[:, 2])
    ang = np.degrees(np.arccos(np.clip(dirs @ dirs[seed], -1, 1)))
    radius = np.quantile(ang, fraction)
    test = np.where(ang <= radius)[0]
    train = np.where(ang > radius)[0]
    cos = np.clip(dirs[test] @ dirs[train].T, -1, 1)
    return train, test, np.degrees(np.arccos(cos.max(axis=1)))


def select_views(poses, centre, frames, count, mode="fps", seed=0):
    """Pick `count` of `frames` for the view-count experiments.

    fps:    farthest-point sampling on view direction. Selections are nested
            (the first 8 of a 16-view pick are the 8-view pick), so adding views
            only ever adds observations.
    random: uniform without replacement.
    arc:    the `count` views nearest the first frame, a narrow-baseline capture.
    """
    frames = np.asarray(frames)
    if not count or count >= len(frames):
        return frames
    dirs = view_directions(poses[frames], centre)
    if mode == "random":
        return np.sort(np.random.default_rng(seed).choice(frames, count, replace=False))
    if mode == "arc":
        return frames[np.argsort(-(dirs @ dirs[0]))[:count]]
    chosen = [int(np.argmax(dirs @ dirs.mean(axis=0)))]
    nearest = dirs @ dirs[chosen[0]]
    while len(chosen) < count:
        nxt = int(np.argmin(nearest))
        chosen.append(nxt)
        nearest = np.maximum(nearest, dirs @ dirs[nxt])
    return frames[chosen]


def equirect_dirs(width, height):
    """Unit direction of every panorama texel centre, shape (height, width, 3).

    Same mapping as SRMModel.lookup: column from the azimuth atan2(y, x), row from
    the polar angle measured from +z, so row 0 is straight up.
    """
    phi = ((np.arange(width) + 0.5) / width - 0.5) * 2 * np.pi
    theta = (np.arange(height) + 0.5) / height * np.pi
    sin = np.sin(theta)[:, None]
    z = np.broadcast_to(np.cos(theta)[:, None], (height, width))
    return np.stack([sin * np.cos(phi)[None], sin * np.sin(phi)[None], z], axis=-1)


def equirect_sample(pano, dirs):
    """Bilinear lookup of an (h, w, c) panorama along unit directions (..., 3)."""
    h, w = pano.shape[:2]
    u = (np.arctan2(dirs[..., 1], dirs[..., 0]) / (2 * np.pi) + 0.5) * w - 0.5
    v = np.clip(np.arccos(np.clip(dirs[..., 2], -1, 1)) / np.pi * h - 0.5, 0, h - 1)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    v1 = np.minimum(v0 + 1, h - 1)
    top = pano[v0, u0 % w] * (1 - fu) + pano[v0, (u0 + 1) % w] * fu
    bottom = pano[v1, u0 % w] * (1 - fu) + pano[v1, (u0 + 1) % w] * fu
    return top * (1 - fv) + bottom * fv


def solid_angle(height, width):
    """Fraction of the sphere covered by each texel, shape (height, width); sums to 1."""
    rows = np.sin((np.arange(height) + 0.5) / height * np.pi)
    return np.repeat(rows[:, None], width, axis=1) / (rows.sum() * width)
