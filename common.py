"""Shared constants, camera model, and dataset split for the Cans scene."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cans_data"
CACHE = ROOT / "cache"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

W, H = 640, 480
FOCAL = 541.961  # same camera for every dataset, from the project page
CX, CY = 320.0, 240.0
GAMMA = 1.6  # dataset linearization: linear = pixel ** 1.6

# Object region in world coordinates (floor is the XY plane). Used only to
# report metrics on the cans, as the paper does in Table 1 and Fig. 8.
CAN_Z_MIN = 0.012
TEST_FRACTION = 0.12


def load_poses():
    """Return (N, 3, 4) camera-to-world matrices; frame i is file {i+1:04d}.png."""
    rows = np.loadtxt(DATA / "xy_poses.txt", skiprows=1)
    assert (rows[:, 0] == np.arange(1, len(rows) + 1)).all()
    t = rows[:, 1:4]
    R = rows[:, 4:].reshape(-1, 3, 3)
    return np.concatenate([R, t[:, :, None]], axis=2)


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
