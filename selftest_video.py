"""CPU self-test of the video front end on synthetic data (a minute or two; no GPU, no COLMAP).

  python selftest_video.py          # deletes its scenes/selftest_* afterwards
  python selftest_video.py --keep

1. frames:  an MP4 of synthetic photos goes through video_frames.py.
2. buffers: a mesh of synth.py's cans and table, ray-cast by `video_scene.py buffers`,
            matches the analytic geometry synth.py wrote (depth, normal, reflection visibility).
3. mesh:    a fake COLMAP reconstruction of a cylinder on a table, under an unknown rotation,
            scale and offset, is brought back by `video_scene.py mesh` (up axis, table at z = 0,
            scale, object at the origin).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent
SCENES = ROOT / "scenes"
SYNTH, MESH, ALIGN = "selftest_synth", "selftest_mesh", "selftest_align"


def run(command, scene=None):
    env = dict(os.environ)
    env.pop("SCENE", None)
    if scene:
        env["SCENE"] = scene
    result = subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        sys.exit(f"FAILED: {' '.join(map(str, command))}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
    return ok


def quaternion(m):
    """wxyz quaternion of a rotation matrix (Shepperd's method; inverse of capture.quat_to_matrix)."""
    if np.trace(m) > 0:
        s = 2 * np.sqrt(1 + np.trace(m))
        return np.array([s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    k = int(np.argmax(np.diag(m)))
    i, j = (k + 1) % 3, (k + 2) % 3
    s = 2 * np.sqrt(1 + m[k, k] - m[i, i] - m[j, j])
    q = np.zeros(4)
    q[0] = (m[j, i] - m[i, j]) / s
    q[1 + k] = s / 4
    q[1 + i] = (m[i, k] + m[k, i]) / s
    q[1 + j] = (m[j, k] + m[k, j]) / s
    return q


def cylinder(x, y, radius, height, segments=256):
    """Wall and lid of a vertical cylinder with exact normals (separate vertices for each)."""
    phi = np.arange(segments) * 2 * np.pi / segments
    ring = np.stack([x + radius * np.cos(phi), y + radius * np.sin(phi)], axis=1)
    radial = np.stack([np.cos(phi), np.sin(phi), np.zeros(segments)], axis=1)
    i, j = np.arange(segments), (np.arange(segments) + 1) % segments
    wall_v = np.concatenate([np.c_[ring, np.zeros(segments)], np.c_[ring, np.full(segments, height)]])
    wall_f = np.concatenate([np.stack([i, j, j + segments], 1), np.stack([i, j + segments, i + segments], 1)])
    wall_n = np.concatenate([radial, radial])
    lid_v = np.concatenate([[[x, y, height]], np.c_[ring, np.full(segments, height)]])
    lid_f = np.stack([np.zeros(segments, int), i + 1, j + 1], 1)
    lid_n = np.tile([0.0, 0.0, 1.0], (segments + 1, 1))
    return [(wall_v, wall_f, wall_n), (lid_v, lid_f, lid_n)]


def test_frames(tmp):
    rgb = np.load(SCENES / SYNTH / "cache" / "rgb.npy", mmap_mode="r")
    video = tmp / "orbit.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (rgb.shape[2], rgb.shape[1]))
    for frame in rgb:
        writer.write(np.ascontiguousarray(frame[..., ::-1]))
    writer.release()
    run(["video_frames.py", video, tmp / "frames", "--count", 8])
    count = len(list((tmp / "frames").glob("frame_*.jpg")))
    return check("frames", count == 8, f"{count} of 8 frames extracted from a {len(rgb)}-frame video")


def test_buffers():
    from video_scene import table_grid, write_mesh

    synth = SCENES / SYNTH
    spec = json.loads((synth / "data" / "scene.json").read_text())
    data = SCENES / MESH / "data"
    shutil.rmtree(SCENES / MESH, ignore_errors=True)
    (data / "rgb").mkdir(parents=True)
    spec.pop("env", None)
    (data / "scene.json").write_text(json.dumps(spec))
    shutil.copy(synth / "data" / "xy_poses.txt", data / "xy_poses.txt")
    rgb = np.load(synth / "cache" / "rgb.npy", mmap_mode="r")
    for i, frame in enumerate(rgb):
        cv2.imwrite(str(data / "rgb" / f"{i + 1:04d}.png"), np.ascontiguousarray(frame[..., ::-1]))

    obj = [part for can in spec["cans"] for part in cylinder(can["x"], can["y"], can["radius"], can["height"])]
    verts = np.concatenate([v for v, _, _ in obj])
    offsets = np.cumsum([0] + [len(v) for v, _, _ in obj[:-1]])
    faces = np.concatenate([f + o for (_, f, _), o in zip(obj, offsets)])
    normals = np.concatenate([n for _, _, n in obj])
    write_mesh(data / "mesh.npz", [(verts, faces, normals), table_grid(spec["floor_half"], spec["floor_half"])])
    run(["video_scene.py", "buffers", "--checks", 1], scene=MESH)

    ours, truth = SCENES / MESH / "cache", synth / "cache"
    load = lambda d, n: np.load(d / f"{n}.npy", mmap_mode="r")  # noqa: E731
    d0, d1 = load(truth, "depth")[..., 0].astype(float) / 1e4, load(ours, "depth")[..., 0].astype(float) / 1e4
    hit0, hit1 = d0 > 0, d1 > 0
    kernel = np.ones((3, 3), np.uint8)
    inner = np.stack([cv2.erode((a & b).astype(np.uint8), kernel) > 0 for a, b in zip(hit0, hit1)])
    n0 = load(truth, "normal").astype(float) / 32767
    n1 = load(ours, "normal").astype(float) / 32767
    v0, v1 = load(truth, "vis")[..., 0], load(ours, "vis")[..., 0]
    depth_ok = np.abs(d0 - d1)[inner] < 1e-3 * d0[inner]
    normal_ok = (n0 * n1).sum(-1)[inner] > 0.999
    vis_ok = (v0 == v1)[inner]
    results = [
        check("buffers/coverage", (hit0 == hit1).mean() > 0.99, f"{(hit0 == hit1).mean():.2%} of pixels agree on hit or miss"),
        check("buffers/depth", depth_ok.mean() > 0.99, f"{depth_ok.mean():.2%} within 0.1%"),
        check("buffers/normal", normal_ok.mean() > 0.99, f"{normal_ok.mean():.2%} within 2.6 degrees"),
        check("buffers/visibility", vis_ok.mean() > 0.98, f"{vis_ok.mean():.2%} agree"),
    ]
    return all(results)


def fake_colmap(dense, seed=0):
    """A COLMAP dense workspace of a cylinder (r 0.06, h 0.25) on a table, under a random similarity.

    Returns the camera centres in the true scene frame (table z = 0, cylinder axis at the origin).
    """
    from synth import camera_poses

    rng = np.random.default_rng(seed)
    radius, height = 0.06, 0.25
    poses = camera_poses(60, np.array([0, 0, 0.12]), radius=0.6, elevation=(5, 50))
    # surface samples with normals
    n_wall, n_lid, n_table = 60000, 6000, 100000
    phi = rng.uniform(0, 2 * np.pi, n_wall)
    wall = np.stack([radius * np.cos(phi), radius * np.sin(phi), rng.uniform(0, height, n_wall)], 1)
    wall_n = np.stack([np.cos(phi), np.sin(phi), np.zeros(n_wall)], 1)
    r, a = radius * np.sqrt(rng.uniform(0, 1, n_lid)), rng.uniform(0, 2 * np.pi, n_lid)
    lid = np.stack([r * np.cos(a), r * np.sin(a), np.full(n_lid, height)], 1)
    table = np.c_[rng.uniform(-0.4, 0.4, (n_table, 2)), np.zeros(n_table)]
    table = table[np.hypot(table[:, 0], table[:, 1]) > radius]
    points = np.concatenate([wall, lid, table]) + rng.normal(0, 5e-4, (len(wall) + len(lid) + len(table), 3))
    normals = np.concatenate([wall_n, np.tile([0, 0, 1.0], (len(lid) + len(table), 1))])

    # unknown similarity: world = s * Q p + T
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.sign(np.linalg.det(Q))
    s, T = 3.7, np.array([1.5, -2.0, 0.7])
    world_points, world_normals = s * points @ Q.T + T, normals @ Q.T
    R_c2w = np.einsum("ij,njk->nik", Q, poses[:, :, :3])
    centres = s * poses[:, :, 3] @ Q.T + T

    (dense / "images").mkdir(parents=True)
    (dense / "sparse_txt").mkdir()
    w, h, f = 320, 240, 270.0
    (dense / "sparse_txt" / "cameras.txt").write_text(f"1 PINHOLE {w} {h} {f} {f} {w / 2} {h / 2}\n")
    lines = []
    for i, (R, c) in enumerate(zip(R_c2w, centres)):
        R_w2c, name = R.T, f"frame_{i + 1:04d}.jpg"  # COLMAP stores world-to-camera
        t = -R_w2c @ c
        qw, qx, qy, qz = quaternion(R_w2c)
        lines += [f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {name}", ""]
        cv2.imwrite(str(dense / "images" / name), np.full((h, w, 3), 128, np.uint8))
    (dense / "sparse_txt" / "images.txt").write_text("\n".join(lines) + "\n")
    (dense / "sparse_txt" / "points3D.txt").write_text(
        "".join(f"{k + 1} {p[0]} {p[1]} {p[2]} 128 128 128 0\n" for k, p in enumerate(world_points[::500]))
    )
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world_points))
    pcd.normals = o3d.utility.Vector3dVector(world_normals)
    o3d.io.write_point_cloud(str(dense / "fused.ply"), pcd)
    return poses[:, :, 3]


def test_mesh(tmp):
    dense = tmp / "dense"
    truth = fake_colmap(dense)
    shutil.rmtree(SCENES / ALIGN, ignore_errors=True)
    run(["video_scene.py", "mesh", "--dense", dense, "--object-height", 0.25, "--width", 320], scene=ALIGN)
    data = SCENES / ALIGN / "data"
    poses = np.loadtxt(data / "xy_poses.txt", skiprows=1, ndmin=2)
    centres = poses[:, 1:4]
    spec = json.loads((data / "scene.json").read_text())
    reach = np.median(np.hypot(truth[:, 0], truth[:, 1]))
    height_err = np.abs(centres[:, 2] - truth[:, 2]).max() / reach
    radial_err = np.abs(np.hypot(centres[:, 0], centres[:, 1]) - np.hypot(truth[:, 0], truth[:, 1])).max() / reach
    down = poses[:, 4:].reshape(-1, 3, 3)[:, :, 1]  # camera y (down in the image) in scene coordinates
    can = spec["cans"][0]
    return all([
        check("mesh/camera heights", height_err < 0.02, f"max error {height_err:.2%} of the orbit radius"),
        check("mesh/camera distances", radial_err < 0.02, f"max error {radial_err:.2%} of the orbit radius"),
        check("mesh/up axis", (down[:, 2] < 0).all(), "image 'down' points down in the scene for every camera"),
        check("mesh/object size", abs(can["height"] - 0.25) < 0.0125 and abs(can["radius"] - 0.06) < 0.015,
              f"height {can['height']:.3f} (0.25), bounding radius {can['radius']:.3f} (0.06)"),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep scenes/selftest_* for inspection")
    args = ap.parse_args()
    tmp = SCENES / "selftest_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    print("building a small synthetic scene ...", flush=True)
    run(["synth.py", "--name", SYNTH, "--layout", "two", "--width", 160, "--height", 120, "--frames", 24, "--ss", 1])
    passed = [test_frames(tmp), test_buffers(), test_mesh(tmp)]
    if not args.keep:
        for name in [SYNTH, MESH, ALIGN, "selftest_tmp"]:
            shutil.rmtree(SCENES / name, ignore_errors=True)
    print("\nall checks passed" if all(passed) else "\nSOME CHECKS FAILED")
    sys.exit(0 if all(passed) else 1)


if __name__ == "__main__":
    main()
