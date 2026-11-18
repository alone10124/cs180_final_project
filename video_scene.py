"""A scene from a video reconstruction of one object on a table, for the rest of the pipeline.

  SCENE=bag python video_scene.py mesh --dense scenes/bag/capture/colmap/dense --object-height 0.25
  SCENE=bag python video_scene.py buffers

mesh: aligns the COLMAP reconstruction (sfm.py) so the table is the plane z = 0 with z up
and the object stands at the origin, scales it so the object is --object-height tall, cuts
the object out of the fused point cloud, meshes it (Poisson), and adds the table as a
triangle grid. Writes data/scene.json, xy_poses.txt, mesh.npz and rgb/0001.png ...
scene.json lists the object's bounding cylinder as "can a", so `--cans a` in splat.py and
train.py means "the object" (only it may reflect the environment).

buffers: ray-casts the mesh from every frame (Open3D, Embree inside) and writes cache/ in
the format of prepare.py, plus tri.npy (first and second hit triangle per pixel). The diffuse
layer is estimated per triangle as the darkest colour it shows across views, the estimator of
label.py on triangles instead of a cylindrical atlas. Check data/check_*.jpg: the traced
normals must sit on the object.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from capture import nearest_to_axes, read_colmap, table_frame
from common import CACHE, DATA, SCENE, save_poses

SELF_HIT_OFFSET = 2e-3  # metres along the normal, as in prepare.py
BUFFERS = {"depth": ("u2", 1), "normal": ("i2", 3), "diffuse": ("u1", 3), "fbi": ("u1", 3), "vis": ("u1", 1), "rgb": ("u1", 3), "tri": ("i4", 2)}
OBJECT, TABLE = 0, 1


# ---------------------------------------------------------------- mesh


def cloud(points, normals=None):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, np.float64)))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, np.float64))
    return pcd


def segment_object(local, normals, focus, reach, crop):
    """Points of the object standing nearest the camera focus, in table coordinates.

    Raised points within `crop` * reach of the focus are voxel-downsampled and clustered
    (DBSCAN); the cluster with the most points near the focus is the object.
    """
    horizontal = np.hypot(*(local[:, :2] - focus[:2]).T)
    keep = (local[:, 2] > 0.02 * reach) & (horizontal < crop * reach)
    assert keep.sum() > 100, "no points stand above the table near the camera focus; is the object on the table?"
    pcd = cloud(local[keep], None if normals is None else normals[keep]).voxel_down_sample(0.004 * reach)
    labels = np.array(pcd.cluster_dbscan(eps=0.02 * reach, min_points=10))
    points = np.asarray(pcd.points)
    near = np.hypot(*(points[:, :2] - focus[:2]).T) < 0.25 * reach
    counts = [(near & (labels == k)).sum() for k in range(labels.max() + 1)]
    assert counts and max(counts) > 0, "could not find the object near the camera focus"
    return pcd.select_by_index(np.flatnonzero(labels == int(np.argmax(counts))))


def orient_outward(pcd):
    """Give a cloud without normals outward-facing ones (fine for a roughly convex object)."""
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
    points, normals = np.asarray(pcd.points), np.asarray(pcd.normals)
    axis = np.array([*np.median(points[:, :2], axis=0), np.median(points[:, 2])])
    flip = ((points - axis) * normals).sum(axis=1) < 0
    normals[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def poisson(pcd, depth, trim, max_faces):
    """Watertight Poisson surface, trimmed to well-supported parts above the table."""
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.quantile(density, trim))
    points = np.asarray(pcd.points)
    margin = 0.02 * (points.max(axis=0) - points.min(axis=0)).max()
    low, high = points.min(axis=0) - margin, points.max(axis=0) + margin
    low[2] = 0.0
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(low, high))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > max_faces:
        mesh = mesh.simplify_quadric_decimation(max_faces)
    mesh.compute_vertex_normals()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_normals)


def table_grid(half_x, half_y, cell=0.01):
    """Square-cell triangle grid on z = 0 covering [-half_x, half_x] x [-half_y, half_y]."""
    nx, ny = max(1, round(2 * half_x / cell)), max(1, round(2 * half_y / cell))
    x, y = np.meshgrid(np.linspace(-half_x, half_x, nx + 1), np.linspace(-half_y, half_y, ny + 1))
    verts = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    i = (np.arange(ny)[:, None] * (nx + 1) + np.arange(nx)[None]).ravel()
    faces = np.concatenate([np.stack([i, i + 1, i + nx + 2], 1), np.stack([i, i + nx + 2, i + nx + 1], 1)])
    return verts, faces, np.tile([0.0, 0.0, 1.0], (len(verts), 1))


def write_mesh(path, parts):
    """Concatenate (verts, faces, normals) per part into mesh.npz with a per-face part id."""
    verts, faces, normals, part, offset = [], [], [], [], 0
    for k, (v, f, n) in enumerate(parts):
        verts.append(v)
        faces.append(f + offset)
        normals.append(n)
        part.append(np.full(len(f), k, np.uint8))
        offset += len(v)
    np.savez(
        path,
        vertices=np.concatenate(verts).astype(np.float32),
        faces=np.concatenate(faces).astype(np.int32),
        normals=np.concatenate(normals).astype(np.float32),
        part=np.concatenate(part),
    )


def build_mesh(args):
    dense = Path(args.dense)
    camera, by_name, _ = read_colmap(dense / "sparse_txt")
    names = sorted(by_name)
    poses = np.stack([by_name[n] for n in names])
    fused = o3d.io.read_point_cloud(str(dense / "fused.ply"))
    points = np.asarray(fused.points)
    normals = np.asarray(fused.normals) if fused.has_normals() else None
    assert len(points) > 1000, f"{dense / 'fused.ply'} has only {len(points)} points"
    rng = np.random.default_rng(args.seed)

    # Fit the table near where the cameras look, so a wall or the floor cannot win the RANSAC.
    focus_world = nearest_to_axes(poses[:, :, 3], poses[:, :, 2])
    radius = np.median(np.linalg.norm(poses[:, :, 3] - focus_world, axis=1))
    near = np.flatnonzero(np.linalg.norm(points - focus_world, axis=1) < radius)
    assert len(near) > 1000, "too few reconstructed points near the camera focus"
    sample = points[rng.choice(near, min(len(near), 200_000), replace=False)]
    R, origin, focus, reach, _ = table_frame(poses, sample, rng)
    local = (points - origin) @ R.T
    local_normals = None if normals is None else normals @ R.T

    obj = segment_object(local, local_normals, focus, reach, args.crop)
    obj_points = np.asarray(obj.points)
    scale = args.object_height / np.quantile(obj_points[:, 2], 0.99)
    low, high = np.quantile(obj_points[:, :2], [0.01, 0.99], axis=0)
    centre = np.array([*(low + high) / 2, 0.0])

    def to_scene(p):
        return scale * (p - centre)

    obj.points = o3d.utility.Vector3dVector(to_scene(obj_points))
    if not obj.has_normals():
        obj = orient_outward(obj)
    obj_v, obj_f, obj_n = poisson(obj, args.poisson_depth, args.trim, args.max_faces)
    obj_v[:, 2] = np.maximum(obj_v[:, 2], 0)

    # Table: plane inliers within the camera orbit, at least a margin around the object.
    table = to_scene(local[(np.abs(local[:, 2]) < 0.01 * reach) & (np.hypot(*(local[:, :2] - focus[:2]).T) < reach)])
    footprint = np.abs(obj_v[:, :2]).max(axis=0) + 0.05
    seen = np.quantile(np.abs(table[:, :2]), 0.98, axis=0) if len(table) else footprint
    half = np.minimum(np.maximum(seen, footprint), args.table_max)
    table_v, table_f, table_n = table_grid(*half)

    world_to_scene = np.einsum("ij,njk->nik", R, poses[:, :, :3])
    centres = to_scene((poses[:, :, 3] - origin) @ R.T)
    scene_poses = np.concatenate([world_to_scene, centres[:, :, None]], axis=2)

    shutil.rmtree(DATA / "rgb", ignore_errors=True)
    for old in DATA.glob("check_*.jpg"):
        old.unlink()
    (DATA / "rgb").mkdir(parents=True)
    write_mesh(DATA / "mesh.npz", [(obj_v, obj_f, obj_n), (table_v, table_f, table_n)])
    save_poses(DATA / "xy_poses.txt", scene_poses)
    (DATA / "frames.txt").write_text("\n".join(names) + "\n")
    s = args.width / camera["width"]
    height = round(camera["height"] * s)
    for i, name in enumerate(names):
        image = cv2.imread(str(dense / "images" / name))
        assert image is not None, f"cannot read {dense / 'images' / name}"
        cv2.imwrite(str(DATA / "rgb" / f"{i + 1:04d}.png"), cv2.resize(image, (args.width, height), interpolation=cv2.INTER_AREA))
    spec = {
        "width": args.width,
        "height": height,
        "focal": camera["focal"] * s,
        "cx": camera["cx"] * s,
        "cy": camera["cy"] * s,
        "gamma": args.gamma,
        "cans": [{"x": 0.0, "y": 0.0, "radius": float(np.hypot(obj_v[:, 0], obj_v[:, 1]).max()), "height": float(obj_v[:, 2].max())}],
        "source": "video",
        "colmap_to_scene_scale": float(scale),
        "table_half": [float(h) for h in half],
    }
    (DATA / "scene.json").write_text(json.dumps(spec, indent=2))
    print(f"{len(names)} frames, object {len(obj_f)} triangles, {obj_v[:, 2].max():.3f} m tall, "
          f"bounding radius {spec['cans'][0]['radius']:.3f} m; table {2 * half[0]:.2f} x {2 * half[1]:.2f} m")


# ---------------------------------------------------------------- buffers


class Tracer:
    def __init__(self, path):
        mesh = np.load(path)
        self.verts, self.faces, self.normals, self.part = mesh["vertices"], mesh["faces"], mesh["normals"], mesh["part"]
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(self.verts), o3d.core.Tensor(self.faces.astype(np.uint32)))

    def cast(self, origins, dirs):
        """Returns (t (inf on a miss), triangle id (-1 on a miss), barycentrics (N, 3))."""
        rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
        out = self.scene.cast_rays(o3d.core.Tensor(rays))
        t = out["t_hit"].numpy()
        tri = out["primitive_ids"].numpy().astype(np.int64)
        tri[~np.isfinite(t)] = -1
        uv = out["primitive_uvs"].numpy()
        return t, tri, np.stack([1 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=1)

    def trace(self, pose, rays):
        """Geometry buffers of one frame: depth (camera z), normal, vis, first/second triangle."""
        h, w = rays.shape[:2]
        d = (rays @ pose[:, :3].T).reshape(-1, 3)
        length = np.linalg.norm(d, axis=1)
        unit = d / length[:, None]
        origin = np.broadcast_to(pose[:, 3], unit.shape)
        t, tri, bary = self.cast(origin, unit)
        depth = np.zeros(h * w, np.float32)
        normal = np.zeros((h * w, 3), np.float32)
        vis = np.zeros(h * w, np.uint8)
        tris = np.full((h * w, 2), -1, np.int32)
        hit = np.flatnonzero(tri >= 0)
        if len(hit):
            ray = unit[hit]
            n = (self.normals[self.faces[tri[hit]]] * bary[hit][:, :, None]).sum(axis=1)
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
            facing = (n * ray).sum(axis=1, keepdims=True)
            n = np.where(facing > 0, -n, n)
            x = origin[hit] + ray * t[hit, None]
            reflected = ray + 2 * np.abs(facing) * n
            _, tri2, _ = self.cast(x + SELF_HIT_OFFSET * n, reflected)
            depth[hit] = t[hit] / length[hit]
            normal[hit] = n
            vis[hit] = tri2 < 0
            tris[hit, 0] = tri[hit]
            tris[hit, 1] = tri2
        return depth.reshape(h, w), normal.reshape(h, w, 3), vis.reshape(h, w), tris.reshape(h, w, 2)


def face_colours(tracer, rgb, tri, frames):
    """Darkest colour of every triangle across `frames` (display-encoded uint8, (F, 3)).

    Each photo is blurred 3x3 and silhouettes are eroded by a pixel, as in label.py.
    Triangles never seen take the median colour of their part (object or table).
    """
    colour = np.full((len(tracer.faces), 3), 256, np.int16)
    kernel = np.ones((3, 3), np.uint8)
    for i in frames:
        first = tri[i][..., 0]
        part = np.where(first >= 0, tracer.part[np.maximum(first, 0)], 255)
        inner = np.zeros(first.shape, bool)
        for k in np.unique(tracer.part):
            inner |= cv2.erode((part == k).astype(np.uint8), kernel) > 0
        ids = first[inner]
        if not len(ids):
            continue
        values = cv2.blur(np.asarray(rgb[i]), (3, 3))[inner].astype(np.int16)
        order = np.argsort(ids, kind="stable")
        ids, values = ids[order], values[order]
        unique, start = np.unique(ids, return_index=True)
        colour[unique] = np.minimum(colour[unique], np.minimum.reduceat(values, start, axis=0))
    seen = colour[:, 0] < 256
    for k in np.unique(tracer.part):
        mine = tracer.part == k
        fill = np.median(colour[mine & seen], axis=0) if (mine & seen).any() else np.full(3, 128)
        colour[mine & ~seen] = fill
    print(f"diffuse estimate: {seen.mean():.1%} of triangles observed", flush=True)
    return colour.astype(np.uint8)


def write_buffers(args):
    from common import H, W, load_cans, load_poses, pixel_rays

    tracer = Tracer(DATA / "mesh.npz")
    poses = load_poses()
    rays = pixel_rays()
    CACHE.mkdir(exist_ok=True)
    maps = {n: np.lib.format.open_memmap(CACHE / f"{n}.npy", "w+", t, (len(poses), H, W, c)) for n, (t, c) in BUFFERS.items()}
    for i, pose in enumerate(poses):
        depth, normal, vis, tris = tracer.trace(pose, rays)
        maps["depth"][i, ..., 0] = np.clip(depth * 1e4, 0, 65535).astype(np.uint16)
        maps["normal"][i] = (normal * 32767).astype(np.int16)
        maps["vis"][i, ..., 0] = vis
        maps["tri"][i] = tris
        photo = cv2.imread(str(DATA / "rgb" / f"{i + 1:04d}.png"))
        assert photo is not None and photo.shape[:2] == (H, W), f"rgb/{i + 1:04d}.png missing or not {W}x{H}"
        maps["rgb"][i] = photo[..., ::-1]
        if i % 20 == 0:
            print(f"traced {i + 1}/{len(poses)}", flush=True)

    # Every frame trains: the pipeline recovers an environment and has no novel-view test.
    train = np.arange(len(poses))
    test = train[:: max(1, len(poses) // 10)]
    cans = load_cans()
    centre = np.array([cans[0, 0], cans[0, 1], cans[0, 3] / 2]) if cans is not None else np.zeros(3)
    np.savez(CACHE / "split.npz", train=train, test=test, angle=np.zeros(len(test)), centre=centre)

    colour = face_colours(tracer, maps["rgb"], maps["tri"], train)
    for i in range(len(poses)):
        first, second = maps["tri"][i][..., 0], maps["tri"][i][..., 1]
        maps["diffuse"][i] = np.where(first[..., None] >= 0, colour[np.maximum(first, 0)], 0)
        maps["fbi"][i] = np.where(second[..., None] >= 0, colour[np.maximum(second, 0)], 0)
    for m in maps.values():
        m.flush()
    (CACHE / "meta.json").write_text(json.dumps({"frames": len(poses)}))

    for i in np.linspace(0, len(poses) - 1, min(args.checks, len(poses))).astype(int):
        normal = maps["normal"][i].astype(np.float32) / 32767
        hit = maps["depth"][i, ..., 0] > 0
        overlay = maps["rgb"][i].astype(np.float32)
        overlay[hit] = 0.4 * overlay[hit] + 0.6 * (normal[hit] + 1) * 127.5
        cv2.imwrite(str(DATA / f"check_{i + 1:04d}.jpg"), overlay[..., ::-1].astype(np.uint8))
    print(f"wrote {CACHE}; check {DATA}/check_*.jpg")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mesh", help="aligned object + table mesh and scene files from a COLMAP dense workspace")
    m.add_argument("--dense", required=True, help="COLMAP dense workspace from sfm.py (images/, sparse_txt/, fused.ply)")
    m.add_argument("--object-height", type=float, default=0.25, help="real height of the object in metres; sets the scale")
    m.add_argument("--width", type=int, default=640, help="frames are resized to this width")
    m.add_argument("--gamma", type=float, default=2.2, help="display gamma of the video")
    m.add_argument("--crop", type=float, default=0.6, help="object search radius, as a fraction of the camera orbit radius")
    m.add_argument("--poisson-depth", type=int, default=9)
    m.add_argument("--trim", type=float, default=0.05, help="drop this quantile of least-supported Poisson vertices")
    m.add_argument("--max-faces", type=int, default=300_000)
    m.add_argument("--table-max", type=float, default=1.0, help="largest table half-size in metres")
    m.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("buffers", help="ray-cast cache/ from data/mesh.npz")
    b.add_argument("--checks", type=int, default=6, help="number of check images")
    args = ap.parse_args()
    # Without SCENE the paths are the Cans dataset's (cans_data/, cache/), which must not be overwritten.
    assert SCENE, "set SCENE=<name>; this writes scenes/<name>/data and scenes/<name>/cache"
    {"mesh": build_mesh, "buffers": write_buffers}[args.command](args)


if __name__ == "__main__":
    main()
