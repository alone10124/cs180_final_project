"""Ray trace per-frame geometry buffers from the mesh (run once).

Writes one memory-mapped array per buffer into cache/, each shaped (N, H, W, C):
  depth  uint16  camera z * 1e4; 0 where no geometry
  normal int16   world-space shading normal * 32767
  diffuse uint8  texture rendered into the view, display-encoded RGB (D_P)
  fbi    uint8   first-bounce image: diffuse colour at the 2nd mirror hit (Sec 3.3)
  vis    uint8   V(u) of Eq. 4: 1 if the reflected ray escapes the scene
  rgb    uint8   the input photo, display-encoded RGB (G_P)
"""
import argparse
import json
import time

import cv2
import numpy as np
from embreex import rtcore_scene as rtcs
from embreex.mesh_construction import TriangleMesh

from common import CACHE, DATA, H, W, load_poses, make_split, object_centre, pixel_rays

BUFFERS = {
    "depth": ("u2", 1),
    "normal": ("i2", 3),
    "diffuse": ("u1", 3),
    "fbi": ("u1", 3),
    "vis": ("u1", 1),
    "rgb": ("u1", 3),
}
SELF_HIT_OFFSET = 2e-3  # metres along the normal, about one mesh triangle


def read_mesh():
    path = DATA / "xy_mesh.ply"
    with open(path, encoding="ascii") as f:
        header = 0
        for line in f:
            header += 1
            if line.startswith("element vertex"):
                num_v = int(line.split()[-1])
            elif line.startswith("element face"):
                num_f = int(line.split()[-1])
            elif line.startswith("end_header"):
                break
    v = np.loadtxt(path, skiprows=header, max_rows=num_v, dtype=np.float32)
    f = np.loadtxt(path, skiprows=header + num_v, max_rows=num_f, dtype=np.int64)
    return v[:, :3], v[:, 3:6], v[:, 6:8], f[:, 1:4]


class Tracer:
    def __init__(self, verts, norms, uvs, faces, texture):
        self.tri_v = verts[faces]
        self.tri_n = norms[faces]
        self.tri_uv = uvs[faces]
        self.tex = texture
        self.scene = rtcs.EmbreeScene()
        self.mesh = TriangleMesh(self.scene, self.tri_v)

    def trace(self, origins, dirs):
        """Return (hit mask, prim id, barycentrics (N, 3), ray t)."""
        out = self.scene.run(origins.astype(np.float32), dirs.astype(np.float32), output=1)
        hit = out["geomID"] != -1
        bary = np.stack([1 - out["u"] - out["v"], out["u"], out["v"]], axis=-1)
        return hit, out["primID"], bary, out["tfar"]

    def attr(self, table, prim, bary):
        return (table[prim] * bary[:, None, :].transpose(0, 2, 1)).sum(axis=1)

    def texture(self, prim, bary):
        uv = self.attr(self.tri_uv, prim, bary)
        h, w = self.tex.shape[:2]
        mx = (uv[:, 0] * w - 0.5).astype(np.float32)
        my = ((1 - uv[:, 1]) * h - 0.5).astype(np.float32)
        # cv2.remap rejects outputs wider than 32767 pixels.
        chunks = range(0, len(mx), 30000)
        return np.concatenate(
            [cv2.remap(self.tex, mx[i : i + 30000][None], my[i : i + 30000][None], cv2.INTER_LINEAR)[0] for i in chunks]
        )


def trace_frame(tracer, pose, rays):
    R, t = pose[:3, :3], pose[:3, 3]
    d = (rays @ R.T).reshape(-1, 3)  # world direction with camera z = 1
    length = np.linalg.norm(d, axis=1)
    unit = d / length[:, None]
    org = np.broadcast_to(t, unit.shape)

    hit, prim, bary, t_hit = tracer.trace(org, unit)
    depth = np.zeros((H, W), np.float32)
    normal = np.zeros((H, W, 3), np.float32)
    diffuse = np.zeros((H, W, 3), np.uint8)
    fbi = np.zeros((H, W, 3), np.uint8)
    vis = np.zeros((H, W), np.uint8)
    if not hit.any():
        return depth, normal, diffuse, fbi, vis

    idx = np.nonzero(hit)[0]
    p, b, ray = prim[idx], bary[idx], unit[idx]
    n = tracer.attr(tracer.tri_n, p, b)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
    facing = (n * ray).sum(axis=1, keepdims=True)
    n = np.where(facing > 0, -n, n)
    facing = np.abs(facing)

    x = org[idx] + ray * t_hit[idx][:, None]
    refl = ray + 2 * facing * n
    hit2, p2, b2, _ = tracer.trace(x + SELF_HIT_OFFSET * n, refl)

    px = np.unravel_index(idx, (H, W))
    depth[px] = (t_hit[idx] / length[idx]).astype(np.float32)
    normal[px] = n
    diffuse[px] = np.clip(tracer.texture(p, b), 0, 255).astype(np.uint8)
    vis[px] = (~hit2).astype(np.uint8)
    if hit2.any():
        second = (px[0][hit2], px[1][hit2])
        fbi[second] = np.clip(tracer.texture(p2[hit2], b2[hit2]), 0, 255).astype(np.uint8)
    return depth, normal, diffuse, fbi, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only the first N frames")
    args = ap.parse_args()

    poses = load_poses()
    verts, norms, uvs, faces = read_mesh()
    tex = cv2.imread(str(DATA / "texturemapflip.png"), cv2.IMREAD_COLOR)[..., ::-1].copy()
    # Fill empty atlas gutters so bilinear lookups at chart edges stay on-colour.
    empty = (tex.sum(axis=2) == 0).astype(np.uint8)
    grown = cv2.dilate(tex, np.ones((5, 5), np.uint8))
    tex = np.where(empty[..., None] == 1, grown, tex)

    centre = object_centre(verts)
    train, test, angle = make_split(poses, centre)
    CACHE.mkdir(exist_ok=True)
    np.savez(CACHE / "split.npz", train=train, test=test, angle=angle, centre=centre)
    print(f"train {len(train)}, test {len(test)}, nearest-train angle "
          f"median {np.median(angle):.1f} max {angle.max():.1f} deg")

    count = args.limit or len(poses)
    maps = {}
    for name, (dtype, ch) in BUFFERS.items():
        shape = (len(poses), H, W, ch)
        maps[name] = np.lib.format.open_memmap(CACHE / f"{name}.npy", "w+", dtype, shape)

    tracer = Tracer(verts, norms, uvs, faces, tex)
    rays = pixel_rays()
    start = time.time()
    for i in range(count):
        depth, normal, diffuse, fbi, vis = trace_frame(tracer, poses[i], rays)
        maps["depth"][i, ..., 0] = np.clip(depth * 1e4, 0, 65535).astype(np.uint16)
        maps["normal"][i] = (normal * 32767).astype(np.int16)
        maps["diffuse"][i] = diffuse
        maps["fbi"][i] = fbi
        maps["vis"][i, ..., 0] = vis
        maps["rgb"][i] = cv2.imread(str(DATA / "rgb" / f"{i + 1:04d}.png"))[..., ::-1]
        if i % 100 == 0:
            rate = (i + 1) / (time.time() - start)
            print(f"{i + 1}/{count}  {rate:.2f} frames/s", flush=True)
    for m in maps.values():
        m.flush()
    (CACHE / "meta.json").write_text(json.dumps({"frames": count}))
    print("done")


if __name__ == "__main__":
    main()
