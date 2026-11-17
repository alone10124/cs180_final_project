"""Synthetic scenes with a known environment: cylinders on a table, lit by an HDR panorama.

  python synth.py --name one --layout one      # one mirror can
  python synth.py --name two --layout two      # same can plus a second one
  python synth.py --name label --layout one --label   # printed label on the can

Writes scenes/<name>/data/ (scene.json, poses, ground-truth environment) and
scenes/<name>/cache/ in the format of prepare.py, so `SCENE=<name> python ...`
runs the rest of the code unchanged. Photos come from the analytic ray tracer
here, or from Blender renders of the same scene with --photos (see blender_scene.py).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import ROOT, equirect_dirs, equirect_sample, make_split, save_poses

ENV_W, ENV_H = 2000, 1000
CAN_RADIUS, CAN_HEIGHT, TOP_RADIUS = 0.033, 0.122, 0.027
LAYOUTS = {"one": [(-0.06, 0.0)], "two": [(-0.06, 0.0), (0.06, 0.0)]}
BUFFERS = {"depth": ("u2", 1), "normal": ("i2", 3), "diffuse": ("u1", 3), "fbi": ("u1", 3), "vis": ("u1", 1), "rgb": ("u1", 3)}
EPS = 1e-6
OFFSET = 1e-5
MAX_BOUNCES = 3


def make_env(width=ENV_W, height=ENV_H):
    """A box room seen from a table top: window, ceiling lights, door, picture, sofa, shelf.

    Linear radiance, values above 1 for the lights and window.
    """
    d = equirect_dirs(width, height)
    lo = np.array([-2.2, -1.8, -0.8])
    hi = np.array([2.8, 2.2, 1.9])
    with np.errstate(divide="ignore"):
        t = np.where(d > 0, hi / d, np.where(d < 0, lo / d, np.inf))
    axis = np.argmin(t, axis=-1)
    p = d * t.min(axis=-1, keepdims=True)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    positive = np.take_along_axis(d, axis[..., None], axis=-1)[..., 0] > 0
    env = np.zeros((height, width, 3))

    def paint(mask, colour):
        env[mask] = colour

    def box(a, a0, a1, b, b0, b1):
        return (a > a0) & (a < a1) & (b > b0) & (b < b1)

    east, west = (axis == 0) & positive, (axis == 0) & ~positive
    north, south = (axis == 1) & positive, (axis == 1) & ~positive
    ceiling, floor = (axis == 2) & positive, (axis == 2) & ~positive
    paint(east, (0.60, 0.55, 0.48))
    paint(west, (0.45, 0.52, 0.58))
    paint(north, (0.55, 0.58, 0.50))
    paint(south, (0.58, 0.50, 0.50))
    paint((axis != 2) & (z < -0.7), (0.12, 0.10, 0.09))
    paint(ceiling, (0.75, 0.75, 0.75))
    plank = 0.85 + 0.15 * (np.floor(y / 0.15) % 2)
    env[floor] = np.array([0.30, 0.20, 0.12]) * plank[floor][:, None]

    paint(ceiling & (box(x, -1.0, -0.4, y, -0.3, 0.3) | box(x, 1.0, 1.6, y, -0.3, 0.3)), (12.0, 11.5, 10.5))
    window = east & box(y, -1.0, 0.8, z, 0.1, 1.3)
    paint(window, (2.5, 3.2, 4.5))
    paint(window & ((np.abs(y + 0.1) < 0.03) | (np.abs(z - 0.7) < 0.03)), (0.2, 0.2, 0.2))
    paint(south & box(x, 0.8, 1.7, z, -0.8, 1.2), (0.25, 0.13, 0.06))
    paint(north & box(x, -1.25, 0.25, z, 0.15, 1.15), (0.05, 0.05, 0.05))
    for i, colour in enumerate([(0.8, 0.1, 0.1), (0.9, 0.7, 0.1), (0.1, 0.3, 0.8)]):
        paint(north & box(x, -1.2 + 0.45 * i, -0.75 + 0.45 * i, z, 0.2, 1.1), colour)
    paint(west & box(y, -1.2, 0.9, z, -0.8, -0.1), (0.10, 0.30, 0.22))
    shelf = north & box(x, 1.2, 2.4, z, -0.8, 1.0)
    book = np.floor(x / 0.06).astype(int)
    colours = np.array([(0.6, 0.2, 0.2), (0.2, 0.4, 0.6), (0.7, 0.6, 0.3), (0.3, 0.5, 0.3), (0.5, 0.3, 0.5)])
    env[shelf] = colours[book[shelf] % len(colours)]
    paint(shelf & ((z + 0.8) % 0.36 < 0.04), (0.08, 0.06, 0.05))
    return env.astype(np.float32)


def blur_env(env, degrees):
    """Gaussian blur with horizontal wrap; stands in for a rougher reflection lobe."""
    if degrees <= 0:
        return env
    sigma = degrees / 360 * env.shape[1]
    pad = int(3 * sigma) + 1
    wide = np.concatenate([env[:, -pad:], env, env[:, :pad]], axis=1)
    return cv2.GaussianBlur(wide, (0, 0), sigma)[:, pad:-pad]


def camera_poses(count, target, radius=0.45, elevation=(0, 60)):
    """Camera-to-world poses on a spiral around `target` (x right, y down, z forward)."""
    i = np.arange(count)
    el = np.radians(elevation[0] + (elevation[1] - elevation[0]) * (i + 0.5) / count)
    az = i * np.pi * (3 - np.sqrt(5))
    r = radius * (1 + 0.1 * np.sin(1.7 * i))
    centre = target + r[:, None] * np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    forward = target - centre
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    down = np.cross(forward, right)
    return np.concatenate([np.stack([right, down, forward], axis=2), centre[:, :, None]], axis=2)


def pixel_rays(spec, scale=1):
    w, h = spec["width"] * scale, spec["height"] * scale
    u, v = np.meshgrid((np.arange(w) + 0.5) / scale, (np.arange(h) + 0.5) / scale)
    return np.stack([(u - spec["cx"]) / spec["focal"], (v - spec["cy"]) / spec["focal"], np.ones_like(u)], axis=-1)


def intersect(o, d, spec):
    """Nearest hit of rays with the table and the cans.

    Returns (t, normal, obj): obj is 0 for a miss, 1 for the table, k + 2 for can k.
    """
    t = np.full(len(o), np.inf)
    normal = np.zeros((len(o), 3))
    obj = np.zeros(len(o), int)
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = -o[:, 2] / d[:, 2]
        p = o[:, :2] + tf[:, None] * d[:, :2]
        hit = (tf > EPS) & (np.abs(p).max(axis=1) < spec["floor_half"])
        t[hit], normal[hit], obj[hit] = tf[hit], (0, 0, 1), 1
        for k, can in enumerate(spec["cans"]):
            ox, oy = o[:, 0] - can["x"], o[:, 1] - can["y"]
            radius, top, neck = can["radius"], can.get("top_radius", can["radius"]), can.get("neck", 0)
            z0 = can["height"] - neck
            # body: vertical cylinder from the table to z0
            a = d[:, 0] ** 2 + d[:, 1] ** 2
            b = ox * d[:, 0] + oy * d[:, 1]
            disc = b * b - a * (ox**2 + oy**2 - radius**2)
            ts = (-b - np.sqrt(disc)) / a
            z = o[:, 2] + ts * d[:, 2]
            hit = (disc > 0) & (ts > EPS) & (z > 0) & (z < z0) & (ts < t)
            wall = np.stack([ox + ts * d[:, 0], oy + ts * d[:, 1], np.zeros(len(o))], axis=1) / radius
            t[hit], normal[hit], obj[hit] = ts[hit], wall[hit], k + 2
            # neck: cone frustum narrowing from `radius` at z0 to `top` at the lid
            if neck > 0:
                slope = (radius - top) / neck
                A = radius - slope * (o[:, 2] - z0)
                B = -slope * d[:, 2]
                qa = a - B * B
                qb = b - A * B
                root = np.sqrt(qb * qb - qa * (ox**2 + oy**2 - A * A))
                for tn in ((-qb - root) / qa, (-qb + root) / qa):
                    z = o[:, 2] + tn * d[:, 2]
                    hit = (tn > EPS) & (z >= z0) & (z < can["height"]) & (tn < t)
                    rho = radius - slope * (z - z0)
                    cone = np.stack([(ox + tn * d[:, 0]) / rho, (oy + tn * d[:, 1]) / rho, np.full(len(o), slope)], axis=1)
                    cone /= np.sqrt(1 + slope**2)
                    hit &= (cone * d).sum(axis=1) < 0
                    t[hit], normal[hit], obj[hit] = tn[hit], cone[hit], k + 2
            # lid: flat disc
            tc = (can["height"] - o[:, 2]) / d[:, 2]
            inside = (ox + tc * d[:, 0]) ** 2 + (oy + tc * d[:, 1]) ** 2 < top**2
            hit = (tc > EPS) & inside & (d[:, 2] < 0) & (tc < t)
            t[hit], normal[hit], obj[hit] = tc[hit], (0, 0, 1), k + 2
    return t, normal, obj


def material(p, obj, spec):
    """Linear diffuse colour (N, 3) and mirror reflectance (N,) at surface points."""
    diffuse = np.zeros((len(p), 3))
    ks = np.zeros(len(p))
    table = obj == 1
    check = (np.floor(p[:, 0] / 0.05) + np.floor(p[:, 1] / 0.05)) % 2
    diffuse[table] = np.where(check[table, None] == 1, (0.30, 0.28, 0.25), (0.22, 0.20, 0.18))
    ks[obj >= 2] = spec["reflectance"]
    if spec["label"]:
        for k, can in enumerate(spec["cans"]):
            phi = np.arctan2(p[:, 1] - can["y"], p[:, 0] - can["x"])
            band = (obj == k + 2) & (p[:, 2] > 0.02) & (p[:, 2] < 0.10)
            stripe = np.abs(p[:, 2] - 0.06 - 0.012 * np.sin(3 * phi)) < 0.012
            diffuse[band] = np.where(stripe[band, None], (0.60, 0.60, 0.60), (0.55, 0.03, 0.04))
            ks[band] = spec["label_reflectance"]
    return diffuse, ks


def shade(o, d, spec, env, depth=0):
    """Radiance along rays: diffuse colour plus mirror reflection, traced recursively."""
    t, normal, obj = intersect(o, d, spec)
    colour = np.zeros((len(o), 3))
    miss = obj == 0
    colour[miss] = equirect_sample(env, d[miss])
    index = np.nonzero(~miss)[0]
    if len(index):
        p = o[index] + t[index, None] * d[index]
        diffuse, ks = material(p, obj[index], spec)
        colour[index] = diffuse
        shiny = ks > 0
        if shiny.any():
            n, v = normal[index][shiny], d[index][shiny]
            r = v - 2 * (v * n).sum(axis=1, keepdims=True) * n
            if depth < MAX_BOUNCES:
                seen = shade(p[shiny] + OFFSET * n, r, spec, env, depth + 1)
            else:
                seen = equirect_sample(env, r)
            colour[index[shiny]] += ks[shiny, None] * seen
    return colour


def encode(linear, gamma):
    return (np.clip(linear, 0, 1) ** (1 / gamma) * 255 + 0.5).astype(np.uint8)


def geometry(pose, spec):
    """The buffers of prepare.py for one frame, from the analytic scene."""
    h, w = spec["height"], spec["width"]
    R, origin = pose[:, :3], pose[:, 3]
    d = (pixel_rays(spec) @ R.T).reshape(-1, 3)
    length = np.linalg.norm(d, axis=1)
    unit = d / length[:, None]
    o = np.broadcast_to(origin, unit.shape)
    t, normal, obj = intersect(o, unit, spec)
    hit = obj > 0
    depth = np.where(hit, t / length, 0)
    p = o + np.where(hit, t, 0)[:, None] * unit
    diffuse, _ = material(p, obj, spec)
    r = unit - 2 * (unit * normal).sum(axis=1, keepdims=True) * normal
    t2, _, obj2 = intersect(p + OFFSET * normal, r, spec)
    second, _ = material(p + OFFSET * normal + np.where(obj2 > 0, t2, 0)[:, None] * r, obj2, spec)
    gamma = spec["gamma"]
    return {
        "depth": depth.reshape(h, w),
        "normal": (normal * hit[:, None]).reshape(h, w, 3),
        "diffuse": (encode(diffuse, gamma) * hit[:, None]).reshape(h, w, 3),
        "fbi": (encode(second, gamma) * (hit & (obj2 > 0))[:, None]).reshape(h, w, 3),
        "vis": (hit & (obj2 == 0)).reshape(h, w).astype(np.uint8),
        "obj": obj.reshape(h, w),
    }


def render(pose, spec, env, ss):
    """Display-encoded photo of the scene, supersampled ss x ss."""
    h, w = spec["height"], spec["width"]
    d = (pixel_rays(spec, ss) @ pose[:, :3].T).reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    image = shade(np.broadcast_to(pose[:, 3], d.shape), d, spec, env).reshape(h * ss, w * ss, 3)
    image = cv2.resize(np.clip(image, 0, 1), (w, h), interpolation=cv2.INTER_AREA)
    return encode(image, spec["gamma"])


def write_scene(base, spec, poses, photo, floor_from_photo=False):
    """Write data/ and cache/ for a scene. `photo(i)` returns frame i as display-encoded RGB uint8."""
    data, cache = base / "data", base / "cache"
    data.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=True)
    (data / "scene.json").write_text(json.dumps(spec, indent=2))
    save_poses(data / "xy_poses.txt", poses)
    centre = np.array([np.mean([c["x"] for c in spec["cans"]]), np.mean([c["y"] for c in spec["cans"]]), 0.06])
    train, test, angle = make_split(poses, centre)
    np.savez(cache / "split.npz", train=train, test=test, angle=angle, centre=centre)
    shape = (len(poses), spec["height"], spec["width"])
    maps = {n: np.lib.format.open_memmap(cache / f"{n}.npy", "w+", t, (*shape, c)) for n, (t, c) in BUFFERS.items()}
    for i, pose in enumerate(poses):
        g = geometry(pose, spec)
        rgb = photo(i)
        if floor_from_photo:
            g["diffuse"] = np.where(g["obj"][..., None] == 1, rgb, g["diffuse"])
        maps["depth"][i, ..., 0] = np.clip(g["depth"] * 1e4, 0, 65535).astype(np.uint16)
        maps["normal"][i] = (g["normal"] * 32767).astype(np.int16)
        maps["diffuse"][i] = g["diffuse"]
        maps["fbi"][i] = g["fbi"]
        maps["vis"][i, ..., 0] = g["vis"]
        maps["rgb"][i] = rgb
        if i % 20 == 0:
            print(f"{i + 1}/{len(poses)}", flush=True)
    for m in maps.values():
        m.flush()
    (cache / "meta.json").write_text(json.dumps({"frames": len(poses)}))


def read_hdr(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None, f"cannot read {path}"
    return image[..., 2::-1].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="scene directory under scenes/")
    ap.add_argument("--layout", choices=list(LAYOUTS), default="one")
    ap.add_argument("--label", action="store_true", help="print a diffuse label band on each can")
    ap.add_argument("--neck", type=float, default=0, help="height of a tapered neck (a real can: 0.014); 0 is a pure cylinder")
    ap.add_argument("--table", type=float, default=0.15, help="half-size of the square table top in metres")
    ap.add_argument("--elevation", type=float, nargs=2, default=(0, 60), help="camera elevation range in degrees")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor of the analytic photos")
    ap.add_argument("--blur-deg", type=float, default=0, help="blur the reflected environment (rough metal)")
    ap.add_argument("--env", default=None, help="equirectangular .hdr to use instead of the built-in room")
    ap.add_argument("--photos", default=None, help="directory of Blender renders frame_0001.hdr ...")
    args = ap.parse_args()

    base = ROOT / "scenes" / args.name
    spec = {
        "width": args.width,
        "height": args.height,
        "focal": 541.961 * args.width / 640,
        "cx": args.width / 2,
        "cy": args.height / 2,
        "gamma": 1.6,
        "cans": [
            {"x": x, "y": y, "radius": CAN_RADIUS, "height": CAN_HEIGHT, "neck": args.neck, "top_radius": TOP_RADIUS if args.neck else CAN_RADIUS}
            for x, y in LAYOUTS[args.layout]
        ],
        "floor_half": args.table,
        "reflectance": 0.9,
        "label": args.label,
        "label_reflectance": 0.3,
        "blur_deg": args.blur_deg,
        "env": "env_gt.hdr",
    }
    env = make_env() if args.env is None else cv2.resize(read_hdr(args.env), (ENV_W, ENV_H), interpolation=cv2.INTER_AREA)
    (base / "data").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(base / "data" / "env_gt.hdr"), env[..., ::-1])
    reflected = blur_env(env, args.blur_deg)
    poses = camera_poses(args.frames, np.array([0, 0, 0.06]), elevation=args.elevation)

    def photo(i):
        if args.photos is None:
            return render(poses[i], spec, reflected, args.ss)
        image = read_hdr(Path(args.photos) / f"frame_{i + 1:04d}.hdr")
        image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
        return encode(image, spec["gamma"])

    write_scene(base, spec, poses, photo)
    print(f"wrote {base}")


if __name__ == "__main__":
    main()
