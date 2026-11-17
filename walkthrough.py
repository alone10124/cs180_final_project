"""2.5D walkthrough of a recovered panorama, after Tour Into the Picture.

  SCENE=two python walkthrough.py --input both_telea --room 5 4 2.7 --eye 0.8

--input is a results/ directory holding filled.png (from inpaint.py), or a path to
any equirectangular image.

1. Horizon and vanishing points. The panorama is gravity-aligned (z is the table
   normal), so the horizon is its middle row and the vertical vanishing point is the
   zenith. The two horizontal vanishing points are found by voting: every straight
   edge in a ring of perspective views back-projects to a plane through the viewpoint;
   if the edge is horizontal in the room, its direction is that plane's normal crossed
   with z. A box room's edges vote for two perpendicular headings.
2. Layout. A box aligned with those headings is placed around the viewpoint. Its size
   and the eye height are parameters (--room, --eye, --offset): one panorama does not
   determine them, so treat the result as a visual aid, not a measurement.
3. Walkthrough. The panorama is projected onto the box from the original viewpoint,
   and new views are rendered from a camera moving inside it.

Writes results/walkthrough/: vanishing.png, layout.json, still_*.png, walkthrough.mp4.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import RESULTS, equirect_sample


def camera_rays(yaw, pitch, fov, width, height):
    """World-space unit rays of a perspective camera at heading `yaw` (radians, about z)."""
    f = 0.5 * width / np.tan(np.radians(fov) / 2)
    u, v = np.meshgrid(np.arange(width) + 0.5 - width / 2, np.arange(height) + 0.5 - height / 2)
    forward = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.cross(forward, right)
    d = f * forward + u[..., None] * right + v[..., None] * down
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def perspective(pano, yaw, pitch=0.0, fov=90, width=480, height=480):
    return equirect_sample(pano, camera_rays(yaw, pitch, fov, width, height)).astype(np.float32)


def room_heading(pano, views=8, size=480):
    """Heading (radians, in [0, pi/2)) of the room's walls, from horizontal-edge votes.

    Returns (heading, list of (yaw, segments, directions) for the debug figure).
    """
    votes = np.zeros(90)
    found = []
    for yaw in np.arange(views) * 2 * np.pi / views:
        rays = camera_rays(yaw, 0.0, 90, size, size)
        gray = cv2.cvtColor((np.clip(perspective(pano, yaw, width=size, height=size), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lines = cv2.HoughLinesP(cv2.Canny(gray, 40, 120), 1, np.pi / 360, 40, minLineLength=size // 8, maxLineGap=4)
        segments, directions = [], []
        for x1, y1, x2, y2 in [] if lines is None else np.asarray(lines).reshape(-1, 4):
            a, b = rays[min(y1, size - 1), min(x1, size - 1)], rays[min(y2, size - 1), min(x2, size - 1)]
            normal = np.cross(a, b)
            normal /= np.linalg.norm(normal) + 1e-12
            if abs(normal[2]) < 0.2:  # the plane contains z: a vertical edge
                continue
            direction = np.cross([0, 0, 1.0], normal)
            angle = np.degrees(np.arctan2(direction[1], direction[0])) % 90
            votes[int(angle) % 90] += np.hypot(x2 - x1, y2 - y1)
            segments.append((x1, y1, x2, y2))
            directions.append(np.degrees(np.arctan2(direction[1], direction[0])) % 180)
        found.append((yaw, segments, directions))
    smooth = np.convolve(np.concatenate([votes[-3:], votes, votes[:3]]), np.ones(7) / 7, mode="valid")
    return np.radians(np.argmax(smooth) + 0.5), found


def vanishing_figure(pano, heading, found, size=480):
    """Perspective views with horizontal edges coloured by vanishing point, and the horizon."""
    tiles = []
    for yaw, segments, directions in found[::2]:
        view = (np.clip(perspective(pano, yaw, width=size, height=size), 0, 1) * 255).astype(np.uint8).copy()
        for (x1, y1, x2, y2), angle in zip(segments, directions):
            off = (angle - np.degrees(heading)) % 180
            first = min(off, 180 - off) < 15
            second = abs(off - 90) < 15
            colour = (230, 60, 60) if first else (60, 120, 230) if second else (150, 150, 150)
            cv2.line(view, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2, cv2.LINE_AA)
        cv2.line(view, (0, size // 2), (size, size // 2), (240, 220, 40), 1, cv2.LINE_AA)
        tiles.append(view)
    return np.hstack(tiles)


def box_view(pano, lo, hi, rotation, position, yaw, fov, width, height):
    """Render the panorama-textured box from a camera at `position` (world, viewpoint at origin)."""
    d = camera_rays(yaw, 0.0, fov, width, height) @ rotation.T  # into the box frame
    o = rotation @ position
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d > 0, (hi - o) / d, np.where(d < 0, (lo - o) / d, np.inf)).min(axis=-1, keepdims=True)
    hit = o + t * d
    hit /= np.linalg.norm(hit, axis=-1, keepdims=True)
    return equirect_sample(pano, hit @ rotation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="results/ directory with filled.png, or an image path")
    ap.add_argument("--room", type=float, nargs=3, default=(5.0, 4.0, 2.7), help="box length, width, height in metres")
    ap.add_argument("--eye", type=float, default=0.8, help="height of the viewpoint above the floor")
    ap.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0), help="viewpoint offset from the box centre")
    ap.add_argument("--heading", type=float, default=None, help="wall heading in degrees; default is estimated")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--fov", type=float, default=75)
    ap.add_argument("--size", type=int, nargs=2, default=(640, 480))
    args = ap.parse_args()

    path = RESULTS / args.input / "filled.png" if (RESULTS / args.input / "filled.png").exists() else Path(args.input)
    image = cv2.imread(str(path))
    assert image is not None, f"cannot read {path}"
    pano = image[..., ::-1].astype(np.float32) / 255
    out = RESULTS / "walkthrough"
    out.mkdir(parents=True, exist_ok=True)

    heading, found = room_heading(pano)
    cv2.imwrite(str(out / "vanishing.png"), vanishing_figure(pano, heading, found)[..., ::-1])
    if args.heading is not None:
        heading = np.radians(args.heading)
    c, s = np.cos(heading), np.sin(heading)
    rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])  # world -> box frame
    length, width, height = args.room
    lo = np.array([-length / 2 - args.offset[0], -width / 2 - args.offset[1], -args.eye])
    hi = np.array([length / 2 - args.offset[0], width / 2 - args.offset[1], height - args.eye])
    layout = {
        "horizon": "panorama middle row (z = 0)",
        "vanishing_directions": [[float(c), float(s), 0], [float(-s), float(c), 0], [0, 0, 1]],
        "heading_deg": float(np.degrees(heading)),
        "box_min": lo.tolist(),
        "box_max": hi.tolist(),
    }
    (out / "layout.json").write_text(json.dumps(layout, indent=2))

    writer = cv2.VideoWriter(str(out / "walkthrough.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, tuple(args.size))
    radius = 0.3 * min(length, width) / 2
    for i in range(args.frames):
        phase = 2 * np.pi * i / args.frames
        position = rotation.T @ np.array([radius * np.cos(phase), radius * np.sin(phase), 0.15 * np.sin(2 * phase)])
        yaw = heading + phase + 0.5 * np.pi
        frame = box_view(pano, lo, hi, rotation, position, yaw, args.fov, *args.size)
        frame = (np.clip(frame, 0, 1)[..., ::-1] * 255).astype(np.uint8)
        writer.write(frame)
        if i % (args.frames // 4) == 0:
            cv2.imwrite(str(out / f"still_{i:03d}.png"), frame)
    writer.release()
    print(json.dumps(layout, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
