"""Screenshots and evaluation of a Pano2Room 3DGS scene.

Run in the pano2room env (it imports Pano2Room's own code):
  python p2r_render.py --ply results/p2r_cans/3DGS.ply --pano results/p2r_cans/input_panorama.png \
      --valid results/p2r_cans/valid_mask.png --out results/p2r_cans

Camera handling is copied from pano2room.py: poses are world-to-camera in Pano2Room's mesh
frame (panorama centre at the origin, y vertical), and the w2c -> c2w conversion is the one
used for its training and evaluation cameras.

Writes <out>/shots/*.png (RGB and depth per view), shots/contact_sheet.png, eval.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
P2R = ROOT / "third_party" / "Pano2Room"
sys.path.insert(0, str(P2R))

from gaussian_renderer import render  # noqa: E402
from modules.equilib import cube2equi  # noqa: E402
from scene import GaussianModel  # noqa: E402
from scene.arguments import GSParams  # noqa: E402
from scene.dataset_readers import loadCamerasFromData  # noqa: E402
from utils.functions import colorize_single_channel_image, get_cubemap_views_world_to_cam, rot_x_world_to_cam, rot_y_world_to_cam  # noqa: E402
from utils.graphics import focal2fov  # noqa: E402

H = W = 512
SCENE_DEPTH_MAX = 4.0228885328450446  # pano2room.py, used to colourise depth


def to_frame(mesh_pose):
    """pano2room.py, run(): mesh-frame w2c -> the c2w frame dict its Scene expects."""
    pose_44 = mesh_pose.clone().float()
    pose_44[0:1, :] *= -1
    pose_44[1:2, :] *= -1
    Rw2c = pose_44[:3, :3].cpu().numpy()
    Tw2c = pose_44[:3, 3:].cpu().numpy()
    yz_reverse = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Rc2w = np.matmul(yz_reverse, Rw2c).T
    Tc2w = -np.matmul(Rc2w, np.matmul(yz_reverse, Tw2c))
    Pc2w = np.concatenate((np.concatenate((Rc2w, Tc2w), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
    blank = Image.new("RGB", (W, H))
    return {"image": blank, "transform_matrix": Pc2w.tolist(), "fovx": focal2fov(256, W), "mesh_pose": mesh_pose}


def translate(c):
    """w2c of a camera at position c with identity rotation."""
    t = torch.eye(4)
    t[:3, 3] = -torch.tensor(c, dtype=torch.float32)
    return t


def render_views(gaussians, opt, poses):
    cams = loadCamerasFromData({"frames": [to_frame(p) for p in poses]}, white_background=True)
    bg = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    out = []
    with torch.no_grad():
        for cam, _ in cams:
            r = render(cam, gaussians, opt, bg)
            rgb = r["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            out.append((rgb, r["depth"].squeeze().cpu().numpy()))
    return out


def psnr(a, b, mask):
    if mask.sum() == 0:
        return None
    mse = ((a - b) ** 2)[mask].mean()
    return float(-10 * np.log10(max(mse, 1e-12)))


def ssim_masked(a, b, mask):
    from skimage.metrics import structural_similarity

    if mask.sum() == 0:
        return None
    _, smap = structural_similarity(a, b, channel_axis=2, data_range=1.0, full=True)
    return float(smap.mean(axis=2)[mask].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--pano", required=True, help="the panorama given to Pano2Room")
    ap.add_argument("--valid", default=None, help="mask PNG, white where the panorama was observed (not inpainted)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    shots = out / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    ply, pano_path = Path(args.ply).resolve(), Path(args.pano).resolve()
    valid_path = Path(args.valid).resolve() if args.valid else None
    os.chdir(P2R)

    opt = GSParams()
    gaussians = GaussianModel(opt.sh_degree)
    gaussians.load_ply(str(ply))
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    print(f"{len(xyz)} Gaussians, extent min {xyz.min(0).round(2)} max {xyz.max(0).round(2)}")

    cube = get_cubemap_views_world_to_cam()
    face_names = ["front", "right", "back", "left", "up", "down"]  # order of get_cubemap_views_world_to_cam
    views = {}
    # 1. The six cube faces at the panorama centre: what Pano2Room was trained to reproduce.
    for name, w2c in zip(face_names, cube):
        views[f"centre_{name}"] = w2c.clone()
    # 2. Novel views: step away from the centre and look around (8 headings), y is vertical.
    for step in (0.3, 0.6):
        for k in range(0, 360, 90):
            yaw = rot_y_world_to_cam(k)
            heading = yaw[:3, :3].T @ torch.tensor([0.0, 0.0, 1.0])  # walk the way the camera faces
            c = (-heading * step).tolist()  # step back, so the view spans more of the room
            views[f"back{int(step*100)}cm_yaw{k:03d}"] = yaw @ translate(c)
    # 3. Raised and tilted views, both signs of the vertical axis (its direction is a convention).
    for sign, label in ((1, "a"), (-1, "b")):
        for k in (0, 180):
            w2c = rot_x_world_to_cam(sign * 35) @ rot_y_world_to_cam(k) @ translate([0.0, -sign * 0.5, 0.0])
            views[f"tilt{label}_yaw{k:03d}"] = w2c
    # 4. Outside-in: well behind the centre, to show the extent of the room's geometry.
    for k in (0, 90):
        yaw = rot_y_world_to_cam(k)
        heading = yaw[:3, :3].T @ torch.tensor([0.0, 0.0, 1.0])
        views[f"far_back_yaw{k:03d}"] = yaw @ translate((-heading * 2.0).tolist())

    rendered = render_views(gaussians, opt, list(views.values()))
    tiles = []
    for (name, _), (rgb, depth) in zip(views.items(), rendered):
        rgb8 = (rgb * 255 + 0.5).astype(np.uint8)
        depth_rgb = colorize_single_channel_image(np.clip(depth / SCENE_DEPTH_MAX, 0, 1))
        Image.fromarray(rgb8).save(shots / f"{name}.png")
        Image.fromarray(depth_rgb[..., ::-1] if depth_rgb.shape[-1] == 3 else depth_rgb).save(shots / f"{name}_depth.png")
        tile = rgb8.copy()
        cv2.putText(tile, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        tiles.append(cv2.resize(tile, (256, 256), interpolation=cv2.INTER_AREA))
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    Image.fromarray(sheet).save(shots / "contact_sheet.png")

    # Evaluation: stitch the centre faces back into a panorama and compare with the input.
    faces = [torch.from_numpy(np.concatenate([rgb, depth[..., None]], axis=2)).permute(2, 0, 1).float()
             for rgb, depth in rendered[:6]]
    torch.set_default_tensor_type("torch.FloatTensor")
    pano_rgbd = cube2equi(faces, "list", 1024, 2048).cpu().numpy()
    stitched = np.clip(pano_rgbd[:3].transpose(1, 2, 0), 0, 1)
    distance = pano_rgbd[3]
    Image.fromarray((stitched * 255 + 0.5).astype(np.uint8)).save(out / "stitched_from_3dgs.png")
    target = np.asarray(Image.open(pano_path).convert("RGB").resize((2048, 1024), Image.BICUBIC)) / 255.0
    observed = np.ones((1024, 2048), bool)
    if valid_path is not None:
        observed = np.asarray(Image.open(valid_path).convert("L").resize((2048, 1024), Image.NEAREST)) > 127
    compare = np.vstack([target, stitched, np.abs(target - stitched) * 4]).clip(0, 1)
    Image.fromarray((compare * 255 + 0.5).astype(np.uint8)).save(out / "input_vs_3dgs.png")

    alpha_holes = float((stitched.min(axis=2) > 0.98).mean())  # white background showing through
    result = {
        "gaussians": int(len(xyz)),
        "extent_min": xyz.min(0).round(3).tolist(),
        "extent_max": xyz.max(0).round(3).tolist(),
        "centre_psnr_all": psnr(stitched, target, np.ones_like(observed)),
        "centre_psnr_observed": psnr(stitched, target, observed),
        "centre_psnr_inpainted": psnr(stitched, target, ~observed),
        "centre_ssim_all": ssim_masked(stitched, target, np.ones_like(observed)),
        "centre_ssim_observed": ssim_masked(stitched, target, observed),
        "centre_ssim_inpainted": ssim_masked(stitched, target, ~observed),
        "observed_pixel_fraction": float(observed.mean()),
        "centre_white_hole_fraction": alpha_holes,
        "centre_distance_median": float(np.median(distance)),
        "novel_white_hole_fraction": {
            name: float((rgb.min(axis=2) > 0.98).mean()) for name, (rgb, _) in zip(list(views)[6:], rendered[6:])
        },
    }
    (out / "eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
