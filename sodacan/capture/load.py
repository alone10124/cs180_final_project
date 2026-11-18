"""From a phone video to linear frames with poses in the can frame (stages 1 and 2)."""
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from sodacan.capture.board import load_board
from sodacan.capture.pose_charuco import estimate_poses
from sodacan.capture.video import read_frames
from sodacan.color import linearize
from sodacan.config import load_yaml
from sodacan.geometry.camera import Intrinsics


@dataclass
class Capture:
    frames: torch.Tensor        # (B, 3, H, W) linear, undistorted
    c2w: torch.Tensor           # (B, 4, 4)
    K: Intrinsics
    rgb: list                   # undistorted uint8 frames that kept a pose
    kept: list                  # indices into the decoded frames
    reproj_px: list
    stats: dict = field(default_factory=dict)


def load_capture(video, phone="phone_example", board="board", cfg=None, device="cpu", stride=None, max_frames=None,
                 long_side=None, frames_rgb=None):
    """Decode, undistort, pose and linearise a capture. frames_rgb skips decoding (streaming)."""
    vc = (cfg or {}).get("video", {})
    phone = load_yaml(phone) if isinstance(phone, str) else phone
    if frames_rgb is None:
        frames_rgb, fps, _ = read_frames(video, stride or vc.get("stride", 1), max_frames or vc.get("max_frames"),
                                         long_side or vc.get("long_side"))
    h, w = frames_rgb[0].shape[:2]
    full = Intrinsics.from_dict(phone)
    if (full.width > full.height) != (w > h):  # phone held the other way: swap the calibration
        full = Intrinsics(full.height, full.width, full.fy, full.fx, full.cy, full.cx)
    K = full.scaled(w, h)
    Km = np.array(K.matrix(), dtype=np.float64)
    dist = np.array(phone.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64)
    if np.any(dist != 0):
        frames_rgb = [cv2.undistort(f, Km, dist) for f in frames_rgb]
    board_obj, bcfg = load_board(board)
    poses = estimate_poses(frames_rgb, Km, np.zeros(5), board_obj, bcfg)
    kept = [i for i, p in enumerate(poses) if p.c2w is not None]
    if len(kept) < 3:
        raise RuntimeError(f"board found in only {len(kept)} of {len(frames_rgb)} frames")
    rgb = [frames_rgb[i] for i in kept]
    arr = torch.from_numpy(np.stack(rgb)).to(device).permute(0, 3, 1, 2).float() / 255
    gamma = (cfg or {}).get("gamma", "srgb")
    frames = linearize(arr, gamma)
    c2w = torch.tensor(np.stack([poses[i].c2w for i in kept]), dtype=torch.float32, device=device)
    errs = [poses[i].reproj_px for i in kept]
    stats = {"decoded": len(frames_rgb), "posed": len(kept), "median_reproj_px": float(np.median(errs))}
    return Capture(frames, c2w, K, rgb, kept, errs, stats)


def load_reference(path, H, W):
    """A phone/360 panorama of the room in this package's layout, linear, (3, H, W).

    The heading is unknown; compare with metrics.align_azimuth."""
    from sodacan.geometry.equirect import resize_pano
    from sodacan.sim.rooms import photo_to_pano, read_hdr

    path = str(path)
    if path.lower().endswith((".hdr", ".exr")):
        img = read_hdr(path)
    else:
        bgr = cv2.imread(path)
        if bgr is None:
            raise IOError(f"cannot read {path}")
        img = linearize(torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1).float() / 255)
    h, w = img.shape[1:]
    if abs(w / h - 2) > 0.05:  # phone panoramas are often cropped vertically: pad to 2:1 around the horizon
        full_h = w // 2
        top = max((full_h - h) // 2, 0)
        canvas = torch.zeros(3, full_h, w)
        canvas[:, top : top + min(h, full_h)] = img[:, : min(h, full_h)]
        img = canvas
    return resize_pano(photo_to_pano(img), H, W)
