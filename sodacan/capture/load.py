"""From a phone video to linear frames with poses in the can frame (stages 1 and 2)."""
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from sodacan.capture.board import load_board
from sodacan.capture.pose_charuco import PoseResult, estimate_pose, estimate_poses, make_detector
from sodacan.capture.video import read_frames
from sodacan.color import linearize
from sodacan.config import load_yaml
from sodacan.geometry.camera import Intrinsics


MAX_REPROJ_PX = 2.0  # board poses with a larger RMS reprojection error (working-resolution pixels) are dropped


def _oriented(phone, w, h):
    """The phone calibration for frames of size (w, h): swapped if the phone was held the
    other way, then scaled."""
    full = Intrinsics.from_dict(phone)
    if (full.width > full.height) != (w > h):  # phone held the other way: swap the calibration
        full = Intrinsics(full.height, full.width, full.fy, full.fx, full.cy, full.cx)
    return full.scaled(w, h)


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
                 long_side=None, frames_rgb=None, can=None, roi=None, log=None, fit_can_position=True):
    """Decode, undistort, pose and linearise a capture. frames_rgb skips decoding (streaming).

    board=None (or "none") poses the frames from the can's outline and label instead of a
    ChArUco board (capture/pose_silhouette.py); that needs `can` and a printed can."""
    vc = (cfg or {}).get("video", {})
    phone = load_yaml(phone) if isinstance(phone, str) else phone
    dist = np.array(phone.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64)
    use_board = not (board is None or str(board).lower() == "none")
    full_poses = None
    if frames_rgb is None:
        hook = None
        if use_board:
            # find the board in every frame at full resolution, before the frame is shrunk:
            # 4K markers shrunk to 1280 px are too small to decode
            board_obj, bcfg = load_board(board)
            detector = make_detector(board_obj)
            full_poses = []

            def hook(rgb):
                Kf = np.array(_oriented(phone, rgb.shape[1], rgb.shape[0]).matrix(), dtype=np.float64)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                full_poses.append((estimate_pose(gray, Kf, dist, board_obj, bcfg, detector), rgb.shape[1]))
        frames_rgb, fps, _ = read_frames(video, stride or vc.get("stride", 1), max_frames or vc.get("max_frames"),
                                         long_side or vc.get("long_side"), on_frame=hook)
    h, w = frames_rgb[0].shape[:2]
    K = _oriented(phone, w, h)
    Km = np.array(K.matrix(), dtype=np.float64)
    if np.any(dist != 0):
        frames_rgb = [cv2.undistort(f, Km, dist) for f in frames_rgb]
    gamma = (cfg or {}).get("gamma", "srgb")
    if board is None or str(board).lower() == "none":
        if can is None:
            raise ValueError("posing without a board needs the can (load_can())")
        from sodacan.capture.pose_silhouette import estimate_poses_silhouette

        lin = linearize(torch.from_numpy(np.stack(frames_rgb)).to(device).permute(0, 3, 1, 2).float() / 255, gamma)
        poses = estimate_poses_silhouette(frames_rgb, lin, K, can, roi=roi, log=log)
        kept = [i for i, p in enumerate(poses) if p.c2w is not None]
        if len(kept) < 3:
            raise RuntimeError(f"can outline fitted in only {len(kept)} of {len(frames_rgb)} frames")
        frames = lin[kept]
        rgb = [frames_rgb[i] for i in kept]
        c2w = torch.tensor(np.stack([poses[i].c2w for i in kept]), dtype=torch.float32, device=device)
        errs = [poses[i].residual_px for i in kept]
        stats = {"decoded": len(frames_rgb), "posed": len(kept), "median_outline_px": float(np.median(errs)),
                 "pose": "silhouette", "median_label_match": float(np.median([poses[i].azimuth_score for i in kept]))}
        return Capture(frames, c2w, K, rgb, kept, errs, stats)
    if full_poses is not None:
        # reprojection error in full-resolution pixels, judged at the working resolution
        poses = [r if r.c2w is None or r.reproj_px * w / fw <= MAX_REPROJ_PX else PoseResult(None, r.corners, r.reproj_px)
                 for r, fw in full_poses]
    else:
        board_obj, bcfg = load_board(board)
        poses = estimate_poses(frames_rgb, Km, np.zeros(5), board_obj, bcfg, max_reproj_px=MAX_REPROJ_PX)
    kept = [i for i, p in enumerate(poses) if p.c2w is not None]
    if len(kept) < 3:
        raise RuntimeError(f"board found in only {len(kept)} of {len(frames_rgb)} frames")
    rgb = [frames_rgb[i] for i in kept]
    arr = torch.from_numpy(np.stack(rgb)).to(device).permute(0, 3, 1, 2).float() / 255
    frames = linearize(arr, gamma)
    c2w_np = np.stack([poses[i].c2w for i in kept])
    errs = [poses[i].reproj_px for i in kept]
    stats = {"decoded": len(frames_rgb), "posed": len(kept), "median_reproj_px": float(np.median(errs)), "pose": "board"}
    if can is not None and fit_can_position:
        # the can stands where it was put down, not exactly on the printed circle
        from sodacan.capture.pose_silhouette import fit_can_offset, shift_world

        off, before, after = fit_can_offset(rgb, c2w_np, K, can, log=log or print)
        if after < before:
            c2w_np = shift_world(c2w_np, off)
            stats.update(can_offset_mm=[round(float(v) * 1000, 1) for v in off],
                         outline_px_before=round(before, 2), outline_px_after=round(after, 2))
    c2w = torch.tensor(c2w_np, dtype=torch.float32, device=device)
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
