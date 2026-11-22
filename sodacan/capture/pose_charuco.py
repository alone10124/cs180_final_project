"""Camera pose in every frame from the ChArUco board, as camera-to-world in the can frame."""
from dataclasses import dataclass

import cv2
import numpy as np

from sodacan.capture.board import board_to_world


@dataclass
class PoseResult:
    c2w: np.ndarray        # (4, 4) or None when the board was not found
    corners: int           # ChArUco corners used
    reproj_px: float       # RMS reprojection error


def detect(detector, gray):
    corners, ids, _, _ = detector.detectBoard(gray)
    return corners, ids


def estimate_pose(gray, K, dist, board, cfg, detector=None, min_corners=6):
    """Pose of one greyscale uint8 frame. K is a 3x3 array."""
    detector = detector or cv2.aruco.CharucoDetector(board)
    corners, ids = detect(detector, gray)
    if ids is None or len(ids) < min_corners:
        return PoseResult(None, 0 if ids is None else len(ids), float("nan"))
    obj, img = board.matchImagePoints(corners, ids)
    if obj is None or len(obj) < min_corners or board.checkCharucoCornersCollinear(ids):
        return PoseResult(None, len(ids), float("nan"))
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return PoseResult(None, len(ids), float("nan"))
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2).sum(1).mean()))
    R, _ = cv2.Rodrigues(rvec)
    w2c_board = np.eye(4)
    w2c_board[:3, :3], w2c_board[:3, 3] = R, tvec.ravel()
    c2w = board_to_world(cfg) @ np.linalg.inv(w2c_board)
    return PoseResult(c2w, len(ids), err)


def estimate_poses(frames_rgb, K, dist, board, cfg, max_reproj_px=2.0, min_corners=6):
    """Poses for a list of RGB uint8 frames. Frames without a reliable pose get c2w None."""
    detector = cv2.aruco.CharucoDetector(board)
    out = []
    for f in frames_rgb:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        r = estimate_pose(gray, K, dist, board, cfg, detector, min_corners)
        if r.c2w is not None and r.reproj_px > max_reproj_px:
            r = PoseResult(None, r.corners, r.reproj_px)
        out.append(r)
    return out
