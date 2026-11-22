"""Reading frames from a phone video or a folder of images."""
from pathlib import Path

import cv2
import numpy as np


def read_frames(path, stride=1, max_frames=None, long_side=None):
    """RGB uint8 frames (list of (H, W, 3)), the source fps and the original (width, height).

    path is a video file or a folder of images (sorted by name).
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))[::stride]
        if max_frames:
            files = files[:max_frames]
        frames = [cv2.imread(str(f))[..., ::-1] for f in files]
        fps = 0.0
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames, i = [], 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % stride == 0:
                ok, f = cap.retrieve()
                if not ok:
                    break
                frames.append(f[..., ::-1])
                if max_frames and len(frames) >= max_frames:
                    break
            i += 1
        cap.release()
    if not frames:
        raise IOError(f"no frames read from {path}")
    size = (frames[0].shape[1], frames[0].shape[0])
    if long_side and max(size) > long_side:
        scale = long_side / max(size)
        new = (round(size[0] * scale), round(size[1] * scale))
        frames = [cv2.resize(f, new, interpolation=cv2.INTER_AREA) for f in frames]
    return [np.ascontiguousarray(f) for f in frames], fps, size


def write_video(path, frames_rgb, fps=30):
    """Write RGB uint8 frames to an mp4 (used for simulated test videos and the demo backup)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_rgb:
        writer.write(np.ascontiguousarray(f[..., ::-1]))
    writer.release()
