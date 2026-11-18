"""Extract frames from a video for reconstruction.

  python video_frames.py bag.mp4 scenes/bag/capture/frames --count 300

The video is split into `count` equal windows and the sharpest frame of each window
(variance of the Laplacian) is kept, which avoids the motion-blurred frames of a
hand-held orbit. OpenCV applies the rotation stored in phone videos.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def sharpness(frame, width=480):
    """Variance of the Laplacian on a downscaled grey image."""
    scale = width / frame.shape[1]
    grey = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(grey, cv2.CV_64F).var()


def frames(video):
    """Yield the decoded frames of a video, rotated upright."""
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened(), f"cannot open {video}"
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame
    capture.release()


def extract(video, out, count=300, max_side=1920):
    """Write frame_0001.jpg ... to `out`; returns the number of frames written.

    Two decoding passes (score, then write) keep memory independent of the video length.
    """
    scores = np.array([sharpness(frame) for frame in frames(video)])
    assert len(scores), f"{video} has no decodable frames"
    count = min(count, len(scores))
    window = (np.arange(len(scores)) * count) // len(scores)
    keep = {int(np.flatnonzero(window == w)[np.argmax(scores[window == w])]) for w in range(count)}

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, frame in enumerate(frames(video)):
        if i not in keep:
            continue
        scale = min(1.0, max_side / max(frame.shape[:2]))
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        written += 1
        cv2.imwrite(str(out / f"frame_{written:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out", help="directory for frame_0001.jpg ...")
    ap.add_argument("--count", type=int, default=300, help="number of frames to keep")
    ap.add_argument("--max-side", type=int, default=1920, help="downscale frames whose longer side exceeds this")
    args = ap.parse_args()
    written = extract(args.video, args.out, args.count, args.max_side)
    print(f"wrote {written} frames to {args.out}")


if __name__ == "__main__":
    main()
