"""Phone calibration with the ChArUco board (once per phone), and the printable board.

  python -m sodacan.capture.calibrate board --out boards/board.png
  python -m sodacan.capture.calibrate calibrate --video board_sweep.mp4 --out configs/phone_pixel.yaml

For calibration, film the board alone from many angles and distances (tilted views
matter most), with the same lens, resolution and focus lock you will use for captures.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from sodacan.capture.board import board_image, board_size_m, load_board
from sodacan.capture.video import read_frames
from sodacan.config import load_can


def calibrate(frames_rgb, board, min_corners=12, max_views=60):
    detector = cv2.aruco.CharucoDetector(board)
    objs, imgs = [], []
    for f in frames_rgb:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is None or len(ids) < min_corners:
            continue
        obj, img = board.matchImagePoints(corners, ids)
        objs.append(obj.astype(np.float32))
        imgs.append(img.astype(np.float32))
    if len(objs) < 5:
        raise RuntimeError(f"only {len(objs)} usable board views; film the board from more angles")
    if len(objs) > max_views:
        keep = np.linspace(0, len(objs) - 1, max_views).round().astype(int)
        objs, imgs = [objs[i] for i in keep], [imgs[i] for i in keep]
    h, w = frames_rgb[0].shape[:2]
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, (w, h), None, None)
    return {"width": w, "height": h, "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]),
            "cy": float(K[1, 2]), "dist": [float(x) for x in dist.ravel()[:5]], "rms_px": float(rms), "views": len(objs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("board", help="write a printable board image")
    b.add_argument("--out", default="boards/board.png")
    b.add_argument("--dpi", type=int, default=300)
    b.add_argument("--board", default="board")
    c = sub.add_parser("calibrate", help="calibrate a phone from a board video or image folder")
    c.add_argument("--video", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--board", default="board")
    c.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    board, cfg = load_board(args.board)
    if args.cmd == "board":
        ppm = args.dpi / 0.0254
        img = board_image(board, cfg, ppm, can_radius=load_can().radius)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, img)
        w, h = board_size_m(cfg)
        print(f"wrote {args.out}: {w * 100:.1f} x {h * 100:.1f} cm at {args.dpi} dpi; print at 100% scale")
        return
    frames, _, _ = read_frames(args.video, stride=args.stride)
    result = calibrate(frames, board)
    Path(args.out).write_text(yaml.safe_dump(result, sort_keys=False))
    print(f"RMS reprojection {result['rms_px']:.3f} px over {result['views']} views -> {args.out}")


if __name__ == "__main__":
    main()
