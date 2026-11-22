"""Phase 4 acceptance: the demo path runs end to end in under 60 s for a 10-second clip.

  python scripts/benchmark_demo.py [--model runs/restore/unet/best.pt] [--size 1920 1080] [--fps 30]

Renders a synthetic 10-second clip of the can on its ChArUco board (or uses --video),
writes it as an mp4, then times decoding, ChArUco poses, the classical pipeline and
restoration exactly as the demo app runs them. Rendering the clip is not timed.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sodacan.capture.board import load_board, render_texture  # noqa: E402
from sodacan.capture.video import write_video  # noqa: E402
from sodacan.color import to_uint8  # noqa: E402
from sodacan.config import load_can, load_pipeline  # noqa: E402
from sodacan.demo.run import process  # noqa: E402
from sodacan.geometry.camera import Intrinsics  # noqa: E402
from sodacan.sim.dataset import random_label  # noqa: E402
from sodacan.sim.fast_render import RenderParams, render  # noqa: E402
from sodacan.sim.rooms import procedural_room  # noqa: E402
from sodacan.sim.trajectories import arc_with_tilt  # noqa: E402


def make_clip(path, size, fps, seconds, can, device):
    W, H = size
    K = Intrinsics(W, H, 0.78 * W, 0.78 * W, (W - 1) / 2, (H - 1) / 2)
    n = fps * seconds
    rng = np.random.default_rng(0)
    env = procedural_room(1024, 2048, seed=11, device=device, person=True)
    c2w = arc_with_tilt(n, distance=0.30, arc_deg=50, elevation_deg=(-2, 25), cycles=2, target_z=can.height / 2,
                        jitter_m=0.002, device=device, min_z=0.015)
    _, bcfg = load_board()
    tex = render_texture(bcfg, ppm=2500)
    tex["texture"] = tex["texture"].to(device)
    p = RenderParams(table=tex, roughness_deg=0.5, label=random_label(rng, *can.atlas, device), ambient=0.6, exposure=0.7,
                     noise_read=0.002, noise_shot=0.001, quantize=True, chunk=2)
    rgb = []
    for i in range(0, n, 10):
        rgb += [to_uint8(f) for f in render(env, c2w[i : i + 10], K, can, p)]
    write_video(path, rgb, fps)
    return {"width": W, "height": H, "fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy, "dist": [0] * 5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--phone", default="phone_example")
    ap.add_argument("--model")
    ap.add_argument("--size", type=int, nargs=2, default=[1920, 1080])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=int, default=10)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    can, cfg = load_can(), load_pipeline()
    video, phone = args.video, args.phone
    if not video:
        video = str(Path(tempfile.gettempdir()) / "sodacan_benchmark.mp4")
        print(f"rendering a {args.seconds} s {args.size[0]}x{args.size[1]} clip at {args.fps} fps (not timed) ...", flush=True)
        phone = make_clip(video, args.size, args.fps, args.seconds, can, device)
    model = None
    if args.model:
        from sodacan.restore.infer import load_model

        model = load_model(args.model, device)
    t0 = time.time()
    last = None
    for up in process(video, phone, "board", can, cfg, model, device):
        last = up
        if up.stage != "accumulate":
            print(f"{time.time() - t0:6.1f} s  {up.message}", flush=True)
    total = time.time() - t0
    out = {"device": str(device), "total_s": total, "timing_s": last.timing, "frames_used": last.frames_total,
           "pass": total < 60}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
