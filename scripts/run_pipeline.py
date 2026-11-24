"""Run the whole pipeline on a simulated or real video and save a debug image per stage.

  python scripts/run_pipeline.py --sim --out runs/sim_demo                    # simulated can video
  python scripts/run_pipeline.py --sim --mirror --out runs/sim_mirror         # perfect mirror, no label
  python scripts/run_pipeline.py --video can.mp4 --phone configs/phone_pixel.yaml --out runs/kitchen \
      [--reference room_pano.jpg] [--model runs/train_unet/best.pt]
  python scripts/run_pipeline.py --video can.mp4 --no-board --phone ... --out runs/kitchen   # no ChArUco board

Writes into --out:
  00_frames.png        a few input frames (linear -> sRGB, auto exposure)
  01_poses.png         can outline and world axes drawn with the estimated poses
  02_reflection.png    per-pixel reflection direction as colour, on the usable wall
  03_surface.png       frames warped into surface coordinates (angle x height)
  04_label.png         label estimate; 04_residual.png frame minus label
  05_panorama.png      measured panorama; 05_coverage.png weight map; 05_reference.png if known
  06_enhanced.png      restoration output (with --model)
  panorama.npz         pano, weight, valid (+ enhanced, reference)
  metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sodacan import debug  # noqa: E402
from sodacan.capture.uv_warp import warp_to_surface  # noqa: E402
from sodacan.color import auto_exposure, to_uint8  # noqa: E402
from sodacan.config import load_can, load_pipeline, load_yaml  # noqa: E402
from sodacan.geometry import cylinder  # noqa: E402
from sodacan.geometry.equirect import resize_pano  # noqa: E402
from sodacan.geometry.reflect import can_geometry  # noqa: E402
from sodacan.metrics import align_azimuth, all_metrics  # noqa: E402
from sodacan.pipeline import reconstruct  # noqa: E402


def simulate(args, can, device):
    from sodacan.sim.dataset import sample_scene

    sim = load_yaml(args.sim_config)
    rng = np.random.default_rng(args.seed)
    scene = sample_scene(rng, sim, can, device=device, mirror=args.mirror, frames=args.frames)
    return scene.frames, scene.c2w, scene.K, scene.env, scene


def load_real(args, can, cfg, device):
    from sodacan.capture.load import load_capture

    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    cap = load_capture(args.video, args.phone, None if args.no_board else args.board, cfg, device=device, can=can,
                       roi=roi, log=print if args.no_board else None)
    return cap.frames, cap.c2w, cap.K, cap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim", action="store_true", help="simulate a can video instead of reading one")
    ap.add_argument("--mirror", action="store_true", help="with --sim: perfect mirror, no label or noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--sim-config", default="sim")
    ap.add_argument("--video")
    ap.add_argument("--phone", default="phone_example")
    ap.add_argument("--board", default="board")
    ap.add_argument("--no-board", action="store_true",
                    help="pose from the can's outline and label instead of a ChArUco board (slower, about 1 deg)")
    ap.add_argument("--roi", help="with --no-board: x0,y0,x1,y1 box around the can in the first frame")
    ap.add_argument("--can", default="can")
    ap.add_argument("--pipeline", default="pipeline")
    ap.add_argument("--reference", help="phone/360 panorama of the room taken from the can (photo convention)")
    ap.add_argument("--model", help="restoration checkpoint")
    ap.add_argument("--label", choices=["median", "min", "percentile", "none"])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    can = load_can(args.can, **({"material": {"model": "mirror"}} if args.mirror else {}))
    cfg = load_pipeline(args.pipeline, **({"label": {"method": "none"}} if args.mirror else {}))
    if args.label:
        cfg["label"]["method"] = args.label
    H, W = cfg["pano"]
    t0 = time.time()
    reference = None
    if args.sim:
        frames, c2w, K, env, scene = simulate(args, can, device)
        reference = resize_pano(env, H, W) * scene.params.strength * scene.params.exposure
    else:
        if not args.video:
            ap.error("--video or --sim is required")
        frames, c2w, K, cap = load_real(args, can, cfg, device)
        if args.reference:
            from sodacan.capture.load import load_reference

            reference = load_reference(args.reference, H, W).to(device)
    timing = {"load": time.time() - t0}

    # 00 frames, 01 poses
    idx = np.linspace(0, len(frames) - 1, 6).round().astype(int)
    exp = auto_exposure(frames[idx[0]])
    shots = [to_uint8(frames[i], exp) for i in idx]
    debug.save_rgb(out / "00_frames.png", debug.montage(shots, cols=3, labels=[f"frame {i}" for i in idx]))
    debug.save_rgb(out / "01_poses.png", debug.montage([debug.can_outline(s, c2w[i], K, can) for s, i in zip(shots, idx)], cols=3))

    # 02 reflection directions
    g = can_geometry(c2w[idx[:3]], K, can)
    debug.save_rgb(out / "02_reflection.png", debug.montage([debug.direction_rgb(g.reflected[k], g.mask[k]) for k in range(3)], cols=3))

    # 03 surface space
    obs, valid = warp_to_surface(frames[idx[:3]], c2w[idx[:3]], K, can)
    debug.save_rgb(out / "03_surface.png", debug.stack([debug.display(obs[k], valid[k], exp) for k in range(3)]))

    # 04-05 label and panorama
    t1 = time.time()
    result = reconstruct(frames, c2w, K, can, cfg)
    timing["classical"] = time.time() - t1
    if result.label is not None:
        cnt = result.label_count > 0
        debug.save_rgb(out / "04_label.png", debug.display(result.label, cnt, exp))
        gk = can_geometry(c2w[idx[:1]], K, can)
        res = frames[idx[0]].permute(1, 2, 0).clone()
        m = gk.mask[0]
        res[m] = res[m] - cylinder.atlas_sample(result.label, gk.s[0][m], gk.h[0][m], can.height)
        res[~m] = 0
        debug.save_rgb(out / "04_residual.png", to_uint8(res.permute(2, 0, 1).clamp_min(0), exp * 4))
    debug.save_rgb(out / "05_panorama.png", debug.display(result.pano, result.valid))
    debug.save_rgb(out / "05_coverage.png", debug.heat(result.weight, result.valid))
    saved = {"pano": result.pano.cpu().numpy(), "weight": result.weight.cpu().numpy(), "valid": result.valid.cpu().numpy()}
    metrics = {"frames": int(len(frames)), "timing_s": timing}
    if result.label is not None:
        # the label as an atlas for other runs (run_noboard.py --label-atlas): with board poses
        # the camera circles the can, so every side of the label is seen
        saved["label"] = result.label.cpu().numpy()
        saved["label_count"] = result.label_count.cpu().numpy()
        np.save(out / "label_atlas.npy", saved["label"])

    if reference is not None:
        if not args.sim:
            reference, shift, flip = align_azimuth(result.pano, reference, result.valid)
            metrics["reference_alignment"] = {"shift_columns": shift, "mirrored": flip}
        debug.save_rgb(out / "05_reference.png", debug.display(reference, None, auto_exposure(reference, result.valid)))
        metrics["measured"] = all_metrics(result.pano, reference, result.valid)
        saved["reference"] = reference.cpu().numpy()

    if args.model:
        from sodacan.restore.infer import load_model, enhance

        t2 = time.time()
        model = load_model(args.model, device)
        enhanced = enhance(model, result.pano, result.weight, result.valid)
        timing["restore"] = time.time() - t2
        debug.save_rgb(out / "06_enhanced.png", debug.display(enhanced, result.valid))
        saved["enhanced"] = enhanced.cpu().numpy()
        if reference is not None:
            metrics["enhanced"] = all_metrics(enhanced, reference, result.valid)

    timing["total"] = time.time() - t0
    np.savez_compressed(out / "panorama.npz", **saved)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
