"""Evaluate on held-out views and export the paper's result figures to results/.

Metrics are computed on the cans only, as in the paper's Table 1 and Fig. 8:
  L1    mean absolute error over all pixels and channels, with the background
        zeroed in both images (values in [0, 1], display-encoded)
  LPIPS perceptual distance [79] on the same masked images
"""
import argparse
import json

import cv2
import lpips
import numpy as np
import torch
from torch.amp import autocast

from common import CACHE, CAN_Z_MIN, GAMMA, RESULTS, RUNS
from data import Scene, augment, to_display
from models import EncoderDecoder
from train import load_srm, physical_inputs

ANGLE_BINS = [0, 5, 10, 15, 20]


def object_mask(b, centre):
    """Pixels on the raised geometry near the object centre, i.e. the two cans."""
    pos = b["pos"]
    near = (pos[:, :2] - centre[:2, None, None]).abs().max(dim=1, keepdim=True)[0] < 0.13
    return ((pos[:, 2:3] > CAN_Z_MIN) & near).float() * b["mask"]


def to_bgr(tensor):
    array = tensor.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return (array[..., ::-1] * 255).astype(np.uint8)


def save_srm(srm, materials):
    for i in range(materials):
        pano = (srm.srm[i].detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1) ** (1 / GAMMA))
        name = f"srm_{i}.png"
        cv2.imwrite(str(RESULTS / name), (pano[..., ::-1] * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srm-run", default="srm")
    ap.add_argument("--renderer-run", default="renderer")
    ap.add_argument("--materials", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1, help="evaluate every Nth test frame")
    ap.add_argument("--figures", type=int, default=8, help="number of comparison figures to save")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    device = torch.device("cuda")
    scene = Scene(device)
    srm = load_srm(device, args.materials, args.srm_run)
    save_srm(srm, args.materials)

    generator = EncoderDecoder(13).to(device)
    state = torch.load(RUNS / args.renderer_run / "last.pt", map_location=device)
    generator.load_state_dict(state["generator"])
    generator.eval()
    metric = lpips.LPIPS(net="vgg", verbose=False).to(device)
    centre = torch.tensor(np.load(CACHE / "split.npz")["centre"], dtype=torch.float32, device=device)

    frames = scene.test[:: args.stride].tolist()
    angles = scene.test_angle[:: args.stride]
    figure_at = set(np.linspace(0, len(frames) - 1, args.figures).astype(int).tolist())
    methods = ["diffuse", "diffuse+specular", "neural_renderer"]
    per_frame = {m: {"l1": [], "lpips": []} for m in methods}

    writer = cv2.VideoWriter(str(RESULTS / "test_views.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1920, 480))
    for n, frame in enumerate(frames):
        b = augment(scene.batch([frame]), enabled=False)
        with torch.no_grad(), autocast("cuda"):
            inputs, diffuse, specular, weights = physical_inputs(srm, b)
            neural = generator(inputs).clamp(0, 1)
        neural = neural.float()
        composite = to_display((diffuse + specular).float())
        plain = to_display(diffuse.float())
        object_only = object_mask(b, centre)
        real = b["rgb"] * object_only
        with torch.no_grad():
            for name, image in zip(methods, [plain, composite, neural]):
                masked = image * object_only
                per_frame[name]["l1"].append((masked - real).abs().mean().item())
                score = metric(masked * 2 - 1, real * 2 - 1).item()
                per_frame[name]["lpips"].append(score)

        strip = np.hstack([to_bgr(b["rgb"][0]), to_bgr(composite[0]), to_bgr(neural[0])])
        writer.write(strip)
        if n in figure_at:
            spec = to_bgr(to_display(specular.float())[0])
            bases = weights[0].float()
            shown = torch.zeros(3, *bases.shape[1:], device=device)
            shown[: min(3, len(bases))] = bases[: min(3, len(bases))]
            lower = np.hstack([to_bgr(plain[0]), spec, to_bgr(shown)])
            figure = np.vstack([strip, lower])
            cv2.imwrite(str(RESULTS / f"test_{frame + 1:04d}.png"), figure)
    writer.release()

    summary = {"frames": len(frames), "overall": {}, "by_angle": {}}
    for name in methods:
        l1 = np.array(per_frame[name]["l1"])
        score = np.array(per_frame[name]["lpips"])
        summary["overall"][name] = {"l1": float(l1.mean()), "lpips": float(score.mean())}
        for low, high in zip(ANGLE_BINS, ANGLE_BINS[1:]):
            keep = (angles >= low) & (angles < high)
            if keep.any():
                key = f"{low}-{high}"
                entry = summary["by_angle"].setdefault(key, {"count": int(keep.sum())})
                entry[name] = {"l1": float(l1[keep].mean()), "lpips": float(score[keep].mean())}
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
