#!/bin/bash
# Runs on Kaggle (GPU T4 x2): the real phone video at higher settings than the laptop run.
#   GPU 0: ChArUco-board poses, every 4th frame of the whole orbit (227), 1280 px frames
#   GPU 1: the same video with no-board poses (can outline + label)
# then every restoration checkpoint on both panoramas. The video and checkpoints come as
# split parts in a second dataset; the notebook puts them back together (REPO=/tmp/src).
set -x
W=/kaggle/working
SRC=${REPO:-$(dirname "$(find /kaggle/input -name pyproject.toml | head -1)")}
rm -rf /tmp/repo; cp -r "$SRC" /tmp/repo; cd /tmp/repo
pip install -q -e ".[metrics]" > $W/pip.log 2>&1
nvidia-smi | tee $W/env.log
VID=data/real_world_test.mp4
ARGS="--video $VID --phone configs/phone_real.yaml --board board_printed --can can_pepsi --pipeline configs/pipeline_kaggle.yaml --model runs/restore/unet_lpips/best.pt"
CUDA_VISIBLE_DEVICES=0 python scripts/run_pipeline.py $ARGS --out $W/board > $W/board.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/run_pipeline.py $ARGS --no-board --out $W/noboard > $W/noboard.log 2>&1 &
wait
python - <<'PY' 2>&1 | tee $W/enhance.log
import numpy as np, torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
from sodacan import debug
from sodacan.restore.infer import load_model, enhance
W = "/kaggle/working"
for run in ("board", "noboard"):
    try:
        d = np.load(f"{W}/{run}/panorama.npz")
    except FileNotFoundError:
        print(run, "no panorama"); continue
    pano, weight, valid = (torch.from_numpy(d[k]).to(dev) for k in ("pano", "weight", "valid"))
    for name in ("unet", "unet_lpips", "rtlvit", "vit5_kaggle"):
        out = enhance(load_model(f"runs/restore/{name}/best.pt", dev), pano, weight, valid)
        debug.save_rgb(f"{W}/{run}/06_enhanced_{name}.png", debug.display(out, valid))
        print(run, name, "ok")
PY
echo ALL DONE | tee -a $W/env.log
