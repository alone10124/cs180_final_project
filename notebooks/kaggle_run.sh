#!/bin/bash
# Runs on Kaggle (GPU T4 x2): simulate pairs on both GPUs, then train RTLViT and ViT-5 side by side.
set -x
W=/kaggle/working
SRC=$(dirname "$(find /kaggle/input -name pyproject.toml | head -1)")
rm -rf /tmp/repo; cp -r "$SRC" /tmp/repo; cd /tmp/repo
pip install -q -e ".[metrics]" > $W/pip.log 2>&1
python -c "import torch, lpips; print(torch.__version__, torch.cuda.device_count())" | tee $W/env.log
nvidia-smi | tee -a $W/env.log
CUDA_VISIBLE_DEVICES=0 python -m sodacan.sim.dataset --count 5000 --seed 0 --out /tmp/pairs > $W/sim0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m sodacan.sim.dataset --count 5000 --seed 1 --out /tmp/pairs > $W/sim1.log 2>&1 &
wait
ls /tmp/pairs | wc -l | tee -a $W/env.log
CUDA_VISIBLE_DEVICES=0 python -m sodacan.restore.train --config configs/train_rtlvit.yaml --out $W/rtlvit --steps 15000 --resume --max-minutes 300 --shards /tmp/pairs > $W/rtlvit.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m sodacan.restore.train --config configs/train_vit5.yaml --out $W/vit5 --steps 15000 --resume --max-minutes 300 --shards /tmp/pairs > $W/vit5.log 2>&1 &
wait
echo ALL DONE | tee -a $W/env.log
