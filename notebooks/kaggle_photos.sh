#!/bin/bash
# Runs on Kaggle (CPU is enough): the one-can and two-can photo sets, side by side.
# The notebook has put the code in /tmp/src, the photos in /tmp/src/data and the
# restoration checkpoints in /tmp/src/runs/restore (REPO=/tmp/src).
set -x
W=/kaggle/working
cd ${REPO:-/tmp/src}
pip install -q -e ".[metrics]" > $W/pip.log 2>&1
MODELS="runs/restore/unet/best.pt runs/restore/unet_lpips/best.pt runs/restore/rtlvit/best.pt runs/restore/vit5_kaggle/best.pt"
python scripts/run_photos.py --photos data/one_can --cans 1 --out $W/one_can --models $MODELS > $W/one_can.log 2>&1 &
python scripts/run_photos.py --photos data/two_cans --cans 2 --out $W/two_cans --models $MODELS > $W/two_cans.log 2>&1 &
wait
echo ALL DONE
