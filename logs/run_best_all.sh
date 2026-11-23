#!/usr/bin/env bash
# Best setting so far (corncho1): x_t guidance, saturation mask 0.98, clip 3, self-recurrence K=3, lambda=1/3,
# neutral prompt; lambda=0 (flow prior only) as the reference. Run inside one Slurm GPU allocation.
set -u
cd /home/xiang/cs180/cs180_final_project
export HF_HOME=/overflow/xiang/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 TMPDIR=/tmp/claude-1023/p2r_tmp
echo "CVD=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l
PY=~/miniconda3/envs/chips/bin/python
BEST="--guide-on xt --sat 0.98 --clip 3 --recur 3 --lam 0 0.333 --seeds 0 1 --diffuse texture"
PROMPT="This is a panorama. The image shows an indoor room."

echo "=== corncho1, pixel-space flow model (PixelDiT), same best setting"
SCENE=corncho1 $PY fm_guide_pixel.py --select above --z-min 0.008 $BEST --prompt "$PROMPT" \
  --gt results/paper_corncho_ref/fig5d_corncho1_ground_truth_5376x1827.jpg --gt-rows 0.68 --out results/fm_pixel_corncho1
echo "=== cans"
$PY fm_guide.py --select cans $BEST --prompt "$PROMPT" --out results/fm_best_cans
echo "=== labcat"
SCENE=labcat $PY fm_guide.py --select above --z-min 0.008 $BEST --prompt "$PROMPT" \
  --gt results/paper_labcat_ref/fig5d_labcat_ground_truth_700x238.jpg --gt-rows 0.68 --out results/fm_best_labcat
echo "=== bowls"
SCENE=bowls $PY fm_guide.py --select above --z-min 0.0 $BEST --prompt "$PROMPT" --out results/fm_best_bowls
echo "=== done"
