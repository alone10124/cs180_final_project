#!/usr/bin/env bash
# Remaining steps of setup_linux.sh (pano2room part) after PyTorch3D, with setuptools pinned.
set -eo pipefail
ROOT=/home/xiang/cs180/cs180_final_project
THIRD=$ROOT/third_party
P2R=$THIRD/Pano2Room
LUCID=$THIRD/LucidDreamer
RAST=$LUCID/submodules/depth-diff-gaussian-rasterization-min
eval "$(conda shell.bash hook)"
conda activate pano2room
export PYTHONNOUSERSITE=1 PIP_CACHE_DIR=/overflow/xiang/pip_cache TMPDIR=/tmp/claude-1023/p2r_tmp
export CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 TCNN_CUDA_ARCHITECTURES=90 LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:$CONDA_PREFIX/lib"
which python pip nvcc
python -m pip install "setuptools<70" wheel

echo "== rasterizer and simple-knn"
: already installed

echo "== tiny-cuda-nn torch bindings"
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

echo "== Pano2Room requirements, one by one"
grep -v -E '^[[:space:]]*(#|$|-)' "$P2R/requirements.txt" \
  | grep -v -i -E '^(torch|torchvision|torchaudio|triton|xformers|tinycudann|nvidia-|depth-diff-gaussian|diff-gaussian|simple-knn|simple_knn|pytorch3d|setuptools)([=<>~! ]|$)' \
  | grep -v -E '0\.0\.0|\+cu|@ ' > "$THIRD/p2r_requirements.txt" || true
failed=()
while read -r req; do
  python -m pip install -q --no-deps "$req" >/dev/null 2>&1 || failed+=("$req")
done < "$THIRD/p2r_requirements.txt"
echo "could not install: ${failed[*]}"

echo "== import check"
cd "$P2R"
python - <<'EOF'
import torch, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "numpy", numpy.__version__)
import pytorch3d, tinycudann, depth_diff_gaussian_rasterization_min, simple_knn._C, diffusers, kornia
print("core imports ok")
EOF
