#!/usr/bin/env bash
# One-time setup on Linux with an NVIDIA GPU (Ampere, Ada or Hopper) and conda (Miniforge/Miniconda/Anaconda).
#
#   bash setup_linux.sh                    # everything, ending with Pano2Room on its own demo panorama
#   SKIP_SMOKE=1 bash setup_linux.sh       # skip that last check (it takes a while)
#   STEPS="pano2room" bash setup_linux.sh  # only some of: chips pano2room checkpoints smoke
#
# Creates two conda environments, because the two halves pin incompatible stacks:
#   chips      this repo (part a, the viewer) and COLMAP built with CUDA
#   pano2room  Pano2Room (part b): PyTorch 2.0.1 + CUDA 11.8, PyTorch3D, its Gaussian rasterizer
# and clones Pano2Room (with its checkpoints) and LucidDreamer (for the rasterizer) into third_party/.
set -eo pipefail  # no -u: some conda activation scripts read unset variables

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD="$ROOT/third_party"
P2R="$THIRD/Pano2Room"
LUCID="$THIRD/LucidDreamer"
P2R_COMMIT=bbf93ae57086ed700edc6ee445852d4457a9d704    # TrickyGo/Pano2Room, 2024-08-23
LUCID_COMMIT=4ddf48563140c4254c8ce566955d12531b46829c  # luciddreamer-cvlab/LucidDreamer, 2026-07-23
GLM_TAG=0.9.9.8
CKPT_URL="https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h?rlkey=a96g2incso7g53evzamzo0j0y&dl=1"
STEPS="${STEPS:-chips pano2room checkpoints smoke}"
[ -n "${SKIP_SMOKE:-}" ] && STEPS="${STEPS/smoke/}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
want() { [[ " $STEPS " == *" $1 "* ]]; }
env_exists() { conda env list | awk '{print $1}' | grep -x "$1" >/dev/null; }  # no -q: avoids SIGPIPE under pipefail

command -v conda >/dev/null || { echo "conda not found: install Miniforge (https://github.com/conda-forge/miniforge) first"; exit 1; }
command -v git >/dev/null || { echo "git not found"; exit 1; }
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "warning: nvidia-smi not found; both environments need an NVIDIA GPU"
fi
eval "$(conda shell.bash hook)"
mkdir -p "$THIRD"

# ------------------------------------------------------------------ part (a)
if want chips; then
  say "chips environment: this repo + COLMAP with CUDA"
  env_exists chips || conda create -y -n chips -c conda-forge python=3.11
  conda activate chips
  conda install -y -c conda-forge "colmap=*=*cuda*"
  pip install -r "$ROOT/requirements.txt"
  banner="$(colmap -h 2>&1 || true)"
  printf '%s\n' "$banner" | sed -n '1,3p'
  [[ "$banner" == *"with CUDA"* ]] || echo "warning: this COLMAP does not report CUDA support; dense stereo (sfm stage) needs it"
  python -c "import torch, open3d, viser, cv2; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
  conda deactivate
fi

# ------------------------------------------------------------------ part (b)
if want pano2room; then
  say "Pano2Room and LucidDreamer sources (pinned)"
  [ -d "$P2R/.git" ] || git clone https://github.com/TrickyGo/Pano2Room.git "$P2R"
  git -C "$P2R" fetch -q origin && git -C "$P2R" checkout -q "$P2R_COMMIT"
  [ -d "$LUCID/.git" ] || git clone https://github.com/luciddreamer-cvlab/LucidDreamer.git "$LUCID"
  git -C "$LUCID" fetch -q origin && git -C "$LUCID" checkout -q "$LUCID_COMMIT"
  RAST="$LUCID/submodules/depth-diff-gaussian-rasterization-min"
  # setup.py of the rasterizer includes third_party/glm, which the LucidDreamer copy does not ship
  [ -d "$RAST/third_party/glm" ] || git clone -q --depth 1 --branch "$GLM_TAG" https://github.com/g-truc/glm.git "$RAST/third_party/glm"

  say "pano2room environment: PyTorch 2.0.1 + CUDA 11.8 toolkit + GCC 11"
  env_exists pano2room || conda create -y -n pano2room -c conda-forge python=3.10
  conda activate pano2room
  conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
  conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11  # nvcc 11.8 rejects GCC newer than 11
  conda deactivate && conda activate pano2room                      # activation sets CC/CXX to GCC 11
  pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
  export CUDA_HOME="$CONDA_PREFIX"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"  # A100, RTX 30, RTX 40/L40, H100
  export MAX_JOBS="${MAX_JOBS:-8}"

  say "PyTorch3D (prebuilt wheel, else from source: ~20 min)"
  pip install fvcore iopath
  pip install --no-index --no-cache-dir pytorch3d \
      -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html \
    || pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4"

  say "Gaussian rasterizer with depth, and simple-knn (CUDA builds)"
  pip install --no-build-isolation "$RAST" "$LUCID/submodules/simple-knn"

  say "Pano2Room's other requirements (its own pins, one by one; CUDA and torch packages excluded)"
  # Its requirements.txt is a full freeze that mixes cu11 and cu12 wheels, so it cannot be installed as one set.
  grep -v -E '^[[:space:]]*(#|$|-)' "$P2R/requirements.txt" \
    | grep -v -i -E '^(torch|torchvision|torchaudio|triton|xformers|tinycudann|nvidia-|depth-diff-gaussian|diff-gaussian|simple-knn|simple_knn|pytorch3d)([=<>~! ]|$)' \
    | grep -v -E '0\.0\.0|\+cu|@ ' > "$THIRD/p2r_requirements.txt" || true
  failed=()
  while read -r req; do
    pip install -q --no-deps "$req" >/dev/null 2>&1 || failed+=("$req")
  done < "$THIRD/p2r_requirements.txt"
  [ ${#failed[@]} -eq 0 ] || echo "could not install (usually harmless): ${failed[*]}"

  say "Checking that every module Pano2Room imports resolves"
  python - "$P2R" <<'EOF'
import ast, importlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
local = {p.name for p in root.rglob("*") if p.is_dir()} | {p.stem for p in root.rglob("*.py")}
names = set()
for path in root.rglob("*.py"):
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
missing = []
for name in sorted(names - local - set(sys.stdlib_module_names)):
    try:
        importlib.import_module(name)
    except Exception as e:  # report every failure, keep going
        missing.append(f"{name}: {type(e).__name__}: {e}")
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("all imports resolve" if not missing else "imports that fail (install them into the pano2room env):")
for m in missing:
    print("  ", m)
EOF
  conda deactivate
fi

if want checkpoints; then
  say "Pano2Room checkpoints: big-lama, omnidata depth and normal (Dropbox, ~2 GB)"
  CK="$P2R/checkpoints"
  if ls "$CK/big-lama.ckpt" "$CK/omnidata_dpt_depth_v2.ckpt" "$CK/omnidata_dpt_normal_v2.ckpt" >/dev/null 2>&1; then
    echo "already present"
  else
    curl -L --fail -o "$THIRD/p2r_checkpoints.zip" "$CKPT_URL"
    if command -v unzip >/dev/null; then
      unzip -o "$THIRD/p2r_checkpoints.zip" -d "$CK" -x / || true
    else
      python3 -m zipfile -e "$THIRD/p2r_checkpoints.zip" "$CK"
    fi
    find "$CK" -mindepth 2 -name '*.ckpt' -exec mv -n {} "$CK"/ \;
  fi
  ls -lh "$CK"
fi

if want smoke; then
  say "Pano2Room on its demo panorama (checks all of part b; downloads Stable Diffusion 2 inpainting on first run)"
  conda activate pano2room
  [ -f "$P2R/input/input_panorama.demo.png" ] && cp "$P2R/input/input_panorama.demo.png" "$P2R/input/input_panorama.png"
  (cd "$P2R" && rm -rf output/SDFT_weights output/Pano2Room-results && python pano2room.py)
  ls -lh "$P2R/output/Pano2Room-results/3DGS.ply"
  conda deactivate
  echo "If the Stable Diffusion download failed, run 'huggingface-cli login' in the pano2room env and retry."
fi

say "Done"
cat <<EOF
Next:
  conda activate chips
  python selftest_video.py                                  # CPU checks of the video front end
  python video2room.py --video bag.mp4 --name bag --quick   # end-to-end smoke test
  python video2room.py --video bag.mp4 --name bag --view    # full run, then the viewer on port 8080
EOF
