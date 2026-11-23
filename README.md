# Seeing the World in a Bag of Chips - Cans reconstruction

This repository has one focused purpose: recover a specular reflectance map
(SRM) from the real **Cans** sequence released with Park et al., optionally render
novel views, and test a render-guided flow prior.

## Essential files

| File | Purpose |
|---|---|
| `common.py` | Paths, camera calibration, poses, panorama coordinates, and split helpers. |
| `data.py` | Loads cached RGB/geometry buffers and computes reflection directions. |
| `models.py` | SRM, material, refinement, discriminator, renderer, and perceptual networks. |
| `train.py` | Trains the paper's SRM stage and optional neural novel-view renderer. |
| `envmap.py` | Panorama accumulation, SRM extraction, coverage, and metrics. |
| `eval_env.py` | Converts an SRM checkpoint into a panorama and metrics. |

## Supporting files

| File | Purpose |
|---|---|
| `prepare.py` | One-time mesh ray tracing that creates the arrays in `cache/`. |
| `fit_cans.py` | Estimates coarse can regions; not needed with `--cans all`. |
| `splat.py` | Fast, training-free baseline and geometry sanity check. |
| `label.py` | Estimates a minimum-colour surface layer for `fm_guide.py`. |
| `fm_guide.py` | Experimental FLUX/DiT360 prior guided by reflection-render loss. |
| `evaluate.py` | Evaluates the optional neural novel-view renderer. |
| `INPUTS.md` | Dataset layout and preparation instructions. |

`third_party/DiT360/` is required only by `fm_guide.py`. The dataset
(`cans_data/`), derived cache (`cache/`), checkpoints (`runs/`), and outputs
(`results/`) are retained separately from source code.

## Setup

```bash
cd /Users/maixinnnnn/Desktop/cs180_final_project
source .venv/bin/activate
pip install -r requirements.txt
unset SCENE
```

## Prepare the dataset

Download `cans.zip` from the authors' project page and extract it under
`cans_data/`; see `INPUTS.md`. Then run once:

```bash
python prepare.py
```

The current workspace already contains a prepared cache, so this expensive step
does not need to be repeated unless the raw data changes.

## Validate geometry first

```bash
python splat.py --cans all --views 64 --out splat_check
```

If `results/splat_check/pano.png` lacks recognizable room structure, do not start
neural training: poses, normals, or reflection coordinates are wrong.

## Paper-style SRM training

Full configuration (40 epochs; use a CUDA GPU):

```bash
python train.py srm \
  --run cans_paper \
  --epochs 40 \
  --batch 2 \
  --materials 2 \
  --cans all \
  --lr-srm 0.001 \
  --lambda-p 1.0 \
  --lambda-g 0.05 \
  --log 100 \
  --preview 1000
```

Evaluate:

```bash
python eval_env.py --run cans_paper
```

The panorama is written to `results/cans_paper/pano.png`.

Short CPU smoke test (not a converged reconstruction):

```bash
python train.py srm \
  --run cans_smoke \
  --views 64 \
  --max-steps 100 \
  --materials 2 \
  --cans all \
  --lambda-p 0 \
  --lambda-g 0 \
  --log 10
```

## Optional neural renderer

```bash
python train.py renderer --srm-run cans_paper --run renderer
python evaluate.py --srm-run cans_paper --renderer-run renderer
```

## Optional flow-guided panorama

`fm_guide.py` is separate from the paper reconstruction. It needs a large CUDA
GPU, Hugging Face access to FLUX.1-dev, and the DiT360 LoRA. Validate its data
term before generation:

```bash
python fm_guide.py --samples-only --select above --diffuse texture
```

Do not use flow output to diagnose the SRM renderer; validate `splat.py` and
`train.py srm` first.
