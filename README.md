# Seeing the World in a Can of Coke

How much of a room can be recovered from the reflections on one soda can, and what a second can adds.
Built on a reproduction of Park et al., "Seeing the World in a Bag of Chips" (CVPR 2020).

## Setup

```
pip install -r requirements.txt
```

`embreex` is only needed by `prepare.py` (Cans mesh), `lpips` by `evaluate.py`, `diffusers` by `inpaint.py --method diffusion`, and Blender by `blender_scene.py`.
Training uses CUDA when available and falls back to CPU.

## Scenes

Every script works on one scene.
With no `SCENE` variable that is the Cans dataset at the repo root (`cans_data/`, `cache/`, `runs/`, `results/`).
`SCENE=<name>` switches to `scenes/<name>/` with the same four folders, and `scenes/<name>/data/scene.json` supplies the camera and the can positions.

| Scene | How to create it |
|---|---|
| Cans dataset | download into `cans_data/`, then `python prepare.py` and `python fit_cans.py` |
| Synthetic | `python synth.py --name one --layout one` (see below) |
| Own photos | `python capture.py ...` from a COLMAP model (see below) |

## Synthetic scenes

```
python synth.py --name one --layout one            # one mirror cylinder
python synth.py --name two --layout two            # the same cylinder plus a second one
python synth.py --name neck --layout two --neck 0.014          # cans with a tapered neck
python synth.py --name label --layout two --neck 0.014 --label # printed label on the cans
```

`synth.py` builds a box room as an HDR panorama (`data/env_gt.hdr`), traces it analytically, and writes the cache that `prepare.py` would.
Useful options: `--frames`, `--width/--height`, `--table` (half-size of the table top), `--elevation lo hi`, `--blur-deg` (rough metal), `--env file.hdr`.

To render the photos in Blender instead:

```
blender -b -P blender_scene.py -- two
python synth.py --name two_blender --layout two --photos scenes/two/data/blender
```

## Reconstruction

Training-free baseline (averages photo minus diffuse layer by reflection direction):

```
SCENE=two python splat.py --cans a --views 32
SCENE=two python splat.py --cans both --out both
```

SRM optimisation from the paper, restricted to chosen cans and views:

```
SCENE=two python train.py srm --run srm_a_32 --cans a --views 32 --max-steps 8000 --lambda-g 0
SCENE=two python eval_env.py --run srm_a_32
```

Adam moves a panorama texel by at most `--lr-srm` (default 1e-3) per step, and a texel only moves when a frame that sees it is in the batch.
With many views and a short `--max-steps` the panoramas stay dark; raise `--lr-srm` or the step budget until `gain` in `metrics.json` stops falling.

Both write `results/<name>/pano.png` (estimate above reference, unobserved texels in dark blue) and `metrics.json`:
`coverage` (fraction of the sphere observed, solid-angle weighted), `coverage_upper` (upper hemisphere), and, when the scene has a reference environment, `psnr` and `ssim` over observed texels after gain matching and blur matching (`blur_deg`).

## Experiments

```
SCENE=two python experiments.py sweep --views 4 8 16 32 64 0          # can A, can B, both, both at half the pixels
SCENE=two python experiments.py sweep --method srm --views 16 64 --steps 4000
python experiments.py merge --scenes one two                           # one-can scene against two-can scene
SCENE=label python experiments.py ablation                             # label separation
```

`both_half` uses both cans but drops half their pixels at random, to separate "more pixels" from "new directions".
`psnr_shared` scores every setting on the texels all of them observe.

## Extensions

```
SCENE=two python inpaint.py --input both --method telea       # or --method diffusion
SCENE=two python walkthrough.py --input both_telea --room 5 4 2.7 --eye 0.8
```

Own capture:

```
python capture.py --name mine --images photos --model colmap_txt --cans 2 --reference room_pano.jpg
SCENE=mine python label.py
SCENE=mine python splat.py --buffer diffuse_est --align
```

Check `scenes/mine/data/check_*.jpg` first: the green outlines must sit on the cans.

## Conventions

- World: table is the plane z = 0, z up, metres. Poses are camera-to-world; camera x right, y down, z forward.
- Panoramas: 500 x 250, column from the azimuth `atan2(y, x)`, row from the polar angle (row 0 is straight up), as in `SRMModel.lookup`.
  This layout is the mirror image of a camera panorama: turning right moves left in the image. Flip horizontally for figures meant to look like a photo.
- Can ids: `a` is can 1, `b` is can 2, in the order listed in `scene.json` or `cache/cans.json`.

## Files

| File | Role |
|---|---|
| `common.py` | paths, camera, split, view selection, panorama geometry |
| `prepare.py`, `fit_cans.py` | Cans dataset: geometry buffers from the mesh; can positions |
| `synth.py`, `blender_scene.py` | synthetic scenes |
| `capture.py`, `label.py` | own photos; label estimation |
| `data.py`, `models.py`, `train.py`, `evaluate.py` | the paper's pipeline and novel-view evaluation |
| `envmap.py`, `splat.py`, `eval_env.py` | environment reconstruction, coverage and metrics |
| `experiments.py` | sweeps, ablation, charts |
| `inpaint.py`, `walkthrough.py` | generative fill and 2.5D walkthrough |
