# sodacan: seeing the world in a soda can

A soda can is a 360° camera, if you know its shape. `sodacan` turns a phone video of a can into a panorama of the room around it. The can is a cylinder of known size, so every pixel's reflection direction is computed exactly, with no depth sensor. A small network trained only on simulated data then sharpens the result, and every enhanced output is shown beside the measured one.

This implements the "Seeing the World in a Soda Can — Claude Code Implementation Plan" (Nov 2026), phases 0 to 5. The earlier reproduction of *Seeing the World in a Bag of Chips* (Park et al., CVPR 2020) is kept as the baseline; see [BASELINE.md](BASELINE.md).

## Install

```
pip install -e .[all]        # numpy, torch, opencv, pyyaml, matplotlib, pytest, lpips, scikit-image, gradio
pytest -m "not slow"         # about a minute; pytest runs everything including the slow dent-map test
```

Runs on CPU and on a 4 GB GPU (developed on an RTX 3050 laptop).

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| 1 Capture | `capture/video.py` | phone video (4K, exposure and focus locked) of the can on a printed ChArUco board |
| 2 Pose | `capture/calibrate.py`, `capture/pose_charuco.py`, `capture/load.py` | one-time phone calibration; per-frame pose from the board; can frame = world |
| 3 Geometry | `geometry/` | ray-cylinder intersection, normals, r = d − 2(d·n)n, Fresnel |
| 4 Label removal | `capture/uv_warp.py`, `separate/label_median.py` | warp frames into surface space (angle × height); per-texel temporal percentile/median/min is the label; frame − label is the reflection |
| 5 Accumulation | `accumulate/` | splat reflection samples into an equirectangular panorama, weighted by Fresnel strength, pixel footprint and viewing angle; the weight map is the coverage mask |
| 6 Restoration | `restore/` | U-Net, RTLViT-style and ViT-5-style networks: blurry panorama + coverage → sharp panorama, trained on simulated pairs |
| 7 Display | `demo/app.py` | Gradio app: measured, prior-enhanced and reference panels, coverage, 360° viewer, "where to move the phone" hint |

Conventions (camera, world, can surface, panorama, board) are in [sodacan/geometry/README.md](sodacan/geometry/README.md), each with a hand-checked test. All image maths is in linear light. All can constants are in [configs/can.yaml](configs/can.yaml): **measure your can with calipers**.

## Quick start

```
python scripts/run_pipeline.py --sim --mirror --out runs/sim_mirror     # simulated perfect mirror
python scripts/run_pipeline.py --sim --out runs/sim_label               # simulated printed can
python -m sodacan.demo.app                                               # demo; "Simulated capture" needs no phone
```

`run_pipeline.py` writes one debug image per stage (`00_frames` … `06_enhanced`), so a wrong pose or flipped axis is visible at once.

## Real captures

1. **Day-1 go/no-go.** Photograph the can near a bright window with locked exposure. If the window is not visible in the can by eye, use a chrome tube or steel tumbler (set `material: {model: metal}` in `configs/can.yaml`); the geometry code is the same.
2. **Board.** `python -m sodacan.capture.calibrate board --out boards/board.png`, print at 100 % (fits landscape Letter/A4), measure a square and fix `configs/board.yaml` if the printer scaled it. Stand the can on the printed circle.
3. **Calibrate the phone once.** Film the board alone from many tilted angles: `python -m sodacan.capture.calibrate calibrate --video board.mp4 --out configs/phone_<name>.yaml`.
4. **Capture.** Lock exposure and focus. Sweep 30–60° around the can and tilt up and down, keeping the board in view. Hold the phone low: a vertical cylinder only reflects as high as the camera looks up at it (r_z = d_z), so a can on a table shows a strip around the horizon.
5. **Run.** `python scripts/run_pipeline.py --video can.mp4 --phone configs/phone_<name>.yaml --reference room_pano.jpg --model runs/restore/unet/best.pt --out runs/real/<room>`. The reference is a phone or 360° panorama taken from the can's position; its heading is found automatically (`metrics.align_azimuth`).
6. **Close the sim-to-real gap.** `python -m sodacan.refine.roughness_fit --video can.mp4 --phone ... --reference ... --out runs/fit_<room> --save-label --compare-sim 20` fits roughness, reflection-to-label strength and noise, compares them with the simulator's distribution, and saves the real label atlas. Centre the ranges in `configs/sim.yaml` on the fitted values.

## Training (Kaggle)

```
python -m sodacan.sim.dataset --benchmark 20                      # pairs per second on this machine
python -m sodacan.sim.dataset --count 10000 --out data/pairs      # shards; resumable
python -m sodacan.restore.train --config configs/train_unet.yaml --out runs/restore/unet --resume --max-minutes 690
python scripts/evaluate_restore.py --models runs/restore/*/best.pt
```

`notebooks/kaggle_simulate.ipynb` and `notebooks/kaggle_train.ipynb` are thin wrappers (clone, `pip install -e`, call the commands above). Training checkpoints `last.pt` every 30 minutes and on `--max-minutes`, keeps `best.pt` by validation PSNR on held-out simulated rooms, and `--resume` continues after a cut-off session. Put HDR room panoramas (Poly Haven, CC0) in `data/hdri/`; without them the simulator uses random procedural rooms.

## Figures

```
python scripts/make_figures.py --models runs/restore/*/best.pt [--slow]
```

Regenerates every figure and table in `figures/` from one script: the Phase 0 round trip, single frame vs video, the label-removal ablation, a printed-can example with measured / enhanced / true panels, relighting, the restoration table, every real capture under `runs/real/`, and (with `--slow`) the dent map.

## Status against the plan

Measured on an RTX 3050 laptop (4 GB), 17 Nov 2026. "Sim" means simulated rooms; nothing below has been checked on a real can yet.

| Phase | Acceptance test | Status |
|---|---|---|
| 0 Geometry | perfect-mirror round trip > 30 dB inside coverage | **passes**: 33.4 dB, 42 % of texels (`tests/test_roundtrip.py`); 31.5 dB at 256 × 512 (`figures/roundtrip.png`) |
| 1 Simulator | 10,000 pairs in one Kaggle session | **passes locally**: 1.6 pairs/s on the 3050, so 10k pairs in about 1.8 h (about 5.7 GB of shards) |
| 1 Simulator | Blender frames through Phase 0 code | **not run**: Blender is not installed here. Export, Cycles script and test are written (`sim/blender_export.py`, `sim/blender_scene.py`, `tests/test_sim_vs_blender.py`, skipped until renders exist) |
| 2 Real capture | window and lights in the right places against a phone panorama | **needs a real video**. On synthetic board videos: ChArUco poses within 0.03° and 0.3 mm of the truth, mp4 → panorama at 31.4 dB (`tests/test_capture.py`) |
| 3 Restoration | beats classical on held-out sim rooms in PSNR and LPIPS | **passes (sim)** with a U-Net trained locally for 5k steps on 1.5k pairs: PSNR 17.7 → 21.8 dB, SSIM 0.45 → 0.68, LPIPS 0.53 → 0.43 over 64 held-out rooms (`figures/restore_table.md`) |
| 3 Restoration | visibly sharper on three real rooms | **not met yet**: no real rooms, and the local L1-trained model mostly denoises and smooths (`figures/sim_example.png`). The Kaggle configs add an LPIPS term (0.1) against that blur. RTLViT-style and ViT-5-style models are implemented and smoke-trained only |
| 4 Demo | 10 s clip end to end in < 60 s on the 3050 | **passes**: 9.7 s for 1080p30, 26.5 s for 4K30 (`scripts/benchmark_demo.py`, synthetic clip). The app was driven end to end through its API |
| 5 Report | all figures from one script | **passes**: `scripts/make_figures.py`. Extensions done: relighting a virtual object; dent map (24.9 → 30.2 dB on a dented mirror can in the test) |

Findings worth knowing:

- **Label removal:** the per-texel temporal **minimum** works best on simulated printed cans (18.8 dB, against 19.0 with the true label). The median is far worse (9.5 dB), because a reflection that only slides partway across a texel stays in the median. `min` is the default (`figures/label_ablation.*`).
- **Vertical coverage:** a vertical cylinder keeps the camera ray's vertical component (r_z = d_z). A can standing on a table therefore only reflects a band around the horizon, because the phone cannot look up at it from below. Expect a strip panorama, or raise the can on a stand.
- **Single frame vs video:** one frame covers 13 % of the sphere. About 8 frames reach most of the coverage an arc can give (about 56 %), and more frames then average out noise (`figures/frames_*.png`).
- **Honesty:** every enhanced output is shown beside the measured one, and unobserved texels stay dark blue.

Not done: the day-1 window photo, real captures, phone calibration, Kaggle training of the full 10k-pair runs, Blender renders, and the Bag of Chips dataset comparison through this pipeline. The no-board fallback (pose from the can's silhouette) is not implemented.

## Repository layout

```
sodacan/
  geometry/     camera.py cylinder.py reflect.py equirect.py fresnel.py README.md
  sim/          fast_render.py rooms.py trajectories.py dataset.py blender_export.py blender_scene.py
  capture/      video.py board.py calibrate.py pose_charuco.py segment_can.py uv_warp.py load.py
  separate/     label_median.py
  accumulate/   splat.py weights.py
  restore/      models/ (unet.py rtlvit_style.py vit5_blocks.py) common.py losses.py train.py infer.py
  refine/       dent_map.py roughness_fit.py
  extensions/   relight.py
  demo/         app.py run.py
  pipeline.py metrics.py color.py config.py debug.py
configs/        can.yaml pipeline.yaml board.yaml sim.yaml phone_example.yaml train_*.yaml
notebooks/      kaggle_simulate.ipynb kaggle_train.ipynb (no logic)
tests/          test_roundtrip.py test_geometry.py test_capture.py test_restore.py test_refine.py test_sim_vs_blender.py
scripts/        run_pipeline.py make_figures.py evaluate_restore.py benchmark_demo.py
data/           hdri/ pairs/ blender/ labels/ captures/ (git-ignored)
```

The Bag of Chips baseline scripts (`prepare.py`, `train.py`, `splat.py`, …) remain at the repository root; see [BASELINE.md](BASELINE.md).
