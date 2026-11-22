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
| 2b Pose, no board | `capture/pose_silhouette.py` | fallback: the can's two side walls, rim and base fix five pose numbers; the label's shift in surface space fixes the angle around the can |
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
6. **No board?** Add `--no-board` (and optionally `--roi x0,y0,x1,y1`, a box around the can in the first frame). Poses then come from the can's outline and its printed label (`capture/pose_silhouette.py`): about 1° instead of the board's 0.03°, and about 6 s per frame. Set `rim_radius` and `base_radius` in `configs/can.yaml` for a real can, keep the can upright on a plain surface, and use a printed can: a blank mirror can's angle around its axis cannot be seen.
7. **Close the sim-to-real gap.** `python -m sodacan.refine.roughness_fit --video can.mp4 --phone ... --reference ... --out runs/fit_<room> --save-label --compare-sim 20` fits roughness, reflection-to-label strength and noise, compares them with the simulator's distribution, and saves the real label atlas. Centre the ranges in `configs/sim.yaml` on the fitted values.

## Record3D captures (iPhone LiDAR)

```
python r3d_can.py --r3d capture.r3d --out runs/r3d --inspect     # metadata, frames, depth only
python r3d_can.py --r3d capture.r3d --out runs/r3d [--model runs/restore/unet/best.pt] [--reference room_pano.jpg]
```

Export from Record3D as "Shareable/Internal format (.r3d)". The ARKit poses in its metadata replace the board, and the can's position comes from the LiDAR depth: the table is the dominant horizontal plane near the image centre, and the can is a circle of radius `configs/can.yaml` standing on it. Then `pipeline.reconstruct` runs unchanged.
Check `01_poses.png` (the outline must sit on the can) and `01_fit.png` (red depth points on the near side of the green circle) before reading the panorama; `--center x,z` overrides the fit.
Tested on a simulated `.r3d` (centre within 0.1 mm, panorama 32 dB against the true poses); not yet confirmed on a real Record3D export.

## Training (Kaggle)

```
python -m sodacan.sim.dataset --benchmark 20                      # pairs per second on this machine
python -m sodacan.sim.dataset --count 10000 --out data/pairs      # shards; resumable
python -m sodacan.restore.train --config configs/train_unet.yaml --out runs/restore/unet --resume --max-minutes 690
python scripts/evaluate_restore.py --models runs/restore/*/best.pt
```

`notebooks/kaggle_run.sh` is what was actually run on Kaggle (GPU T4 x2, with the code uploaded as a private dataset rather than cloned): it simulates 5,000 pairs on each T4 (37 min) and then trains RTLViT and ViT-5 side by side, one per T4 (45 min for 15k steps each). `notebooks/kaggle_simulate.ipynb` and `notebooks/kaggle_train.ipynb` are thin wrappers (clone, `pip install -e`, call the commands above). Training checkpoints `last.pt` every 30 minutes and on `--max-minutes`, keeps `best.pt` by validation PSNR on held-out simulated rooms, and `--resume` continues after a cut-off session. Put HDR room panoramas (Poly Haven, CC0) in `data/hdri/`; without them the simulator uses random procedural rooms.

## Figures

```
python scripts/make_figures.py --models runs/restore/*/best.pt [--slow]
```

Regenerates every figure and table in `figures/` from one script: the Phase 0 round trip, single frame vs video, the label-removal ablation, a printed-can example with measured / enhanced / true panels, relighting, the restoration table, every real capture under `runs/real/`, and (with `--slow`) the dent map.

## Status against the plan

Measured on an RTX 3050 laptop (4 GB) and Kaggle T4s, 18 Nov 2026. "Sim" means simulated rooms; nothing below has been checked on a real can yet.

| Phase | Acceptance test | Status |
|---|---|---|
| 0 Geometry | perfect-mirror round trip > 30 dB inside coverage | **passes**: 33.4 dB, 42 % of texels (`tests/test_roundtrip.py`); 31.5 dB at 256 × 512 (`figures/roundtrip.png`) |
| 1 Simulator | 10,000 pairs in one Kaggle session | **passes**: on Kaggle, 10k pairs in 37 min over two T4s (5k each); locally, 1.2–1.6 pairs/s on the 3050, so about 2.4 h (4.5 GB of shards) |
| 1 Simulator | Blender frames through Phase 0 code | **passes**: 10 Cycles scenes (Blender 4.5, OptiX, 128 spp): 9 pass and 1 is skipped because the camera looks down and the can reflects only the table. Above the horizon, Cycles frames reconstruct the room at a median 17.9 dB and fast-renderer frames at 15.6 dB; the two agree at 18.2 dB (`tests/test_sim_vs_blender.py`, `figures/blender.*`) |
| 2 Real capture | window and lights in the right places against a phone panorama | **needs a real video**. On synthetic board videos: ChArUco poses within 0.03° and 0.3 mm of the truth, mp4 → panorama at 31.4 dB (`tests/test_capture.py`) |
| 2 Pose without a board | fallback from the can's outline | **works in simulation** (`capture/pose_silhouette.py`, `--no-board`). The outline gives five of the six pose numbers; the label gives the angle around the can, after removing what is locked to the view (the reflection and the bright edges) by a joint fit of label plus background. On a printed can with no board: 15 of 16 frames posed, median errors 1.3° and 7 mm, 0.5° around the axis (`tests/test_pose_silhouette.py`). On a shinier coat whose reflection outweighs the label (strength 1.0, 40 frames over 60°): all frames posed, median 2.4° and 12 mm, worst angle error 2.4°, and the panorama comes out at 14.5 dB against 15.5 dB from the true poses (`figures/silhouette.*`); matching the raw label there drifted to 51°. About 40× less accurate than the board and about 6 s per frame, so the board stays the default |
| 3 Restoration | beats classical on held-out sim rooms in PSNR and LPIPS | **passes (sim)**, all three architectures, 64 held-out rooms (`figures/restore_table.md`). Classical: 17.7 dB, SSIM 0.45, LPIPS 0.53. U-Net, L1 only, 5k steps: 21.8 dB, 0.68, 0.43. **U-Net with LPIPS, 30k steps on 10k pairs (local, 3.6 h): 23.6 dB, 0.77, 0.23**, the best on every metric. RTLViT-style, 15k steps (local): 22.7 dB, 0.71, 0.26. ViT-5-style, 15k steps (Kaggle T4): 22.4 dB, 0.73, 0.27 |
| 3 Restoration | visibly sharper on three real rooms | **not checked**: no real rooms yet. On simulation the LPIPS term fixed the L1 model's blur (LPIPS 0.43 → 0.23; `figures/sim_example.png`). Training first ran with LPIPS silently doing nothing: `linear_to_srgb` had a NaN gradient at 0, and AMP's scaler skipped every step, so the fix (`color.py`) and fp32 LPIPS (`restore/losses.py`) come with tests |
| 4 Demo | 10 s clip end to end in < 60 s on the 3050 | **passes**: 9.7 s for 1080p30, 26.5 s for 4K30 (`scripts/benchmark_demo.py`, synthetic clip). The app was driven end to end through its API |
| 5 Report | all figures from one script | **passes**: `scripts/make_figures.py`. Extensions done: relighting a virtual object; dent map (24.9 → 30.2 dB on a dented mirror can in the test) |

Findings worth knowing:

- **Label removal:** the per-texel temporal **minimum** works best on simulated printed cans (18.8 dB, against 19.0 with the true label). The median is far worse (9.5 dB), because a reflection that only slides partway across a texel stays in the median. `min` is the default (`figures/label_ablation.*`).
- **Vertical coverage:** a vertical cylinder keeps the camera ray's vertical component (r_z = d_z). A can standing on a table therefore only reflects a band around the horizon, because the phone cannot look up at it from below. Expect a strip panorama, or raise the can on a stand.
- **Single frame vs video:** one frame covers 13 % of the sphere. About 8 frames reach most of the coverage an arc can give (about 56 %), and more frames then average out noise (`figures/frames_*.png`).
- **Honesty:** every enhanced output is shown beside the measured one, and unobserved texels stay dark blue.

- **The table is in the reflection.** Cycles renders the table the can stands on, so every reflection pointing below the horizon shows that table, not the room's floor. The fast renderer treats the room as infinitely far away and shows the floor there instead. This is the largest gap between the two renderers: they agree at 9.7–16 dB over the whole panorama, against 15–22 dB above the horizon. A real can on a table behaves like Cycles, so the simulator should trace the table in its reflections before the next large training run.
- **Blender cylinders need open walls.** When the caps share the wall's vertices, smooth shading bends the wall's normals like a barrel, and a window reflects as a dot instead of a streak. `sim/blender_scene.py` now builds an open tube with a separate lid.

Not done: the day-1 window photo, real captures, phone calibration, and the Bag of Chips dataset comparison through this pipeline.

## Repository layout

```
sodacan/
  geometry/     camera.py cylinder.py reflect.py equirect.py fresnel.py README.md
  sim/          fast_render.py rooms.py trajectories.py dataset.py blender_export.py blender_scene.py
  capture/      video.py board.py calibrate.py pose_charuco.py segment_can.py uv_warp.py load.py pose_silhouette.py
  separate/     label_median.py
  accumulate/   splat.py weights.py
  restore/      models/ (unet.py rtlvit_style.py vit5_blocks.py) common.py losses.py train.py infer.py
  refine/       dent_map.py roughness_fit.py
  extensions/   relight.py
  demo/         app.py run.py
  pipeline.py metrics.py color.py config.py debug.py
configs/        can.yaml pipeline.yaml board.yaml sim.yaml phone_example.yaml train_*.yaml
notebooks/      kaggle_simulate.ipynb kaggle_train.ipynb (no logic)
tests/          test_roundtrip.py test_geometry.py test_capture.py test_restore.py test_refine.py test_sim_vs_blender.py test_pose_silhouette.py
scripts/        run_pipeline.py make_figures.py evaluate_restore.py benchmark_demo.py
r3d_can.py      Record3D .r3d -> poses in the can frame -> pipeline
data/           hdri/ pairs/ blender/ labels/ captures/ (git-ignored)
```

The Bag of Chips baseline scripts (`prepare.py`, `train.py`, `splat.py`, …) remain at the repository root; see [BASELINE.md](BASELINE.md).
