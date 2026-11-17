# Inputs you need to provide

What each part of the project needs from you, in what format, and where it goes.
Formats are what the code reads; anything marked "generated" is produced by a script and needs nothing from you.

There are three kinds of scene. Each needs the same four ingredients: photos, camera poses, object geometry, and a diffuse (label) layer. They differ in who supplies them.

| Ingredient | Cans dataset | Synthetic | Your own photos |
|---|---|---|---|
| Photos | you download | generated | **you shoot** |
| Camera poses | you download | generated | **you run COLMAP** |
| Geometry (3D model) | you download (scanned mesh) | generated (exact cylinders) | fitted cylinders, from **the can's real size** |
| Diffuse / label layer | you download (texture atlas) | generated | estimated from the photos |
| Reference environment | none exists | generated | **optional: you shoot a room panorama** |

---

## 1. Cans dataset (real data from the paper)

Download from the paper's project page, https://grail.cs.washington.edu/projects/bagofchips/, and place the files in `cans_data/` at the repo root.
I have not seen the download itself, so check the file names against this table and rename if they differ.

| File | What it is | Format the code expects |
|---|---|---|
| `cans_data/rgb/0001.png` ... | the photos | 640 x 480, numbered from 1, one per pose (2265 frames in the existing log) |
| `cans_data/xy_poses.txt` | camera poses | one header line; then per frame: index, translation (3), rotation (9, row-major); camera-to-world |
| `cans_data/xy_mesh.ply` | 3D model of the scene: both cans and the floor in one mesh | ASCII PLY; vertex columns x y z nx ny nz u v; triangle faces; z up, floor at z = 0, metres |
| `cans_data/texturemapflip.png` | surface colour (the printed labels and floor) | texture atlas indexed by the mesh UVs |

Camera intrinsics are constants in `common.py` (focal 541.961 px, principal point 320, 240, no distortion). The dataset's depth images are not used.

Then run, once:

```
python prepare.py      # CPU ray tracing, about 22 min, writes cache/ (about 12.5 GB)
python fit_cans.py     # finds the two cans in the mesh, writes cache/cans.json
```

Check that `fit_cans.py` prints a radius near 0.033 m and a height near 0.122 m for each can. If it does not, the per-can masks will be wrong.

There is no ground-truth environment for this scene, so it yields coverage and qualitative panoramas, not PSNR/SSIM.

---

## 2. Synthetic scenes

Nothing is required. `synth.py` creates the room, the cans, the cameras, the photos and the ground-truth environment.

Optional inputs:

| Input | Flag | Format |
|---|---|---|
| Your own environment | `--env room.hdr` | equirectangular Radiance `.hdr`, linear, any size (resampled to 2000 x 1000); column from azimuth, row 0 at the zenith |
| Blender photos instead of the analytic ones | `--photos <dir>` | `frame_0001.hdr` ..., written by `blender_scene.py`; needs Blender 4 or 5 installed |

Scene choices are flags, not files: `--layout one|two`, `--neck`, `--label`, `--table`, `--elevation`, `--frames`, `--width/--height`, `--blur-deg`.

---

## 3. Your own photos of a can

### 3a. Photos (required)

- 50 to 200 photos of the can(s) standing on a flat table, taken while moving around them. Include low viewpoints near table height: a can's side wall only reflects above the horizon when seen from low down.
- **Nothing in the room may move**, including you where possible: your own reflection is part of the environment the method recovers.
- Lock exposure, white balance and focus (the method assumes one exposure for all photos).
- The table needs visible texture (a tablecloth, newspaper) and the surroundings need features, because COLMAP cannot match points on the mirror surface itself.
- One camera, one zoom setting. JPEG or PNG; they are resized to 640 wide.

### 3b. Camera poses: a COLMAP model (required)

Run COLMAP on the photos and export the sparse model as text:

```
colmap automatic_reconstructor --workspace_path ws --image_path photos --single_camera 1
colmap model_converter --input_path ws/sparse/0 --output_path ws/txt --output_type TXT
```

`capture.py` reads `cameras.txt`, `images.txt` and `points3D.txt` from that folder.
Lens distortion is ignored, so for a wide lens undistort first (`colmap image_undistorter`) and use the undistorted images and model.

### 3c. Geometry: the can's real dimensions (required, defaults provided)

No 3D scan is needed. Each can is modelled as a vertical cylinder fitted to the sparse points, and the scene is scaled so the cylinder has the real radius.

| Input | Flag | Default |
|---|---|---|
| Number of cans | `--cans` | 1 |
| Real radius | `--radius` | 0.033 m (standard 12 oz can) |
| Real height | `--height` | 0.122 m |

After `capture.py`, open `scenes/<name>/data/check_*.jpg`. The green outlines must sit on the cans. If they do not, the reflection directions are wrong and nothing downstream is meaningful.
The cylinder model has no tapered neck or lid detail, so those parts of a real can are modelled incorrectly.

### 3d. Label layer (generated)

`SCENE=<name> python label.py` estimates the printed label from the photos. No scan of the label is needed.

### 3e. Reference panorama of the room (optional, needed for PSNR/SSIM)

| Input | Flag | Format |
|---|---|---|
| A 360 photo of the room taken from where the can stood | `capture.py --reference room.jpg` | equirectangular, 2:1, upright (horizon on the middle row), as produced by a 360 camera or a phone panorama app |

Its heading is unknown, so evaluate with `--align`. Without a reference you still get coverage and the panorama itself.

---

## 4. Pretrained weights (downloaded automatically, need internet)

| Used by | Weights | Source |
|---|---|---|
| `train.py` (perceptual loss; skip with `--lambda-p 0`) | VGG-19 ImageNet | torchvision |
| `evaluate.py` | LPIPS (VGG) | `lpips` package |
| `inpaint.py --method diffusion` | `stabilityai/stable-diffusion-2-inpainting`, or any inpainting checkpoint via `--model` | Hugging Face; may require accepting a licence or logging in |

---

## 5. Parameters the walkthrough cannot infer

`walkthrough.py` estimates the wall heading from the panorama, but one panorama does not fix the room's size. You provide:

| Input | Flag | Default |
|---|---|---|
| Room length, width, height | `--room` | 5 4 2.7 m |
| Height of the can above the floor | `--eye` | 0.8 m |
| Can's offset from the room centre | `--offset` | 0 0 |

---

## 6. Compute

| Step | Needs |
|---|---|
| `prepare.py` | CPU, about 22 min, 12.5 GB disk |
| `synth.py`, `splat.py`, `experiments.py sweep` (splat), `label.py`, `capture.py`, `walkthrough.py` | CPU is enough |
| `train.py`, `experiments.py sweep --method srm`, `evaluate.py`, `inpaint.py --method diffusion` | GPU; see `colab_gpu.ipynb` |
