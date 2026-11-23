# Real Cans dataset inputs

Download the **Cans** archive from:

https://grail.cs.washington.edu/projects/bagofchips/

Extract it under `cans_data/`:

```text
cans_data/
  rgb/0001.png ...          640x480 RGB frames
  depth/0001.png ...        released depth frames (not used by `prepare.py`)
  xy_poses.txt              camera-to-world poses
  xy_mesh.ply               reconstructed scene mesh with normals and UVs
  texturemapflip.png        diffuse texture atlas
```

Camera calibration is fixed in `common.py`: focal length 541.961 px and principal
point `(320, 240)`. RGB linearization uses `pixel ** 1.6`, as specified by the
dataset authors.

Run `python prepare.py` once. It writes approximately 12.5 GB under `cache/`:

- `rgb.npy`: captured images
- `depth.npy`: rendered mesh depth
- `normal.npy`: world-space normals
- `diffuse.npy`: rendered diffuse texture
- `fbi.npy`: first-bounce buffer
- `vis.npy`: reflected-ray visibility
- `split.npz`: train/test indices and scene centre

The Cans scene has no reference environment panorama, so evaluation is qualitative
plus reflection-direction coverage. It is a different capture from the paper's
Corncho/bag panorama.
