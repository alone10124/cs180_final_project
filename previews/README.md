# Previews from the test runs

Small test scenes: 480x360, 120 frames (105 training views), mirror cans, splatting baseline unless noted.
Panoramas are in the repo's lookup layout, which is mirrored relative to a camera panorama.
In estimate/reference pairs the estimate is on top; dark blue marks unobserved texels.

- `photos_*.png`: six analytic photos per scene
- `environment_ground_truth.png`: the room used to light every scene
- `analytic_vs_blender.png`: analytic tracer (left) against Blender Cycles (right), 320x240
- `sweep_<scene>_*.png|csv`: can A / can B / both / both at half the pixels, over 4, 16, 64 and all views
  (`sweep_*_panoramas.png`: rows are view counts, columns are settings)
- `pano_*.png`: estimate over reference at all views
- `merge_one_vs_two_*.png`: can A in the one-can scene against the two-can scene
- `label_ablation_panoramas.png`: no subtraction, given diffuse layer, estimated label, reference
- `inpaint_telea_compare.png`: recovered, filled (OpenCV Telea, not generative), reference
- `walkthrough*.{mp4,png}`: box-room walkthrough of the filled neck-scene panorama
- `srm_8views_*.png`: SRM training on a 160x120 scene, 8 views, with and without the refinement network
