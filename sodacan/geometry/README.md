# Coordinate conventions

Each convention below has a hand-checked test in `tests/test_geometry.py`.

## World = can frame
- Metres, right-handed, **z up**.
- The can's axis is the z axis, its base centre is the origin. The table (or ChArUco board) is the plane z = 0.
- The can is the cylinder x² + y² = R², 0 ≤ z ≤ height (`configs/can.yaml`). Only the straight band `usable_z` is used for reflections.

## Camera
- OpenCV axes: **x right, y down, z forward**. Poses are **camera-to-world** 4×4 matrices: columns 0–2 are the camera axes in world coordinates, column 3 is the camera centre.
- Pixel (column j, row i) has its centre at image coordinate (j, i). A pixel's ray is ((j − cx)/fx, (i − cy)/fy, 1), normalised.
- Check: a camera at +x looking at the origin has right = +y, down = −z (`test_look_at_axes`). A point at +y projects right of centre, a point at +z above it (`test_projection_right_and_down`).

## Can surface (atlas)
- Surface coordinates (s, h): s = atan2(y, x) ∈ [−π, π) is the angle around the can, h = z.
- The atlas is an image over (s, h). Column j ↔ s = −π + (j + ½)/W · 2π, and row i ↔ h = height · (1 − (i + ½)/H), so **row 0 is the rim**. Increasing s reads left to right when the can is seen from outside, so a printed label appears unmirrored (`test_surface_atlas_not_mirrored`).
- The atlas wraps in s.

## Reflection
- r = d − 2(d·n)n, where d is the unit camera ray and n the outward normal (x/R, y/R, 0). The incidence cosine is −d·n.
- A vertical cylinder keeps the vertical component: r_z = d_z (`test_reflection_keeps_vertical_component`). The can widens the view only horizontally; vertical coverage comes from tilting the phone.
- A head-on ray reflects straight back toward the camera (`test_render_head_on_sees_behind_camera`).
- The room is assumed to be far away: the reflection depends only on r, not on the hit point.

## Panorama (equirectangular)
- (C, H, W), linear light. Column from the azimuth φ = atan2(y, x): u = (φ/2π + ½)·W. Row from the polar angle θ = acos(z): v = θ/π · H, so **row 0 is straight up** and the horizon is the middle row. Texel (i, j) has its centre at (u, v) = (j + ½, i + ½).
- Landmarks: +x is at the centre column, +y is three quarters across, −x is on the seam (`test_equirect_landmarks`).
- This is the sphere **seen from outside**: +y (to the left of someone facing +x) sits right of +x. `equirect.to_viewer` flips it into the usual photo panorama, and `sim.rooms.photo_to_pano` flips photo panoramas (Poly Haven, phone panoramas) into this layout.
- Splatting and sampling are bilinear, wrap in azimuth and clamp at the poles. Sampling then splatting at texel centres is the identity (`test_sample_and_splat_are_consistent`).

## ChArUco board
- OpenCV board coordinates: origin at the board's top-left corner as printed, x along the columns, y along the rows, z into the table.
- World = diag(1, −1, −1) · (board − c), where c is `can_centre` from `configs/board.yaml`. So world x follows the board's x, and world z points up out of the board (`tests/test_capture.py`).

## Light
- All image maths is in linear light. Video frames are decoded with the sRGB curve (`color.linearize`) before any subtraction or averaging.
