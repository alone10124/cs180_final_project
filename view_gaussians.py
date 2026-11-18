"""Interactive browser viewer for a 3D Gaussian splatting .ply (e.g. Pano2Room's 3DGS.ply).

  python view_gaussians.py scenes/bag/results/room/3DGS.ply            # http://localhost:8080
  ssh -L 8080:localhost:8080 user@gpu-machine                          # from your laptop, then open the URL

Reads the standard 3DGS layout (x y z, f_dc_*, opacity as a logit, scale_* as logs,
rot_* as a wxyz quaternion) and shows it with viser. The up axis of Pano2Room's world is
not documented; switch it in the panel if the room looks tipped over. The same file
also opens in SuperSplat (https://superspl.at/editor) by drag and drop.
"""
import argparse
import time

import numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814
AXES = ("-y", "+y", "+z", "-z", "+x", "-x")


def load(path, max_splats=0):
    """Returns centers (N, 3), covariances (N, 3, 3), rgbs (N, 3) in [0, 1], opacities (N, 1)."""
    v = PlyData.read(str(path))["vertex"]
    centers = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1))
    q = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    w, x, y, z = q.T
    R = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )
    covariances = np.einsum("nij,nj,nkj->nik", R, scales**2, R).astype(np.float32)
    rgbs = np.clip(0.5 + SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1), 0, 1).astype(np.float32)
    opacities = (1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64))))[:, None].astype(np.float32)
    if max_splats and len(centers) > max_splats:
        keep = np.argsort(-opacities[:, 0])[:max_splats]
        centers, covariances, rgbs, opacities = centers[keep], covariances[keep], rgbs[keep], opacities[keep]
    return centers, covariances, rgbs, opacities


def axis_vector(name):
    vector = np.zeros(3)
    vector["xyz".index(name[1])] = 1.0 if name[0] == "+" else -1.0
    return vector


def start_view(centre, up):
    """A camera at the scene centre looking horizontally (perpendicular to `up`)."""
    up = axis_vector(up)
    forward = np.roll(np.abs(up), 1)  # any axis perpendicular to up
    return centre, centre + forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--up", choices=AXES, default="-y", help="world up axis (3DGS scenes are usually y-down)")
    ap.add_argument("--max-splats", type=int, default=3_000_000, help="keep the most opaque N; 0 keeps all")
    args = ap.parse_args()

    import viser

    centers, covariances, rgbs, opacities = load(args.ply, args.max_splats)
    centre = np.median(centers, axis=0)
    print(f"{len(centers)} Gaussians, median position {np.round(centre, 3).tolist()}")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction(args.up)
    server.scene.add_gaussian_splats("/room", centers=centers, covariances=covariances, rgbs=rgbs, opacities=opacities)
    up_choice = server.gui.add_dropdown("Up axis", options=AXES, initial_value=args.up)
    reset = server.gui.add_button("Back to the centre")

    def place(client):
        position, target = start_view(centre, up_choice.value)
        client.camera.up_direction = axis_vector(up_choice.value)
        client.camera.position = position
        client.camera.look_at = target

    @up_choice.on_update
    def _(_event):
        server.scene.set_up_direction(up_choice.value)
        for client in server.get_clients().values():
            place(client)

    @reset.on_click
    def _(event):
        place(event.client)

    server.on_client_connect(place)
    print(f"viewer at http://localhost:{args.port}  (remote: ssh -L {args.port}:localhost:{args.port} user@host)")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
