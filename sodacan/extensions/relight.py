"""Extension: relight a virtual object with the recovered panorama.

Renders a sphere that is half diffuse, half glossy, lit by a panorama: diffuse shading
from the 9-term spherical-harmonic irradiance (Ramamoorthi and Hanrahan 2001), glossy
shading as a lookup of the blurred panorama along the reflection direction. Unobserved
texels are filled with the mean observed radiance of their row band (or the global
mean), which is a guess and is labelled as such in figures. Comparing the sphere lit by
the recovered panorama with the sphere lit by the true room shows whether the panorama
is physically useful, not just recognisable.

  python -m sodacan.extensions.relight --run runs/sodacan/sim_mirror --out relight.png
"""
import argparse
import math

import numpy as np
import torch

from sodacan.geometry import equirect


def fill_unobserved(pano, valid):
    out = pano.clone()
    if not valid.any():
        return out
    mean = pano[:, valid].mean(1)
    H = pano.shape[1]
    for r0 in range(0, H, H // 8):
        rows = slice(r0, r0 + H // 8)
        v = valid[rows]
        m = pano[:, rows][:, v].mean(1) if v.sum() > 20 else mean
        out[:, rows] = torch.where(v[None], pano[:, rows], m[:, None, None])
    return out


def sh_irradiance_coeffs(pano):
    H, W = pano.shape[1:]
    d = equirect.pano_dirs(H, W, pano.device)
    x, y, z = d.unbind(-1)
    basis = torch.stack([
        torch.full_like(x, 0.282095), 0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)], 0)
    w = equirect.solid_angle(H, W, pano.device)
    L = torch.einsum("chw,khw->ck", pano * w, basis)  # (3, 9)
    A = torch.tensor([math.pi, *[2 * math.pi / 3] * 3, *[math.pi / 4] * 5], device=pano.device)
    return L * A


def sh_eval(coeffs, n):
    x, y, z = n.unbind(-1)
    basis = torch.stack([
        torch.full_like(x, 0.282095), 0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)], -1)
    return (basis @ coeffs.T).clamp_min(0) / math.pi


def render_sphere(pano, size=256, albedo=(0.7, 0.7, 0.7), gloss_deg=4.0, view_az_deg=0.0):
    """A sphere seen from azimuth view_az_deg (horizontal view): left half diffuse, right half glossy."""
    dev = pano.device
    u = (torch.arange(size, device=dev) + 0.5) / size * 2 - 1
    yy, xx = torch.meshgrid(-u, u, indexing="ij")
    inside = xx**2 + yy**2 < 1
    zz = torch.sqrt((1 - xx**2 - yy**2).clamp_min(0))
    a = math.radians(view_az_deg)
    fwd = torch.tensor([-math.cos(a), -math.sin(a), 0.0], device=dev)  # camera looks along fwd
    right = torch.tensor([-math.sin(a), math.cos(a), 0.0], device=dev)  # fwd x up
    up = torch.tensor([0.0, 0.0, 1.0], device=dev)
    n = xx[..., None] * right + yy[..., None] * up - zz[..., None] * fwd
    diffuse = sh_eval(sh_irradiance_coeffs(pano), n) * torch.tensor(albedo, device=dev)
    r = fwd - 2 * (n * fwd).sum(-1, keepdim=True) * n
    glossy = equirect.sample(equirect.blur(pano, gloss_deg), r)
    img = torch.where((xx < 0)[..., None], diffuse, glossy)
    return torch.where(inside[..., None], img, torch.zeros_like(img)).permute(2, 0, 1), inside


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output folder of scripts/run_pipeline.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from pathlib import Path

    from sodacan import debug

    d = np.load(Path(args.run) / "panorama.npz")
    pano, valid = torch.from_numpy(d["pano"]), torch.from_numpy(d["valid"])
    imgs, labels = [], []
    for name in ("pano", "enhanced"):
        if name in d:
            s, _ = render_sphere(fill_unobserved(torch.from_numpy(d[name]), valid))
            imgs.append(s)
            labels.append(f"lit by {'measured' if name == 'pano' else 'enhanced'}")
    if "reference" in d:
        s, _ = render_sphere(torch.from_numpy(d["reference"]))
        imgs.append(s)
        labels.append("lit by true room")
    e = 0.18 / float(torch.stack(imgs).mean(1).median().clamp_min(1e-6))
    out = args.out or str(Path(args.run) / "relight.png")
    debug.save_rgb(out, debug.montage([debug.display(i, None, e) for i in imgs], cols=len(imgs), labels=labels))
    print(out)


if __name__ == "__main__":
    main()
