"""Hand-checked cases for every coordinate convention in geometry/README.md."""
import math

import pytest
import torch

from sodacan.accumulate.splat import splat
from sodacan.config import load_can
from sodacan.geometry import cylinder, equirect, fresnel
from sodacan.geometry.camera import Intrinsics, look_at, project, world_rays
from sodacan.geometry.reflect import can_geometry, reflect
from sodacan.sim.fast_render import RenderParams, render

R = 0.033


def close(a, b, tol=1e-5):
    return torch.allclose(torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32), atol=tol)


# --- camera: OpenCV axes, camera-to-world poses -------------------------------------

def test_look_at_axes():
    # camera at +x looking at the origin, z up: forward -x, right +y, down -z
    c2w = look_at(torch.tensor([[1.0, 0, 0]]), torch.zeros(3))
    assert close(c2w[0, :3, 0], [0, 1, 0]) and close(c2w[0, :3, 1], [0, 0, -1]) and close(c2w[0, :3, 2], [-1, 0, 0])


def test_projection_right_and_down():
    K = Intrinsics(101, 101, 100, 100, 50, 50)
    c2w = look_at(torch.tensor([[1.0, 0, 0]]), torch.zeros(3))
    pts = torch.tensor([[[0, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]], dtype=torch.float32)
    uv, z = project(pts, c2w, K)
    assert close(uv[0, 0], [50, 50]) and close(z[0], [1, 1, 1])
    assert close(uv[0, 1], [60, 50])  # +y is to the right
    assert close(uv[0, 2], [50, 40])  # +z (up) is toward the top row


def test_pixel_centre_ray():
    K = Intrinsics(101, 81, 100, 100, 50, 40)
    c2w = look_at(torch.tensor([[1.0, 0, 0]]), torch.zeros(3))
    o, d = world_rays(c2w, K)
    assert close(d[0, 40, 50], [-1, 0, 0]) and close(o[0, 0, 0], [1, 0, 0])


# --- cylinder and reflection --------------------------------------------------------

def test_head_on_hit_reflects_back():
    o = torch.tensor([1.0, 0, 0.05])
    d = torch.tensor([-1.0, 0, 0])
    t, hit = cylinder.intersect(o, d, R, 0, 0.122)
    assert hit and close(t, 1 - R)
    p = o + t * d
    n = cylinder.normal(p)
    assert close(n, [1, 0, 0]) and close(reflect(d, n), [1, 0, 0])


def test_oblique_hit_doubles_the_angle():
    # a ray parallel to -x hitting the wall at s = 30 deg reflects to azimuth 60 deg
    s = math.radians(30)
    o = torch.tensor([1.0, R * math.sin(s), 0.05])
    d = torch.tensor([-1.0, 0, 0])
    t, hit = cylinder.intersect(o, d, R, 0, 0.122)
    n = cylinder.normal(o + t * d)
    r = reflect(d, n)
    assert hit and close(n, [math.cos(s), math.sin(s), 0], 1e-4)
    assert close(r, [math.cos(2 * s), math.sin(2 * s), 0], 1e-4)


def test_reflection_keeps_vertical_component():
    o = torch.tensor([0.3, 0.0, 0.0])
    d = torch.tensor([-1.0, 0.0, 0.2])
    d = d / d.norm()
    t, hit = cylinder.intersect(o, d, R, 0, 0.122)
    r = reflect(d, cylinder.normal(o + t * d))
    assert hit and close(r[2], d[2])  # a vertical cylinder does not widen the vertical view


def test_misses_and_height_limits():
    _, hit = cylinder.intersect(torch.tensor([1.0, 0.1, 0.05]), torch.tensor([-1.0, 0, 0]), R, 0, 0.122)
    assert not hit
    _, hit = cylinder.intersect(torch.tensor([1.0, 0, 0.2]), torch.tensor([-1.0, 0, 0]), R, 0, 0.122)
    assert not hit


def test_surface_atlas_not_mirrored():
    # from a camera on +x, increasing s (toward +y) appears to the right
    K = Intrinsics(101, 101, 300, 300, 50, 50)
    c2w = look_at(torch.tensor([[0.3, 0, 0.06]]), torch.tensor([0, 0, 0.06]))
    p = cylinder.surface_point(torch.tensor([0.0, 0.2]), torch.tensor([0.06, 0.06]), R)
    uv, _ = project(p[None], c2w, K)
    assert uv[0, 1, 0] > uv[0, 0, 0]
    # atlas column increases with s, row 0 is the rim
    s, h = cylinder.atlas_coords(4, 8, 0.12)
    assert s[0, 1] > s[0, 0] and h[0, 0] > h[1, 0]


def test_atlas_sample_hits_texel_centres_and_wraps():
    atlas = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    s, h = cylinder.atlas_coords(4, 8, 0.12)
    assert close(cylinder.atlas_sample(atlas, s, h, 0.12)[..., 0], atlas[0])
    # halfway across the seam averages the first and last column
    v = cylinder.atlas_sample(atlas, torch.tensor([math.pi - 1e-6]), h[:1, 0], 0.12)
    assert close(v[0, 0], (atlas[0, 0, 0] + atlas[0, 0, 7]) / 2, 1e-3)


# --- panoramas ----------------------------------------------------------------------

def test_equirect_landmarks():
    H, W = 64, 128
    u, v = equirect.dir_to_uv(torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0]]), H, W)
    assert close(u[:2], [W / 2, 3 * W / 4]) and close(v[[0, 1, 2]], [H / 2, H / 2, 0])
    assert close(torch.remainder(u[3], W), 0, 1e-4)  # -x sits on the seam


def test_pano_dirs_round_trip():
    H, W = 16, 32
    d = equirect.pano_dirs(H, W)
    u, v = equirect.dir_to_uv(d, H, W)
    assert close(u, torch.arange(W)[None].expand(H, W) + 0.5, 1e-4)
    assert close(v, torch.arange(H)[:, None].expand(H, W) + 0.5, 1e-4)


def test_sample_and_splat_are_consistent():
    H, W = 16, 32
    pano = torch.rand(3, H, W)
    d = equirect.pano_dirs(H, W)
    assert close(equirect.sample(pano, d).permute(2, 0, 1), pano, 1e-4)
    p = splat(pano.permute(1, 2, 0).reshape(-1, 3), d.reshape(-1, 3), torch.ones(H * W), H, W)
    assert close(p.pano, pano, 1e-4) and p.valid.all()


def test_splat_wraps_across_the_seam():
    H, W = 8, 16
    # u = W is halfway between the last and the first column centres
    d = equirect.uv_to_dir(torch.tensor([W * 1.0]), torch.tensor([4.5]), H, W)
    p = splat(torch.ones(1, 1), d, torch.ones(1), H, W, min_weight=0.1)
    assert close(p.weight[4, 0], 0.5) and close(p.weight[4, W - 1], 0.5)


def test_solid_angle_sums_to_sphere():
    assert close(equirect.solid_angle(32, 64).sum(), 4 * math.pi, 1e-3)


# --- Fresnel --------------------------------------------------------------------------

def test_fresnel_limits():
    cos = torch.tensor([1.0, 1e-4])
    f = fresnel.dielectric(cos, 1.5)
    assert close(f[0], 0.04, 1e-4) and f[1] > 0.99
    assert close(fresnel.schlick(torch.tensor([1.0]), 0.9), [0.9])


# --- renderer: the centre pixel of a head-on view sees the camera's own direction ----

@pytest.mark.parametrize("az_deg", [0.0, 90.0])
def test_render_head_on_sees_behind_camera(az_deg):
    H, W = 64, 128
    can = load_can(material={"model": "mirror"})
    env = torch.full((3, H, W), 0.1)
    d = equirect.pano_dirs(H, W)
    az = math.radians(az_deg)
    bright = (d[..., 0] * math.cos(az) + d[..., 1] * math.sin(az)) > math.cos(math.radians(10))
    env[:, bright] = 5.0
    K = Intrinsics(41, 41, 400, 400, 20, 20)
    eye = torch.tensor([[0.3 * math.cos(az), 0.3 * math.sin(az), 0.06]])
    c2w = look_at(eye, torch.tensor([0, 0, 0.06]))
    img = render(env, c2w, K, can, RenderParams())
    assert img[0, 0, 20, 20] > 4.0
    g = can_geometry(c2w, K, can)
    assert g.mask[0, 20, 20] and close(g.reflected[0, 20, 20], [math.cos(az), math.sin(az), 0], 1e-4)
