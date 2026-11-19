"""Restoration models, normalisation, the training loop and checkpoint resume."""
import json
import subprocess
import sys

import pytest
import torch
import yaml

from sodacan.restore.common import from_log, sample_scale, to_log
from sodacan.restore.infer import enhance, load_model
from sodacan.restore.models import build_model, count_parameters
from sodacan.sim.dataset import band


@pytest.mark.parametrize("name", ["unet", "rtlvit", "vit5"])
def test_models_shapes_size_and_identity_start(name):
    model = build_model(name)
    assert 0.5e6 < count_parameters(model) < 2.5e6
    x = torch.rand(2, 3, 70, 256) * 3
    w = torch.rand(2, 70, 256)
    v = w > 0.2
    y = model(x, w, v)
    assert y.shape == x.shape
    # zero-initialised heads: an untrained model returns its input on observed texels
    assert torch.allclose(y[v[:, None].expand_as(y)], x[v[:, None].expand_as(x)], rtol=1e-4, atol=1e-5)


def test_models_wrap_in_azimuth():
    model = build_model("unet").eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    x = torch.rand(1, 3, 64, 128)
    w = torch.ones(1, 64, 128)
    v = w > 0
    y1 = model(x, w, v)
    # circular padding: rotating the panorama by a multiple of the total stride rotates the output
    y2 = model(torch.roll(x, 48, -1), w, v)
    assert torch.allclose(torch.roll(y1, 48, -1), y2, atol=1e-3)


def test_log_normalisation_round_trip():
    x = torch.rand(2, 3, 8, 16) * 10
    v = torch.ones(2, 8, 16, dtype=torch.bool)
    s = sample_scale(x, v)
    assert torch.allclose(from_log(to_log(x, s), s), x, rtol=1e-4, atol=1e-5)


def test_band_covers_observed_rows():
    v = torch.zeros(256, 512, dtype=torch.bool)
    v[100:150, 10:20] = True
    r0, r1 = band(v)
    assert r0 <= 100 and r1 >= 150 and r1 - r0 >= 64


def test_train_resume_and_infer(tmp_path):
    cfg = {
        "model": {"name": "unet", "base": 8, "depth": 3}, "sim": "sim", "pano": [64, 128],
        "data": {"on_the_fly": True, "pool": 4, "fresh_per_step": 0}, "crop": [32, 64], "batch": 2, "steps": 4,
        "lr": 1e-3, "warmup": 1, "amp": False, "loss": {"fidelity": 0.05}, "val": {"pairs": 2, "every": 2},
        "ckpt_minutes": 0, "seed": 0,
    }
    sim = yaml.safe_load(open("configs/sim.yaml"))
    sim.update(frames=6, image=[160, 120], focal=[250, 300], hdri_dir="nonexistent")
    (tmp_path / "sim_small.yaml").write_text(yaml.safe_dump(sim))
    cfg["sim"] = str(tmp_path / "sim_small.yaml")
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(cfg))
    run = [sys.executable, "-m", "sodacan.restore.train", "--config", str(tmp_path / "train.yaml"), "--out", str(tmp_path / "run")]
    subprocess.run(run, check=True, capture_output=True)
    ck = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert ck["step"] == 4
    subprocess.run(run + ["--resume", "--steps", "6"], check=True, capture_output=True)
    ck = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert ck["step"] == 6
    hist = [json.loads(l) for l in open(tmp_path / "run" / "history.jsonl")]
    assert any("val" in h for h in hist)
    model = load_model(tmp_path / "run" / "best.pt")
    pano = torch.rand(3, 64, 128)
    valid = torch.zeros(64, 128, dtype=torch.bool)
    valid[20:40] = True
    out = enhance(model, pano * valid, valid.float(), valid)
    assert out.shape == pano.shape and (out[:, ~valid] == 0).all()


def test_srgb_gradient_is_finite_at_zero():
    """x ** (1/2.4) has an infinite slope at 0; where() must not turn it into a NaN gradient."""
    from sodacan.color import linear_to_srgb

    x = torch.tensor([0.0, 1e-5, 0.003, 0.5, 1.0], requires_grad=True)
    linear_to_srgb(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_losses_give_finite_gradients_with_unobserved_regions():
    """The full loss (L1, fidelity, LPIPS) must train a model whose crops are partly unobserved.
    A NaN gradient makes the AMP scaler skip every step, and the network then never moves."""
    from sodacan.restore.losses import restoration_loss

    torch.manual_seed(0)
    model = build_model("unet", base=8, depth=2)
    B, H, W = 2, 32, 64
    valid = torch.zeros(B, H, W, dtype=torch.bool)
    valid[:, 8:24] = True
    batch = {"input": torch.rand(B, 3, H, W) * valid[:, None], "weight": valid.float(), "valid": valid,
             "target": torch.rand(B, 3, H, W)}
    loss, terms = restoration_loss(model, batch, {"fidelity": 0.05, "lpips": 0.1})
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)
