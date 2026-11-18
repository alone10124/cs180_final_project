"""Apply a trained restoration model to a measured panorama."""
import torch

from sodacan.restore.models import build_model
from sodacan.sim.dataset import band


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    mcfg = dict(ck["config"]["model"])
    model = build_model(mcfg.pop("name"), **mcfg).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def enhance(model, pano, weight, valid, fill=False):
    """Enhanced panorama (3, H, W). Runs on the observed band only. With fill=False
    (the default) unobserved texels stay 0: the network output is only shown where the
    can measured something."""
    dev = next(model.parameters()).device
    r0, r1 = band(valid)
    x = pano[:, r0:r1][None].to(dev).float()
    w = weight[r0:r1][None].to(dev).float()
    v = valid[r0:r1][None].to(dev)
    y = model(x, w, v)[0].float()
    out = torch.zeros_like(pano, dtype=torch.float32, device=dev)
    out[:, r0:r1] = y if fill else y * v
    return out.to(pano.device)
