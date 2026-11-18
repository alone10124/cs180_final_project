"""Training losses, all on observed texels only.

  reconstruction  L1 between prediction and target in the normalised log domain
  fidelity        L1 between the prediction and the measured input: keeps the network
                  from inventing objects the can never reflected (raise if it does)
  perceptual      optional LPIPS on display-mapped crops
"""
import torch

from sodacan.color import linear_to_srgb
from sodacan.restore.common import from_log, to_log

_LPIPS = {}


def masked_l1(a, b, mask):
    m = mask[:, None].float()
    return ((a - b).abs() * m).sum() / (m.sum() * a.shape[1]).clamp_min(1)


def restoration_loss(model, batch, weights):
    pred_log, in_log, scale = model.forward_log(batch["input"], batch["weight"], batch["valid"])
    tgt_log = to_log(batch["target"], scale)
    valid = batch["valid"]
    terms = {"rec": masked_l1(pred_log, tgt_log, valid)}
    if weights.get("fidelity", 0) > 0:
        terms["fid"] = masked_l1(pred_log, in_log, valid)
    if weights.get("lpips", 0) > 0:
        terms["lpips"] = perceptual(from_log(pred_log, scale), batch["target"].float(), valid, scale)
    names = {"fid": "fidelity", "lpips": "lpips"}
    total = terms["rec"] + sum(float(weights[names[k]]) * v for k, v in terms.items() if k != "rec")
    return total, {k: float(v.detach()) for k, v in terms.items()}


def perceptual(pred, target, valid, scale):
    import lpips

    dev = pred.device
    if dev not in _LPIPS:
        _LPIPS[dev] = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        for p in _LPIPS[dev].parameters():
            p.requires_grad_(False)
    e = 0.18 / scale
    m = valid[:, None].float()
    a = linear_to_srgb((pred * e).clamp(0, 1)) * m
    b = linear_to_srgb((target * e).clamp(0, 1)) * m
    return _LPIPS[dev](a * 2 - 1, b * 2 - 1).mean()
