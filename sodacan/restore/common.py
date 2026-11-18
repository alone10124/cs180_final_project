"""Input/output normalisation shared by every restoration model.

Panoramas are HDR with an unknown gain. Each sample is divided by the median of its
observed luminance and mapped with log1p, so the network sees the same range for any
exposure. Inputs: 3 log-colour channels, the coverage mask and the normalised log
weight (how much evidence each texel has). The network predicts a residual in the log
domain; the result is mapped back with the same scale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 5


def sample_scale(pano, valid):
    """(B, 3, H, W), (B, H, W) -> (B, 1, 1, 1) median observed luminance."""
    lum = pano.float().mean(1)
    out = []
    for b in range(pano.shape[0]):
        v = lum[b][valid[b]]
        v = v[v > 0]
        out.append(v.median() if v.numel() else torch.tensor(1.0, device=pano.device))
    return torch.stack(out).clamp_min(1e-6).view(-1, 1, 1, 1)


def encode(pano, weight, valid, scale):
    v = valid[:, None].float()
    x = torch.log1p(pano.float().clamp_min(0) / scale) * v
    w = weight[:, None].float()
    wmed = torch.stack([w[b][valid[b][None]].median() if valid[b].any() else torch.tensor(1.0, device=w.device)
                        for b in range(w.shape[0])]).clamp_min(1e-9).view(-1, 1, 1, 1)
    lw = torch.log1p(w / wmed) * v
    return torch.cat([x, v, lw], 1)


def to_log(pano, scale):
    return torch.log1p(pano.float().clamp_min(0) / scale)


def from_log(y, scale):
    return torch.expm1(y.clamp(min=0, max=12)) * scale


class PanoConv(nn.Conv2d):
    """Conv2d with circular padding in azimuth (columns) and replicate padding in rows."""

    def __init__(self, cin, cout, k=3, stride=1, groups=1, bias=True):
        super().__init__(cin, cout, k, stride, 0, groups=groups, bias=bias)
        self.p = k // 2

    def forward(self, x):
        if self.p:
            x = F.pad(x, (self.p, self.p, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, self.p, self.p), mode="replicate")
        return super().forward(x)


class Restorer(nn.Module):
    """Wraps a backbone (5 -> 3 channels, residual in the log domain) with normalisation."""

    multiple = 16

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.multiple = getattr(backbone, "multiple", 16)

    def forward_log(self, pano, weight, valid):
        """Returns (prediction in log domain, input in log domain, scale)."""
        scale = sample_scale(pano, valid)
        x = encode(pano, weight, valid, scale)
        H, W = x.shape[-2:]
        m = self.multiple
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, 0, ph), mode="replicate")
        y = x[:, :3] + self.backbone(x)
        return y[..., :H, :W], x[:, :3, :H, :W], scale

    def forward(self, pano, weight, valid):
        y, _, scale = self.forward_log(pano, weight, valid)
        return from_log(y, scale)
