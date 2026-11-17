"""Networks, the SRM renderer (Sec. 3.2), and losses, following supplement Sec. H.1."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19

SRM_W, SRM_H = 500, 250  # panorama size given in Sec. 3.2


def conv(k, c_in, c_out, s, norm=True, act="relu"):
    """C / CN / CNR / CR(k, c_in, c_out, s). s < 1 means nearest upsample then convolve."""
    layers = []
    if s < 1:
        layers.append(nn.Upsample(scale_factor=round(1 / s), mode="nearest"))
        s = 1
    layers.append(nn.Conv2d(c_in, c_out, k, stride=s, padding=k // 2))
    if norm:
        layers.append(nn.InstanceNorm2d(c_out, affine=True))
    if act == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif act == "lrelu":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(conv(3, ch, ch, 1), conv(3, ch, ch, 1, act=None))

    def forward(self, x):
        return x + self.body(x)


class EncoderDecoder(nn.Module):
    """Johnson et al. generator; used for diffuse refinement and the neural renderer."""

    def __init__(self, c_in):
        super().__init__()
        self.net = nn.Sequential(
            conv(9, c_in, 32, 1),
            conv(3, 32, 64, 2),
            conv(3, 64, 128, 2),
            *[ResBlock(128) for _ in range(5)],
            conv(3, 128, 64, 0.5),
            conv(3, 64, 32, 0.5),
            conv(3, 32, 3, 1, norm=False, act=None),
        )

    def forward(self, x):
        return self.net(x)


class MaterialNet(nn.Module):
    def __init__(self, materials):
        super().__init__()
        self.net = nn.Sequential(
            conv(5, 3, 64, 2),
            conv(3, 64, 64, 2),
            ResBlock(64),
            ResBlock(64),
            conv(3, 64, 32, 0.5),
            conv(3, 32, materials, 0.5, norm=False, act=None),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv(4, 3, 64, 2, norm=False, act="lrelu"),
            conv(4, 64, 128, 2, act="lrelu"),
            conv(4, 128, 256, 2, act="lrelu"),
            conv(4, 256, 512, 2, act="lrelu"),
            conv(1, 512, 1, 1, norm=False, act=None),
        )

    def forward(self, x):
        return self.net(x)


class SRMModel(nn.Module):
    """Refined diffuse D' plus specular S from M basis SRMs (Eqs. 4 and 7)."""

    def __init__(self, materials=2, refine=True):
        super().__init__()
        self.srm = nn.Parameter(0.05 * torch.rand(materials, 3, SRM_H, SRM_W))
        self.material = MaterialNet(materials)
        self.refine = EncoderDecoder(3) if refine else None
        self.cans = "all"

    def lookup(self, refl):
        """Sample every SRM along reflection directions; returns B, M, 3, H, W."""
        x, y, z = refl[:, 0], refl[:, 1], refl[:, 2]
        u = (torch.atan2(y, x) / (2 * math.pi) + 0.5) * SRM_W
        v = torch.acos(z.clamp(-1, 1)) / math.pi * SRM_H
        # One column of wrap-around padding keeps the panorama seam continuous.
        grid = torch.stack([2 * (u + 1) / (SRM_W + 2) - 1, 2 * v / SRM_H - 1], dim=-1)
        pad = torch.cat([self.srm[..., -1:], self.srm, self.srm[..., :1]], dim=-1).float()
        n, m = len(refl), len(self.srm)
        sampled = F.grid_sample(
            pad[None].expand(n, -1, -1, -1, -1).reshape(n * m, 3, SRM_H, SRM_W + 2),
            grid.float().repeat_interleave(m, dim=0),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.reshape(n, m, 3, *refl.shape[-2:])

    def forward(self, batch, diffuse_linear, diffuse_display):
        weights = self.material(diffuse_display)
        residual = self.refine(diffuse_display) if self.refine is not None else 0
        diffuse = (diffuse_linear + residual).clamp(min=0) * batch["mask"]
        specular = (weights[:, :, None] * self.lookup(batch["refl"])).sum(dim=1)
        specular = specular * batch["vis"] * batch["mask"]
        if "spec_mask" in batch:
            specular = specular * batch["spec_mask"]
        return diffuse, specular, weights

    def clamp_srm(self):
        self.srm.data.clamp_(min=0)


class Perceptual(nn.Module):
    """VGG-19 feature loss on the layers of Chen and Koltun [9].

    Their weights balance layer magnitudes for a Caffe VGG on 0-255 inputs. On
    torchvision's VGG they let conv5_2 supply 95% of the loss, so the weights here
    are 1 / (5 * mean layer error) measured on this scene, one fifth per layer.
    """

    LAYERS = {2: 0.78, 7: 0.32, 12: 0.21, 21: 0.31, 30: 0.13}

    def __init__(self):
        super().__init__()
        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:31]
        self.vgg = features.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _features(self, x):
        x = (x - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.LAYERS:
                out.append(x)
        return out

    def forward(self, pred, target):
        with torch.no_grad():
            target_features = self._features(target)
        loss = 0
        for weight, a, b in zip(self.LAYERS.values(), self._features(pred), target_features):
            loss = loss + weight * F.l1_loss(a, b)
        return loss


def masked_l1(a, b, mask):
    return ((a - b).abs() * mask).sum() / (mask.sum() * a.shape[1] + 1)


def gan_d_loss(disc, real, fake):
    return ((disc(real) - 1) ** 2).mean() + (disc(fake.detach()) ** 2).mean()


def gan_g_loss(disc, fake):
    return ((disc(fake) - 1) ** 2).mean()
