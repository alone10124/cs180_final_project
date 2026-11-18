"""Restoration backbones: a small U-Net baseline, an RTLViT-style transformer, and ViT-5-style blocks."""
from sodacan.restore.common import Restorer


def build_model(name, **kwargs):
    if name == "unet":
        from sodacan.restore.models.unet import UNet

        backbone = UNet(**kwargs)
    elif name == "rtlvit":
        from sodacan.restore.models.rtlvit_style import RTLViTStyle

        backbone = RTLViTStyle(**kwargs)
    elif name == "vit5":
        from sodacan.restore.models.vit5_blocks import ViT5Style

        backbone = ViT5Style(**kwargs)
    else:
        raise ValueError(f"unknown model {name!r}")
    return Restorer(backbone)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
