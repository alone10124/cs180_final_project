"""Linear light <-> display encoding. All image maths happens in linear light."""
import numpy as np
import torch


def srgb_to_linear(x):
    x = x.clamp(0, 1) if torch.is_tensor(x) else np.clip(x, 0, 1)
    lib = torch if torch.is_tensor(x) else np
    return lib.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1) if torch.is_tensor(x) else np.clip(x, 0, 1)
    lib = torch if torch.is_tensor(x) else np
    return lib.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def linearize(x, gamma="srgb"):
    """Display-encoded [0, 1] -> linear, by the sRGB curve or a power law."""
    if gamma == "srgb":
        return srgb_to_linear(x)
    x = x.clamp(0, 1) if torch.is_tensor(x) else np.clip(x, 0, 1)
    return x ** float(gamma)


def encode(x, gamma="srgb"):
    if gamma == "srgb":
        return linear_to_srgb(x)
    x = x.clamp(0, 1) if torch.is_tensor(x) else np.clip(x, 0, 1)
    return x ** (1 / float(gamma))


def to_uint8(linear, exposure=1.0):
    """(3, H, W) or (H, W, 3) linear tensor/array -> (H, W, 3) uint8 RGB for saving."""
    x = linear.detach().float().cpu().numpy() if torch.is_tensor(linear) else np.asarray(linear, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.moveaxis(x, 0, -1)
    return (linear_to_srgb(x * exposure) * 255 + 0.5).astype(np.uint8)


def auto_exposure(linear, mask=None, key=0.18):
    """Exposure that maps the median of the (masked) image to `key`."""
    x = linear if torch.is_tensor(linear) else torch.as_tensor(linear)
    lum = x.mean(0) if x.dim() == 3 and x.shape[0] == 3 else x
    vals = lum[mask] if mask is not None else lum.reshape(-1)
    vals = vals[vals > 0]
    if vals.numel() == 0:
        return 1.0
    return float(key / vals.median().clamp_min(1e-6))
