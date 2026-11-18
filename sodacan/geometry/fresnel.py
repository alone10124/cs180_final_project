"""Fresnel reflectance of the can surface as a function of the incidence cosine."""
import torch


def dielectric(cos_i, ior=1.5):
    """Exact unpolarised Fresnel reflectance of a clear coat (air -> ior)."""
    cos_i = cos_i.clamp(0, 1)
    sin_t2 = (1 - cos_i**2) / ior**2
    cos_t = torch.sqrt((1 - sin_t2).clamp_min(0))
    rs = (cos_i - ior * cos_t) / (cos_i + ior * cos_t)
    rp = (ior * cos_i - cos_t) / (ior * cos_i + cos_t)
    return 0.5 * (rs**2 + rp**2)


def schlick(cos_i, f0):
    return f0 + (1 - f0) * (1 - cos_i.clamp(0, 1)) ** 5


def reflectance(cos_i, material):
    """Fresnel factor for a material dict from configs/can.yaml."""
    model = material.get("model", "mirror")
    if model == "mirror":
        return torch.ones_like(cos_i)
    if model == "coated":
        return dielectric(cos_i, float(material.get("ior", 1.5)))
    if model == "metal":
        return schlick(cos_i, float(material.get("f0", 0.9)))
    raise ValueError(f"unknown material model {model!r}")
