"""Label estimate: a per-texel temporal statistic of the surface-space frames.

The label is fixed to the can, the reflection slides across it as the camera moves, so
a low percentile (or the median) over frames approximates the label. Whatever part of
the reflection never leaves a texel stays in the estimate; that is the method's bias.
"""
import torch


@torch.no_grad()
def estimate_label(obs, valid, method="percentile", percentile=0.2, min_count=3, rows_per_chunk=16):
    """obs (B, 3, Ha, Wa), valid (B, Ha, Wa) -> label (3, Ha, Wa), count (Ha, Wa).

    method: median | min | percentile | none. Texels seen fewer than min_count times
    get label 0 (they are dropped later through count).
    """
    B, C, Ha, Wa = obs.shape
    count = valid.sum(0)
    if method == "none":
        return torch.zeros(C, Ha, Wa, device=obs.device), count
    q = {"median": 0.5, "min": 0.0}.get(method, percentile)
    label = torch.zeros(C, Ha, Wa, device=obs.device)
    for r in range(0, Ha, rows_per_chunk):
        o = obs[:, :, r : r + rows_per_chunk].clone()
        v = valid[:, None, r : r + rows_per_chunk].expand_as(o)
        o[~v] = float("nan")
        est = torch.nanquantile(o.float(), q, dim=0)
        label[:, r : r + rows_per_chunk] = torch.nan_to_num(est, nan=0.0)
    label[:, count < min_count] = 0
    return label, count


def residual(values, label_values):
    """Reflection = frame - label, in linear light."""
    return values - label_values
