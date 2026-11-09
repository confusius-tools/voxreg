"""Per-frame differentiable objectives on a common physical sampling domain."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from voxreg.sampling import compute_global_ncc


@dataclass(frozen=True)
class _Metric:
    """Select a loss without changing sampling, masks or optimizer state."""

    name: str
    bins: int = 32
    sigma: float = 0.5

    def __call__(self, moving, fixed, valid):
        if self.name == "ncc":
            return compute_global_ncc(moving, fixed, valid)
        return _mutual_information(moving, fixed, valid, self.bins, self.sigma)


def _validate_metric(metric, bins, sigma):
    """Validate shared options before any stage starts fitting."""
    if metric not in (None, "ncc", "local_ncc", "mi"):
        raise ValueError("metric must be None, ncc, local_ncc or mi.")
    if isinstance(bins, bool) or not isinstance(bins, Integral) or not 2 <= bins <= 256:
        raise ValueError("mi_bins must be an integer between 2 and 256.")
    if (
        isinstance(sigma, bool)
        or not isinstance(sigma, Real)
        or not np.isfinite(sigma)
        or not 0.1 <= sigma <= 10
    ):
        raise ValueError("mi_sigma must be finite and between 0.1 and 10 bin widths.")


def _mutual_information(moving, fixed, valid, bins=32, sigma=0.5, *, chunk_size=4096):
    """Negative Gaussian-Parzen MI in nats, independently over each valid ROI.

    Inputs use fixed native-image intensity ranges mapped to [0, 1] before
    pyramid smoothing or sampling. Each sample's Gaussian bin weights sum to one.
    Histograms are accumulated in spatial chunks; checkpointing recomputes bin
    weights during backward instead of retaining an image-by-bins activation.
    Both images remain differentiable, including SyN's two halfway pulls.
    """
    dtype = torch.float64 if moving.dtype == torch.float64 else torch.float32
    fixed = torch.broadcast_to(fixed, moving.shape)
    x = torch.where(valid, moving.to(dtype), 0).flatten(1)
    y = torch.where(valid, fixed.to(dtype), 0).flatten(1)
    w = valid.flatten(1).to(dtype)
    centers = torch.arange(bins, dtype=dtype, device=moving.device)
    chunk_size = min(chunk_size, max(1, 2**20 // (len(x) * bins)))

    def histogram(a, b, weights):
        # Subtracting the largest log weight gives stable, normalized kernels.
        a = torch.softmax(
            -0.5 * ((a[..., None] * (bins - 1) - centers) / sigma).square(), dim=-1
        )
        b = torch.softmax(
            -0.5 * ((b[..., None] * (bins - 1) - centers) / sigma).square(), dim=-1
        )
        return (a * weights[..., None]).transpose(1, 2) @ b

    joint = x.new_zeros((len(x), bins, bins))
    for start in range(0, x.shape[1], chunk_size):
        stop = start + chunk_size
        inputs = (x[:, start:stop], y[:, start:stop], w[:, start:stop])
        if torch.is_grad_enabled() and (x.requires_grad or y.requires_grad):
            contribution = checkpoint(
                histogram, *inputs, use_reentrant=False, preserve_rng_state=False
            )
        else:
            contribution = histogram(*inputs)
        joint = joint + contribution
    joint = joint / w.sum(dim=1).clamp_min(1)[:, None, None]
    tiny = torch.finfo(dtype).tiny
    log_ratio = (
        joint.clamp_min(tiny).log()
        - joint.sum(dim=2, keepdim=True).clamp_min(tiny).log()
        - joint.sum(dim=1, keepdim=True).clamp_min(tiny).log()
    )
    loss = -(joint * log_ratio).sum(dim=(1, 2))
    with torch.no_grad():
        _, meaningful = compute_global_ncc(x, y, valid.flatten(1))
    meaningful = meaningful & torch.isfinite(loss)
    return torch.where(meaningful, loss, 0), meaningful
