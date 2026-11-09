"""Per-frame differentiable objectives on a common physical sampling domain."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class Metric:
    """Select a loss without changing sampling, masks or optimizer state."""

    name: str
    bins: int = 32
    sigma: float = 0.5

    def __call__(self, moving, fixed, valid):
        if self.name == "ncc":
            return compute_global_ncc(moving, fixed, valid)
        return _mutual_information(moving, fixed, valid, self.bins, self.sigma)


def validate_metric(metric: str | None, bins: int, sigma: float) -> None:
    """Validate shared metric options before any stage starts fitting.

    Parameters
    ----------
    metric : str, optional
        Explicit global NCC, local NCC or mutual information selection.
    bins : int
        Number of histogram bins for MI.
    sigma : float
        MI Gaussian standard deviation in bin widths.

    Raises
    ------
    ValueError
        If the metric or MI parameters are unsupported.
    """
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


def compute_global_ncc(
    moved: torch.Tensor, fixed: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute independent, valid-sample negative squared global correlations.

    Parameters
    ----------
    moved : (B, ...) torch.Tensor
        Sampled moving intensities.
    fixed : (...) torch.Tensor
        Fixed image intensities.
    valid : (B, ...) torch.Tensor
        Valid moving sample centers.

    Returns
    -------
    loss : (B,) torch.Tensor
        Negative squared correlations, with guards for invalid metric arithmetic.
    meaningful : (B,) torch.Tensor
        Whether support and both centered variances are nonzero.
    """
    axes = tuple(range(1, moved.ndim))
    # Fitting samples use float32; explicit double inputs retain double metric math.
    dtype = torch.float64 if moved.dtype == torch.float64 else torch.float32
    moved, fixed = moved.to(dtype), fixed.to(dtype)
    padding = (-moved[0].numel()) % 32
    if dtype == torch.float32 and padding:
        # Align frame rows so CUDA reductions do not change with row-address alignment.
        fixed = torch.broadcast_to(fixed, moved.shape)
        moved = F.pad(moved.flatten(1), (0, padding))
        fixed = F.pad(fixed.flatten(1), (0, padding))
        valid = F.pad(valid.flatten(1), (0, padding), value=False)
        axes = (1,)
    weights = valid.to(moved.dtype)
    count = weights.sum(dim=axes, keepdim=True).clamp_min(1)
    mean_x = (moved * weights).sum(dim=axes, keepdim=True) / count
    mean_y = (fixed * weights).sum(dim=axes, keepdim=True) / count
    x = (moved - mean_x) * weights
    y = (fixed - mean_y) * weights
    vx = x.square().sum(dim=axes)
    vy = y.square().sum(dim=axes)
    denominator = vx * vy
    tolerance_x = tolerance_y = 0.0
    if dtype == torch.float32:
        epsilon = 16 * torch.finfo(dtype).eps ** 2
        tolerance_x = (
            epsilon * count.reshape(len(moved)) * mean_x.reshape(len(moved)).square()
        )
        tolerance_y = (
            epsilon * count.reshape(len(moved)) * mean_y.reshape(len(moved)).square()
        )
    meaningful = (
        (count.reshape(len(moved)) > 1)
        & (vx > tolerance_x)
        & (vy > tolerance_y)
        & torch.isfinite(denominator)
        & (denominator > torch.finfo(moved.dtype).tiny)
    )
    safe_denominator = torch.where(
        meaningful, denominator, torch.ones_like(denominator)
    )
    loss = -(x * y).sum(dim=axes).square() / safe_denominator
    return torch.where(meaningful, loss, torch.zeros_like(loss)), meaningful


def _sum_neighborhoods(values, radius):
    """Sum neighborhoods separably in metric precision."""
    dimensions = values.ndim - 2
    for axis in range(dimensions):
        padding = [0] * (2 * dimensions)
        padding[2 * (dimensions - axis - 1) : 2 * (dimensions - axis)] = [
            radius + 1,
            radius,
        ]
        width = 2 * radius + 1
        if values.dtype == torch.float32:
            # Local accumulation avoids cancellation of long float32 prefix sums.
            padding[2 * (dimensions - axis - 1)] = radius
            kernel = [1] * dimensions
            kernel[axis] = width
            pool = F.avg_pool2d if dimensions == 2 else F.avg_pool3d
            values = pool(F.pad(values, padding), kernel, stride=1, divisor_override=1)
            continue
        cumulative = F.pad(values, padding).cumsum(dim=axis + 2)
        values = cumulative.narrow(
            axis + 2, width, values.shape[axis + 2]
        ) - cumulative.narrow(axis + 2, 0, values.shape[axis + 2])
    return values


def compute_local_ncc(moving, fixed, valid, radius):
    """Compute masked squared neighborhood correlation independently per frame.

    Parameters
    ----------
    moving : torch.Tensor
        Sampled moving images, shaped (batch, *spatial_shape).
    fixed : torch.Tensor
        Sampled fixed images with the same shape.
    valid : torch.Tensor
        Boolean common-support mask.
    radius : int
        Neighborhood radius in active voxel axes.

    Returns
    -------
    loss : torch.Tensor
        Per-frame negative mean squared correlation; lower is better.
    meaningful : torch.Tensor
        Whether each frame has sufficient support and nonconstant neighborhoods.
    """
    dtype = torch.float64 if moving.dtype == torch.float64 else torch.float32
    x, y = moving.to(dtype), fixed.to(dtype)
    w = valid.to(dtype)
    stats = _sum_neighborhoods(
        torch.stack((w, w * x, w * y, w * x * x, w * y * y, w * x * y), dim=1), radius
    )
    n, sx, sy, sxx, syy, sxy = stats.unbind(dim=1)
    count = n.clamp_min(1)
    vx = (sxx - sx * sx / count).clamp_min(0)
    vy = (syy - sy * sy / count).clamp_min(0)
    tolerance_x = tolerance_y = 1e-12
    if dtype == torch.float32:
        # Raw-moment cancellation must not make constant neighborhoods meaningful.
        epsilon = 16 * torch.finfo(dtype).eps
        tolerance_x = epsilon * (sxx + sx * sx / count).abs()
        tolerance_y = epsilon * (syy + sy * sy / count).abs()
    usable = valid & (n > 1) & (vx > tolerance_x) & (vy > tolerance_y)
    cc = ((sxy - sx * sy / count).square() / (vx * vy).clamp_min(1e-24)).clamp(max=1)
    axes = tuple(range(1, x.ndim))
    counts = usable.sum(dim=axes)
    loss = -torch.where(usable, cc, 0).sum(dim=axes) / counts.clamp_min(1)
    return loss, counts > 1
