"""Final-image cubic B-spline interpolation on CPU and CUDA."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import torch

Interpolation: TypeAlias = Literal["linear", "nearest", "bspline"]
"""Supported final-image interpolation methods; fitting remains linear."""


def _round_integer_samples(values: npt.NDArray, method: Interpolation) -> npt.NDArray:
    """Round final integer-image samples, stabilizing cubic half-integer ties.

    Parameters
    ----------
    values : numpy.ndarray
        Final float64 intensities, after support fills and before dtype clipping.
    method : str
        Selected final-image interpolator.

    Returns
    -------
    numpy.ndarray
        Nearest-even rounded values. Cubic samples within
        32 * float64 epsilon * max(1, abs(value)) of a half-integer are snapped
        to that half-integer first, so coefficient-solve roundoff does not
        arbitrarily change exact ties. Floating-image outputs are unaffected.
    """
    if method == "bspline":
        half = np.rint(values * 2) / 2
        tolerance = 32 * np.finfo(np.float64).eps * np.maximum(1, np.abs(values))
        values = np.where(np.abs(values - half) <= tolerance, half, values)
    return np.rint(values)


def restore_image_dtype(
    values: npt.NDArray, dtype: np.dtype, method: Interpolation
) -> npt.NDArray:
    """Restore original image dtype after interpolation and source-support fills.

    Parameters
    ----------
    values : numpy.ndarray
        Final floating samples on CPU, retaining double precision for cubic.
    dtype : numpy.dtype
        Original moving-image intensity type.
    method : str
        Selected final-image interpolator.

    Returns
    -------
    numpy.ndarray
        Original-dtype intensities. Integers use nearest-even rounding and dtype
        clipping; floating cubic values retain representable overshoot.

    Raises
    ------
    ValueError
        If cubic values cannot be represented finitely in the original floating dtype.
    """
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        values = np.clip(_round_integer_samples(values, method), limits.min, limits.max)
    elif method == "bspline" and np.issubdtype(dtype, np.floating):
        if (
            not np.isfinite(values).all()
            or (np.abs(values) > np.finfo(dtype).max).any()
        ):
            raise ValueError(
                "Cubic interpolation produced nonfinite values in the source dtype."
            )
    return values.astype(dtype, copy=False)


@lru_cache(maxsize=64)
def _mirror_factors(size: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Factor the cubic cardinal interpolation equations for one mirrored axis.

    Parameters
    ----------
    size : int
        Number of samples on a nonsingleton native axis.

    Returns
    -------
    diagonal : tuple of float
        Modified pivots for the tridiagonal solve.
    upper : tuple of float
        Normalized upper diagonal used in back substitution.
    """
    diagonal, upper = [4.0], [0.5]
    for index in range(1, size):
        lower = 2.0 if index == size - 1 else 1.0
        diagonal.append(4.0 - lower * upper[-1])
        upper.append(0.0 if index == size - 1 else 1.0 / diagonal[-1])
    return tuple(diagonal), tuple(upper)


def _prefilter_axis(values: torch.Tensor, axis: int) -> torch.Tensor:
    """Solve one axis of an owned float64 coefficient workspace in place.

    Parameters
    ----------
    values : torch.Tensor
        Owned, normalized image/coefficient workspace; no original-image alias.
    axis : int
        Native spatial axis to filter, excluding the leading frame axis.

    Returns
    -------
    torch.Tensor
        Coefficients with whole-sample mirror boundaries on the selected axis.
    """
    size = values.shape[axis]
    if size == 1:
        return values
    diagonal, upper = _mirror_factors(size)
    work = values.movedim(axis, -1).contiguous()
    work[..., 0] *= 1.5
    for index in range(1, size):
        lower = 2.0 if index == size - 1 else 1.0
        work[..., index] = (
            6.0 * work[..., index] - lower * work[..., index - 1]
        ) / diagonal[index]
    for index in range(size - 2, -1, -1):
        work[..., index] -= upper[index] * work[..., index + 1]
    return work.movedim(-1, axis)


@dataclass(frozen=True)
class CubicSpline:
    """Native images and normalized float64 cardinal B-spline coefficients.

    Attributes
    ----------
    images : torch.Tensor
        Original source frames, retained for exact evaluation at integer knots.
    coefficients : torch.Tensor
        Tensor-product coefficients with whole-sample mirror extension.
    offset, scale : torch.Tensor
        Per-frame intensity offset and scale used to preserve contrast and avoid
        overflowing intermediate coefficient solves.
    """

    images: torch.Tensor
    coefficients: torch.Tensor
    offset: torch.Tensor
    scale: torch.Tensor


@torch.no_grad()
def prepare_cubic_spline(images: torch.Tensor) -> CubicSpline:
    """Prepare interpolating cubic coefficients once per native source batch.

    The separable tridiagonal solve uses f[i] = (c[i-1] + 4c[i] + c[i+1])/6,
    with c[-1] = c[1] and c[n] = c[n-2]. This is a coefficient prefilter,
    not Gaussian smoothing or an additional spatial resampling. Operations stay
    on the input device, with no CPU fallback for CUDA inputs.

    Parameters
    ----------
    images : torch.Tensor
        Finite real source frames, shaped batch followed by two or three axes.

    Returns
    -------
    CubicSpline
        Float64 normalized coefficients and original intensity reconstruction data.
    """
    values = images.double()
    broadcast = (slice(None),) + (None,) * (images.ndim - 1)
    offset = values.flatten(1)[:, 0]
    centered = values - offset[broadcast]
    overflow = ~torch.isfinite(centered).flatten(1).all(dim=1)
    offset = torch.where(overflow, 0, offset)
    centered = torch.where(overflow[broadcast], values, centered)
    scale = centered.abs().flatten(1).amax(dim=1)
    scale = torch.where(scale > 0, scale, 1)
    coefficients = centered / scale[broadcast]
    for axis in range(1, images.ndim):
        coefficients = _prefilter_axis(coefficients, axis)
    return CubicSpline(images, coefficients.contiguous(), offset, scale)


@torch.no_grad()
def sample_cubic_spline(
    spline: CubicSpline, coordinates: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Evaluate tensor-product cubic B-splines at native array-position queries.

    Parameters
    ----------
    spline : CubicSpline
        Native source batch prepared once for these queries.
    coordinates : torch.Tensor
        Float64 array-position queries with a leading frame axis and native
        coordinate components last (j/i for planes, k/j/i for volumes).
    valid : torch.Tensor
        Source-support mask supplied by geometry. Invalid queries are replaced
        by an integer knot during evaluation; the caller applies its outside fill.

    Returns
    -------
    torch.Tensor
        Float64 samples, preserving singleton axes and batch broadcasting.
        The caller restores the original image dtype once, after outside fills.
        Floating overshoot is retained, without clipping to the input range.

    Raises
    ------
    ValueError
        If cubic overshoot cannot be represented finitely in the source dtype.
    """
    count = max(len(spline.images), len(coordinates))
    shape = coordinates.shape[1:-1]
    sizes = spline.images.shape[1:]
    queries = coordinates.expand(count, *coordinates.shape[1:]).reshape(
        count, -1, len(sizes)
    )
    support = valid.expand(count, *valid.shape[1:]).reshape(count, -1)
    coefficients = spline.coefficients.reshape(len(spline.images), -1).expand(count, -1)
    originals = spline.images.reshape(len(spline.images), -1).expand(count, -1)
    offsets, scales = spline.offset.expand(count), spline.scale.expand(count)
    output = torch.empty(
        (count, queries.shape[1]), dtype=torch.float64, device=queries.device
    )
    chunk_size = max(1, 2**18 // count)
    for start in range(0, queries.shape[1], chunk_size):
        stop = start + chunk_size
        query = torch.where(support[:, start:stop, None], queries[:, start:stop], 0)
        indices, weights = [], []
        knots = torch.ones(query.shape[:2], dtype=torch.bool, device=query.device)
        knot_index = torch.zeros(
            query.shape[:2], dtype=torch.int64, device=query.device
        )
        for axis, size in enumerate(sizes):
            position = (
                query[..., axis].clamp(-0.5, size - 0.5)
                if size > 1
                else torch.zeros_like(query[..., axis])
            )
            base = position.floor()
            fraction = position - base
            knots &= fraction == 0
            knot_index = knot_index * size + base.long().clamp(0, size - 1)
            if size == 1:
                indices.append([torch.zeros_like(base, dtype=torch.int64)])
                weights.append([torch.ones_like(base)])
                continue
            square, cube = fraction.square(), fraction.pow(3)
            weights.append(
                [
                    (1 - fraction).pow(3) / 6,
                    (4 - 6 * square + 3 * cube) / 6,
                    (1 + 3 * fraction + 3 * square - 3 * cube) / 6,
                    cube / 6,
                ]
            )
            neighbors = []
            for step in (-1, 0, 1, 2):
                mirrored = (base.long() + step).remainder(2 * (size - 1))
                neighbors.append(
                    torch.where(mirrored < size, mirrored, 2 * (size - 1) - mirrored)
                )
            indices.append(neighbors)
        sampled = torch.zeros(query.shape[:2], dtype=torch.float64, device=query.device)
        for neighbor in product(*(range(len(values)) for values in indices)):
            index = torch.zeros_like(knot_index)
            weight = torch.ones_like(sampled)
            for axis, choice in enumerate(neighbor):
                index = index * sizes[axis] + indices[axis][choice]
                weight = weight * weights[axis][choice]
            sampled += coefficients.gather(1, index) * weight
        sampled = sampled * scales[:, None] + offsets[:, None]
        sampled = torch.where(knots, originals.gather(1, knot_index).double(), sampled)
        output[:, start:stop] = sampled
    if not torch.isfinite(output).all():
        raise ValueError(
            "Cubic interpolation produced nonfinite values in the source dtype."
        )
    return output.reshape(count, *shape)


def sample_cubic_normalized(
    images: torch.Tensor, normalized: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Sample original images at normalized xyz queries using cubic B-splines.

    Parameters
    ----------
    images : torch.Tensor
        Original source frames with native spatial axes last.
    normalized : torch.Tensor
        Double normalized sampler coordinates, with components in reverse array order.
    valid : torch.Tensor
        Existing geometry support mask, without any cubic-specific erosion.

    Returns
    -------
    torch.Tensor
        Original intensities at the requested points, with unchanged support policy.
    """
    size = normalized.new_tensor(images.shape[1:])
    coordinates = (normalized.flip(-1).double() + 1) * ((size - 1) / 2)
    return sample_cubic_spline(prepare_cubic_spline(images), coordinates, valid)
