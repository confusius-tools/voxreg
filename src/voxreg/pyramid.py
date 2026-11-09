"""Physical image pyramids and per-level registration diagnostics."""

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import torch
import torch.nn.functional as F

from voxreg.geometry import VOXEL_DIMS, position_to_world


def _validate_schedule(shrink_factors, smoothing_sigmas, max_iterations):
    """Normalize explicit coarse-to-fine schedules without choosing a preset."""
    for name, values in (
        ("shrink_factors", shrink_factors),
        ("smoothing_sigmas", smoothing_sigmas),
    ):
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
        ):
            raise ValueError(f"{name} must be a nonempty sequence.")
    factors, sigmas = tuple(shrink_factors), tuple(smoothing_sigmas)
    if any(type(value) is not int or value < 1 for value in factors):
        raise ValueError("shrink_factors must contain positive integers.")
    if any(a < b for a, b in pairwise(factors)):
        raise ValueError(
            "shrink_factors must be ordered coarse to fine (nonincreasing)."
        )
    if len(sigmas) != len(factors):
        raise ValueError("smoothing_sigmas and shrink_factors must have equal lengths.")
    if any(
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, bool)
        or not np.isfinite(value)
        or value < 0
        for value in sigmas
    ):
        raise ValueError(
            "smoothing_sigmas must contain finite nonnegative values in mm."
        )
    if type(max_iterations) is int:
        budgets = (max_iterations,) * len(factors)
    elif isinstance(max_iterations, Sequence) and not isinstance(
        max_iterations, (str, bytes)
    ):
        budgets = tuple(max_iterations)
    else:
        raise ValueError(
            "max_iterations must be a positive integer or a per-level sequence."
        )
    if len(budgets) != len(factors) or any(
        type(value) is not int or value < 1 for value in budgets
    ):
        raise ValueError("max_iterations must contain one positive integer per level.")
    return factors, tuple(float(value) for value in sigmas), budgets


def _smooth_images(images, data, sigma):
    """Apply a physical Gaussian, including covariance on sheared lattices."""
    if sigma == 0:
        return images
    dimensions = images.ndim - 1
    lattice = position_to_world(data)[:3, 3 - dimensions : 3]
    active = np.array(images.shape[1:]) > 1
    gram = lattice.T @ lattice
    covariance = sigma**2 * np.linalg.inv(gram)
    radii = np.where(active, np.ceil(3 * np.sqrt(np.diag(covariance))).astype(int), 0)
    convolution = F.conv2d if dimensions == 2 else F.conv3d
    values = images[:, None]
    if np.allclose(
        gram, np.diag(np.diag(gram)), rtol=0, atol=1e-12 * np.diag(gram).max()
    ):
        for axis, radius in enumerate(radii):
            if radius == 0:
                continue
            positions = torch.arange(
                -int(radius), int(radius) + 1, dtype=images.dtype, device=images.device
            )
            kernel = torch.exp(-0.5 * positions.square() / covariance[axis, axis])
            shape = [1] * dimensions
            shape[axis] = len(kernel)
            padding = [0] * (2 * dimensions)
            padding[2 * (dimensions - axis - 1) : 2 * (dimensions - axis)] = [
                int(radius)
            ] * 2
            values = convolution(
                F.pad(values, padding, mode="replicate"),
                (kernel / kernel.sum()).reshape(1, 1, *shape),
            )
    else:
        # A separable index-space Gaussian would smooth the wrong physical directions.
        axes = torch.meshgrid(
            *(
                torch.arange(
                    -int(r), int(r) + 1, dtype=images.dtype, device=images.device
                )
                for r in radii
            ),
            indexing="ij",
        )
        offsets = torch.stack(axes, dim=-1)
        physical = (
            offsets
            @ torch.as_tensor(lattice, dtype=images.dtype, device=images.device).T
        )
        kernel = torch.exp(-0.5 * physical.square().sum(-1) / sigma**2)
        padding = [int(r) for r in radii[::-1] for _ in range(2)]
        # FFT convolution avoids an im2col workspace proportional to kernel volume.
        padded = F.pad(values, padding, mode="replicate")
        fft_shape = tuple(
            n + k - 1 for n, k in zip(padded.shape[2:], kernel.shape, strict=True)
        )
        axes = tuple(range(2, values.ndim))
        transformed = torch.fft.rfftn(padded, s=fft_shape, dim=axes)
        weights = torch.fft.rfftn(kernel / kernel.sum(), s=fft_shape)
        filtered = torch.fft.irfftn(transformed * weights, s=fft_shape, dim=axes)
        slices = tuple(
            slice(k - 1, k - 1 + n)
            for k, n in zip(kernel.shape, images.shape[1:], strict=True)
        )
        values = filtered[(slice(None), slice(None), *slices)]
    return values[:, 0]


def _make_level(images, data, factor, sigma):
    """Smooth native fitting images and stride their original physical lattice."""
    strides = {dim: min(factor, max(1, data.sizes[dim] - 1)) for dim in VOXEL_DIMS}
    selectors = {dim: slice(None, None, stride) for dim, stride in strides.items()}
    level = data.isel(selectors)
    dimensions = images.ndim - 1
    samples = tuple(selectors[dim] for dim in VOXEL_DIMS[-dimensions:])
    return _smooth_images(images, data, sigma)[(slice(None), *samples)], level


def _resize_halfway(field, source, destination):
    """Interpolate physical field components between same-origin pyramid grids."""
    from voxreg.fields import sample_field_at

    if source is destination:
        return field
    # All pyramid grids select index zero, so origins cancel exactly, even at large origins.
    basis_map = source.basis.T @ destination.basis
    mapped = (
        destination.indices
        @ (source.inverse_lattice @ basis_map @ destination.lattice).T
    )
    sampled = sample_field_at(
        field.double(), mapped.expand(len(field), *mapped.shape[1:]), source
    )
    result = torch.einsum("ij,bj...->bi...", basis_map.T, sampled)
    return result.masked_fill(destination.boundary, 0).float()


def _add_level_diagnostics(diagnostics, records, factors, sigmas, budgets):
    """Expose level-local outcomes and cumulative frame update counts."""
    diagnostics.coords["level"] = np.arange(len(factors))
    diagnostics["shrink_factor"] = ("level", np.asarray(factors))
    diagnostics["smoothing_sigma"] = ("level", np.asarray(sigmas), {"units": "mm"})
    diagnostics["iteration_budget"] = ("level", np.asarray(budgets))
    for name in ("n_iterations", "status", "objective"):
        values = np.concatenate([record[name] for record in records])
        diagnostics[f"level_{name}"] = (("time", "level"), values)
    diagnostics["n_iterations"] = diagnostics.level_n_iterations.sum("level")
    diagnostics["objective"] = diagnostics.level_objective.isel(level=-1, drop=True)
    diagnostics.attrs["objective_domain"] = (
        "final pyramid level; losses are not compared across levels"
    )
