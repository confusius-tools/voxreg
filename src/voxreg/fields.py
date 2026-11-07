"""Physical displacement operations for compositive SyN.

Fields have shape (batch, components, *spatial_shape) and contain millimetres.
Volumes use world z/y/x components; planes use the existing orthonormal plane
basis. These are field operations, not yet a public registration model.
"""

from dataclasses import dataclass
from math import ceil, isfinite, log

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from voxreg.geometry import _validate_registration_geometry, position_to_world
from voxreg.sampling import _map_displacement, make_grid


@dataclass(frozen=True)
class FieldGrid:
    """Resident geometry; origin-free offsets avoid large-world cancellation."""

    shape: tuple[int, ...]
    basis: torch.Tensor
    lattice: torch.Tensor
    inverse_lattice: torch.Tensor
    indices: torch.Tensor
    half: torch.Tensor
    boundary: torch.Tensor
    voxel_axes: tuple[int, ...] | None = None
    voxel_scale: torch.Tensor | None = None


def make_field_grid(data: xr.DataArray, *, device: str = "cpu") -> FieldGrid:
    """Build a float64 physical field lattice for a volume or native k=1 plane.

    Parameters
    ----------
    data : xarray.DataArray
        Spatial VoxelData defining the field shape and millimetre geometry.
    device : str, default: "cpu"
        Torch device holding the resident lattice.

    Returns
    -------
    FieldGrid
        Geometry for component-first physical displacement fields.
    """
    _validate_registration_geometry(data)
    geometry = make_grid(data, data, device=device)
    planar = data.sizes["k"] == 1
    shape = tuple(data.sizes[d] for d in (("j", "i") if planar else ("k", "j", "i")))
    basis = geometry.basis
    lattice = basis.T @ position_to_world(data)[:3, 1 if planar else 0 : 3]
    inverse = np.linalg.inv(lattice)
    # Exact zeros only; oblique/sheared lattices keep the full mapping.
    sparse = np.count_nonzero(inverse) == len(shape)
    voxel_axes = (
        tuple(int(i) for i in np.abs(inverse).argmax(axis=1)) if sparse else None
    )
    voxel_scale = (
        torch.tensor(
            np.abs(inverse).sum(axis=0), dtype=torch.float64, device=device
        ).reshape(1, len(shape), *([1] * len(shape)))
        if sparse
        else None
    )
    axes = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float64, device=device) for n in shape),
        indexing="ij",
    )
    indices = torch.stack(axes, dim=-1)[None]
    boundary = torch.zeros(shape, dtype=torch.bool, device=device)
    for axis, coordinate in enumerate(axes):
        boundary = boundary | (coordinate == 0) | (coordinate == shape[axis] - 1)
    return FieldGrid(
        shape,
        torch.tensor(basis, dtype=torch.float64, device=device),
        torch.tensor(lattice, dtype=torch.float64, device=device),
        torch.tensor(inverse, dtype=torch.float64, device=device),
        indices,
        torch.tensor([(n - 1) / 2 for n in shape], dtype=torch.float64, device=device),
        boundary[None, None],
        voxel_axes,
        voxel_scale,
    )


def _check_field(field: torch.Tensor, grid: FieldGrid) -> None:
    """Check component/spatial shape and the resident grid's precision and device."""
    if field.shape[1:] != (len(grid.shape), *grid.shape):
        raise ValueError("Field shape must be (batch, components, *grid.shape).")
    if field.dtype != grid.indices.dtype or field.device != grid.indices.device:
        raise ValueError(
            f"Fields must match the grid device and {grid.indices.dtype} dtype."
        )


def _sample_field(
    field: torch.Tensor, displacement: torch.Tensor, grid: FieldGrid
) -> torch.Tensor:
    """Sample a field at physical displacements from its native voxel centers."""
    offsets = _map_displacement(
        grid.inverse_lattice[None], displacement, grid.voxel_axes
    )
    mapped = grid.indices + offsets
    return sample_field_at(field, mapped, grid)


def sample_field_at(
    field: torch.Tensor, mapped: torch.Tensor, grid: FieldGrid
) -> torch.Tensor:
    """Evaluate a resident field at batched voxel positions on any query grid.

    Parameters
    ----------
    field : torch.Tensor
        Float64 millimetre field, shaped (batch, components, *grid.shape).
    mapped : torch.Tensor
        Voxel query positions with components last; may broadcast over frames.
    grid : FieldGrid
        Resident source-field lattice.

    Returns
    -------
    torch.Tensor
        Component-first sampled displacements, zero outside field support.
    """
    sizes = 2 * grid.half + 1
    valid = ((mapped >= -0.5) & (mapped < sizes - 0.5)).all(dim=-1)
    normalized = (mapped / grid.half - 1).flip(-1)
    sampled = F.grid_sample(
        field, normalized, mode="bilinear", padding_mode="border", align_corners=True
    )
    # Like ITK composition: no field contribution outside the finite field domain.
    return torch.where(valid[:, None], sampled, 0)


def compose_fields(
    outer: torch.Tensor, inner: torch.Tensor, grid: FieldGrid
) -> torch.Tensor:
    """Return outer ∘ inner: inner(x) + outer(x + inner(x)), not vector addition.

    Parameters
    ----------
    outer : torch.Tensor
        Outer float64 millimetre displacement field.
    inner : torch.Tensor
        Inner field with the same shape, applied first.
    grid : FieldGrid
        Shared resident field lattice.

    Returns
    -------
    torch.Tensor
        Component-first displacement for the complete composed pull.
    """
    _check_field(outer, grid)
    _check_field(inner, grid)
    if outer.shape != inner.shape:
        raise ValueError("Composed fields must have equal batch and spatial shapes.")
    return inner + _sample_field(outer, inner, grid)


def smooth_field(field: torch.Tensor, grid: FieldGrid, *, sigma: float) -> torch.Tensor:
    """Smooth fields with a Gaussian in voxel units and fixed active boundaries.

    As in ITK SyN, sigma=0 bypasses smoothing/boundary enforcement. A plane has
    only two active boundaries; its singleton acquisition axis is never expanded.

    Parameters
    ----------
    field : torch.Tensor
        Float64 millimetre field, shaped (batch, components, *grid.shape).
    grid : FieldGrid
        Resident lattice and fixed active-boundary mask.
    sigma : float
        Finite, nonnegative Gaussian standard deviation in voxel units.

    Returns
    -------
    torch.Tensor
        Smoothed field with fixed boundaries, or the original field for sigma zero.
    """
    _check_field(field, grid)
    if not isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative (voxel units).")
    if sigma == 0:
        return field
    radius = int(3 * sigma + 0.5)
    positions = torch.arange(
        -radius, radius + 1, dtype=field.dtype, device=field.device
    )
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel = kernel / kernel.sum()
    dimensions = len(grid.shape)
    convolution = F.conv2d if dimensions == 2 else F.conv3d
    channels = 1 if field.is_cuda and dimensions == 2 else dimensions
    result = field.reshape(-1, 1, *grid.shape) if channels == 1 else field
    for axis in range(dimensions):
        shape = [1] * dimensions
        shape[axis] = len(kernel)
        weights = kernel.reshape(1, 1, *shape).expand(channels, 1, *shape)
        padding = [0] * (2 * dimensions)
        padding[2 * (dimensions - axis - 1) : 2 * (dimensions - axis)] = [
            radius,
            radius,
        ]
        result = convolution(
            F.pad(result, padding, mode="replicate"), weights, groups=channels
        )
    return result.reshape_as(field).masked_fill(grid.boundary, 0)


def _measure_norm(field: torch.Tensor, grid: FieldGrid) -> torch.Tensor:
    """Measure physical vectors in native voxel-coordinate units."""
    if grid.voxel_scale is not None:
        values = field * grid.voxel_scale
        if field.device.type == "cpu":
            # CPU short-vector reductions need contiguous components.
            return torch.linalg.vector_norm(values.movedim(1, -1).contiguous(), dim=-1)
        return torch.linalg.vector_norm(values, dim=1)
    return torch.linalg.vector_norm(
        _map_displacement(grid.inverse_lattice[None], field), dim=-1
    )


def _update_inverse(field, inverse, grid, damping):
    """Apply one damped, clipped inverse update without host-side polling."""
    residual = compose_fields(field, inverse, grid)
    norms = _measure_norm(residual, grid)
    maximum = norms.flatten(1).amax(dim=1).reshape(-1, *([1] * len(grid.shape)))
    fraction = (
        damping * maximum / norms.clamp_min(torch.finfo(field.dtype).tiny)
    ).clamp(max=1)
    return (inverse - damping * residual * fraction[:, None]).masked_fill(
        grid.boundary, 0
    )


def _compute_contracted_cpu_inverse(field, grid, iterations):
    """Use fewer fixed-point updates only when a voxel-space bound certifies them."""
    voxels = _map_displacement(
        grid.inverse_lattice[None], field, grid.voxel_axes
    ).movedim(-1, 1)
    row_bound = voxels.new_zeros(field.shape[:2])
    for axis in range(len(grid.shape)):
        row_bound += voxels.diff(dim=axis + 2).abs().flatten(2).amax(2)
    kappa = float(row_bound.amax())
    amplitude = float(voxels.abs().amax())
    if not (isfinite(kappa) and isfinite(amplitude) and kappa < 1):
        return None
    if torch.where(grid.boundary, field, 0).any():
        return None
    # From -u, the inverse error is at most 2*amplitude*kappa**count.
    if 2 * amplitude <= 1e-12:
        count = 0
    elif kappa == 0:
        count = 1
    else:
        count = max(1, ceil(log(1e-12 / (2 * amplitude)) / log(kappa)))
    if count >= iterations:
        return None
    inverse = -field.clone()
    for _ in range(count):
        inverse = -_sample_field(field, inverse, grid)
    inverse = inverse.masked_fill(grid.boundary, 0)
    left = _measure_norm(compose_fields(field, inverse, grid), grid).flatten(1).amax(1)
    if not (left <= 1e-10).all():
        return None
    right = _measure_norm(compose_fields(inverse, field, grid), grid).flatten(1).amax(1)
    return inverse, torch.maximum(left, right)


def invert_field(
    field: torch.Tensor,
    grid: FieldGrid,
    *,
    iterations: int = 20,
    initial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate a zero-boundary inverse; return it and two-sided max error in pixels.

    ITK-style damped, clipped composition-residual updates, batched independently.
    No per-iteration host polling. Intended for the zero-boundary smooth fields
    used by compositive SyN; a finite iteration budget does not guarantee an inverse.
    Callers must inspect the returned error and Jacobians before accepting a warp.

    Parameters
    ----------
    field : torch.Tensor
        Float64 millimetre field to invert.
    grid : FieldGrid
        Resident field lattice.
    iterations : int, default: 20
        Positive number of clipped residual updates.
    initial : torch.Tensor, optional
        Starting inverse with the same shape as the field; copied before updates.

    Returns
    -------
    inverse : torch.Tensor
        Estimated component-first inverse displacement in millimetres.
    error : torch.Tensor
        Per-frame maximum two-sided composition residual in native pixels.
    """
    _check_field(field, grid)
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations < 1
    ):
        raise ValueError("iterations must be a positive integer.")
    inverse = torch.zeros_like(field) if initial is None else initial.clone()
    _check_field(inverse, grid)
    if inverse.shape != field.shape:
        raise ValueError("Initial inverse must match the field shape.")
    for iteration in range(iterations):
        inverse = _update_inverse(field, inverse, grid, 0.75 if iteration == 0 else 0.5)
    left = (
        _measure_norm(compose_fields(field, inverse, grid), grid).flatten(1).amax(dim=1)
    )
    right = (
        _measure_norm(compose_fields(inverse, field, grid), grid).flatten(1).amax(dim=1)
    )
    return inverse, torch.maximum(left, right)


def compute_jacobian_determinant(field: torch.Tensor, grid: FieldGrid) -> torch.Tensor:
    """Compute det(I + du/dphysical) on oblique, sheared or reversed grids.

    Positive sampled determinants alone do not prove global invertibility.

    Parameters
    ----------
    field : torch.Tensor
        Float64 millimetre field, shaped (batch, components, *grid.shape).
    grid : FieldGrid
        Resident lattice, including shear and axis orientation.

    Returns
    -------
    torch.Tensor
        Determinant at each field voxel, shaped (batch, *grid.shape).
    """
    _check_field(field, grid)
    derivatives = torch.gradient(field, dim=tuple(range(2, field.ndim)))
    voxel_axes = grid.voxel_axes
    if voxel_axes is not None:
        # Exact axis permutations/reversals need no packed physical-derivative tensor.
        def entry(component, physical_axis):
            """Read one physical derivative through the exact axis permutation."""
            voxel_axis = voxel_axes.index(physical_axis)
            return (
                grid.inverse_lattice[voxel_axis, physical_axis]
                * derivatives[voxel_axis][:, component]
            )

        a, b = entry(0, 0) + 1, entry(0, 1)
        d, e = entry(1, 0), entry(1, 1) + 1
        if len(grid.shape) == 2:
            return a * e - b * d
        c, f = entry(0, 2), entry(1, 2)
        g, h, i = entry(2, 0), entry(2, 1), entry(2, 2) + 1
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    derivative = torch.stack(derivatives, dim=2)
    physical = (
        _map_displacement(
            grid.inverse_lattice.T[None],
            derivative.flatten(0, 1),
        )
        .reshape(field.shape[0], field.shape[1], *grid.shape, len(grid.shape))
        .movedim(1, -2)
    )
    # Explicit small determinants avoid per-voxel LU workspaces.
    a = physical[..., 0, 0] + 1
    b = physical[..., 0, 1]
    d = physical[..., 1, 0]
    e = physical[..., 1, 1] + 1
    if len(grid.shape) == 2:
        return a * e - b * d
    c = physical[..., 0, 2]
    f = physical[..., 1, 2]
    g = physical[..., 2, 0]
    h = physical[..., 2, 1]
    i = physical[..., 2, 2] + 1
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
