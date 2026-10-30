"""Research small-matrix Jacobians without per-voxel LU factorizations."""

import torch

from voxreg.fields import _check_field
from voxreg.sampling import _map_displacement


def jacobian_direct(field, grid):
    _check_field(field, grid)
    dimensions = len(grid.shape)
    derivative = torch.stack(
        torch.gradient(field, dim=tuple(range(2, field.ndim))), dim=2
    )
    axes = (
        tuple(grid.voxel_axes.index(k) for k in range(dimensions))
        if grid.voxel_axes is not None
        else None
    )
    physical = _map_displacement(
        grid.inverse_lattice.T[None], derivative.flatten(0, 1), axes
    ).reshape(field.shape[0], dimensions, *grid.shape, dimensions)
    a = physical[:, 0, ..., 0] + 1
    b = physical[:, 0, ..., 1]
    d = physical[:, 1, ..., 0]
    e = physical[:, 1, ..., 1] + 1
    if dimensions == 2:
        return a * e - b * d
    c = physical[:, 0, ..., 2]
    f = physical[:, 1, ..., 2]
    g = physical[:, 2, ..., 0]
    h = physical[:, 2, ..., 1]
    i = physical[:, 2, ..., 2] + 1
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
