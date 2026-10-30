"""Research-only FP64 inverse update avoiding per-voxel sqrt followed by divide."""

import torch

from voxreg.fields import _inverse_update, compose_fields
from voxreg.sampling import _map_displacement


def inverse_update(field, inverse, grid, damping):
    if not field.is_cuda:
        return _inverse_update(field, inverse, grid, damping)
    residual = compose_fields(field, inverse, grid)
    if grid.voxel_scale is not None:
        square = (residual * grid.voxel_scale).square().sum(1)
    else:
        square = (
            _map_displacement(grid.inverse_lattice[None], residual).square().sum(-1)
        )
    maximum = square.flatten(1).amax(1).reshape(-1, *([1] * len(grid.shape)))
    fraction = (damping * maximum.sqrt() * torch.rsqrt(square.clamp_min(5e-324))).clamp(
        max=1
    )
    fraction = torch.where(square > 0, fraction, (maximum > 0).to(field.dtype))
    return (inverse - damping * residual * fraction[:, None]).masked_fill(
        grid.boundary, 0
    )
