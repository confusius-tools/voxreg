"""Research eager float64 fusion using existing Torch pointwise operators."""

import torch

from voxreg import fields
from voxreg.sampling import _map_displacement


def geometry_map(matrix, field, axes=None):
    if not field.is_cuda or axes is not None:
        return _map_displacement(matrix, field, axes)
    expand = (-1,) + (1,) * (field.ndim - 2)
    components = []
    for row in range(matrix.shape[-2]):
        value = matrix[:, row, 0].reshape(expand) * field[:, 0]
        for column in range(1, field.shape[1]):
            value = torch.addcmul(
                value, matrix[:, row, column].reshape(expand), field[:, column]
            )
        components.append(value)
    return torch.stack(components, dim=-1)


def inverse_update(field, inverse, grid, damping):
    residual = fields.compose_fields(field, inverse, grid)
    norms = fields._pixel_norm(residual, grid)
    maximum = norms.flatten(1).amax(1).reshape(-1, *([1] * len(grid.shape)))
    fraction = (
        damping * maximum / norms.clamp_min(torch.finfo(field.dtype).tiny)
    ).clamp(max=1)
    return torch.addcmul(
        inverse, residual, fraction[:, None], value=-damping
    ).masked_fill(grid.boundary, 0)
