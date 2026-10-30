"""Research-only float32 fitting-gradient filter; displacement states stay double."""

import torch
import torch.nn.functional as F

from voxreg.fields import smooth_field


def gradient_filter(field, grid, *, sigma):
    # The candidate phase uses 1.5 for gradients and 0.5 for displacement states.
    if not field.is_cuda or sigma != 1.5:
        return smooth_field(field, grid, sigma=sigma)
    scale = field.abs().flatten(1).amax(1).reshape(-1, 1, *([1] * len(grid.shape)))
    scale = scale.clamp_min(torch.finfo(field.dtype).tiny)
    result = (field / scale).float()
    radius = int(3 * sigma + 0.5)
    positions = torch.arange(
        -radius, radius + 1, dtype=result.dtype, device=result.device
    )
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel = kernel / kernel.sum()
    dimensions = len(grid.shape)
    convolution = F.conv2d if dimensions == 2 else F.conv3d
    for axis in range(dimensions):
        shape = [1] * dimensions
        shape[axis] = len(kernel)
        weights = kernel.reshape(1, 1, *shape).expand(dimensions, 1, *shape)
        padding = [0] * (2 * dimensions)
        padding[2 * (dimensions - axis - 1) : 2 * (dimensions - axis)] = [
            radius,
            radius,
        ]
        result = convolution(
            F.pad(result, padding, mode="replicate"), weights, groups=dimensions
        )
    return result.masked_fill(grid.boundary, 0).clamp(-1, 1).double() * scale
