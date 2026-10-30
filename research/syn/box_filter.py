"""Research CUDA 2D box sums as separable native double convolutions."""

import torch.nn.functional as F

from voxreg.syn import _box_sum


def box_sum(values, radius):
    if not values.is_cuda or values.ndim != 4 or radius == 0:
        return _box_sum(values, radius)
    batch, channels, height, width = values.shape
    out = values.reshape(batch * channels, 1, height, width)
    width_kernel = values.new_ones((1, 1, 1, 2 * radius + 1))
    out = F.conv2d(out, width_kernel, padding=(0, radius))
    height_kernel = values.new_ones((1, 1, 2 * radius + 1, 1))
    return F.conv2d(out, height_kernel, padding=(radius, 0)).reshape_as(values)
