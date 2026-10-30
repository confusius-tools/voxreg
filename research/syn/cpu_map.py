"""Research CPU one-GEMM-per-image geometry, never one tiny GEMM per voxel."""

import torch

from voxreg.sampling import _map_displacement


def cpu_map(matrix, field, axes=None):
    if field.device.type != "cpu" or axes is not None:
        return _map_displacement(matrix, field, axes)
    batch = max(matrix.shape[0], field.shape[0])
    values = field.flatten(2).expand(batch, -1, -1)
    mapped = torch.bmm(matrix.expand(batch, -1, -1), values)
    return (
        mapped.reshape(batch, matrix.shape[1], *field.shape[2:])
        .movedim(1, -1)
        .contiguous()
    )
