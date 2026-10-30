"""Research inverse acceleration with a conservative voxel-space contraction bound."""

from math import ceil, isfinite, log

import torch

from voxreg.fields import _pixel_norm, _sample_field, compose_fields, invert_field
from voxreg.sampling import _map_displacement

COUNTS = {"certified": 0, "fallback": 0, "iterations": [], "kappa": []}


def contract_inverse(field, grid, *, iterations=20, initial=None):
    # Public differentiable inversion and dense restart behavior remain untouched.
    if initial is not None or (torch.is_grad_enabled() and field.requires_grad):
        return invert_field(field, grid, iterations=iterations, initial=initial)
    voxels = _map_displacement(
        grid.inverse_lattice[None], field, grid.voxel_axes
    ).movedim(-1, 1)
    row_bound = torch.zeros(field.shape[:2], dtype=field.dtype, device=field.device)
    for axis in range(len(grid.shape)):
        row_bound += voxels.diff(dim=axis + 2).abs().flatten(2).amax(2)
    kappa, amplitude, boundary = torch.stack(
        (
            row_bound.amax(),
            voxels.abs().amax(),
            torch.where(grid.boundary, field, 0).abs().amax(),
        )
    ).tolist()
    COUNTS["kappa"].append(kappa if isfinite(kappa) else None)
    count = iterations
    if isfinite(kappa) and isfinite(amplitude) and boundary == 0 and kappa < 1:
        # Starting at -u, inverse error is at most 2*amplitude*kappa**count.
        if 2 * amplitude <= 1e-12:
            count = 0
        elif kappa == 0:
            count = 1
        else:
            count = max(1, ceil(log(1e-12 / (2 * amplitude)) / log(kappa)))
    if count >= iterations:
        COUNTS["fallback"] += 1
        return invert_field(field, grid, iterations=iterations)
    inverse = -field.clone()
    for _ in range(count):
        inverse = -_sample_field(field, inverse, grid)
    inverse = inverse.masked_fill(grid.boundary, 0)
    left = _pixel_norm(compose_fields(field, inverse, grid), grid).flatten(1).amax(1)
    right = _pixel_norm(compose_fields(inverse, field, grid), grid).flatten(1).amax(1)
    if not (left <= 1e-10).all().item():
        COUNTS["fallback"] += 1
        return invert_field(field, grid, iterations=iterations)
    COUNTS["certified"] += 1
    COUNTS["iterations"].append(count)
    return inverse, torch.maximum(left, right)
