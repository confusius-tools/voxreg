"""Research: take a square root only after the maximum squared voxel norm."""

import torch

from voxreg import fields, syn


def maximum_norm(field, grid):
    if grid.voxel_scale is not None:
        scaled = field * grid.voxel_scale.view(1, -1, *([1] * len(grid.shape)))
        squared = scaled.square().sum(dim=1)
    else:
        mapped = fields._map_displacement(grid.inverse_lattice[None], field)
        squared = mapped.square().sum(dim=-1)
    return squared.flatten(1).amax(dim=1).sqrt()


def candidate_fields(uf, um, gf, gm, grid, learning_rate, update_sigma, total_sigma):
    broadcast = (slice(None),) + (None,) * (len(grid.shape) + 1)
    candidates = []
    for old, gradient in ((uf, gf), (um, gm)):
        gradient = syn.smooth_field(
            torch.nan_to_num(gradient), grid, sigma=update_sigma
        )
        maximum = maximum_norm(gradient, grid)
        scale = torch.where(
            maximum > 0,
            learning_rate / maximum.clamp_min(torch.finfo(uf.dtype).tiny),
            0,
        )
        delta = (-gradient * scale[broadcast]).masked_fill(grid.boundary, 0)
        candidate = syn.compose_fields(old, delta, grid)
        candidates.append(syn.smooth_field(candidate, grid, sigma=total_sigma))
    admissible = uf.new_ones((len(uf),), dtype=torch.bool)
    for candidate in candidates:
        determinant = syn.jacobian_determinant(candidate, grid)
        admissible = (
            admissible
            & torch.isfinite(determinant).flatten(1).all(dim=1)
            & (determinant.flatten(1).amin(dim=1) > 0)
        )
    return candidates[0], candidates[1], admissible
