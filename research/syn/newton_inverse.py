"""Research bounded planar Newton inversion in float64 voxel coordinates."""

import torch

from voxreg import fields, syn

_original = syn._final_inverse
COUNTS = {"accepted": 0, "fallback": 0, "left_residual": []}


def _newton_update(values, inverse, indices, boundary):
    height, width = values.shape[-2:]
    query = indices.movedim(-1, 1) + inverse
    j, i = query.unbind(1)
    cj, ci = j.clamp(0, height - 1), i.clamp(0, width - 1)
    j0, i0 = cj.floor().long(), ci.floor().long()
    tj, ti = cj - j0, ci - i0
    flat = values.flatten(2)
    corners = []
    for dj, di in ((0, 0), (0, 1), (1, 0), (1, 1)):
        key = (
            (j0 + dj).clamp(max=height - 1) * width + (i0 + di).clamp(max=width - 1)
        ).flatten(1)
        corners.append(
            flat.gather(2, key[:, None].expand(-1, 2, -1)).reshape_as(values)
        )
    p00, p01, p10, p11 = corners
    u = (p00 * (1 - ti[:, None]) + p01 * ti[:, None]) * (1 - tj[:, None]) + (
        p10 * (1 - ti[:, None]) + p11 * ti[:, None]
    ) * tj[:, None]
    dj = ((p10 - p00) * (1 - ti[:, None]) + (p11 - p01) * ti[:, None]) * (
        (j > 0) & (j < height - 1)
    )[:, None]
    di = ((p01 - p00) * (1 - tj[:, None]) + (p11 - p10) * tj[:, None]) * (
        (i > 0) & (i < width - 1)
    )[:, None]
    inside = ((j >= -0.5) & (j < height - 0.5) & (i >= -0.5) & (i < width - 0.5))[
        :, None
    ]
    u, dj, di = (torch.where(inside, x, 0) for x in (u, dj, di))
    residual = inverse + u
    a, b, c, d = 1 + dj[:, 0], di[:, 0], dj[:, 1], 1 + di[:, 1]
    det = a * d - b * c
    regular = torch.isfinite(det) & (det > 1e-12)
    denominator = torch.where(regular, det, 1)
    delta = torch.stack(
        (
            (d * residual[:, 0] - b * residual[:, 1]) / denominator,
            (a * residual[:, 1] - c * residual[:, 0]) / denominator,
        ),
        dim=1,
    )
    delta = torch.where(regular[:, None], delta, residual)
    scale = (0.5 / delta.abs().amax(1, keepdim=True).clamp_min(1e-30)).clamp(max=1)
    return (inverse - delta * scale).masked_fill(boundary, 0)


def final_inverse(field, grid):
    if len(grid.shape) != 2 or field.requires_grad or not field.is_cuda:
        return _original(field, grid)
    voxel = (
        fields._map_displacement(grid.inverse_lattice[None], field, grid.voxel_axes)
        .movedim(-1, 1)
        .contiguous()
    )
    inverse = -voxel.clone()
    for _ in range(12):
        inverse = _newton_update(voxel, inverse, grid.indices, grid.boundary)
    physical = (
        fields._map_displacement(grid.lattice[None], inverse)
        .movedim(-1, 1)
        .contiguous()
    )
    left = (
        fields._pixel_norm(fields.compose_fields(field, physical, grid), grid)
        .flatten(1)
        .amax(1)
    )
    COUNTS["left_residual"].append(float(left.max()))
    if not (torch.isfinite(physical).all() & (left <= 1e-10).all()).item():
        COUNTS["fallback"] += 1
        return _original(field, grid)
    COUNTS["accepted"] += 1
    right = (
        fields._pixel_norm(fields.compose_fields(physical, field, grid), grid)
        .flatten(1)
        .amax(1)
    )
    return physical, torch.maximum(left, right)
