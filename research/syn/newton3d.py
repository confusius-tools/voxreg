"""Research volume counterpart of the bounded float64 voxel Newton solve."""

from itertools import product

import torch
from newton_inverse import final_inverse as planar_inverse

from voxreg import fields, syn

_original = syn._final_inverse
COUNTS = {"accepted": 0, "fallback": 0, "left_residual": []}


def _lerp(a, b, t):
    return a * (1 - t) + b * t


def _bilinear(a, b, c, d, t, s):
    return _lerp(_lerp(a, b, s), _lerp(c, d, s), t)


def _newton_update3d(values, inverse, indices, boundary):
    query = indices.movedim(-1, 1) + inverse
    sizes = values.shape[2:]
    clamped = [query[:, axis].clamp(0, size - 1) for axis, size in enumerate(sizes)]
    lower = [q.floor().long() for q in clamped]
    fraction = [(q - lo)[:, None] for q, lo in zip(clamped, lower, strict=True)]
    flat = values.flatten(2)
    corners = {}
    for bits in product((0, 1), repeat=3):
        coords = [
            (lo + bit).clamp(max=size - 1)
            for lo, bit, size in zip(lower, bits, sizes, strict=True)
        ]
        key = ((coords[0] * sizes[1] + coords[1]) * sizes[2] + coords[2]).flatten(1)
        corners[bits] = flat.gather(2, key[:, None].expand(-1, 3, -1)).reshape_as(
            values
        )
    p000, p001 = corners[(0, 0, 0)], corners[(0, 0, 1)]
    p010, p011 = corners[(0, 1, 0)], corners[(0, 1, 1)]
    p100, p101 = corners[(1, 0, 0)], corners[(1, 0, 1)]
    p110, p111 = corners[(1, 1, 0)], corners[(1, 1, 1)]
    tz, ty, tx = fraction
    u = _lerp(
        _bilinear(p000, p001, p010, p011, ty, tx),
        _bilinear(p100, p101, p110, p111, ty, tx),
        tz,
    )
    dz = _bilinear(p100 - p000, p101 - p001, p110 - p010, p111 - p011, ty, tx)
    dy = _bilinear(p010 - p000, p011 - p001, p110 - p100, p111 - p101, tz, tx)
    dx = _bilinear(p001 - p000, p011 - p010, p101 - p100, p111 - p110, tz, ty)
    inside = query[:, 0].new_ones(query.shape[:1] + query.shape[2:], dtype=torch.bool)
    derivatives = [dz, dy, dx]
    for axis, size in enumerate(sizes):
        inside = inside & (query[:, axis] >= -0.5) & (query[:, axis] < size - 0.5)
        derivatives[axis] = (
            derivatives[axis]
            * ((query[:, axis] > 0) & (query[:, axis] < size - 1))[:, None]
        )
    u, dz, dy, dx = [torch.where(inside[:, None], v, 0) for v in (u, *derivatives)]
    r = inverse + u
    a, b, c = 1 + dz[:, 0], dy[:, 0], dx[:, 0]
    d, e, f = dz[:, 1], 1 + dy[:, 1], dx[:, 1]
    g, h, i = dz[:, 2], dy[:, 2], 1 + dx[:, 2]
    co0, co1, co2 = e * i - f * h, f * g - d * i, d * h - e * g
    det = a * co0 + b * co1 + c * co2
    regular = torch.isfinite(det) & (det > 1e-12)
    den = torch.where(regular, det, 1)
    delta = torch.stack(
        (
            (co0 * r[:, 0] + (c * h - b * i) * r[:, 1] + (b * f - c * e) * r[:, 2])
            / den,
            (co1 * r[:, 0] + (a * i - c * g) * r[:, 1] + (c * d - a * f) * r[:, 2])
            / den,
            (co2 * r[:, 0] + (b * g - a * h) * r[:, 1] + (a * e - b * d) * r[:, 2])
            / den,
        ),
        dim=1,
    )
    delta = torch.where(regular[:, None], delta, r)
    scale = (0.5 / delta.abs().amax(1, keepdim=True).clamp_min(1e-30)).clamp(max=1)
    return (inverse - delta * scale).masked_fill(boundary, 0)


def final_inverse(field, grid):
    if len(grid.shape) == 2:
        return planar_inverse(field, grid)
    if field.requires_grad or not field.is_cuda:
        return _original(field, grid)
    voxel = (
        fields._map_displacement(grid.inverse_lattice[None], field, grid.voxel_axes)
        .movedim(-1, 1)
        .contiguous()
    )
    inverse = -voxel.clone()
    for _ in range(12):
        inverse = _newton_update3d(voxel, inverse, grid.indices, grid.boundary)
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
