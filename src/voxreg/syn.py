"""Batched compositive greedy SyN, with two midpoint-to-image pull fields.

The midpoint fields are optimized independently (as in FireANTs); smoothed
physical updates compose with the accumulated maps. Inversion uses the ITK-style
field solver. No vendor implementation or optimizer source is transplanted.
"""

from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from confusius._utils.progress import progress_bar

from voxreg.convergence import WindowConvergence
from voxreg.fields import (
    _pixel_norm,
    compose_fields,
    invert_field,
    jacobian_determinant,
    make_field_grid,
    sample_field_at,
    smooth_field,
)
from voxreg.geometry import position_to_world
from voxreg.sampling import (
    _prepare_displaced_sampling,
    _sample_prepared,
    global_ncc,
    make_grid,
    sample_rigid,
    world_matrices,
)


def _box_sum(values, radius):
    """Separable sliding sums: float64 statistics without a cubic convolution."""
    dimensions = values.ndim - 2
    for axis in range(dimensions):
        padding = [0] * (2 * dimensions)
        padding[2 * (dimensions - axis - 1) : 2 * (dimensions - axis)] = [
            radius + 1,
            radius,
        ]
        cumulative = F.pad(values, padding).cumsum(dim=axis + 2)
        width = 2 * radius + 1
        values = cumulative.narrow(
            axis + 2, width, values.shape[axis + 2]
        ) - cumulative.narrow(axis + 2, 0, values.shape[axis + 2])
    return values


def local_ncc(moving, fixed, valid, radius):
    """Masked squared neighborhood correlation; lower is better, per frame."""
    x, y = moving.double(), fixed.double()
    w = valid.double()
    stats = _box_sum(
        torch.stack((w, w * x, w * y, w * x * x, w * y * y, w * x * y), dim=1), radius
    )
    n, sx, sy, sxx, syy, sxy = stats.unbind(dim=1)
    count = n.clamp_min(1)
    vx = (sxx - sx * sx / count).clamp_min(0)
    vy = (syy - sy * sy / count).clamp_min(0)
    usable = valid & (n > 1) & (vx > 1e-12) & (vy > 1e-12)
    cc = ((sxy - sx * sy / count).square() / (vx * vy).clamp_min(1e-24)).clamp(max=1)
    axes = tuple(range(1, x.ndim))
    counts = usable.sum(dim=axes)
    loss = -torch.where(usable, cc, 0).sum(dim=axes) / counts.clamp_min(1)
    return loss, counts > 1


def _midpoint_loss(images, fixed_images, moving_sampler, fixed_sampler, uf, um, radius):
    """One differentiable objective; optimizer clocks and callbacks remain outside."""
    moved, mv = _sample_prepared(images, um, moving_sampler)
    stationary, fv = _sample_prepared(fixed_images, uf, fixed_sampler)
    valid = mv & fv
    loss, meaningful = local_ncc(moved, stationary, valid, radius)
    return loss, meaningful, valid


def _candidate_fields(uf, um, gf, gm, grid, learning_rate, update_sigma, total_sigma):
    """Independent halfway updates and their sampled-Jacobian acceptance guard."""
    broadcast = (slice(None),) + (None,) * (len(grid.shape) + 1)
    candidates = []
    for old, gradient in ((uf, gf), (um, gm)):
        gradient = smooth_field(torch.nan_to_num(gradient), grid, sigma=update_sigma)
        maximum = _pixel_norm(gradient, grid).flatten(1).amax(dim=1)
        scale = torch.where(
            maximum > 0,
            learning_rate / maximum.clamp_min(torch.finfo(uf.dtype).tiny),
            0,
        )
        delta = -gradient * scale[broadcast]
        # The acquisition/field boundary remains fixed even when sigma=0.
        delta = delta.masked_fill(grid.boundary, 0)
        candidate = compose_fields(old, delta, grid)
        candidate = smooth_field(candidate, grid, sigma=total_sigma)
        candidates.append(candidate)
    admissible = uf.new_ones((len(uf),), dtype=torch.bool)
    for candidate in candidates:
        determinant = jacobian_determinant(candidate, grid)
        admissible = (
            admissible
            & torch.isfinite(determinant).flatten(1).all(dim=1)
            & (determinant.flatten(1).amin(dim=1) > 0)
        )
    return candidates[0], candidates[1], admissible


def _fit(
    images,
    target,
    plane,
    grid,
    fixed_plane,
    parameters,
    reference,
    *,
    max_iterations,
    learning_rate,
    convergence_window_size,
    convergence_minimum_value,
    update_sigma,
    total_sigma,
    radius,
    metric_callback,
    metric_interval,
):
    b = len(images)
    shape = (b, len(grid.shape), *grid.shape)
    uf = torch.zeros(
        shape, dtype=torch.float64, device=images.device, requires_grad=True
    )
    um = torch.zeros_like(uf, requires_grad=True)
    bf, bm = uf.detach().clone(), um.detach().clone()
    best_loss = uf.new_full((b,), torch.inf)
    steps = torch.zeros(b, dtype=torch.int64, device=images.device)
    axes = tuple(range(1, images.ndim))
    constant = images.amax(dim=axes) == images.amin(dim=axes)
    status = torch.where(reference, 4, torch.where(constant, 3, 0))
    fixed_parameters = uf.new_zeros((b, len(fixed_plane.parameter_scale)))
    fixed_images = target.expand(b, *target.shape)
    moving_sampler = _prepare_displaced_sampling(images, parameters, plane)
    fixed_sampler = _prepare_displaced_sampling(
        fixed_images, fixed_parameters, fixed_plane
    )
    monitor = (
        WindowConvergence(b, convergence_window_size, images.device)
        if convergence_window_size <= max_iterations
        else None
    )
    broadcast = (slice(None),) + (None,) * (len(grid.shape) + 1)
    for iteration in range(max_iterations + 1):
        loss, meaningful, valid = _midpoint_loss(
            images,
            fixed_images,
            moving_sampler,
            fixed_sampler,
            uf,
            um,
            radius,
        )
        good = meaningful & torch.isfinite(loss) & ~constant
        if metric_callback is not None and (
            iteration % metric_interval == 0 or iteration == max_iterations
        ):
            metric_callback(steps, torch.where(good, loss.detach(), torch.nan))
        with torch.no_grad():
            active = status == 0
            overlap = valid.sum(dim=axes) > 1
            status = torch.where(active & ~overlap, 2, status)
            status = torch.where(active & overlap & ~good, 3, status)
            improved = (status == 0) & good & (loss < best_loss)
            bf = torch.where(improved[broadcast], uf.detach(), bf)
            bm = torch.where(improved[broadcast], um.detach(), bm)
            best_loss = torch.where(improved, loss.detach(), best_loss)
        if iteration == max_iterations:
            break
        gf, gm = torch.autograd.grad(torch.where(status == 0, loss, 0).sum(), (uf, um))
        with torch.no_grad():
            finite = torch.isfinite(gf).flatten(1).all(dim=1) & torch.isfinite(
                gm
            ).flatten(1).all(dim=1)
            status = torch.where((status == 0) & ~finite, 3, status)
            active = status == 0
            if monitor is not None:
                trend = monitor.update(loss, active)
                status = torch.where(
                    active & (trend <= convergence_minimum_value), 1, status
                )
            active = status == 0
            candidate_f, candidate_m, admissible = _candidate_fields(
                uf, um, gf, gm, grid, learning_rate, update_sigma, total_sigma
            )
            steps = steps + active
            status = torch.where(active & ~admissible, 6, status)
            uf = torch.where((status == 0)[broadcast], candidate_f, uf)
            um = torch.where((status == 0)[broadcast], candidate_m, um)
        uf = uf.detach().requires_grad_(True)
        um = um.detach().requires_grad_(True)
        cadence = 10 if images.is_cuda else 1
        if (
            monitor is not None
            and iteration + 1 >= convergence_window_size
            and (iteration + 1 - convergence_window_size) % cadence == 0
            and not (status == 0).any().item()
        ):
            break
    if metric_callback is not None and iteration < max_iterations:
        metric_callback(steps, torch.where(good, loss.detach(), torch.nan))
    with torch.no_grad():
        fi, _ = invert_field(bf, grid, iterations=80)
        mi, _ = invert_field(bm, grid, iterations=80)
        forward = compose_fields(bm, fi, grid)
        inverse = compose_fields(bf, mi, grid)
        determinant = jacobian_determinant(forward, grid)
        inverse_det = jacobian_determinant(inverse, grid)
        admissible = (
            torch.isfinite(forward).flatten(1).all(dim=1)
            & torch.isfinite(inverse).flatten(1).all(dim=1)
            & (determinant.flatten(1).amin(dim=1) > 0)
            & (inverse_det.flatten(1).amin(dim=1) > 0)
        )
        status = torch.where(~admissible, 6, status)
        forward = torch.where(admissible[broadcast], forward, 0)
        inverse = torch.where(admissible[broadcast], inverse, 0)
        bf = torch.where(admissible[broadcast], bf, 0)
        bm = torch.where(admissible[broadcast], bm, 0)
        left = (
            _pixel_norm(compose_fields(forward, inverse, grid), grid)
            .flatten(1)
            .amax(dim=1)
        )
        right = (
            _pixel_norm(compose_fields(inverse, forward, grid), grid)
            .flatten(1)
            .amax(dim=1)
        )
        objective, meaningful, _ = _midpoint_loss(
            images,
            fixed_images,
            moving_sampler,
            fixed_sampler,
            bf,
            bm,
            radius,
        )
        objective = torch.where(meaningful & ~constant, objective, torch.nan)
        status = torch.where(status == 0, 5, status)
    return forward, inverse, steps, status, objective, torch.maximum(left, right)


def _world_field(residual, grid, matrices, data):
    """Complete affine-plus-residual world displacement at native fixed centers."""
    af = position_to_world(data)
    linear = torch.tensor(
        matrices[:, :3, :3], dtype=torch.float64, device=residual.device
    )
    offset = torch.tensor(
        np.einsum("bij,j->bi", matrices[:, :3, :3] - np.eye(3), af[:3, 3])
        + matrices[:, :3, 3],
        dtype=torch.float64,
        device=residual.device,
    )
    lattice = torch.tensor(
        af[:3, 1 if len(grid.shape) == 2 else 0 : 3], device=residual.device
    )
    local = grid.indices @ lattice.T
    disp = torch.einsum(
        "bij,...j->bi...", linear - torch.eye(3, device=residual.device), local[0]
    )
    world = torch.einsum("bij,jc,bc...->bi...", linear, grid.basis, residual)
    return disp + world + offset.reshape(len(residual), 3, *([1] * len(grid.shape)))


def _inverse_world_field(residual, grid, matrices, fixed, moving):
    """Complete inverse on the original moving grid, including the inverse seed."""
    af, am = position_to_world(fixed), position_to_world(moving)
    inv = np.linalg.inv(matrices)
    planar = len(grid.shape) == 2
    shape = tuple(moving.sizes[d] for d in (("j", "i") if planar else ("k", "j", "i")))
    axes = torch.meshgrid(
        *(torch.arange(n, device=residual.device, dtype=torch.float64) for n in shape),
        indexing="ij",
    )
    indices = torch.stack(axes, dim=-1)
    lattice = torch.tensor(am[:3, 1 if planar else 0 : 3], device=residual.device)
    local = indices @ lattice.T
    linear = torch.tensor(inv[:, :3, :3], device=residual.device)
    origin = (
        np.einsum("bij,j->bi", inv[:, :3, :3], am[:3, 3]) + inv[:, :3, 3] - af[:3, 3]
    )
    query = torch.einsum("bij,...j->b...i", linear, local) + torch.tensor(
        origin, device=residual.device
    ).reshape(len(residual), *([1] * len(shape)), 3)
    voxel_map = torch.tensor(
        np.linalg.inv(af[:3, :3])[1 if planar else 0 : 3], device=residual.device
    )
    sampled = sample_field_at(residual, query @ voxel_map.T, grid)
    offset = (
        np.einsum("bij,j->bi", inv[:, :3, :3] - np.eye(3), am[:3, 3]) + inv[:, :3, 3]
    )
    return (
        torch.einsum(
            "bij,...j->bi...", linear - torch.eye(3, device=residual.device), local
        )
        + torch.einsum("ic,bc...->bi...", grid.basis, sampled)
        + torch.tensor(offset, device=residual.device).reshape(
            len(residual), 3, *([1] * len(shape))
        )
    )


def _labeled_field(values, template, convention):
    # Scalar VoxelData coordinates/affine index survive the extra vector dimension.
    result = xr.concat(
        [template.copy(data=values[:, component]) for component in range(3)],
        dim=xr.IndexVariable("world_component", ["z", "y", "x"]),
    )
    result = result.transpose("time", "k", "j", "i", "world_component")
    result.attrs = {
        "convention": convention,
        "representation": "displacement",
        "spatial_units": "mm",
        "units": "mm",
    }
    result.name = "displacement"
    return result


def register_syn(
    moving,
    fixed,
    fit,
    target,
    plane,
    initial,
    *,
    implicit_fixed,
    return_transforms,
    spatial_input,
    same_grid,
    timings,
    batch_size,
    max_iterations,
    learning_rate,
    convergence_window_size,
    convergence_minimum_value,
    device,
    progress,
    metric_callback,
    metric_interval,
    interpolation,
    update_sigma,
    total_sigma,
    radius,
):
    from voxreg.rigid import RegistrationResult, _output_template

    grid = make_field_grid(fixed, device=device)
    fixed_plane = make_grid(fixed, fixed, device=device)
    planar = len(grid.shape) == 2
    raw = moving.values[:, 0] if planar else moving.values
    axes = tuple(range(1, raw.ndim))
    integer = np.issubdtype(raw.dtype, np.integer)
    dtype = np.float64 if integer or raw.dtype.itemsize > 4 else np.float32
    initial = (
        np.zeros((len(raw), len(plane.parameter_scale))) if initial is None else initial
    )
    matrices = world_matrices(initial, plane)
    if np.any(np.linalg.det(matrices[:, :3, :3]) <= 0):
        raise ValueError("SyN affine seeds must preserve orientation.")
    outputs = []
    masks = []
    forwards = []
    inverses = []
    clocks = []
    statuses = []
    objectives = []
    inverse_errors = []
    jacobians = []
    nccs = []
    with progress_bar("Registering frames", len(raw), show=progress is True) as advance:
        for first in range(0, len(raw), batch_size):
            last = min(first + batch_size, len(raw))
            if device == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            images = torch.tensor(fit[first:last], device=device)
            stationary = torch.tensor(target, device=device)
            parameters = torch.tensor(initial[first:last], device=device)
            reference = torch.zeros(last - first, dtype=torch.bool, device=device)
            if implicit_fixed and first == 0:
                reference[0] = True
            if device == "cuda":
                torch.cuda.synchronize()
            timings["upload"] += perf_counter() - start

            def report(steps, loss, first=first, last=last):
                metric_callback(
                    np.arange(first, last, dtype=np.int64),
                    steps.cpu().numpy().copy(),
                    loss.cpu().numpy().copy(),
                )

            start = perf_counter()
            forward, inverse, steps, status, objective, error = _fit(
                images,
                stationary,
                plane,
                grid,
                fixed_plane,
                parameters,
                reference,
                max_iterations=max_iterations,
                learning_rate=learning_rate,
                convergence_window_size=convergence_window_size,
                convergence_minimum_value=convergence_minimum_value,
                update_sigma=update_sigma,
                total_sigma=total_sigma,
                radius=radius,
                metric_callback=report if metric_callback is not None else None,
                metric_interval=metric_interval,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            timings["optimize"] += perf_counter() - start
            start = perf_counter()
            with torch.no_grad():
                original = torch.tensor(
                    np.ascontiguousarray(raw[first:last], dtype=dtype), device=device
                )
                sampled, valid = sample_rigid(
                    original,
                    parameters,
                    plane,
                    displacement=forward,
                    interpolation=interpolation,
                )
                fills = original.amin(dim=axes).reshape(
                    last - first, *([1] * len(axes))
                )
                sampled = torch.where(valid, sampled, fills)
                if implicit_fixed:
                    index = (slice(None),) + (None,) * len(axes)
                    sampled = torch.where(reference[index], original, sampled)
                    valid = valid | reference[index]
                values = sampled.cpu().numpy()
                if integer:
                    limits = np.iinfo(raw.dtype)
                    values = np.clip(np.rint(values), limits.min, limits.max)
                outputs.append(values.astype(raw.dtype, copy=False))
                masks.append(valid.cpu().numpy())
                if return_transforms:
                    forwards.append(
                        _world_field(forward, grid, matrices[first:last], fixed)
                        .cpu()
                        .numpy()
                    )
                    inverses.append(
                        _inverse_world_field(
                            inverse, grid, matrices[first:last], fixed, moving
                        )
                        .cpu()
                        .numpy()
                    )
                jacobians.append(
                    (
                        jacobian_determinant(forward, grid).flatten(1).amin(dim=1)
                        * torch.tensor(
                            np.linalg.det(matrices[first:last, :3, :3]), device=device
                        )
                    )
                    .cpu()
                    .numpy()
                )
                fitted, fv = sample_rigid(
                    images, parameters, plane, displacement=forward
                )
                loss, meaningful = global_ncc(fitted, stationary, fv)
                nccs.append(torch.where(meaningful, -loss, torch.nan).cpu().numpy())
                clocks.append(steps.cpu().numpy())
                statuses.append(status.cpu().numpy())
                objectives.append(objective.cpu().numpy())
                inverse_errors.append(error.cpu().numpy())
            timings["resample_download"] += perf_counter() - start
            if progress is True:
                advance(last - first)
            elif callable(progress):
                progress(last, len(raw))
    template = _output_template(moving, fixed, same_grid)
    values, validity = np.concatenate(outputs), np.concatenate(masks)
    if planar:
        values, validity = values[:, None], validity[:, None]
    registered = template.copy(data=values)
    valid_data = template.copy(data=validity)
    valid_data.attrs = {"description": "Valid moving-image sample centers"}
    transforms = inverse_transforms = None
    if return_transforms:
        forward, inverse = np.concatenate(forwards), np.concatenate(inverses)
        if planar:
            forward, inverse = forward[:, :, None], inverse[:, :, None]
        transforms = _labeled_field(
            forward, template, "fixed-world-to-moving-world pull"
        )
        inverse_transforms = _labeled_field(
            inverse, moving, "moving-world-to-fixed-world pull"
        )
    labels = np.array(
        [
            "running",
            "converged",
            "insufficient_overlap",
            "invalid_metric",
            "reference",
            "max_iterations",
            "invalid_transform",
        ]
    )
    diagnostics = xr.Dataset(
        {
            "objective": ("time", np.concatenate(objectives)),
            "ncc_squared": ("time", np.concatenate(nccs)),
            "n_iterations": ("time", np.concatenate(clocks)),
            "status": ("time", labels[np.concatenate(statuses)]),
            "valid_fraction": ("time", validity.reshape(len(raw), -1).mean(axis=1)),
            "min_jacobian": ("time", np.concatenate(jacobians)),
            "inverse_error": (
                "time",
                np.concatenate(inverse_errors),
                {
                    "units": "voxel-coordinate norm",
                    "domain": "fixed virtual grid; residual maps before affine seed",
                },
            ),
        },
        coords={"time": moving.time},
        attrs={
            "transform": "syn",
            "batch_size": batch_size,
            "metric": "negative squared local NCC",
            "cc_radius": radius,
            "update_sigma_voxels": update_sigma,
            "total_sigma_voxels": total_sigma,
            "timings_seconds": timings,
            "regularity": "positive sampled Jacobians are checked; not a proof of global invertibility",
            "inverse_error_note": "sampled inverse is approximate; two-sided residual is reported, not forced to zero",
        },
    )
    if spatial_input:
        return RegistrationResult(
            registered.isel(time=0, drop=True),
            valid_data.isel(time=0, drop=True),
            transforms.isel(time=0, drop=True) if transforms is not None else None,
            diagnostics.isel(time=0, drop=True),
            inverse_transforms.isel(time=0, drop=True)
            if inverse_transforms is not None
            else None,
        )
    return RegistrationResult(
        registered, valid_data, transforms, diagnostics, inverse_transforms
    )
