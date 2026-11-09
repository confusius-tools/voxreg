"""Batched compositive greedy SyN, with two midpoint-to-image pull fields.

The midpoint fields are optimized independently (as in FireANTs); smoothed
physical updates compose with the accumulated maps. Inversion uses the ITK-style
field solver. No vendor implementation or optimizer source is transplanted.
"""

from dataclasses import replace
from functools import lru_cache
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from confusius._utils.progress import progress_bar

from voxreg.convergence import WindowConvergence
from voxreg.fields import (
    _check_field,
    _compute_contracted_cpu_inverse,
    _measure_norm,
    compose_fields,
    compute_jacobian_determinant,
    invert_field,
    make_field_grid,
    sample_field_at,
)
from voxreg.fields import (
    smooth_field as _smooth_field,
)
from voxreg.geometry import position_to_world
from voxreg.masks import _MetricMasks, _source
from voxreg.sampling import (
    _prepare_displaced_sampling,
    _recover_initial_parameters,
    _sample_prepared,
    build_world_matrices,
    compute_global_ncc,
    make_grid,
    sample_rigid,
)


@lru_cache(maxsize=4)
def _build_cpu_smoothing_weights(sigma, dimensions):
    """Build cached CPU smoothing kernels for SyN's two fixed sigmas."""
    radius = int(3 * sigma + 0.5)
    positions = torch.arange(-radius, radius + 1, dtype=torch.float32, device="cpu")
    kernel = torch.exp(-0.5 * (positions / sigma).square())
    kernel = kernel / kernel.sum()
    weights = []
    for axis in range(dimensions):
        shape = [1] * dimensions
        shape[axis] = len(kernel)
        weights.append(
            kernel.reshape(1, 1, *shape).expand(dimensions, 1, *shape).contiguous()
        )
    return radius, tuple(weights)


def smooth_field(field, grid, *, sigma):
    """Filter CPU SyN fields in scaled float32, preserving their storage dtype.

    Parameters
    ----------
    field : torch.Tensor
        Component-first float32 or float64 millimetre displacement field.
    grid : FieldGrid
        Resident physical field lattice and fixed boundary mask.
    sigma : float
        Gaussian standard deviation in voxel units.

    Returns
    -------
    torch.Tensor
        Same-dtype field with the same smoothing boundary and support conventions.
    """
    if (
        torch.compiler.is_compiling()
        or field.device.type != "cpu"
        or (torch.is_grad_enabled() and field.requires_grad)
        or sigma not in (0.5, 1.5)
    ):
        return _smooth_field(field, grid, sigma=sigma)
    _check_field(field, grid)
    dimensions = len(grid.shape)
    radius, weights = _build_cpu_smoothing_weights(sigma, dimensions)
    scale = field.abs().flatten(1).amax(1).reshape(-1, 1, *([1] * dimensions))
    scale = scale.clamp_min(torch.finfo(field.dtype).tiny)
    result = (field / scale).float()
    convolution = F.conv2d if dimensions == 2 else F.conv3d
    for axis, weight in enumerate(weights):
        padding = [0] * (2 * dimensions)
        padding[2 * (dimensions - axis - 1) : 2 * (dimensions - axis)] = [
            radius,
            radius,
        ]
        result = convolution(
            F.pad(result, padding, mode="replicate"), weight, groups=dimensions
        )
    result = result.masked_fill(grid.boundary, 0)
    return result.clamp(-1, 1).to(field.dtype) * scale


def _sum_neighborhoods(values, radius):
    """Sum neighborhoods separably in metric precision."""
    dimensions = values.ndim - 2
    for axis in range(dimensions):
        padding = [0] * (2 * dimensions)
        padding[2 * (dimensions - axis - 1) : 2 * (dimensions - axis)] = [
            radius + 1,
            radius,
        ]
        width = 2 * radius + 1
        if values.dtype == torch.float32:
            # Local accumulation avoids cancellation of long float32 prefix sums.
            padding[2 * (dimensions - axis - 1)] = radius
            kernel = [1] * dimensions
            kernel[axis] = width
            pool = F.avg_pool2d if dimensions == 2 else F.avg_pool3d
            values = pool(F.pad(values, padding), kernel, stride=1, divisor_override=1)
            continue
        cumulative = F.pad(values, padding).cumsum(dim=axis + 2)
        values = cumulative.narrow(
            axis + 2, width, values.shape[axis + 2]
        ) - cumulative.narrow(axis + 2, 0, values.shape[axis + 2])
    return values


def compute_local_ncc(moving, fixed, valid, radius):
    """Compute masked squared neighborhood correlation independently per frame.

    Parameters
    ----------
    moving : torch.Tensor
        Sampled moving images, shaped (batch, *spatial_shape).
    fixed : torch.Tensor
        Sampled fixed images with the same shape.
    valid : torch.Tensor
        Boolean common-support mask.
    radius : int
        Neighborhood radius in active voxel axes.

    Returns
    -------
    loss : torch.Tensor
        Per-frame negative mean squared correlation; lower is better.
    meaningful : torch.Tensor
        Whether each frame has sufficient support and nonconstant neighborhoods.
    """
    dtype = torch.float64 if moving.dtype == torch.float64 else torch.float32
    x, y = moving.to(dtype), fixed.to(dtype)
    w = valid.to(dtype)
    stats = _sum_neighborhoods(
        torch.stack((w, w * x, w * y, w * x * x, w * y * y, w * x * y), dim=1), radius
    )
    n, sx, sy, sxx, syy, sxy = stats.unbind(dim=1)
    count = n.clamp_min(1)
    vx = (sxx - sx * sx / count).clamp_min(0)
    vy = (syy - sy * sy / count).clamp_min(0)
    tolerance_x = tolerance_y = 1e-12
    if dtype == torch.float32:
        # Raw-moment cancellation must not make constant neighborhoods meaningful.
        epsilon = 16 * torch.finfo(dtype).eps
        tolerance_x = epsilon * (sxx + sx * sx / count).abs()
        tolerance_y = epsilon * (syy + sy * sy / count).abs()
    usable = valid & (n > 1) & (vx > tolerance_x) & (vy > tolerance_y)
    cc = ((sxy - sx * sy / count).square() / (vx * vy).clamp_min(1e-24)).clamp(max=1)
    axes = tuple(range(1, x.ndim))
    counts = usable.sum(dim=axes)
    loss = -torch.where(usable, cc, 0).sum(dim=axes) / counts.clamp_min(1)
    return loss, counts > 1


def _compute_midpoint_loss(
    images,
    fixed_images,
    moving_sampler,
    fixed_sampler,
    uf,
    um,
    radius,
    metric_masks=None,
    parameters=None,
):
    """Compute the differentiable halfway objective and common support."""
    moved, mv = _sample_prepared(images, um, moving_sampler, fitting=True)
    stationary, fv = _sample_prepared(fixed_images, uf, fixed_sampler, fitting=True)
    valid = mv & fv
    if metric_masks is not None:
        valid = metric_masks.apply(valid, parameters, um, uf)
    loss, meaningful = compute_local_ncc(moved, stationary, valid, radius)
    return loss, meaningful, valid


def _build_candidate_fields(
    uf, um, gf, gm, grid, learning_rate, update_sigma, total_sigma
):
    """Build independent halfway updates and check their sampled Jacobians."""
    broadcast = (slice(None),) + (None,) * (len(grid.shape) + 1)
    candidates = []
    for old, gradient in ((uf, gf), (um, gm)):
        gradient = smooth_field(torch.nan_to_num(gradient), grid, sigma=update_sigma)
        maximum = _measure_norm(gradient, grid).flatten(1).amax(dim=1)
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
        determinant = compute_jacobian_determinant(candidate, grid)
        admissible = (
            admissible
            & torch.isfinite(determinant).flatten(1).all(dim=1)
            & (determinant.flatten(1).amin(dim=1) > 0)
        )
    return candidates[0], candidates[1], admissible


def _compute_final_inverse(field, grid):
    """Invert final fields, using certified CPU contraction when available."""
    if (
        field.device.type == "cpu"
        and not field.requires_grad
        and not torch.compiler.is_compiling()
    ):
        contracted = _compute_contracted_cpu_inverse(field, grid, 80)
        if contracted is not None:
            return contracted
    return invert_field(field, grid, iterations=80)


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
    initial_fields=None,
    finalize=True,
    active_frames=None,
    metric_masks=None,
):
    """Fit independent halfway fields and finalize their best valid state in double."""
    physical_grid = grid
    # Only origin-free working field arithmetic is reduced; export geometry stays double.
    grid = replace(
        grid,
        inverse_lattice=grid.inverse_lattice.float(),
        indices=grid.indices.float(),
        half=grid.half.float(),
        voxel_scale=None if grid.voxel_scale is None else grid.voxel_scale.float(),
    )
    if (
        not torch.isfinite(grid.inverse_lattice).all()
        or torch.linalg.det(grid.inverse_lattice.double()) == 0
    ):
        raise ValueError(
            "SyN working lattice must remain finite and nonsingular in float32."
        )
    b = len(images)
    shape = (b, len(grid.shape), *grid.shape)
    if initial_fields is None:
        uf = torch.zeros(
            shape, dtype=torch.float32, device=images.device, requires_grad=True
        )
        um = torch.zeros_like(uf, requires_grad=True)
        seed_valid = torch.ones(b, dtype=torch.bool, device=images.device)
    else:
        seed_valid = torch.ones(b, dtype=torch.bool, device=images.device)
        for field in initial_fields:
            determinant = compute_jacobian_determinant(field.double(), physical_grid)
            seed_valid &= (
                torch.isfinite(field).flatten(1).all(1)
                & torch.isfinite(determinant).flatten(1).all(1)
                & (determinant.flatten(1).amin(1) > 0)
            )
        mask = (seed_valid & ~reference).reshape(b, *([1] * (len(grid.shape) + 1)))
        uf, um = (
            torch.where(mask, field.float(), 0).detach().requires_grad_(True)
            for field in initial_fields
        )
    bf, bm = uf.detach().clone(), um.detach().clone()
    best_loss = uf.new_full((b,), torch.inf)
    steps = torch.zeros(b, dtype=torch.int64, device=images.device)
    axes = tuple(range(1, images.ndim))
    constant = images.amax(dim=axes) == images.amin(dim=axes)
    status = torch.where(reference, 4, torch.where(constant, 3, 0))
    if metric_masks is not None:
        status = torch.where(reference, 4, 0)
    status = torch.where(~seed_valid & ~reference, 6, status)
    if active_frames is not None:
        status = torch.where(~active_frames & ~reference, 7, status)
    fixed_parameters = parameters.new_zeros((b, len(fixed_plane.parameter_scale)))
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
        loss, meaningful, valid = _compute_midpoint_loss(
            images,
            fixed_images,
            moving_sampler,
            fixed_sampler,
            uf,
            um,
            radius,
            metric_masks=metric_masks,
            parameters=parameters,
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
            candidate_f, candidate_m, admissible = _build_candidate_fields(
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
    if not finalize:
        with torch.no_grad():
            objective, meaningful, _ = _compute_midpoint_loss(
                images,
                fixed_images,
                moving_sampler,
                fixed_sampler,
                bf,
                bm,
                radius,
                metric_masks=metric_masks,
                parameters=parameters,
            )
            objective = torch.where(meaningful & ~constant, objective, torch.nan)
        return (
            bf.detach(),
            bm.detach(),
            steps,
            torch.where(status == 0, 5, status),
            objective,
            steps.new_zeros(b),
        )
    with torch.no_grad():
        # Final inversion, validation and exported displacements retain double arithmetic.
        grid = physical_grid
        bf, bm = bf.double(), bm.double()
        fi, _ = _compute_final_inverse(bf, grid)
        mi, _ = _compute_final_inverse(bm, grid)
        forward = compose_fields(bm, fi, grid)
        inverse = compose_fields(bf, mi, grid)
        determinant = compute_jacobian_determinant(forward, grid)
        inverse_det = compute_jacobian_determinant(inverse, grid)
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
            _measure_norm(compose_fields(forward, inverse, grid), grid)
            .flatten(1)
            .amax(dim=1)
        )
        right = (
            _measure_norm(compose_fields(inverse, forward, grid), grid)
            .flatten(1)
            .amax(dim=1)
        )
        objective, meaningful, _ = _compute_midpoint_loss(
            images,
            fixed_images,
            moving_sampler,
            fixed_sampler,
            bf,
            bm,
            radius,
            metric_masks=metric_masks,
            parameters=parameters,
        )
        objective = torch.where(meaningful & ~constant, objective, torch.nan)
        status = torch.where(status == 0, 5, status)
    return forward, inverse, steps, status, objective, torch.maximum(left, right)


def _fit_pyramid(
    images,
    target,
    plane,
    grid,
    fixed_plane,
    parameters,
    reference,
    *,
    moving,
    fixed,
    schedule,
    records,
    **options,
):
    """Carry both physical halfway maps between levels, then finalize natively."""
    from voxreg.pyramid import _make_level, _resize_halfway
    from voxreg.rigid import _STATUS_LABELS
    from voxreg.transforms import configure_transform

    factors, sigmas, budgets = schedule
    matrices = build_world_matrices(parameters.cpu().numpy(), plane)
    fields = None
    previous_grid = None
    level_steps, level_status, level_objectives = [], [], []
    metric_masks = options.get("metric_masks")
    for factor, sigma, budget in zip(factors, sigmas, budgets, strict=True):
        level_images, level_moving = _make_level(images, moving, factor, sigma)
        level_target, level_fixed = _make_level(target[None], fixed, factor, sigma)
        geometry = configure_transform(
            make_grid(
                level_fixed,
                level_moving,
                device=str(images.device),
                center=plane.center,
                plane_seed=matrices,
            ),
            level_fixed,
            "affine",
        )
        level_grid = make_field_grid(level_fixed, device=str(images.device))
        fixed_geometry = make_grid(level_fixed, level_fixed, device=str(images.device))
        if fields is not None:
            fields = tuple(
                _resize_halfway(field, previous_grid, level_grid) for field in fields
            )
        seed = torch.tensor(
            _recover_initial_parameters(matrices, geometry, len(images)),
            device=images.device,
        )
        level_masks = (
            None
            if metric_masks is None
            else _MetricMasks(*metric_masks.sources, level_fixed, geometry)
        )
        uf, um, steps, status, objective, _ = _fit(
            level_images,
            level_target[0],
            geometry,
            level_grid,
            fixed_geometry,
            seed,
            reference,
            **{**options, "max_iterations": budget, "metric_masks": level_masks},
            initial_fields=fields,
            finalize=False,
        )
        fields, previous_grid = (uf, um), level_grid
        level_steps.append(steps.cpu().numpy())
        level_status.append(_STATUS_LABELS[status.cpu().numpy()])
        level_objectives.append(objective.cpu().numpy())
    records.append(
        {
            "n_iterations": np.stack(level_steps, axis=1),
            "status": np.stack(level_status, axis=1),
            "objective": np.stack(level_objectives, axis=1),
        }
    )
    assert fields is not None
    fields = tuple(_resize_halfway(field, previous_grid, grid) for field in fields)
    # Only finalization uses the native lattice; no additional optimizer update.
    final_options = {**options, "max_iterations": 0, "metric_callback": None}
    forward, inverse, _, native_status, _, error = _fit(
        images,
        target,
        plane,
        grid,
        fixed_plane,
        parameters,
        reference,
        **final_options,
        initial_fields=fields,
    )
    status = torch.where(native_status == 6, native_status, status)
    return (
        forward,
        inverse,
        torch.as_tensor(np.sum(level_steps, axis=0), device=images.device),
        status,
        objective,
        error,
    )


def _build_world_field(residual, grid, matrices, data):
    """Build complete affine-plus-residual displacements at native fixed centers."""
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


def _build_inverse_world_field(residual, grid, matrices, fixed, moving):
    """Build the complete inverse on the moving grid, including the inverse seed."""
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


def _build_labeled_field(values, template, convention):
    """Label complete world displacements on the template's native lattice."""
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
    schedule=None,
    active_frames=None,
    fixed_mask=None,
    moving_mask=None,
):
    """Register prepared native images with batched compositive midpoint fields.

    Parameters
    ----------
    moving : xarray.DataArray
        Original moving VoxelData with a leading frame dimension.
    fixed : xarray.DataArray
        Spatial-only fixed VoxelData defining the output grid.
    fit : numpy.ndarray
        Normalized float32 moving fitting images.
    target : numpy.ndarray
        Normalized float32 fixed fitting image.
    plane : Plane or SpatialGrid
        Prepared independent-grid sampling geometry and seed parameterization.
    initial : numpy.ndarray, optional
        Physical parameters for the fixed affine prealignment.
    implicit_fixed : bool
        Whether the first moving frame is the exact reference.
    return_transforms : bool
        Whether to export complete forward and inverse displacement fields.
    spatial_input : bool
        Whether the caller supplied a volume without a time dimension.
    same_grid : bool
        Whether moving and fixed use the same native spatial lattice.
    timings : dict
        Accumulated upload, optimization and resampling times; updated in place.
    batch_size : int
        Number of resident moving frames per fitting batch.
    max_iterations : int
        Maximum number of field updates per frame.
    learning_rate : float
        Maximum normalized update magnitude in native pixels.
    convergence_window_size : int
        Number of active objective samples in the convergence window.
    convergence_minimum_value : float
        Minimum convergence value before a frame stops.
    device : str
        Torch execution device.
    progress : bool or callable or None
        Whether to display progress, or a frame-completion callback.
    metric_callback : callable, optional
        Callback receiving frame indices, update counts and objective values.
    metric_interval : int
        Number of updates between metric callback reports.
    interpolation : {"linear", "nearest"}
        Final original-intensity sampling rule; fitting remains linear.
    update_sigma : float
        Update-gradient Gaussian standard deviation in voxels.
    total_sigma : float
        Accumulated-field Gaussian standard deviation in voxels.
    radius : int
        Local correlation neighborhood radius in active voxel axes.
    schedule : tuple, optional
        Validated shrink factors, image smoothing sigmas in mm and iteration
        budgets. None retains the native single-resolution fitting path.
    active_frames : numpy.ndarray, optional
        Per-frame mask for staged fitting. Inactive frames retain the complete
        affine seed with zero residual deformation and status skipped.
    fixed_mask, moving_mask : xarray.DataArray, optional
        Boolean ROI masks in their own native physical grids. A spatial moving
        mask broadcasts; per-frame moving masks follow the recording time labels.

    Returns
    -------
    RegistrationResult
        Calibrated original intensities, validity, diagnostics and optional fields.
    """
    from voxreg.rigid import _STATUS_LABELS, _build_output_template, _build_result

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
    matrices = build_world_matrices(initial, plane)
    if np.any(np.linalg.det(matrices[:, :3, :3]) <= 0):
        raise ValueError("SyN affine seeds must preserve orientation.")
    outputs, masks, forwards, inverses = [], [], [], []
    clocks, statuses, objectives = [], [], []
    inverse_errors, jacobians, nccs = [], [], []
    level_records = []
    metric_overlaps = []
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
                """Report owned CPU arrays with recording-wide frame indices."""
                metric_callback(
                    np.arange(first, last, dtype=np.int64),
                    steps.cpu().numpy().copy(),
                    loss.double().cpu().numpy().copy(),
                )

            start = perf_counter()
            metric_masks = None
            if fixed_mask is not None or moving_mask is not None:
                metric_masks = _MetricMasks(
                    _source(fixed_mask, device),
                    _source(moving_mask, device, first, last),
                    fixed,
                    plane,
                )
            fit_method = _fit_pyramid if schedule is not None else _fit
            pyramid_options = (
                {
                    "moving": moving.isel(time=slice(first, last)),
                    "fixed": fixed,
                    "schedule": schedule,
                    "records": level_records,
                }
                if schedule is not None
                else {}
            )
            forward, inverse, steps, status, objective, error = fit_method(
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
                **(
                    {
                        "active_frames": torch.tensor(
                            active_frames[first:last], device=device
                        )
                    }
                    if active_frames is not None
                    else {}
                ),
                **pyramid_options,
                **({"metric_masks": metric_masks} if metric_masks is not None else {}),
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
                        _build_world_field(forward, grid, matrices[first:last], fixed)
                        .cpu()
                        .numpy()
                    )
                    inverses.append(
                        _build_inverse_world_field(
                            inverse, grid, matrices[first:last], fixed, moving
                        )
                        .cpu()
                        .numpy()
                    )
                jacobians.append(
                    (
                        compute_jacobian_determinant(forward, grid)
                        .flatten(1)
                        .amin(dim=1)
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
                if metric_masks is not None:
                    fv = metric_masks.apply(fv, parameters, forward)
                    metric_overlaps.append(fv.float().flatten(1).mean(1).cpu().numpy())
                loss, meaningful = compute_global_ncc(fitted, stationary, fv)
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
    template = _build_output_template(moving, fixed, same_grid)
    values, validity = np.concatenate(outputs), np.concatenate(masks)
    transforms = inverse_transforms = None
    if return_transforms:
        forward, inverse = np.concatenate(forwards), np.concatenate(inverses)
        if planar:
            forward, inverse = forward[:, :, None], inverse[:, :, None]
        transforms = _build_labeled_field(
            forward, template, "fixed-world-to-moving-world pull"
        )
        inverse_transforms = _build_labeled_field(
            inverse, moving, "moving-world-to-fixed-world pull"
        )
    diagnostics = xr.Dataset(
        {
            "objective": ("time", np.concatenate(objectives)),
            "ncc_squared": ("time", np.concatenate(nccs)),
            "n_iterations": ("time", np.concatenate(clocks)),
            "status": ("time", _STATUS_LABELS[np.concatenate(statuses)]),
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
    if metric_overlaps:
        diagnostics["roi_valid_fraction"] = (
            "time",
            np.concatenate(metric_overlaps),
            {
                "domain": "native fixed image; endpoint ROI intersection with geometric support"
            },
        )
    if schedule is not None:
        from voxreg.pyramid import _add_level_diagnostics

        _add_level_diagnostics(diagnostics, level_records, *schedule)
    return _build_result(
        template,
        values,
        validity,
        transforms,
        diagnostics,
        inverse_transforms,
        spatial_input,
    )
