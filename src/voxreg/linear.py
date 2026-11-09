"""Batched translation, rigid and affine fitting on physical native grids."""

import warnings
from collections.abc import Callable
from dataclasses import replace
from time import perf_counter
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from confusius._utils.progress import progress_bar

from voxreg.convergence import WindowConvergence
from voxreg.interpolation import restore_image_dtype
from voxreg.masks import (
    MetricMasks,
    get_mask_source,
)
from voxreg.metrics import Metric, compute_global_ncc
from voxreg.preparation import PreparedRegistration
from voxreg.results import (
    STATUS_LABELS,
    LinearFit,
    RegistrationResult,
    build_output_template,
    build_result,
)
from voxreg.sampling import (
    Plane,
    SpatialGrid,
    build_world_matrices,
    make_grid,
    recover_initial_parameters,
    sample_rigid,
)
from voxreg.transforms import (
    build_linear_diagnostics,
    check_affine_parameters,
    configure_transform,
)


class _MetricOptions(TypedDict, total=False):
    """Type optional resident metric callback arguments."""

    metric_callback: Callable[[torch.Tensor, torch.Tensor], None] | None
    metric_interval: int
    active_frames: torch.Tensor
    metric_masks: MetricMasks
    metric_function: Metric


class _OptimizerOptions(_MetricOptions, total=False):
    """Type shared level-local optimizer controls and optional resident metrics."""

    convergence_window_size: int
    convergence_minimum_value: float
    initial_parameters: torch.Tensor | None


def _check_all_frames_stopped(status: torch.Tensor) -> bool:
    """Read whether a whole batch has stopped, synchronizing CUDA once.

    Parameters
    ----------
    status : torch.Tensor
        Per-frame optimizer status, where zero denotes active fitting.

    Returns
    -------
    bool
        Whether every frame has a terminal status.
    """
    return not (status == 0).any().item()


def _optimize(
    images: torch.Tensor,
    fixed: torch.Tensor,
    plane: Plane | SpatialGrid,
    reference: torch.Tensor,
    max_iterations: int,
    learning_rate: float,
    convergence_window_size: int = 50,
    convergence_minimum_value: float = 1e-6,
    *,
    initial_parameters: torch.Tensor | None = None,
    metric_callback: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    metric_interval: int = 10,
    active_frames: torch.Tensor | None = None,
    metric_masks: MetricMasks | None = None,
    metric_function: Metric | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run masked batched Adam with bounded whole-batch stopping checks.

    Parameters
    ----------
    images : torch.Tensor
        Batched fitting images.
    fixed : torch.Tensor
        Shared fixed fitting image.
    plane : Plane or SpatialGrid
        Resident fixed/moving sampling geometry.
    reference : torch.Tensor
        Boolean mask of frames that must stay exactly identity.
    max_iterations : int
        Maximum number of parameter updates.
    learning_rate : float
        Step size in physically scaled parameters.
    convergence_window_size : int
        ITK energy-profile window; larger than the iteration budget disables it.
    convergence_minimum_value : float
        Stop when the normalized improvement trend is at or below this threshold.
    initial_parameters : torch.Tensor, optional
        Starting physical parameters; Adam state starts fresh, not from a previous fit.
    metric_callback : callable, optional
        Resident callback receiving per-frame clocks and trial objective values.
    metric_interval : int, default: 10
        Number of optimizer rounds between callback reports.
    active_frames : torch.Tensor, optional
        Per-frame eligibility inherited from preceding stages.
    metric_masks : MetricMasks, optional
        Physical fixed and moving ROIs on the current metric domain.
    metric_function : Metric, optional
        Explicit per-frame loss; omitted uses global squared NCC.

    Returns
    -------
    parameters : torch.Tensor
        Best valid radian/mm parameters per frame.
    steps : torch.Tensor
        Number of optimizer updates per frame.
    status : torch.Tensor
        1 converged, 2 insufficient overlap, 3 invalid metric, 4 reference, 5 limit,
        6 invalid affine transform, 7 skipped after an earlier stage failure.
    """
    objective = compute_global_ncc if metric_function is None else metric_function
    b = len(images)
    u = (
        plane.parameter_scale.new_zeros((b, len(plane.parameter_scale)))
        if initial_parameters is None
        else initial_parameters / plane.parameter_scale
    ).requires_grad_(True)
    spatial_axes = tuple(range(1, images.ndim))
    best = u.detach().clone()
    best_loss = u.new_full((b,), torch.inf)
    moment, variance = torch.zeros_like(u), torch.zeros_like(u)
    steps = torch.zeros(b, dtype=torch.int64, device=images.device)
    monitor = (
        WindowConvergence(b, convergence_window_size, images.device)
        if convergence_window_size <= max_iterations
        else None
    )
    constant = images.amax(dim=spatial_axes) == images.amin(dim=spatial_axes)
    status = torch.where(reference, 4, torch.where(constant, 3, 0))
    if metric_masks is not None:
        status = torch.where(reference, 4, 0)
    if active_frames is not None:
        status = torch.where(~active_frames & ~reference, 7, status)
    check_interval = 10 if images.is_cuda else 1
    orientation = (
        check_affine_parameters(u.detach() * plane.parameter_scale, plane)[1]
        if plane.transform == "affine"
        else None
    )
    for iteration in range(max_iterations):
        moved, valid = sample_rigid(images, u * plane.parameter_scale, plane)
        if metric_masks is not None:
            valid = metric_masks.apply(valid, u * plane.parameter_scale)
        loss, meaningful = objective(moved, fixed, valid)
        if metric_callback is not None and iteration % metric_interval == 0:
            metric_callback(
                steps, torch.where(meaningful & ~constant, loss.detach(), torch.nan)
            )
        overlap = valid.sum(dim=spatial_axes) > 1
        good = overlap & meaningful & torch.isfinite(loss)
        active = status == 0
        with torch.no_grad():
            if orientation is not None:
                admissible, sign = check_affine_parameters(
                    u.detach() * plane.parameter_scale, plane
                )
                status = torch.where(
                    active & (~admissible | (sign != orientation)), 6, status
                )
                active = status == 0
                good = good & admissible & (sign == orientation)
            status = torch.where(active & ~overlap, 2, status)
            status = torch.where(active & overlap & ~good, 3, status)
            improved = active & good & (loss < best_loss)
            best = torch.where(improved[:, None], u.detach(), best)
            best_loss = torch.where(improved, loss.detach(), best_loss)
        active = status == 0
        gradient = torch.autograd.grad(torch.where(active, loss, 0).sum(), u)[0]
        with torch.no_grad():
            status = torch.where(
                active & ~torch.isfinite(gradient).all(dim=1), 3, status
            )
            active = status == 0
            if monitor is not None:
                value = monitor.update(loss, active)
                status = torch.where(
                    active & (value <= convergence_minimum_value), 1, status
                )
                active = status == 0
            gradient = torch.nan_to_num(gradient)
            steps = steps + active
            moment = torch.where(active[:, None], 0.9 * moment + 0.1 * gradient, moment)
            variance = torch.where(
                active[:, None], 0.999 * variance + 0.001 * gradient.square(), variance
            )
            clock = steps.clamp_min(1).to(u.dtype)[:, None]
            delta = (
                learning_rate
                * (moment / (1 - 0.9**clock))
                / ((variance / (1 - 0.999**clock)).sqrt() + 1e-8)
            )
            candidate = u - torch.where(active[:, None], delta, 0)
        u = candidate.detach().requires_grad_(True)
        if (
            monitor is not None
            and iteration + 1 >= convergence_window_size
            and (iteration + 1 - convergence_window_size) % check_interval == 0
            and iteration + 1 < max_iterations
            and _check_all_frames_stopped(status)
        ):
            break
    with torch.no_grad():
        moved, valid = sample_rigid(images, u * plane.parameter_scale, plane)
        if metric_masks is not None:
            valid = metric_masks.apply(valid, u * plane.parameter_scale)
        loss, meaningful = objective(moved, fixed, valid)
        if metric_callback is not None:
            metric_callback(steps, torch.where(meaningful & ~constant, loss, torch.nan))
        overlap = valid.sum(dim=spatial_axes) > 1
        good = overlap & meaningful & torch.isfinite(loss)
        eligible = (status == 0) | (status == 1)
        if orientation is not None:
            admissible, sign = check_affine_parameters(u * plane.parameter_scale, plane)
            status = torch.where(
                eligible & (~admissible | (sign != orientation)), 6, status
            )
            eligible = (status == 0) | (status == 1)
            good = good & admissible & (sign == orientation)
        improved = eligible & good & (loss < best_loss)
        best = torch.where(improved[:, None], u, best)
        status = torch.where(eligible & ~overlap, 2, status)
        status = torch.where(eligible & overlap & ~good, 3, status)
        status = torch.where(status == 0, 5, status)
    return best * plane.parameter_scale, steps, status


def _optimize_pyramid(
    images: torch.Tensor,
    fixed: torch.Tensor,
    plane: Plane | SpatialGrid,
    reference: torch.Tensor,
    max_iterations: int,
    learning_rate: float,
    *,
    moving: xr.DataArray,
    fixed_data: xr.DataArray,
    schedule: tuple[tuple[int, ...], tuple[float, ...], tuple[int, ...]],
    records: list[dict[str, npt.NDArray]],
    convergence_window_size: int = 50,
    convergence_minimum_value: float = 1e-6,
    initial_parameters: torch.Tensor | None = None,
    metric_callback: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    metric_interval: int = 10,
    active_frames: torch.Tensor | None = None,
    metric_masks: MetricMasks | None = None,
    metric_function: Metric | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refine complete world matrices with fresh per-level optimizer state.

    Parameters
    ----------
    images, fixed : torch.Tensor
        Native moving frames and shared fixed fitting image.
    plane : Plane or SpatialGrid
        Native physical sampling geometry and model.
    reference : torch.Tensor
        Frames preserved exactly as the implicit reference.
    max_iterations : int
        Native budget retained in the common fitting call signature.
    learning_rate : float
        Step size in physically scaled parameters.
    moving, fixed_data : xarray.DataArray
        Original native grids used to build pyramid levels.
    schedule : tuple
        Validated factors, physical smoothing sigmas and per-level budgets.
    records : list of dict
        Batch-local level outcomes appended in place.
    convergence_window_size : int, default: 50
        Level-local objective-history window.
    convergence_minimum_value : float, default: 1e-6
        Threshold for the normalized objective trend.
    initial_parameters : torch.Tensor, optional
        Complete centered physical seed on the native grid.
    metric_callback : callable, optional
        Resident per-frame clock and trial-objective reporter.
    metric_interval : int, default: 10
        Number of rounds between reports.
    active_frames : torch.Tensor, optional
        Frames eligible to continue after preceding stages.
    metric_masks : MetricMasks, optional
        Native physical ROI sources re-evaluated on every level.
    metric_function : Metric, optional
        Selected per-frame loss; omitted uses global squared NCC.

    Returns
    -------
    physical : torch.Tensor
        Complete final centered physical parameters on the native grid.
    steps : torch.Tensor
        Total updates per frame across levels.
    status : torch.Tensor
        Final level status per frame.
    """
    from voxreg.pyramid import make_level

    objective = compute_global_ncc if metric_function is None else metric_function
    factors, sigmas, budgets = schedule
    physical = (
        plane.parameter_scale.new_zeros((len(images), len(plane.parameter_scale)))
        if initial_parameters is None
        else initial_parameters
    )
    matrices = build_world_matrices(physical.cpu().numpy(), plane)
    level_steps, level_status, level_objectives = [], [], []
    for factor, sigma, budget in zip(factors, sigmas, budgets, strict=True):
        level_images, level_moving = make_level(images, moving, factor, sigma)
        level_fixed, level_data = make_level(fixed[None], fixed_data, factor, sigma)
        geometry = configure_transform(
            make_grid(
                level_data,
                level_moving,
                device=str(images.device),
                center=plane.center,
                plane_seed=matrices,
            ),
            level_data,
            plane.transform,
        )
        seed = torch.tensor(
            recover_initial_parameters(matrices, geometry, len(images)),
            device=images.device,
        )
        level_masks = (
            None
            if metric_masks is None
            else MetricMasks(*metric_masks.sources, level_data, geometry)
        )
        level_options = _MetricOptions(
            metric_callback=metric_callback, metric_interval=metric_interval
        )
        if metric_function is not None:
            level_options["metric_function"] = metric_function
        if active_frames is not None:
            level_options["active_frames"] = active_frames
        if level_masks is not None:
            level_options["metric_masks"] = level_masks
        physical, steps, status = _optimize(
            level_images,
            level_fixed[0],
            geometry,
            reference,
            budget,
            learning_rate,
            convergence_window_size=convergence_window_size,
            convergence_minimum_value=convergence_minimum_value,
            initial_parameters=seed,
            **level_options,
        )
        with torch.no_grad():
            fitted, valid = sample_rigid(level_images, physical, geometry)
            if level_masks is not None:
                valid = level_masks.apply(valid, physical)
            loss, meaningful = objective(fitted, level_fixed[0], valid)
            nonconstant = level_images.flatten(1).amax(1) > level_images.flatten(
                1
            ).amin(1)
            level_objectives.append(
                torch.where(meaningful & nonconstant, loss, torch.nan).cpu().numpy()
            )
        level_steps.append(steps.cpu().numpy())
        level_status.append(STATUS_LABELS[status.cpu().numpy()])
        matrices = build_world_matrices(physical.cpu().numpy(), geometry)
    records.append(
        {
            "n_iterations": np.stack(level_steps, axis=1),
            "status": np.stack(level_status, axis=1),
            "objective": np.stack(level_objectives, axis=1),
        }
    )
    physical = torch.tensor(
        recover_initial_parameters(matrices, plane, len(images)), device=images.device
    )
    # Preserve an implicit reference exactly despite matrix reparameterization roundoff.
    physical = torch.where(reference[:, None], 0, physical)
    return (
        physical,
        torch.as_tensor(np.sum(level_steps, axis=0), device=images.device),
        status,
    )


def run_linear(
    prepared: PreparedRegistration, *, fit_only: bool = False
) -> RegistrationResult | LinearFit:
    """Fit a prepared linear model and assemble native-grid outputs.

    Parameters
    ----------
    prepared : PreparedRegistration
        Validated native images, physical geometry, seed and resolved settings.
    fit_only : bool, default: False
        Whether to return only linear state and diagnostics for a later stage.

    Returns
    -------
    RegistrationResult or LinearFit
        Resampled original intensities and maps, or the complete linear handoff.
    """
    settings = prepared.settings
    batch_size = settings.batch_size
    convergence_minimum_value = settings.convergence_minimum_value
    convergence_window_size = settings.convergence_window_size
    device = settings.device
    fixed_mask = settings.fixed_mask
    initialization = settings.initialization
    interpolation = settings.resample_interpolation
    learning_rate = settings.learning_rate
    metric = settings.metric
    metric_callback = settings.metric_callback
    metric_interval = settings.metric_interval
    mi_bins = settings.mi_bins
    mi_sigma = settings.mi_sigma
    moving_mask = settings.moving_mask
    progress = settings.progress
    return_transforms = settings.return_transforms
    active_frames = prepared.active_frames
    budgets = prepared.budgets
    factors = prepared.factors
    fit = prepared.fit
    fixed = prepared.fixed
    implicit_fixed = prepared.implicit_fixed
    initial = prepared.initial
    metric_function = prepared.metric_function
    moving = prepared.moving
    multiresolution = prepared.multiresolution
    plane = prepared.plane
    same_grid = prepared.same_grid
    sigmas = prepared.sigmas
    spatial_input = prepared.spatial_input
    target = prepared.target
    timings = prepared.timings
    transform = prepared.transform
    assert batch_size is not None
    planar = isinstance(plane, Plane) or plane.planar
    raw = moving.values[:, 0] if planar else moving.values
    spatial_axes = tuple(range(1, raw.ndim))
    reference_shape = (slice(None),) + (None,) * len(spatial_axes)
    integer_output = np.issubdtype(raw.dtype, np.integer)
    resampling_dtype = (
        np.float64 if integer_output or raw.dtype.itemsize > 4 else np.float32
    )
    max_iterations = budgets[0]
    outputs, masks, parameters = [], [], []
    clocks, statuses, objectives, overlaps = [], [], [], []
    level_records = []
    metric_overlaps = []
    with progress_bar("Registering frames", len(raw), show=progress is True) as advance:
        for first in range(0, len(raw), batch_size):
            last = min(first + batch_size, len(raw))
            if device == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            images = torch.tensor(fit[first:last], device=device)
            target_tensor = torch.tensor(target, device=device)
            reference = torch.zeros(last - first, dtype=torch.bool, device=device)
            if implicit_fixed and first == 0:
                reference[0] = True
            initial_tensor = (
                torch.tensor(initial[first:last], device=device)
                if initial is not None
                else None
            )
            if device == "cuda":
                torch.cuda.synchronize()
            timings["upload"] += perf_counter() - start
            start = perf_counter()
            metric_options: _MetricOptions = {}
            if metric_function is not None:
                metric_options["metric_function"] = metric_function
            metric_masks = None
            if fixed_mask is not None or moving_mask is not None:
                metric_masks = MetricMasks(
                    get_mask_source(fixed_mask, device),
                    get_mask_source(moving_mask, device, first, last),
                    fixed,
                    plane,
                )
                metric_options["metric_masks"] = metric_masks
            if active_frames is not None:
                metric_options["active_frames"] = torch.tensor(
                    active_frames[first:last], device=device
                )
            if metric_callback is not None:

                def report_metrics(clocks, objective, first=first, last=last):
                    """Report owned CPU arrays with recording-wide frame indices."""
                    metric_callback(
                        np.arange(first, last, dtype=np.int64),
                        clocks.detach().cpu().numpy().copy(),
                        objective.detach().double().cpu().numpy().copy(),
                    )

                metric_options.update(
                    {
                        "metric_callback": report_metrics,
                        "metric_interval": metric_interval,
                    }
                )
            optimizer_options = _OptimizerOptions(
                convergence_window_size=convergence_window_size,
                convergence_minimum_value=convergence_minimum_value,
                initial_parameters=initial_tensor,
                **metric_options,
            )
            if multiresolution:
                physical, steps, status = _optimize_pyramid(
                    images,
                    target_tensor,
                    plane,
                    reference,
                    max_iterations,
                    learning_rate,
                    moving=moving.isel(time=slice(first, last)),
                    fixed_data=fixed,
                    schedule=(factors, sigmas, budgets),
                    records=level_records,
                    **optimizer_options,
                )
            else:
                physical, steps, status = _optimize(
                    images,
                    target_tensor,
                    plane,
                    reference,
                    max_iterations,
                    learning_rate,
                    **optimizer_options,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            timings["optimize"] += perf_counter() - start
            start = perf_counter()
            with torch.no_grad():
                if not fit_only:
                    original = torch.tensor(
                        np.ascontiguousarray(raw[first:last], dtype=resampling_dtype),
                        device=device,
                    )
                    sampled, valid = sample_rigid(
                        original, physical, plane, interpolation=interpolation
                    )
                    fills = original.amin(dim=spatial_axes)[reference_shape]
                    sampled = torch.where(valid, sampled, fills)
                    if implicit_fixed:
                        sampled = torch.where(
                            reference[reference_shape], original, sampled
                        )
                        valid = valid | reference[reference_shape]
                fitted, fit_valid = sample_rigid(images, physical, plane)
                metric_valid = (
                    fit_valid
                    if metric_masks is None
                    else metric_masks.apply(fit_valid, physical)
                )
                objective = (
                    compute_global_ncc if metric_function is None else metric_function
                )
                loss, meaningful = objective(fitted, target_tensor, metric_valid)
                if metric_masks is not None:
                    metric_overlaps.append(
                        metric_valid.float().mean(dim=spatial_axes).cpu().numpy()
                    )
                nonconstant = images.amax(dim=spatial_axes) > images.amin(
                    dim=spatial_axes
                )
                loss = torch.where(meaningful & nonconstant, loss, torch.nan)
                if fit_only:
                    valid = fit_valid
                else:
                    values = sampled.cpu().numpy()
                    outputs.append(
                        restore_image_dtype(values, raw.dtype, interpolation)
                    )
                masks.append(valid.cpu().numpy())
                parameters.append(physical.cpu().numpy())
                clocks.append(steps.cpu().numpy())
                statuses.append(status.cpu().numpy())
                objectives.append(loss.cpu().numpy())
                overlaps.append(valid.float().mean(dim=spatial_axes).cpu().numpy())
            timings["resample_download"] += perf_counter() - start
            if progress is True:
                advance(last - first)
            elif callable(progress):
                progress(last, len(raw))
    params = np.concatenate(parameters)
    validity = np.concatenate(masks)
    values = None if fit_only else np.concatenate(outputs)
    diagnostic_params, diagnostic_plane = params, plane
    mixed = isinstance(plane, SpatialGrid) and plane.slice_basis is not None
    matrices = (
        build_world_matrices(params, plane)
        if fit_only or return_transforms or (mixed and transform == "affine")
        else None
    )
    if mixed and transform == "affine":
        assert matrices is not None
        shifts = matrices[:, :3, 3] + np.einsum(
            "bij,j->bi", matrices[:, :3, :3] - np.eye(3), plane.center
        )
        diagnostic_params = np.column_stack(
            ((matrices[:, :3, :3] - np.eye(3)).reshape(-1, 9), shifts)
        )
        diagnostic_plane = replace(plane, slice_basis=None, resident_slice_basis=None)
    if transform != "rigid":
        motion = build_linear_diagnostics(diagnostic_params, diagnostic_plane)
    elif planar:
        motion = {
            "angle": (
                "time",
                diagnostic_params[:, 0],
                {"units": "rad", "basis": "intrinsic i/j plane"},
            ),
            **{
                f"translation_{axis}": (
                    "time",
                    diagnostic_params[:, column],
                    {
                        "units": "mm",
                        "meaning": f"center displacement along {direction}",
                    },
                )
                for column, axis, direction in (
                    (1, "i", "intrinsic unit i"),
                    (2, "j", "orthogonalized intrinsic j"),
                )
            },
        }
    else:
        motion = {
            name: (
                "time",
                diagnostic_params[:, column],
                {"units": "rad" if column < 3 else "mm", "basis": "world z/y/x"},
            )
            for column, name in enumerate(
                (
                    "rotation_z",
                    "rotation_y",
                    "rotation_x",
                    "translation_z",
                    "translation_y",
                    "translation_x",
                )
            )
        }
    diagnostics = xr.Dataset(
        {
            "objective": ("time", np.concatenate(objectives)),
            "valid_fraction": ("time", np.concatenate(overlaps)),
            "n_iterations": ("time", np.concatenate(clocks)),
            "status": ("time", STATUS_LABELS[np.concatenate(statuses)]),
            **motion,
        },
        coords={"time": moving.time},
        attrs={
            "timings_seconds": timings,
            **(
                {"transform": transform, "center_zyx_mm": plane.center.tolist()}
                if transform != "rigid"
                else {}
            ),
            **(
                {
                    "initialization": initialization,
                    "rotation_center_zyx_mm": plane.center.tolist(),
                }
                if initialization is not None
                else {}
            ),
        },
    )
    diagnostics.attrs["batch_size"] = batch_size
    diagnostics.attrs["metric"] = (
        "negative mutual information"
        if metric == "mi"
        else "negative squared global NCC"
    )
    if metric == "mi":
        diagnostics.attrs.update(
            mi_bins=mi_bins,
            mi_sigma_bin_widths=mi_sigma,
            mi_kernel="gaussian",
            objective_units="nats",
            mi_range="native per-image min/max mapped to [0, 1] before pyramid smoothing",
        )
    if metric_overlaps:
        diagnostics["roi_valid_fraction"] = (
            "time",
            np.concatenate(metric_overlaps),
            {"domain": "native fixed image; ROI intersection with geometric support"},
        )
    if multiresolution:
        from voxreg.pyramid import add_level_diagnostics

        add_level_diagnostics(diagnostics, level_records, factors, sigmas, budgets)
    if isinstance(plane, SpatialGrid) and plane.slice_basis is not None:
        diagnostics.attrs.update(
            metric_domain="fixed_slice" if fixed.sizes["k"] == 1 else "fixed_volume",
            parameter_convention="fixed-world-to-moving-world centered map; sampled on native fixed grid",
            optimized_parameters=len(plane.parameter_scale),
            rotation_center_zyx_mm=plane.center.tolist(),
        )
        if moving.sizes["k"] == 1:
            diagnostics["output_valid_fraction"] = (
                "time",
                validity.reshape(len(raw), -1).mean(axis=1),
            )
            if np.any(diagnostics.output_valid_fraction.values == 0):
                warnings.warn(
                    "No fixed-grid sample centers lie within the transformed slice's voxel extent for some frames. Fitting uses the native fixed grid; try a better initialization or choose the slice as fixed.",
                    UserWarning,
                    stacklevel=2,
                )
    if isinstance(plane, SpatialGrid) and plane.plane_rotation is not None:
        diagnostics.attrs["plane_alignment"] = (
            "fixed-to-moving plane geometry; only in-plane motion optimized"
        )
    if fit_only:
        assert matrices is not None
        return LinearFit(matrices, diagnostics, plane.center.copy())
    assert values is not None
    axes = ["z", "y", "x", "1"]
    transforms = (
        xr.DataArray(
            matrices,
            dims=("time", "world_out", "world_in"),
            coords={"time": moving.time, "world_out": axes, "world_in": axes},
            attrs={
                "convention": "fixed-world-to-moving-world pull",
                "spatial_units": "mm",
            },
        )
        if return_transforms
        else None
    )
    inverse_transforms = None
    if transforms is not None:
        assert matrices is not None
        inverse_transforms = transforms.copy(data=np.linalg.inv(matrices))
        inverse_transforms.attrs["convention"] = "moving-world-to-fixed-world pull"
    template = build_output_template(moving, fixed, same_grid)
    return build_result(
        template,
        values,
        validity,
        transforms,
        diagnostics,
        inverse_transforms,
        spatial_input,
    )
