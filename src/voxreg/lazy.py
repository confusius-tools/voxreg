"""Bounded time-task graphs sharing eager registration and resampling outputs."""

from __future__ import annotations

import socket
import threading
from collections.abc import Generator, Hashable
from contextlib import contextmanager
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from filelock import FileLock

from voxreg.geometry import VOXEL_DIMS, validate_registration_geometry
from voxreg.interpolation import Interpolation
from voxreg.preparation import RegistrationSettings, resolve_device
from voxreg.progress import enable_lazy_progress
from voxreg.pyramid import validate_schedule
from voxreg.results import assemble_result_tree, build_output_template
from voxreg.stages import Stage, resolve_stages

_LOCAL_EXECUTION_LOCK = threading.Lock()
"""Conservative one-fitting-task-at-a-time limit for local threaded execution."""


def has_lazy_data(*values: Any) -> bool:
    """Detect Dask spatial data without coercing arrays.

    Parameters
    ----------
    *values : object
        Public image, mask or transform arguments.

    Returns
    -------
    bool
        Whether any argument has a Dask array backend.
    """
    return any(
        isinstance(value.data if isinstance(value, xr.DataArray) else value, da.Array)
        for value in values
    )


@contextmanager
def _execution_slot(device: str) -> Generator[None]:
    """Bound fitting concurrency on a host or a physical GPU across workers.

    Parameters
    ----------
    device : str
        Resolved CPU or CUDA choice.

    Yields
    ------
    None
        Exclusive execution slot, released on errors as well as success.
    """
    from distributed import Semaphore, get_worker

    identity = f"cpu-{socket.gethostname()}"
    if device == "cuda":
        import torch

        identity = str(
            torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
        )
    try:
        get_worker()
    except ValueError:
        # A file lease also bounds local multiprocessing, not just worker threads.
        directory = Path.home() / ".cache" / "voxreg" / "execution"
        directory.mkdir(parents=True, exist_ok=True)
        with _LOCAL_EXECUTION_LOCK, FileLock(directory / f"{identity}.lock"):
            yield
        return
    # One lease per host CPU or physical GPU avoids oversubscribing fitting state.
    with Semaphore(max_leases=1, name=f"voxreg-{identity}"):
        yield


def _check_chunks(data: Any, limit: int) -> None:
    """Reject source time blocks larger than the eager task memory contract.

    Parameters
    ----------
    data : object
        Optional image, mask or transform.
    limit : int
        Maximum source frames admitted by a time-task chunk.

    Returns
    -------
    None
        Acceptable sources need no rechunking.

    Raises
    ------
    ValueError
        If a source time block exceeds the task limit.
    """
    oversized = (
        isinstance(data, xr.DataArray)
        and isinstance(data.data, da.Array)
        and "time" in data.dims
        and max(data.data.chunks[data.get_axis_num("time")]) > limit
    ) or (isinstance(data, da.Array) and data.ndim == 3 and max(data.chunks[0]) > limit)
    if oversized:
        raise ValueError(
            "Auxiliary Dask source time chunks must not exceed the largest primary task chunk. Align chunks at the data source; splitting a large stored block does not avoid loading it."
        )


def _task_ranges(
    frames: int, batch_size: int | None, *sources: Any
) -> list[tuple[int, int]]:
    """Follow primary temporal source chunks independently of optimizer batches.

    Parameters
    ----------
    frames : int
        Total recording frames.
    batch_size : int, optional
        Eager fitting override, also used as a task fallback without time chunks.
    *sources : object
        Sources in priority order: moving data, then masks or transforms.

    Returns
    -------
    list of tuple of int
        Half-open time ranges matching the first time-chunked source. Without
        temporal Dask chunks, use the explicit batch size or one frame.

    Raises
    ------
    ValueError
        If auxiliary time chunks exceed the largest primary task chunk.
    """
    chunks = None
    for source in sources:
        if isinstance(source, xr.DataArray):
            if isinstance(source.data, da.Array) and "time" in source.dims:
                chunks = source.data.chunks[source.get_axis_num("time")]
                break
        elif isinstance(source, da.Array) and source.ndim == 3:
            chunks = source.chunks[0]
            break
    if chunks is None:
        limit = batch_size or 1
        chunks = tuple(min(limit, frames - first) for first in range(0, frames, limit))
    for source in sources:
        _check_chunks(source, max(chunks))
    ranges = []
    first = 0
    for chunk in chunks:
        if not chunk:
            continue
        last = first + int(chunk)
        ranges.append((first, last))
        first = last
    return ranges


def _select_frames(value: Any, first: int, last: int) -> Any:
    """Select only required recording masks or seed matrices.

    Parameters
    ----------
    value : object
        Optional shared or per-frame input.
    first, last : int
        Half-open global frame range.

    Returns
    -------
    object
        Shared input unchanged, or a bounded frame selection.
    """
    if isinstance(value, xr.DataArray):
        return value.isel(time=slice(first, last)) if "time" in value.dims else value
    if value is not None and not isinstance(value, str) and np.ndim(value) == 3:
        return value[first:last]
    return value


def _restore_data(
    template: xr.Dataset,
    dims: tuple[Hashable, ...],
    name: Hashable | None,
    encoding: dict[str, Any],
    values: np.ndarray,
) -> xr.DataArray:
    """Attach metadata to one materialized bounded input block.

    Parameters
    ----------
    template : xarray.Dataset
        Coordinates, indexes and attributes without a signal variable.
    dims : tuple of hashable
        Original dimension order, including dimensions without coordinates.
    name : hashable, optional
        Original array name.
    encoding : dict
        Original array encoding.
    values : numpy.ndarray
        One complete bounded task block.

    Returns
    -------
    xarray.DataArray
        Eager VoxelData retaining the original index and metadata.
    """
    result = xr.DataArray(
        values, coords=template.coords, dims=dims, name=name, attrs=template.attrs
    )
    result.encoding = encoding.copy()
    return result


def _eager_dependency(value: Any) -> Any:
    """Materialize a bounded input while preserving upstream shared graph keys.

    Parameters
    ----------
    value : object
        Optional bounded image, mask or transform selection.

    Returns
    -------
    object
        Shared Delayed input, or an unchanged eager value.
    """
    data = value.data if isinstance(value, xr.DataArray) else value
    if not isinstance(data, da.Array):
        return value
    # Xarray-to-delayed conversion optimizes each collection independently,
    # cloning shared fits; raw blocks keep their original task identities.
    block = data.rechunk(data.shape).to_delayed(optimize_graph=False).item()
    if isinstance(value, xr.DataArray):
        # Broadcast NumPy placeholders expand to full dense buffers when serialized.
        template = xr.Dataset(coords=value.coords, attrs=value.attrs)
        return delayed(_restore_data)(
            template, value.dims, value.name, value.encoding, block
        )
    return block


def _run_registration_batch(
    moving: xr.DataArray,
    fixed: xr.DataArray,
    specification: Any,
    options: dict[str, Any],
    first: int,
    total: int,
    implicit: bool,
) -> xr.DataTree:
    """Fit one bounded eager batch at execution time.

    Parameters
    ----------
    moving, fixed : xarray.DataArray
        Eager task inputs, including a shared fixed dependency.
    specification : object
        Complete transform pipeline.
    options : dict
        Selected shared settings.
    first, total : int
        Global frame offset and recording length for callbacks.
    implicit : bool
        Whether the global first frame is the reference.

    Returns
    -------
    xarray.DataTree
        One shared result for every lazy projection.
    """
    from voxreg.registration import execute_registration

    device = resolve_device(options["device"])
    settings = RegistrationSettings(**options)
    callback = settings.metric_callback
    progress = settings.progress
    if callback is not None:

        def report(indices, steps, objectives):
            """Report task-time metrics with global frame indices."""
            callback(indices + first, steps, objectives)

        settings = replace(settings, metric_callback=report)
    if callable(progress):

        def report_progress(completed, count):
            """Report task-local completion with a global completed-frame offset."""
            local = len(moving.time)
            stage_count = (
                len(specification) if isinstance(specification, (tuple, list)) else 1
            )
            stage_index = max(0, completed - 1) // local
            progress(
                stage_index * total + first + completed - stage_index * local,
                total * stage_count,
            )

        settings = replace(settings, progress=report_progress)
    else:
        settings = replace(settings, progress=False)
    if implicit and first:
        settings = replace(
            settings,
            fixed_intensity_scaling=settings.moving_intensity_scaling,
            fixed_winsorize_percentiles=settings.moving_winsorize_percentiles,
        )
    settings = replace(settings, device=device, allow_zero_moving=True)
    if implicit and first == 0 and moving.sizes["time"] == 1:
        moving = moving.copy(data=fixed.values[None])
    with _execution_slot(device):
        return execute_registration(
            moving, None if implicit and first == 0 else fixed, specification, settings
        )


def _registration_values(result: xr.DataTree, name: str) -> np.ndarray:
    """Project one eager spatial result without repeating fitting.

    Parameters
    ----------
    result : xarray.DataTree
        Shared batch result.
    name : str
        Image, mask or map attribute.

    Returns
    -------
    numpy.ndarray
        Bounded eager values for a Dask output block.
    """
    path = {
        "registered": "images/registered",
        "valid": "images/valid",
        "forward_transforms": "forward/transform",
        "inverse_transforms": "inverse/transform",
    }[name]
    group, variable = path.split("/")
    value = result[group].to_dataset()[variable]
    if "world_component" in value.dims:
        value = value.transpose("time", *VOXEL_DIMS, "world_component")
    return value.values


def _diagnostic_values(
    result: xr.DataTree, name: str, stage: int | None, first: int
) -> np.ndarray:
    """Project lazy diagnostics, including task execution metadata.

    Parameters
    ----------
    result : xarray.DataTree
        Shared eager batch result.
    name : str
        Diagnostic variable or execution-metadata field.
    stage : int, optional
        Stage index; not provided selects the summary.
    first : int
        Global batch offset, used to identify repeated batch timings.

    Returns
    -------
    numpy.ndarray
        Per-frame values or bounded level diagnostics.
    """
    data = result["diagnostics" if stage is None else f"stages/{stage}"].to_dataset()
    count = data.sizes["time"]
    if name in data:
        values = data[name].values
        dtype = (
            np.float64
            if values.dtype.kind == "f"
            else "<U32"
            if values.dtype.kind in "US"
            else np.int64
        )
        return values.astype(dtype, copy=False)
    if name == "execution_device":
        return np.full(count, data.attrs["device"], dtype="<U4")
    if name == "execution_task_size":
        return np.full(count, count, dtype=np.int64)
    if name == "execution_batch_size":
        limit = data.attrs["batch_size"]
        return np.minimum(limit, count - np.arange(count) // limit * limit).astype(
            np.int64
        )
    if name == "execution_batch_start":
        return np.full(count, first, dtype=np.int64)
    if name == "batch_stage_overhead_seconds":
        return np.full(count, data.attrs["stage_overhead_seconds"], dtype=np.float64)
    if name == "rotation_center_zyx_mm":
        center = data.attrs.get(name, data.attrs.get("center_zyx_mm"))
        return np.broadcast_to(center, (count, 3)).copy()
    return np.full(
        count,
        data.attrs["timings_seconds"][
            name.removeprefix("batch_").removesuffix("_seconds")
        ],
        dtype=np.float64,
    )


def _diagnostics(
    tasks: list[Any],
    ranges: list[tuple[int, int]],
    moving: xr.DataArray,
    fixed: xr.DataArray,
    model: str,
    settings: RegistrationSettings,
    stage: int | None,
    *,
    stage_summary: bool = False,
) -> xr.Dataset:
    """Build a data-free diagnostic schema whose variables project shared tasks.

    Parameters
    ----------
    tasks : list
        Shared eager fits.
    ranges : list of tuple of int
        Half-open frame ranges.
    moving, fixed : xarray.DataArray
        Metadata defining time labels and dimensionality.
    model : str
        Effective model for this dataset.
    settings : RegistrationSettings
        Effective stage settings.
    stage : int, optional
        Stage projection or pipeline summary.
    stage_summary : bool, default: False
        Whether to include orchestration overhead for a staged summary.

    Returns
    -------
    xarray.Dataset
        Lazy per-frame, stage and level diagnostics with static schema metadata.
    """
    factors, sigmas, budgets = validate_schedule(
        settings.shrink_factors, settings.smoothing_sigmas, settings.max_iterations
    )
    schema: dict[str, tuple[tuple[str, ...], Any]] = {
        "objective": (("time",), np.float64),
        "valid_fraction": (("time",), np.float64),
        "n_iterations": (("time",), np.int64),
        "status": (("time",), "<U32"),
        "execution_device": (("time",), "<U4"),
        "execution_batch_size": (("time",), np.int64),
        "execution_task_size": (("time",), np.int64),
        "execution_batch_start": (("time",), np.int64),
    }
    if stage_summary:
        schema["batch_stage_overhead_seconds"] = (("time",), np.float64)
    for phase in ("prepare", "upload", "optimize", "resample_download"):
        schema[f"batch_{phase}_seconds"] = (("time",), np.float64)
    planar = moving.sizes["k"] == fixed.sizes["k"] == 1
    mixed = (moving.sizes["k"] == 1) != (fixed.sizes["k"] == 1)
    if model == "syn":
        for name in ("ncc_squared", "min_jacobian", "inverse_error"):
            schema[name] = (("time",), np.float64)
    else:
        axes = ("i", "j") if planar else ("z", "y", "x")
        if model == "rigid":
            for name in ["angle"] if planar else [f"rotation_{axis}" for axis in axes]:
                schema[name] = (("time",), np.float64)
        if model == "affine":
            for row in axes:
                for col in axes:
                    schema[f"linear_{row}{col}"] = (("time",), np.float64)
        for axis in axes:
            schema[f"translation_{axis}"] = (("time",), np.float64)
        if model != "rigid" or mixed or isinstance(settings.initialization, str):
            schema["rotation_center_zyx_mm"] = (("time", "world_component"), np.float64)
        if mixed and moving.sizes["k"] == 1:
            schema["output_valid_fraction"] = (("time",), np.float64)
    if settings.fixed_mask is not None or settings.moving_mask is not None:
        schema["roi_valid_fraction"] = (("time",), np.float64)
    multiresolution = factors != (1,) or sigmas != (0.0,)
    if multiresolution:
        for name, dtype in (
            ("objective", np.float64),
            ("status", "<U32"),
            ("n_iterations", np.int64),
        ):
            schema[f"level_{name}"] = (("time", "level"), dtype)
    sizes = {"level": len(factors), "world_component": 3}
    # One layer per variable avoids constructing and merging a graph per block.
    prefix = f"voxreg-diagnostics-{uuid4().hex}"
    base = HighLevelGraph.from_collections(prefix, {}, dependencies=tasks)
    source_layers = base.dependencies[prefix]
    layers = {key: value for key, value in base.layers.items() if key != prefix}
    dependencies = {
        key: value for key, value in base.dependencies.items() if key != prefix
    }
    for name, (dims, _) in schema.items():
        label = f"{prefix}-{name}"
        layers[label] = MaterializedLayer(
            {
                (label, index, *(0 for _ in dims[1:])): (
                    _diagnostic_values,
                    task.key,
                    name,
                    stage,
                    first,
                )
                for index, (task, (first, _)) in enumerate(
                    zip(tasks, ranges, strict=True)
                )
            }
        )
        dependencies[label] = source_layers
    graph = HighLevelGraph(layers, dependencies)
    data = xr.Dataset(coords={"time": moving.time})
    for name, (dims, dtype) in schema.items():
        chunks = (tuple(last - first for first, last in ranges),) + tuple(
            (sizes[dim],) for dim in dims[1:]
        )
        data[name] = (
            dims,
            # Dask defines its public array constructor in __new__, not __init__.
            da.Array(graph, f"{prefix}-{name}", chunks=chunks, dtype=dtype),  # ty: ignore[too-many-positional-arguments, unknown-argument]
        )
    if "world_component" in data.dims:
        data.coords["world_component"] = ["z", "y", "x"]
        data.rotation_center_zyx_mm.attrs["units"] = "mm"
    if multiresolution:
        data.coords["level"] = np.arange(len(factors))
        for name, values in (
            ("shrink_factor", factors),
            ("smoothing_sigma", sigmas),
            ("iteration_budget", budgets),
        ):
            data[name] = ("level", np.asarray(values))
        data.smoothing_sigma.attrs["units"] = "mm"
    metric = (
        "mutual information"
        if settings.metric == "mi"
        else "squared local NCC"
        if model == "syn" and settings.metric != "ncc"
        else "squared global NCC"
    )
    for name in schema:
        if name.startswith("translation_"):
            data[name].attrs.update(
                units="mm", basis="intrinsic i/j" if planar else "world z/y/x"
            )
        elif (
            name == "angle"
            or name.startswith("rotation_")
            and name != "rotation_center_zyx_mm"
        ):
            data[name].attrs.update(
                units="rad", basis="intrinsic i/j plane" if planar else "world z/y/x"
            )
        elif name.startswith("linear_"):
            data[name].attrs.update(
                units="1", basis="intrinsic i/j" if planar else "world z/y/x"
            )
    if "inverse_error" in data:
        data.inverse_error.attrs.update(
            units="voxel-coordinate norm",
            domain="fixed virtual grid; residual maps before affine seed",
        )
    if "roi_valid_fraction" in data:
        data.roi_valid_fraction.attrs["domain"] = (
            "native fixed image; endpoint ROI intersection with geometric support"
            if model == "syn"
            else "native fixed image; ROI intersection with geometric support"
        )
    if isinstance(settings.initialization, str):
        data.attrs["initialization"] = settings.initialization
    if settings.metric == "mi":
        data.attrs.update(
            mi_bins=settings.mi_bins,
            mi_sigma_bin_widths=settings.mi_sigma,
            mi_kernel="gaussian",
            objective_units="nats",
            mi_range="native per-image min/max mapped to [0, 1] before pyramid smoothing",
        )
    if model == "syn":
        data.attrs.update(
            cc_radius=settings.syn_cc_radius,
            update_sigma_voxels=settings.syn_update_sigma,
            total_sigma_voxels=settings.syn_total_sigma,
            regularity="positive sampled Jacobians are checked; not a proof of global invertibility",
            inverse_error_note="sampled inverse is approximate; two-sided residual is reported, not forced to zero",
        )
    if multiresolution:
        data.attrs["objective_domain"] = (
            "final pyramid level; losses are not compared across levels"
        )
    if stage is not None:
        data.attrs["stage"] = stage
    data.attrs.update(
        transform=model,
        metric=f"negative {metric}",
        execution="lazy time batches; device and timings resolve in tasks",
        timing_policy="batch timings repeat per frame; deduplicate by execution_batch_start",
    )
    return data


def register_lazy(
    moving: xr.DataArray,
    fixed: xr.DataArray | None,
    specification: Any,
    settings: RegistrationSettings,
) -> xr.DataTree:
    """Construct shared bounded fits without reading spatial data.

    Parameters
    ----------
    moving : xarray.DataArray
        Recording with a NumPy or Dask backend.
    fixed : xarray.DataArray, optional
        Spatial reference; not provided selects the global first frame.
    specification : object
        Complete model or stage pipeline.
    settings : RegistrationSettings
        Unresolved public settings, with optional lazy masks and seeds.

    Returns
    -------
    xarray.DataTree
        Lazy images, masks, optional maps and execution diagnostics.

    Raises
    ------
    ValueError
        If metadata, task bounds or labels are invalid.
    TypeError
        If lazy registration is requested for a spatial pair without time.
    """
    from voxreg.registration import run_stage

    validate_registration_geometry(moving)
    if "time" not in moving.dims:
        raise TypeError(
            "Lazy registration requires a time recording; compute spatial image pairs before register."
        )
    implicit = fixed is None
    target = moving.isel(time=0, drop=True) if implicit else fixed
    assert target is not None
    if isinstance(specification, str):
        run_stage(moving, fixed, specification, settings, validate_only=True)
        stages = ((Stage(specification), settings),)
        staged = False
    else:
        stages = resolve_stages(moving, fixed, specification, settings)
        staged = True
    ranges = _task_ranges(
        moving.sizes["time"],
        settings.batch_size,
        moving,
        settings.moving_mask,
        settings.initial_transform,
        settings.initialization,
    )
    fixed_dependency = delayed(_eager_dependency(target))
    shared = {field.name: getattr(settings, field.name) for field in fields(settings)}
    shared["fixed_mask"] = (
        delayed(_eager_dependency(settings.fixed_mask))
        if settings.fixed_mask is not None
        else None
    )
    show_progress = settings.progress is True or (
        settings.progress == "auto" and moving.sizes["time"] > 1
    )
    if show_progress:
        enable_lazy_progress()
    progress_id = uuid4().hex
    tasks = []
    for first, last in ranges:
        options = shared.copy()
        for key in ("moving_mask", "initial_transform", "initialization"):
            options[key] = _eager_dependency(_select_frames(options[key], first, last))
        selected = moving.isel(time=slice(first, last))
        if implicit and first == 0 and last == 1:
            # Reuse the fixed dependency instead of reading the reference twice.
            selected = selected.copy(data=np.broadcast_to(np.uint8(0), selected.shape))
        task = delayed(_run_registration_batch)(
            _eager_dependency(selected),
            fixed_dependency,
            specification,
            options,
            first,
            moving.sizes["time"],
            implicit,
            dask_key_name=(
                f"voxregframes_{progress_id}_{first}_{last}_end"
                if show_progress
                else f"voxreg-fit-{progress_id}-{first}"
            ),
        )
        tasks.append(task)
    template = build_output_template(moving, target, False)
    image_shape = tuple(target.sizes[d] for d in VOXEL_DIMS)

    def assemble(name, shape, dtype):
        """Project bounded eager output blocks onto a shared lazy array.

        Parameters
        ----------
        name : str
            Eager result attribute to project.
        shape : tuple of int
            Output dimensions excluding time.
        dtype : numpy.dtype or type
            Output array dtype.

        Returns
        -------
        dask.array.Array
            Shared task projections concatenated along time.
        """
        return da.concatenate(
            [
                da.from_delayed(
                    delayed(_registration_values)(task, name),
                    shape=(last - first, *shape),
                    dtype=dtype,
                )
                for task, (first, last) in zip(tasks, ranges, strict=True)
            ],
            axis=0,
        )

    registered = template.copy(data=assemble("registered", image_shape, moving.dtype))
    valid = template.copy(data=assemble("valid", image_shape, bool))
    valid.attrs = {"description": "Valid moving-image sample centers"}
    model, final_settings = stages[-1][0].transform, stages[-1][1]
    if staged and len(stages) > 1:
        final_settings = replace(final_settings, initialization=None)
    diagnostics = _diagnostics(
        tasks, ranges, moving, target, model, final_settings, None, stage_summary=staged
    )
    if staged:
        diagnostics.attrs["stages"] = [stage.transform for stage, _ in stages]
        diagnostics.attrs["stage_failure_policy"] = (
            "Stop each failed frame; retain its best valid transform; later stages mark it skipped."
        )
    stage_diagnostics = (
        tuple(
            _diagnostics(
                tasks,
                ranges,
                moving,
                target,
                stage.transform,
                replace(options, initialization=None) if index else options,
                index,
            )
            for index, (stage, options) in enumerate(stages)
        )
        if staged
        else None
    )
    forward = inverse = None
    if settings.return_transforms:
        if model == "syn":
            forward = template.expand_dims(
                world_component=["z", "y", "x"], axis=-1
            ).copy(data=assemble("forward_transforms", (*image_shape, 3), np.float64))
            inverse = moving.expand_dims(world_component=["z", "y", "x"], axis=-1).copy(
                data=assemble(
                    "inverse_transforms",
                    (*(moving.sizes[d] for d in VOXEL_DIMS), 3),
                    np.float64,
                )
            )
            forward.name = inverse.name = "displacement"
            attrs = {
                "representation": "displacement",
                "spatial_units": "mm",
                "units": "mm",
            }
        else:
            coords = {
                "time": moving.time,
                "world_out": ["z", "y", "x", "1"],
                "world_in": ["z", "y", "x", "1"],
            }
            forward = xr.DataArray(
                assemble("forward_transforms", (4, 4), np.float64),
                dims=("time", "world_out", "world_in"),
                coords=coords,
            )
            inverse = forward.copy(
                data=assemble("inverse_transforms", (4, 4), np.float64)
            )
            attrs = {"spatial_units": "mm"}
        forward.attrs = {**attrs, "convention": "fixed-world-to-moving-world pull"}
        inverse.attrs = {**attrs, "convention": "moving-world-to-fixed-world pull"}
    return assemble_result_tree(
        registered, valid, forward, diagnostics, inverse, stage_diagnostics
    )


def _run_resampling_batch(
    moving: xr.DataArray,
    reference: xr.DataArray,
    transforms: Any,
    interpolation: Interpolation,
    device: Literal["auto", "cpu", "cuda"],
    batch_size: int | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Resample one eager batch with worker-local device resolution.

    Parameters
    ----------
    moving, reference : xarray.DataArray
        Eager source batch and shared reference metadata.
    transforms : object
        Required eager map frames only.
    interpolation : str
        Public final interpolation.
    device : str
        Unresolved public device choice.
    batch_size : int, optional
        Fitting/resampling batch override; not provided uses worker memory policy.

    Returns
    -------
    image : xarray.DataArray
        Bounded eager output.
    valid : xarray.DataArray
        Bounded eager support mask.
    """
    from voxreg.resampling import resample

    resolved = resolve_device(device)
    with _execution_slot(resolved):
        return resample(
            moving,
            reference,
            transforms,
            interpolation=interpolation,
            device=resolved,
            batch_size=batch_size,
        )


def _resampling_values(
    result: tuple[xr.DataArray, xr.DataArray], index: int
) -> np.ndarray:
    """Project an eager resampling image or support mask.

    Parameters
    ----------
    result : tuple of xarray.DataArray
        Shared resampling result.
    index : int
        Image or validity selector.

    Returns
    -------
    numpy.ndarray
        Eager task values.
    """
    return result[index].values


def resample_lazy(
    moving: xr.DataArray,
    reference: xr.DataArray,
    transforms: Any,
    interpolation: Interpolation,
    device: Literal["auto", "cpu", "cuda"],
    batch_size: int | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Construct lazy resampling tasks for eager/lazy image and map combinations.

    Parameters
    ----------
    moving, reference : xarray.DataArray
        Spatial source or recording and reference metadata.
    transforms : object
        Shared or time-labelled matrix/displacement pulls.
    interpolation : str
        Final image interpolation.
    device : str
        Worker-local execution choice.
    batch_size : int, optional
        Eager resampling batch override, independent of source task chunks.

    Returns
    -------
    image : xarray.DataArray
        Lazy intensities on the reference grid.
    valid : xarray.DataArray
        Lazy support on the same grid.

    Raises
    ------
    ValueError
        If device settings or source chunk bounds are invalid.
    """
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda.")
    source = moving
    labeled = isinstance(transforms, xr.DataArray)
    per_frame = "time" in transforms.dims if labeled else np.ndim(transforms) == 3
    count = (
        transforms.sizes["time"]
        if labeled and per_frame
        else np.shape(transforms)[0]
        if per_frame
        else 1
    )
    spatial = "time" not in moving.dims and not per_frame
    if "time" not in moving.dims:
        time = transforms.time if labeled and per_frame else np.arange(count)
        moving = moving.expand_dims(time=time)
        validate_registration_geometry(moving)
    ranges = _task_ranges(moving.sizes["time"], batch_size, source, transforms)
    # Reference intensities are unused; replacing data avoids loading a lazy image.
    reference = reference.copy(data=np.broadcast_to(np.uint8(0), reference.shape))
    template = build_output_template(moving, reference, False)
    shared = delayed(_eager_dependency(transforms)) if not per_frame else None
    reference_dependency = delayed(reference)
    outputs, masks = [], []
    shape = tuple(reference.sizes[d] for d in VOXEL_DIMS)
    for first, last in ranges:
        selected = (
            _select_frames(transforms, first, last)
            if labeled
            else transforms[first:last]
            if per_frame
            else shared
        )
        if not per_frame:
            selected = shared
        task = delayed(_run_resampling_batch)(
            _eager_dependency(moving.isel(time=slice(first, last))),
            reference_dependency,
            _eager_dependency(selected),
            interpolation,
            device,
            batch_size,
        )
        outputs.append(
            da.from_delayed(
                delayed(_resampling_values)(task, 0),
                shape=(last - first, *shape),
                dtype=moving.dtype,
            )
        )
        masks.append(
            da.from_delayed(
                delayed(_resampling_values)(task, 1),
                shape=(last - first, *shape),
                dtype=bool,
            )
        )
    image = template.copy(data=da.concatenate(outputs))
    valid = template.copy(data=da.concatenate(masks))
    valid.attrs = {"description": "Valid moving-image sample centers"}
    if spatial:
        return image.isel(time=0, drop=True), valid.isel(time=0, drop=True)
    return image, valid
