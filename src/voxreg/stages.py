"""Explicit registration stages and independent per-frame pipeline execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, fields, replace
from time import perf_counter
from typing import Literal, cast

import numpy as np
import xarray as xr

from voxreg.preparation import RegistrationSettings
from voxreg.progress import progress_bar
from voxreg.results import LinearFit


@dataclass(frozen=True)
class Stage:
    """Specify a model and explicit overrides for one registration stage.

    Unset options inherit the corresponding shared register argument. Shared
    arguments in turn default to the existing model settings. Sequence options
    are copied to tuples; inputs and this specification are not modified.
    Moving/fixed winsorization and intensity scaling are shared across stages
    and always start from original intensities, never preceding fitting images.

    Attributes
    ----------
    transform : {"translation", "rigid", "affine", "syn"}
        Model fitted at this stage. Pipelines follow this order, skipping models
        as needed. Repeated or decreasing-complexity stages are unsupported.
    max_iterations : int or sequence of int, optional
        Maximum updates per frame per pyramid level.
    shrink_factors : sequence of int, optional
        Coarse-to-fine downsampling factors.
    smoothing_sigmas : sequence of float, optional
        Gaussian image smoothing standard deviations in mm, one per level.
    learning_rate : float, optional
        Model-specific optimizer step size.
    convergence_window_size : int, optional
        Number of level-local objective values in the convergence window.
    convergence_minimum_value : float, optional
        Threshold for the normalized convergence trend.
    metric : {"ncc", "local_ncc", "mi"}, optional
        Override the shared objective. Unset inherits the shared metric setting.
    mi_bins : int, optional
        Number of MI histogram bins, from 2 through 256.
    mi_sigma : float, optional
        MI Gaussian standard deviation in bin widths, from 0.1 through 10.
    syn_cc_radius : int, optional
        SyN rectangular CC window radius in current-level voxels.
    syn_update_sigma : float, optional
        SyN update smoothing standard deviation in current-level voxels.
    syn_total_sigma : float, optional
        SyN total-field smoothing standard deviation in current-level voxels.
    """

    transform: Literal["translation", "rigid", "affine", "syn"]
    _: KW_ONLY
    max_iterations: int | Sequence[int] | None = None
    shrink_factors: Sequence[int] | None = None
    smoothing_sigmas: Sequence[float] | None = None
    learning_rate: float | None = None
    convergence_window_size: int | None = None
    convergence_minimum_value: float | None = None
    metric: Literal["ncc", "local_ncc", "mi"] | None = None
    mi_bins: int | None = None
    mi_sigma: float | None = None
    syn_cc_radius: int | None = None
    syn_update_sigma: float | None = None
    syn_total_sigma: float | None = None

    def __post_init__(self):
        """Own sequence specifications without mutating the caller's lists."""
        for name in ("max_iterations", "shrink_factors", "smoothing_sigmas"):
            value = getattr(self, name)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                object.__setattr__(self, name, tuple(value))

    def _overrides(self):
        """Return only values explicitly supplied for this stage."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "transform" and getattr(self, field.name) is not None
        }


def resolve_stages(
    moving: xr.DataArray,
    fixed: xr.DataArray | None,
    specification: Stage | Sequence[str | Stage],
    shared: RegistrationSettings,
) -> tuple[tuple[Stage, RegistrationSettings], ...]:
    """Resolve and validate every effective stage without fitting images.

    Parameters
    ----------
    moving : xarray.DataArray
        Original spatial image or recording.
    fixed : xarray.DataArray, optional
        Shared spatial reference, or the implicit first recording frame.
    specification : Stage or sequence of str or Stage
        Increasing-complexity transform stages and their explicit overrides.
    shared : RegistrationSettings
        Settings inherited by every stage unless overridden.

    Returns
    -------
    tuple of tuple of Stage and RegistrationSettings
        Ordered stage specifications and validated effective settings.

    Raises
    ------
    ValueError
        If stage order or any effective stage configuration is invalid.
    """
    from voxreg.registration import run_stage

    stages = (
        (specification,) if isinstance(specification, Stage) else tuple(specification)
    )
    if not stages:
        raise ValueError("transform stage sequence must not be empty.")
    stages = tuple(
        Stage(cast(Literal["translation", "rigid", "affine", "syn"], stage))
        if isinstance(stage, str)
        else stage
        for stage in stages
    )
    order = {"translation": 0, "rigid": 1, "affine": 2, "syn": 3}
    previous = -1
    resolved = []
    has_syn = any(
        isinstance(stage, Stage) and stage.transform == "syn" for stage in stages
    )
    for stage in stages:
        if (
            not isinstance(stage, Stage)
            or not isinstance(stage.transform, str)
            or stage.transform not in order
        ):
            raise ValueError(
                "Each transform stage must be translation, rigid, affine, syn or a Stage specification."
            )
        rank = order[stage.transform]
        if rank <= previous:
            raise ValueError(
                "Stages must increase in complexity: translation, rigid, affine, syn; SyN must be last."
            )
        previous = rank
        options = shared
        if stage.transform != "syn" and has_syn:
            options = replace(
                options, syn_cc_radius=2, syn_update_sigma=1.5, syn_total_sigma=0.5
            )
        overrides = stage._overrides()
        if stage.transform != "syn" and any(
            name.startswith("syn_") for name in overrides
        ):
            raise ValueError("Stage syn_ options require transform='syn'.")
        options = replace(options, **overrides)
        # Validate every effective stage configuration before fitting the first one.
        run_stage(moving, fixed, stage.transform, options, validate_only=True)
        resolved.append(options)

    return tuple(zip(stages, resolved, strict=True))


def register_stages(
    moving: xr.DataArray,
    fixed: xr.DataArray | None,
    specification: Stage | Sequence[str | Stage],
    shared: RegistrationSettings,
) -> xr.DataTree:
    """Fit stages eagerly and resample original intensities once.

    Parameters
    ----------
    moving : xarray.DataArray
        Original spatial image or recording.
    fixed : xarray.DataArray, optional
        Explicit reference; not provided uses the first frame.
    specification : Stage or sequence of str or Stage
        Increasing-complexity stages and overrides.
    shared : RegistrationSettings
        Settings shared by the complete pipeline.

    Returns
    -------
    xarray.DataTree
        Complete outputs, including independent stage diagnostics.

    Raises
    ------
    ValueError
        If a stage specification or effective settings are invalid.
    """
    from voxreg.registration import run_stage

    start = perf_counter()
    stages, resolved = zip(
        *resolve_stages(moving, fixed, specification, shared), strict=True
    )
    spatial = moving.dims == ("k", "j", "i")
    count = 1 if spatial else moving.sizes["time"]
    active = np.ones(count, dtype=bool)
    failure = np.full(count, "", dtype="<U32")
    total_updates = np.zeros(count, dtype=np.int64)
    diagnostics = []
    center = None
    seed = None
    totals = dict.fromkeys(("prepare", "upload", "optimize", "resample_download"), 0.0)
    progress = shared.progress
    if isinstance(progress, str):
        progress = count > 1
    total_work = count * len(stages)
    with progress_bar(
        "Registering stages", total_work, show=progress is True
    ) as advance:
        for index, (stage, options) in enumerate(zip(stages, resolved, strict=True)):
            if index:
                options = replace(options, initialization=None, initial_transform=seed)
            options = replace(
                options,
                return_transforms=shared.return_transforms
                if index == len(stages) - 1
                else False,
            )

            completed_before = [0]

            def report(
                completed, total, index=index, completed_before=completed_before
            ):
                """Report monotonically increasing stage/frame work units."""
                if progress is True:
                    advance(completed - completed_before[0])
                elif callable(progress):
                    progress(index * count + completed, total_work)
                completed_before[0] = completed

            options = replace(
                options,
                progress=report if progress is True or callable(progress) else False,
            )
            result = run_stage(
                moving,
                fixed,
                stage.transform,
                options,
                active_frames=active,
                center_override=center,
                fit_only=index < len(stages) - 1,
            )
            assert isinstance(result, (xr.DataTree, LinearFit))
            stage_diagnostics = (
                result.diagnostics
                if isinstance(result, LinearFit)
                else result["diagnostics"].to_dataset()
            ).copy(deep=True)
            if isinstance(result, LinearFit) and spatial:
                stage_diagnostics = stage_diagnostics.isel(time=0, drop=True)
            stage_diagnostics.attrs.update(stage=index, transform=stage.transform)
            diagnostics.append(stage_diagnostics)
            status = np.atleast_1d(stage_diagnostics.status.values)
            failed = active & ~np.isin(
                status, ["converged", "max_iterations", "reference"]
            )
            failure[failed] = status[failed]
            active &= ~failed
            total_updates += np.atleast_1d(stage_diagnostics.n_iterations.values)
            for name, seconds in stage_diagnostics.attrs["timings_seconds"].items():
                totals[name] += seconds
            if isinstance(result, LinearFit):
                seed, center = result.matrices, result.center

    assert isinstance(result, xr.DataTree)
    summary = result["diagnostics"].to_dataset().copy(deep=True)
    final_status = np.atleast_1d(summary.status.values).copy()
    final_status[failure != ""] = failure[failure != ""]
    summary["status"] = summary.status.copy(
        data=final_status[0] if spatial else final_status
    )
    summary["n_iterations"] = summary.n_iterations.copy(
        data=total_updates[0] if spatial else total_updates
    )
    summary.attrs.update(
        stages=[stage.transform for stage in stages],
        timings_seconds=totals,
        stage_failure_policy="Stop each failed frame; retain its best valid transform; later stages mark it skipped.",
        stage_overhead_seconds=max(0.0, perf_counter() - start - sum(totals.values())),
    )
    result["diagnostics"] = xr.DataTree(summary)
    for index, data in enumerate(diagnostics):
        result[f"stages/{index}"] = xr.DataTree(data)
    return result
