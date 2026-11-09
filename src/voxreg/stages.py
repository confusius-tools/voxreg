"""Explicit registration stages and independent per-frame pipeline execution."""

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, fields, replace
from time import perf_counter
from typing import Literal

import numpy as np
from confusius._utils.progress import progress_bar


@dataclass(frozen=True)
class Stage:
    """Specify a model and explicit overrides for one registration stage.

    Unset options inherit the corresponding shared register argument. Shared
    arguments in turn default to the existing model settings. Sequence options
    are copied to tuples; inputs and this specification are not modified.

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


def _register_stages(moving, fixed, specification, shared):
    """Fit original images through ordered stages and resample only at the end."""
    from voxreg.rigid import RegistrationResult, _LinearFit, _register

    start = perf_counter()
    stages = (
        (specification,) if isinstance(specification, Stage) else tuple(specification)
    )
    if not stages:
        raise ValueError("transform stage sequence must not be empty.")
    stages = tuple(
        Stage(stage) if isinstance(stage, str) else stage for stage in stages
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
        options = shared.copy()
        if stage.transform != "syn" and has_syn:
            options.update(syn_cc_radius=2, syn_update_sigma=1.5, syn_total_sigma=0.5)
        overrides = stage._overrides()
        if stage.transform != "syn" and any(
            name.startswith("syn_") for name in overrides
        ):
            raise ValueError("Stage syn_ options require transform='syn'.")
        options.update(overrides)
        # Validate every effective stage configuration before fitting the first one.
        _register(
            moving, fixed, transform=stage.transform, _validate_only=True, **options
        )
        resolved.append(options)

    spatial = moving.dims == ("k", "j", "i")
    count = 1 if spatial else moving.sizes["time"]
    active = np.ones(count, dtype=bool)
    failure = np.full(count, "", dtype="<U32")
    total_updates = np.zeros(count, dtype=np.int64)
    diagnostics = []
    center = None
    seed = None
    totals = dict.fromkeys(("prepare", "upload", "optimize", "resample_download"), 0.0)
    progress = shared["progress"]
    if isinstance(progress, str):
        progress = count > 1
    total_work = count * len(stages)
    with progress_bar(
        "Registering stages", total_work, show=progress is True
    ) as advance:
        for index, (stage, options) in enumerate(zip(stages, resolved, strict=True)):
            options = options.copy()
            if index:
                options.update(initialization=None, initial_transform=seed)
            options["return_transforms"] = (
                shared["return_transforms"] if index == len(stages) - 1 else False
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

            options["progress"] = (
                report if progress is True or callable(progress) else False
            )
            result = _register(
                moving,
                fixed,
                transform=stage.transform,
                _active=active,
                _center=center,
                _fit_only=index < len(stages) - 1,
                **options,
            )
            assert isinstance(result, (RegistrationResult, _LinearFit))
            stage_diagnostics = result.diagnostics.copy(deep=True)
            if isinstance(result, _LinearFit) and spatial:
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
            if isinstance(result, _LinearFit):
                seed, center = result.matrices, result.center

    assert isinstance(result, RegistrationResult)
    summary = result.diagnostics.copy(deep=True)
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
    return replace(result, diagnostics=summary, stage_diagnostics=tuple(diagnostics))
