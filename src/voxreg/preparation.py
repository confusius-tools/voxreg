"""Shared registration settings, prepared inputs and fitting preparation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr

from voxreg.interpolation import Interpolation

if TYPE_CHECKING:
    from voxreg.metrics import Metric
    from voxreg.sampling import Plane, SpatialGrid


IntensityScaling: TypeAlias = Literal["log1p", "sqrt"] | float | None
"""Fitting-only magnitude transform, or no transform."""


@dataclass(frozen=True, kw_only=True)
class RegistrationSettings:
    """Carry shared registration settings between the entry point and stages.

    This internal execution record owns no image data and defines no workflow
    presets. Attribute names and units match the public register API; device is
    already resolved to CPU or CUDA. Stage overrides create a new record with
    dataclasses.replace.
    """

    fixed_mask: xr.DataArray | None = None
    moving_mask: xr.DataArray | None = None
    initial_transform: npt.ArrayLike | xr.DataArray | None = None
    initialization: (
        Literal["center_geometry", "center_moments", "align_planes"]
        | npt.ArrayLike
        | xr.DataArray
        | None
    ) = None
    moving_intensity_scaling: IntensityScaling = None
    fixed_intensity_scaling: IntensityScaling = None
    moving_winsorize_percentiles: Sequence[float] | None = None
    fixed_winsorize_percentiles: Sequence[float] | None = None
    metric: Literal["ncc", "local_ncc", "mi"] | None = None
    mi_bins: int = 32
    mi_sigma: float = 0.5
    resample_interpolation: Interpolation = "linear"
    return_transforms: bool = False
    batch_size: int | None = None
    max_iterations: int | Sequence[int] = 100
    shrink_factors: Sequence[int] = (1,)
    smoothing_sigmas: Sequence[float] = (0.0,)
    learning_rate: float = 0.1
    syn_cc_radius: int = 2
    syn_update_sigma: float = 1.5
    syn_total_sigma: float = 0.5
    convergence_window_size: int = 50
    convergence_minimum_value: float = 1e-06
    device: Literal["cpu", "cuda"] = "cpu"
    progress: Literal["auto"] | bool | Callable[[int, int], None] | None = "auto"
    metric_callback: (
        Callable[
            [npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]],
            None,
        ]
        | None
    ) = None
    metric_interval: int = 10


@dataclass(frozen=True)
class PreparedRegistration:
    """Carry native fitting inputs and resolved settings into one model engine.

    Native original images remain separate from conditioned fitting intensities.
    Physical geometry and initial parameters are in mm. Preparation and validation
    are owned by the registration entry point, not repeated by fitting engines.
    """

    moving: xr.DataArray
    fixed: xr.DataArray
    settings: RegistrationSettings
    transform: str
    plane: Plane | SpatialGrid
    fit: npt.NDArray[np.float32]
    target: npt.NDArray[np.float32]
    initial: npt.NDArray[np.float64] | None
    metric_function: Metric | None
    timings: dict[str, float]
    factors: tuple[int, ...]
    sigmas: tuple[float, ...]
    budgets: tuple[int, ...]
    multiresolution: bool
    spatial_input: bool
    implicit_fixed: bool
    same_grid: bool
    active_frames: npt.NDArray[np.bool_] | None


def resolve_device(device: Literal["auto", "cpu", "cuda"]) -> Literal["cpu", "cuda"]:
    """Resolve a public execution choice before preparation or device allocation.

    Parameters
    ----------
    device : {"auto", "cpu", "cuda"}
        Auto selects CUDA when available, otherwise CPU. Explicit CPU bypasses
        availability checks; explicit CUDA requires availability. CUDA execution
        failures are never retried here.

    Returns
    -------
    {"cpu", "cuda"}
        Concrete Torch execution device for the complete call, including stages.

    Raises
    ------
    ValueError
        If the choice is unsupported or explicit CUDA is unavailable.
    """
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda.")
    if device == "cpu":
        return "cpu"
    available = torch.cuda.is_available()
    if device == "auto":
        return "cuda" if available else "cpu"
    if not available:
        raise ValueError("CUDA was requested but is unavailable.")
    return "cuda"


def validate_intensity_scaling(value: IntensityScaling, name: str) -> None:
    """Validate a fitting transform before any stage executes.

    Parameters
    ----------
    value : {"log1p", "sqrt"} or float, optional
        Magnitude transform or a positive finite power exponent.
    name : str
        Public argument name used in errors.

    Returns
    -------
    None
        Valid settings require no conversion.

    Raises
    ------
    ValueError
        If the transform is unknown or the exponent is not positive and finite.
    """
    if value is None or (isinstance(value, str) and value in ("log1p", "sqrt")):
        return
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        try:
            if np.isfinite(float(value)) and value > 0:
                return
        except OverflowError:
            pass
    raise ValueError(
        f"{name} must be None, 'log1p', 'sqrt' or a positive finite exponent."
    )


def validate_winsorize_percentiles(value: Sequence[float] | None, name: str) -> None:
    """Validate optional lower and upper fitting percentiles.

    Parameters
    ----------
    value : sequence of float, optional
        Two finite percentile bounds in increasing order within [0, 100].
    name : str
        Public argument name used in errors.

    Returns
    -------
    None
        Valid bounds need no conversion.

    Raises
    ------
    ValueError
        If bounds are not two finite real non-boolean values in range and order.
    """
    if value is None:
        return
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(
            isinstance(bound, (int, float, np.integer, np.floating))
            and not isinstance(bound, (bool, np.bool_))
            and 0 <= bound <= 100
            for bound in value
        )
        and value[0] < value[1]
    ):
        return
    raise ValueError(
        f"{name} must be None or two finite percentiles with 0 <= lower < upper <= 100."
    )


def _winsorize_intensities(
    values: npt.NDArray,
    percentiles: Sequence[float] | None,
    axes: tuple[int, ...],
    name: str,
) -> npt.NDArray:
    """Clip native signed intensities using per-image linear percentiles.

    Parameters
    ----------
    values : numpy.ndarray
        Finite intensities in preprocessing precision.
    percentiles : sequence of float, optional
        Validated lower and upper bounds; not provided leaves values unchanged.
    axes : tuple of int
        Complete spatial axes, excluding time.
    name : str
        Image side used in errors.

    Returns
    -------
    numpy.ndarray
        Clipped values, or unchanged input when clipping is disabled.

    Raises
    ------
    ValueError
        If percentile interpolation produces nonfinite bounds.
    """
    if percentiles is None:
        return values
    with np.errstate(over="ignore", invalid="ignore"):
        lower, upper = np.percentile(
            values, percentiles, axis=axes, keepdims=True, method="linear"
        )
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError(f"{name} winsorization percentile bounds overflowed float64.")
    return np.clip(values, lower, upper)


def _scale_intensities(
    values: npt.NDArray, scaling: IntensityScaling, name: str
) -> npt.NDArray:
    """Transform magnitudes in float64, rejecting nonfinite power results.

    Parameters
    ----------
    values : numpy.ndarray
        Finite native intensities, with any leading time axis retained.
    scaling : {"log1p", "sqrt"} or float, optional
        Validated transform. No transform preserves signed values.
    name : str
        Image side used in errors.

    Returns
    -------
    numpy.ndarray
        Unchanged values or transformed float64 magnitudes.

    Raises
    ------
    ValueError
        If power scaling overflows.
    """
    if scaling is None:
        return values
    magnitude = np.abs(values.astype(np.float64))
    if scaling == "log1p":
        return np.log1p(magnitude)
    exponent = 0.5 if scaling == "sqrt" else float(scaling)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        transformed = magnitude**exponent
    if not np.isfinite(transformed).all():
        raise ValueError(f"{name} intensity scaling overflowed float64.")
    return transformed


def prepare_fitting_images(
    raw: npt.NDArray,
    target_raw: npt.NDArray,
    moving_intensity_scaling: IntensityScaling = None,
    fixed_intensity_scaling: IntensityScaling = None,
    metric: str | None = None,
    *,
    implicit_fixed: bool = False,
    moving_winsorize_percentiles: Sequence[float] | None = None,
    fixed_winsorize_percentiles: Sequence[float] | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Prepare float32 fitting images while preserving double/integer contrast.

    Parameters
    ----------
    raw : numpy.ndarray
        Native original intensities with a leading frame axis.
    target_raw : numpy.ndarray
        Shared native fixed image without a frame axis.
    moving_intensity_scaling, fixed_intensity_scaling : {"log1p", "sqrt"} or float, optional
        Independent fitting transforms, applied before metric conditioning.
        log1p computes ln(1 + abs(x)); powers use magnitudes. Transforms are
        evaluated in float64. Numerical conditioning is per image, never across
        frames or shared between moving and fixed images.
    metric : str, optional
        Selected metric; MI uses native per-image ranges instead of NCC conditioning.
    implicit_fixed : bool, default: False
        Whether to reuse the first transformed moving frame as the fitting target.
    moving_winsorize_percentiles, fixed_winsorize_percentiles : sequence of float, optional
        Lower/upper percentiles in [0, 100], independently per native image before
        intensity transforms. Bounds include values outside masks; interpolation
        is NumPy's linear method. Clipping is disabled when not provided.

    Returns
    -------
    fit : numpy.ndarray
        Conditioned float32 moving frames.
    target : numpy.ndarray
        Conditioned float32 fixed image.

    Raises
    ------
    ValueError
        If all moving values are zero, the fixed image is constant, a scaling
        option is invalid, or a transform produces nonfinite values.
    """
    validate_intensity_scaling(moving_intensity_scaling, "moving_intensity_scaling")
    validate_intensity_scaling(fixed_intensity_scaling, "fixed_intensity_scaling")
    validate_winsorize_percentiles(
        moving_winsorize_percentiles, "moving_winsorize_percentiles"
    )
    validate_winsorize_percentiles(
        fixed_winsorize_percentiles, "fixed_winsorize_percentiles"
    )
    spatial_axes = tuple(range(1, raw.ndim))
    preprocessing_dtype = (
        np.float64
        if moving_intensity_scaling is not None
        or fixed_intensity_scaling is not None
        or moving_winsorize_percentiles is not None
        or fixed_winsorize_percentiles is not None
        or any(
            np.issubdtype(data.dtype, np.integer) or data.dtype.itemsize > 4
            for data in (raw, target_raw)
        )
        else np.float32
    )
    fit_raw = raw.astype(preprocessing_dtype, copy=False)
    target_raw = target_raw.astype(preprocessing_dtype, copy=False)
    if not np.any(fit_raw):
        raise ValueError("An all-zero recording cannot be registered.")
    fit_raw = _winsorize_intensities(
        fit_raw, moving_winsorize_percentiles, spatial_axes, "moving"
    )
    fit_raw = _scale_intensities(fit_raw, moving_intensity_scaling, "moving")
    target_raw = (
        fit_raw[0].copy()
        if implicit_fixed
        else _scale_intensities(
            _winsorize_intensities(
                target_raw,
                fixed_winsorize_percentiles,
                tuple(range(target_raw.ndim)),
                "fixed",
            ),
            fixed_intensity_scaling,
            "fixed",
        )
    )
    if not np.any(fit_raw) and not np.any(target_raw):
        raise ValueError("Intensity scaling produced all-zero fitting images.")
    fit, target = fit_raw, target_raw
    if metric == "mi":
        # Native ranges stay fixed across all levels, samples and ROI changes.
        def normalize(values, axes):
            low = values.min(axis=axes, keepdims=True)
            high = values.max(axis=axes, keepdims=True)
            with np.errstate(over="ignore"):
                overflow = ~np.isfinite(high - low)
            if overflow.any():
                scale = np.maximum(np.abs(low), np.abs(high))
                values = values / np.where(overflow, scale, 1)
                low = values.min(axis=axes, keepdims=True)
                high = values.max(axis=axes, keepdims=True)
            return (values - low) / np.where(high > low, high - low, 1)

        fit = normalize(fit, spatial_axes)
        target = normalize(target, tuple(range(target.ndim)))
    else:
        conditioned = []
        for values in (fit, target):
            axes = tuple(range(-len(spatial_axes), 0))
            amplitude = np.abs(values).max(axis=axes, keepdims=True)
            # Bound each image independently before subtraction or float32 fitting.
            values = values / np.where(amplitude > 0, amplitude, 1)
            if preprocessing_dtype == np.float64:
                values -= values[(...,) + (slice(0, 1),) * len(spatial_axes)]
                amplitude = np.abs(values).max(axis=axes, keepdims=True)
                values /= np.where(amplitude > 0, amplitude, 1)
            conditioned.append(values)
        fit, target = conditioned
    fit, target = fit.astype(np.float32), target.astype(np.float32)
    if np.ptp(target) == 0:
        raise ValueError("The fixed image must not be constant.")
    return fit, target


def choose_batch_size(
    moving: xr.DataArray, fixed: xr.DataArray, transform: str, device: str
) -> int:
    """Choose a frame batch from a conservative working-set estimate.

    Parameters
    ----------
    moving, fixed : xarray.DataArray
        Recording and shared native fixed image.
    transform : str
        Model used to estimate per-frame memory.
    device : {"cpu", "cuda"}
        Explicit execution device.

    Returns
    -------
    int
        Automatic frame batch bounded by recording length and available capacity.
    """
    frames = moving.sizes["time"]
    if device == "cpu":
        return min(frames, 32)
    planar = moving.sizes["k"] == fixed.sizes["k"] == 1
    # Ref20 benefits from large linear batches; larger SyN/volume batches do not.
    cap = 256 if planar and transform != "syn" else 32
    voxels = max(moving.size // frames, fixed.size)
    bytes_per_voxel = (1536 if planar else 3072) if transform == "syn" else 512
    free, _ = torch.cuda.mem_get_info()
    reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    # Leave half the available capacity for workspaces, fragmentation and other users.
    capacity = max(1, int((free + reusable) * 0.5) // (voxels * bytes_per_voxel))
    batch = min(cap, capacity)
    return min(frames, 1 << (batch.bit_length() - 1))
