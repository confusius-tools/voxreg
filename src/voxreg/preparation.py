"""Shared registration settings, prepared inputs and fitting preparation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr

from voxreg.interpolation import Interpolation

if TYPE_CHECKING:
    from voxreg.metrics import Metric
    from voxreg.sampling import Plane, SpatialGrid


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
    intensity_scaling: Literal["none", "db"] = "db"
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


def prepare_fitting_images(
    raw: npt.NDArray,
    target_raw: npt.NDArray,
    intensity_scaling: Literal["none", "db"],
    metric: str | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Prepare float32 fitting images while preserving double/integer contrast.

    Parameters
    ----------
    raw : numpy.ndarray
        Native original intensities with a leading frame axis.
    target_raw : numpy.ndarray
        Shared native fixed image without a frame axis.
    intensity_scaling : {"none", "db"}
        Intensity transform applied only to fitting images.
    metric : str, optional
        Selected metric; MI uses native per-image ranges instead of NCC conditioning.

    Returns
    -------
    fit : numpy.ndarray
        Conditioned float32 moving frames.
    target : numpy.ndarray
        Conditioned float32 fixed image.

    Raises
    ------
    ValueError
        If all moving values are zero, the fixed image is constant, or dB inputs
        contain negative values.
    """
    spatial_axes = tuple(range(1, raw.ndim))
    preprocessing_dtype = (
        np.float64
        if any(
            np.issubdtype(data.dtype, np.integer) or data.dtype.itemsize > 4
            for data in (raw, target_raw)
        )
        else np.float32
    )
    fit_raw = raw.astype(preprocessing_dtype, copy=False)
    target_raw = target_raw.astype(preprocessing_dtype, copy=False)
    moving_scale = float(np.abs(fit_raw).max())
    scale = max(moving_scale, float(np.abs(target_raw).max()))
    if moving_scale == 0:
        raise ValueError("An all-zero recording cannot be registered.")
    if intensity_scaling == "db":
        if fit_raw.min() < 0 or target_raw.min() < 0:
            raise ValueError("dB scaling requires nonnegative power intensities.")
        fit = 10 * np.log10(np.maximum(fit_raw / scale, 1e-12))
        target = 10 * np.log10(np.maximum(target_raw / scale, 1e-12))
    elif metric == "mi":
        # Subtract native offsets before dividing so double/integer contrast
        # survives even when it is tiny relative to an image's DC intensity.
        fit, target = fit_raw, target_raw
    else:
        fit, target = fit_raw / scale, target_raw / scale
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
    elif preprocessing_dtype == np.float64:
        for values in (fit, target):
            values -= values[(...,) + (slice(0, 1),) * len(spatial_axes)]
            amplitude = np.abs(values).max(
                axis=tuple(range(-len(spatial_axes), 0)), keepdims=True
            )
            values /= np.where(amplitude > 0, amplitude, 1)
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
