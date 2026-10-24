"""Independent per-frame rigid optimization on a shared physical voxel plane."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import torch
import xarray as xr
from confusius._utils.progress import progress_bar

from voxreg.geometry import VOXEL_DIMS, validate_voxeldata
from voxreg.sampling import Plane, global_ncc, make_plane, sample_rigid, world_matrices


@dataclass(frozen=True)
class RegistrationResult:
    """Labeled registration outputs, not a replacement spatial container.

    Parameters
    ----------
    registered : xarray.DataArray
        Original intensities resampled on the fixed grid, retaining moving dtype.
        Integer output is rounded to nearest-even and clipped to its type range.
    valid : xarray.DataArray
        Boolean VoxelData mask of valid moving-image sample centers.
    transforms : xarray.DataArray
        Homogeneous world z/y/x pull matrices, indexed by time.
    diagnostics : xarray.Dataset
        Per-frame objective, iteration count, status and intrinsic motion.
    """

    registered: xr.DataArray
    valid: xr.DataArray
    transforms: xr.DataArray
    diagnostics: xr.Dataset


def _optimize(
    images: torch.Tensor,
    fixed: torch.Tensor,
    plane: Plane,
    reference: torch.Tensor,
    max_iterations: int,
    learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run masked batched Adam without host reads in the iteration loop.

    Parameters
    ----------
    images : torch.Tensor
        Batched fitting images.
    fixed : torch.Tensor
        Shared fixed fitting image.
    plane : Plane
        Resident sampling geometry.
    reference : torch.Tensor
        Boolean mask of frames that must stay exactly identity.
    max_iterations : int
        Maximum number of parameter updates.
    learning_rate : float
        Step size in physically scaled parameters.

    Returns
    -------
    parameters : torch.Tensor
        Best valid angle/mm parameters per frame.
    steps : torch.Tensor
        Number of optimizer updates per frame.
    status : torch.Tensor
        1 converged, 2 insufficient overlap, 3 invalid metric, 4 reference, 5 limit.
    """
    b = len(images)
    u = plane.parameter_scale.new_zeros((b, 3), requires_grad=True)
    best = torch.zeros_like(u)
    best_loss = u.new_full((b,), torch.inf)
    moment, variance = torch.zeros_like(u), torch.zeros_like(u)
    steps = torch.zeros(b, dtype=torch.int64, device=images.device)
    stable = torch.zeros_like(steps)
    constant = images.amax(dim=(1, 2)) == images.amin(dim=(1, 2))
    status = torch.where(reference, 4, torch.where(constant, 3, 0))
    for _ in range(max_iterations):
        moved, valid = sample_rigid(images, u * plane.parameter_scale, plane)
        loss, meaningful = global_ncc(moved, fixed, valid)
        overlap = valid.float().mean(dim=(1, 2)) >= 0.9
        good = overlap & meaningful & torch.isfinite(loss)
        active = status == 0
        with torch.no_grad():
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
            stable = torch.where(
                active & (delta.abs().amax(dim=1) < 1e-4), stable + 1, 0
            )
            status = torch.where(active & (steps >= 20) & (stable >= 5), 1, status)
        u = candidate.detach().requires_grad_(True)
    with torch.no_grad():
        moved, valid = sample_rigid(images, u * plane.parameter_scale, plane)
        loss, meaningful = global_ncc(moved, fixed, valid)
        overlap = valid.float().mean(dim=(1, 2)) >= 0.9
        good = overlap & meaningful & torch.isfinite(loss)
        eligible = (status == 0) | (status == 1)
        improved = eligible & good & (loss < best_loss)
        best = torch.where(improved[:, None], u, best)
        status = torch.where(eligible & ~overlap, 2, status)
        status = torch.where(eligible & overlap & ~good, 3, status)
        status = torch.where(status == 0, 5, status)
    return best * plane.parameter_scale, steps, status


def register_rigid(
    moving: xr.DataArray,
    fixed: xr.DataArray | None = None,
    *,
    intensity_scaling: Literal["none", "db"] = "db",
    interpolation: Literal["linear", "nearest"] = "linear",
    batch_size: int = 32,
    max_iterations: int = 100,
    learning_rate: float = 0.1,
    device: Literal["cpu", "cuda"] = "cpu",
    progress: bool | Callable[[int, int], None] | None = True,
) -> RegistrationResult:
    """Register independent single-slice frames to a shared fixed VoxelData grid.

    Parameters
    ----------
    moving : xarray.DataArray
        Finite real NumPy-backed VoxelData, (time, k=1, j, i). Integer values must
        lie within [-2**53, 2**53] for exact conversion to floating-point samples.
    fixed : xarray.DataArray, optional
        Spatial-only VoxelData on exactly the moving grid. Uses the first frame
        when not provided; that reference frame is then copied exactly.
    intensity_scaling : {"none", "db"}, default: "db"
        Fitting preprocessing only; original intensities are resampled for output.
        dB requires nonnegative power; normalized zeros are floored at 1e-12.
    interpolation : {"linear", "nearest"}, default: "linear"
        Final original-image resampling only; fitting always uses linear sampling.
        Nearest is suitable for labels, not automatically for integer intensities.
    batch_size : int, default: 32
        Maximum number of independent frames per resident optimization chunk.
    max_iterations : int, default: 100
        Maximum Adam steps per frame.
    learning_rate : float, default: 0.1
        Step size in physically scaled rotation/translation parameters.
    device : {"cpu", "cuda"}, default: "cpu"
        Explicit Torch device; unavailable CUDA is an error, not a fallback.
    progress : bool or callable, default: True
        True displays ConfUSIus's Rich progress bar with frame counts, elapsed time
        and ETA. False or None disables reporting. A callable overrides the bar and
        receives ``(completed_frames, total_frames)`` after each batch's results
        reach the CPU, on either device. Counts include reference and failed frames,
        not optimizer updates. No preparation/final-assembly events are emitted.
        Callback exceptions propagate; reporting updates are outside stage timings.

    Returns
    -------
    result : RegistrationResult
        Registered original intensities and validity as VoxelData, world pull
        matrices, and per-frame diagnostics. Invalid frames retain their best
        valid transform, or identity if no valid metric was observed.

    Raises
    ------
    TypeError
        If intensities or the progress option have unsupported types.
    ValueError
        If geometry, input values or numerical options are unsupported.

    Notes
    -----
    Convergence requires five consecutive scaled parameter steps below 1e-4
    after at least twenty updates. Each frame has independent Adam moments,
    clock and stopping state. Physical geometry, optimizer state and NCC statistics
    use float64 to stabilize chunk-size comparisons; fitting images use float32.
    Original float64 and integer images are resampled in float64, other images in
    float32, then restored to their original dtype. Integer results use nearest-even
    rounding and range clipping. Metrics exclude invalid sample centers; outside
    output values use each original moving frame's minimum.
    """
    start = perf_counter()
    validate_voxeldata(moving)
    if moving.dims != ("time", *VOXEL_DIMS):
        raise ValueError("moving must have dimensions (time, k, j, i).")
    implicit_fixed = fixed is None
    if fixed is None:
        fixed = moving.isel(time=0, drop=True)
    validate_voxeldata(fixed)
    if fixed.dims != VOXEL_DIMS:
        raise ValueError("fixed must have dimensions (k, j, i), without time.")
    if not moving.xindexes["z"].equals(fixed.xindexes["z"]):
        raise ValueError(
            "fixed and moving must have identical native coordinates and physical geometry."
        )
    for name, data in (("moving", moving), ("fixed", fixed)):
        integer = np.issubdtype(data.dtype, np.integer)
        if not integer and not (
            np.issubdtype(data.dtype, np.floating) and data.dtype.itemsize <= 8
        ):
            raise TypeError(
                f"{name} intensities must be real integer or floating-point values up to float64."
            )
        if not np.isfinite(data.values).all():
            raise ValueError(f"{name} intensities must be finite.")
        if integer and (
            int(data.values.min()) < -(2**53) or int(data.values.max()) > 2**53
        ):
            raise ValueError(
                f"{name} integers must lie within [-2**53, 2**53] for exact floating-point representation."
            )
    if (
        "time" not in moving.coords
        or moving.time.dims != ("time",)
        or not moving.get_index("time").is_unique
    ):
        raise ValueError("moving must have unique one-dimensional time labels.")
    if (
        type(batch_size) is not int
        or batch_size < 1
        or type(max_iterations) is not int
        or max_iterations < 1
    ):
        raise ValueError("batch_size and max_iterations must be positive integers.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if device not in ("cpu", "cuda") or (
        device == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("device must be cpu or an available cuda device.")
    if intensity_scaling not in ("none", "db"):
        raise ValueError("intensity_scaling must be none or db.")
    if interpolation not in ("linear", "nearest"):
        raise ValueError("interpolation must be linear or nearest.")
    if (
        progress is not None
        and not isinstance(progress, bool)
        and not callable(progress)
    ):
        raise TypeError(
            "progress must be True, False, None, or a callable accepting (completed, total)."
        )
    # Double physical geometry/state keeps gradient reductions stable across chunk sizes.
    plane = make_plane(fixed, device=device, dtype=torch.float64)
    raw = moving.values[:, 0]
    integer_output = np.issubdtype(raw.dtype, np.integer)
    resampling_dtype = (
        np.float64 if integer_output or raw.dtype.itemsize > 4 else np.float32
    )
    # Normalize before downcasting so float64/integer values are not quantized first.
    preprocessing_dtype = (
        np.float64
        if any(
            np.issubdtype(data.dtype, np.integer) or data.dtype.itemsize > 4
            for data in (moving, fixed)
        )
        else np.float32
    )
    fit_raw = raw.astype(preprocessing_dtype)
    target_raw = fixed.values[0].astype(preprocessing_dtype)
    moving_scale = float(np.abs(fit_raw).max())
    scale = max(moving_scale, float(np.abs(target_raw).max()))
    if moving_scale == 0:
        raise ValueError("An all-zero recording cannot be registered.")
    if intensity_scaling == "db":
        if fit_raw.min() < 0 or target_raw.min() < 0:
            raise ValueError("dB scaling requires nonnegative power intensities.")
        fit = 10 * np.log10(np.maximum(fit_raw / scale, 1e-12))
        target = 10 * np.log10(np.maximum(target_raw / scale, 1e-12))
    else:
        fit, target = fit_raw / scale, target_raw / scale
    if preprocessing_dtype == np.float64:
        # NCC is affine-intensity invariant; keep low-contrast detail before float32 conversion.
        for values in (fit, target):
            values -= values[..., :1, :1]
            amplitude = np.abs(values).max(axis=(-2, -1), keepdims=True)
            values /= np.where(amplitude > 0, amplitude, 1)
    fit, target = fit.astype(np.float32), target.astype(np.float32)
    if np.ptp(target) == 0:
        raise ValueError("The fixed image must not be constant.")
    timings = {
        "prepare": perf_counter() - start,
        "upload": 0.0,
        "optimize": 0.0,
        "resample_download": 0.0,
    }
    outputs, masks, parameters, clocks, statuses, objectives, overlaps = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
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
            if device == "cuda":
                torch.cuda.synchronize()
            timings["upload"] += perf_counter() - start
            start = perf_counter()
            physical, steps, status = _optimize(
                images, target_tensor, plane, reference, max_iterations, learning_rate
            )
            if device == "cuda":
                torch.cuda.synchronize()
            timings["optimize"] += perf_counter() - start
            start = perf_counter()
            with torch.no_grad():
                original = torch.tensor(
                    raw[first:last].astype(resampling_dtype, copy=False),
                    device=device,
                )
                sampled, valid = sample_rigid(
                    original, physical, plane, interpolation=interpolation
                )
                fills = original.amin(dim=(1, 2))[:, None, None]
                sampled = torch.where(valid, sampled, fills)
                sampled = torch.where(reference[:, None, None], original, sampled)
                valid = valid | reference[:, None, None]
                fitted, fit_valid = sample_rigid(images, physical, plane)
                loss, meaningful = global_ncc(fitted, target_tensor, fit_valid)
                nonconstant = images.amax(dim=(1, 2)) > images.amin(dim=(1, 2))
                loss = torch.where(meaningful & nonconstant, loss, torch.nan)
                values = sampled.cpu().numpy()
                if integer_output:
                    limits = np.iinfo(raw.dtype)
                    values = np.clip(np.rint(values), limits.min, limits.max)
                outputs.append(values.astype(raw.dtype, copy=False))
                masks.append(valid.cpu().numpy())
                parameters.append(physical.cpu().numpy())
                clocks.append(steps.cpu().numpy())
                statuses.append(status.cpu().numpy())
                objectives.append(loss.cpu().numpy())
                overlaps.append(valid.float().mean(dim=(1, 2)).cpu().numpy())
            timings["resample_download"] += perf_counter() - start
            if progress is True:
                advance(last - first)
            elif callable(progress):
                progress(last, len(raw))
    params = np.concatenate(parameters)
    labels = np.array(
        [
            "running",
            "converged",
            "insufficient_overlap",
            "invalid_metric",
            "reference",
            "max_iterations",
        ]
    )
    diagnostics = xr.Dataset(
        {
            "objective": ("time", np.concatenate(objectives)),
            "valid_fraction": ("time", np.concatenate(overlaps)),
            "n_iterations": ("time", np.concatenate(clocks)),
            "status": ("time", labels[np.concatenate(statuses)]),
            "angle": (
                "time",
                params[:, 0],
                {"units": "rad", "basis": "intrinsic i/j plane"},
            ),
            "translation_i": (
                "time",
                params[:, 1],
                {
                    "units": "mm",
                    "meaning": "center displacement along intrinsic unit i",
                },
            ),
            "translation_j": (
                "time",
                params[:, 2],
                {
                    "units": "mm",
                    "meaning": "center displacement along orthogonalized intrinsic j",
                },
            ),
        },
        coords={"time": moving.time},
        attrs={"timings_seconds": timings},
    )
    axes = ["z", "y", "x", "1"]
    transforms = xr.DataArray(
        world_matrices(params, plane),
        dims=("time", "world_out", "world_in"),
        coords={"time": moving.time, "world_out": axes, "world_in": axes},
        attrs={"convention": "fixed-world-to-moving-world pull", "spatial_units": "mm"},
    )
    valid_data = moving.copy(data=np.concatenate(masks)[:, None])
    valid_data.attrs = {"description": "Valid moving-image sample centers"}
    return RegistrationResult(
        moving.copy(data=np.concatenate(outputs)[:, None]),
        valid_data,
        transforms,
        diagnostics,
    )
