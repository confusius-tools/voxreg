"""VoxelData registration and shared input/output contracts."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from confusius._utils.progress import progress_bar

from voxreg.convergence import WindowConvergence
from voxreg.geometry import (
    VOXEL_DIMS,
    _center_positions,
    _plane_alignment,
    _validate_registration_geometry,
    position_to_world,
)
from voxreg.sampling import (
    Plane,
    SpatialGrid,
    _initial_parameters,
    global_ncc,
    make_grid,
    sample_rigid,
    world_matrices,
)
from voxreg.transforms import (
    affine_properties,
    configure_transform,
    linear_diagnostics,
)


class _MetricOptions(TypedDict, total=False):
    metric_callback: Callable[[torch.Tensor, torch.Tensor], None]
    metric_interval: int


@dataclass(frozen=True)
class RegistrationResult:
    """Labeled registration outputs, not a replacement spatial container.

    Attributes
    ----------
    registered : xarray.DataArray
        Original intensities resampled on the fixed grid, retaining moving dtype.
        Integer output is rounded to nearest-even and clipped to its type range.
    valid : xarray.DataArray
        Boolean VoxelData mask of valid moving-image sample centers.
    transforms : xarray.DataArray, optional
        None unless register(return_transforms=True). Linear models: homogeneous world z/y/x pull matrices. SyN: complete
        fixed-world-to-moving-world displacement in mm on the fixed grid, with
        a final world_component axis labeled z/y/x. Recordings retain time.
    diagnostics : xarray.Dataset
        Objective, update count and status, plus linear motion parameters or
        SyN regularity/inverse-consistency diagnostics.
    inverse_transforms : xarray.DataArray, optional
        SyN only: complete moving-world-to-fixed-world displacement in mm on
        the original moving grid. Sampled inverses are approximate. None for
        linear models or return_transforms=False; exported linear matrices can
        be inverted directly.
    """

    registered: xr.DataArray
    valid: xr.DataArray
    transforms: xr.DataArray | None
    diagnostics: xr.Dataset
    inverse_transforms: xr.DataArray | None = None


def _fitting_images(raw, target_raw, intensity_scaling):
    """Shared preprocessing; preserve float64/integer detail before float32 fitting."""
    spatial_axes = tuple(range(1, raw.ndim))
    preprocessing_dtype = (
        np.float64
        if any(
            np.issubdtype(data.dtype, np.integer) or data.dtype.itemsize > 4
            for data in (raw, target_raw)
        )
        else np.float32
    )
    fit_raw = raw.astype(preprocessing_dtype)
    target_raw = target_raw.astype(preprocessing_dtype)
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
        for values in (fit, target):
            corner = (slice(None),) * (values.ndim - len(spatial_axes)) + (
                slice(0, 1),
            ) * len(spatial_axes)
            values -= values[corner]
            amplitude = np.abs(values).max(
                axis=tuple(range(-len(spatial_axes), 0)), keepdims=True
            )
            values /= np.where(amplitude > 0, amplitude, 1)
    fit, target = fit.astype(np.float32), target.astype(np.float32)
    if np.ptp(target) == 0:
        raise ValueError("The fixed image must not be constant.")
    return fit, target


def _output_template(moving, fixed, same_grid):
    if same_grid:
        return moving
    template = fixed.drop_vars("time", errors="ignore").expand_dims(time=moving.time)
    template = template.assign_coords(
        {
            name: coord
            for name, coord in moving.coords.items()
            if coord.dims == ("time",)
        }
    )
    template.attrs = moving.attrs.copy()
    template.name = moving.name
    return template


def _all_frames_stopped(status: torch.Tensor) -> bool:
    """Read one batch-wide scalar; on CUDA this synchronizes the device stream."""
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

    Returns
    -------
    parameters : torch.Tensor
        Best valid radian/mm parameters per frame.
    steps : torch.Tensor
        Number of optimizer updates per frame.
    status : torch.Tensor
        1 converged, 2 insufficient overlap, 3 invalid metric, 4 reference, 5 limit,
        6 invalid affine transform.
    """
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
    check_interval = 10 if images.is_cuda else 1
    orientation = (
        affine_properties(u.detach() * plane.parameter_scale, plane)[1]
        if plane.transform == "affine"
        else None
    )
    for iteration in range(max_iterations):
        moved, valid = sample_rigid(images, u * plane.parameter_scale, plane)
        loss, meaningful = global_ncc(moved, fixed, valid)
        if metric_callback is not None and iteration % metric_interval == 0:
            metric_callback(
                steps, torch.where(meaningful & ~constant, loss.detach(), torch.nan)
            )
        overlap = valid.sum(dim=spatial_axes) > 1
        good = overlap & meaningful & torch.isfinite(loss)
        active = status == 0
        with torch.no_grad():
            if orientation is not None:
                admissible, sign = affine_properties(
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
            and _all_frames_stopped(status)
        ):
            break
    with torch.no_grad():
        moved, valid = sample_rigid(images, u * plane.parameter_scale, plane)
        loss, meaningful = global_ncc(moved, fixed, valid)
        if metric_callback is not None:
            metric_callback(steps, torch.where(meaningful & ~constant, loss, torch.nan))
        overlap = valid.sum(dim=spatial_axes) > 1
        good = overlap & meaningful & torch.isfinite(loss)
        eligible = (status == 0) | (status == 1)
        if orientation is not None:
            admissible, sign = affine_properties(u * plane.parameter_scale, plane)
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


def _automatic_batch_size(moving, fixed, transform, device):
    """Conservative working-set estimate, not an optimizer probe or OOM retry."""
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


def register(
    moving: xr.DataArray,
    fixed: xr.DataArray | None = None,
    *,
    transform: Literal["translation", "rigid", "affine", "syn"] = "rigid",
    initial_transform: npt.ArrayLike | xr.DataArray | None = None,
    initialization: Literal["center_geometry", "center_moments", "align_planes"]
    | npt.ArrayLike
    | xr.DataArray
    | None = None,
    intensity_scaling: Literal["none", "db"] = "db",
    interpolation: Literal["linear", "nearest"] = "linear",
    return_transforms: bool = False,
    batch_size: int | None = None,
    max_iterations: int = 100,
    learning_rate: float = 0.1,
    convergence_window_size: int = 50,
    convergence_minimum_value: float = 1e-6,
    device: Literal["cpu", "cuda"] = "cpu",
    progress: Literal["auto"] | bool | Callable[[int, int], None] | None = "auto",
    metric_callback: Callable[
        [npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]], None
    ]
    | None = None,
    metric_interval: int = 10,
) -> RegistrationResult:
    """Register a volume or independent frames onto a fixed physical VoxelData grid.

    Parameters
    ----------
    moving : xarray.DataArray
        Finite real NumPy-backed VoxelData, (k, j, i) or (time, k, j, i).
        Slice pairs use k=1; volume pairs require all spatial axes >= 2.
        Integer values must lie within [-2**53, 2**53] for exact float conversion.
    fixed : xarray.DataArray, optional
        Spatial-only VoxelData with independent shape, spacing, origin and
        orientation. Required for spatial-only moving input. For recordings,
        omission uses the first frame and preserves that reference exactly.
        Noncoplanar slice pairs require a plane-compatible seed or align_planes.
        Mixed slice/volume inputs sample the moving image on the native fixed grid.
    transform : {"translation", "rigid", "affine", "syn"}, default: "rigid"
        Translation/rigid/affine use 2/3/6 in-plane parameters for slice pairs,
        3/6/12 for volume pairs, and 3/6/9 for mixed slice/volume inputs.
        Euler belongs to the rigid family; the rigid model uses rotation vectors,
        not ITK's Euler optimizer parameterization. Affine refines the full linear
        map, including scale and shear, rather than freezing an affine prefix.
        SyN uses two smoothed compositive midpoint-to-image fields on one native
        resolution, with local NCC radius 2, update sigma 1.5 and total sigma 0.5
        voxels. It supports two k=1 planes or two volumes, not mixed-dimensional
        deformation. Its complete forward/inverse outputs are dense displacements,
        not matrices. Physical geometry and field state remain float64.
    initial_transform : array-like or xarray.DataArray, optional
        Model-compatible fixed-world-to-moving-world pull matrix in world z/y/x,
        with translations in mm. Shape (4, 4) broadcasts to all frames;
        (N, 4, 4) follows moving frame order. SyN accepts an orientation-preserving
        affine seed as its global prealignment; the dense result includes that
        seed. Dense restart seeds are not supported. Labeled arrays must use
        (time, world_out, world_in) or (world_out, world_in); supplied axis/time
        labels must match. Copied, never modified, like ITK InitialTransform
        with InPlaceOff: initializes the optimized linear transform, not a frozen
        pre-transform. For SyN the global affine is fixed during field fitting,
        following the usual affine-then-SyN workflow. Rigid seeds require proper rotations (float32 roundoff
        within 1e-6 is normalized); translation seeds require an identity linear
        part. Affine seeds must pass the normalized determinant guard. Negative
        determinant affine seeds are supported, but fitting preserves their
        orientation sign. For two slices, seeds must map the fixed plane into the
        moving plane with the supported normal completion. Mixed affine seeds
        require unit orthogonal normal completion.
        For an implicit first-frame reference its initial transform must be identity;
        use explicit fixed to optimize every frame from a shared nonidentity pose.
    initialization : {"center_geometry", "center_moments", "align_planes"} or array-like, optional
        Named centering mode or model-compatible matrix seed, mutually exclusive
        with initial_transform. Matrices alias initial_transform with the same
        shape, world-coordinate, label and model requirements.
        Geometry aligns physical grid centers. Moments aligns
        original-intensity centers of gravity and uses the fixed centroid as the
        rotation center, including physical lever-arm scaling, following ITK.
        Moments requires nonnegative original values with positive mass in every
        image. Named modes estimate no rotation; the implicit reference stays
        exactly identity.
    intensity_scaling : {"none", "db"}, default: "db"
        Fitting preprocessing only; original intensities are resampled for output.
        dB requires nonnegative power; normalized zeros are floored at 1e-12.
        Pass original linear intensities; for dB fitting use this option instead
        of scaling the inputs yourself. Logarithms, square roots and other nonlinear
        preprocessing change center_moments weights. Already-transformed data needs
        center_geometry or an explicit seed instead. Only fitting is scaled;
        initialization and original-intensity output remain unscaled.
    interpolation : {"linear", "nearest"}, default: "linear"
        Final original-image resampling only; fitting always uses linear sampling.
        Nearest is suitable for labels, not automatically for integer intensities.
    return_transforms : bool, default: False
        Return linear pull matrices or both complete SyN displacement fields.
        Otherwise transforms and inverse_transforms are None; fitting, resampling
        and diagnostics are unchanged. Dense fields are not exported or accumulated.
        Warning: on equal fixed/moving grids, the two float64, three-component
        SyN fields require 12x the uncompressed image storage of a float32 recording
        (48 bytes per voxel versus 4), plus temporary assembly memory. For example,
        a 525 MB recording needs about 6.3 GB for fields alone. Different grid sizes
        or input dtypes change this ratio. batch_size does not bound returned fields.
    batch_size : int, optional
        Maximum number of independent frames per resident optimization chunk.
        None (default) chooses automatically: CPU uses up to 32 frames; CUDA uses
        available VRAM (including reusable Torch cache), the larger image size and
        a conservative method-specific working-set estimate with 50% headroom.
        Performance caps are 256 for CUDA linear slice pairs, 32 for SyN or volumes;
        automatic CUDA batches round down to a power of two. diagnostics.attrs
        records the selected batch_size. This is not a runtime autotuner or a
        guarantee against OOM under concurrent allocations; one frame must fit.
        An explicit positive integer overrides this policy. Returned images/masks
        and opt-in transforms still require full-recording CPU storage.
    max_iterations : int, default: 100
        Maximum optimizer updates per frame (Adam for linear models, smoothed
        compositive gradient descent for SyN).
    learning_rate : float, default: 0.1
        Step size in physically scaled parameters. Affine coefficients use
        minimum spacing divided by the fixed-domain RMS input-axis lever arm.
        For SyN this bounds each halfway update's maximum voxel-coordinate norm.
    convergence_window_size : int, default: 50
        ITK-style metric-history window, at least two values. A window larger
        than max_iterations disables convergence checks for a fixed-budget run.
    convergence_minimum_value : float, default: 1e-6
        Stop when ITK's normalized B-spline improvement trend is at or below this
        nonnegative threshold. This is not a raw NCC difference or parameter step.
    device : {"cpu", "cuda"}, default: "cpu"
        Explicit Torch device; unavailable CUDA is an error, not a fallback.
    progress : {"auto"} or bool or callable, default: "auto"
        Auto shows the frame bar only for multiple moving frames, remaining quiet
        for a spatial pair or a single-frame recording. True explicitly displays
        ConfUSIus's Rich progress bar with frame counts, elapsed time and ETA even
        for a single pair.
        False or None disables reporting. A callable overrides the bar and
        receives ``(completed_frames, total_frames)`` after each batch's results
        reach the CPU, on either device. Counts include reference and failed frames,
        not optimizer updates. No preparation/final-assembly events are emitted.
        Callback exceptions propagate; reporting updates are outside stage timings.
    metric_callback : callable, optional
        Live optimizer reporting, independent of frame progress. Receives owned
        NumPy arrays ``(frame_indices, n_iterations, objective)`` for each chunk;
        indices follow moving frame order, clocks count each frame's updates.
        Reports the initial trial, every metric_interval optimizer rounds, and
        the final trial. Stopped rows may repeat. Objective is raw negative NCC²
        (negative mean squared local NCC between halfway images for SyN),
        NaN for unusable variance/support; lower is better. The returned best
        valid pose can differ from the final trial. No history is retained unless
        the callback stores it. Exceptions propagate. CPU copies and CUDA
        synchronization are opt-in and included in optimization timing.
    metric_interval : int, default: 10
        Positive optimizer-round reporting cadence when metric_callback is set.
        Set to 1 for a diagnostic trace; throttle plot redraws separately.
        With no metric callback, no per-round host transfers are added.

    Returns
    -------
    RegistrationResult
        Registered intensities and validity on the fixed grid, diagnostics, and
        optional world pull transforms (None by default). Spatial-only input has no synthetic time
        label: linear transforms are (4, 4), SyN fields are (k, j, i, world_component).
        SyN inverse_transforms use the original moving grid and the reverse pull.
        Invalid frames retain their best
        valid transform, or the initial transform if no valid metric was observed.
        Motion diagnostics describe the full transform, not an increment.

    Raises
    ------
    TypeError
        If intensities, progress or return_transforms have unsupported types.
    ValueError
        If geometry, input values or numerical options are unsupported.

    Notes
    -----
    Optimization starts at initial_transform or the selected automatic seed
    (identity when neither is provided) in physical
    world space, not voxel coordinates. Sampling composes
    inverse(A_moving) @ T @ A_fixed, without pre-resampling the input. Rigid volumes
    use a centered axis-angle rotation vector (right-handed XYZ, stored z/y/x).
    Affine parameters are row-major deviations from identity plus physical center
    displacement; diagnostics report actual linear coefficients, not deviations.
    Slice pairs optimize only their intrinsic orthonormal i/j plane, leaving its normal
    fixed. Linear models share their optimizer; SyN shares preprocessing, physical
    sampling, output contracts and independent convergence monitoring.
    align_planes explicitly assumes corresponding sections: align centers and
    unoriented plane normals, then refine in-plane. Centering alone corrects
    parallel-plane offsets, not tilt. Translation cannot align nonparallel planes.
    Mixed slice/volume fitting always uses the caller's native fixed grid, without
    reversing roles or replicating slices. Rigid has six pose parameters. Affine
    retains nine tangent/translation parameters: unit orthogonal normal completion
    constrains the physical slice-to-volume map, inverted when moving is the slice.
    Unidentifiable affine normal scale/shear seeds are rejected. Mixed fits use the
    geometric fixed-image pivot; moments changes the seed, not that pivot.
    Mixed source support is half-open [-0.5, size-0.5) on every voxel axis, with
    border-clamped interpolation. A singleton axis has constant intensity over its
    finite voxel extent, not measured through-plane variation. An empty moving-slice
    support mask warns. valid_fraction and output_valid_fraction (when present)
    describe the same fixed-grid domain, not confidence or anatomical correctness.
    Swapping inputs changes the fitting problem and need not return inverse poses.
    Exported transforms retain the fixed-world-to-moving-world pull convention.
    SyN operates on two independent halfway fields and composes them through
    numerically estimated inverses. It returns the best usable halfway-metric
    state, not necessarily its last trial; endpoint ncc_squared is a separate
    diagnostic. Positive sampled Jacobians are enforced for accepted states but
    are not a proof of global invertibility; inverse_error reports two-sided
    residual-map error on the fixed virtual lattice, before the affine seed.
    Final composition failure falls back to the seed with invalid_transform.
    Separate arbitrary ITK-style pre-transform chains remain unsupported.
    Fewer than two valid sample centers flags insufficient overlap; zero/invalid
    variance flags an invalid metric. No fixed coverage percentage is required.
    valid_fraction reports coverage, not confidence: tiny overlaps can give
    misleading correlations or fail to identify the transform. Physical scaling
    uses minimum fixed spacing and RMS lever arms, not ITK's estimator.
    Affine candidates with normalized |determinant| <= 1e-8 or a changed orientation
    sign stop only that frame with invalid_transform, retaining its best valid pose.
    Normalization divides the linear map by its largest absolute entry first.
    This guard prevents numerical singularity, not implausible anatomical scaling.
    The default convergence threshold was calibrated on rigid slices, not affine
    models or general 3D.

    Convergence follows ITK v4's accumulated-absolute-energy normalization and
    first-order B-spline window gradient, checked before the next Adam update.
    Each frame has independent metric history, Adam moments, clock and stopping
    state. After a full history window, whole-batch completion is checked each
    iteration on CPU and every ten iterations on CUDA, using one batch-wide scalar
    read (a CUDA stream synchronization). These checks do not alter per-frame
    decisions; frozen frames remain resident until the entire batch finishes.
    Fitting images and NCC statistics use float32. Physical geometry and
    optimizer state use float64. Benchmark accuracy is measured against known
    truth independently for each device, not by cross-device map agreement.
    Original float64 and integer images are resampled in float64, other images in
    float32, then restored to their original dtype. Integer results use nearest-even
    rounding and range clipping. Metrics exclude invalid sample centers; outside
    output values use each original moving frame's minimum.
    """
    start = perf_counter()
    _validate_registration_geometry(moving)
    if not isinstance(transform, str) or transform not in (
        "translation",
        "rigid",
        "affine",
        "syn",
    ):
        raise ValueError(
            "transform must be translation, rigid, affine or syn. Euler transforms belong to the rigid family."
        )
    spatial_input = moving.dims == VOXEL_DIMS
    if spatial_input:
        if fixed is None:
            raise ValueError("fixed is required for spatial-only input without time.")
        moving = moving.expand_dims(time=[0])
    if moving.dims != ("time", *VOXEL_DIMS):
        raise ValueError("moving must have dimensions (time, k, j, i).")
    implicit_fixed = fixed is None
    if fixed is None:
        fixed = moving.isel(time=0, drop=True)
    _validate_registration_geometry(fixed)
    if fixed.dims != VOXEL_DIMS:
        raise ValueError("fixed must have dimensions (k, j, i), without time.")
    same_grid = moving.xindexes["z"].equals(fixed.xindexes["z"])
    if transform == "syn" and (moving.sizes["k"] == 1) != (fixed.sizes["k"] == 1):
        raise ValueError(
            "SyN requires two native k=1 planes or two 3D volumes; mixed-dimensional deformation is unsupported."
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
        (batch_size is not None and (type(batch_size) is not int or batch_size < 1))
        or type(max_iterations) is not int
        or max_iterations < 1
    ):
        raise ValueError(
            "batch_size must be None or a positive integer; max_iterations must be a positive integer."
        )
    if type(convergence_window_size) is not int or convergence_window_size < 2:
        raise ValueError("convergence_window_size must be an integer of at least two.")
    if (
        not isinstance(convergence_minimum_value, (int, float, np.integer, np.floating))
        or isinstance(convergence_minimum_value, bool)
        or not np.isfinite(convergence_minimum_value)
        or convergence_minimum_value < 0
    ):
        raise ValueError("convergence_minimum_value must be finite and nonnegative.")
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
        and not (isinstance(progress, str) and progress == "auto")
    ):
        raise TypeError(
            "progress must be True, False, None, 'auto', or a callable accepting (completed, total)."
        )
    if type(return_transforms) is not bool:
        raise TypeError("return_transforms must be a bool.")
    if metric_callback is not None and not callable(metric_callback):
        raise TypeError("metric_callback must be callable or None.")
    if type(metric_interval) is not int or metric_interval < 1:
        raise ValueError("metric_interval must be a positive integer.")
    if batch_size is None:
        batch_size = _automatic_batch_size(moving, fixed, transform, device)
    if isinstance(progress, str):
        progress = moving.sizes["time"] > 1
    if initialization is not None and initial_transform is not None:
        raise ValueError("Use initialization or initial_transform, not both.")
    if initialization is not None:
        if isinstance(initialization, str):
            if initialization not in (
                "center_geometry",
                "center_moments",
                "align_planes",
            ):
                raise ValueError(
                    "initialization must be None, center_geometry, center_moments, align_planes or a model-compatible matrix."
                )
        else:
            if np.ndim(initialization) not in (2, 3):
                raise ValueError(
                    "initialization must be a model-compatible (4, 4) or (N, 4, 4) matrix, or a named mode."
                )
            initial_transform, initialization = initialization, None
    center = None
    if initialization is not None:
        if initialization == "align_planes" and (
            fixed.sizes["k"] != 1 or moving.sizes["k"] != 1
        ):
            raise ValueError("align_planes is only defined for two k=1 slices.")
        mode = "center_geometry" if initialization == "align_planes" else initialization
        fixed_position = _center_positions(fixed, mode, "fixed")[0]
        moving_positions = _center_positions(moving, mode, "moving")
        af, am = position_to_world(fixed), position_to_world(moving)
        fixed_offset = af[:3, :3] @ fixed_position
        initial_transform = np.broadcast_to(
            np.eye(4), (moving.sizes["time"], 4, 4)
        ).copy()
        # Subtract origins before local offsets to avoid common-large-origin cancellation.
        initial_transform[:, :3, 3] = (
            am[:3, 3] - af[:3, 3] + moving_positions @ am[:3, :3].T - fixed_offset
        )
        if initialization == "align_planes":
            rotation, _ = _plane_alignment(fixed, moving)
            initial_transform[:, :3, :3] = rotation
            initial_transform[:, :3, 3] -= (rotation - np.eye(3)) @ (
                af[:3, 3] + fixed_offset
            )
        if initialization == "center_moments" and (
            (fixed.sizes["k"] == 1) == (moving.sizes["k"] == 1)
        ):
            center = af[:3, 3] + fixed_offset
    # Double physical geometry/state keeps gradient reductions stable across chunk sizes.
    plane = make_grid(
        fixed,
        moving,
        device=device,
        dtype=torch.float64,
        center=center,
        plane_seed=initial_transform,
    )
    if (
        transform == "translation"
        and isinstance(plane, SpatialGrid)
        and plane.plane_rotation is not None
        and not np.allclose(plane.plane_rotation, np.eye(3), atol=1e-6, rtol=0)
    ):
        raise ValueError(
            "Translation cannot align nonparallel slice planes; choose rigid or affine."
        )
    plane = configure_transform(
        plane, fixed, "affine" if transform == "syn" else transform
    )
    initial = None
    if initial_transform is not None:
        if isinstance(initial_transform, xr.DataArray):
            if initial_transform.dims not in (
                ("world_out", "world_in"),
                ("time", "world_out", "world_in"),
            ):
                raise ValueError(
                    "initial_transform has unsupported labeled dimensions."
                )
            for axis in ("world_out", "world_in"):
                if axis in initial_transform.coords and not np.array_equal(
                    initial_transform[axis], ["z", "y", "x", "1"]
                ):
                    raise ValueError("initial_transform world axes must be z/y/x/1.")
            if (
                "time" in initial_transform.dims
                and "time" in initial_transform.coords
                and not initial_transform.get_index("time").equals(
                    moving.get_index("time")
                )
            ):
                raise ValueError(
                    "initial_transform time labels must match moving in order."
                )
            for key, expected in (
                ("spatial_units", "mm"),
                ("convention", "fixed-world-to-moving-world pull"),
            ):
                if (
                    key in initial_transform.attrs
                    and initial_transform.attrs[key] != expected
                ):
                    raise ValueError(f"initial_transform {key} must be {expected!r}.")
        initial = _initial_parameters(initial_transform, plane, moving.sizes["time"])
        if implicit_fixed:
            if not np.allclose(initial[0], 0, rtol=0, atol=1e-12):
                raise ValueError(
                    "The implicit reference initial_transform must be identity; supply explicit fixed otherwise."
                )
            initial[0] = 0
    planar = isinstance(plane, Plane) or plane.planar
    raw = moving.values[:, 0] if planar else moving.values
    spatial_axes = tuple(range(1, raw.ndim))
    reference_shape = (slice(None),) + (None,) * len(spatial_axes)
    integer_output = np.issubdtype(raw.dtype, np.integer)
    resampling_dtype = (
        np.float64 if integer_output or raw.dtype.itemsize > 4 else np.float32
    )
    fit, target = _fitting_images(
        raw, fixed.values[0] if planar else fixed.values, intensity_scaling
    )
    timings = {
        "prepare": perf_counter() - start,
        "upload": 0.0,
        "optimize": 0.0,
        "resample_download": 0.0,
    }
    if transform == "syn":
        from voxreg.syn import register_syn

        return register_syn(
            moving,
            fixed,
            fit,
            target,
            plane,
            initial,
            implicit_fixed=implicit_fixed,
            return_transforms=return_transforms,
            spatial_input=spatial_input,
            same_grid=same_grid,
            timings=timings,
            batch_size=batch_size,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            convergence_window_size=convergence_window_size,
            convergence_minimum_value=convergence_minimum_value,
            device=device,
            progress=progress,
            metric_callback=metric_callback,
            metric_interval=metric_interval,
            interpolation=interpolation,
            update_sigma=1.5,
            total_sigma=0.5,
            radius=2,
        )
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
            if metric_callback is not None:

                def report_metrics(clocks, objective, first=first, last=last):
                    metric_callback(
                        np.arange(first, last, dtype=np.int64),
                        clocks.detach().cpu().numpy().copy(),
                        objective.detach().double().cpu().numpy().copy(),
                    )

                metric_options = {
                    "metric_callback": report_metrics,
                    "metric_interval": metric_interval,
                }
            physical, steps, status = _optimize(
                images,
                target_tensor,
                plane,
                reference,
                max_iterations,
                learning_rate,
                convergence_window_size=convergence_window_size,
                convergence_minimum_value=convergence_minimum_value,
                initial_parameters=initial_tensor,
                **metric_options,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            timings["optimize"] += perf_counter() - start
            start = perf_counter()
            with torch.no_grad():
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
                    sampled = torch.where(reference[reference_shape], original, sampled)
                    valid = valid | reference[reference_shape]
                fitted, fit_valid = sample_rigid(images, physical, plane)
                loss, meaningful = global_ncc(fitted, target_tensor, fit_valid)
                nonconstant = images.amax(dim=spatial_axes) > images.amin(
                    dim=spatial_axes
                )
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
                overlaps.append(valid.float().mean(dim=spatial_axes).cpu().numpy())
            timings["resample_download"] += perf_counter() - start
            if progress is True:
                advance(last - first)
            elif callable(progress):
                progress(last, len(raw))
    params = np.concatenate(parameters)
    matrices = world_matrices(params, plane)
    diagnostic_params, diagnostic_plane = params, plane
    mixed = isinstance(plane, SpatialGrid) and plane.slice_basis is not None
    if mixed and transform == "affine":
        shifts = matrices[:, :3, 3] + np.einsum(
            "bij,j->bi", matrices[:, :3, :3] - np.eye(3), plane.center
        )
        diagnostic_params = np.column_stack(
            ((matrices[:, :3, :3] - np.eye(3)).reshape(-1, 9), shifts)
        )
        diagnostic_plane = replace(plane, slice_basis=None, resident_slice_basis=None)
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
    motion = (
        {}
        if transform != "rigid"
        else (
            {
                "angle": (
                    "time",
                    diagnostic_params[:, 0],
                    {"units": "rad", "basis": "intrinsic i/j plane"},
                ),
                "translation_i": (
                    "time",
                    diagnostic_params[:, 1],
                    {
                        "units": "mm",
                        "meaning": "center displacement along intrinsic unit i",
                    },
                ),
                "translation_j": (
                    "time",
                    diagnostic_params[:, 2],
                    {
                        "units": "mm",
                        "meaning": "center displacement along orthogonalized intrinsic j",
                    },
                ),
            }
            if planar
            else {
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
        )
    )
    if transform != "rigid":
        motion = linear_diagnostics(diagnostic_params, diagnostic_plane)
    diagnostics = xr.Dataset(
        {
            "objective": ("time", np.concatenate(objectives)),
            "valid_fraction": ("time", np.concatenate(overlaps)),
            "n_iterations": ("time", np.concatenate(clocks)),
            "status": ("time", labels[np.concatenate(statuses)]),
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
                np.concatenate(masks).reshape(len(raw), -1).mean(axis=1),
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
    template = _output_template(moving, fixed, same_grid)
    values, validity = np.concatenate(outputs), np.concatenate(masks)
    if planar:
        values, validity = values[:, None], validity[:, None]
    registered = template.copy(data=values)
    valid_data = template.copy(data=validity)
    valid_data.attrs = {"description": "Valid moving-image sample centers"}
    if spatial_input:
        return RegistrationResult(
            registered.isel(time=0, drop=True),
            valid_data.isel(time=0, drop=True),
            transforms.isel(time=0, drop=True) if transforms is not None else None,
            diagnostics.isel(time=0, drop=True),
        )
    return RegistrationResult(registered, valid_data, transforms, diagnostics)
