"""Public registration entry point, validation and model/stage dispatch."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from time import perf_counter
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr

from voxreg.geometry import (
    VOXEL_DIMS,
    compute_center_positions,
    compute_plane_alignment,
    position_to_world,
    validate_registration_geometry,
)
from voxreg.interpolation import Interpolation
from voxreg.masks import (
    validate_masks,
)
from voxreg.metrics import Metric, validate_metric
from voxreg.preparation import (
    IntensityScaling,
    PreparedRegistration,
    RegistrationSettings,
    choose_batch_size,
    prepare_fitting_images,
    resolve_device,
    validate_intensity_scaling,
)
from voxreg.results import LinearFit, RegistrationResult
from voxreg.sampling import (
    Plane,
    SpatialGrid,
    make_grid,
    recover_initial_parameters,
)
from voxreg.stages import Stage
from voxreg.transforms import (
    configure_transform,
)


def register(
    moving: xr.DataArray,
    fixed: xr.DataArray | None = None,
    *,
    transform: Literal["translation", "rigid", "affine", "syn"]
    | Stage
    | Sequence[str | Stage] = "rigid",
    fixed_mask: xr.DataArray | None = None,
    moving_mask: xr.DataArray | None = None,
    initial_transform: npt.ArrayLike | xr.DataArray | None = None,
    initialization: Literal["center_geometry", "center_moments", "align_planes"]
    | npt.ArrayLike
    | xr.DataArray
    | None = None,
    moving_intensity_scaling: IntensityScaling = None,
    fixed_intensity_scaling: IntensityScaling = None,
    metric: Literal["ncc", "local_ncc", "mi"] | None = None,
    mi_bins: int = 32,
    mi_sigma: float = 0.5,
    resample_interpolation: Interpolation = "linear",
    return_transforms: bool = False,
    batch_size: int | None = None,
    max_iterations: int | Sequence[int] = 100,
    shrink_factors: Sequence[int] = (1,),
    smoothing_sigmas: Sequence[float] = (0.0,),
    learning_rate: float = 0.1,
    syn_cc_radius: int = 2,
    syn_update_sigma: float = 1.5,
    syn_total_sigma: float = 0.5,
    convergence_window_size: int = 50,
    convergence_minimum_value: float = 1e-6,
    device: Literal["auto", "cpu", "cuda"] = "auto",
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
        World coordinates may use ConfUSIus length units (m, cm, mm, um/µm, nm);
        fixed, moving and masks may use different units. Geometry is normalized
        to mm. Output images retain the fixed grid's native coordinates and units.
        Slice pairs use k=1; volume pairs require all spatial axes >= 2.
        Integer values must lie within [-2**53, 2**53] for exact float conversion.
    fixed : xarray.DataArray, optional
        Spatial-only VoxelData with independent shape, spacing, origin and
        orientation. Required for spatial-only moving input. For recordings,
        omission uses the first frame and preserves that reference exactly.
        Noncoplanar slice pairs require a plane-compatible seed or align_planes.
        Mixed slice/volume inputs sample the moving image on the native fixed grid.
    transform : str, Stage or sequence of str or Stage, default: "rigid"
        Translation/rigid/affine use 2/3/6 in-plane parameters for slice pairs,
        3/6/12 for volume pairs, and 3/6/9 for mixed slice/volume inputs.
        Euler belongs to the rigid family; the rigid model uses rotation vectors,
        not ITK's Euler optimizer parameterization. Affine refines the full linear
        map, including scale and shear, rather than freezing an affine prefix.
        SyN uses two smoothed compositive midpoint-to-image fields, with
        configurable local NCC and Gaussian field smoothing.
        It supports two k=1 planes or two volumes, not mixed-dimensional
        deformation. Its complete forward/inverse outputs are dense displacements,
        not matrices. Working fields use float32; physical geometry, final
        inversion and exported displacements retain float64.
        A nonempty sequence runs increasing-complexity stages (translation,
        rigid, affine, syn), skipping models as needed. Repeated or reversed
        stages are rejected; SyN must be last. Stage specifications override
        shared fitting options, while strings inherit them. Shared syn_ settings
        apply only to SyN in pipelines containing it. No preset is selected.
    fixed_mask : xarray.DataArray, optional
        Spatial-only boolean ROI VoxelData in its own physical grid. Only
        included sample centers contribute to fitting statistics. Does not
        modify output intensities, output validity or automatic initialization.
    moving_mask : xarray.DataArray, optional
        Boolean ROI VoxelData sampled through the complete moving pull. A
        spatial mask broadcasts over frames; a time-dependent mask must match
        moving time labels exactly. Its grid may differ from the moving image.
        Both masks restrict every stage/level. Masks are evaluated in physical
        space at each level without Gaussian smoothing. Image interpolation,
        intensity scaling and image-pyramid smoothing retain their usual rules;
        image values outside an ROI can influence interpolation near its boundary.
    initial_transform : array-like or xarray.DataArray, optional
        Model-compatible fixed-world-to-moving-world pull matrix in world z/y/x,
        with translations in mm for bare arrays. Labelled spatial_units may
        declare any supported length unit; translations are converted to mm.
        Shape (4, 4) broadcasts to all frames;
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
    moving_intensity_scaling : {"log1p", "sqrt"} or float, optional
        Fitting-only transform; not provided means no nonlinear transform.
        log1p uses ln(1 + abs(x)), preserving zeros without a floor or reference.
        Its compression depends on the absolute intensity scale. sqrt and
        positive finite numeric exponents use abs(x)**exponent.
        Signed input is allowed; magnitude transforms discard its sign.
        Transforms are evaluated in float64 before metric conditioning and
        float32 fitting. Power overflow raises ValueError; underflow rounds to zero.
        Numerical conditioning is independent per image/frame before batching,
        including outside masks; there is no recording-wide intensity reference.
        Original intensities drive initialization and final resampling.
        Pass native intensities rather than pretransformed moment weights.
    fixed_intensity_scaling : {"log1p", "sqrt"} or float, optional
        Independent transform of an explicit fixed image, with the same rules.
        Not provided means no transform, not inheritance from moving scaling.
        Without explicit fixed, the first-frame reference uses moving scaling;
        specifying this argument then is invalid.
        The implicit reference output remains exactly the original first frame.
    metric : {"ncc", "local_ncc", "mi"}, optional
        Fitting objective. None preserves global squared NCC for linear models
        and rectangular local squared NCC for SyN. Explicit ncc selects global
        squared NCC for any model. local_ncc explicitly selects SyN local NCC.
        mi selects Gaussian-Parzen mutual information.
        MI independently maps each native fitting image's min/max to [0, 1]
        after the respective intensity transform, before pyramid smoothing. Ranges include the
        entire native image, including outside ROIs, and stay fixed during fitting.
        Original intensities and final interpolation remain unchanged.
    mi_bins : int, default: 32
        Histogram bins for MI, from 2 through 256, uniformly covering [0, 1].
    mi_sigma : float, default: 0.5
        MI Gaussian standard deviation in bin widths, from 0.1 through 10.
        Bin memberships are normalized per sample. These are intensity kernels,
        independent of image-pyramid and deformation smoothing.
    resample_interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Final original-image resampling only; fitting always uses linear sampling.
        bspline uses prefiltered cubic B-splines with mirror boundaries in 2D and
        3D,
        computed on the selected device with float64 coefficients and queries.
        Floating outputs may overshoot the input intensity range. Integer outputs
        are rounded nearest-even and clipped to their dtype range. Unsupported
        points keep the existing frame-minimum fill and support mask. Nearest
        is suitable for labels, not automatically for integer intensities.
    return_transforms : bool, default: False
        Return forward/inverse linear matrices or complete SyN displacement fields.
        Otherwise forward_transforms and inverse_transforms are None; fitting, resampling
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
    max_iterations : int or sequence of int, default: 100
        Maximum optimizer updates per frame at each pyramid level (Adam for
        linear models, smoothed compositive gradient descent for SyN). An integer
        repeats at every level; a sequence must match shrink_factors in length.
    shrink_factors : sequence of int, default: (1,)
        Positive integer downsampling factors, ordered coarse to fine
        (nonincreasing). Each axis retains at least two samples, except singleton
        slice axes, which remain singleton. Levels stride the original grids
        starting at index zero; trailing samples may be omitted. The final factor
        need not be one: output always uses the original fixed grid.
    smoothing_sigmas : sequence of float, default: (0.0,)
        Finite, nonnegative Gaussian image smoothing standard deviations in mm,
        one per shrink factor. Each level is smoothed directly from the native
        prepared fitting images before downsampling, with replicated edges.
        Zero disables image smoothing. Schedules must be supplied explicitly;
        no multi-resolution preset or automatic antialiasing is selected.
    learning_rate : float, default: 0.1
        Step size in physically scaled parameters. Affine coefficients use
        minimum spacing divided by the fixed-domain RMS input-axis lever arm.
        For SyN this bounds each halfway update's maximum voxel-coordinate norm.
    syn_cc_radius : int, default: 2
        Positive local NCC neighborhood radius in active voxel axes, using a
        uniformly weighted rectangular window of width 2 * syn_cc_radius + 1.
        A radius of one is allowed but can give unstable local statistics.
        Applies only to SyN; other models require the default value.
    syn_update_sigma : float, default: 1.5
        Finite, nonnegative Gaussian standard deviation in voxel units for
        smoothing each SyN gradient before update normalization and composition.
        Zero disables this smoothing. Applies only to SyN; other models require
        the default value. Units are current-level voxels, not mm. This is a
        standard deviation, not an ANTs variance.
    syn_total_sigma : float, default: 0.5
        Finite, nonnegative Gaussian standard deviation in voxel units for
        smoothing each accumulated SyN halfway field after composition.
        Zero disables this smoothing. Applies only to SyN; other models require
        the default value. Units are current-level voxels, not mm.
    convergence_window_size : int, default: 50
        ITK-style metric-history window, at least two values. A window larger
        than a level's iteration budget disables convergence checks at that level.
    convergence_minimum_value : float, default: 1e-6
        Stop when ITK's normalized B-spline improvement trend is at or below this
        nonnegative threshold. This is not a raw NCC difference or parameter step.
    device : {"auto", "cpu", "cuda"}, default: "auto"
        Auto selects CUDA when available, otherwise CPU, once per call. Explicit
        CPU bypasses CUDA selection; unavailable explicit CUDA is an error.
        CUDA execution errors, including out-of-memory, propagate without a CPU
        retry. The concrete device is recorded in diagnostics.attrs["device"]
        and in each stage's diagnostics. Outputs remain NumPy-backed on CPU.
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
        For pipelines, progress counts completed stage/frame work units, with
        total equal to frame count times stage count, including skipped frames.
    metric_callback : callable, optional
        Live optimizer reporting, independent of frame progress. Receives owned
        NumPy arrays ``(frame_indices, n_iterations, objective)`` for each chunk;
        indices follow moving frame order, clocks count each frame's updates.
        For pyramid fitting, reports arrive coarse to fine within each batch;
        clocks reset at each level. Losses from different levels are not comparable.
        Pipeline reports run stage by stage; clocks also reset at stage boundaries.
        Reports the initial trial, every metric_interval optimizer rounds, and
        the final trial. Stopped rows may repeat. Objective is the selected loss:
        negative NCC² (mean local NCC² for default SyN) or negative MI in nats,
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
        inverse_transforms reverse the pull: inverse matrices for linear models,
        or displacements on the original moving grid for SyN.
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
    Pipelines initialize once, retain the physical pivot, and pass complete world
    matrices between linear stages and into terminal SyN. Every stage fits the
    original images; original intensities are sampled only for the final output.
    A failed frame stops its pipeline after that stage and retains its best valid
    map; subsequent stage diagnostics mark it skipped. Other frames continue.
    Converged and max_iterations statuses allow continuation. Summary status
    preserves the original failure; stage_diagnostics contains separate datasets
    so stages can have different pyramid schedules without padding or loss
    comparisons between models. An implicit first-frame reference stays unchanged.
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
    Pyramid levels derive directly from the prepared native fitting images.
    Linear world matrices and both SyN halfway fields carry between levels;
    physical pivots remain fixed. Each level starts fresh optimizer and
    convergence state, with a new best-loss baseline. Failed/coarsely converged
    frames are retried at the next level. SyN field interpolation uses physical
    components without voxel-size scaling, zeros outside coarse field support
    and fixed native boundaries. Invalid interpolated fields are rejected with
    invalid_transform and zero residual fields before any finer-level retry.
    Native images are resampled only for final output, regardless of final shrink.
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
    Fitting images, NCC/MI statistics and SyN working fields use float32 with native
    grid_sample fitting interpolation. Physical geometry and linear pose parameters
    use float64, as do final SyN inversion and exported displacements. Benchmark accuracy is measured against known
    truth independently for each device, not by cross-device map agreement.
    Original float64 and integer images are resampled in float64, other images in
    float32, then restored to their original dtype. Integer results use nearest-even
    rounding and range clipping. Metrics exclude invalid sample centers; outside
    output values use each original moving frame's minimum.
    """
    settings = RegistrationSettings(
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        initial_transform=initial_transform,
        initialization=initialization,
        moving_intensity_scaling=moving_intensity_scaling,
        fixed_intensity_scaling=fixed_intensity_scaling,
        metric=metric,
        mi_bins=mi_bins,
        mi_sigma=mi_sigma,
        resample_interpolation=resample_interpolation,
        return_transforms=return_transforms,
        batch_size=batch_size,
        max_iterations=max_iterations,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
        learning_rate=learning_rate,
        syn_cc_radius=syn_cc_radius,
        syn_update_sigma=syn_update_sigma,
        syn_total_sigma=syn_total_sigma,
        convergence_window_size=convergence_window_size,
        convergence_minimum_value=convergence_minimum_value,
        device=resolve_device(device),
        progress=progress,
        metric_callback=metric_callback,
        metric_interval=metric_interval,
    )
    if isinstance(transform, Stage) or (
        isinstance(transform, Sequence) and not isinstance(transform, (str, bytes))
    ):
        from voxreg.stages import register_stages

        return register_stages(moving, fixed, transform, settings)
    if not isinstance(transform, str) or transform not in (
        "translation",
        "rigid",
        "affine",
        "syn",
    ):
        raise ValueError(
            "transform must be translation, rigid, affine or syn. Euler transforms belong to the rigid family."
        )
    result = run_stage(moving, fixed, transform, settings)
    assert isinstance(result, RegistrationResult)
    return result


def run_stage(
    moving: xr.DataArray,
    fixed: xr.DataArray | None,
    transform: str,
    settings: RegistrationSettings,
    *,
    fit_only: bool = False,
    active_frames: npt.NDArray[np.bool_] | None = None,
    center_override: npt.NDArray[np.float64] | None = None,
    validate_only: bool = False,
) -> RegistrationResult | LinearFit | None:
    """Validate and prepare one model, then dispatch to its fitting engine.

    Parameters
    ----------
    moving : xarray.DataArray
        Original spatial image or recording.
    fixed : xarray.DataArray, optional
        Shared spatial reference; omitted uses the first recording frame.
    transform : str
        Single transform model for this stage.
    settings : RegistrationSettings
        Effective shared and stage-specific options.
    fit_only : bool, default: False
        Whether to retain only a linear handoff instead of resampling originals.
    active_frames : numpy.ndarray, optional
        Frames eligible to continue after earlier stages.
    center_override : numpy.ndarray, optional
        Physical world z/y/x pivot in mm retained from an earlier stage.
    validate_only : bool, default: False
        Whether to validate shared input contracts without preparing or fitting.

    Returns
    -------
    RegistrationResult or LinearFit or None
        Complete outputs, linear handoff, or no result for validation-only calls.

    Raises
    ------
    TypeError
        If image, geometry or option types are unsupported.
    ValueError
        If input values, labels, geometry or numerical settings are invalid.
    """
    fixed_mask = settings.fixed_mask
    moving_mask = settings.moving_mask
    initial_transform = settings.initial_transform
    initialization = settings.initialization
    moving_intensity_scaling = settings.moving_intensity_scaling
    fixed_intensity_scaling = settings.fixed_intensity_scaling
    metric = settings.metric
    mi_bins = settings.mi_bins
    mi_sigma = settings.mi_sigma
    resample_interpolation = settings.resample_interpolation
    return_transforms = settings.return_transforms
    batch_size = settings.batch_size
    max_iterations = settings.max_iterations
    shrink_factors = settings.shrink_factors
    smoothing_sigmas = settings.smoothing_sigmas
    learning_rate = settings.learning_rate
    syn_cc_radius = settings.syn_cc_radius
    syn_update_sigma = settings.syn_update_sigma
    syn_total_sigma = settings.syn_total_sigma
    convergence_window_size = settings.convergence_window_size
    convergence_minimum_value = settings.convergence_minimum_value
    device = settings.device
    progress = settings.progress
    metric_callback = settings.metric_callback
    metric_interval = settings.metric_interval
    start = perf_counter()
    validate_metric(metric, mi_bins, mi_sigma)
    if metric == "local_ncc" and transform != "syn":
        raise ValueError(
            "metric=local_ncc requires transform=syn; its radius is syn_cc_radius."
        )
    metric_function = (
        None if metric in (None, "local_ncc") else Metric(metric, mi_bins, mi_sigma)
    )
    from voxreg.pyramid import validate_schedule

    factors, sigmas, budgets = validate_schedule(
        shrink_factors, smoothing_sigmas, max_iterations
    )
    multiresolution = factors != (1,) or sigmas != (0.0,)
    max_iterations = budgets[0]
    validate_registration_geometry(moving)
    validate_masks(fixed_mask, moving_mask, moving)
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
    implicit_fixed = fixed is None
    if fixed is None:
        fixed = moving.isel(time=0, drop=True)
    validate_registration_geometry(fixed)
    if fixed.dims != VOXEL_DIMS:
        raise ValueError("fixed must have dimensions (k, j, i), without time.")
    same_grid = moving.coords["z"].attrs["units"] == fixed.coords["z"].attrs[
        "units"
    ] and moving.xindexes["z"].equals(fixed.xindexes["z"])
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
    if (
        not isinstance(learning_rate, (int, float, np.integer, np.floating))
        or isinstance(learning_rate, bool)
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive.")
    if type(syn_cc_radius) is not int or syn_cc_radius < 1:
        raise ValueError("syn_cc_radius must be a positive integer.")
    for name, value in (
        ("syn_update_sigma", syn_update_sigma),
        ("syn_total_sigma", syn_total_sigma),
    ):
        if (
            not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative (voxel units).")
    if transform != "syn" and (
        syn_cc_radius != 2 or syn_update_sigma != 1.5 or syn_total_sigma != 0.5
    ):
        raise ValueError(
            'syn_cc_radius, syn_update_sigma and syn_total_sigma require transform="syn" when nondefault.'
        )
    validate_intensity_scaling(moving_intensity_scaling, "moving_intensity_scaling")
    validate_intensity_scaling(fixed_intensity_scaling, "fixed_intensity_scaling")
    if implicit_fixed and fixed_intensity_scaling is not None:
        raise ValueError("fixed_intensity_scaling requires an explicit fixed image.")
    if resample_interpolation not in ("linear", "nearest", "bspline"):
        raise ValueError("resample_interpolation must be linear, nearest or bspline.")
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
    if validate_only:
        return None
    if batch_size is None:
        batch_size = choose_batch_size(moving, fixed, transform, device)
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
    center = center_override
    if initialization is not None:
        if initialization == "align_planes" and (
            fixed.sizes["k"] != 1 or moving.sizes["k"] != 1
        ):
            raise ValueError("align_planes is only defined for two k=1 slices.")
        mode = "center_geometry" if initialization == "align_planes" else initialization
        fixed_position = compute_center_positions(fixed, mode, "fixed")[0]
        moving_positions = compute_center_positions(moving, mode, "moving")
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
            rotation, _ = compute_plane_alignment(fixed, moving)
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
            for key, expected in (("convention", "fixed-world-to-moving-world pull"),):
                if (
                    key in initial_transform.attrs
                    and initial_transform.attrs[key] != expected
                ):
                    raise ValueError(f"initial_transform {key} must be {expected!r}.")
        initial = recover_initial_parameters(
            initial_transform, plane, moving.sizes["time"]
        )
        if implicit_fixed:
            if not np.allclose(initial[0], 0, rtol=0, atol=1e-12):
                raise ValueError(
                    "The implicit reference initial_transform must be identity; supply explicit fixed otherwise."
                )
            initial[0] = 0
    planar = isinstance(plane, Plane) or plane.planar
    raw = moving.values[:, 0] if planar else moving.values
    fit, target = prepare_fitting_images(
        raw,
        fixed.values[0] if planar else fixed.values,
        moving_intensity_scaling,
        fixed_intensity_scaling,
        metric,
        implicit_fixed=implicit_fixed,
    )
    timings = {
        "prepare": perf_counter() - start,
        "upload": 0.0,
        "optimize": 0.0,
        "resample_download": 0.0,
    }
    resolved = replace(
        settings,
        batch_size=batch_size,
        progress=progress,
        initialization=initialization,
    )
    prepared = PreparedRegistration(
        moving=moving,
        fixed=fixed,
        settings=resolved,
        transform=transform,
        plane=plane,
        fit=fit,
        target=target,
        initial=initial,
        metric_function=metric_function,
        timings=timings,
        factors=factors,
        sigmas=sigmas,
        budgets=budgets,
        multiresolution=multiresolution,
        spatial_input=spatial_input,
        implicit_fixed=implicit_fixed,
        same_grid=same_grid,
        active_frames=active_frames,
    )
    if transform == "syn":
        from voxreg.syn import run_syn

        return run_syn(prepared)
    from voxreg.linear import run_linear

    return run_linear(prepared, fit_only=fit_only)
