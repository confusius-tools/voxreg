# VoxelData-native batched registration

Status: eager translation/rigid/affine and symmetric compositive SyN are available.
[SyN contracts and ITK/FireANTs measurements](research/syn/README.md) cover native
2D/3D pairs, recordings and explicit dense forward/inverse results. Linear models
retain their original numerical path, with synthetic CPU/CUDA tests and
real-notebook quality comparison. Isolated ITK/FireANTs time/memory comparisons are
available in [BENCHMARK.md](BENCHMARK.md). [PERFORMANCE.md](PERFORMANCE.md) identifies
the dense-grid bottleneck that motivated the production scalar-affine sampler.
Fresh production time/memory comparisons validate the sampler rewrite. The
ITK-style PyTorch convergence monitor is reference-tested, with initial NCC
threshold calibration on four real windows and bounded whole-batch termination.
Broader data qualification remains pending; see [research/convergence/CALIBRATION.md](research/convergence/CALIBRATION.md).
See [VALIDATION.md](VALIDATION.md).

## Shared linear models (current)

The sole public fitting entry point is `register(..., transform="rigid")`, with
`translation`, `rigid` and `affine` families. The pre-release `register_rigid`
wrapper is removed; live examples, tests and research callers use `register`.
The original rigid/default numerical path remains unchanged. Euler is a rigid
parameterization, not another family; this API does not claim ITK Euler-angle
optimizer trajectories.

`transforms.py` owns linear-model scaling, centered state, seed validation,
world export and diagnostics. `sampling.py` retains the qualified rigid kernels
and dispatches the same centered physical sampling composition on other linear
maps. `rigid.py` is the shared engine despite its historical internal filename:
one preprocessing, NCC, masked Adam, reference/stop/failure state and final
original-intensity resampling path, not three registration implementations.

Translation uses d physical center-displacement parameters and a fixed identity
linear map. Affine uses d*d row-major deviations from identity and d center shifts,
where d=2 in the fixed plane and d=3 for volumes. The pull is
`y = center + A @ (x-center) + t`; `b = t - (A-I) @ center` in exported world matrices.
An affine initializer seeds this full optimized state, not a frozen prefix or
rigid residual. Its rotation/scale/shear can all change during fitting.

Affine coefficients are dimensionless. An entry in column j has optimizer scale
`minimum_active_spacing / RMS(fixed_coordinate_j - center_j)`; physical translations
use minimum spacing. These are unweighted fixed-domain lever arms including pivot
offsets, not intensity covariance. The dense-domain scale oracle and gradients
are tested in world and orthonormal-plane coordinates, including oblique/sheared
and cropped/strided sampling geometry.

The 2D affine embedding is `I + B @ (A2-I) @ B.T`; its normal is unchanged. Seeds
with unidentifiable normal components or normal displacement are rejected, rather
than silently fitting a 3D affine to a plane. Translation seeds require identity
linear parts; rigid keeps its existing SVD-normalized proper-rotation policy.
Affine allows either determinant sign, but a fit preserves its seed orientation.
A finite active matrix passes the numerical guard only when
`abs(det(A / max(abs(A)))) > 1e-8`. This is scale-normalized to avoid determinant
overflow/underflow. Invalid or orientation-changing candidates mark only their
frame `invalid_transform`, do not update its best state, and retain the best
valid full pose (or validated seed if no metric was valid). No new CUDA scalar
reads are introduced by this vectorized guard.

Affine diagnostics expose actual linear entries plus center shifts, not rotation
angles obtained by pretending a shear is rigid. Default rigid diagnostic fields
and attributes are retained unchanged. Affine is unconstrained beyond numerical
invertibility/orientation; no anatomical regularizer or plausible-scale bound is
claimed. Fixed-budget analytic ITK/SciPy qualification is in
[research/linear/README.md](research/linear/README.md); default rigid-slice convergence
thresholds are not calibrated for affine or general 3D. Historical timing results
remain rigid measurements.

## Symmetric compositive SyN

`register(..., transform="syn")` dispatches after shared validation, initialization
and preprocessing into `syn.py`. Two independent physical midpoint-to-image fields
are optimized with masked squared local NCC and normalized smoothed compositive
updates. Complete pulls are `moving_half ∘ inverse(fixed_half)` and the reverse;
`fields.py` owns interpolation, composition, smoothing, approximate inversion and
full-lattice Jacobians. This is neither stationary velocity nor a renamed one-way
warp. ITK/ANTs provides the algorithmic baseline; FireANTs informs batched execution,
without source transplantation or optimizer-trajectory parity.

An orientation-preserving affine seed prealigns the moving image. Returned dense
world z/y/x millimetre displacements include that seed; the forward lives on the
fixed grid and the inverse on the original moving grid. Linear matrices and
metadata are unchanged. Native planes have two internal components, embedded in
world coordinates for export; mixed-dimensional deformation is explicitly rejected.
The single-resolution optimizer uses radius-2 neighborhoods and voxel sigmas
1.5 (updates) and 0.5 (total field). Fitting statistics are float32;
geometry/state and original float64-image final interpolation remain float64. Displaced float32 image fitting uses
fixed-order multilinear interpolation and roundoff-only voxel-knot snapping to
avoid inconsistent CPU/CUDA one-sided gradients; linear sampling is unchanged.

Accepted states require positive finite sampled halfway Jacobians. Final complete
residual maps are also checked; failure falls back to the seed with an explicit
status. These finite-difference checks do not prove global invertibility, and
sampled inverse interpolation has a nonzero two-sided residual. Diagnostics expose
that limitation rather than forcing agreement. Best-state selection uses the
halfway objective; endpoint global NCC is reported separately. Detailed numerical
proof and comparison caveats are in [research/syn](research/syn/README.md).

## Independent-grid / 3D rigid extension

`register` now accepts spatial-only fixed/moving volume pairs, or batches
`(time, k, j, i)`, with independent shapes, spacings, origins and orientations.
The shared engine evaluates the metric on fixed-grid points and resamples original
moving intensities onto that grid using `inverse(G_moving) @ T @ G_fixed`.
Spatial-only input requires fixed and returns no synthetic time dimension.

Volumes with every axis >= 2 fit a centered world z/y/x rotation vector and world
z/y/x translation. Rotation vectors follow right-handed XYZ handedness, stored in
z/y/x order, and are not Euler angles. Per-axis rotation scales use RMS physical
lever arms; translation scales use minimum fixed spacing. This remains distinct
from ITK's physical-shift estimator.

Coplanar k=1 pairs retain fixed-plane 3-DOF constraints, now with independent
in-plane grids. The identical-grid slice sampling kernel retains its qualified
arithmetic; preprocessing, NCC, optimizer, convergence, failure guards and dtype
handling are shared. The subsequent slice-geometry extension accepts compatible
plane seeds/explicit align_planes. Mixed pairs now sample the caller's native fixed
grid without role reversal. The nine-parameter mixed affine constrains the physical
slice-to-volume map; its inverse is used when the slice is moving, with translation
centered on the fixed image. Mixed support uses half-open half-voxel bounds on all
source axes and border-clamped interpolation; a singleton remains one finite cell.
Other singleton configurations remain errors; no slice is replicated into a volume.
See [native fixed-grid validation](research/slice-geometry/NATIVE-FIXED.md).

Initialization defaults to identity world motion; explicit rigid seeds now use
the optimized ITK InitialTransform model described below. No fixed coverage
percentage is required; fewer than two valid sample centers is insufficient overlap.
Automatic geometry/moments centering is described below. Explicit plane alignment
and mixed-domain fitting are described in [research/slice-geometry/README.md](research/slice-geometry/README.md).
Masks, pyramids and other metrics remain deferred. Initial 3D synthetic qualification and
bitwise slice regression evidence are in [research/geometry/README.md](research/geometry/README.md).
The subsequent [shared pair/series benchmark](research/registration/README.md)
provides bounded ITK/FireANTs 3D time/memory comparisons, with independent physical
truth controls and separate native-convergence qualification.

## Automatic centering

`initialization=None` preserves identity by default. `center_geometry` and
`center_moments` are mutually exclusive with an explicit `initial_transform`.
Both compute `moving_center - fixed_center` in physical world z/y/x as a
translation-only seed, then enter the existing validated transform/optimizer path.
Geometry uses array-position midpoints mapped through crop/stride-aware affines.
Moments uses original linear intensities, never dB fitting buffers or NCC-conditioned
values. Nonnegative finite values and positive mass are required per image; signed
input is rejected rather than shifted or assigned absolute-value weights.

Moment accumulation uses float64, per-image peak normalization (which leaves the
centroid unchanged and prevents overflowing the mass sum), and centered marginal
index moments without dense world-coordinate grids. Each moving frame is treated
independently. Origin differences are formed before local centroid offsets to
avoid common-large-origin cancellation. The implicit reference stays exactly
identity; centering does not introduce temporal coupling or extra interpolation.

Following ITK CenteredTransformInitializer, the fixed geometric center or fixed
intensity centroid is also the rotation pivot. The existing physical scales use
unweighted fixed-grid RMS lever arms **about that pivot**, not the moments
covariance. Off-axis variances include the squared geometric-center/pivot offset.
A resident small rotation-offset vector adjusts the centered sampler when the
pivot differs from the geometric center; image grids and sampling domains are not
changed. The qualified identity/default kernels retain their original arithmetic
when no pivot override is requested. Returned matrices and motion diagnostics still
describe the full transform; automatic-mode diagnostics record the world pivot.

Unlike NCC, moments is not invariant to additive offsets or nonlinear scaling.
Users should pass unscaled images and request dB fitting via `intensity_scaling`,
not pre-apply dB/log/sqrt/power transforms. Pre-scaled nonnegative inputs cannot be
reliably detected, so this is also a documented caller responsibility. Geometry
centering or an explicit matrix remains available for signed images. The API does
not infer the original units, invert display transforms, estimate rotation, align
plane tilt (use explicit align_planes or a compatible seed for two slices), or
infer a reliable match from a small overlap.

## Initial-transform model

A matrix-valued `initialization` aliases `initial_transform`, using the same
validation and optimizer path; it is not a frozen affine pre-transform. Named
initialization modes remain centering-only. Supplying both arguments is an error.
Scale/shear require `transform="affine"`, not silent projection into rigid state.
Euler is a rigid parameterization rather than a separate transform family.

`initial_transform` is a full fixed-world-to-moving-world rigid pull matrix, one
shared `(4, 4)` matrix or a positional `(N, 4, 4)` batch. It initializes the
**optimized transform**, like ITK `SetInitialTransform` with `InPlaceOff`.
It is neither a frozen `MovingInitialTransform` nor a residual transform chain.
There are no additional public transform objects or ITK runtime dependencies.

The matrix `T = [R, b; 0, 1]` is converted once to the existing centered parameters:
`d = b + (R - I) @ fixed_center`, plus a stable quaternion-derived rotation vector
for volumes, or fixed-plane angle/i/j displacement for coplanar slices. Rotation
extraction is stable at zero and pi. A matrix does not carry an ITK rotation center;
voxreg retains the fixed-grid center and existing physical scales. Fresh Adam
moments and clocks start at these parameters. Every metric/final sample uses the
full transform directly, so the input is never pre-resampled or interpolated twice.

Input validation checks shape, real/finite values, homogeneous row, orthogonality,
positive unit determinant, and slice-plane preservation. Float32 rotational
roundoff within 1e-6 is normalized to SO(3), preserving the mapped fixed center.
Labeled matrices require canonical dimensions and matching supplied world/time
labels, units and pull convention. User storage is never mutated. The implicit
reference must start at identity within 1e-12 centered-parameter roundoff and is
forced to exactly zero parameters. A shared nonidentity start requires explicit
fixed. Failure retains the best valid pose, or the supplied start if no metric
was valid; returned diagnostics always describe the full transform.

### Importing a rigid ITK transform

For a rigid ITK transform already expressed in the **same physical world frame**
as the VoxelData:

```python
matrix_xyz = np.eye(4)
matrix_xyz[:3, :3] = np.asarray(itk_transform.GetMatrix())
matrix_xyz[:3, 3] = np.asarray(itk_transform.GetOffset())  # Not GetTranslation().
order = [2, 1, 0, 3]
initial_zyx = matrix_xyz[np.ix_(order, order)]
result = voxreg.register(moving, fixed, initial_transform=initial_zyx)
```

`GetOffset` incorporates a nonzero ITK center, whereas `GetTranslation` does not.
Axis permutation is **not** an LPS/RAS coordinate-frame conversion: reconcile
world frames separately if independent readers use different conventions. No
inverse is applied for ITK's usual registration pull transform. Composite chains
are not imported by this snippet.

The ITK model is checked against installed ITK 5.4.7: a zero-update v4 registration
with `SetInitialTransform`/`InPlaceOff` maps fixed physical points identically to
voxreg's seed, and mutating the cloned ITK output leaves its initial object intact.
The corresponding API contract is documented by ITK's
[ImageRegistrationMethodv4](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1ImageRegistrationMethodv4.html).
This is initialization compatibility, not equality with ITK's Euler/GD optimizer.

**The numbered sections below preserve the original slice milestone.** Its full-3D
and identical-grid exclusions are superseded by this extension; its historical
performance evidence is not evidence for 3D throughput.

## 1. Deliverable and stop condition

Build a standalone PyTorch proof of concept that corrects **in-plane rigid motion
in a single-slice recording**. Its public inputs and registered output are xarray
DataArrays using ConfUSIus's VoxelData implementation. ConfUSIus is temporarily
a runtime dependency for geometry; registration remains voxreg's PyTorch engine,
not an ITK, FireANTs, syntx, ANTs or ConfUSIus registration call.

The first milestone is complete when it:

1. Recovers known translations and rotations in synthetic images, including
   anisotropic and oblique geometry.
2. Registers the actual ConfUSIus motion-correction example's 120-frame window,
   with inspectable transforms, diagnostics, voxel traces and before/after rasters.
3. Compares quality and runtime with a frozen ITK reference on those same frames.
4. Demonstrates that batching does not change a frame's estimated transform beyond
   the stated numerical tolerance.

This is not yet a general replacement for the existing registration API.

### Not in the first milestone (historical rigid-only scope)

Affine, independent grids, full 3D and SyN were added subsequently as described
above. The original exclusions below describe that first milestone, not today's
supported transform list.

- Affine, B-spline, dense deformation, SyN or numerical field inversion.
- Full 3D registration, multiple poses, or arbitrary extra feature dimensions.
- Different fixed/moving grids, arbitrary masks, multiresolution or multiple metrics.
- Dask execution, distributed computation, napari integration or dataset downloaders.
- A second numerical backend, backend plugin system or broad configuration framework.
- Bitwise agreement with ITK, or reproducing its optimizer and learning-rate values.

These restrictions must be validated explicitly; unsupported inputs must not be
silently reinterpreted.

## 2. Python and dependencies

Accept Python **3.11–3.14**; retain 3.14 as the preferred development interpreter.
Syntax and dependency resolution cover this range; full numerical/CUDA
qualification is still on Python 3.14.8 with NumPy 2.5.4, xarray 2026.9.0 and
PyTorch 2.14.1 on the current Linux machine. The earlier 3.14-only package floor
was unnecessarily strict, not imposed by ConfUSIus's Python requirement.

The engine directly uses:

- NumPy for host-side geometry and validation.
- xarray for the public data model and labeled results.
- PyTorch for interpolation, differentiation and batched optimization.
- ConfUSIus >=0.8.0 for the canonical VoxelData index and Rich progress reporter.
  This temporarily replaces the duplicated geometry implementation.

Pytest/Ruff are development dependencies. SciPy >=1.16.3 remains a compatible
dependency floor for ConfUSIus and supplies independent interpolation/filtering
references; the engine does not call SciPy for runtime smoothing. Matplotlib, nibabel and SciPy are in the optional
`examples` extra. The optional reference exporter runs with an existing ConfUSIus
interpreter; ConfUSIus is now installed into voxreg and brings its full transitive
dependency stack, including ITK/napari. FireANTs remains development-only. The
released ConfUSIus package supports Python >=3.11,<3.15 and requires SciPy >=1.16.3;
its SciPy floor still prevents sharing an environment with ANTsPy's scipy<1.16.
No sibling checkout or dependency override is required. Historical benchmarks
predate these compatibility declarations.

## 3. VoxelData contract

VoxelData remains an **xarray.DataArray**, not a replacement image container.

The general spatial convention is `(..., time, pose, k, j, i)`, with optional
time/pose dimensions and mandatory trailing `k/j/i`. A `VoxelToWorldIndex` on
`z/y/x` derives world coordinates lazily from the native voxel coordinates and
an affine. World coordinates are not stored as eager spatial meshes. Affines map
native `(k, j, i, 1)` coordinates to world `(z, y, x, 1)` coordinates.

For this milestone accept only:

- Moving recording: `(time, k, j, i)`, with `k=1`.
- Fixed image: `(k, j, i)`, with `k=1`.
- Real, finite NumPy-backed integer or floating data up to float64. Integers must
  be exactly representable in float64 (values within [-2**53, 2**53]).
- Integer, uniform, strictly monotonic native spatial coordinates; nonzero offsets and
  strides are supported. Spatial units must be consistently declared as mm.
- A finite, nonsingular voxel-to-world affine and the required lazy world index.
- Fixed and moving images with identical native coordinates and physical geometry.

A missing fixed image means the first moving frame. An explicit fixed image lets
callers select another frame or construct a mean reference without another
reference-selection option. Time labels need not start at zero and are preserved.
A pose dimension is rejected in this first version; the general model requires
one affine per pose, never a shared affine when pose is present.

### Temporary shared implementation via ConfUSIus

Use ConfUSIus's actual `VoxelToWorldIndex`; do not duplicate its index/transform
classes. Both `cf.load(...).compute()` and `cf.create_voxeldata(...)` feed the
registration API directly, and registered/valid outputs retain that same model.
Geometry stays authoritative in the index rather than mutable DataArray attrs.

Use ConfUSIus's constructor and general validator directly; voxreg does not export
VoxelData construction/validation wrappers. Keep only private registration-specific
geometry validation and array-position affine conversion locally. The bridge
imports ConfUSIus's index class in one module; changing private upstream helpers is
therefore an explicit integration risk, covered by shared-class and regression
tests. ConfUSIus's indexing features are inherited, not reimplemented.

A future independent VoxelData package is deferred. Reverting this dependency
migration does not require rewriting or resampling stored image arrays.

## 4. Physical geometry and transform convention

Public transforms are pull transforms:

> A transform maps a fixed world position to the moving world position sampled
> to produce the registered image at that fixed position.

Return homogeneous `(4, 4)` matrices in world **z/y/x** order, not normalized
Torch coordinates. Inverse rigid transforms are ordinary matrix inverses; no
numerical displacement-field inverter is needed.

### Array positions are not native coordinate values

For each spatial axis with uniform coordinates:

    native_coordinate = first_coordinate + step * array_position

Let `C` be that homogeneous position-to-native-coordinate map and `A` the stored
voxel-to-world affine. The position-to-world map is `G = A @ C`.

The sampling relation is:

    moving_position = inverse(G_moving) @ T @ G_fixed @ fixed_position

This prevents an especially easy bug: applying the stored affine directly to
zero-based array positions after cropping or striding. A singleton axis has its
selected native coordinate incorporated in the offset; its unobserved step does
not need to be inferred from a single sample.

### Genuine 2D optimization, full 3D geometry

Do not expand the singleton spatial axis. Build an orthonormal plane basis from
the physical i/j directions, not from assumed world x/y axes. Optimize only:

- One rotation about the slice normal, centered on the fixed image center.
- Two translations in that plane, expressed in physical distance.

Lift that transform back into a `(4, 4)` world matrix. This remains physically
rigid with anisotropic sampling and oblique orientation. The slice normal is
unchanged, and no unobservable through-plane parameters are optimized.

Inside PyTorch, image shape is `(batch, channel=1, H=j, W=i)`. Sampling grids are
normalized in **i/j** component order with `align_corners=True`; this ordering is
an implementation detail and never becomes the public transform convention.

## 5. Numerical engine

### Preprocessing

Use dB scaling for the notebook experiment, independently of output resampling:

    fit_image = 10 * log10(max(power / scale, 1e-12))

`scale` is a shared positive maximum over the moving recording and fixed image,
computed once, not once per chunk. Negative power values, nonfinite inputs and an
all-zero recording are errors. The explicit floor makes zeros finite; it is a
numerical policy, not a modification of the returned original intensities.
A `none` mode permits already-prepared finite fitting intensities.

### Similarity and interpolation

Start with **negative squared global NCC**, computed independently per frame.
Use bilinear interpolation and only geometrically valid moving samples in the
objective. Zero padding must not be included as real dB observations. Reject
constant fixed images; diagnose insufficient valid variance or overlap rather
than returning a misleading finite loss from a large denominator epsilon.

Sum per-frame losses to obtain a scalar for differentiation. Never center across
the batch or compute covariance across frames. Fitting NCC statistics use
float32, matching the resident fitting images; explicit float64 metric inputs
retain float64 for analytic checks. Float32 frame reductions are padded with
invalid samples to align rows, avoiding row-address-dependent CUDA reductions.
Benchmark acceptance uses independent truth RMS < 0.25 voxels and successful
execution within available memory; median runtime is the optimization objective.
CPU/CUDA and old/new mapped-point differences are diagnostics, not benchmark gates. Use a small numerical denominator
guard only after checking that the metric is meaningful.

No minimum coverage fraction is enforced. Both optimizer evaluation sites require
more than one valid sample center, nonzero finite variances and a finite metric.
Fewer than two valid centers retains the existing insufficient_overlap status;
constant or invalid arithmetic yields invalid_metric. Frames retain their best
valid transform, or their seed when no metric was valid. valid_fraction continues
to report coverage, not confidence. Two samples only satisfy the mathematical NCC
minimum, not transform identifiability or anatomical validity. The former 90% rule
rejected valid smaller fields of view and blocked the real same-subject pair;
its removal is documented in [research/overlap-policy/README.md](research/overlap-policy/README.md).

### Optimization

Start with batched Adam and identity transforms. Scale translation and rotation
parameters according to their physical effect rather than giving radians and mm
identical raw step sizes. A proposed scaling is:

- `ell`: minimum in-plane physical pixel spacing.
- `r`: RMS physical distance of fixed-grid centers from the rotation center.
- Physical translations: `ell * u_translation`.
- Rotation angle: `(ell / r) * u_rotation`.

This makes the learning rate approximately comparable across pixel sizes and
image extents. It is not equivalent to ITK's physical-shift scales estimator and
must be calibrated on the synthetic tests, then held fixed for evaluation.

Convergence and failure state are per frame. Converged/reference/failed frames
must have their parameters **and optimizer state** frozen while other frames
continue. The reference frame is exactly identity, not numerically re-estimated.

Keep images, grids, parameters and optimizer state resident for each chunk.
Working fitting images/interpolation remain float32. Physical geometry and optimizer state use
float64 after real-data tests exposed chunk-shape-dependent float32 rounding.
Only the final normalized interpolation grid is cast to the image dtype. This
is a correctness-first baseline, not a throughput-optimal precision policy.
Do not call `.item()`, collect garbage or clear the CUDA cache on every iteration
or frame. Transfer final transforms and diagnostics after the numerical loop.

First implement an eager, readable correctness baseline. Measure a compiled step
or loop only after correctness, without adding a second engine or claiming a
speedup in advance. Compilation and warm steady-state timings are separate.

### Output resampling

Apply the final mapping once to **original, unscaled intensities**, not to the
fitting dB images. Retain the moving dtype, original time coordinates and physical
intensity units. Float64 and integer images use float64 final interpolation;
float32/float16 images use float32, then return their original dtype. Integer
outputs are rounded nearest-even and range-clipped; reject integer inputs outside
[-2**53, 2**53]. `interpolation="linear"` or `"nearest"` affects only this final
resampling, not the linear sampler used for fitting. Use the original
moving frame's minimum as the fill value and return a validity mask so that edge
pixels need not be interpreted as genuine measurements. The reference frame can
be copied directly.

## 6. Proposed public surface

Implemented API (the device is explicitly selected below; CPU is the default):

```python
import confusius as cf
import voxreg

recording = cf.create_voxeldata(
    values,
    dims=("time", "k", "j", "i"),
    time=times,
    k=k,
    j=j,
    i=i,
    voxel_to_world=affine_zyx,
    units="mm",
)

result = voxreg.register(
    recording,
    fixed=None,  # First frame; explicit fixed images use the same grid.
    intensity_scaling="db",  # Fitting only; output retains original intensities.
    batch_size=None,  # Automatic, VRAM-aware on CUDA; explicit integers override.
    return_transforms=False,  # Opt in only when matrices/fields are needed.
    max_iterations=100,
    learning_rate=0.1,  # Proposed scaled-parameter rate, not ITK's rate.
    device="cuda",  # CPU also supported; no automatic device switch.
)
```

Return a small `RegistrationResult` with:

- `registered`: VoxelData `(time, k, j, i)` on the fixed grid.
- `valid`: boolean VoxelData with the same geometry and dimensions.
- `transforms`: None by default; with `return_transforms=True`, linear models
  return DataArray `(time, world_out, world_in)` with labeled homogeneous z/y/x
  matrix axes and explicit fixed-world-to-moving-world semantics. SyN instead
  exports complete forward/inverse float64 world-displacement fields. Both fields
  cost 12× equal-grid float32 image storage, plus assembly temporaries; opting out
  skips export/download/accumulation, not fitting, resampling or diagnostics.
- `diagnostics`: Dataset indexed by time, containing final objective, valid
  fraction, iteration count and a per-frame status.

These are results with named fields, not a second spatial data container. Avoid
putting a DataFrame or device tensor into opaque DataArray attrs. Matrix inversion
already supplies inverse rigid transforms; no separate inverse API is needed yet.

## 7. Reproduce the actual motion-correction example

The target is **not** the two-frame ref20 microbenchmark. The current example is:

- Dataset: Cybis Pereira 2026 rawdata.
- Subject `rat75`, session `20220523`, acquisition `slice32`.
- Window: zero-based frames **220:340**, 120 frames in total.
- Fixed image: first frame of that window.
- Fit: power-to-dB scaling and global correlation.
- Output: registered original power Doppler intensities.

Reference source:
[ConfUSIus motion-correction example](../../confusius-tools/confusius/docs/examples/03_registration/02_volumewise_motion_correction.py).
The example uses ITK learning rate 1.0; that value is not transplanted into Adam.

Prepare a frozen input/reference artifact outside the voxreg engine, including
original intensities, time labels, native coordinates, affine, ITK pull matrices,
registered outputs and exact reference settings. NIfTI reading in the standalone
example must reconstruct VoxelData correctly using the shared index. ConfUSIus is
now an explicit runtime dependency. Identify the input artifact/version in results.

Produce:

1. Per-frame motion and optimizer plots. State whether a reported translation is
   center displacement or the homogeneous matrix offset; they are not identical.
2. A trace at the voxel with largest pre-registration temporal standard deviation.
3. A fixed native-column time raster before/after, using the same column and color
   limits. For this example choose the corresponding column once, not separately
   after registration.
4. Registered and difference images with ITK as a comparison reference.
5. Machine-readable transforms, diagnostics and napari-readable outputs.

For mean/max/RMS framewise displacement, the inspected reference computes
`norm(T[t+1] @ p - T[t] @ p)` on fixed-grid world points, associates the result
with time t, and sets the final entry to zero. The example uses that same rule,
not transform composition or a different temporal convention. Do not substitute a generic
sum of Euler-angle differences or claim the diagnostic curves are identical.

Matching the example means comparable geometric recovery and stabilization,
not merely a smooth-looking raster or a lower image variance. True fUSI signal
changes must not be mistaken for motion.

## 8. Tests and acceptance gates

### Geometry and numerical correctness

- Asymmetric seeded phantoms with known translations and rotations.
- Identity and exact reference-frame preservation.
- Anisotropic spacing, oblique orientation, nonzero origins, cropped/strided
  coordinates, singleton k offsets and physical-unit consistency.
- Pull-transform direction, rotation center and matrix inverse round trips.
- Agreement between the sampler and an independent interpolation reference on
  interior points, including image boundaries and the validity mask.
- Invalid dimensions, missing index, incompatible grids/units, nonfinite inputs,
  invalid affines, constants and insufficient-overlap cases.
- Transform estimates invariant to chunk composition within floating-point
  tolerance; repeat with two frames moving in opposite directions to catch a
  loss that accidentally mixes the batch.
- CPU float64 gradient checks on the interior and CPU/CUDA comparison with
  dtype-appropriate tolerances.

Proposed recovery gate: on noiseless asymmetric cases with translations up to two
pixels and rotations up to two degrees, achieve RMS physical mapping error below
0.25 of the minimum in-plane pixel spacing over a declared interior region.
The synthetic tests pass this gate without changing it for the real recording.
The measured recovery errors and other remaining gates are recorded in
[VALIDATION.md](VALIDATION.md).

### Real recording and performance

Compare ITK and voxreg on **the same** 120-frame window and common valid support.
Report per-frame transform disagreement, NCC, overlap and output differences;
inspect outliers rather than accepting only a better aggregate score.

Record loading/preprocessing, upload, optimization, resampling/download, cold
compilation and warm end-to-end timings separately. Synchronize CUDA timers.
Run one benchmark at a time. Compare serial and chunked execution, including a
fair multicore ITK control before claiming a public-pipeline speedup.

Do not add affine/deformable models until this milestone passes. There is no
assumed speed or quality advantage over batched FireANTs.

## 9. Minimal implementation order

1. Add numerical/test dependencies and a small geometry adapter with
   constructor/validation/slicing tests.
2. Implement physical pull-grid construction and sampling, tested against known
   mappings before adding any optimizer.
3. Add batched rigid parameters, per-frame NCC and optimizer state.
4. Assemble the result and resample original intensities.
5. Reproduce the real example, compare with the frozen reference, and benchmark.
6. Consider compilation only where profiling shows a useful opportunity.

Split code by these actual responsibilities, not into a framework of backends,
registries and transform subclasses. Reuse the same geometry and sampling path
for the optimizer and final resampling. Stop at a validated rigid PoC.
