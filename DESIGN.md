# First design: VoxelData-native batched rigid registration

Status: eager rigid implementation available, with synthetic CPU/CUDA tests and
real-notebook quality comparison. Isolated ITK/FireANTs time/memory comparisons are
available in [BENCHMARK.md](BENCHMARK.md). [PERFORMANCE.md](PERFORMANCE.md) identifies
the dense-grid bottleneck that motivated the production scalar-affine sampler.
Fresh production time/memory comparisons validate the rewrite; real-data
convergence calibration remains pending.
See [VALIDATION.md](VALIDATION.md).

## 1. Deliverable and stop condition

Build a standalone PyTorch proof of concept that corrects **in-plane rigid motion
in a single-slice recording**. Its public inputs and registered output are xarray
DataArrays following the VoxelData model. No ConfUSIus, ITK, FireANTs, syntx or
ANTs imports belong in the engine.

The first milestone is complete when it:

1. Recovers known translations and rotations in synthetic images, including
   anisotropic and oblique geometry.
2. Registers the actual ConfUSIus motion-correction example's 120-frame window,
   with inspectable transforms, diagnostics, voxel traces and before/after rasters.
3. Compares quality and runtime with a frozen ITK reference on those same frames.
4. Demonstrates that batching does not change a frame's estimated transform beyond
   the stated numerical tolerance.

This is not yet a general replacement for the existing registration API.

### Not in the first milestone

- Affine, B-spline, dense deformation, SyN or numerical field inversion.
- Full 3D registration, multiple poses, or arbitrary extra feature dimensions.
- Different fixed/moving grids, arbitrary masks, multiresolution or multiple metrics.
- Dask execution, distributed computation, napari integration or dataset downloaders.
- A second numerical backend, backend plugin system or broad configuration framework.
- Bitwise agreement with ITK, or reproducing its optimizer and learning-rate values.

These restrictions must be validated explicitly; unsupported inputs must not be
silently reinterpreted.

## 2. Python and dependencies

Use Python **3.14**. Python 3.12 was needed for some comparison-library packages,
not for voxreg. The implementation and CUDA tests run on Python 3.14.8 with
NumPy 2.5.4, xarray 2026.9.0 and PyTorch 2.14.1 on the current Linux machine.

The engine needs only:

- NumPy for host-side geometry and validation.
- xarray for the public data model and labeled results.
- PyTorch for interpolation, differentiation and batched optimization.

Pytest, Ruff and the independent SciPy interpolation reference are development
dependencies. Matplotlib, nibabel and SciPy are in the optional `examples` extra.
The optional reference exporter runs with an existing ConfUSIus interpreter;
ConfUSIus is not installed into voxreg. ITK and a pinned upstream FireANTs archive
are development-only comparison dependencies, never engine/runtime dependencies.

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
- Uniform, strictly monotonic native spatial coordinates; nonzero offsets and
  strides are supported. Spatial units must be consistently declared as mm.
- A finite, nonsingular voxel-to-world affine and the required lazy world index.
- Fixed and moving images with identical native coordinates and physical geometry.

A missing fixed image means the first moving frame. An explicit fixed image lets
callers select another frame or construct a mean reference without another
reference-selection option. Time labels need not start at zero and are preserved.
A pose dimension is rejected in this first version; the general model requires
one affine per pose, never a shared affine when pose is present.

### Independent implementation, not a ConfUSIus dependency

Provide a small `create_voxeldata` constructor, validator and independent xarray
index implementation. Geometry is authoritative in the index, not duplicated in
mutable DataArray attrs. Test that spatial slicing and reduction over time retain
correct geometry. Do not implement world-coordinate lookup/reindexing features
that registration does not need in the first milestone.

Following the same model does not automatically make two separately implemented
index classes interchangeable. No class-identity compatibility with ConfUSIus's
private index is promised. Reference-data interchange uses arrays, native
coordinates and the affine to construct voxreg VoxelData at the boundary. A shared
index protocol or neutral geometry package is a separate decision, not a reason
to import ConfUSIus internals.

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
the batch or compute covariance across frames. Accumulate NCC statistics in
float64: float32 loss quantization near unit NCC destabilized best-iterate selection. Use a small numerical denominator
guard only after checking that the metric is meaningful.

A frame that loses more than 10% overlap is flagged, with its best valid transform
retained (or identity when no valid metric has been observed). This threshold is a deliberately narrow PoC policy, not an assertion
that larger motion is invalid for registration in general.

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
recording = voxreg.create_voxeldata(
    values,
    dims=("time", "k", "j", "i"),
    coords={"time": times, "k": k, "j": j, "i": i},
    voxel_to_world=affine_zyx,
    units="mm",
)

result = voxreg.register_rigid(
    recording,
    fixed=None,                 # First frame; explicit fixed images use the same grid.
    intensity_scaling="db",     # Fitting only; output retains original intensities.
    batch_size=32,
    max_iterations=100,
    learning_rate=0.1,           # Proposed scaled-parameter rate, not ITK's rate.
    device="cuda",              # CPU also supported; no automatic device switch.
)
```

Return a small `RegistrationResult` with:

- `registered`: VoxelData `(time, k, j, i)` on the fixed grid.
- `valid`: boolean VoxelData with the same geometry and dimensions.
- `transforms`: DataArray `(time, world_out, world_in)` with labeled homogeneous
  z/y/x matrix axes and explicit fixed-world-to-moving-world semantics.
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
example must reconstruct VoxelData correctly; it must not sneak in a ConfUSIus
runtime dependency. Identify the input artifact/version in the results.

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

1. Add numerical/test dependencies and a small independent geometry module with
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
