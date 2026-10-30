# voxreg

A proof of concept for **batched physical-space registration** using
PyTorch and ConfUSIus's xarray-based VoxelData implementation.

The engine registers **single 3D volume pairs or batched frames**, with independent
fixed/moving shapes, spacings, origins and orientations. Native single-slice pairs
retain identifiable in-plane transforms. CPU and CUDA share each model's numerical
path, convergence policy and dtype-preserving resampling contracts.

Translation, rigid and affine fitting are joined by **symmetric compositive SyN**
for native 2D–2D and 3D–3D pairs/recordings. This is not a complete ConfUSIus
replacement or a claim of anatomical validation. See the new
[SyN implementation and measured ITK/FireANTs comparison](research/syn/README.md).
The historical performance benchmarks below measure rigid registration only. A [shared benchmark protocol](research/registration/README.md) now
covers 3D pairs and motion correction against ITK/FireANTs, including 64³/128³
phantoms and a real-template known-pose control—not a natural 3D session pair.

See [DESIGN.md](DESIGN.md) for conventions and scope, and
[VALIDATION.md](VALIDATION.md) for the measured quality checks and remaining gates.
For a small research-only live composite/metric plot, see
[native-pair exploration](research/confusius-examples/EXPLORATION.md).
[ROADMAP.md](ROADMAP.md) records possible multistart/line-search work and existing
manual rigid→affine staging; these do not change optimizer defaults.

## Install and test

Requires Python **3.14** (`>=3.14,<3.15`, matching the temporary ConfUSIus dependency). CUDA execution has been checked on Linux with an RTX
5080 Laptop GPU. The default PyTorch distribution includes large CUDA runtime
packages; a CPU-only package source is not configured here.

```bash
uv sync
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples

# Also exercise the optional NIfTI/validation example.
uv sync --extra examples
uv run --extra examples pytest
```

The CUDA test runs when CUDA is available and otherwise skips. The registration
engine uses NumPy, xarray and PyTorch, with **ConfUSIus temporarily required for
VoxelData** (`confusius>=0.8.0`). `uv sync` installs the released package; no
sibling checkout is required. ConfUSIus brings its full dependency stack,
including ITK/napari and Rich. The NumPy minimum matches ConfUSIus (2.3.4) to
keep its platform dependency constraints resolvable; this Linux environment
uses NumPy 2.5.4 and xarray 2026.9.0.

FireANTs remains a pinned development-only comparison dependency. Matplotlib,
nibabel and SciPy are listed in the `examples` extra, although ConfUSIus also
installs them transitively. No registration is delegated to ConfUSIus or ITK.

## Use

```python
import numpy as np
import confusius as cf
import voxreg

# values: real NumPy array (time, 1, j, i); its dtype is retained.
# Affine rows/columns are world z/y/x and native voxel k/j/i, respectively.
recording = cf.create_voxeldata(
    values,
    dims=("time", "k", "j", "i"),
    time=time_seconds,
    voxel_to_world=np.diag([0.4, 0.1, 0.1, 1.0]),
    units="mm",
)

result = voxreg.register(recording, device="cuda", return_transforms=True)
registered = result.registered  # VoxelData, original intensities and dtype.
valid = result.valid  # VoxelData, boolean sample-support mask.
matrices = result.transforms  # (time, world_out, world_in), float64.
print(result.diagnostics)
```

For recordings, registration defaults to the first frame, preserved exactly,
including its original dtype. An explicit spatial-only `fixed=` image may have
independent geometry. Output always uses the **fixed spatial grid** and the moving
intensity dtype, units and time labels. Defaults are 100 iterations, learning rate 0.1 in scaled
physical parameters, and dB fitting intensities. Use `intensity_scaling="none"`
for already-prepared fitting intensities or signed integer images. dB mode
requires nonnegative power.

Transforms are **off by default**: ordinary `voxreg.register(recording)` returns
corrected intensities, validity and diagnostics, with both transform attributes
set to `None`. Set **`return_transforms=True`** only when you need matrices or
fields. For equal grids and float32 images, both exported float64 SyN fields cost
**12× the uncompressed image storage**, plus assembly temporaries: a 525 MB movie
needs about 6.3 GB for fields alone. Other dtypes/grid sizes change this ratio.
With transforms disabled, dense exports, downloads and accumulation are skipped;
only batch-sized deformation state is retained during fitting/resampling.

### Transform models

Use one entry point, `voxreg.register(..., transform=...)`:

| Model | Coplanar slice | 3D volume |
| --- | ---: | ---: |
| `"translation"` | 2 parameters | 3 parameters |
| `"rigid"` (default) | 3 parameters | 6 parameters |
| `"affine"` | 6 parameters | 12 parameters |
| `"syn"` | Two 2-component halfway fields | Two 3-component halfway fields |

Euler transforms belong to the rigid family. Voxreg's rigid optimizer uses a
rotation vector in 3D, not ITK's Euler-angle parameterization. Affine adds scale
and shear; it optimizes the **full** supplied matrix, not a rigid residual after
freezing a manual affine prefix.

```python
registered = voxreg.register(
    pwd, template,
    transform="affine",
    initialization=np.linalg.inv(affine),  # If affine maps pwd world -> template world.
    intensity_scaling="none",
).registered
```

The pre-release `register_rigid` name has been removed. Change existing calls to
`register`; omitting `transform` keeps rigid fitting and its numerical behavior.
The three linear models accept matrix or geometry/moments initialization and
support pairs/recordings on the same sampling/optimizer path.

Slices use an orthonormal intrinsic i/j plane; affine slices leave the world
normal unchanged, rather than fitting unidentifiable 3D coefficients. Affine
parameters are centered linear deviations from identity plus center displacement;
diagnostics expose **actual** `linear_ii`/`linear_ij`/... or world `linear_zz`/...,
plus translations in mm. They do not report fictitious rigid angles for a shear.

Affine seeds must be finite and numerically nonsingular:
`abs(det(A / max(abs(A)))) > 1e-8` for the active linear map `A`.
The optimizer preserves the initial determinant sign. A reflected affine can be
supplied explicitly; identity initialization does not search for reflections.
An invalid/singular/orientation-changing candidate stops only that frame with
`invalid_transform` and retains its best valid full pose. This is not an anatomical
regularizer: implausible but invertible scale/shear can still overfit the metric.

New-model qualification uses fixed budgets and analytic ITK/SciPy controls, not
natural inter-session data or new performance benchmarks. Defaults were calibrated
on **rigid slices**, not affine fitting or general 3D; see
[research/linear/README.md](research/linear/README.md) and [VALIDATION.md](VALIDATION.md).

### Symmetric nonlinear registration

```python
# Optional global prealignment: keep using the original images in both calls.
seed = voxreg.register(
    moving, fixed, transform="affine", intensity_scaling="none", return_transforms=True
)
result = voxreg.register(
    moving, fixed, transform="syn", initial_transform=seed.transforms,
    intensity_scaling="none", device="cuda", batch_size=1, return_transforms=True,
)
forward = result.transforms          # (*fixed spatial shape, world_component).
inverse = result.inverse_transforms # (*moving spatial shape, world_component).
```

For recordings, fields also retain `time`. Components are labeled **z/y/x in mm**:
`p_moving = p_fixed + forward(p_fixed)`; the inverse is the reverse pull on the
original moving grid. These are complete displacements, **including the affine
seed**, not homogeneous matrices. Linear results remain matrices and have
`inverse_transforms=None`. Plane outputs embed two identifiable in-plane degrees
of freedom into world components; they do not estimate depth variation.

SyN independently optimizes two halfway maps using masked squared local NCC,
smoothed normalized compositive updates, and approximate numerical inversion.
The current implementation is **single-resolution**, with neighborhood radius 2,
update sigma 1.5 and total sigma 0.5 voxels. `learning_rate` bounds each halfway
update's maximum voxel-coordinate norm. The affine seed is fixed during field
fitting and must preserve orientation. Mixed slice/volume deformation and dense
restart seeds are rejected. Use small volume batches to limit memory.

`min_jacobian` and `inverse_error` expose regularity and two-sided sampled inverse
residuals. Positive sampled Jacobians do not prove global invertibility; sampled
inverses are not exact. The live research plotter currently supports linear
models only. See [contracts, tests and measured limitations](research/syn/README.md).

### Single 3D volume registration

```python
# moving_volume and fixed_volume: VoxelData with dimensions (k, j, i).
# Their shapes, spacings, orientations and origins need not match.
result = voxreg.register(
    moving_volume, fixed_volume, intensity_scaling="none", device="cuda",
    return_transforms=True,
)
registered = result.registered  # (k, j, i), on fixed_volume's grid.
matrix = result.transforms     # (world_out, world_in), no synthetic time axis.
print(result.diagnostics)       # Scalar objective, status and motion parameters.
```

All three axes must contain at least two samples for volume fitting (6 DOFs for
the default rigid model).
Recordings `(time, k, j, i)` use the same volume engine in batches. Choose a small
`batch_size` override when needed. The default (`None`) is automatic: CPU caps at
32; CUDA uses available VRAM, image size and a conservative per-method working-set
estimate with 50% headroom. Performance caps are 256 for linear slice pairs and
32 for SyN/volumes; larger batches are not always faster. The memory-limited CUDA
choice rounds down to a power of two, then is limited to the frame count. This
rounding is a conservative policy, **not evidence that powers of two are always
faster**; short recordings and final partial batches need not be powers of two.
Seeing 256 for linear slices and 32 for SyN usually means the performance cap,
not the VRAM limit, was reached. CPU currently uses a fixed cap, not a memory or
speed autotuner. Reusable Torch cache counts as
available; concurrent allocations can still cause OOM, and one frame must fit.
No optimizer probes or OOM retries run. The effective size is recorded in
`result.diagnostics.attrs["batch_size"]`. Explicit positive integers override the
policy. Single-pair 3D memory is benchmarked at documented sizes; large 3D batches are not. Spatial-only input requires an explicit fixed image. Coplanar `k=1` slice pairs instead fit
three identifiable in-plane parameters with the default rigid model.

For linear pairs, the metric is evaluated on the caller's native fixed-grid points, sampling moving array positions as
`inverse(A_moving) @ T @ A_fixed`, where each `A` includes native crop/stride
coordinates. Thus **identity world motion can resample unequal grids** without
estimating their known geometry as motion.

### Slice placement and slice-to-volume fitting

All geometry remains in physical 3D space. Registration selects the observable
parameters and metric domain from the input geometry:

| Inputs | Translation / rigid / affine parameters | Metric domain |
| --- | --- | --- |
| Two slices | 2 / 3 / 6 | Fixed slice, constrained plane-to-plane map |
| Two volumes | 3 / 6 / 12 | Fixed volume |
| One slice, one volume | 3 / 6 / 9 | Caller’s fixed grid; no role reversal or replicated slab |

**Noncoplanar slice pairs** need a physical seed mapping the fixed plane into the
moving plane, or an explicit `initialization="align_planes"`. This new initializer
aligns geometric centers and chooses the shortest unoriented-normal rotation,
then image content refines the in-plane pose. It assumes corresponding anatomical
sections, not two genuinely different sections. Ordinary center initialization
can align parallel displaced planes but cannot fix tilt. A translation-only model
cannot align nonparallel planes: use rigid or affine.

```python
result = voxreg.register(
    moving_slice, fixed_slice, initialization="align_planes",
    intensity_scaling="none",
)

# The entire supplied 3D reference is available as the slice moves and tilts.
# This direction directly returns the template sampled onto the recording slice.
atlas_on_slice = voxreg.register(
    atlas_volume, recording_slice, transform="affine",
    initialization=slice_world_to_atlas_world,
    intensity_scaling="none",
)
```

Mixed fitting uses the **same optimizer**, with the caller's roles unchanged.
To fit the recording on the template grid, use `register(recording_slice,
atlas_volume, ...)`; this is a genuinely different objective from the example
above, not a hidden inverse fit. There is no direction flag or replicated slab.

Affine fits six entries of a 3×2 tangent map plus three world translations. The
physical **slice-to-volume map** retains a unit orthogonal, orientation-preserving
normal completion in both directions; when the slice is moving, its inverse is
used for the requested pull. Unobservable normal scale/shear seeds are rejected.
Rigid allows all six spatial pose parameters. Mixed modes use the geometric
**fixed-image pivot**; `center_moments` aligns intensity centroids for the seed
without changing this pivot. Batch seeds and frame failures remain independent.

**The returned pull always maps the original fixed world into the original moving
world**, and images retain the original moving dtype on the original fixed grid.
If moving is the slice and fixed the volume, samples within its **finite voxel
extent** are valid: the singleton axis occupies `[-0.5, 0.5)` in source voxel
coordinates, mapped through the full affine. Its intensity is constant through
that extent; no independent measurements are replicated. Mixed pairs use ITK-style
half-voxel support `[-0.5, size-0.5)` on **all source axes**, with border-clamped
interpolation. Fitting uses the original fixed-grid centers, including when fixed
is the volume. If none falls inside the transformed slice extent, fitting reports
insufficient overlap and a warning; it does not silently switch domains.
For a useful atlas overlay, prefer the volume-as-moving call above, or sample the
volume onto the slice with the inverse of the slice-as-moving result matrix.
`diagnostics.valid_fraction` describes native fixed-grid metric support;
`output_valid_fraction`, retained for moving-slice calls, now describes that same
domain. Coverage is not confidence.
Diagnostic pose parameters, like the exported matrix, describe the requested
fixed-to-moving direction. Mixed affine diagnostics report the completed 3×3
matrix; only its nine identifiable slice-fitting parameters are optimized.

Swapped mixed calls no longer coincide by construction. They use different fixed
samples and moving intensities/gradients, so inverse poses are not guaranteed.
This follows the caller-fixed sampling convention, **not CF's historical four-slice
expansion**. Our constrained affine, autograd derivatives and Adam optimizer still
differ from generic ITK affine/GD; sampling agreement is not full algorithmic parity.
See the [native fixed-grid validation](research/slice-geometry/NATIVE-FIXED.md).
The [FireANTs/slab experiments](research/confusius-examples/ATLAS-FIREANTS.md) remain
historical diagnostics, not the implementation target.

The [slice-geometry qualification](research/slice-geometry/README.md) records
synthetic truth/CPU–CUDA checks and the real ConfUSIus atlas retry.

### Explicit initial transforms

A matrix passed as `initialization=matrix` is an alias for
`initial_transform=matrix`; supply only one. Both follow **ITK's optimized `InitialTransform` model with
`InPlaceOff`**, not its separate frozen `MovingInitialTransform`. It supplies the
starting full pose of the selected model; voxreg copies it, starts fresh Adam state, refines it and
returns the **full transform**, not an increment to compose afterward. Omission
still starts at identity in physical world space. Original images are not
pre-resampled.

```python
import numpy as np

initial = np.eye(4)
initial[:3, 3] = [5.0, -2.0, 1.0]  # World z/y/x offset in mm, fixed → moving.
result = voxreg.register(
    moving_volume, fixed_volume,
    initial_transform=initial, intensity_scaling="none", return_transforms=True,
)
# initial is unchanged; result.transforms contains the complete refined pose.
```

A shared `(4, 4)` matrix broadcasts across a recording; `(N, 4, 4)` supplies one
matrix per frame in moving order. `result.transforms` can be reused directly:
labeled arrays must use `world_out/world_in` (and optional leading `time`), with
any supplied axis/time labels in the exact input order. No implicit reindexing or
inversion occurs. Supplied convention/unit metadata must agree with world z/y/x
fixed-to-moving pulls in mm.

For a pose adjusted manually in napari:

```python
registered = voxreg.register(
    pwd, template,
    intensity_scaling="none",
    initialization=np.linalg.inv(affine),
).registered
```

The inverse is correct **if** `affine` maps moving (`pwd`) physical world
coordinates into fixed (`template`) physical world coordinates. Voxreg needs the
opposite, fixed-to-moving pull. The matrix must use **world z/y/x in mm**; a napari
matrix in layer-data/pixel coordinates or x/y/z order needs conversion first.
Do not invert a matrix that is already a fixed-to-moving pull.

A homogeneous 4×4 matrix is often called an affine even when it contains only a
rotation and translation. Select `transform="affine"` to initialize and optimize
scale/shear too; the default `"rigid"` model still rejects them. Slice-to-volume
registration remains unsupported.

Rigid seeds must be finite proper rigid homogeneous matrices. Scale, shear and
reflection are rejected beyond a 1e-6 rotation-roundoff tolerance. Translation
seeds require an identity linear part; affine seeds use the determinant guard
above and are not projected into rotations. Accepted
float32 rotation roundoff is normalized while preserving the mapped fixed center.
Explicit matrix seeds keep the fixed geometric center and the selected model's
parameterization/Adam policy; following ITK's initialization model does not mean
identical optimizer trajectories.
For slices the seed must preserve the existing fixed plane. With an omitted
`fixed`, the first frame's seed must be identity (up to float64 roundoff), since
that reference is preserved exactly. Supply explicit `fixed` to fit every frame
from a shared nonidentity pose.

On failure the best valid full transform is retained, or the initial pose if no
valid metric was observed. Motion diagnostics describe that full transform.
There is no minimum coverage percentage: a smaller moving image can register
against a larger fixed image. Fewer than two valid sample centers still flags
`insufficient_overlap`; zero/invalid variance flags `invalid_metric`. The sample
minimum only makes correlation mathematically usable—it does not establish enough
information to identify every transform. Large displacements can be recovered from
a suitable seed, but broad capture range and robust partial-overlap alignment remain
unqualified.

### Automatic centering

```python
result = voxreg.register(
    moving_volume, fixed_volume,
    initialization="center_moments",  # Or "center_geometry".
    intensity_scaling="db",          # Scale fitting images, not the input data.
)
```

- **`center_geometry`** aligns physical grid centers, independently of intensities.
  The rotation pivot is the fixed geometric center.
- **`center_moments`** aligns intensity-weighted physical centroids of the original
  images. Following ITK, its rotation pivot is the **fixed intensity centroid**;
  physical lever-arm scales are recomputed around that pivot.

Both modes produce translation-only seeds (`moving_center - fixed_center` for pull
sampling), then use the shared optimizer and single final resampling path.
For recordings, moving moments are computed separately per frame; the fixed
centroid/pivot is shared. The implicit reference stays exactly identity.
`initialization` and `initial_transform` are mutually exclusive. Omitting both
retains identity initialization. Inspect `result.diagnostics.attrs` for
`initialization` and `rotation_center_zyx_mm` when automatic centering is selected.

**Warning—pass original linear intensities, not manually scaled images.** Use
`intensity_scaling="db"` for dB fitting instead of calling a logarithm/dB transform
before registration. Moments are computed **before** fitting scaling or NCC
conditioning, and outputs retain the original intensities. Nonlinear preprocessing
(including logarithms, square roots and power transforms) changes centroid weights;
max-referenced dB can make dark background dominate the centroid instead of bright
signal. Arbitrary intensity offsets also change moments. Voxreg cannot recover the
original weights or reliably identify every pre-scaled input.

Moments requires finite nonnegative original values with positive mass in **every**
image. Signed/already-dB input should use `center_geometry` or an explicit matrix
(with `intensity_scaling="none"`), not automatic shifting, absolute-value weights
or silent inverse scaling. The current fitting options remain `"none"` and `"db"`.
Bright artifacts, background, different crops and incomplete anatomy can still bias
linear-intensity moments; neither centering mode is an anatomical alignment guarantee.

Centering modes do not estimate rotation or search for an initial pose.
`align_planes` is a separate, explicit geometry alignment assumption, not an
image-based out-of-plane pose search for two slices. Separate fixed/moving pre-transform chains and pyramids
remain unsupported. See [VALIDATION.md](VALIDATION.md).

### Convergence

Each frame uses the ITK v4 window convergence calculation, implemented in PyTorch:
a normalized energy profile fitted with ITK's first-order B-spline construction.
Defaults use a **50-value window and NCC-calibrated 1e-6 threshold**:

```python
result = register(
    pwd,
    convergence_window_size=50,
    convergence_minimum_value=1e-6,
)
```

The monitor checks before the next Adam update; stopped frames retain independent
optimizer state and their best-valid transforms. This matches the convergence
calculation, not ITK's optimizer or necessarily its final transforms. A larger
threshold permits earlier stopping; it needs quality validation for your data.
Use `convergence_minimum_value=1e-8` for ITK gradient descent's literal threshold
default. Set `convergence_window_size=max_iterations + 1` for a fixed-budget control.
The calibrated threshold passed unchanged numerical gates and comparisons with
100/400-update controls across four real slice windows; it is not a universal
or general-3D calibration guarantee.

History stays on the selected device. Once a full window exists, whole-batch
completion is checked every iteration on CPU and every ten iterations on CUDA.
Each CUDA check reads one batch-wide boolean and synchronizes that stream—at most
five checks with the default 100-update budget. Individual stopping decisions do
not depend on batch companions. Frozen frames remain resident until all frames
finish; CUDA may execute up to nine extra frozen iterations between checks.
Speedups depend on the slowest frame, and polling can cost time when no batch
finishes early. See [calibration and timing evidence](research/convergence/CALIBRATION.md)
and the [ITK reference calculation](research/convergence/README.md).

### Image dtype and final interpolation

Input arrays are never modified. Working fitting images are float32; geometry,
Adam state, NCC and world matrices remain float64. Integer/float64 preprocessing
runs in float64 and uses NCC's intensity-offset/scale invariance to retain
low-contrast detail before conversion to fitting tensors.

Multithreaded CPU 3D SyN uses scaled float32 Gaussian convolutions for its two
regularizers, restoring float64 fields afterward. Coordinates, field storage,
composition, NCC statistics, inversion and Jacobians remain float64. Planar,
single-threaded, differentiable and compiled paths retain their prior smoothing;
there is no new precision option. See the [AC-qualified comparison with ITK and
FireANTs](research/syn/AC-CPU-SMOOTHING.md).

Final resampling uses the **original values**, once, independently of fitting:

- Float64 images are interpolated in float64 and returned as float64—not sampled
  in float32 and merely cast back. Float32 images stay float32; float16 output is
  retained after float32 interpolation.
- Integer images are interpolated in float64, rounded to nearest-even, clipped
  to their type range and returned in their original dtype. This necessarily
  quantizes fractional intensities. Integer values outside `[-2**53, 2**53]` are
  rejected rather than silently losing precision on conversion.
- `interpolation="linear"` is the default; `"nearest"` changes **final output only**.
  Fitting always uses differentiable linear interpolation. Nearest is appropriate
  for categorical labels, not automatically for integer-valued intensity images.
- Boolean/complex/extended-precision floating intensity arrays are not supported.

There is no public compute-precision or compiler switch. Float32 fitting and
float32-only NCC failed the existing CPU/CUDA agreement gates; see
[the dtype validation report](research/performance/DTYPE.md).

**Transform direction:** `moving_world = matrix @ fixed_world`, using homogeneous
**z/y/x** vectors. These are physical pull mappings, not normalized Torch grids.
Matrix inversion gives inverse world transforms. Slice `translation_i/j` describe
center displacement along an orthonormal fixed-plane basis. Volume
`rotation_z/y/x` are components of a rotation vector (radians, right-handed XYZ
convention, stored z/y/x), **not Euler angles**; `translation_z/y/x` describe center
displacement in world mm. These translations are not the homogeneous matrix's
translation column. Parameter scaling uses minimum fixed spacing and RMS physical
lever arms, not ITK's physical-shift estimator.

Spatial cropping/striding and oblique/anisotropic geometry are handled through
native coordinates and a lazy `VoxelToWorldIndex`. Use
`voxreg.position_to_world(data)` when a zero-based array-position affine is
needed. Assigning new native coordinates without rebuilding the world index is
an error, not a supported way to relocate an image.

### Using ConfUSIus VoxelData directly

There is now one index implementation: ConfUSIus's. No conversion is needed:

```python
import confusius as cf
import voxreg

pwd = cf.load(path).compute()
result = voxreg.register(pwd, device="cuda", batch_size=120)
```

Inputs and registered/valid outputs use the same geometry class, including
ConfUSIus's world-coordinate lookup. Voxreg still enforces its registration subset:
NumPy-backed arrays, integer native voxel indices, uniform spatial steps and mm.
Slices retain the orthogonal-axis requirement, with k=1 and j/i >= 2;
full volumes require all axes >= 2. Noncoplanar
slice pairs need a compatible initialization. It does not modify source values
or geometry.
Start with `pwd.isel(time=slice(0, 120))` before processing a long recording.
The default device is CPU; select CUDA explicitly when available.

VoxelData construction and general validation belong to ConfUSIus:
`cf.create_voxeldata(...)` and `confusius.validation.validate_voxeldata(...)`.
Voxreg no longer exports either helper. Migrate old `coords={...}` constructor
calls to ConfUSIus's `time=`, `k=`, `j=` and `i=` arguments. Its timing metadata rules
also apply; a singleton time coordinate needs `volume_acquisition_duration=` when
no duration can be inferred. Solver-specific geometry checks remain private and
run automatically inside registration and `position_to_world`.

Existing floating native-coordinate arrays must use integer voxel indices; no
rounding or coordinate rebasing is performed automatically. Restart an existing
Python kernel after updating to discard the old exports/index implementation.

### CPU and CUDA progress

**Progress defaults to `"auto"`** on both CPU and CUDA: spatial-only pairs and
single-frame recordings are quiet; multiple moving frames show ConfUSIus's Rich
progress-bar style with percentage, frame counts, elapsed time and ETA.
Use `progress=True` to force a bar, including for a single pair.
No callback is needed:

```python
result = voxreg.register(pwd, device="cuda", batch_size=120)
# Quiet execution: progress=False (or None).
result = voxreg.register(pwd, device="cpu", progress=False)
```

A callback overrides the built-in display and receives
`(completed_frames, total_frames)` after each completed batch:

```python
def report_progress(completed, total):
    print(f"Registered {completed:,}/{total:,} frames")


result = voxreg.register(pwd, progress=report_progress)
```

Progress counts processed frames, including reference/failed frames, not optimizer
updates. Updates happen after the existing CPU downloads, with no additional CUDA
synchronization. Preparation and final output assembly do not emit events; smaller
batches give more frequent updates but may reduce throughput. Reporting updates
are outside stage timings; callback exceptions propagate and the built-in display
is closed on errors. Timed/profiling harnesses explicitly use `progress=False`.
Frame progress alone adds no extra dependency or per-iteration GPU polling.

### Live optimizer metrics (opt-in)

Use a separate metric callback to print, collect a history, or feed a live plot:

```python
def report_metric(frame_indices, updates, objective):
    for frame, step, loss in zip(frame_indices, updates, objective, strict=True):
        print(f"Frame {frame}, update {step}: trial NCC²={-loss:.6f}")


result = voxreg.register(
    moving, fixed,
    metric_callback=report_metric,
    metric_interval=10,  # Optimizer rounds; use 1 for a diagnostic trace.
)
```

The callback receives owned NumPy arrays: global moving-frame indices, per-frame
update counts, and raw negative NCC² (lower is better; invalid metrics are NaN).
Initial and final trials are always reported; stopped rows can repeat. The
returned result retains the **best valid pose**, not necessarily the final trial,
so read its final metric from `result.diagnostics.objective`.

No history is stored automatically and no plotting dependency is added. CUDA
reporting transfers small arrays and synchronizes **only when opted in**; throttle
plot redraws separately (for example, a few times per second). Metric-callback
work is included in optimization timing and exceptions propagate. With no metric
callback, numerical execution and default frame progress are unchanged.

## Reproduce the real motion-correction example

The example uses Cybis Pereira 2026, rat75, session 20220523, slice32, frames
**220:340**. Supply the downloaded NIfTI and its BIDS JSON sidecar; there is no
built-in downloader. The sidecar supplies the acquisition's irregular time labels.

From the voxreg directory:

```bash
INPUT="$HOME/.cache/confusius/cybis-pereira-2026-bids/1.0.0/sub-rat75/ses-20220523/fusi/sub-rat75_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz"

uv run --extra examples python examples/notebook_motion_correction.py \
    "$INPUT" --device cuda --output artifacts/notebook
```

This produces original-power NIfTI recordings, validity and difference volumes,
exact timing sidecars, transforms/diagnostics in NPZ and CSV, and motion, voxel
trace and raster PNGs. Open the NIfTI outputs in napari with a NIfTI reader.
Artifacts stay local and are ignored by Git.

### Optional frozen ITK comparison

The reference exporter uses **an existing ConfUSIus environment**, solely to run
its original public registration functions. ConfUSIus is also installed into
voxreg for shared VoxelData; the reference still runs a separate numerical pipeline. With the neighboring checkout layout used during development:

```bash
# Run with ConfUSIus's interpreter; quality export only, no timing repetitions.
uv run --directory ../../confusius-tools/confusius python \
    "$PWD/examples/freeze_itk_reference.py" "$INPUT" \
    "$PWD/artifacts/notebook/itk_reference.npz"

# Back in voxreg's environment: registration still uses voxreg's PyTorch engine.
uv run --extra examples python examples/notebook_motion_correction.py \
    "$INPUT" --device cuda --output artifacts/notebook \
    --reference artifacts/notebook/itk_reference.npz
```

The frozen reference records source hashes, resolved ITK registration settings,
input arrays/coordinates, pull matrices and original-power registered outputs.
The comparison uses one independent SciPy bilinear sampler on dB inputs and
**common valid support**, not each implementation's internal optimizer score.
Both outputs retain the source's decoded intensity dtype. Voxreg uses float32
working fitting images with float64 NCC statistics, then resamples the original
float64 intensities for this NIfTI example.

Timing repetitions default to **zero**. Only on an otherwise idle machine, add
`--repeats 3` to the respective scripts. The reference exporter measures the warm
public pipeline at `n_jobs=1` and `--jobs 8`, using its default joblib backend.
Run each configuration separately; do not infer speedups from contested timings.

## Standalone time and memory comparisons

The motion-correction example also writes `artifacts/notebook/input.npz`. This
neutral input artifact lets the ITK/FireANTs benchmark run entirely in voxreg's
Python environment, without calling ConfUSIus's registration pipeline:

```bash
# Practical subset; run only on an otherwise idle machine.
uv run python examples/benchmark_registration.py \
    --cases itk-1-t8 itk-8 cpu-32-t8 cuda-120 \
            fireants-cpu-32-t8 fireants-cuda-120

# Omitting --cases also includes deliberately slow unbatched controls.
# The full suite can take more than ten minutes.
```

Each configuration uses fresh interpreters, with **separate time and memory
passes**. Warm timings include preprocessing, original-power resampling and
return to host. Memory reporting includes process-tree RSS/PSS, exact PyTorch
CUDA allocator peaks, and sampled per-process GPU VRAM including context memory.
Process-parallel ITK includes scheduling and result-transfer overhead. Startup
costs and loaded-input/warm memory baselines are recorded separately.

See [BENCHMARK.md](BENCHMARK.md) for measured timings, memory, quality and caveats.
The historical float32-input benchmark takes **0.219 s on full-batch CUDA** and
**2.128 s on eight-thread CPU**. Fresh FireANTs controls take 0.355 s and 0.963 s
respectively: CUDA is competitive, while CPU remains slower than native float32
FireANTs. These are not precision-
matched solvers or universal speedup guarantees. Comparisons use direct 2D APIs,
not the ConfUSIus pipeline or its earlier frozen-reference optimizer configuration.
See the report for memory, quality, source hashes and the pinned FireANTs archive.

[PERFORMANCE.md](PERFORMANCE.md) preserves the profiler investigation that led to
this rewrite. Float64 physical geometry/state/NCC and independent frame policies
are unchanged; only dense coordinate evaluation was reorganized.

### Consistent 3D-pair and series comparisons

Use the same runner with an explicit standardized workload. Historical commands
without `--workload` retain their older configuration; historical results are not
overwritten or mixed with the new measurements.

```bash
# Prepare independent 3D grids and analytic known motion (outside timing).
uv run python examples/prepare_registration_benchmark.py \
    --size 128 --independent-grids --output artifacts/registration-inputs/pair128.npz

# Native ITK / voxreg / FireANTs, CPU and CUDA; one cold + three warm calls,
# with separate fresh-process memory passes. Output directory must be new/empty.
uv run python examples/benchmark_registration.py \
    --workload pair --policy fixed \
    --input artifacts/registration-inputs/pair128.npz \
    --output artifacts/registration-results/pair128-fixed

# Same protocol, existing motion-correction input and explicit batches.
uv run python examples/benchmark_registration.py \
    --workload series --policy convergence --intensity-scaling db \
    --input artifacts/notebook/input.npz \
    --cases itk-1-t8 cpu-32-t8 cuda-120 fireants-cpu-32-t8 fireants-cuda-120 \
    --output artifacts/registration-results/series-convergence
```

`fixed` compares a common update budget, not identical optimizers. `convergence`
uses explicitly recorded backend stopping rules; native FireANTs stops too early
on two 3D controls and fails their unchanged RMS truth screen. Its shorter times
are not equal-quality speedups. Independent SciPy quality uses common support;
known physical truth error, versions, precision, source/input hashes, update
counts, statuses and power-state snapshots are recorded in the shared schema.

See [the AC-powered results and reproduction guide](research/registration/README.md)
for latency, memory, timing ranges and limitations. Earlier uncontrolled-power
runs are kept separately, not mixed into the AC tables.

## Current limits

**Real-example capture warning:** the default 100-update budget still under-captures
ConfUSIus's same-subject session pair. The former 90% coverage cutoff has been
removed: full-image fits can now continue to better alignment without a manual
crop. See the [coverage-policy follow-up](research/overlap-policy/README.md) and
[original guarded results](research/confusius-examples/README.md). This does not
calibrate defaults or guarantee robust matching from small overlaps.

The [public-API example benchmark](research/confusius-examples/BENCHMARK.md) compares
fresh float64 same-subject and motion-correction inputs with ConfUSIus 0.8.0:
actual call times, common-support scores and inspected overlays. It retains a
fixed100 motion-control CPU/CUDA precision failure (0.001965 px versus the unchanged
0.001-pixel limit); native-stopping motion correction passes that screen. Atlas
slice-to-volume and B-spline stages had no counterpart at that benchmark boundary
and were not timed; the new slice-to-volume retry is documented separately.

- Translation, rigid and affine: slice pairs, volume pairs and mixed slice/volume
  fitting. No deformable, multi-pose or unconstrained 12-parameter slice affine.
  Noncoplanar slice pairs require an explicit alignment assumption or seed.
- Independent fixed/moving grids, but uniform native coordinates, mm and finite
  NumPy-backed real input with supported dtype/range remain required. No masks,
  Dask or multiresolution fitting.
- No fixed overlap percentage is required. Fewer than two valid sample centers
  is diagnosed as insufficient overlap; constant/nonfinite metrics separately
  fail validation. Frames retain their best valid pose, or the seed if no valid
  metric was observed. `valid_fraction` reports coverage, **not confidence**;
  small overlaps can yield misleadingly high correlations.
- Output fill is each original frame's minimum. Use `result.valid` to exclude
  padding from downstream analyses.
- Per-frame status distinguishes convergence, iteration limit, invalid metric,
  insufficient overlap, invalid affine transform and the fixed reference.
  An iteration limit is **not** a
  convergence claim. With the calibrated defaults, one of 476 moving frames across
  four checked windows still reaches the 100-step limit; other data may need more.
- Fitting interpolation is float32; physical geometry, optimizer state, NCC
  statistics and world matrices are float64 for stability. Final interpolation
  preserves original image precision as described above. Cross-device transforms
  are numerically, not bitwise, equal; output dtype is not a transform-accuracy guarantee.
- The lazy index builds on xarray's experimental coordinate-transform index API.

No compiled engine, deformable extension or compatibility wrapper is included.
Performance claims are limited to the documented pair/series controls and their
measurement boundaries. Natural independent-session 3D registration, broad capture
range and general 3D-series memory scaling remain unqualified.
