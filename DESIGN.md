# Registration design

`voxreg.register` fits physical fixed→moving pulls with PyTorch. ConfUSIus
provides the xarray-based VoxelData model and progress display—not the fitter.
See the [public API](README.md#voxregregister) for usage and the function's
[docstring](src/voxreg/rigid.py) for all options.

## Geometry and inputs

- Moving data is `(k, j, i)` or `(time, k, j, i)`; fixed data is spatial-only.
  A recording without explicit fixed uses its first frame, preserved exactly.
- Inputs are finite, real, NumPy-backed VoxelData in mm. Native spatial coordinates
  are uniform, monotonic integers; offsets, crops, strides and reversal are supported.
- Fixed/moving shapes, spacings, origins and orientations may differ. Geometry
  comes from ConfUSIus's lazy world index, not mutable metadata shortcuts.
- Slices have `k=1` and at least two samples on `j/i`; volumes have at least two
  samples on every axis. Unsupported singleton layouts and extra dimensions fail.

Public transforms use **world z/y/x in mm**. For linear models:

```text
moving_world = T @ fixed_world
moving_position = inverse(G_moving) @ T @ G_fixed @ fixed_position
```

`G` maps zero-based array positions to world coordinates, including the native
coordinate offset and stride. It is not necessarily the stored voxel-to-world
affine; `voxreg.position_to_world(data)` supplies this map. Identity world motion
can therefore resample unequal grids without estimating geometry as motion.

Opt-in `forward_transforms` and `inverse_transforms` expose both directions for
all models. `voxreg.resample` applies either complete pull to original intensities
on CPU/CUDA without fitting. Dense fields must match the reference lattice;
linear matrices can be applied on any supported reference lattice.

## Models

| Model | Slice pair parameters | Volume pair parameters | Mixed slice/volume parameters |
| --- | ---: | ---: | ---: |
| Translation | 2 | 3 | 3 |
| Rigid | 3 | 6 | 6 |
| Affine | 6 | 12 | 9 |
| SyN | Two 2-component halfway fields | Two 3-component halfway fields | Unsupported |

**Linear models** share preprocessing, per-frame metrics (global squared NCC by default), Adam,
convergence and final resampling. Translations are physically scaled center shifts;
rotations/affine coefficients use fixed-grid RMS lever arms and minimum spacing.
Volume rotations are rotation vectors, not Euler angles. Centered parameters and
the exported matrix's translation column are not interchangeable.

Slice pairs optimize an orthonormal intrinsic plane, without fitting unobservable
normal motion. Noncoplanar planes require a compatible seed or explicit
`align_planes`, which assumes corresponding sections. Mixed fitting always uses
the caller's fixed grid: no role reversal or replicated slab. Mixed affine retains
nine identifiable parameters with a unit orthogonal normal completion. Singleton
source support has finite half-voxel extent, not zero thickness.

**SyN** optimizes two compositive midpoint-to-image fields using masked squared
local NCC by default (radius 2), normalized updates and Gaussian regularization
(default update sigma 1.5, total sigma 0.5 voxels). The public `syn_cc_radius`,
`syn_update_sigma` and `syn_total_sigma` options control these values. CC windows
are rectangular; field sigmas are standard deviations in voxel units, with zero
disabling the corresponding smoothing. Complete pulls compose one halfway map
with the numerical inverse of the other. An orientation-preserving affine seed is fixed
during field fitting and included in the returned displacements.

Sampled halfway and final residual Jacobians must be
finite and positive; inversion is approximate. These checks do not prove global
invertibility or anatomical correctness. The guard rejects sampled collapse
(det = 0), orientation reversal (det < 0), and nonfinite derivatives; it is a
voxreg safety policy, not a required FireANTs optimizer step. It was introduced
with native SyN (`2a65a6d`) from the local orientation-preserving requirement
for diffeomorphisms, not copied from a vendor acceptance rule.
See [validation limits](VALIDATION.md).

## Image pyramids

All models support explicit `shrink_factors`, `smoothing_sigmas` in mm, and scalar
or per-level `max_iterations`. The default `(1,)` / `(0.0,)` schedule retains the
original native fitting path and diagnostics. Each nondefault level derives
directly from normalized native fitting images, without cumulative image
interpolation. Physical Gaussian smoothing precedes striding; orthogonal grids
use separable kernels, sheared grids use a full physical-distance kernel and FFT
convolution. Edges are replicated. Singleton axes remain singleton; strides are
capped to retain at least two samples on active axes. All levels share index zero
and preserve the original coordinate offsets, orientation and native strides.

Linear fitting transfers complete world matrices and retains the native physical
pivot while recalculating level-specific parameter scaling. SyN transfers both
halfway fields in mm through physical interpolation, without voxel-size magnitude
scaling. Interpolation uses zero residual outside coarse field support and fixes
the destination boundary. Interpolated fields must pass finite/positive sampled
Jacobian checks; invalid seeds are marked and replaced by zero residuals. Fresh
optimizer, convergence and best-loss state starts at every level, including a
retry for frames stopped at coarser levels. Losses are never compared across
levels. Per-level status, objective and actual update counts remain in diagnostics.

SyN inversion/composition and safety checks run once on the native fixed grid
after the final level. Final original-intensity sampling and optional world-field
export use the original fixed/moving grids, even with a coarse final level.
Pyramid preparation and transfer are included in optimization timings; runtime
and anatomical benefits require separate measurements.

## Initialization and failure handling

`register(transform=("rigid", "affine", "syn"))` runs explicit ordered stages;
`Stage` supplies immutable per-stage overrides of shared fitting arguments.
Only increasing model complexity is accepted, with optional skipped models and
at most one terminal SyN. Single-model string calls retain the existing path.
Workflow presets and automatic stage selection are not implemented.

All effective fitting configurations are validated before optimization begins.
Initialization runs once. Intermediate linear stages return matrices, their
native physical pivot and diagnostics through a private fit-only path; they do
not allocate resampled original intensities. Subsequent stages recover their own
parameters from those complete matrices, retaining the pivot. Terminal SyN holds
the affine seed fixed and includes it in its complete fields. Output resampling
runs only for the final stage. All stages fit the original inputs.

Per-frame stage failure freezes that frame for subsequent stages (status skipped,
zero updates), retaining its best valid map, while companions continue. Converged
and max_iterations outcomes permit the next stage. The final summary preserves
the originating failure status and sums update counts and phase timings across
stages; validation/orchestration overhead is recorded separately. Independent
stage datasets retain model-specific variables and differing pyramid schedules
without padding or comparing their losses. Metric callback clocks restart per
stage/level; progress counts completed stage/frame work units.

Identity is the default. Matrix-valued `initialization` aliases
`initial_transform`; supplying both is an error. Seeds are copied and returned
transforms describe the complete fit, not an increment. Linear fitting refines
the full seed rather than freezing a pre-transform.

- `center_geometry` aligns physical grid centers.
- `center_moments` uses original nonnegative intensities with positive mass.
  Native-pair rotation pivots follow the fixed intensity centroid; mixed fits
  retain the fixed geometric pivot. Do not supply manually logged images.
- `align_planes` aligns centers and unoriented normals before in-plane fitting;
  ordinary centering does not correct tilt.

Rigid seeds require proper rotations; affine seeds must be numerically
nonsingular and fitting preserves their determinant sign. A reflected affine
seed is permitted for linear affine fitting, not SyN. An implicit reference must
start at identity; use explicit fixed to fit every frame from a nonidentity seed.

State, optimizer clocks and stopping decisions are independent per frame. The
returned pose is the best valid trial, or the seed if no trial was valid. Invalid
metrics, insufficient overlap and invalid transforms are explicit statuses.
There is no minimum coverage percentage: two supported samples only make a metric
mathematically possible, not a reliable registration. Coverage is not confidence.

## Metric interface and mutual information

The metric boundary accepts moving/fixed samples and a shared boolean support
mask, returning a per-frame differentiable loss and usability flag. Geometry,
ROI construction, best-state selection, callbacks and stopping remain outside it.
Defaults retain global NCC for linear models and local NCC for SyN; explicit
global NCC and Gaussian-Parzen MI also operate on SyN's two halfway images.

MI maps each native fitting image's min/max to [0, 1] after optional dB scaling.
These full-image ranges include values outside masks and are fixed before pyramid
smoothing. Native offset subtraction precedes division for high-offset double
or integer images; an overflow-safe path handles opposite extreme intensities.
Original image values and final interpolation are independent of this conditioning.

Uniform bin centers span [0, 1]. For intensity u, memberships are proportional
to exp(-0.5 * ((u * (bins - 1) - center) / sigma)^2), normalized over bins per
sample. The joint distribution averages membership outer products only over
valid samples; marginals come from the same joint. The loss is negative MI in
natural logarithms. Nonconstant supported intensities are required on both sides.
Finite-difference checks differentiate both images, including the halfway pulls.

This implementation was written independently. Gaussian Parzen estimation is
also described in [MONAI's MI documentation](https://docs.monai.io/en/stable/losses.html#globalmutualinformationloss).
It differs from [ITK Mattes MI](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1MattesMutualInformationImageToImageMetric.html),
which uses distinct fixed/moving B-spline kernels; values are not interchangeable.

At most 4096 samples per spatial chunk and approximately 2^20 bin memberships
per side are formed at once (at least one sample per frame). Non-reentrant
checkpointing recomputes memberships during backward. Workspace scales with
batch times bins squared plus chunked memberships, rather than voxels times bins;
images, gradients and outputs still scale with voxel count. The estimator does
not randomly subsample the ROI. Bins and sigma are bounded public controls.
Objective units, estimator settings and range policy are recorded in diagnostics;
SyN endpoint NCC remains an independent quality measure. Metric-specific stopping
calibration, outlier handling and real-data performance remain qualification work.

## ROI masks

`fixed_mask` and `moving_mask` are boolean VoxelData in independent physical grids.
Fixed masks are spatial-only; moving masks can be spatial/shared or follow the
recording's exact time labels. Physical queries use origin-free float64 geometry
and binary cell membership, with ITK-style half-integer rounding upwards. Masks
are evaluated on their original grids at each level and candidate transform;
they are not blurred or repeatedly resampled. Fixed ROI, moving ROI and image
support intersect before the metric calculation. Local CC uses that intersection
for all neighbourhood sums and for eligible window centres.

Linear fitting holds the fixed ROI stationary and pulls the moving ROI through
the complete physical matrix. SyN moves both ROIs through the two halfway maps,
including affine prealignment on the moving side. Hard ROI membership has no
autograd derivative. Image interpolation/smoothing and intensity conditioning
remain independent of the masks; outside-ROI intensities can influence smoothed
or interpolated boundary values. Initialization still uses the original images.

Masks apply to every stage and level. Frame isolation and skipped-stage rules
remain unchanged. Geometric output validity is not replaced by ROI support:
`roi_valid_fraction` reports native fixed-grid endpoint ROI/support intersection
separately, while `valid` continues to cover the complete resampled output.
Regularization can carry deformation outside the ROI; whole-field Jacobian
checks still apply. Empty/coarsely vanished ROIs fail explicitly and pyramid
fitting can retry at finer levels under its existing rules.

Reference choices were checked against
[ITK physical mask membership](https://github.com/InsightSoftwareConsortium/ITK/blob/main/Modules/Core/SpatialObjects/include/itkImageMaskSpatialObject.hxx)
and native masked global correlation. This is not a claim of full optimizer or
local-CC equivalence. FireANTs' inspected masked-CC path weights the loss map after
computing neighbourhood statistics; voxreg excludes masked neighbours from those
statistics. FireANTs also permits interpolated fractional mask channels, whereas
voxreg uses explicit binary membership. ANTsPy defaults to masks on the final
stage; voxreg deliberately applies shared masks throughout its explicit pipeline.

## Precision, output and execution

| Computation | Precision/policy |
| --- | --- |
| Fitting images and NCC/MI metrics | Float32; explicit double metric inputs retain double for checks |
| Physical geometry and linear pose parameters | Float64 |
| SyN fitting | Float32 working fields, origin-free field arithmetic and native PyTorch `grid_sample`; support checks retain float64 physical queries |
| SyN finalization | Float64 inversion, composition, validation and world displacement export; casting does not recover fitting precision |
| Smoothing | PyTorch in field dtype; CPU SyN regularizers use scaled float32, restoring the field dtype |
| Final interpolation | Original images, once; float64/integer inputs retain double; outputs retain moving dtype |

Float32 NCC uses aligned global reductions and local pooling sums with
roundoff-aware variance guards. SyN rejects lattices that become nonfinite or
singular in float32 working arithmetic rather than silently corrupting fields.
dB scaling affects fitting only; original inputs
are never modified. Integer results are rounded nearest-even and clipped;
integer inputs outside `[-2**53, 2**53]` are rejected.

Offline linear resampling uses native double-image/double-coordinate sampling on
CPU/CUDA, without fitting-only coordinate round trips. Single CPU volumes partition
query points over shared image storage; CUDA affine queries use separable coordinates
instead of a dense skinny matrix product. Fitting sampling is unchanged.

Output uses the fixed grid and moving intensity dtype/units/time labels. Spatial
pairs acquire no synthetic time axis. Out-of-bounds output is filled with the
original frame's minimum; `valid` identifies supported samples. Transform export
is opt-in: linear matrices or complete forward/inverse SyN displacement fields
on the fixed/moving grids, respectively.

Automatic batching uses a CPU cap and conservative CUDA memory estimates;
`batch_size` overrides it. It is not an autotuner or an OOM retry. Full CPU outputs
and optional dense exports are not bounded by the batch size. Two equal-grid
float64 SyN world fields alone cost 12× float32 image storage.

Convergence uses a device-resident ITK-style window calculation, defaulting to
50 values and threshold `1e-6`. A window greater than a level's iteration budget
disables convergence checks there. Stopped rows freeze while companions continue; whole-batch
completion is polled every CPU iteration or every ten eligible CUDA iterations.
Matching ITK's monitor does not mean matching its optimizer or final pose.

Frame progress reports completed batches. Optional metric callbacks report owned
NumPy arrays of frame indices, update counts and trial objectives; they can add
CUDA synchronization and do not change best-valid return semantics.

## Code ownership

| Module | Responsibility |
| --- | --- |
| [geometry.py](src/voxreg/geometry.py) | VoxelData validation and array-position geometry |
| [transforms.py](src/voxreg/transforms.py) | Linear parameterization, scaling and seed constraints |
| [resampling.py](src/voxreg/resampling.py) | Offline world-pull application on CPU/CUDA |
| [sampling.py](src/voxreg/sampling.py) | Physical pull sampling, global NCC and matrix conversion |
| [rigid.py](src/voxreg/rigid.py) | Public API, shared linear fitting and result assembly |
| [fields.py](src/voxreg/fields.py) | Field interpolation, composition, smoothing, inversion and Jacobians |
| [syn.py](src/voxreg/syn.py) | Symmetric field fitting and dense outputs |
| [convergence.py](src/voxreg/convergence.py) | Independent window convergence |
| [pyramid.py](src/voxreg/pyramid.py) | Physical image smoothing, pyramid lattices, field transfer and level diagnostics |
| [stages.py](src/voxreg/stages.py) | Stage specifications, shared-option inheritance and per-frame pipeline execution |
| [masks.py](src/voxreg/masks.py) | Physical binary ROI membership and metric-domain mask intersections |
| [metrics.py](src/voxreg/metrics.py) | Per-frame objective selection and chunked differentiable MI |

No backend registry, public precision/compiler switch or transform-object
framework is needed. [Tests](tests/) own mathematical contracts;
[BENCHMARK.md](BENCHMARK.md) owns measurement policy. Older design milestones are
linked in the [historical guide index](research/README.md#pre-cleanup-guides).
