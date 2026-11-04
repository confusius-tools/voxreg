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

**Linear models** share preprocessing, masked squared global NCC, per-frame Adam,
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
local NCC (radius 2), normalized updates and Gaussian regularization (update sigma
1.5, total sigma 0.5 voxels). Complete pulls compose one halfway map with the
numerical inverse of the other. An orientation-preserving affine seed is fixed
during field fitting and included in the returned displacements.

SyN is single-resolution. Sampled halfway and final residual Jacobians must be
finite and positive; inversion is approximate. These checks do not prove global
invertibility or anatomical correctness. See [validation limits](VALIDATION.md).

## Initialization and failure handling

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
There is no minimum coverage percentage: two supported samples only make NCC
mathematically possible, not a reliable registration. Coverage is not confidence.

## Precision, output and execution

| Computation | Precision/policy |
| --- | --- |
| Fitting images and global/local NCC | Float32; explicit double metric inputs retain double for checks |
| Geometry, optimizer state and fields | Float64 |
| Smoothing | PyTorch; CPU SyN regularizers use scaled float32, restoring double fields; general/differentiable/compiled and CUDA paths retain double filtering |
| Final interpolation | Original images, once; float64/integer inputs retain double; outputs retain moving dtype |

Float32 NCC uses aligned global reductions and local pooling sums with
roundoff-aware variance guards. dB scaling affects fitting only; original inputs
are never modified. Integer results are rounded nearest-even and clipped;
integer inputs outside `[-2**53, 2**53]` are rejected.

Offline CPU linear resampling uses native double-image/double-coordinate sampling,
without fitting-only coordinate round trips. Single volumes partition query points
over shared image storage; fitting and CUDA sampling paths are unchanged.

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
50 values and threshold `1e-6`. A window greater than `max_iterations` disables
convergence checks. Stopped rows freeze while companions continue; whole-batch
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

No backend registry, public precision/compiler switch or transform-object
framework is needed. [Tests](tests/) own mathematical contracts;
[BENCHMARK.md](BENCHMARK.md) owns measurement policy. Older design milestones are
linked in the [historical guide index](research/README.md#pre-cleanup-guides).
