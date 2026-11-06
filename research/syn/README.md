# Symmetric compositive SyN

> Historical qualification record. Exploratory Python runners were removed;
> commands below describe the original checkpoint. Use the [current benchmark](../../BENCHMARK.md)
> for new measurements; see the [archive scope](../README.md) for retained source/evidence.

`register(..., transform="syn")` now supports native **2D–2D and 3D–3D** pairs
and recordings. It shares linear registration's validation, initialization,
preprocessing, physical sampling and original-intensity output contracts, without
changing the linear optimizer or its defaults. This is a single-resolution research
implementation, not a claim of general anatomical registration success.

See [accuracy and speed against native ITK and FireANTs](BENCHMARK.md) and the
[source-hashed measurements](benchmark-results.json).
[Acceleration investigation](SPEED.md) records current full-recording timings,
qualified eager changes, CPU work and compiler failures separately from prototypes.

## Algorithm and directions

`src/voxreg/syn.py` independently optimizes two physical displacement fields:
`F: midpoint → fixed` and `M: midpoint → prealigned moving`. Both images are sampled
at those halfway maps. Masked squared neighborhood NCC supplies gradients for both
fields; normalized smoothed updates compose with their accumulated maps. The
residual endpoint pulls are **`M ∘ F⁻¹`** and **`F ∘ M⁻¹`**. This is not a generic
one-way displacement fit or a stationary velocity field relabeled SyN.

An orientation-preserving affine seed `A` prealigns the moving image. The complete
forward is `A ∘ M ∘ F⁻¹`; the inverse is `F ∘ M⁻¹ ∘ A⁻¹`. The seed is fixed during
field fitting. Use the original moving/fixed images for affine then SyN, not the
affine-resampled image. Identity, geometry/moments and compatible plane-alignment
initialization reuse the existing input path. Dense restart seeds are unsupported.

Current settings are radius-2 local NCC, update sigma 1.5 and total sigma 0.5
voxels. `learning_rate` bounds each halfway update's maximum voxel-coordinate norm.
These are explicit initial settings, not broadly calibrated capture-range defaults.
The best usable halfway-objective state is retained per frame. `ncc_squared`
measures the endpoint image separately; callbacks report raw halfway trials.

References inspected (not runtime delegates):

- ITK `090ded25eff27f6d76888836538f7a16900cb327`:
  `Modules/Registration/RegistrationMethodsv4/include/itkSyNImageRegistrationMethod.hxx`
  (`StartOptimization`, `ScaleUpdateField`, `InvertDisplacementField`,
  `GaussianSmoothDisplacementField`, `GenerateData`) and
  `Modules/Filtering/DisplacementField/include/itkInvertDisplacementFieldImageFilter.hxx`.
- FireANTs `f8a9cc259ab9113fc43bd99797000fa8fc34eb59`:
  `fireants/registration/syn.py` and `registration/deformation/compositive.py`.
  Its batched halfway structure informs execution, not copied optimizer code.
  FireANTs has a restrictive license; no source is transplanted or vendored.

Matching library optimizer trajectories is not an acceptance requirement. The
Gaussian kernels, update rules, inverse solvers and best-state policies differ.

## Public results

```python
from voxreg import register

result = register(
    moving,
    fixed,
    transform="syn",
    intensity_scaling="none",
    device="cuda",
    batch_size=1,
    return_transforms=True,
)
registered, valid = result.registered, result.valid
forward, inverse = result.transforms, result.inverse_transforms
```

- `registered`/`valid`: original moving intensity dtype/units/time metadata on the
  fixed grid, with original-intensity final interpolation and per-frame-minimum
  fill outside valid moving sample centers. An implicit first reference frame is
  preserved exactly. Integer output uses clipped nearest-even rounding.
Transforms are opt-in and **None by default**. The example explicitly requests
them. With equal grids and float32 images, both float64 world fields occupy
**12× uncompressed image storage**, plus assembly temporaries. Disabling export
skips field construction, CPU download and accumulation; per-batch state is still
needed for fitting, final interpolation and inverse/regularity diagnostics.
Automatic CUDA batching uses a VRAM estimate with a measured performance cap,
not a fill-all-VRAM policy; explicit integers override it. See
[batch measurements](../performance/BATCH-SIZE.md).

- `transforms`: complete fixed-world → moving-world **displacements in mm** on
  the fixed grid, ending in `world_component=["z", "y", "x"]`.
- `inverse_transforms`: complete reverse displacements on the **original moving
  grid**, with the same component convention. Add a displacement to its physical
  query point; do not interpret either array as a 4×4 matrix.
- Recordings retain `time`; spatial-only pairs have no synthetic time dimension.
  Linear outputs remain matrices, with `inverse_transforms=None`.
- `diagnostics`: objective, endpoint global NCC², updates/status, validity,
  minimum sampled Jacobian, two-sided residual inverse error and stage timings.
  The inverse-error diagnostic is measured in voxel-coordinate norm on the fixed
  virtual lattice **before the affine seed**, not the full unequal-grid map.

Halfway candidates require finite positive sampled Jacobians. Final residual maps
are also checked; final composition failure returns the seed and an explicit
`invalid_transform` status. Sampled positive determinants are **not proof of global
invertibility**. Inverse residuals are exposed, not asserted to vanish.

## Field and numerical contracts

`src/voxreg/fields.py` defaults to float64 millimetre geometry/state and tensor layout
`(batch, components, *active_spatial_shape)`. Volumes use world z/y/x; planes use
the existing orthonormal two-component basis, embedded into world components on
export. No fabricated depth, replicated planes or hidden role reversal is used.

Composition is `outer ∘ inner`: `inner(x) + outer(x + inner(x))`. Fields have finite
half-open voxel-cell support `[-0.5, size-0.5)`, border interpolation within it and
zero contribution outside. Smoothing is a separable sampled Gaussian in voxel
units with zero active boundaries; sigma zero bypasses smoothing/boundary handling
in the primitive. The optimizer still holds its update boundary fixed. These
filters are not claimed identical to ITK's discrete Gaussian/small-variance blend.
SyN fitting uses float32 working fields and origin-free lattice arithmetic.
The physical master geometry and linear seed parameters stay float64. CPU SyN
performs its two fixed Gaussian convolutions in magnitude-scaled float32,
restoring the input field dtype; the standalone smoothing primitive uses field
precision. [AC-CPU-SMOOTHING.md](AC-CPU-SMOOTHING.md) records the earlier policy.

Inversion uses clipped/damped ITK-style residual updates and independent per-frame
reductions without per-iteration host polling. Endpoint construction uses 80
iterations per best halfway inverse. Full-lattice Jacobians retain shear,
obliquity and reversed coordinates. Best halfway fields are promoted to float64
before final inversion, composition and validation; exported world fields remain
double. This does not recover float32 fitting precision. Displaced image fitting
uses native PyTorch `grid_sample` with float32 images and normalized grids, while
support is decided before that quantization. Final original-image interpolation
retains its existing dtype/geometry policy. Exact axis-permuted/reversed Jacobians
avoid packed derivative tensors; general/sheared lattices retain the full path.

## Proof, preserved failures and limits

The current acceleration checkpoint passes **920 tests**, with **32 actual-wheel
CPU/CUDA API checks**; see [SPEED.md](SPEED.md) for default versus experimental
paths, full-recording timings and cold compilation costs. The initial milestone
passed 675 tests: 585 previous checks, 49 field checks, 40 SyN/comparison checks,
and one explicit linear-plotter rejection case. Tests share
package-wide fresh factories and independent SciPy/native ITK oracles. Multi-axis
sinusoidal compositive truth is inverted independently with reverse-order Newton
solves, never production field code. Even its best affine correspondence fails
the unchanged **0.25-pixel RMS** truth gate. Default and fixed-budget recovery,
forward/inverse CPU/CUDA points, exchange symmetry, true original-intensity
resampling, dtype/metadata, chunk/reference/failure isolation and callbacks pass.
The reference adapters separately preserve a complete nonidentity seed through
native zero-update ITK/FireANTs outputs.

Fresh pair, real-image and eight-volume comparisons pass voxreg's unchanged
**0.001-pixel maximum physical CPU/CUDA point gate** for both forward and inverse
outputs. Worst measured discrepancy is **0.000139742 pixels**. This is distinct
from sampled inverse residual, and distinct from agreement between libraries.

Initial failed checks remain under `artifacts/syn-fields/01/` and
`artifacts/syn-implementation/01/`. A coarse sheared field's reverse interpolation
residual remains approximately **0.0141 pixels after 80 inverse iterations**, even
when its forward-equation residual reaches about 1e-11. Solving one sampled
inverse equation more accurately does not make two interpolated fields exact
inverses. The initial float32-coordinate displaced sampler also failed the CPU/CUDA
gate; that failure is retained alongside the fixed-order-sampler proof.

`pytest.ini` limits default collection to `tests`, so preserved artifact/source
snapshots are not collected as duplicate test modules. The initial discovery
failure is retained; no historical snapshot was deleted to make collection pass.

Mixed-dimensional deformation, pyramids, dense restarts, multistart, line search,
large-brain/peak-memory qualification and anatomical validation remain outside
this implementation. The research live plotter supports linear models only.
Historical linear default-underfit and fixed-100 float64 motion precision failures
remain documented; these SyN checks do not erase or relax them.
