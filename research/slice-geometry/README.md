# Slice geometry extension

One physical-space optimizer, three metric domains; no replicated slices and no
separate slice-to-volume fitter. Original default rigid slice arithmetic is retained.

## Models and mappings

- Two slices: 2 translation / 3 rigid / 6 affine parameters. Plane correspondence
  is a known geometric constraint; fitting remains in-plane.
- Two volumes: unchanged 3 / 6 / 12 parameter models.
- Mixed slice/volume: 3 / 6 / **9** parameters. The actual supplied volume is
  sampled along the moving/tilting native slice throughout optimization. Nothing
  pre-extracts or freezes a template plane.
- Noncoplanar slice pairs need a compatible physical seed, or explicit
  `initialization="align_planes"`. This initializer aligns geometric centers and
  chooses the shortest unoriented-normal rotation. Centering alone handles parallel
  offsets, not tilt. Translation cannot align nonparallel planes. A compatible seed
  can select the opposite normal orientation, including a coplanar half-turn.
  Two-slice recording seeds must share the chosen normal orientation; independent
  slice-to-volume poses are not restricted this way.
- Slice-to-volume affine uses a 3×2 tangent map and three world translations.
  Its normal column is completed with a unit orthogonal, orientation-preserving
  normal. Full affine normal scale/shear/reflection is unobservable on one slice:
  incompatible seeds are rejected, not silently optimized or discarded.
- A mixed fit uses the geometric native-slice pivot, also with moments centering.
  Moments still sets centroid translation from original nonnegative linear data.
  Other pre-existing moment restrictions remain unchanged.

With the volume as moving and the slice as fixed, output is the volume sampled
on the native slice. With the slice as moving and volume as fixed, fitting reverses
internally and the exported pull is inverted. The exported matrix and diagnostic
motion parameters **always describe the original fixed-world → moving-world map**.
Mixed affine diagnostics describe its completed 3×3 matrix, not twelve optimized
parameters. Geometry remains world z/y/x, mm; original moving dtype is retained.

A zero-thickness moving slice generally misses every fixed-volume voxel center.
Then output is fill with an empty validity mask, **not a failed slice-domain
metric**. A warning explains how to view the volume on the native slice instead.
`valid_fraction` describes fitting-domain support; `output_valid_fraction` describes
reverse-direction output support. Numerical on-plane tolerance is retained, not a
new slice-thickness model. Original values are not changed. Reversed NumPy views
are now made contiguous at the shared final tensor-upload boundary; this also
fixes an existing negative-stride final-resampling error without changing values.

## Qualification

Run from the repository root:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python research/slice-geometry/qualify.py --output artifacts/new-slice-check
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python research/confusius-examples/compare_atlas.py --output artifacts/new-atlas-check
```

Output destinations must not already exist. Atlas data must be cached; neither
script downloads data. The synthetic script reuses the unit fixtures rather than
maintaining a second phantom implementation.

[results.json](results.json) records the gates, native retry and historical-series
regression. The limits remain **0.25 px RMS truth** and **0.001 px maximum pointwise
CPU/CUDA disagreement**, not relaxed limits for the new modes.

- Fixed400 known SciPy-interpolated 3D anatomy, independent native slice grid,
  initialization identity: all six model/device truth checks pass. Worst RMS
  **7.55e-7 px**. All three CPU/CUDA checks pass, worst **5.23e-7 px** on the slice;
  canonical off-plane extension agreement is also recorded. This favorable
  interpolation-consistency control is not general physical accuracy evidence.
- Unit checks cover both argument directions, gradients of every identifiable
  parameter, seed/export roundtrips, native ITK point-action oracles, oblique and
  reversed/strided grids, uint16 rounding, exact on-plane output, empty output,
  chunk-independent batched seeds and invalid frames, affine collapse/normal
  restrictions, and nonparallel plane initialization. Analytic two-slice cases
  additionally refine unknown in-plane motion after plane placement on CPU/CUDA.
- The previous six analytic native controls were rerun without overwriting their
  frozen report: all 18 CPU/CUDA/ITK truth and six precision gates pass unchanged.
- Four frozen real-series controls (CPU8/b32 and CUDA1/b120, fixed100 and native
  convergence) retain bitwise-equal arrays, transforms, masks, updates and statuses.
- Full suite and packaging checks are recorded in the bounded result/report.

## Cached ConfUSIus atlas example

Exact example03 inputs: CR022 session20201007 slice02, mean of 755 frames then dB;
Pepe–Mariani qform template, original approximately ±1 mm slab. Native shapes are
slice **1×125×80**, volume slab **20×91×116**. Voxreg receives the entire slab,
not a single extracted plane. Both pipelines start from the example's manual
physical placement. ConfUSIus is unmodified, using public register_volume, affine,
LR1/max500/window50/8 threads; voxreg keeps LR0.1, affine9/max500/window50, CPU8 or
CUDA1 host threads. Its original float32 moving and float64 fixed inputs are kept.

Common native-slice support is **91.76%**. Independent SciPy double sampling gives:

| Pose | Common native-slice NCC² | Updates |
| --- | ---: | ---: |
| Manual placement | 0.048897 | — |
| ConfUSIus affine | 0.118258 | 320 |
| voxreg CPU affine9 | 0.153258 | 91, converged |
| voxreg CUDA affine9 | 0.153258 | 91, converged |

CPU/CUDA maximum pointwise disagreement is **9.76e-7 px**, below 0.001 px.
Reversing argument roles agrees within 5.7e-14 px. Slice-as-moving output has zero
volume-grid coverage, as expected; inverse-direction template-on-slice output is
useful and retains the template's float64 dtype. Overlays were inspected at
`artifacts/atlas-and-single-pair/atlas-qualified/atlas-overlays.png`.

**These scores are low.** They show an improvement on this shared slice-domain
measure, not a validated anatomical atlas placement. CF optimizes its expanded
thin-image objective; voxreg optimizes native-slice sampling of the volume. Their
internal losses and affine degrees of freedom are not identical. No landmark,
annotation or biological truth claim, and no atlas speed claim: elapsed diagnostic
fields include unlike cold setup and reverse-direction verification. Annotation
and mesh downloads, atlas-cache export and B-spline stages were not run.

## Boundaries retained

Slice lattices keep the pre-existing orthogonal-axis requirement. Other singleton
configurations remain unsupported. No masks, pyramids, finite slice-thickness
model, deformable transform or new convergence tuning was introduced. Geometry
alignment cannot create missing anatomy in genuinely different sections.

The earlier default3D stopping precision failure and original-float64 motion
fixed100 failure (**0.00196507 px**, still above the 0.001 px gate) remain recorded;
this extension neither fixes nor hides them. General natural-data 3D/affine/mixed
stopping and anatomical correctness remain unqualified.
