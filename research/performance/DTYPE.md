# Dtype-preserving resampling and precision qualification

This change separates source/output representation from fitting computation.
Implementation commits are `9692efe` (dtype contract) and `88b491b` (eager NCC).
The compact record is [dtype-results.json](dtype-results.json). All prior CPU/NCC
reports are historical investigations, not additional runtime backends.

## Shipped behavior

- Source arrays are never modified. Float16/32/64 and integer outputs retain the
  moving dtype. Boolean, complex and extended-precision floating inputs are rejected.
- Fitting images are float32. Geometry, independent Adam state, NCC and exported
  world matrices stay float64. No public compute-precision selector was added.
- Integer/float64 preparation normalizes in float64, then uses NCC's affine
  intensity invariance to center/scale before converting fitting copies. This
  prevents low-contrast float64/integer information disappearing at the initial
  cast. Source values remain untouched.
- Final original-power sampling happens once, in float64 for float64/integers and
  float32 for float16/32. A float64 output is not a float32 result cast back.
- Integers round to nearest-even and clip to their type range. Inputs outside
  [-2**53, 2**53] are rejected so the initial float64 conversion is exact.
- `interpolation="linear"` (default) or `"nearest"` affects final output only.
  Fitting remains linear. Nearest is appropriate for categorical labels, not
  automatically for integer intensity images.
- References are copied exactly; invalid support still receives the frame minimum.
  Native coordinates, world geometry, units and time labels are preserved.
- The NIfTI example now preserves decoded array dtype, exports integer types
  explicitly, and records fitting/output dtype separately.

## Numerical tests and rejected precision changes

**118 tests pass** without weakening tolerances. New cases cover fractional
linear/nearest output against SciPy, CPU/CUDA, float16/32/64 and signed/unsigned
integers, source immutability, exact reference identity, rounding/clipping, unsafe
integer rejection, low contrast, float64 values outside float32 range, known
physical recovery, and NIfTI dtype round trips.

The eager NCC cleanup masks centered images once and shares those arrays between
covariance and variance, following FireANTs' organization. It preserves all
meaningfulness, finite, overlap and best-valid checks. On the measured neutral
float32 recording, transforms, registered outputs and iteration counts are
**bitwise identical** to `857e792` for all four before/after configurations.

Two alternative fitting policies were investigated through the same optimizer
and public API, with final resampling kept independent:

1. Full float32 fitting (grid, parameters/moments and NCC), returning physical
   parameters to the original float64 final geometry.
2. Float32 NCC only, retaining float64 fitting geometry/state.

Both finish with **116 passed, 2 failed**: synthetic recovery passes, but the
existing CPU/CUDA mapping agreement tests fail (batch sizes 1 and 3). Full-float32 maximum pointwise discrepancies
are about 0.0048–0.0051 minimum-spacing pixels; NCC-only float32 about 0.0099,
against the unchanged 0.001-pixel maximum-pointwise gate. This does not mean
float32 is intrinsically inaccurate or unusable in another engine. These variants
are simply not qualified substitutes under this package's current contract.
Qualification stops at those failures; no further real-data speed claim or public
mode is offered for them. Production retains the validated mixed policy.

## Additional real windows

Alongside slice32 frames220:340, the checker exercises slice32 frames800:920 and
slice37 frames220:340. The neutral benchmark/checker deliberately converts samples
to float32, as in earlier backend comparisons. The saved source windows decode
as float64; separate direct API runs verify native float64 output as well.

| Window | Neutral CPU batch RMS, pixels | Neutral CUDA batch RMS, pixels |
|---|---:|---:|
| slice32, 800:920 | 1.41e-16 | 4.29e-9 |
| slice37, 220:340 | 2.11e-16 | 1.04e-8 |

These compare batch1 with32/full using a separately supplied fixed image. Both
are below the unchanged 0.001-pixel batch gate. Independent common-support median
NCC² is approximately 0.964161 and 0.959219 respectively.

Native float64 direct API calls retain float64 registered intensities and exact
reference/time labels. Batch32/full agree exactly on CPU and CUDA in both windows.
Minimum valid support stays above 97.7%, with 119 iteration-limit frames and one
reference per window. CPU/CUDA mapping RMS differences are 0.00213 and 0.00106
pixels: reported as inter-device disagreement, **not** ground-truth errors or a
claim of meeting the separate 0.001 maximum-pointwise synthetic tolerance. These
are still two slices of one animal/session, not broad biological validation.

The complete native float64 NIfTI example was also rerun against the frozen
ConfUSIus reference, including exact input equality, diagnostics, plots and
napari-readable exports under `artifacts/dtype-policy/notebook/`.

## Fresh time and memory benchmark

All configurations ran sequentially, using fresh interpreters and **separate
unmonitored timing and monitored memory passes**, with three warm calls. There
were no interrupted phases. Baseline `857e792` was selected via an archived
source `PYTHONPATH`, using the same dependencies and benchmark harness.

The benchmark is the neutral float32 120-frame input, not a performance claim
for every supported source dtype. Normal desktop activity remained present;
clocks/affinity were not fixed. Small CPU differences overlap across repetitions.
Before/after cases were rescored with ITK/FireANTs on one common support.

| voxreg configuration | Before, s | After, s |
|---|---:|---:|
| CPU batch32, 1 thread | 5.935 | 5.691 |
| CPU batch32, 8 threads | 2.458 | 2.128 |
| CUDA batch32 | 0.829 | 0.780 |
| CUDA batch120 | 0.234 | 0.219 |

Full-batch CUDA allocated/reserved peaks fall from **189.2/218 to175.2/204 MiB**;
process VRAM from **478 to464 MiB**. Current backend controls, PSS/RSS and all warm
samples are in the snapshot and [BENCHMARK.md](../../BENCHMARK.md). Common-support
median NCC² remains 0.961168 voxreg, 0.961191 FireANTs and 0.961099 ITK.

The only post-measurement runtime cleanup hoists dtype decisions out of the batch
loop; it changes no array operation, dtype, tensor or numerical result. Final
source hashes and tests are captured separately from that timing boundary.

## Package structure review

Expected use is a single caller registering one recording in bounded resident
chunks, on CPU or one CUDA device—not a concurrent registration service.

The four-module runtime remains appropriate:

- `geometry.py`: VoxelData construction/validation and lazy world coordinates.
- `sampling.py`: physical plane geometry, sampler, NCC and world matrices.
- `rigid.py`: input preparation, independent optimizer and output assembly.
- `__init__.py`: small public export surface.

No mandatory structural refactor was found. The new preprocessing/restoration
blocks are small and stay beside their owning registration pipeline. Static dtype
choices were moved out of the chunk loop. No backend registry, precision-policy
objects, compiler requirement, extra runtime dependency, metric selector or
separate integer engine was introduced. The optimizer retains its original state
and stopping logic. Research harnesses remain outside the runtime package.

**Verdict: Ship** the mixed-precision implementation. Connected callers, reference
I/O, dtype boundaries, optimizer guards and numerical regressions were checked;
no mandatory structural changes remain.

Potential future work remains measurable, not architectural: calibrate convergence
on more datasets, then revisit fitting precision if there is a demonstrated need.
Neither compiled NCC nor the mixed CPU/GPU per-axis bounds tradeoff is shipped.

## Reproduction

```bash
uv run --extra examples pytest -q
uv run python research/performance/validate_probe.py production --fit32
uv run python research/performance/validate_probe.py production --ncc32
# The latter two intentionally fail the unchanged CPU/CUDA agreement tests.

uv run python examples/benchmark_registration.py \
    --output artifacts/benchmark-dtype-policy \
    --cases itk-1 itk-1-t8 itk-8 cpu-32 cpu-32-t8 cuda-32 cuda-120 \
            fireants-cpu-32 fireants-cpu-32-t8 fireants-cuda-32 fireants-cuda-120
```

To repeat before controls, archive `857e792` as in BENCHMARK.md's earlier recipe.
Additional window inputs are prepared with `load_recording(path, slice(...))`
and saved with values, time/k/j/i and the native voxel-to-world matrix. Run
`check_elementwise.py --sampler production --input WINDOW.npz --devices cpu cuda`
for neutral float32 batch checks, or call `register_rigid` directly on the decoded
float64 data for native-dtype checks. Raw outputs/logs stay ignored under
`artifacts/dtype-policy/` and `artifacts/benchmark-dtype-{baseline,policy}/`.
