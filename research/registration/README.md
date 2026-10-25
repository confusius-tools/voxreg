# Shared 3D-pair and motion-correction benchmark

**Protocol:** `standardized-v1`, implemented by the existing
`examples/benchmark_registration.py` runner with workload helpers in
`examples/benchmark_workloads.py`. No registration is delegated to ConfUSIus.
At this frozen benchmark boundary (`e263aa0`), the numerical engine was unchanged
from `cc0b4de`. Later initial-transform support is qualified separately in
[VALIDATION.md](../../VALIDATION.md); these remain identity-start measurements.

[results.json](results.json) contains all warm samples, setup/cold times, memory
measurements, versions, source/input hashes, update counts, statuses, per-frame
scores, native ITK stop conditions and timing/memory output-equality checks.
Raw logs, arrays and per-configuration summaries are under the ignored
`artifacts/registration-ac/` directory.

## Power-state boundary

The operator reported plugging in the laptop after the initial suite. Its prior
power state was **not recorded**, so those timings are not used in this report.
They remain under `artifacts/registration-v2/`, including a frozen harness-source
snapshot and `results-uncontrolled.json`; their hash is recorded in results.json.
Do not splice their medians into the AC session or call them a controlled battery
baseline.

The **entire same suite was rerun**, using the exact same frozen inputs. All 196
worker endpoint snapshots (49 configurations × time/memory × before/after) show:

- `/sys/class/power_supply/ADP0/online = 1`;
- `/sys/firmware/acpi/platform_profile = performance`.

Power snapshots are outside timed calls. They do not continuously enforce power,
clocks or temperature. CPU/GPU clocks remain unfixed, desktop activity is not
blocked, and there is no randomized configuration-order/thermal experiment.
Observed differences between sessions are not a causal measurement of AC power.

Hardware: Core Ultra 9 275HX, 24 logical CPUs, approximately 31 GiB RAM,
RTX 5080 Laptop GPU, Linux. Python 3.14, NumPy 2.5.4, xarray 2026.9.0,
ConfUSIus 0.8.0, PyTorch 2.14.1+cu130, ITK 5.4.7, SimpleITK 2.5.6,
FireANTs 1.5.1 at the repository's pinned archive. FireANTs fused ops are **off**.
CPU controls use eight internal threads, with BLAS/OMP environment defaults set
to one; Torch/ITK explicitly select eight. GPU cases use one host Torch thread.

## What is consistent

Two workloads use one orchestration, measurement and reporting path:

- **Pair:** one spatial moving/fixed 3D pair, independent shapes and physical grids.
- **Series:** the existing 120-frame rat75 recording, registered independently to
  frame zero. All backends copy that reference exactly; aggregate quality excludes it.

Each configuration gets a fresh interpreter for timing and another for memory,
run sequentially. Timing includes one cold call followed by **three warm calls**.
Memory uses one cold call and a separate warm call. The reported time is the
median of the three unmonitored warm end-to-end calls.

The warm boundary starts with resident host inputs and includes:

1. Fitting preprocessing and image/geometry construction needed by registration.
2. Transfers, optimization and stopping checks.
3. Final linear resampling of original intensities onto the fixed grid.
4. Registered images, transforms, validity and diagnostics returned to the CPU.

Loading, scoring, compression and progress rendering are outside that boundary.
Fixed fitting images are **not cached across standardized calls** for any backend.
Voxreg's original-input VoxelData metadata containers are built during setup;
its fitting tensors, derived sampling geometry and validation are still timed.
ITK/FireANTs working image containers are built inside calls from the preprocessed
arrays. This is a declared host-input API boundary, not identical wrapper work.
Setup and cold-call times are separate; they are not complete interpreter-launch
latencies. CUDA timings are explicitly synchronized.

All source inputs/output arrays are deliberately **float32** for this comparison.
This is not a change to voxreg's production dtype contract. Working precision is
not forced equal:

| Backend | Numerical policy | Optimizer / initialization |
|---|---|---|
| ITK | Common float32 fitting samples converted to double images; double metric, transform and interpolation; float32 final output | Physical-shift scales, GD rate 0.1, return best; centered Euler3D / Euler2D |
| voxreg | Float32 fitting/interpolation; float64 geometry, state, NCC and matrices | RMS physical lever arms, Adam rate 0.1; centered rotation vector / constrained plane angle |
| FireANTs | Native float32 images, geometry, state, NCC and interpolation | Adam rate 0.01, 3D quaternion / 2D angle, no scaling; centered rotation |

Every backend starts at **identity world motion**, uses global squared NCC and one
resolution. ITK uses its native correlation metric. FireANTs uses a guarded native
float32 custom global NCC with a warped ones channel for support. Its support
mask is approximate near interpolation boundaries; external scoring does not use
that internal mask. Voxreg retains its minimum 90% overlap guard and best-valid
transform policy. These are not equal optimizers or equal-quality guarantees.

Final center-support masks/fill are explicit. Output outside support uses each
original moving image's minimum. The independent scorer uses SciPy float64 linear
interpolation on the *same float32 fitting samples*, intersecting valid pixel-center
support across every returned transform, identity and known truth when available.
Physical truth error is evaluated over the entire fixed grid, not just metric
support: exact affine RMS from grid mean/variance and maximum displacement at box
corners, both divided by minimum active-axis fixed spacing.

For fixed-versus-convergence comparisons, `joint_policy_quality` in results.json
rescores both policies on **one joint support**, rather than subtracting scores
computed on different masks. Individual run scores remain available as well.

## Inputs and scope

### Analytic 64³ and 128³ pairs

`examples/prepare_registration_benchmark.py` paints asymmetric continuous Gaussian
mixtures on each grid independently. The moving image is not generated with a
registration sampler. Both resolutions cover the same physical extent, 8×10×7 mm.
The declared physical rotation vector is `[0.015, -0.02, 0.01]` radians in world
z/y/x, with center displacement `[0.1, -0.08, 0.12]` mm.

Each resolution has two cases:

- Same fixed/moving grid, with oblique anisotropic geometry.
- Independent orientations/spacings and moving arrays of 80³ / 144³ respectively.

### ConfUSIus example-data control

The same-subject registration example uses **two single slices**, not two 3D
volumes. The Allen-template example is **slice-to-volume with manual initialization
and affine fitting**, also not a direct 3D rigid control. Neither was relabeled as
3D-to-3D registration.

Instead, the genuine cached **Pepe, Mariani 2026 fUSI template** from the latter
example supplies a native `(81, 91, 116)` volume. Its NIfTI qform is loaded through
ConfUSIus. The source contains negative values, so fitting uses `none`, not dB.
It is intentionally converted from float64 to the common float32 benchmark input.

A known centered rigid pose is applied to the moving grid's physical affine,
leaving the voxel intensities unchanged. This tests real 3D intensity structure
with an exactly known pose and no motion-generation interpolation. It is **not
an independent-session pair or biological truth**.

Source: Pepe et al., *Structural and dynamic embedding of the mouse functional
connectome revealed by functional ultrasound imaging (fUSI)*,
[doi:10.64898/2026.02.05.704055](https://doi.org/10.64898/2026.02.05.704055),
CC BY 4.0. The source NIfTI hash, original dtype, affine, preparation hash and
warning are stored in the frozen input provenance and results.json.

### Motion correction

The existing neutral float32 artifact is retained: rat75 / 20220523 / slice32,
frames 220:340, shape `(120, 1, 112, 128)`, original time labels and physical grid.
Its SHA-256 is recorded; this is not a newly regenerated window. All series cases
use dB fitting. There is no known biological motion truth for this recording.

## Fixed-budget 3D results

All moving inputs perform **100 updates**; every backend passes the unchanged
**RMS truth error < 0.25 minimum-spacing pixels** screen on all five 3D controls.
A fixed budget is not a convergence claim or equal metric-evaluation count.

Median warm seconds; CPU columns use eight threads, GPU pair batch size is one:

| Pair | ITK CPU | voxreg CPU | voxreg CUDA | FireANTs CPU | FireANTs CUDA |
|---|---:|---:|---:|---:|---:|
| 64³ same grid | 1.264 | 2.060 | 0.300 | 1.685 | 0.265 |
| 64³ independent grids | 1.336 | 2.144 | 0.269 | 1.623 | 0.259 |
| 128³ same grid | 11.192 | 14.803 | 0.958 | 11.641 | 1.426 |
| 128³ independent grids | 11.463 | 14.687 | 0.960 | 11.837 | 1.295 |
| Template known pose | 3.811 | 5.800 | 0.376 | 4.716 | 0.369 |

RMS truth error in minimum-spacing pixels:

| Pair | ITK | voxreg CPU / CUDA | FireANTs CPU / CUDA |
|---|---:|---:|---:|
| 64³ same | 0.037534 | 0.005595 / 0.005595 | 0.007927 / 0.007927 |
| 64³ independent | 0.040940 | 0.007442 / 0.007442 | 0.008502 / 0.008502 |
| 128³ same | 0.076204 | 0.014594 / 0.014594 | 0.014825 / 0.014826 |
| 128³ independent | 0.077450 | 0.014625 / 0.014625 | 0.015272 / 0.015271 |
| Template known pose | 0.000024 | 0.003930 / 0.003930 | 0.003966 / 0.003961 |

For 128³ independent grids, common-support NCC² is approximately 0.99998848 ITK,
0.99999645 voxreg and 0.99999648 FireANTs, from 0.96114796 at identity. Passing
one RMS screen does not make the estimates equally accurate. The template case
also shows that ITK can be closer to its known transform than either Adam solver.

Timings are not universal speed ratios. For example, 128³ independent-grid warm
ranges are 11.429–11.472 s ITK, 14.669–14.797 s voxreg CPU,
0.960–0.961 s voxreg CUDA, 11.825–11.933 s FireANTs CPU and
**0.820–1.296 s FireANTs CUDA**. The last range overlaps voxreg despite its slower
median. Unfixed clocks/thermals and only three repetitions limit conclusions.

### Memory: 128³ independent-grid control

| Configuration | Sampled tree peak PSS, MiB | CUDA peak allocated, MiB | Sampled GPU process peak, MiB |
|---|---:|---:|---:|
| ITK CPU8 | 1600 | — | — |
| voxreg CPU8 | 1153 | — | — |
| voxreg CUDA | 1935 | 283.4 | 562 |
| FireANTs CPU8 | 1220 | — | — |
| FireANTs CUDA | 2039 | 190.8 | 542 |

Host peaks include imports/setup/cold/warm, while exact CUDA allocator peaks are
reset after cold warmup. Process VRAM includes context/driver allocations; do not
add it to allocator figures. Loaded and warm host baselines, RSS, PSS and reserved
CUDA memory are retained in the JSON. RSS/PSS reads are non-atomic and peaks are
sampled separately, so tiny PSS/RSS ordering inconsistencies are possible.

One FireANTs CUDA template-convergence memory pass reaches **3225 MiB PSS** versus
1944 MiB in its fixed-budget pass, with the same 114.3 MiB CUDA allocator peak.
The transient host-memory cause is unproven; the observation is retained rather
than silently replaced with a lower rerun. Host sampling is approximate and may
miss brief transients; GPU process polling is approximately 250 ms, host sampling
approximately 20 ms plus query overhead.

## Convergence is not quality-equivalent

The second mode is deliberately backend-specific:

- ITK: window50 / minimum1e-6, physical-shift-scaled GD, return best.
- voxreg: independent ITK-formula window50 / minimum1e-6 and coarse whole-batch polling.
- FireANTs: native ten-value **scalar batch-loss slope** monitor, tolerance1e-6.
  Its implementation stops on a fitted loss increase above tolerance, not the
  same normalized-energy trend or a generic flat-loss test.

Selected pair results, using joint-policy support for NCC comparisons:

| Pair / backend | Updates | Warm median, s | RMS truth pixels | RMS < 0.25? |
|---|---:|---:|---:|---|
| 64³ independent, ITK | 98 | 1.293 | 0.043224 | yes |
| 64³ independent, voxreg CPU / CUDA | 91 | 1.922 / 0.282 | 0.008636 | yes |
| 64³ independent, FireANTs CPU / CUDA | 19 | 0.321 / 0.056 | 0.483444 | **no** |
| Template, ITK | 68 | 2.565 | 0.000614 | yes |
| Template, voxreg CPU / CUDA | 100 | 5.803 / 0.377 | 0.003930 | yes; iteration limit |
| Template, FireANTs CPU / CUDA | 19 | 0.915 / 0.080 | 0.540017 / 0.540020 | **no** |

For the template, joint-support NCC² is about 0.9999994 for the fixed-budget Adam
solutions but **0.9889738** for native-convergence FireANTs CUDA. Its short latency
is not an equal-quality win. No stopping threshold was retuned after seeing these
results, and no error tolerance was relaxed.

Voxreg's default threshold is still not a broad 3D calibration: the template hits
the cap, and coarse CUDA terminal polling can cost more than skipped updates save.
For the 64³ independent pair, CUDA convergence is slightly slower than fixed budget
despite fewer actual updates. A stopped optimizer and a faster call are different
claims.

## Retimed motion correction under the same protocol

Seconds per entire 120-frame call; aggregate quality excludes the reference.
NCC² below uses the intersection across both policies and all backends.

| Configuration | Fixed100 s | Convergence s | Convergence updates min/median/max | Median NCC² fixed / convergence |
|---|---:|---:|---:|---:|
| ITK, one process × eight threads | 6.011 | 3.450 | 49 / 53 / 78 | 0.96109907 / 0.96109907 |
| voxreg CPU8, batch32 | 1.723 | 1.825 | 51 / 79 / 95 | 0.96116762 / 0.96116720 |
| voxreg CUDA, batch32 | 0.822 | 0.889 | 51 / 79 / 95 | 0.96116762 / 0.96116720 |
| voxreg CUDA, batch120 | 0.215 | 0.222 | 51 / 79 / 95 | 0.96116762 / 0.96116720 |
| FireANTs CPU8, batch32 | 0.945 | 0.618 | 41 / 43 / 100 | 0.96119125 / 0.96108326 |
| FireANTs CUDA, batch32 | 0.885 | 0.602 | 41 / 43 / 100 | 0.96119119 / 0.96108320 |
| FireANTs CUDA, batch120 | 0.305 | 0.236 | 75 / 75 / 75 | 0.96119119 / 0.96117468 |

Native FireANTs convergence changes with batch composition because its monitor
uses a summed batch loss; voxreg keeps per-frame stopping independent. The
slower convergence-mode voxreg calls in this session are reported, not discarded.
Frozen-loop work, synchronization, monitor overhead and ordinary timing variation
all matter; there is no universal early-stopping speedup.

This is **not a claim against the fastest ITK motion-correction pipeline**. The
standardized protocol currently uses one ITK process with eight internal threads,
not the legacy eight-process scheduling control. Older ITK process-pool results
remain in [BENCHMARK.md](../../BENCHMARK.md), under their different protocol.

## Reproduce

Use an otherwise quiet Linux machine, keep the power/profile stable, and inspect
worker power snapshots. This script does not set clocks, plug in AC, or stop other
processes. Input preparation is outside benchmarking. All output directories
below must be new/empty; worker outputs are not silently overwritten.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Prepare the four analytic controls once; benchmark reuses exactly these bytes.
for N in 64 128; do
  uv run python examples/prepare_registration_benchmark.py \
    --size "$N" --output "artifacts/registration-inputs/synthetic-${N}-same.npz"
  uv run python examples/prepare_registration_benchmark.py \
    --size "$N" --independent-grids \
    --output "artifacts/registration-inputs/synthetic-${N}-independent.npz"
done

# Five default controls: ITK CPU8, voxreg CPU8/CUDA, FireANTs CPU8/CUDA; batch1.
uv run python examples/benchmark_registration.py \
  --workload pair --policy fixed --repeats 3 \
  --input artifacts/registration-inputs/synthetic-128-independent.npz \
  --output artifacts/registration-results/synthetic-128-independent-fixed

# Native/practical stopping: use a separate result directory.
uv run python examples/benchmark_registration.py \
  --workload pair --policy convergence --repeats 3 \
  --input artifacts/registration-inputs/synthetic-64-independent.npz \
  --output artifacts/registration-results/synthetic-64-independent-convergence

# A genuine cached template, as fetched by the ConfUSIus example. No implicit download.
TEMPLATE="$HOME/.cache/confusius/pepe-mariani-2026-template/1.0.0/pepe-mariani-2026-fusi-template.nii.gz"
uv run python examples/prepare_registration_benchmark.py \
  --template "$TEMPLATE" --output artifacts/registration-inputs/template-pose.npz
uv run python examples/benchmark_registration.py \
  --workload pair --policy fixed --input artifacts/registration-inputs/template-pose.npz \
  --output artifacts/registration-results/template-pose-fixed

# Existing frozen motion-correction input, same standardized boundary/policies.
for POLICY in fixed convergence; do
  uv run python examples/benchmark_registration.py \
    --workload series --policy "$POLICY" --intensity-scaling db --repeats 3 \
    --input artifacts/notebook/input.npz \
    --cases itk-1-t8 cpu-32-t8 cuda-32 cuda-120 \
            fireants-cpu-32-t8 fireants-cuda-32 fireants-cuda-120 \
    --output "artifacts/registration-results/series-${POLICY}"
done
```

The measured suite covers five fixed-budget pairs, convergence on 64³ independent
and template pairs, and both series policies: 49 configurations / 98 fresh
processes. It does not include convergence-enabled 128³ sweeps. Every stored final
timing output is **bitwise identical to its separate memory-pass output**, including
transforms, images, masks, updates and statuses. CPU/CUDA equality is numerical,
not claimed bitwise across devices.

For an independently prepared natural pair, freeze `moving`, `fixed`,
`moving_affine`, `fixed_affine` and optional `truth`/JSON-string `provenance` in NPZ.
Arrays must be finite float32; affines map array positions to world z/y/x in mm.
`--workload pair` accepts spatial-only arrays. The series format also accepts the
existing `values/time/k/j/i/voxel_to_world` archive and folds native crop/stride
coordinates into its position affine.

The cross-backend protocol requires orthogonal axes (obliquity and anisotropy are
supported, shear is rejected) and at least 32 samples per active axis to avoid
FireANTs's minimum-size upsampling. Its 2D controls are positive axis-aligned
coplanar slices. These benchmark restrictions do not narrow voxreg's solver API.
For 3D series, select small batches explicitly; batch32 defaults are slice-sized,
not a claim of memory safety for large volumes.

## Verification and remaining gates

At this benchmark boundary the full suite passed **229 tests**, including 18 new protocol tests: native 3D
physical recovery against analytic truth, output interpolation against SciPy,
series reference/batching, native convergence modes, joint support, exact affine
truth-error statistics, malformed inputs, no-overwrite CLI behavior and optional
power-file handling. Existing solver precision/synthetic/CUDA gates are unchanged.
Ruff/format/lock/build and wheel smoke checks pass.

Remaining: a natural independent-session 3D pair, large-motion capture range,
broader modalities/volume sizes, process-parallel ITK under the new boundary,
precision-matched experiments and controlled clock/thermal statistics. The current
results establish bounded registration/benchmark behavior, not biological accuracy
or a general replacement for ConfUSIus's broader registration workflow.
