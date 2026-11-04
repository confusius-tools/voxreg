# Performance investigation: threads, dense grids and CUDA

## Current numerical-policy work

The policy is **minimize median runtime subject to truth RMS < 0.25 voxels and
fitting in memory**. Peak RAM/VRAM are resource constraints, not a composite score.
CPU/CUDA and old/new map differences are diagnostic only.

Fitting NCC is float32 and runtime smoothing is PyTorch-only; CPU planar/volume
SyN regularizers use scaled float32. Geometry, state, final interpolation and
batching defaults are unchanged. Full-budget SyN controls still fail truth RMS;
one-call diagnostics are not formal throughput claims. See [validation](VALIDATION.md)
and the [single benchmark command](BENCHMARK.md). Historical measurements below
retain their original policy; preliminary raw probes are now compactly archived
under `artifacts/archive/`.

## Historical investigation

At benchmark cleanup, the [consolidated five-run pipeline](BENCHMARK.md) was ready
but had not been run. Measurements below remain historical qualification evidence,
not its median/range scorecard.

**Finding:** the largest identified bottleneck is generic matrix multiplication
on dense float64 coordinate fields, including unfavorable shapes/layout and
associated copies/backpropagation. It is not xarray overhead, and reducing
precision is not necessary to remove it.

This is the historical investigation of baseline commit `83b6dd6`. The float64
scalar-affine candidate described below has **now been brought into the engine**;
see [BENCHMARK.md](BENCHMARK.md) for the current production measurement protocol,
source capture and regression gates. Historical prototype timings are retained
here rather than presented as current benchmark results.

## CPU volume follow-up

[The CPU volume report](research/performance/CPU-VOLUME.md) records the latest
precision-preserving changes: a shared float64 lattice/native coordinate layout
for batched CPU volumes, and zero-copy query partitioning for single volumes.
On full-size rigid controls with 100 updates, one first 32-volume call fell from
97.861 to 77.694 s, and one pair from 5.877 to 2.689 s. NCC precision, production
batch caps and CUDA arithmetic are unchanged. The FireANTs benchmark adapter now
uses 32 linear-model volumes after separate CPU/CUDA batch and memory checks;
historical four-volume timings retain their old scheduling label.

## SyN acceleration checkpoint

[The current SyN report](research/syn/SPEED.md) records qualified eager geometry,
layout, CPU interpolation/smoothing and inverse improvements. Default single-thread
CPU controls use 25–37% less time than the original CPU baseline. The latest
[AC-qualified multithreaded CPU 3D smoothing](research/syn/AC-CPU-SMOOTHING.md)
uses scaled float32 convolutions with double field storage and unchanged physical
checks; fresh CPU 2D/3D/batch8 times are 0.431/2.956/8.133 s. The checkpoint passes
920 tests, 85 real eight-thread registration checks and 32 actual-wheel CPU/CUDA
checks. Planar, serial, differentiable, compiled and CUDA smoothing are unchanged.

A separate bounded-Newton compiled experiment beats FireANTs on the warm native
CUDA controls, but cold registrations cost 21–52 seconds. Its complete recording
takes 127.118 versus 102.979 seconds plus separate warmups; it is not a default
compiler requirement or a universal parity claim. Eager-only Newton and fused
pointwise candidates did not earn production promotion from whole-call timings.

## CPU SyN short-vector norm layout

The axis-aligned physical-field norm now packs CPU vector components contiguously
before the float64 reduction. CUDA and general-lattice arithmetic are unchanged;
no precision, smoothing, update budget or guard was relaxed. The strided
component-first reduction consumed 60% of the planar and 34% of the volumetric
operator profiles in the bounded investigation.

Five eight-update CPU calls with 32 fitted native planes improved from
**3.189 [2.990–3.227] s** to **1.229 [1.225–1.296] s**. A separate 100-update,
32-fitted-plane export check preserves registered images exactly and physical
maps within **9.44e-16 minimum-spacing voxels**. Its one export-inclusive call
fell from 30.846 to 13.332 s; those are diagnostics, not five-run benchmark
medians. Full-budget volumetric performance has not been remeasured.

The full-budget check also exposes an **existing CPU/CUDA maximum-point gate
failure: 0.054765 voxels**, identical before and after this patch. Matching worst
truth RMS (CPU 0.264182, CUDA 0.264185 voxels) does not establish equivalence;
both still fail the unchanged 0.25-voxel truth screen. The layout change preserves
the prior CPU result, not a claim of newly passing SyN accuracy or CPU/CUDA parity.

A source-guided rigid control also tried packing the affine grid contiguously,
as in native `affine_grid`. Five 32-volume/eight-update CPU calls worsened from
7.244 [6.993–7.400] to 7.939 [7.665–8.127] s despite bitwise-equal results, so
**that candidate was rejected**. Rigid production code remains unchanged.

Evidence and retained failures: `artifacts/cpu-single-resolution-investigation-01/`
and `artifacts/cpu-norm-qualification-01/`. Full production suite: **683 passed**.
These local Core Ultra 9 measurements are not a replacement coruscant scorecard.

## Single-pair GPU follow-up

[Measured diagnosis](research/performance/SINGLE-PAIR.md): a small real pair remains
faster on CPU, with CUDA launching 209 tiny kernels per sampled fitting step.
Resident replay that removes repeated host dispatch preserves optimizer outputs
bitwise and substantially reduces device time; this is a static-graph diagnostic,
not a shipped optimization or an end-to-end speed claim. Bulk image uploads are a
small fraction of the observed warm call. Dtype and stopping policies are unchanged.

## Current dtype and eager NCC result

[DTYPE.md](research/performance/DTYPE.md) records the dtype-preserving output
contract and the small eager NCC cleanup now in production, with a fresh full
backend time/memory comparison. Float32 fitting candidates fail existing
CPU/CUDA gates, so mixed computation precision is retained. That historical package review
kept the four-module runtime and added no compiler/precision/backend framework.
Earlier CPU/NCC experiments below are historical evidence.

## CPU follow-up after the sampler rewrite

[research/performance/CPU.md](research/performance/CPU.md) profiles the remaining
CPU gap at `857e792`. Isolating NCC arithmetic dtype removes about 40–44% of
voxreg's CPU runtime; the reverse FireANTs ablation corroborates the cost. The
remaining signal is bulk grid/mask/autograd work, not primarily independent Adam
bookkeeping or xarray. These are diagnostics, not a new production precision
policy; source remains unchanged.

The subsequent [NCC follow-up](research/performance/NCC.md) tests a simpler eager
loss and compiled float64 NCC. Compilation improves warm throughput but introduces
startup/toolchain constraints; neither a public compiler mode nor automatic dtype
selection was added.

## 1. Single-thread controls

Single-thread voxreg was measured in the original benchmark but omitted from the
shortened FireANTs comparison. Fresh sequential measurements on the same complete
120-frame recording, three warm calls and separate memory passes:

| Engine | Batch | Torch threads | Median seconds |
|---|---:|---:|---:|
| voxreg | 1 | 1 | 9.484 |
| voxreg | 32 | 1 | 8.823 |
| voxreg | 32 | 8 | 3.305 |
| voxreg | 120 | 1 | 8.683 |
| FireANTs | 1 | 1 | 7.154 |
| FireANTs | 32 | 1 | 2.647 |
| FireANTs | 32 | 8 | 0.943 |

Eight threads help both engines. Single threading does not resolve voxreg's
performance gap. See `artifacts/benchmark-threads/summary.json` for full timings,
memory and shared-support quality. These workstation measurements vary between
sessions; do not mix them into the earlier table as if measured simultaneously.

## 2. What the profiler actually shows

Profiles capture **five steady optimizer iterations** after 20 waiting and two
warmup iterations, separately from timing runs. CPU profiles use 32 moving frames;
CUDA profiles use all 119 moving frames. Both engines optimize against the same
first-frame reference. FireANTs uses native float32; voxreg retains its float64
geometry/state/NCC and float32 interpolation. No optional FireANTs fused extension,
Torch compilation or CUDA Graphs are involved.

### CPU

Selected inclusive forward/backward scope times over five profiled iterations:

| Region | voxreg, 1 thread | FireANTs, 1 thread | voxreg, 8 threads | FireANTs, 8 threads |
|---|---:|---:|---:|---:|
| Sampling forward, including grid | 48.05 ms | 9.45 ms | 15.00 ms | 2.96 ms |
| NCC forward | 21.22 ms | 6.51 ms | 3.60 ms | 1.66 ms |
| Backward | 43.03 ms | 18.93 ms | 14.99 ms | 8.06 ms |

These are instrumented scope times, not throughput benchmarks. Voxreg's hot
operators include large multiplies, copies, matrix multiplications and divisions.
The bilinear image sampler itself is **not** the primary explanation: FireANTs
actually samples two channels (intensity plus validity) versus voxreg's one.

`sample_rigid` first rotates every physical point with an `einsum`, adds a dense
translation field, then multiplies the entire field by the inverse lattice.
`mapped @ plane.inverse_lattice.T` treats `(batch, height)` as matrix batch axes.
For CUDA batch 119 and shape 112 × 128 that becomes:

```text
[13328, 128, 2] @ [13328, 2, 2]       # forward and backward
[1, 238, 2] @ [1, 2, 14336]          # dense rotation
[1, 238, 14336] @ [1, 14336, 2]      # rotation-gradient reduction
```

Profiler traces also expose image-sized clones/copies around these operations.
These are poor general-purpose GEMM workloads for a two-dimensional affine map.
Voxreg then makes multiple full-image float64 passes for weighted NCC, its guards
and gradients. That is extra work versus FireANTs's float32 custom loss, but the
sampler ablation below shows that the dense-grid implementation is independently
responsible for a substantial part of the gap.

### CUDA

GPU activity is counted from trace events with category `kernel`, avoiding double
counting the CPU operator aggregates and corresponding GPU events:

| Five-iteration capture | GPU kernel time sum | Kernel count |
|---|---:|---:|
| Current voxreg | 52.98 ms | 1,061 |
| Native float32 FireANTs | 11.47 ms | 795 |
| Float64 scalar-affine prototype | 8.58 ms | 1,190 |

About **80%** of current voxreg's captured GPU kernel time is in float64 GEMM
kernels. The prototype is much faster **despite launching more kernels**: avoiding
the wrong operations matters before reducing launch count.

The per-call stage timings also rule out wrapper/I/O as the main problem: in the
matched CUDA baseline, optimization takes 1.143 s out of 1.164 s; preparation is
about 7 ms and upload below 1 ms. Xarray/validation/upload are not useful first
optimization targets here.

## 3. Isolated sampler experiments

The experiments leave Adam, per-frame guards, independent clocks, best-valid
selection, stopping rules, NCC, output resampling and world-matrix export intact.
They substitute only the sampler used by the public registration API.

The matched ablation processes **119 moving frames with an explicitly supplied
fixed reference**, from the same 120-frame recording. Three warm calls are timed,
with no profiler/memory monitor running. CPU uses batch 32; CUDA uses one full
119-frame batch. Thus these are prototype comparisons against their own local
baseline, not a replacement for the isolated-process benchmark leaderboard.

| Configuration | Current sampler | Float64 scalar-affine prototype | Speed ratio |
|---|---:|---:|---:|
| CPU, 1 thread | 8.824 s | 5.820 s | 1.5× |
| CPU, 8 threads | 3.203 s | 1.524 s | 2.1× |
| CUDA, full batch | 1.164 s | 0.232 s | 5.0× |

### The change in mathematical organization

Compose the tiny physical-to-normalized affine **once per frame and iteration**,
not by applying several transforms separately to every voxel. Cache constant
lattice/normalization factors and one-dimensional normalized coordinate axes.
Then evaluate the two grid coordinates as broadcast scalar expressions.

Writing `L` for the intrinsic physical lattice and `D` for the diagonal matrix of
half image extents:

```text
Q = D⁻¹ L⁻¹ (R - I) L D
c = D⁻¹ L⁻¹ t

grid_x = nx + Q00*nx + Q01*ny + c0
grid_y = ny + Q10*nx + Q11*ny + c1
```

Only the small matrices use matrix multiplication. There is no image-sized GEMM
or `(batch × height)` batch of tiny matrices. The displacement-from-identity form
also avoids needless identity cancellation. All geometry expressions remain
float64; the completed grid is cast to float32 for the existing image sampler,
just as in the current implementation. Obliquity/anisotropy/native-coordinate
semantics remain represented by the intrinsic lattice.

### Validation of that prototype

- **All 43 existing tests pass** with the diagnostic sampler temporarily substituted,
  including independent interpolation, gradient checks, synthetic recovery,
  oblique/strided geometry, failure/convergence behavior and CPU/CUDA tests.
- Worst real-recording mapping disagreement with the matched baseline:
  **6.69e-6 minimum-spacing pixels RMS** across frames.
- Real-recording batch sizes 1 versus 32 versus 119: worst RMS difference below
  **3e-16 pixels on CPU**, **6.95e-9 pixels on CUDA**, below the unchanged 0.001-pixel
  batch gate.
- Independent common-support median NCC² remains **0.961167622**, indistinguishable
  from baseline at displayed precision.

### Rejected shortcut: just use float32 affine grids

A standard `affine_grid` with float32 coefficients was also faster, but changed
the worst real-frame mapping by about **0.13 pixel RMS** relative to baseline.
Both standard affine-grid variants initially failed one tight low-level identity
interpolation test (42/43 passed). Keeping `affine_grid` float64 helped CPU but did
not improve CUDA. These experiments do not justify silently relaxing precision
or weakening tests: the scalar float64 formulation is the better first candidate.

## 4. What to borrow from FireANTs—and what not to copy

1. **Borrow compact affine composition and efficient normalized-grid evaluation.**
   This is the demonstrated first fix. Preserve physical float64 parameters and
   statistics; avoid making the entire dense transform a BLAS problem.
2. **Then fuse repeated elementwise work/reductions.** Once the GEMM bottleneck is
   gone, NCC, coordinate evaluation and small optimizer/status kernels are the
   next candidates for a compiled functional step. Profile that version before
   choosing `torch.compile`, CUDA Graphs or a custom kernel. None is needed to
   demonstrate the first improvement above.
3. **Keep per-frame correctness policies.** Native batch Adam/convergence is not
   automatically a substitute for independently frozen moments/clocks, overlap
   handling and best-valid snapshots. Those policies were retained in this probe.
4. **Do not copy FireANTs's per-iteration `loss.item()`.** It forces a CUDA-to-host
   synchronization for its convergence monitor. Voxreg already avoids that.
   The measured FireANTs advantage did not require a fused extension or secret
   CUDA backend; its better-organized computation mattered more here.
5. **Calibrate convergence separately.** ITK's median 31 updates versus voxreg's
   100 contributes to their comparison, but cannot explain the gap to FireANTs,
   which also takes 100 updates. Do not merely lower the iteration count to make
   the benchmark look better.

## Evidence and next action

The four harnesses are tracked in [research/performance/](research/performance/README.md),
with reproducible commands and a [compact historical snapshot](research/performance/results.json).
Raw JSON, CPU/CUDA traces, output matrices, quality and full historical test logs
remain local under ignored `artifacts/performance-investigation/`. Historical
43-test counts above predate the research CLI cleanup tests. No engine source or
dependency was changed for these probes.

The narrow sampler rewrite and fresh-process time/**memory** benchmarking are now
complete; [sampler-results.json](research/performance/sampler-results.json) records
the production results. Dense pixel-sized GEMMs are replaced by cached normalized
axes and scalar affine expressions. The baseline optimizer, precision and frame
policies remain intact. A compiled variant and broader datasets remain untested;
the unchanged real-data convergence rule still reaches its iteration limit.
