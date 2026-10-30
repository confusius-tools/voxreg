# SyN acceleration investigation

## Scope and status

**Latest checkpoint:** [AC-qualified CPU smoothing](AC-CPU-SMOOTHING.md), **920
passing tests**, 85 real eight-thread registration/output checks and 32 actual-wheel
CPU/CUDA checks. A fresh complete ITK–FireANTs–voxreg comparison records CPU times
of **0.431/2.956/8.133 s** for 2D/3D/batch8, with unchanged truth and point gates.
Earlier measurements below remain historical checkpoints, not current mixed rows.

The target is FireANTs-class recording throughput **without relaxing** the
0.25-native-pixel RMS truth gate or the 0.001-pixel maximum CPU/CUDA point gate.
Identical scores are not an anatomical oracle. Native optimizer, arithmetic and
diagnostic differences remain explicit; see [LANDEMARD.md](LANDEMARD.md).

The default/eager checkout has these qualified changes:

1. Replace per-voxel tiny double-precision batched GEMMs with explicit component
   arithmetic. The measured baseline spent 76.79% of device time in `aten::bmm`.
   General CPU maps now use one GEMM per image, explicitly restoring contiguous
   component-last output; sparse maps retain exact component arithmetic.
2. Prepare complete seed queries and displacement-to-sampling matrices **once per
   batch**, rather than rebuilding constant geometry every optimizer update.
   Images are still sampled from the originals, not from prewarped intermediates.
3. Specialize exactly sparse/permuted-diagonal field lattices. Detection uses
   exact zeros, not a tolerance; general oblique/sheared mappings are retained.
4. On CPU, use native double interpolation of float32 fitting images, returning
   float32 samples with the same right-sided knot convention. Float64 final
   interpolation and nearest-neighbor tie rules keep their original path.
5. On CPU without an autograd tape, use SciPy's compiled separable float64
   Gaussian filter rather than grouped convolution. Differentiable field
   smoothing and compiled tensor paths retain the original Torch implementation.
6. Evaluate 2×2/3×3 Jacobian determinants directly in float64, avoiding packed
   per-voxel LU workspaces.
7. At SyN finalization on CPU, use a shorter inverse solve only when a conservative
   contraction bound and residual check certify it. The public field primitive's
   iteration, initial-seed, resume and autograd behavior remain unchanged.
8. Reverse dense coordinate components with `flip`, not advanced indexing.
   This removes generic index-scatter backward work while preserving component
   values and gradients; planar queries first exclude the normal component.
9. Enforce CPU Gaussian boundaries in its fresh output buffer, avoiding a clone.
   On CUDA **2D only**, filter independent components as batch entries; 3D keeps
   grouped filtering because channel-as-batch was substantially slower there.

10. On multithreaded CPU 3D SyN, use magnitude-scaled float32 convolutions for
    the fixed update/total Gaussian filters, then restore float64 fields. Planar,
    serial, differentiable, compiled and CUDA paths keep their prior smoothing.
    Kernel extent, padding, fixed boundaries and public field primitives are unchanged.

Coordinates, stored displacement fields, composition, inversion, Jacobians and CC
statistics remain float64; the scoped CPU convolutions are the explicit exception.
Fitting image samples remain float32; float64-image final interpolation stays genuine
float64. Fitting budgets, convergence, two-sided inverse diagnostics, regularity checks,
callbacks and optional-output behavior have not been removed to win a timing.

Compilation, CUDA native-double fitting interpolation, CUDA float32 gradient
filtering and squared-norm inverse updates are **research-only monkeypatches**.
There is no new compiler dependency or public compiler switch.

## Measurements

### Diagnostic, not an end-to-end claim

Actual first 32 ref20 frames, saved complete affine seeds, 20 updates, native
resolution, CUDA, transforms off; synchronized unprofiled calls:

| Default-path stage | Seconds |
|---|---:|
| Original | 0.815926 |
| Tiny-matrix arithmetic | 0.287092 |
| Cached seed queries | 0.189084 |
| Exact sparse field mappings | 0.171306 |

This is a 4.76× improvement **for this diagnostic**. In particular, its fixed
inverse work has a different weight than in a 100-update whole-recording run.
Do not extrapolate this ratio to the full recording.

### Matched 1,024-frame recordings

Original inputs, shared rigid/affine seeds, 100 updates, automatic voxreg batch
32, FireANTs public-call batch 256. One warm measurement per arm, including
original-power output interpolation but excluding seed fitting, QC and export:

| Experiment | voxreg seconds | FireANTs seconds | Status |
|---|---:|---:|---|
| Native-double fitting sampler | 22.171 | 9.000 | Research, not promoted |
| Compiled loss only | 15.129 | 8.945 | Research |
| Compiled loss/update/inverse step | 14.721 | 8.950 | Research |
| CUDA-graph backend, owned outputs | 22.932 | 8.995 | Research, no speed win |
| Cached geometry + native-grid compiler lowering | 13.344 | 8.947 | Research |
| Default eager, all three qualified changes | 21.207 | 8.973 | Qualified checkout |
| Scaled fitting-gradient filter + FP64 rsqrt inverse | 12.779 | 9.159 | Research; 85 focused tests pass |

The latest completed default-prefix comparison is still **2.36× slower** than
FireANTs; the native-grid compiled experiment is **1.49× slower**. Neither is a
claim of parity. Pearson correlation is reported in the raw JSON but is not
used as a substitute for physical truth.

### Whole recording

Historical complete 11,310-frame baseline: voxreg **758.861 s**, FireANTs
**130.172 s**, a 5.83× gap. Peak voxreg allocated memory was only 373.8 MiB.

Fresh complete recording, `artifacts/syn-speed/full-eager-01/`: default voxreg
**234.897 s** versus FireANTs **102.506 s**. Voxreg is **3.23× faster than its
historical self**, but still **2.29× slower than the concurrently measured
FireANTs arm**. The FireANTs timing also changed between runs; compare the
matched arms, not the historical FireANTs number. Automatic voxreg batch is 32,
peak allocated memory 368.08 MiB, and 11,309 fitted frames retain the 100-update
budget. Warmups were 2.828/1.311 s and are separate. This is one full measurement,
not a repeated-run confidence interval.

## CPU milestone

Single CPU thread, 120 updates, two warm repeats, shared independent fixtures:

| Case | Before CPU changes | Native CPU kernels | Matched FireANTs CPU |
|---|---:|---:|---:|
| 2D medium | 0.413 s | 0.377 s | 0.202 s |
| 3D medium | 7.707 s | 6.153 s | 4.281 s |
| 3D batch eight | 43.607 s | 34.745 s | 19.615 s |

Those first CPU changes improve these controls by roughly 9–20%, **not yet
FireANTs parity**. With direct Jacobians and the CPU-only SyN inverse shortcut,
actual default-path times are **0.326 / 5.797 / 31.319 s**; matched FireANTs is
**0.206 / 4.449 / 19.398 s**. Relative to the original CPU baseline this is about
21%, 25% and 28% less time, respectively. Results are retained in
`contract-core-cpu-01/`. Restoring contiguous one-GEMM-per-image CPU maps further
reduces those default-path times to **0.308 / 5.282 / 27.651 s**, versus matched
FireANTs **0.214 / 4.327 / 19.556 s** (`cpu-map-core-cpu-01/`). That is about
**25%, 31% and 37% less CPU time** than the original CPU baseline, without changing
the fitting budget. All truth gates pass. Medium 3D RMS changes from 0.1001063 to
0.1001070 pixels; native ITK's corresponding RMS is 0.0797510 pixels. That is not
identical accuracy, but the optimization has negligible measured effect and
retains the 0.25-pixel RMS gate. CPU/CUDA point gates also remain unchanged.

The baseline's expensive eight-frame ITK timing was stopped by the investigator
budget after the whole baseline process exceeded eleven minutes; that arm is
incomplete. Completed ITK 2D/3D controls and all completed backend rows remain in
`artifacts/syn-speed/cpu-baseline-01/`. Native-kernel prototype results are in
`cpu-fast-01/`; CPU profiles are `cpu-profile-fast-{01,2d-01}/`.

The first layout experiment moved independent field channels into the batch
axis. It helped the measured CUDA 2D filter but was up to about **60× slower** for
CUDA 3D and inconsistent on CPU. It is not a universal layout replacement.

## Follow-up experiments: measure the whole call

Further research keeps all fitting state, field arithmetic and statistics in
float64; it does not rely on the optional float32 gradient-filter experiment.

- Combining Inductor fusion with CUDA-graph replay passed all eight physical
  point controls, but took **14.188 s versus FireANTs 8.923 s** on the same
  1,024-frame prefix. Lower allocation (234 MB) did not make it faster than the
  roughly 13.3-second ordinary Inductor experiment. It is not promoted.
- One CPU GEMM per image made isolated geometry mapping **4–7× faster**, but
  returning strided coordinates made the whole CPU call slower. Restoring the
  original component-last contiguous layout brought medium 2D to **0.346 s**
  and medium 3D to **6.110 s**. This is not a blanket GEMM win.
- Direct 2×2/3×3 double Jacobian algebra and a contraction-bounded inverse
  prototype, combined with the contiguous CPU mapper, passed **187 eager
  controls** and **85 compiled controls**. CPU warm times were **0.303 / 5.266 /
  27.412 s** for medium 2D / medium 3D / batch eight. Matched FireANTs times were
  **0.212 / 4.317 / 19.505 s**. These are still not parity.
- That combined prototype took **12.759 s versus FireANTs 8.909 s** on the CUDA
  prefix. Every production-sized inverse batch fell back to the original
  solver: its conservative contraction bound exceeded one. The shortcut helped
  the tested CPU controls, not this recording's GPU batches.

The inverse prototype uses a voxel-space Lipschitz bound and a 1e-12-voxel
iteration-error target. It only replaces the original iterations when the bound
certifies fewer iterations; it verifies the left residual and retains the
right residual, boundaries and original fallback. This does **not** imply an
exact interpolated inverse between lattice points. Initially placing this in
all public primitive calls produced three strict resume-improvement failures at
the float64 roundoff floor (about 1e-16 pixels). Rather than loosen those tests,
the shortcut was scoped to SyN finalization; the primitive kept its original
algorithm. `contract-core-full-suite-01.log` preserves that failed attempt.

Moving square roots after the maximum norm and using separable native-double
box convolutions also passed qualification, but yielded about **12.99 s** on
the prefix versus **12.76 s** for their precursor. Neither is promoted.

The latest eager/default prefix is **20.315 s versus FireANTs 8.930 s**. No
whole-recording speedup is inferred from that prefix. Details and failures remain
under `artifacts/syn-speed/{direct-contract*,cpu-map*,inductor-graphs*}`.

## CPU smoothing precision follow-up

[CPU-SMOOTHING.md](CPU-SMOOTHING.md) isolates Torch float64, float32 gradient
filtering and float32 filtering of both gradients/states. Each candidate passes
85 unchanged registration controls, plus 51 independent filter checks. Larger
CPU workloads show promise, but small 2D calls regress; the laptop was observed
on battery and reference timings varied. No production precision/backend or cap
change was made from these exploratory timings.

## Dense coordinate reorder follow-up

The eight-thread CPU batch profile attributed 6.83% self time to 40 generic
index-scatter calls in coordinate-reorder backward. Replacing two advanced
component indices with slice/flip removes those calls; it does not change the
coordinate arithmetic or introduce compilation.

The unchanged suite passes **840 tests**, 1,098 warnings, in 124.10 seconds
(`flip-core-full-suite-01.log`). Native warm voxreg CPU/CUDA times at eight threads
are now **0.411 / 0.612 s** (2D), **3.421 / 0.992 s** (3D), and
**10.729 / 1.531 s** (eight volumes). The prior matched checkout was
0.421 / 0.625, 3.605 / 1.032 and 11.769 / 1.774 s, respectively. The batch is
about **9% faster on CPU and 14% on CUDA** in this follow-up; all truth and
CPU/CUDA point gates pass. Raw results are in `reorder-eager-results.json`.
Reference-library rows from the preceding three-way comparison were not rerun
for this isolated change.

## Smoothing buffer/layout follow-up

The next small checkpoint passes **847 tests**, 1,098 warnings, in 129.58 seconds
(`smoothing-core-full-suite-01.log`). CPU Gaussian outputs remain independent of
the input; CPU/GPU gradient, SciPy boundary and physical-point gates pass.
Native warm voxreg CPU/CUDA times are **0.412 / 0.613 s**, **3.354 / 0.991 s**,
and **10.582 / 1.531 s** for 2D / 3D / batch eight at eight CPU threads
(`layout-eager-results.json`). Compared with the original latest three-way
checkout, the cumulative dense-reorder/buffer changes save about **10% CPU and
14% CUDA time** on the batch.

For the larger recording grid, a baseline/layout/baseline sequence measured
**24.056 / 23.474 / 24.119 s** on the 1,024-frame prefix (FireANTs
10.419 / 10.456 / 10.475 s). The new default measured **23.550 / 10.501 s**
voxreg/FireANTs (`layout-prefix-results.json`). These are small gains, not parity;
absolute timings vary with the machine's thermal/power state.

A scoped SciPy thread-pool candidate passed seven exact independent controls and
sped up some isolated 3D filters, but did not establish a convincing whole-call
win (3.754 s single 3D / 11.030 s batch). It remains research-only; the library
does not create a global thread pool or change the user's Torch thread count.

## Bounded Newton inverse candidate: qualified, not a compiler default

A separate research solver converts each halfway field to float64 voxel
coordinates once, evaluates the exact bilinear/trilinear field derivatives, and
runs twelve bounded Newton updates. It only accepts a finite inverse with a
left residual at most 1e-10 pixels; otherwise it calls the original solver. The
right residual, endpoint Jacobians and original-image interpolation remain.
CPU and differentiable paths retain their existing algorithms.

Both eager and compiled candidate qualification passed **89 controls**, including
independent two-sided SciPy composition and unchanged CPU/CUDA point gates.

Native CUDA controls, 120 updates, two warm repeats:

| Case | Compiled voxreg warm | FireANTs warm | Voxreg cold registration |
|---|---:|---:|---:|
| 2D medium | 0.172 s | 0.342 s | 20.979 s |
| 3D medium | 0.207 s | 0.420 s | 41.368 s |
| 3D batch eight | 0.515 s | 0.572 s | 52.331 s |

All truth gates pass. Medium 3D RMS is 0.1001065 pixels versus FireANTs 0.0995142
and the prior native ITK control 0.0797510. Warm wins do **not** erase the cold
cost. These cold registrations include compilation and registration; they are
not isolated compiler timers. Without compilation, this candidate was roughly
neutral on those native CUDA controls (0.551 / 0.908 / 1.490 s versus the default
0.548 / 0.895 / 1.492 s). It is not promoted merely from warm compiled timings.

On the **complete 11,310-frame** recording, the compiled candidate took
**127.118 s versus FireANTs 102.979 s**. It remains **23.4% slower**, not parity.
Separate two-update warmups were **19.390 / 0.766 s**; adding them gives about
146.5 / 103.7 s before I/O, shared seed fitting and export. All **710** inverse
calls (including warmup) passed the residual check without fallback; worst left
residual was **2.43e-14 pixels**. This is a root residual, not an anatomical or
between-lattice inverse guarantee. Full exports/QC/provenance are retained in
`newton-full-01/`.

Smaller batches did not improve it: compiled prefix times at batches 8 / 16 / 32
were **17.087 / 12.165 / 10.549 s**. Eager batch 16 took 30.610 s versus batch 32's
19.314 s. No speculative cache-capacity batching heuristic was added. A separate
float64 `addcmul` fusion candidate passed 203 controls but took 20.413 s on the
prefix versus the default's 20.315 s; it also remains research-only.

CPU thread count also matters. On medium 3D, default voxreg/FireANTs took
**3.807 / 4.061 s** with four threads and **4.064 / 4.524 s** with eight. But the
eight-volume batch remained slower: **14.624 / 7.395 s** and **12.362 / 5.608 s**,
respectively. At 24 threads medium 3D was 7.766 / 17.018 s, while the batch was
25.022 / 7.775 s. All truth gates pass; there is no blanket CPU parity claim and
no library-global thread-setting change.

## Why low VRAM allocation is not low GPU utilization

Allocated memory measures the live tensor working set, not useful arithmetic,
SM occupancy, memory-bandwidth utilization, or launch efficiency. Tiny matrix
kernels, repeated geometry construction, many short launches, reductions and
slow double-precision arithmetic can all consume time with little live memory.

The post-mapping 1,024-frame batch sweep actually became **slower** as memory
usage grew:

| Batch | Median seconds, two repeats | Peak allocated MiB |
|---|---:|---:|
| 32 | 22.382 | 341.97 |
| 64 | 25.542 | 649.01 |
| 128 | 36.040 | 1262.19 |
| 256 | 43.457 | 2489.28 |

This sweep is its own protocol, not the seeded comparison table above. It
supports keeping the default performance cap rather than filling VRAM. Native
3D batch 256 also exhausted the available 16 GB in the earlier sweep. No runtime
batch probes or silent OOM retries were added.

## What compilation means here

Ordinary eager PyTorch dispatches the tensor operations individually.
`torch.compile` captures selected tensor functions; Inductor generates
specialized kernels, often using Triton, and can fuse pointwise operations to
avoid launches and intermediate memory traffic. This does not compile the
NIfTI data or alter the registration model intentionally.

CUDA graphs are different: they capture/replay GPU launches with stable buffer
addresses. The `cudagraphs` backend was tested separately from Inductor fusion.

Compilation is not free. Recorded cold two-update registrations were 14.49 s
for compiled leaves, 36.84 s for phase compilation, and 34.33 s for the
native-grid compiler experiment. These are cold-registration measurements for
those cases, not universal compiler startup costs. Cache hits, shapes, layout,
device and compiler versions matter. Warmup is now separately timed in the
recording harness; imports are still outside that timing.

Fusion can also change floating-point operation order. Research Inductor runs
use `emulate_precision_casts=True`; that setting does **not** replace numerical
qualification. Native PyTorch field interpolation was retained in one custom
research decomposition table because automatically generated interpolation was
not invariably faster. That backend uses private compiler APIs and is not a
production compatibility promise.

### Distinct failures, not one diagnosis

- **Parallel worker pool exited:** later submissions raised `Attempting to use
  a closed pool`. Serial compilation (`compile_threads=1`) worked. The worker
  crash's root cause remains unknown. This does **not** establish VRAM or host
  OOM; no confirming OOM event was found in the checked kernel-log interval.
- **Dynamic scalar validation:** `dynamic=True` fullgraph tracing could not
  handle scalar `isfinite`; a minimal reproduction worked with `False`/`None`.
  The research update phase uses `dynamic=None`, not deleted validation.
- **Entire inverse unrolled:** capturing all 80 iterations exceeded a 120-second
  experiment budget and was stopped. That is a bounded experiment termination,
  not a measured final compilation latency. Compiling one inverse update avoids
  the giant graph.
- **Compiled 3D invalid pointer:** initial qualification had 10 failures/63
  passes and CUDA driver pointer/allocation errors. Moving constant geometry
  outside the graph and using the native-grid experimental backend subsequently
  passed all four then-current CPU/CUDA point cases. This localizes a workable
  graph boundary, not a proven upstream root cause.
- **CUDA graph ownership:** a later graph overwrote backward output storage.
  Owned output/gradient clones fixed that run, but the matched recording was
  slower than the Inductor experiment.
- **Specialization cache exhausted:** mixed ranks/batches/layouts exceeded
  Dynamo's default eight specializations (three failures, 82 passes). The
  research runner now scopes `recompile_limit=64`; physical tolerances were not
  changed. The retry passed **all 85 tests**. This additional compilation/memory
  cost is not silently imposed on public registration.

## Additional bounded experiments

`gradient_filter.py` scales a fitting gradient into a safe float32 range before
filtering, then restores its magnitude in double. CPU fitting and displacement
state smoothing stay on the original double path. Range checks cover zero and
magnitudes 1e-200 through 1e200. This is **not** a float32 displacement-state or
CC-statistics mode, and it is not enabled in the library.

`inverse_rsqrt.py` tests double squared voxel norms and reciprocal square roots
instead of a per-voxel square root followed by division. Zero-norm behavior is
handled explicitly. The inverse iteration count/damping and final residual
checks are retained. Its qualification is separate from the default-path proof.

## Verification and evidence

- Final smoothing/layout checkout: **847 passed**, 1,098 warnings, 129.58 s
  (`smoothing-core-full-suite-01.log`), including seven threaded-candidate controls.
- Contiguous CPU-GEMM checkout: **840 passed**, 1098 existing warnings, 135.56 s
  (`cpu-map-core-full-suite-01.log`), including expanded independent matrix/field
  broadcast controls and four Newton inverse research controls.
- Jacobian/CPU-SyN-inverse checkout: **828 passed**, 1098 existing warnings,
  152.31 s, CPU and CUDA, one CPU thread (`contract-core-full-suite-02.log`),
  including 40 norm/box research controls. No existing tolerance was loosened.
- Previous CPU native-kernel checkout: **780 passed**, 1098 existing warnings,
  135.20 s (`cpu-core-full-suite-01.log`); before that, 772 passed
  (`full-suite-02.log`).
- Added independent SciPy interpolation/one-sided-knot derivative controls and
  an explicit check that differentiable smoothing retains its tensor path.
- This includes independent nonlinear truth on newly added exactly sparse
  native 2D/3D grids, fixed/convergent budgets, CPU/CUDA point gates, general-grid
  preservation, gradients, final interpolation, metadata/dtype and output-off
  contracts. No gate was relaxed.
- Compiled native-grid/scaled-gradient/FP64-rsqrt experiment: **85 passed**,
  26 warnings, 192.86 s (`gradient32-rsqrt-qualification-02.log`). It remains
  research-only; 1,024-frame warmup took 26.02 s with that process/cache state.
- Scaled gradient range controls: **16 passed**, including float64 output and
  unchanged displacement-state smoothing (`gradient32-range-02.log`).
- Scoped Ruff and `git diff --check` pass. `ty check src` retains the **38**
  baseline diagnostics; it is not a clean type-check claim.
- Latest native-grid/Newton compiled qualification: **89 passed**, 23 warnings,
  222.77 s (`final-compiled-qualification-01.log`); the seven additional warnings
  are Triton's Python-AST deprecation for future Python 3.15, not numerical failures.
- Latest offline wheel/sdist rebuilt under `artifacts/syn-speed/final-package-01/`.
  All eight wheel modules match the checkout; **32 actual-wheel CPU/CUDA checks**
  cover all four models, native 2D/3D, double transforms and output-off/opt-in parity.
- Both complete experimental recording exports passed finite/float32/shape,
  exact-reference, exact-acquisition-time and 1e-8-mm lattice checks. All 11,310 QC
  rows are finite; captured library/research source hashes match. Dataset/export
  hashes are retained in `newton-full-01/validation.json`.

Raw traces/operator tables/timing JSON, source snapshots and preserved failures
are under `artifacts/syn-speed/`. The runners are `profile_syn.py`,
`qualify_compiled.py`, and `landemard.py`. The baseline full comparison and its
source hashes remain under `artifacts/syn-ref20/full-01/`.

Qualified production milestones were committed separately. No push, vendor
patches, public precision-mode knobs, GUI qualification,
or claim of anatomical equivalence/global invertibility is included.
