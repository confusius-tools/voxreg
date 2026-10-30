# SyN acceleration investigation

## Scope and status

The target is FireANTs-class recording throughput **without relaxing** the
0.25-native-pixel RMS truth gate or the 0.001-pixel maximum CPU/CUDA point gate.
Identical scores are not an anatomical oracle. Native optimizer, arithmetic and
diagnostic differences remain explicit; see [LANDEMARD.md](LANDEMARD.md).

The default/eager checkout has three qualified changes:

1. Replace per-voxel tiny double-precision batched GEMMs with explicit component
   arithmetic. The measured baseline spent 76.79% of device time in `aten::bmm`.
2. Prepare complete seed queries and displacement-to-sampling matrices **once per
   batch**, rather than rebuilding constant geometry every optimizer update.
   Images are still sampled from the originals, not from prewarped intermediates.
3. Specialize exactly sparse/permuted-diagonal field lattices. Detection uses
   exact zeros, not a tolerance; general oblique/sheared mappings are retained.

All coordinates, displacement states, field operations and CC statistics remain
float64. Fitting image samples remain float32; final interpolation remains genuine
float64. Iteration budgets, convergence, inverse iterations, regularity checks,
callbacks and optional-output behavior have not been removed to win a timing.

Compilation, native-double fitting interpolation, scaled float32 gradient
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

- Default checkout: **772 passed**, 1098 existing warnings, 130.76 s, CPU and
  CUDA, one CPU thread (`full-suite-02.log`).
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
- Offline wheel/sdist rebuilt under `artifacts/syn-speed/build/`.

Raw traces/operator tables/timing JSON, source snapshots and preserved failures
are under `artifacts/syn-speed/`. The runners are `profile_syn.py`,
`qualify_compiled.py`, and `landemard.py`. The baseline full comparison and its
source hashes remain under `artifacts/syn-ref20/full-01/`.

No commit/push, vendor patches, public precision-mode knobs, GUI qualification,
or claim of anatomical equivalence/global invertibility is included.
