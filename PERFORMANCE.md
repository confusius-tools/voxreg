# Performance

## Objective

**Minimize runtime subject to the [benchmark accuracy gates](BENCHMARK.md#scorecard-and-protocol)
and fitting in memory:** all registration models use truth RMS ≤ 0.30 voxels.
The current priority is profiling runtime gaps, not further accuracy tuning.
Report median [min–max] seconds and peak RAM/VRAM; do not combine them into a score.
CPU/CUDA and old/new map differences are diagnostics, not benchmark gates.

Use **[the single benchmark pipeline](BENCHMARK.md)** for commands, timing
boundaries, configuration selection and outputs. Run heavy jobs serially.
Resident-array and native CLI workflows have different I/O/reference/schedule
costs and stay in separate tables.

## Current implementation

Production work already includes:

- Compact physical-to-sampling affine composition instead of dense grid GEMMs.
- Cached CPU volume coordinates and query partitioning for single-volume sampling.
- Contiguous CPU field-vector reductions for axis-aligned SyN norms.
- Float32 fitting NCC and PyTorch-only smoothing; CPU SyN regularizers use scaled
  float32 filtering, now preserving the working field dtype.
- Native PyTorch `grid_sample` for SyN image fitting and float32 working fields.
- Direct determinant entries for exact axis-aligned/permuted/reversed Jacobians,
  retaining both candidate rejection checks and the general geometry fallback.

Geometry, optimizer state and final output precision follow the
[design contract](DESIGN.md#precision-output-and-execution). These changes are not
proof of being faster than ANTs or FireANTs on every workload. The latest precision
policy has the focused five-call CUDA comparison below, not a full matrix;
volume SyN still exceeds the shared 0.30-voxel gate. Historical results retain their
original [accuracy verdicts](VALIDATION.md#benchmark-accuracy).

## Latest runtime investigation

[Controlled CUDA precision/implementation profiling](research/syn/PRECISION-PROFILE.md)
found that float32 field work explains a substantial part of the SyN gap, but is
not the only cause. An unpromoted **FP64-preserving** sampler/Jacobian combination
reduced warmed volume-pair time by about 23%; pairing the halfway updates regressed
and was rejected. Those measurements precede the new float32-field/native-float32
fitting policy and must not be presented as its benchmark results. Final SyN
inversion and export remain double. The direct Jacobian specialization is now
implemented without removing the guards. Both benchmark backends explicitly
disable TF32; the public API leaves application-wide flags alone.
These earlier diagnostics are not a newly qualified performance scorecard.

## Current matched SyN CUDA subset

RTX 5080 Laptop, eight threads, AC/performance, 100 updates, identical inputs and
matched batches. Five complete calls **including the first**, independent timing,
memory and full-workload accuracy workers. TF32 is disabled for both backends.
Sources and phase provenance were independently checked. The plane workload is
**100 frames / 99 fits**, not a single 32-frame diagnostic batch; the volume is
one `(80,150,80)` pair, not a 100-volume series.

| Workload / backend | Batch | Median [min–max], s | Peak VRAM, GiB | Worst truth RMS, vox | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 100 planes / voxreg | 32 | 1.789 [1.784–2.401] | 0.318 | 0.268294 | Pass |
| 100 planes / FireANTs | 32 | 1.505 [1.488–2.443] | 0.167 | 0.299721 | Pass |
| Volume pair / voxreg | 1 | 1.412 [1.408–4.219] | 0.643 | 0.306457 | Fail |
| Volume pair / FireANTs | 1 | 1.589 [1.544–3.634] | 0.310 | 0.358709 | Fail |

The planar voxreg call remains about 19% slower. The volume median is about 11%
lower, but **neither volume result is accuracy-qualified**. No gate, update budget,
support oracle or regularity check was relaxed. This is not anatomical evidence
or a CPU/large-recording qualification. Evidence and offline graphs:
`artifacts/benchmark-syn-float32-01/{results.json,report.html}`; verification:
`artifacts/syn-float32-production-01/verification.json`.

## Next optimization

1. Establish the current benchmark baseline; retain failures and actual batch sizes.
2. Retain accuracy verdicts, but defer accuracy tuning; raw timings of non-passing
   cases remain useful diagnostics, not accuracy-qualified performance wins.
3. Profile the largest runtime gap and choose one credible candidate.
4. Measure the whole registration, not only an isolated kernel. Keep a change only
   when the gain survives repeated timings without losing accuracy or validity.

Batch defaults are conservative policies, not measured universal optima. Compare
small, workload-specific choices with `--batch-size`; do not force equal batches
as a production policy or silently retry OOMs. Thread counts are limits, not a
promise that every operator scales. Avoid changing several policies at once.

Compilation, custom gradients and extra backend/configuration layers are not
requirements. Reuse library operations first; inspect competitor mechanisms
without transplanting restrictively licensed code.

## Historical evidence

These reports retain their original source, precision and workload boundaries;
their timings are **not** the current scorecard:

| Investigation | Record |
| --- | --- |
| CPU volume layout and sampler partitioning | [CPU-VOLUME](research/performance/CPU-VOLUME.md) |
| Coordinate/GEMM, NCC and launch-cost studies | [Performance index](research/performance/README.md) |
| SyN eager/compiled experiments and smoothing | [SPEED](research/syn/SPEED.md), [CPU smoothing](research/syn/AC-CPU-SMOOTHING.md) |
| Batch-size tradeoffs | [BATCH-SIZE](research/performance/BATCH-SIZE.md) |
| Native ANTs workflows | [ANTs report](research/registration/ANTS.md) |

The local completed coruscant reports remain in
`artifacts/benchmark-review-coruscant-complete-02/` and
`artifacts/benchmark-review-native-syn-01/`. Preliminary logs/sources are compactly
archived under `artifacts/archive/`; generated bulk arrays/images were discarded.
The earlier performance narrative is linked in the
[historical guide index](research/README.md#pre-cleanup-guides).
