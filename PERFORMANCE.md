# Performance

## Objective

**Minimize runtime subject to truth RMS < 0.25 voxels and fitting in memory.**
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
  float32 filtering with float64 field storage.

Geometry, optimizer state and final output precision follow the
[design contract](DESIGN.md#precision-output-and-execution). These changes are not
proof of being faster than ANTs or FireANTs on every workload. The latest precision
policy needs a reviewed five-run scorecard; full-budget SyN controls still fail
[truth accuracy](VALIDATION.md#benchmark-accuracy).

## Next optimization

1. Establish the current benchmark baseline; retain failures and actual batch sizes.
2. Fix failing truth accuracy before calling a configuration a performance winner.
3. For a passing case, profile its largest gap and choose one credible candidate.
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
