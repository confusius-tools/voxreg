# Performance

## Objective

Minimize runtime subject to the [benchmark accuracy gates](BENCHMARK.md#scorecard-and-protocol)
and fitting in memory. Registration requires worst fitted-frame truth RMS
≤ 0.30 minimum-spacing voxels; resampling requires relative RMSE < 1e-5.
The current priority is runtime profiling, not further accuracy tuning.

Use the Python-API benchmarks in [BENCHMARK.md](BENCHMARK.md). Run registration
and resampling serially in new output directories. Report median [min–max],
first call, peak RAM/VRAM and accuracy separately. Do not pool hosts, revisions
or operations. Failed quality gates remain failures, not qualified speed wins.

## Current implementation

- Compact physical sampling maps, cached CPU queries and single-volume partitioning.
- Native PyTorch `grid_sample` for SyN fitting; float32 images, NCC and working fields.
- Float64 physical master geometry, linear pose parameters, final SyN inversion,
  validation and world-field export.
- Direct Jacobian determinants for exact axis-aligned/permuted/reversed grids;
  general oblique/sheared fallback and both candidate rejection checks retained.
- Original-data final resampling and output dtype preservation.

See the [precision contract](DESIGN.md#precision-output-and-execution).

## Measurement plan

Rerun both complete 45-case matrices to establish the scorecard for the current
ANTsPy-only comparison policy. The ANTs backend is always its Python API, not CLI
executables. ANTsPy processes one frame at a time and retains its own optimizer
and convergence; settings are approximate matches, not algorithmic equivalence.

Choose the next optimization from the largest measured whole-call gap. Keep it
only when repeated timings improve without weakening geometry, support, output
precision or safety checks. No custom compiler/kernel, silent OOM retry or
accuracy tuning is required to produce a favorable result.
