# Roadmap

Goal: small, physically correct physical-space registration of VoxelData on CPU/CUDA.
Work one item at a time, then stop for review and a separate commit.

The core eager/lazy APIs, native result trees, local Rich progress and unified
benchmark commands are implemented. [README](README.md) owns supported behavior;
[VALIDATION](VALIDATION.md) owns evidence and limits. Anatomical registration
quality is not yet validated.

## Next priorities

1. **Benchmark throughput and memory scaling.** Review eager comparisons and run
   [cold-input lazy size sweeps](BENCHMARK.md#lazy-streaming-benchmark), normally
   1/4 GiB under a fixed chunk/batch/worker policy. Compare peak RAM/VRAM and
   throughput; inputs larger than physical RAM are an optional stronger check.
   Include coruscant results when available. Preserve frozen sources/settings and
   failures; do not pool incompatible hosts, revisions or protocols.
2. **Profile the SyN runtime gap to FireANTs.** Use the baseline to choose one
   measured bottleneck: optimizer work, batching, sampling or graph overhead.
   Preserve the [0.30-voxel registration gate](research/syn/REFERENCE.md), numerical
   safety and complete workloads. Report non-passing cases rather than retuning
   thresholds or shortening work to manufacture speedups. See [PERFORMANCE](PERFORMANCE.md).
3. **Broaden real-data qualification.** Evaluate pair, motion and atlas workflows
   on native grids with explicit quality limits. Qualify intensity scaling and
   winsorization, MI bandwidth/convergence/runtime, and pyramid capture range and
   whole-call performance. Establish useful workflow settings only after these
   checks; anatomical quality is not implied by synthetic accuracy.

## Deferred or evidence-driven

- **Complex IQ:** define a real fitting representation and phase-preserving
  resampling of original complex data. Spatial interpolation is not temporal
  phase correction; lazy chunking alone does not provide complex support.
- **Deployment scaling:** qualify multi-host/GPU performance and resource bounds.
  Distributed execution already works; distributed Rich progress is optional
  future work, with Dask's own tools available today.
- **Large single volumes:** spatial-block registration/distributed fitting of one
  volume is separate from temporal streaming. Mixed-dimensional deformation
  remains out of scope.
- **Other algorithms/controls:** multistart, line search, Gaussian-weighted CC,
  MSE/custom losses, optimizer/regularization hooks, velocity integration,
  compute-dtype selection and rigid-plus-scale need demonstrated use cases and
  measured benefit. Keep current optimizers and defaults until justified.
- **Additional resampling qualification:** measure auto-device/cubic performance;
  use nearest interpolation for categorical labels. Label-aware interpolation is
  not currently planned implementation work.

Do not compete with heavy jobs during measurements. Preserve historical outputs
and minimal failure evidence; job completion is not tracked here. Research details
and experiment limitations remain in [research](research/README.md).
