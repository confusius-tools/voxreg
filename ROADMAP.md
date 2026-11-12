# Roadmap

Goal: small, physically correct physical-space registration of VoxelData on CPU/CUDA.
Work one item at a time, then stop for review and a separate commit.

The core eager/lazy APIs, native result trees, local Rich progress and unified
benchmark commands are implemented. [README](README.md) owns supported behavior;
[VALIDATION](VALIDATION.md) owns evidence and limits. Anatomical registration
quality is not yet validated.

The initial coruscant eager comparisons and fixed-policy cold-input 1/4 GiB
rigid sweeps are reviewed. The input-sized serialized-graph bug is fixed;
[PERFORMANCE](PERFORMANCE.md#coruscant-baseline) records results and remaining limits.

## Next priorities

1. **Qualify 3D SyN accuracy under fixed work.** Earlier 100-update volume-pair
   and recording cases failed the 0.30-voxel gate: voxreg was about 0.306 on
   CPU/CUDA; FireANTs also failed, while ANTsPy passed. Benchmarks now request
   200 fixed updates on all backends; broader CPU/CUDA/recording qualification
   is still needed. The CPU pair pilot passes for voxreg/ANTsPy but FireANTs is
   still at 0.303709, above 0.30. [CPU diagnostics](research/syn/ACCURACY-20261112.md)
   show continued improvement at 300 updates, but larger learning rates are not a
   robust fix. Retain the current ITK-like scalar normalization; isolate metric,
   smoothing and inverse-maintenance differences before changing the algorithm.
   Do not relax gates or retune only to the benchmark.
2. **Attribute the SyN runtime gap before optimizing.** Local profiling and small
   prototypes are complete, but no worthwhile safe optimization was established.
   Use diagnostic ablations to separate additional validation/finalization work
   from avoidable overhead; the profiled spans do not causally explain the gap.
   Preserve numerical safety, precision and complete workloads. See
   [PERFORMANCE](PERFORMANCE.md) and the [profile](research/syn/PROFILE-20261111.md).
3. **Broaden real-data qualification.** Evaluate pair, motion and atlas workflows
   on native grids with explicit quality limits. Qualify intensity scaling and
   winsorization, MI bandwidth/convergence/runtime, and pyramid capture range and
   whole-call performance. Establish useful workflow settings only after these
   checks; anatomical quality is not implied by synthetic accuracy.

## Deferred or evidence-driven

- **Extended memory qualification:** larger-than-host-RAM input is an optional
  stronger check. Current 1/4 GiB evidence covers one GPU, rigid fitting and known-map
  resampling, not arbitrary recordings, lazy SyN or a universal memory bound.
  Rerun relevant benchmarks after meaningful changes; preserve frozen settings
  and failures without pooling incompatible hosts, revisions or protocols.
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
