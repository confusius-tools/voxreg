# Next steps

Goal: a small, physically correct CPU/CUDA registration engine for ConfUSIus.
Current model support is documented in the [README](README.md); numerical
contracts and unresolved accuracy limits belong in [VALIDATION.md](VALIDATION.md).

## Priorities

1. **Review the coruscant benchmark when complete.** Establish the current
   precision-policy baseline using [one pipeline](BENCHMARK.md), without competing
   heavy jobs or changing the running job's frozen sources/settings.
2. **Profile the SyN runtime gap to FireANTs.** Keep the shared
   [0.30-voxel registration gate](research/syn/REFERENCE.md) and recorded errors;
   defer further accuracy tuning while investigating runtime.
3. **Optimize one measured bottleneck.** Compare workload/device-specific
   batching or a source-guided implementation change without weakening precision
   or safety checks; label non-passing cases rather than hiding them.
   Preserve timing boundaries and report memory; see [PERFORMANCE.md](PERFORMANCE.md).
4. **Broaden real-data qualification.** Revisit pair, motion and atlas workflows
   with original images, native geometry and explicit quality limitations.

## Candidates, not enabled defaults

- **Fixed and moving masks:** respect each mask's native geometry, warp moving
  masks into the metric domain, and define mask-aware local statistics/pyramids.
  Keep user ROIs distinct from geometric support and the output validity mask.
- **Multi-resolution:** shrink factors, per-level iteration budgets and image
  smoothing sigmas **in mm**, respecting anisotropy and singleton axes. Carry
  physical transforms/fields between levels and resample original images once.
  Define per-level optimizer/convergence state; do not compare losses across levels.
- **Mutual information:** a differentiable, mask-aware estimator with explicit
  bins/ranges/kernels and bounded memory. Add a small per-frame metric interface;
  revisit NCC-specific conditioning, diagnostics and stopping calibration.
- **FireANTs-inspired controls:** configurable local-CC windows/kernels, MSE/custom
  losses, optimizer choice/hyperparameters, SyN gradient/warp smoothing and
  regularization hooks. Follow ITK's unit distinction: image-pyramid smoothing
  sigmas in mm, deformation regularization in voxel units. Keep existing SyN
  sigmas/defaults; ITK's field parameters are variances, not interchangeable values.
  Keep current optimizers for now. Velocity-based deformation/integration,
  compute-dtype selection and rigid-plus-scale are further options, not
  requirements for parity.
  Reimplement mechanisms independently; do not transplant licensed vendor code.
- **Multistart:** only with a selection rule that rejects tiny-overlap or
  anatomically wrong metric optima, especially for slice placement.
- **Line search:** investigate measured overshoot and extra evaluation cost before
  adding another optimizer policy.
- **Automatic staging:** manual rigid→affine and affine→SyN initialization already
  work. Reuse full returned matrices and the original images; do not resample an
  intermediate image or invent a separate transform chain.

These are candidates, not implemented features or a commitment to expose every
vendor option. Streaming/Dask execution and mixed-dimensional deformation also
remain outside the implementation. Add scope for a demonstrated need, not as a
prerequisite to benchmarking or fixing existing accuracy failures.

Keep source/settings and minimal failure evidence before cleanup. Never retune a
truth threshold or shorten work merely to manufacture a speedup. Historical
experiments and their original limitations remain in [research](research/README.md).
