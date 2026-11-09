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

## Implemented, opt-in

- **Multi-resolution:** explicit coarse-to-fine shrink factors, Gaussian image
  smoothing sigmas in mm and per-level iteration budgets for every model.
  Physical transforms/halfway fields carry between levels with fresh optimizer
  and convergence state; original images are resampled once. See the
  [API and diagnostic conventions](README.md#multi-resolution-fitting).
  Qualifying useful schedules, capture range and whole-call performance on real
  workflows remains future work; no multi-resolution default preset is enabled.

## Candidates, not enabled defaults

- **Fixed and moving masks:** respect each mask's native geometry, warp moving
  masks into the metric domain, and define mask-aware local statistics/pyramids.
  Keep user ROIs distinct from geometric support and the output validity mask.
- **Mutual information:** a differentiable, mask-aware estimator with explicit
  bins/ranges/kernels and bounded memory. Add a small per-frame metric interface;
  revisit NCC-specific conditioning, diagnostics and stopping calibration.
- **Further metric/optimizer controls:** Gaussian-weighted local-CC kernels,
  MSE/custom losses, optimizer choice/hyperparameters and regularization hooks.
  SyN's rectangular CC radius and update/total smoothing strengths are public
  options already. Follow ITK's unit distinction: image-pyramid smoothing
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
