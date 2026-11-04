# Next steps

Goal: a small, physically correct CPU/CUDA registration engine for ConfUSIus.
Current model support is documented in the [README](README.md); numerical
contracts and unresolved accuracy limits belong in [VALIDATION.md](VALIDATION.md).

## Priorities

1. **Review a fresh full benchmark.** Establish the current precision-policy
   baseline using [one pipeline](BENCHMARK.md), without competing heavy jobs.
2. **Address SyN truth accuracy.** Current full-budget controls exceed 0.25 voxels
   RMS. Higher NCC or agreement with another library is not a substitute.
3. **Optimize one measured bottleneck.** For accuracy-passing cases, compare
   workload/device-specific batching or a source-guided implementation change.
   Preserve timing boundaries and report memory; see [PERFORMANCE.md](PERFORMANCE.md).
4. **Broaden real-data qualification.** Revisit pair, motion and atlas workflows
   with original images, native geometry and explicit quality limitations.

## Candidates, not enabled defaults

- **Multistart:** only with a selection rule that rejects tiny-overlap or
  anatomically wrong metric optima, especially for slice placement.
- **Line search:** investigate measured overshoot and extra evaluation cost before
  adding another optimizer policy.
- **Automatic staging:** manual rigid→affine and affine→SyN initialization already
  work. Reuse full returned matrices and the original images; do not resample an
  intermediate image or invent a separate transform chain.

Masks, pyramids, streaming/Dask execution and mixed-dimensional deformation are
not part of the current implementation. Add scope only for a demonstrated need,
not as a prerequisite to benchmarking or fixing the existing accuracy failures.

Keep source/settings and minimal failure evidence before cleanup. Never retune a
truth threshold or shorten work merely to manufacture a speedup. Historical
experiments and their original limitations remain in [research](research/README.md).
