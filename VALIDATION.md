# Validation and known limits

## Checks

```bash
uv run pytest
uv run prek run --all-files
```

CUDA tests skip when unavailable; CPU-only success does not qualify CUDA.
The production suite covers physical geometry, unequal/oblique/cropped/strided
lattices and large origins; SciPy/native-library interpolation oracles; original
dtype and integer safeguards; per-frame state, convergence and failure isolation;
field composition, inverses and Jacobians; and optional transform export.

SyN checks cover float32 working fields and native fitting gradients, unchanged
original float64 data, double finalization/export, large-origin invariance,
unrepresentable working lattices, and direct/packed Jacobian parity.

Benchmark adapters have separate bounded local checks for staged-input hashes,
ANTsPy Python-only dispatch, single-resolution settings, complete forward-pull
composition, reference-frame preservation, phase isolation and report generation.
These checks are not full-workload performance or anatomical qualification.

## Acceptance

[BENCHMARK.md](BENCHMARK.md) defines the current measurement and scoring boundaries:

- Registration: worst fitted-frame full-grid physical pull RMS **≤ 0.30
  minimum-spacing voxels**, measured against analytic truth, for every model.
- Resampling: worst-frame relative intensity RMSE **< 1e-5**, against independent
  float64 SciPy interpolation with physical support checks.
- Timing: five complete calls, first included; memory and accuracy use separate
  workers, identical prepared inputs and the same effective backend batching.

Passing unit tests, higher NCC, agreement with another backend or CPU/CUDA
agreement does not establish correct correspondence. ANTsPy is a comparison
backend, not the ground-truth deformation.

## Limits

- Positive sampled Jacobians do not prove global invertibility; numerical inverse
  residuals are nonzero and exposed.
- Float32 fitting has finite precision; double export cannot recover information
  already lost during optimization.
- Initialization, overlap and iteration budgets affect capture. ANTsPy convergence
  is enabled, unlike the forced resident update budget; iteration counts are not
  fabricated when its API does not expose them.
- Batching does not bound full CPU inputs/outputs or optional dense exports.
- Synthetic qualification does not establish anatomical accuracy.
- Platform, CUDA, large-volume and full-recording behavior require their own
  measurement. Do not infer them from another machine or a small diagnostic.
