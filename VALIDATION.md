# Validation and known limits

## Checks

```bash
uv run pytest
uv run prek run --all-files
```

The suite checks physical geometry, unequal/oblique/cropped/strided grids,
large origins, interpolation against SciPy/native-library oracles, dtype
preservation, failure isolation, convergence, transform export and offline
resampling. SyN checks include nonlinear truth recovery, field composition,
inverse residuals, Jacobians and float32 fitting with float64 finalization.
Pyramid checks include anisotropic/sheared/reversed physical smoothing against
independent convolution, field interpolation in mm, linear/nonlinear truth
recovery, per-level counts/statuses, batching and native-grid output resampling.
Staging checks cover manual full-transform handoff, independent physical truth,
shared/overridden settings, single output interpolation, initialization once,
motion batching, reference preservation and failure isolation into terminal SyN.
ROI checks cover ITK physical-cell rounding and native masked global NCC,
independent mask-grid world queries and both SyN halfway pulls, masked local
statistics, per-frame motion masks, empty ROI failures and unchanged output support.
CUDA tests skip when unavailable; CPU success does not qualify CUDA.

Benchmark workers verify staged-input hashes, output contracts and full-grid
accuracy. Keep adapter diagnostics separate from full-workload performance
and anatomical validation. Comparisons include **ANTsPy, FireANTs and voxreg**;
analytic truth defines acceptance, not agreement between backends.

## Acceptance

[BENCHMARK.md](BENCHMARK.md) defines the measurement boundaries:

- Registration: worst fitted-frame physical pull RMS **≤ 0.30 minimum-spacing
  voxels** against analytic truth over the full grid, for every model.
- Resampling: worst-frame relative intensity RMSE **< 1e-5** against independent
  float64 SciPy interpolation, with physical support checks.
- Timing: five complete calls, first included. Separate memory and accuracy
  workers use identical inputs and the same effective backend batching.

Passing unit tests or improving NCC does not establish correct correspondence.

## Limits

- Positive sampled Jacobians do not prove global invertibility; sampled inverses
  have nonzero residuals, reported in diagnostics.
- Float32 fitting loses precision that double export cannot recover.
- Initialization, overlap and budgets affect capture. ANTsPy convergence remains
  enabled; its API does not expose observed iteration counts.
- Pyramid schedules are explicit, without a qualified automatic preset. Coarse
  striding can omit trailing samples; small axes retain at least two samples.
  Coarse field boundaries are fixed, and residuals outside their support are zero.
  Passing synthetic pyramid checks does not establish wider capture on real data.
- Masks gate metric samples, without altering image interpolation or smoothing;
  ROI boundary values can include neighbouring intensities from outside the ROI.
  Small ROIs can vanish at coarse levels or yield unreliable local statistics.
  Shared masks apply to all stages; soft masks and stage-specific ROI overrides
  are not implemented. Mask support does not establish anatomical correspondence.
- Batching does not bound full CPU inputs/outputs or optional dense exports.
- Synthetic results do not establish anatomical accuracy. CUDA, other platforms,
  large volumes and full recordings require their own measurements.
