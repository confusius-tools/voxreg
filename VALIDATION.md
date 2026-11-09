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
MI checks include an independent NumPy density oracle, double finite-difference
gradients through both sampled images, excluded-sample gradients, chunk and
frame independence, native-range contrast preservation, backward activation
storage and nonlinear-contrast physical truth in 2D/3D translation and SyN.
Integration covers masks, stage overrides, failure isolation and pyramids.
CUDA tests skip when unavailable; CPU success does not qualify CUDA.

Length-unit regressions cover all ConfUSIus-supported length scales, mixed-unit
fixed/moving/mask grids, cropped/strided/oblique geometry, large world origins,
every transform model, staged pyramids, labelled seeds and initialization aliases.
Resampling is checked against independent SciPy pulls, including displacement
component units independent of their native grid units. Native output geometry,
input immutability and invalid/mismatched unit metadata are checked separately.

Batch comparisons of float32 fits allow a maximum physical pull difference of
0.001 minimum-spacing voxels over the full grid. Linear matrices are evaluated
at world points; their raw translation entries depend on the world origin.
Reference transforms remain exact across batch sizes; failed-frame exports allow
only float64 finalization roundoff (1e-12 in matrix entries or mm).
Registered intensities use a per-frame relative RMS difference below 1e-5.

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

## First real-data motion-quality case

The paper Figure 2 rat75 / 20220523 / slice32 window (frames 220–339) now has a
retained, non-timing reproduction in
`artifacts/realdata-rat75-quality-20261109-01/`. The paper script and caches were
not changed. Raw DVARS uses unnormalized power differences over 119 adjacent
pairs; source hashes, package versions, options, per-frame diagnostics, transforms
and curves are retained with the local artifacts.

For fresh CPU rigid runs, mean DVARS in the original brain ROI was 193.52 raw,
160.69 with default voxreg dB fitting, and 156.50 with ConfUSIus's script settings.
Seven default-voxreg pairs exceeded raw DVARS, versus none for ConfUSIus. Changing
voxreg to unscaled fitting did not remove these differences. These configurations
have different optimizers, intensity transforms and stopping policies.

Intersecting the brain ROI with fresh voxreg support across every frame removed
31 of 10,168 voxels. On this restricted ROI, mean DVARS was 191.67 raw, 154.26
voxreg and 154.94 ConfUSIus; neither fresh method had a pair above raw. ConfUSIus
did not export its own support mask here, so this is a sensitivity check rather
than a fully matched spatial-support or anatomical-accuracy qualification.

All 191 unsupported brain samples in the default voxreg run were within the
half-voxel cell extent and were filled with the frame minimum. Voxreg's ordinary
axes currently use center-range output support; ITK's image-buffer checks include
[half-voxel cells](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1ImageFunction.html).
Audit this support/interpolation contract before interpreting the observed
spikes as optimizer failure. The older voxreg cache differs materially from the
fresh runs and has no established configuration provenance. No default, support
contract or optimizer was changed to improve these QC curves. Smaller DVARS
alone does not demonstrate better spatial registration.

## Limits

- Positive sampled Jacobians do not prove global invertibility; sampled inverses
  have nonzero residuals, reported in diagnostics.
- Float32 fitting loses precision that double export cannot recover.
- Gaussian-Parzen MI is a distinct estimator from Mattes MI. Native min/max
  ranges include intensities outside masks and can be affected by outliers.
  Sparse ROIs or overly fine bins can give misleading histogram estimates.
  MI bandwidth, convergence thresholds and performance need real-data qualification;
  successful small synthetic fits do not qualify multimodal clinical workflows.
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
