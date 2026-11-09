# Validation and known limits

## Checks

```bash
uv run pytest
uv run prek run --all-files
```

The registration-module refactor was checked against ten saved pre-refactor
public calls spanning all models and full staged pipelines in 2D and 3D with
pyramids. Images, support, forward/inverse maps and stage/level diagnostics were
identical, excluding runtime timing metadata. The existing full suite remains
the numerical regression gate; no precision or quality threshold was changed.

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

Final cubic-image tests exercise public `register` and `resample` against
independent SciPy `map_coordinates(order=3, mode="mirror", prefilter=True)` and
native SimpleITK B-spline samples in 2D/3D. Cases include unequal/oblique and
reversed/strided grids, physical units, short/singleton axes, float16/32/64 and
integer intensities, large DC contrast, exact knots, integer half ties,
overshoot/range failures, invalid queries, chunk boundaries, mixed dimensions,
and shared/per-frame matrix and dense-field pulls. All models and a full staged
pyramid check unchanged fitting diagnostics and exported-transform reuse.
Float64 oracle discrepancies in the initial unequal-grid checks were below
5e-14. CUDA cases are collected but skip on this CPU-only system; no CUDA or
performance qualification is claimed. The existing paper-case support policy
and artifacts are unchanged.

Automatic device tests compare public default/auto calls with explicit devices
for linear and SyN models, complete staged pyramids and all final resampling
interpolators with matrix and dense-field pulls. They check resolved-device
metadata, once-per-pipeline selection, explicit CPU without CUDA probing,
unavailable CUDA rejection, and propagation of simulated CUDA runtime/OOM errors
without CPU retry. CPU selection is exercised locally; actual CUDA comparisons
skip when unavailable. Selecting auto does not establish new CUDA accuracy or
performance qualification.

Batch comparisons of float32 fits allow a maximum physical pull difference of
0.01 minimum-spacing voxels (1% of one voxel) over the full grid. This is a
user-selected reproducibility tolerance, not a ground-truth accuracy threshold;
float32 NCC rounding can select different nearly tied best iterations. Linear
matrices are evaluated at world points; raw translations depend on world origin.
Reference transforms remain exact across batch sizes; failed-frame exports allow
only float64 finalization roundoff (1e-12 in matrix entries or mm).
Registered intensities use a per-frame relative RMS difference below 1e-5.

Benchmark workers verify staged-input hashes, output contracts and full-grid
accuracy. Keep adapter diagnostics separate from full-workload performance
and anatomical validation. Comparisons include **ANTsPy, FireANTs and voxreg**;
analytic truth defines acceptance, not agreement between backends.

## Intensity scaling

Moving/fixed controls default independently to `None`. `"log1p"` now computes
`ln(1 + abs(x))`, replacing `"db"`; `"sqrt"` and positive finite powers still use
magnitudes. The shared `intensity_scaling` argument and `"none"`/`"db"` spellings
are rejected. Transform evaluation remains float64 before float32 fitting.
Recording-wide numerical normalization is removed: NCC conditioning is per image,
MI ranges remain per image, and no transform requires a temporal intensity scan.
Zeros need no log floor; log1p compression depends on absolute intensity units.
See [the complete policy](README.md#fitting-intensity-scaling).

The scaling checks pass with CUDA available on an NVIDIA GeForce RTX 5080
Laptop GPU; integration cases explicitly cover CPU and CUDA. They compare
conditioning to independent NumPy magnitude/range calculations and fits to
pretransformed-image runs, across NCC/MI, linear/SyN/staged pyramids, masks and
native resampling. They check signed inputs, zeros, finite positive exponents,
overflow/underflow, low contrast, invalid settings, input immutability,
log1p zeros/reference behavior and unchanged fitting images across batches.
Direct conditioning comparisons verify independence from unrelated frames for
float32/float64/integer inputs, NCC/MI and every transform.
Independent native-centroid calculations and frozen-optimizer regressions cover
explicit/implicit references and stages, including rejection of negative native
moment weights despite magnitude-based fitting transforms. Expanded synthetic
rigid recovery checks retain the 0.25-minimum-spacing-voxel gate for no transform,
log1p, square root and power 2. The full CUDA-enabled suite had 1368 passed;
a subsequent bright-frame isolation regression also passes (final focused scaling
suite: 125 passed). A repeat full run timed out under high host load without
reported failures; pre-commit checks pass. No anatomical, performance or real-data
default qualification is claimed. Lazy execution must preserve per-image conditioning.

At the intensity-scaling commit, the CPU-only suite had 956 passed, 297 skipped;
the CUDA-enabled run had 1327 passed and 13 failures.
All 13 failing cases were reproduced on unchanged pre-scaling `HEAD`: two
initialization batch comparisons, six auto-device availability-probe-count checks,
three 3D batch comparisons and two per-frame seed comparisons. These existing
CUDA/batching failures were recorded without changing tolerances, truth gates,
optimizers or production device-selection behavior at that point. Prior
CPU-only qualification notes above describe the original feature checks.

Test-only follow-up fixes count voxreg's resolver calls rather than Torch's
internal probes and restore explicit CPU baselines. Centering now has separate
strict CPU and physical CUDA batch checks. At that point 1341 tests passed and
`center_moments` differed by 0.00306 voxels against the former 0.001-voxel gate.
Traces show nearly identical trajectories (parameter differences around 1e-8),
but NCC rounding selects iteration 61 versus 81. The user subsequently selected
0.01 voxels for physical batch agreement, accepting this case without a runtime
change. Ground-truth accuracy, strict CPU, device-equivalence and image-output
checks are unchanged. Slower precision/reduction candidates were rejected.
With this policy, the full CUDA-enabled suite passes: 1342 passed; pre-commit
checks pass.

The live benchmark adapter now explicitly requests no moving/fixed transform,
explicitly selecting no nonlinear transform. The live notebook example now
requests log1p; this is not numerically equivalent to its historical dB settings. Frozen historical sources,
settings, paper outputs and artifacts were not migrated or rerun. New comparisons
must disclose the default/convention changes rather than relabel old results.

## Fitting winsorization

Optional moving/fixed percentile pairs are independently validated before any
stage runs. Each native signed image/frame is clipped using NumPy linear
percentiles before intensity transforms, metric conditioning and pyramids, with
no ROI-restricted or recording-wide bounds. Native moments and final output
resampling remain untouched; disabled clipping skips percentile calculation.

The 163 focused CPU/CUDA checks compare public fits to independently NumPy-clipped
inputs for all models and staged pyramids, NCC/MI, full-image bounds outside masks,
and moving-only/fixed-only/both settings. Native output pulls, input immutability,
implicit references, float32/float64/integer frames, batch independence, signed
outliers, endpoint percentiles, invalid bounds, percentile overflow and constant
image handling are covered. Frozen-optimizer cases verify native centroid weights
through stages. The full CUDA-enabled suite passes: 1532 passed; all pre-commit
checks pass. No real-data clipping preset, performance or lazy-execution
qualification is claimed; Dask support remains separate roadmap work.

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

The following reproduction used the then-default shared dB fitting option;
it does not qualify the new untransformed default or revised dB conventions.

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
