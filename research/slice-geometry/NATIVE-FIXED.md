# Native caller-fixed mixed sampling

**Current contract:** original images, caller-fixed metric grid, no automatic role
reversal, no replicated slice/slab. The shared optimizer/API remains unchanged.

```python
# Evaluate template intensities on native recording centers.
voxreg.register(moving=template, fixed=recording, ...)

# Evaluate native recording intensities at template centers inside its voxel extent.
voxreg.register(moving=recording, fixed=template, ...)
```

These are different fitting problems, not inverse-equivalent executions. The
second direction cannot obtain a measured through-plane intensity gradient from
one recording plane. A singleton is constant over its affine-defined voxel cell;
its spacing is not a measured acquisition PSF. Fixed centers missing that cell
provide no usable samples. No hidden fallback or replication is performed.

## Implementation and compatibility

- `rigid.register` keeps the supplied roles throughout preprocessing, fitting,
  export and final interpolation. Public pulls remain fixed-world→moving-world.
- `sampling.make_grid` supports a moving singleton against a fixed volume directly.
  Mixed source support follows ITK's half-open `[-.5,size-.5)` bounds on all axes,
  with border-clamped interpolation. Mixed fixed coordinates are integer-centered,
  avoiding a normalize/denormalize roundoff at exact voxel faces.
- Translation/rigid/affine remain3/6/9 parameters for mixed pairs. Affine still
  constrains the **physical slice-to-volume map** to tangent deformation and a unit
  perpendicular normal. When the slice is moving, the sampler/exporter uses the
  inverse constrained linear map, with translation centered on the caller's fixed
  image. Tilt and in-plane shear remain free; normal scale/shear seed rejection is
  preserved. There is no new full12-parameter mixed affine or direction flag.
- Mixed pivots are geometric **fixed-image centers**, including with moments seed
  initialization. Seeds remain complete optimized world mappings, not frozen chains.
- `metric_domain` is now `fixed_slice` or `fixed_volume`; there is no `moving_slice`
  fitting domain. `valid_fraction` describes that fixed grid. For compatibility,
  moving-slice calls retain `output_valid_fraction`, now the same domain.
- Output retains moving dtype and fixed grid. Float64 final interpolation and
  nearest-even/clipped integer conversion are unchanged. Empty moving-slice support
  warns and reports insufficient overlap, rather than fitting on a different grid.
- Same-dimension slice/volume arithmetic and optimizer defaults are unchanged.
  Historical mixed-direction invariance is intentionally no longer the contract.

This follows ITK's fixed-domain sampling convention, **not CF's historical
four-slice workaround**. Our constrained affine, autograd derivative, Adam and
best-state return are still distinct from generic ITK affine/GD. Do not claim full
algorithmic parity from domain agreement.

## Verification

- **531 tests passed**,1096 warnings,239.74s. An earlier full run timed out at200s;
  the successful longer run and failed development checks are retained separately.
- **22 new CPU/CUDA cases** verify unexpanded optimizer input/target shapes,
  nonidentity seeds on oblique/common-shifted grids, fixed pivots, sampling/export
  agreement, finite gradients, singular/normal-scale seed rejection and voxel faces.
- Twelve of those cases compare the exact seeded metric against native singleton
  SimpleITK `MetricEvaluate`, with gradient filters disabled, to1e-6 absolute loss.
  No image expansion is used in that comparison.
- Existing slice-fixed synthetic truth gates remain. Only assertions depending on
  the retired symmetric objective/no-update-from-its-truth assumption were replaced
  by independent fixed-domain/seed/metric tests, not loosened numerical gates.
- All **four frozen120-frame controls** remain bitwise identical: fixed/convergence
  × CPU32/CUDA120, including registered values, masks, matrices, clocks and statuses.
- Full suite includes batching, failed frames, callbacks, float64/integer outputs,
  strided grids, physical seeds and CPU/CUDA coverage. Gates remain.25px reference
  RMS and.001px maximum CPU/CUDA point discrepancy.

- Offline lock/build, Ruff check/format and `git diff --check` pass. The built wheel
  matches every current package source byte and passes18 isolated CPU smoke cases:
  both directions, all three models, float32/float64/uint16 output, reversed recording
  strides and metric callbacks. No editable-source import is used in that smoke test.
- Source/input hashes, unchanged installed CF sources, all precision gates, all
  failed atlas reference gates, and the executed self-contained notebook are verified.
  The report has4 executed code cells,3 embedded figures and no cell errors.

Logs and snapshots: `artifacts/native-fixed-domain/validation/`. Verified metrics:
[atlas-native-fixed-results.json](../confusius-examples/atlas-native-fixed-results.json).

## Fresh atlas check, no expansion

Open **`artifacts/native-fixed-domain/atlas/native-fixed.html`**. Eight actual fits:
rigid/affine × both supplied directions × CPU/CUDA. Original recording1×125×80,
template20×91×116, existing manual inverse-equivalent seeds, rate.1,500/window50.
The observer records actual optimizer array shapes, fixed pivots, raw histories,
physical states and best-return checks. The old good CF matrix is a **frozen pose
comparison**, not a newly executed expanded-data registration or anatomical truth.

| CPU arm | Updates | Returned best clock | RMS px vs old good CF |
| --- | ---: | ---: | ---: |
| Rigid / recording moving | 50 | 29 | 5.415465 |
| Rigid / template moving | 49 | 24 | 7.818690 |
| Affine / recording moving | 143 | 131 | 5.894310 |
| Affine / template moving | 90 | 56 | 6.161640 |

All four CPU/CUDA point gates pass; worst maximum discrepancy **4.89e-6px**.
The unit-normal/orthogonality checks pass throughout. Unsupported template pixels
are gray; plotted native-center coverage is separate from half-voxel metric
coverage. The recording-moving metric's≈8–9% denominator is the entire template
volume; it is not comparable to≈93–95% of the much smaller fixed recording grid.

**Rigid-panel interpretation:** its original good CF reference is a **full affine**,
not a rigid fit. The read-only diagnosis in
`artifacts/native-fixed-domain/rigid-investigation/` shows≈11.4% vertical scale in
that reference and an irreducible4.23px mapping error for any rigid approximation.
Actual historical CF rigid and current voxreg rigid both lack this deformation;
they still differ by2.42px and are not a matched native-protocol parity pass.
Template FOV coverage is useful clipping QA, not an alignment score; the companion
plots identify models explicitly and move percentages out of panel headings.

**No atlas reference gate passes.** This completes the requested sampling-policy
change, not anatomical alignment. The earlier.889px replicated/full-affine ablation
is not this native/constrained protocol and must not be substituted for these
results. No defaults, references or acceptance gates were tuned to improve them.
No performance qualification or corrected unexpanded CF fit was run.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/atlas_native_fixed.py \
  --output artifacts/native-fixed-domain/new-atlas
```

Output must be a fresh directory. Existing slab/native-slice reports and failed
harnesses remain historical evidence. The old centered-followup report asserts its
retired direction-invariance protocol; use its measured-source snapshots/old engine
to reproduce that experiment, and the new runner for current sampling behavior.
