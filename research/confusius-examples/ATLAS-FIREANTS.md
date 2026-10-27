# FireANTs atlas controls and the sampling-fidelity question

**The mixed native-slice voxreg fitter is an alternative model, not a literal
optimized reproduction of CF/ITK sampling. This was not inherited from FireANTs.**

Open **`artifacts/atlas-fireants/qualified/fireants.html`** for executed traces,
best/returned support-aware overlays and the expanded-slab voxreg ablation.
Measurements: [atlas-fireants-results.json](atlas-fireants-results.json).

## Why ITK does not always retain the best by default

For the gradient-descent optimizer specifically, installed ITK returns
`GetReturnBestParametersAndValue()==False`. Its
[documented option](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1GradientDescentOptimizerv4Template.html)
explicitly addresses overshoot/oscillation. The documented downside is additional
storage for best parameters, potentially large for displacement fields; copying
new best states also costs work. The docs do not establish the historical reason
for choosing false, so we should not invent one. SimpleITK's tested high-level
registration wrapper does not expose this flag.

For our deterministic6/12-parameter linear fits, the storage cost is negligible:
best retention is sensible, and voxreg already does it. The separate caveat is
that best metric need not mean best anatomy, especially when coverage changes.
That calls for validation, not a presumption that the final state is preferable.

## Where voxreg departed from the reference

The mixed-grid extension introduced in `ce8636f` fits the native slice domain,
internally reverses a moving-slice call, and completes a9-parameter tangent affine
with a unit normal. This avoids estimating unmeasured full affine normal freedoms
and avoids fabricating extra observations. It is a deliberate **model/domain
change**, not merely a faster implementation of the same CF calculation.

Both installed CF and ordinary FireANTs registration evaluate on the caller's
fixed grid and change problems when the caller swaps roles. FireANTs did not
supply voxreg's role-reversal convention. Its affine/rigid implementations already
use centered parameterizations, preserving their world-map seeds.

If the requirement is reference reproduction, we must first establish the same
fixed-domain/expansion/transform-space baseline, then isolate optimizer,
gradient and arithmetic changes. Invariance under swapping cannot substitute
for that comparison. ITK itself is not inherently limited to CF's chosen slab
protocol; the user's separate successful unexpanded ITK configuration remains
unreproduced without its settings.

## FireANTs experiment

Installed FireANTs1.5.1, pinned `f8a9cc259ab9113fc43bd99797000fa8fc34eb59`;
its four core installed sources match that checkout. CUDA, native float32, Torch
interpolator fallback (fused ops unavailable). No package/source modification.

- Original cached template slab and recording mean/dB images/manual seeds.
- The installed3D FireANTs `Image` adapter rejects the native singleton recording:
  normalized-to-physical geometry is singular. This is an adapter failure, not
  a measured registration result or a claim that2D FireANTs is unsupported.
- Controlled3D fits therefore use **exactly CF's four-slice expansion**, preserving
  voxel-cell extent and center. No extra measured through-plane information.
- Eight fits: rigid/affine × both directions × two losses.
- Native local CC, default kernel3; separately the existing benchmark's custom
  masked squared global NCC. These are not interchangeable scores.
- Single scale,500 Adam updates, native API rate.03, `around_center=True`, no blur,
  stopping disabled to inspect the entire trajectory. This is a bounded controlled
  setup, **not a fully tuned/default end-to-end FireANTs workflow**.
- Actual initial transforms and normalized warp grids are checked against the
  physical seeds. CF's transpose convention makes SITK physical x/y/z numerically
  our world z/y/x; no extra axis permutation is applied to exported matrices.

### Best observed poses

RMS values refer to the original good CF recording-moving affine, not independent
anatomical truth. Native coverage counts recording-grid centers inside the
original template-center domain. Gray pixels are unsupported, not extrapolation.
Normal thickness is measured **perpendicular to the mapped recording plane**,
not just the length of one matrix column; factor1 preserves thickness.

| FireANTs loss / model | Moving image | Best state | RMS native px | Native coverage | Normal thickness factor |
| --- | --- | ---: | ---: | ---: | ---: |
| Local CC / rigid | Recording | 123 | 91.948 | 0% | 1 |
| Local CC / rigid | Template | 112 | 11.008 | 99.7% | 1 |
| Local CC / affine | Recording | 490 | 44.154 | 100% | .5011 |
| Local CC / affine | Template | 276 | 15.863 | 99.9% | .00024 |
| Masked global NCC / rigid | Recording | 404 | 31.493 | 0% | 1 |
| Masked global NCC / rigid | Template | 44 | 26.721 | 0% | 1 |
| Masked global NCC / affine | Recording | 67 | 8.117 | 94.7% | .6509 |
| Masked global NCC / affine | Template | 490 | 79.922 | 13.9% | .00229 |

**None passes the unchanged.25px reference gate**, including best checkpoints.
Some are physically unacceptable affine compressions; rigid cannot collapse but
can leave the template crop. Best retention would not rescue these runs.

### Why apparently good scores are unsafe here

Two additional read-only checks in `metric_probe.py`/`metric-probe.json`:

1. Native local CC's default numerator/denominator smoothing gives an **all-zero
   prediction** a loss of **−.5646128058** against this dB template. This is exactly
   the final recording-moving rigid loss. Without numerator smoothing, the
   all-zero prediction's loss is0. Unmasked padding/background and these smoothing
   constants make that native score misleading in this intensity/domain setup.
   We did not silently tune away this failure or change FireANTs defaults.
2. The masked-global recording-moving rigid peak has NCC².9030 but only about
   **7 of211120 fixed-grid centers** geometrically supported. Its native recording
   plane has no template overlap. A replicated slab can retain a few off-center
   samples even when the native center plane misses the template. The other
   global rigid peak has1838/40000 supported fixed centers but likewise no native
   plane support. Counts use double physical geometry and may differ at individual
   floating-point mask boundaries from actual float32 interpolation masks.

The custom mask excludes zero-padded interpolation outside center support via an
all-ones channel. ITK instead uses half-voxel support and border-clamped values;
this custom control is **not exact ITK NCC**. Native local CC is unmasked and also
has different smoothing/windowing. Joint native support across all poses is empty,
so common-domain NCC is correctly reported as `None`, not fabricated/ranked.
These findings do not establish that FireANTs cannot register this anatomy under
a better protocol; they show this single-scale configuration is not a solution.

## Explicit caller-fixed voxreg ablation

We also fed the same four-slice recording to the **unchanged shared engine**. With
both inputs volumetric it no longer reverses roles; affine uses12 parameters and
evaluates on the caller's fixed grid. The affine pivots match each fixed image's
geometric center. This changes inputs/domain/model together, not only one variable.

| Voxreg expanded-slab arm | Steps | Status | RMS px vs original good CF |
| --- | ---: | --- | ---: |
| Rigid / recording moving | 75 | converged | 7.113 |
| Rigid / template moving | 49 | converged | 6.999 |
| Affine / recording moving | 68 | converged | **.889** |
| Affine / template moving | 74 | invalid_transform; best retained | 6.298 |

The recording-moving affine improves reference agreement from native-plane6.218px
to.889px; perpendicular normal factor is.99951. It still fails.25px. The reversed
expanded affine's retained best already compresses thickness to.662; the later
trajectory triggers the existing transform guard. This is diagnostic, not a new
accepted model or replacement reference.

Thus the domain/model choice materially contributes to the mismatch. The ablation
still differs from ITK in support bounds, gradient construction, parameter scaling,
Adam versus GD and stopping/best retention. It is a useful closer baseline, not
proof of full algorithmic parity or an instruction to ship replicated data.

## Verification and scope

-10 new focused tests cover nonidentity rigid/affine seeds, physical warp grids,
  extent/center-preserving expansion, masked loss and real CPU/CUDA execution.
-8 FireANTs500-update fits and4 expanded-voxreg fits actually executed. They were
  checkpointed before an export-only import failure. The qualified notebook
  reloads those checkpoints without repeating optimization; hashes and the original
  measured driver are retained. Fitting function ASTs are unchanged in the renderer.
- Production and installed library sources are unchanged; no implicit downloads,
  new dependency, reference substitution or numerical gate changes.
- No multi-resolution tuning, rate sweep, masked-local-CC optimization, exact
  half-voxel FireANTs objective, full suite or performance comparison was run.

```sh
# New fits, fresh directory required:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/atlas_fireants.py \
  --output artifacts/atlas-fireants/new-run

# Rendering saved checkpoints only:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/atlas_fireants.py \
  --output artifacts/atlas-fireants/new-render \
  --replay-from artifacts/atlas-fireants/comparison
```
