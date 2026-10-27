# Late CF overshoot; revisiting voxreg centering and both directions

**Late worsening is not necessarily sustained divergence. Neither continuing nor
centering alone establishes correct anatomy. Voxreg was already centered.**

Open **`artifacts/atlas-centered-followup/validated/followup.html`** for executed
curves, support-aware overlays, before-stop best poses, continued best/final poses,
and both public voxreg directions. Measurements/provenance:
[atlas-centered-followup-results.json](atlas-centered-followup-results.json).

## Why the stopped traces end with a jump

CF uses physically scaled gradient descent with rate1 fixed here, no line search
or trial-step rejection. Its window50 convergence test observes the objective's
recent improvement trend, not pose stability; a negative trend can satisfy the
stopping criterion. Thus the window reacts **after** worsening, making a late
upturn a characteristic ending of these stopped curves. CF returns the final
transform, not the best observed transform.

We disabled convergence only (window1001, budget1000), kept seeds/centers/rate and
all image settings unchanged, and verified native metric and physical-state
prefixes bitwise. We also shrank actual parameter updates, not matrix chords
that would leave a rigid/unit-normal model.

At the unit-normal fit's original best state89:

| Fraction of actual update | Negative NCC² | Changed support points |
| --- | ---: | ---: |
| 0 | −.1606301831 | 0 |
| .03 | **−.1606391555** | 0 |
| 1 | −.1601911356 | 20 |

A smaller step improves while the full step worsens: **finite-step overshoot is
measured**, not inferred from the plot. At the later jump94, a1% step improves
with no support changes; the full step worsens and changes66 support points.
Rigid and full affine probes likewise improve for smaller fractions. The
hard, pose-dependent support also creates sharp changes: at full-affine state158,
a.01% step improves without changing support, while a.1% step admits/excludes one
point and worsens. This supports overshoot plus moving-support effects, not a
claim that the gradient is universally wrong. No fixed-support causal ablation
or line-search fix was applied.

## What continuing actually does

These are recording-fixed CF fits. Coverage is the geometric fraction of expanded
fixed-grid centers inside the moving template's ITK half-voxel support. It is
measured, **not a new overlap rejection threshold**.

| Model | Best NCC² before stop | Best NCC² over1000 | State of best over1000 | Coverage there | RMS px vs original good CF there |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full affine | .169674 | .169674 | 146 | 90.68% | 14.33 |
| True rigid | .127309 | **.165398** | 956 | **28.20%** | 38.44 |
| Unit-normal + in-plane affine | .160630 | **.382802** | 998 | **20.42%** | 43.37 |

Rigid and unit-normal fits recover the metric and go much lower after the initial
bad excursion. Thus the early upturn is **not sustained objective divergence**.
However, the later fits leave most of the cropped template's field of view:
unit-normal native-center coverage drops from90.02% at the earlier best to18.61%,
and mapped recording-center depth moves to7.798mm, tilt18.51°. Their scores then
refer to a much smaller subset of pixels. The unit-normal constraint remains
satisfied; eliminating collapse does not prevent this escape.

The unrestricted affine instead becomes nearly singular, changes determinant
sign during the trajectory, and ends with minimum singular value≈1.1e-12. Its
state1000 NCC² is only.111412. That transform is not an acceptable physical map;
its inverse and derived placement statistics are ill-conditioned.

**Preserving/inspecting the best pose is appropriate, but stopping at the first
increase is not a reliable anatomical rule.** It would miss later metric recovery,
and even the pre-upturn best poses already fail the original good reference.
Step acceptance/backtracking can address local worsening; best-pose retention,
geometry checks and coverage/anatomical validation are separate requirements.
No automatic rule can infer the correct section from these scores alone.

### Display correction

The shared historical SciPy helper returns nearest-border extrapolation **and a
separate validity mask**. Earlier composite plots used the values without that
mask. This is especially misleading when80% of the recording leaves the template:
the extrapolated edge can look like a complete brain overlay.

The validated report now colors unsupported native-center pixels **gray**, adds
coverage to each panel, and never presents edge extrapolation as matched anatomy.
Fit matrices, objectives and numerical gates were unaffected. Earlier unmasked
figures remain archived, not overwritten; use the validated report for assessment.
ITK half-voxel coverage and stricter native-center display coverage are reported
separately, not conflated.

## Voxreg: the pivot was not lost

Inspection and live optimizer instrumentation confirm:

- Mixed fits already use the native recording's geometric center, in world z/y/x:
  **[.40000000596,5.99168001860,.00000001118]mm**.
- Matrix seeds are decoded as centered displacement `b+(A-I)c`; re-export preserves
  the supplied initial world mapping. No second centering translation is applied.
- Both public directions fit the same native-slice domain internally, with inverse
  seeds/output maps as appropriate. This intentionally differs from CF's swap.
- All observed rigid/affine states preserve a unit perpendicular normal. Affine
  has9 native-plane parameters, not an unconstrained12-parameter slab fit.

Eight fresh runs: rigid/affine × both directions × CPU/CUDA, unchanged rate.1,
max500/window50. Both CPU directions have **bitwise-identical canonical matrices
and raw histories**. The CUDA directions also coincide. Maximum CPU/CUDA point
errors are1.31e-7px rigid /9.76e-7px affine, below the unchanged.001px gate.
All four affine matrices reproduce the earlier unmodified direction experiment
**bitwise**; there was no missing-center bug to fix in voxreg.

| Voxreg model | Stops at | Returns best at | RMS px vs original good CF affine, either direction |
| --- | ---: | ---: | ---: |
| Rigid | 49 | 23 | 7.710 |
| Native-plane affine | 91 | 56 | 6.218 |

Voxreg already returns its best valid state rather than the last worsening trial.
Its raw last errors would be9.924px /7.627px respectively. Best retention helps
but does not meet the unchanged.25px reference gate.

Direct comparison with **returned, centered CF fits in the same public direction**:

| Model / moving image | Voxreg-vs-CF RMS native px |
| --- | ---: |
| Rigid / recording | 4.575 |
| Rigid / template | 14.359 |
| Affine / recording | 6.301 |
| Affine / template | 11.237 |

The recording-moving CF affine is the12-parameter expanded-slice model; the
template-moving comparator is the9-free-parameter unit-normal control. Rigid
comparators are genuine Euler3D, not the misleading matrix-seeded affine fallback.
CF expands the slice; voxreg uses native centers. GD versus Adam, rotation/affine
parameterization, gradient construction and physical scaling also differ. These
are explicit comparisons, not claims of identical optimization or cross-domain
metric rankings. Reference disagreement is not independent anatomical truth.

## Scope, verification and reproduction

- **26 focused tests pass**, including CUDA state capture, actual CF execution,
  parameter-path probes, seed/pivot checks and support-aware rendering.
- Each successful run has11 fits:3 CF continuations and8 native voxreg fits. Native CF metric and
  physical-state prefixes match saved controls bitwise; the two independent
  follow-up runs reproduce matrices and objective histories bitwise.
- Input/CF/production-source hashes match the prior boundary. All research patches
  restore on failure; neither installed CF nor production voxreg was edited.
- Validated notebook:5 executed code cells,5 figures, no error cells.
- Historical helper versions, constructor/CPU-to-CUDA observer harness failures,
  and unmasked renderings are retained separately. No optimizer defaults, masks,
  reference, truth/precision gates or production stopping behavior changed.
- No new smaller-rate optimization, fixed-support objective, line search, full
  suite or performance run. Prior full suite remains470 passed.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/atlas_centered_followup.py \
  --output artifacts/atlas-centered-followup/new-run
```
