# Why the atlas alignment differs: direct ITK and matched-freedom controls

Open **`artifacts/atlas-investigation/native-itk/investigation.html`** for the
executed tables, actual metric histories and identical-sampler overlays. Matrices,
raw histories, PNGs, notebook and source/input fingerprints are alongside it.
[atlas-alignment-results.json](atlas-alignment-results.json) retains the measurements.

## Main finding

**The original CF affine fit is effectively anchored to the manually supplied
recording plane. Voxreg's native-slice fit is allowed to change that plane.**
There is also an optimizer/gradient/capture difference after fixing the plane.
This is not explained by a simple world-axis inversion, CUDA error or inability of
the9-parameter model to represent CF's native placement.

This conclusion applies to the actual **matrix-seeded affine experiment**, not all
possible CF transform families or future acquisitions.

## 1. Direct native ITK reproduces CF

All controls use the same original template slab, recording mean, dB data, physical
manual seed and500-update budget. CF-style controls use rate1, window50, physical
shift parameter scales and one unsmoothed resolution level. No downloads/tuning.

| Optimizer control | Updates | RMS discrepancy from saved CF, native pixels |
| --- | ---: | ---: |
| Direct SimpleITK, without CF registration wrapper | 320 | **0; matrix bitwise identical** |
| Direct Python ITKv4 `ImageRegistrationMethodv4` / `GradientDescentOptimizerv4` | 320 | **2.27e-14** |

Thus the designated CF result is reproduced by ITK directly. Its registration
wrapper does not account for the voxreg discrepancy. Image conversion helpers
are reused to avoid introducing a new axis convention; the native ITKv4 optimizer,
metric and physical-shift scales are configured explicitly in the research code.

## 2. Why CF cannot update depth and tilt in this affine setup

For optimization, CF expands the single recording voxel to four identical samples
through the slice, **preserving its0.4mm voxel-cell extent** rather than creating a
1.6mm slab. The moving intensity field is constant in its first physical coordinate
(world z in our convention). Its through-slice gradient is essentially zero.

An independent **native ITKv4 derivative probe** at the manual seed measures:

- Maximum derivative for the affine first row and its translation: **1.49e-19**.
- Maximum derivative among all parameters: **0.05675**.
- With the real3D template as moving image instead: through-plane parameter
  derivative maximum **0.04843**, not zero.

The first row of CF's returned world pull remains `[1,0,0,−5.5946386564]`, up to
~1e-18 numerical noise. The implicit recording-plane equation therefore stays at
the manual depth/tilt. Other affine rows can change, including in-plane alignment
and off-plane shears that affect the finite-slab metric.

By contrast, voxreg samples the varying3D template at native recording centers:

| Native recording placement | World-z center, mm | Tilt from manual plane |
| --- | ---: | ---: |
| Manual / original CF | **5.994639** | **0°** |
| voxreg native retained pose | **6.216296** | **3.639°** |
| voxreg fixed500/1000 retained pose | **6.563191** | **5.536°** |

This directly explains a substantial part of the apparent residual rotation and
depth shift. CF's attractive result is conditional on the supplied plane, not
independent evidence that gradient descent recovered that plane from the images.
Neither voxel spacing nor expansion establishes the acquisition's actual PSF.

As an intervention, starting CF at the voxreg pose keeps that *new* plane equation
fixed too. It refines in-plane but remains5.73px from original CF: it does not
recover the original plane.

## 3. Matching plane freedoms helps, but does not close the gap

A scoped **research-only gradient hook** freezes the two out-of-plane tangent
components and the through-plane translation in voxreg's9-parameter native map.
All other fitting arithmetic/settings remain unchanged. It is not a product API
or a proposed silent restriction on3D registration.

| voxreg control | RMS pixels vs CF | Best native NCC² |
| --- | ---: | ---: |
| Native, manual seed | 6.217505 | 0.153627 |
| Fix depth/tilt, manual seed | **3.037602** | 0.132117 |
| Same frozen plane, no stopping /500 updates | **3.037602** | 0.132117 |
| CF-native placement seed, free plane | **0.810294** | 0.143007 |
| CF-native placement seed, frozen plane | **0.440597** | 0.141409 |

Fixing the plane halves the RMS disagreement; it does not reach the0.25px gate.
The same frozen-plane optimizer reaches a better metric from the CF placement than
from the manual seed, even when the latter runs500 updates. This shows capture/
trajectory dependence in addition to differing geometric freedoms. It is not
just "needs more iterations".

The CF seed is projected to a unit-orthogonal normal completion **without moving
any of the10000 native centers**. Production zero-update decoding/export verifies
its native placement and metric. Hence nominal "CF12 versus voxreg9 parameters"
is **not an inability to express the CF native pose**; normal-gauge differences
can still affect off-plane/slab samples.

## 4. ITK's image-gradient construction materially affects the trajectory

Keeping CF's images/domain/seed/GD optimizer/scales/rate otherwise fixed, disabling
its recursive-Gaussian moving-image gradient produces **30.89px** disagreement
from CF instead of the matching result. The curve becomes strongly oscillatory.

This is a causal one-setting control: gradient construction matters. It is not
proof that adding one blur would fix voxreg. ITK's unsmoothed gradient is also not
identical to autograd through trilinear interpolation; its magnitude/direction
changes the effective GD steps at the same configured rate. Adam, parameter
scaling and stopping/capture remain separate differences.

Direct ITK on the unexpanded native-slice domain, with the3D template moving,
also drifts from original CF (18.86px with smoothed moving gradients,29.22px without).
Those rate1 controls are not a tuned comparison or a limit on ITK accuracy; they
show that swapping the backend while allowing plane motion does not inherently
recover CF's plane-anchored answer.

## 5. Numerical and metric checks do not show a basic sampler/backpropagation bug

At the same native placements, explicit center-masked ITK correlation and voxreg's
actual held-pose objective agree within **4.7e-8 NCC²** across five compatible poses.
Unmasked native ITK includes half-voxel borders, so it is reported separately.

At both manual and projected-CF seeds, all9 normalized-coordinate gradients were
checked against float64-interpolation central differences with fixed support:

- Double autograd versus finite difference: **<=3.3e-9 relative L2 error**.
- Actual mixed-precision gradient versus double interpolation: **<=1.7e-7**.

These are local checks at two real poses, not a proof over every interpolation
knot or changing-support boundary. An exploratory1e-5 finite-difference step was
too coarse at the CF pose; the preserved failed check disagreed by2.15e-5 relative.
The final1e-6 step gives the local agreement above. Failed exploratory notebook
outputs and the initially over-specific tangent-basis guard are retained rather
than overwritten; the final report has no failed cells.

## 6. Even CF's own score prefers the voxreg candidates

Evaluating the **same saved matrices**, not running different optimizers:

| Candidate pose | CF original-domain NCC² | Native voxreg NCC² |
| --- | ---: | ---: |
| CF | **0.160658** | **0.139352**, equivalent projected native pose |
| voxreg native | **0.166578** | **0.153627** |
| voxreg longer-run best | **0.167836** | **0.163318** |

Each objective uses its own ordinary pose-dependent support. This is not a common
support anatomical ranking. Crucially, the explanation cannot simply be
"voxreg's different metric likes its own pose, whereas CF's metric prefers CF."
CF's metric also scores the voxreg poses higher. Its returned result reflects a
constrained search and gradient/optimizer trajectory, not a unique global NCC
optimum. Maximizing NCC alone will not validate anatomical correspondence.

## Consequence / next step

Do **not** silently remove reversal, freeze3D motion, increase iteration budgets,
or call the higher-score fit successful. The controlled next step is to separate:

1. **CF-equivalent, plane-preserving affine alignment**, including the finite-slab
   domain and gradient/update construction, for reference parity.
2. **Actual3D plane refinement**, which needs independent placement evidence or
   a justified acquisition/sampling model; CF's frozen plane cannot validate it.

The residual in-plane gap still needs a matched gradient/optimizer/scaling
ablation. This investigation identifies concrete causes and falsifies several
simple alternatives; it does not yet provide a passing voxreg correction.

## Verification and reproduction

- **11 fresh optimizer fits**, three native ITK derivative probes, two real-pose
  finite-difference checks and28 cross-objective evaluations.
- **24 focused tests pass**, including three new projection/gradient helper cases.
  The prior470-test full suite was not rerun for this research-only change.
- Production source hashes are unchanged, including the fitter/sampling kernels.
  No new dependencies, default-policy changes or timing claims.
- FireANTs was **not run**. The matched direct ITK controls already isolate these
  differences; another engine's default loss/optimizer would confound them.

Requires the prior direction/convergence JSON reference artifacts; their hashes
and original input/source fingerprints are checked/retained.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/atlas_alignment_investigation.py \
  --output artifacts/atlas-investigation/new-run
```
