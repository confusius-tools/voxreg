# Centered CF without through-plane collapse

**The constraint works, but alignment still fails the reference gate.**

Open **`artifacts/atlas-rigid-centered/final/rigid.html`** for actual metric traces,
returned-pose overlays and best-metric overlays. Red=template, cyan=recording;
all sampled on the same native recording grid with the same −30…0dB display.
Machine-readable states and provenance: [atlas-rigid-centered-results.json](atlas-rigid-centered-results.json).

**Display follow-up:** historical overlays above extrapolate outside template
support. Use the support-masked panels and1000-step continuation in
[ATLAS-CENTERED-FOLLOWUP.md](ATLAS-CENTERED-FOLLOWUP.md) when assessing alignment;
unsupported pixels are gray. Fit matrices and measured objectives are unchanged.

## Controls

Three final arms, original cached images/manual placement, fixed-image physical
centering, correlation, rate1, max500/window50,8threads, unchanged slice expansion:

1. **Genuine rigid, recording moving:** Euler3DTransform,6 verified parameters.
2. **Genuine rigid, template moving:** Euler3DTransform,6 verified parameters.
3. **Tilt + in-plane affine, template moving:**
   ComposeScaleSkewVersor3DTransform,12 parameters with9 allowed to update.

The third arm uses the recording's axial world-z normal as its first input axis:
`A = R @ diag(1,sy,sx) @ U`, with `U[1,2]` the only free upper-triangular shear.
The first column is a **unit normal perpendicular to both mapped plane tangents**.
Rotation/tilt, translation, both in-plane scales and in-plane shear remain free.
This is not freezing a row/column of the world affine. Unmeasured through-plane
shear is not estimated. This recording-input constraint was only tested in the
recording-fixed direction; naively applying it to the opposite pull would not
impose the same native-slice geometry.

CF's matrix-seed path otherwise substitutes an affine even for requested rigid.
Scoped research setup selects the actual ITK class and preserves the full seed
while centering. The composed transform also needs a scoped CF matrix exporter
and a workaround for this SimpleITK build's missing Compose downcast entry.
**No installed sources were edited; all patches restore on failure.** This is
not currently a selectable constrained model in CF's public API. Its
`optimizer_weights` ordering here is specific to the composed transform, not the
ordinary affine matrix coefficients.

## Results

RMS values measure disagreement with the original operator-designated good CF
recording-moving affine, not independently measured anatomical error. Absolute
metrics from opposite directions have different sampling domains.

| Model / moving image | Updates | Best NCC² (state) | Returned NCC² | Returned RMS px vs good original affine |
| --- | ---: | ---: | ---: | ---: |
| True rigid / recording | 67 | .117828 (2) | .114765 | 5.010 |
| True rigid / template | 113 | .127309 (99) | .116745 | 18.025 |
| Unit-normal in-plane affine / template | 106 | **.160630 (89)** | **.146172** | **13.717** |

The last arm's best state still differs by **12.100px**, above the unchanged
0.25px reference gate. Its mapped recording center moves from the manual
**5.994639mm** depth to **6.536975mm** at best / **6.603950mm** returned, with
**4.756° / 6.415°** tilt. The artery/cortex mismatch remains visible in both
best and returned overlays. Retaining the best metric alone would not fix it.

Unlike the previous centered full affine, there is **no hidden normal collapse**:

- All rigid-state singular values differ from1 by at most6.7e-16.
- Every constrained-affine state has unit normal length (error≤1.2e-16), normal
  perpendicular to tangents (dot magnitude≤2.5e-17), and positive determinant.
- Best constrained-affine singular values: **[1.171486,1.011394,1.000000]**;
  returned: **[1.170829,1.008010,1.000000]**. Nonunit scales are in-plane.

Thus compression was a serious failure mode, **not the only explanation** for
wrong reverse-direction placement. With a noncollapsing map, the current setup
still drifts in plane depth/orientation and stops after worsening. Rigid also
removes legitimate in-plane deformation, so its weaker score is not evidence
that tilt should be frozen. Euler and versor rotation parameterizations differ;
these are model/parameterization controls, not identical gradient trajectories.
No rate tuning, no-stop continuation or claim of optimizer optimality is made.

## Verification

**13 focused tests pass**, including8 new seed geometry, true CF execution,
normal/thickness geometry and exception-restoration cases. Each real arm checks
actual transform class/parameter count at every event, pairs pre-step metric with
its matching pose, independently re-evaluates seed/peak/last states, and validates
geometry throughout the trajectory. Both rigid matrices/histories reproduce the
earlier two-arm run bitwise. Inputs, CF source hashes and all production engine
sources are unchanged. The prior470-test full suite was not rerun.

The final notebook has4 executed code cells,3 figures and no error cells. Earlier
unit-harness failures and the missing-downcast export failure remain archived;
no final results were overwritten. The abandoned world-row/column-freezing
proposal was **not run**.

Centering issue: <https://github.com/confusius-tools/confusius/issues/537>.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/atlas_rigid_centered.py \
  --output artifacts/atlas-rigid-centered/new-run
```

No production API, optimizer defaults, acceptance gates or reference were changed.
