# CF affine centering in both registration directions

Open **`artifacts/atlas-centering/comparison/centering.html`** for the executed
curves,2×2 returned-pose overlays and2×2 best-metric overlays.
[atlas-centering-results.json](atlas-centering-results.json) contains measured
states/metrics, source fingerprints and summary statistics; full parameter
trajectories and sampled images are saved in the artifact NPZ.

**Display follow-up:** these historical overlays use nearest-border extrapolation
outside the template. Support-masked comparisons are now available in
[ATLAS-CENTERED-FOLLOWUP.md](ATLAS-CENTERED-FOLLOWUP.md); fit matrices and numerical
results are unchanged. Do not interpret extrapolated regions as matched anatomy.

## Controlled change

Four actual CF affine/correlation fits, all rate1, max500/window50,8threads,
original images/dB preprocessing/manual placement and existing slice expansion.
Only the transform center changes:

- Recording moving / template fixed: origin versus **template slab center**
  `[5.979563,5.219548,5.729273]mm` in fixed world z/y/x.
- Template moving / recording fixed: origin versus **recording center**
  `[0.400000,5.991680,0.000000]mm` in fixed world z/y/x.

This is ITK's fixed-image geometric-centering convention. It is not the only valid
pivot or a claim that it is optimal for an anatomical ROI. No extra rate tuning,
plane constraint or production code change was introduced.

ITK applies `A(x-c)+c+t`. For the same supplied world affine `(A,b)`, changing the
center to `c` uses **`t=b+(A-I)c`**. The converter asserts the full initial matrix
is preserved; initial metrics match within1e-12. Unit tests also verify a
nonidentity affine on explicit world points, not only this translation-only seed.

## Results

Errors below are relative to the original good, recording-moving CF result, not
independent anatomical truth. Native pixels follow the existing minimum-spacing
convention. Absolute losses across the two directions are not directly comparable.

| Moving image / pivot | Updates | Returned NCC² | Best NCC² (state) | Returned RMS px vs original good CF |
| --- | ---: | ---: | ---: | ---: |
| Recording / origin | 320 | 0.160658 | 0.161180 (298) | 0 |
| Recording / template center | **135** | 0.158713 | 0.159252 (114) | **0.410089** |
| Template / origin | 127 | 0.125246 | 0.131866 (102) | 17.231877 |
| Template / recording center | **160** | 0.145032 | 0.169674 (146) | **14.939912** |

**Centering materially changes conditioning/trajectory, but does not recover
inverse-consistent or anatomically comparable results.**

The recording-moving result looks similar with either pivot; centering uses fewer
updates but has a slightly worse metric and differs0.41px from the original
reference, outside the0.25px gate. No wall-time speed claim is made.

The template-moving result improves its own metric but still shows the inferior
artery/cortex mismatch. Its curve reaches a deeper minimum, then worsens sharply.
Saving the best state does not solve the anatomical problem: that state remains
14.33px from the original good reference.

## A new warning: affine compression

For the template-moving, recording-centered fit:

- Best-metric linear singular values: **[1.175624,0.995780,0.032061]**.
- Returned linear singular values: **[1.177144,0.993340,0.019142]**.
- Returned determinant: **0.022382**, still positive but strongly compressive.

The weakest input singular direction aligns almost exactly with recording world z
(absolute alignment0.9999999 at the returned pose). This is about98% compression
along an approximately through-plane direction, not a modest rotation about a
better pivot. It exposes a weakly constrained affine mode in the expanded2D-data
objective. The normal extent of the represented voxel cell is not a measured PSF.

Thus the higher metric is **not evidence of a successful correction**. A sensible
center changes the path; it does not constrain affine scale/shear or resolve
single-slice depth ambiguity. The next causal control should address these
through-plane affine freedoms rather than silently accept the higher score.

## Measurement and checks

- Uses actual CF calls with scoped seed-converter/observer patches; installed CF
  and production voxreg sources are not edited. Patches are restored on failure.
- Both origin controls reproduce saved matrices and recorded curves bitwise.
- Observer metrics and parameter snapshots are paired correctly: this installed
  build emits a pre-update metric alongside a post-update position. Selected
  paired states are independently re-evaluated; final returned metric is appended.
- **11 focused tests pass**, including3 new center/mapping/restoration tests.
  The prior470-test full suite was not rerun for these research-only changes.
- Final report:4 executed code cells,3 real figures, no failed cells. Earlier
  check/export failures (overly strict scalar equality and notebook string
  escaping) remain in separate artifact directories; they are not fitter failures.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/atlas_centering.py \
  --output artifacts/atlas-centering/new-run
```

No default center, optimizer, transform model or acceptance gate was changed.
