# Atlas initial-placement control: zero optimizer updates

**Historical pre-fix measurements.** The zero-thickness error identified here is
now corrected: see [the finite-extent fix](../slice-geometry/THICKNESS.md) and its
fresh zero-update/atlas-fit reports. The counts below deliberately retain the
original failing behavior, not current output support.

Open `artifacts/example-notebooks/atlas-seed-control-final/initial-placement.html`.
The executed notebook, five embedded/standalone figures, masks and arrays are
alongside it. [Recorded results](atlas-seed-results.json) retain source/input
fingerprints, dtypes, matrices' mapping errors and every support comparison.

## Question and scope

Does the supplied manual affine already place the recording differently in CF
and voxreg, before either optimizer runs?

The previous notebook's **"original affine" panels are after registration**, not
initial placement: both calls run up to500 updates. "Original" refers to the
original example workflow. Its initial manual panel was a shared SciPy sampling
helper, not an independent CF-versus-voxreg initialization control.

This experiment uses exactly the same cached slab/recording, preprocessing and
manual matrix. Both directions are tested with CF public `resample_like` and
voxreg CPU/CUDA. No registration, rate changes or fitted-pose substitution.

Voxreg currently has no public resample-only API and rejects max_iterations0.
A scoped **research-only optimizer bypass** therefore returns the decoded seed
parameters unchanged and zero updates, exercising production `register` preparation,
final sampling, dtype conversion and export. The optimizer function is restored on
exit. Its artificial status must not be interpreted as a convergence result.
Neither the voxreg engine nor ConfUSIus is modified.

## Results

**The transform is not inverted, composed or placed incorrectly.** All four
CPU/CUDA/direction cases preserve the supplied4×4 matrix exactly. Maximum mapping
error on the actual target grid is **0mm /0 source pixels**. Zero optimizer updates
are asserted, as is unchanged input matrix and output dtype.

However, **the whole output arrays do not match**, because sampling support differs.

### Template sampled on native recording: same interior placement

This is the useful view used for the report's template/recording overlays.

| Quantity | CPU | CUDA |
| --- | ---: | ---: |
| Native centers | 10000 | 10000 |
| CF-supported centers | 9680 | 9680 |
| voxreg-supported/common centers | 9600 | 9600 |
| Maximum difference on common support (dB) | 6.04e-14 | 6.04e-14 |
| RMS difference on common support (dB) | 8.97e-15 | 9.19e-15 |
| Maximum voxreg–SciPy difference there (dB) | 4.53e-14 | 4.53e-14 |

There is a real **80-pixel boundary difference**, the entire native row `j=4`.
CF includes the half-voxel boundary outside the first template sample center;
voxreg requires coordinates within the sample-center bounds (plus1e-5-pixel
numerical tolerance). CF interpolates/clamps that row, whereas voxreg fills it.
The whole-output maximum difference is7.17dB. Do not describe the entire image as
identical or silently hide this row behind a common-support score.

The interior template overlays nevertheless coincide to double precision. This
rules out a wrong manual placement/pull conversion as the explanation for the
large post-registration native-slice pose discrepancy.

### Recording sampled on original3D template grid: different slice extent

| Quantity | CF | voxreg CPU/CUDA |
| --- | ---: | ---: |
| Supported template centers | 16936 | 0 |
| Nonempty template planes | 4 | 0 |
| Whole-output maximum difference | — | 17.88dB |

The placed recording plane is at **z=5.9946386624mm**. The closest template plane
is at **z=6.0295630544mm**, offset **0.0349243921mm**. No template center lies on
the exact recording plane.

CF gives the singleton recording voxel finite support, using its header spacing
of approximately0.4mm:16936 centers across four template planes are valid. Voxreg
models this measured slice as zero thickness, so it returns fill everywhere on
this volume grid and warns. No common-support intensity error is defined here;
it is recorded as null, not incorrectly reported as zero agreement error.

This is a **real CF-output compatibility gap**, not evidence that the supplied
affine was decoded incorrectly. It is not excused as a reference-acceptance pass.

## Interpretation and next boundary

- Initial interior placement is correct on both devices. The later visibly
  divergent native-slice poses arise after optimization, not from a wrong initial
  matrix or an inverse applied twice.
- The previous post-fit template overlays use the same SciPy sampler for both
  fitted poses; the Allen-colored overlays use the same CF nearest-neighbor
  resampler for both. Their visible interior differences therefore reflect different
  fitted matrices, not different plotting/interpolation backends. They do not display
  the fill-only original-direction voxreg volume.
- Border and singleton-slice support semantics already differ with zero updates.
  These policies also differ in fitting: CF expands thin dimensions while preserving
  physical voxel extent; voxreg evaluates a native-slice metric. Support/domain
  differences may contribute to different fitted optima; this control does not
  isolate their causal contribution.
- CF's matrix-seeded requested-rigid stage is additionally affine-valued, as audited
  in [NOTEBOOKS.md](NOTEBOOKS.md). That is separate from this affine seed test.
- No optimizer tuning, support-policy change or engine fix is made. CF remains the
  designated reference; the previous failed fitted-pose gates remain failures.
  Next compare objectives/updates with explicitly controlled support/domain rather
  than treating this as an initial-transform bug or changing learning rate blindly.

## Reproduce and checks

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/atlas_seed_control.py \
  --output artifacts/example-notebooks/new-atlas-seed-control
```

Destination must be new. No downloads or package patches. Both CPU/CUDA and both
sampling directions executed; all four notebook cells completed. Four new tests
cover zero-step sampling/optimizer restoration on both devices, the independent
CF half-voxel-border behavior, and undefined error on empty common support.
**56 focused tests passed** including notebook, atlas-chain and slice/volume tests.
Two fresh controls produced all14 saved arrays bitwise identically; the earlier
run is retained separately. Source fingerprints, embedded PNGs and local HTML links
were checked, and the final scoped-bypass tests were rerun (4 passed).
No fresh full engine suite, timing benchmark or anatomical-accuracy pass is claimed.
