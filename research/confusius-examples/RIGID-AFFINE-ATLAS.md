# Rigid → affine atlas comparison against ConfUSIus ground truth

**Subsequent transform-family audit:** the [executed notebooks](NOTEBOOKS.md)
show that installed CF uses an AffineTransform for matrix initialization even
when rigid is requested. Its first-stage matrix contains scale/shear. “CF rigid”
below therefore means the **requested stage**, not a genuine six-DOF result.
The frozen measurements/reference failures remain unchanged, but this is not a
like-family rigid-optimizer comparison. CF itself was not modified.

**Result: fail against the designated ground truth.** Both libraries accept their
own rigid result as the affine initializer, but voxreg's final placement remains
about **0.370 mm / 7.65 native pixels RMS** from ConfUSIus. The unchanged truth
limit is **0.25 px RMS**. CPU/CUDA numerical agreement passes separately.

ConfUSIus is ground truth **by operator instruction** for this experiment. A higher
native-slice correlation does not override disagreement with that reference.
This supersedes the acceptance criterion—not the frozen measurements—of the
previous affine-only atlas retry in [../slice-geometry/README.md](../slice-geometry/README.md).

## Procedure

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLBACKEND=Agg uv run python research/confusius-examples/rigid_affine_atlas.py \
  --output artifacts/rigid-affine-atlas/new-run
```

The output directory must not already exist. Recording, template and Allen labels
must already be cached. No downloads, engine changes, optimizer tuning, default
changes, speed measurements or sibling-repository edits.

- Same example03 recording: CR022/session20201007/slice02, mean of 755 frames then
  dB, original float32 native slice **1×125×80**.
- Same original example **3D slab**, approximately ±1 mm around manual placement:
  **20×91×116**, float64. It is not one extracted/frozen template plane. We did not
  replace the example's preprocessing with a different full-template representation.
- Both backends receive exactly the same original images and manual pull seed.
  Argument order stays **moving recording / fixed template slab** for both stages.
- Each backend fits rigid, copies its returned **full fixed-world → moving-world
  pull**, and passes it directly as affine initialization. No inversion at handoff,
  pre-resampling, extra matrix composition or frozen rigid prefix. The literal
  seed/output matrix equality is asserted and recorded for every backend.
- Installed ConfUSIus **0.8.0**, public register_volume: correlation, learning rate1,
  max500/window50, 8 SimpleITK threads, no multiresolution. Its exact installed
  source hashes are preserved, rather than treating the version string alone as
  an immutable implementation identifier.
- voxreg CPU8 or CUDA1 host threads: rigid6 then slice-affine9, learning rate0.1,
  max500/window50, intensity_scaling none because inputs are already dB. These are
  the existing method settings, not identical optimizers or internal objectives.

## Placement and labels

At every native recording pixel center, compare its 3D template placement under
`inverse(candidate_pull)` and `inverse(ConfUSIus_pull)`, using the corresponding
rigid or affine reference stage. This evaluates the actual slice embedding, not
arbitrary/unobservable full-volume normal extensions.

Errors are Euclidean template-world distances in mm, normalized by the smaller
native in-plane spacing **0.0483199991 mm** for pixel units. All 10,000 native
centers participate, including those outside the supplied template slab; changing
metric overlap cannot hide pose error. Independent SciPy double image sampling
and double NCC use a common template support of **91.02%**.

| Backend / stage | Updates | RMS from CF, mm | RMS, native px | Maximum, px | Truth gate | Common NCC² |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| CF rigid | 320 | 0 | 0 | 0 | Reference | 0.108877 |
| voxreg CPU rigid | 49 | 0.372554 | 7.710132 | 15.087103 | **Fail** | 0.059804 |
| voxreg CUDA rigid | 49 | 0.372554 | 7.710132 | 15.087103 | **Fail** | 0.059804 |
| CF affine, from CF rigid | 72 | 0 | 0 | 0 | Reference | 0.110256 |
| voxreg CPU affine, from voxreg rigid | 94 | 0.369850 | 7.654183 | 11.113469 | **Fail** | 0.155406 |
| voxreg CUDA affine, from voxreg rigid | 94 | 0.369850 | 7.654183 | 11.113468 | **Fail** | 0.155406 |

All stages stop through their native convergence checks, not the 500-update cap.
CF's internal slab objective and the shared native-slice NCC are recorded separately.
**The higher voxreg affine NCC is not a successful ground-truth result.**

For Allen labels, compose the example's template `world_to_sform` with the inverse
returned pull. Use the **same public ConfUSIus nearest-neighbor resample_like** for
all poses, with zero background fill. Thus label differences here do not compare
two different interpolation implementations.

| Candidate stage vs corresponding CF truth | Label agreement, entire image | Agreement on foreground union | Macro region Dice |
| --- | ---: | ---: | ---: |
| voxreg rigid, CPU/CUDA | 47.54% | 34.65% | 0.21711 |
| voxreg affine, CPU/CUDA | 56.22% | 45.40% | 0.21781 |

Foreground union includes pixels where either map has a nonzero region ID. The
entire-image number includes matching background; it is not the foreground result.
Per-region Dice values, all six matrices and all six affine/rigid seeds are retained
in [rigid-affine-atlas-results.json](rigid-affine-atlas-results.json).

## Separate precision check

Native-slice maximum pointwise CPU/CUDA disagreement:

- Rigid: **1.3082e-7 px** — passes 0.001 px.
- Affine: **2.9366e-6 px** — passes 0.001 px.

Agreement between devices does not imply agreement with ConfUSIus. No threshold
was relaxed, failed arm omitted, or higher correlation substituted for the truth gate.

## Evidence and checks

Ignored raw evidence: `artifacts/rigid-affine-atlas/current-reference-final/`:

- `results.json`, `fit-records.json`: poses, literal handoffs, convergence metadata,
  options, input/source/annotation hashes and failed gates;
- `arrays.npz`: original images, common support, six sampled images and six label maps;
- `template-overlays.png`: native recording cyan/template red, same display scale;
- `atlas-boundaries.png`: CF boundaries cyan/candidate magenta/shared white;
- `measured-source.py`: exact script snapshot matching the recorded source hash;
- `repeatability.json`: all **15 saved arrays** are bitwise identical across the two
  fresh quality runs. The first run completed fitting/rendering but hit a report
  serialization error; it remains preserved, not replaced. This is not a timing
  repeatability or an unrecorded-matrix repeatability claim.

Both figures were inspected. The label-boundary discrepancy is visible, especially
in deeper regions; CPU and CUDA overlays are indistinguishable at this scale.

Five new harness tests cover the full rigid-to-affine handoff for all three
backends, inverse-pull/native-pixel error units (including zero-origin grids), and
foreground/region label comparison. **49 focused tests pass**, covering this
harness, slice geometry and the prior public-workflow comparison helpers. Lint,
format and diff checks pass. The prior full422 engine qualification is reused;
no new full427 suite or package build is claimed for this research-only change.

### Cached label loading limitation

The installed BrainGlobe public constructor fails while parsing its cached
structures CSV with Pandas: `TypeError: cannot safely cast non-equivalent int64
to uint16`. Neither package nor cache was patched. Instead the script reads the
already-cached compressed uint32 annotation array at 100um/v3.1, views labels as
int32, and constructs exactly the resolution-diagonal, zero-origin geometry used
by current ConfUSIus's atlas builder. Shape, orientation, resolution and array hash
are checked/recorded. NGFF pyramid translations are not newly substituted for CF's
chosen normalized atlas-world convention. No meshes, structures lookup or atlas
cache export is performed.

## Next diagnostic, not an implemented fix

The disagreement already appears at the first requested stage; chaining affine
does not close it. The subsequent transform-family audit above must be accounted
for before the previously proposed controlled-CF-seed diagnostic: CF's returned
matrix is not a valid genuine rigid initializer, and mixed-affine normal-completion
compatibility also needs checking. Do not attribute the gap to rigid capture/stopping
alone or tune for a higher native-slice NCC and call that agreement. This experiment
does not establish the cause and changes no production behavior.
