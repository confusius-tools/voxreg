# Executed visual comparisons of all three examples

Open **`artifacts/example-notebooks/comparison/index.html`** in a browser.

The follow-up [zero-update atlas control](ATLAS-SEED-CONTROL.md) independently
compares initial placement before registration. Its zero-thickness error is now
[corrected and requalified](../slice-geometry/THICKNESS.md); that page links the
fresh zero-update and fitted-atlas HTML reports. The original exports here remain
at their measured pre-fix boundary. The original “affine” panels below
already contain500-update-budget fits, not just application of the manual matrix.
Each HTML page embeds its figures and tables; no server, Python, network or
external image assets are needed to view it. Expand the code blocks to inspect
the actual executed calls. Executed `.ipynb` files are provided alongside HTML.

| Comparison | HTML | Executed cells | Figures |
| --- | --- | ---: | ---: |
| Same-subject sessions | `01-same-subject.html` | 6 | 6 |
| 120-frame motion correction | `02-motion-correction.html` | 5 | 7 |
| Allen atlas, original and chained workflows | `03-allen-atlas.html` | 7 | 9 |

All **18 code cells executed successfully**, producing **22 actual figures**.
Standalone PNGs, original/registered arrays, motion CSVs and provenance are also
saved in that directory. Images are embedded in HTML/ipynb, so copying only an
HTML file still retains its plots. The index links the three files locally.

## What each page contains

1. **Same-subject registration:** original cached rat75 sessions, qform geometry,
   dB images, initial/CF/voxreg rigid overlays, real CF convergence history and
   voxreg's actual final metric. The original CF B-spline stage is executed too,
   including its convergence curve and approximate inverse combined/local
   displacement plots. Voxreg has no B-spline equivalent; it is explicitly marked
   unsupported, not replaced with affine. The native100-update voxreg rigid result
   still under-captures; this is shown, not tuned away.
2. **Motion correction:** all120 original frames220–339, original power values,
   current source example's CF learning_rate0.1, both motion/FD diagnostics with
   matched axes, the highest-raw-std voxel's power trace, matched-scale rasters,
   initial worst/median frame overlays and final-image QC. Both sets of poses pass
   through the same public CF motion/FD helper. Reference frame excluded from QC
   summaries. This is a fresh current workflow, not the old learning_rate1 timing
   archive or historical float32 input.
3. **Atlas registration:** original manual placement and3D template slab, original
   affine-only calls **and** each backend's own rigid→affine chain. Template/recording
   composites use red/cyan, and actual Allen region maps use **official CSV RGB
   colors**, overlaid at alpha0.45 on the native recording. Label-only views and a
   colored region-name legend accompany them. CF-reference placement and label
   failures remain visible; higher NCC does not count as passing.

Display ranges are matched within comparison figures. Motion diagnostic axes are
matched across libraries. CF displacement arrows retain the original approximate
inverse/auto-scaled-arrow convention, explicitly labeled. Native stopping policies
are retained and described, not represented as identical optimizers. Voxreg does
not expose per-iteration metric histories, so its curve is **not fabricated**.

## Important discovery: CF's matrix-seeded "rigid" request

The atlas notebook now audits the actual returned linear matrices. With a matrix
initializer, installed CF0.8.0 uses `affine_to_sitk_linear_transform(initialization)`
as its optimized initial transform, regardless of the requested linear model.
That helper creates a SimpleITK **AffineTransform**. Thus the call requested as
`transform_type="rigid"` in this atlas chain actually returns scale/shear:

| Requested rigid stage | ‖LᵀL−I‖ | Singular values | Rigid matrix? |
| --- | ---: | --- | --- |
| CF, matrix seed | 0.198727 | 1.000520 /0.983321 /0.896688 | **No** |
| voxreg, matrix seed | 3.69e-17 | 1 /1 /1 | Yes |

This is an **affine-valued linear result**, not a B-spline deformation. The original
CF affine-only atlas call and its first chained "rigid" call consequently give the
same pose here. The direct original affine workflow remains a valid like-family
comparison; the added requested-rigid chain is **not** a six-DOF-versus-six-DOF
comparison. Same-subject/motion CF calls use named geometry initialization and do
not take this matrix-seed path.

CF outputs remain the operator-designated ground truth. The earlier failed
reference-placement/label checks are still failures; their values are not changed.
However, they cannot be attributed simply to two different rigid optimizers.
This notebook/report corrects the interpretation of the earlier chain labels.
CF is read-only: no package or sibling-repository correction is included.

## Reproduce

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/render_notebooks.py \
  --output artifacts/example-notebooks/new-comparison
```

The destination must not already exist. Use the project Python environment and
run from the repository root. Cached original data and Allen100um/v3.1 labels and
terminology CSV are required; no implicit download. No Jupyter/nbconvert dependency
was added: the small exporter executes trusted code cells, captures stdout/tables
and real Matplotlib figures, then writes nbformat4 JSON and self-contained HTML.
Failed cells are preserved as notebook errors and raised, not hidden as successes.
To re-execute an `.ipynb`, select the project environment and check its `OUT` path.

`manifest.json` and the tracked [notebook-results.json](notebook-results.json)
record summaries, device/library versions, input hashes and source fingerprints.
The files were rendered with CF0.8.0, Torch2.14.1+cu130, NumPy2.5.4, CUDA.
Six new tests cover real execution/error outputs, offline PNG embedding, official
colors with large Allen IDs, background-preserving alpha overlays, red/cyan roles,
and refusal to overwrite an existing destination. **55 focused tests passed**
(the six new tests plus the existing atlas-chain, slice/volume and benchmark tests).
All three actual notebooks executed end-to-end; their code counts, PNG signatures, absence
of cell errors and local HTML links were additionally checked. No engine changes,
new performance benchmarks or fresh full engine suite are claimed.

## Explicit omissions

- No voxreg B-spline implementation or invented nonrigid equivalent.
- No full Atlas structures/mesh save/reload. The installed BrainGlobe/Pandas CSV
  parser fails on UInt16 conversion; cached labels and official CSV colors are read
  directly, using the same geometry as CF's current builder. Native label maps and
  fitted matrices are saved as NPZ instead of pretending to export a full Atlas.
- No default changes, convergence tuning, GUI/napari session or timing claims.
