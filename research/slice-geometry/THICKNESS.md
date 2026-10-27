# Fix: finite extent for a singleton slice voxel

**Historical output-only fix.** The finite-cell principle remains; current mixed
registration also fits on the caller's original fixed grid, with half-voxel source
support and no replication. See [NATIVE-FIXED.md](NATIVE-FIXED.md).

**The zero-thickness output assumption was wrong.** A k=1 image still has a
finite voxel cell defined by its affine. The mixed slice→volume resampler now
uses that extent rather than requiring output centers exactly on the slice plane.

## Smallest responsible change

`src/voxreg/sampling.py::sample_inverse_slice` computes source voxel coordinates
through the full physical affine and returned transform. For a singleton axis,
valid coordinates are now **`[-0.5, 0.5)`**: lower face inclusive, upper face
exclusive, matching SimpleITK. Existing border interpolation gives constant
intensity along that axis without storing replicated slices or inventing
independent measurements. Source dtype and actual affine spacing are retained.

This is voxel-cell extent, **not a claim about the acquisition's measured slice
point-spread function**. With an orthogonal0.4mm k-axis, extent is approximately
±0.2mm about the center; rotated/reversed lattices use the full inverse affine,
not a hard-coded world-axis distance.

Active-axis support still uses the existing sample-center convention plus its
numerical tolerance. The qualified default2D kernel, optimizer, initialization,
stopping rules and native-slice fitting metric are unchanged. Empty output is
still possible if no output centers intersect the finite transformed voxel
extent, but warnings no longer describe a zero-thickness slice.

## Regression proof

`tests/test_slice_thickness.py` failed on the old kernel before the edit. It now
passes all16 CPU/CUDA × float32/float64 × linear/nearest × axial/oblique cases,
comparing values and validity directly with CF public `resample_like`.

Coverage includes:

- Exact lower/upper slab faces and points on both sides of the slab.
- Different singleton-axis spacings, translated/rotated poses.
- Reversed/strided active-axis grids and noncontiguous input views.
- Original moving dtype and outside-support fill.

The existing mixed-direction tests use an independent SciPy/full-affine support
oracle and no longer expect an erroneous empty output warning.

**453 full tests passed** (1092 warnings,77.33s), including the16 new regressions.
The focused thickness/slice-volume run passed57 tests. All four frozen120-frame
CPU8/batch32 and CUDA1/batch120 fixed100/native-convergence controls remain
**bitwise identical** for images, validity, matrices, status and update counts.

The mixed synthetic qualification was rerun with unchanged gates: six truth
checks pass, worst RMS **7.55e-7px** against the0.25px limit; three CPU/CUDA
precision checks pass, worst pointwise error **5.23e-7px** against0.001px.
`uv lock --check`, lint, format, diff checks, source/wheel builds and an isolated
wheel affine9/float64 finite-slab smoke pass. The smoke returns nine valid volume
centers for a slice plane between volume planes (the old mask would be empty).

## Real cached atlas: corrected initial volume output

Open **`artifacts/slice-thickness/atlas-seed/initial-placement.html`** for the
executed zero-update comparison and five embedded figures. Same cached inputs,
manual pull and display scales; both CPU/CUDA preserve the matrix exactly.

| Original recording→template-grid output | Before fix | After fix | CF |
| --- | ---: | ---: | ---: |
| Valid centers | 0 | **16472** | 16936 |
| Nonempty template planes | 0 | **4** | 4 |
| Empty-output warning | Yes | **No** | — |

The remaining464-center mismatch is at **active-axis borders**, where CF includes
half-voxel margins and voxreg retains center bounds. This fix does not claim
whole-output CF parity. On shared support, maximum CF/voxreg differences are
2.77e-5dB CPU /2.86e-5dB CUDA for the original float32 recording. The native
volume-on-recording view remains unchanged, agreeing on9600 common centers to
6.04e-14dB, with the previously recorded80-pixel active-border difference.

## Real atlas fits: output fixed, registration disagreement remains

Open **`artifacts/slice-thickness/atlas-notebook/03-allen-atlas.html`** for the
fresh affine-only and chained atlas fits, including official Allen-colored labels.
All14 saved image/label/matrix arrays in that native-view report are bitwise
identical to the pre-fix notebook: this patch changes reverse output support,
not the fitted poses or native-center metric.

**The CF-reference placement gate still fails:** original affine6.218px RMS;
chained affine7.654px RMS, versus0.25px allowed. Fixing slice extent alone is
not an atlas optimizer/objective correction. CF's finite-slab fitting domain and
matrix-seeded requested-rigid transform-family difference remain separate issues;
no learning-rate, model or convergence tuning is slipped into this change.

## Evidence and reproduction

[thickness-results.json](thickness-results.json) binds source fingerprints,
synthetic gates, frozen-motion comparisons, zero-update atlas measurements and
post-fit atlas scores. Raw logs, arrays, measured source and packages are under
`artifacts/slice-thickness/`. Pre-fix reports/results remain unchanged and labeled
historical, not rewritten to look successful.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run pytest -q tests/test_slice_thickness.py tests/test_slice_volume.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/atlas_seed_control.py \
  --output artifacts/slice-thickness/new-atlas-seed
```

No CF source/cache changes, new dependency, slice-data replication, extra fit
parameters, acquisition-specific thickness/PSF option or performance claim.
