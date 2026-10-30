# ConfUSIus examples: current exploration and historical validation

**Start here:** [native-pair live/headless exploration](EXPLORATION.md), using
`explore_atlas.py` and the reusable research-only `plot_registration.py` helper.
Optimizer defaults are unchanged. Library comparisons below are diagnostics, not
an ITK/FireANTs output-equivalence product requirement or anatomical ground truth.

## Quick Landemard motion QC

[`compare_landemard_motion.py`](compare_landemard_motion.py) fetches ALD001/ref20
through ConfUSIus and compares raw, rigid, rigid→affine and rigid→affine→SyN on
the same original recording and first selected reference frame. Default: **the
whole recording**, 100 updates per stage, CUDA if available. First download is
about 0.5 GB. There are **exactly three whole-recording `voxreg.register` calls**,
one per method; batching happens inside voxreg. Every call uses the original
images and the same first-frame reference. The comparison deliberately reuses
rigid/affine results as the next method's optional seed; ordinary motion correction
needs just `result = voxreg.register(pwd)` for the default rigid model.

```sh
uv run python research/confusius-examples/compare_landemard_motion.py \
  --output artifacts/landemard-ref20-full
# Optional short window: --start 1000 --frames 120
```

Writes `rigid.nii`, `affine.nii`, `syn.nii`, their ConfUSIus geometry/timing JSON
sidecars, `qc.png`, full-resolution `qc.csv` and settings/status/timing JSON.
**No GIFs or dense fields.** NIfTIs retain original power intensities and dtype;
invalid samples use the registration engine's per-frame-minimum fill. Three full
corrected recordings occupy about 1.6 GB uncompressed, in addition to the cached
input. Each method is exported and released before proceeding. There is no
script-level registration chunking or output scratch. Linear matrices are
explicitly requested for stage seeds; **SyN transform export is disabled** (as it
is by default in the library), avoiding the full-recording 6.3 GB dense fields.
Only per-batch deformation state is needed; full corrected images/masks still
occupy CPU RAM. Existing output directories are never overwritten.

The four stacked raster panels share −30…0 dB relative to the first reference's
maximum. Whole-run rasters use temporal means in at most 1200 bins; gray means
no supported samples. Correlation plots show bin means **and min/max envelopes**
so brief dips remain visible; CSV retains every frame. Long runs use minutes.
`--batch-size` (optional override of voxreg's automatic policy), `--iterations` and
`--device cpu|cuda` are optional. QC reads one exported recording at a time,
after dense fields are released; it does not add registration calls.
Correlation improvement is QC, not anatomical truth or proof physiology is retained.

After saving, the script opens **uncorrected, rigid, affine and SyN** in a **2×2
Napari grid**, using `cf.plotting.plot_napari`. All four share a time slider,
physical calibration and the same gray colormap/power contrast limits. Corrected
NIfTIs are loaded lazily through ConfUSIus. Pass **`--no-view`** on headless machines;
the saved outputs are unchanged. Display uses original power rather than the QC
figure's dB scale. ConfUSIus uses median temporal spacing for the viewer; exact
acquisition times remain in the sidecars and CSV.

To reopen an existing run without repeating registration, from the repository root:

```python
import json
import sys
from pathlib import Path

import confusius as cf

sys.path.insert(0, "research/confusius-examples")
from compare_landemard_motion import show_recordings

root = Path("artifacts/landemard-ref20-full")
settings = json.loads((root / "settings.json").read_text())
start = settings["start"]
raw = cf.load(settings["source"]).isel(
    time=slice(start, start + len(settings["time_seconds"]))
)
show_recordings(raw, root)
```

## Pair/atlas exploration and historical validation

**Current mixed-pair protocol:** [native fixed-grid sampling](../slice-geometry/NATIVE-FIXED.md),
with `atlas_native_fixed.py`: no role reversal or replicated slab. Eight fresh
native fits pass CPU/CUDA precision but still fail the atlas reference gate.
Older direction-invariance reports below describe the retired sampling policy.

For **FireANTs atlas results and explicit sampling-fidelity controls**, see
[ATLAS-FIREANTS.md](ATLAS-FIREANTS.md):8 actual FireANTs fits,4 caller-fixed/slab
voxreg controls, support-masked overlays and background-score checks. The native
slice-domain model was not inherited from FireANTs and is not literal CF sampling.

For **late overshoot,1000-step CF continuation and a fresh voxreg centering audit**, see
[ATLAS-CENTERED-FOLLOWUP.md](ATLAS-CENTERED-FOLLOWUP.md). Metric recovery can hide
coverage loss; updated overlays mark unsupported pixels gray. Voxreg was already
centered; direction equality in that historical report followed the now-retired
sampling policy, not independent evidence of alignment accuracy.

For **fits that allow tilt but prevent normal collapse**, see
[ATLAS-RIGID-CENTERED.md](ATLAS-RIGID-CENTERED.md): true rigid in both directions,
and a9-DOF tilt/in-plane-affine control with verified unit perpendicular normal.
Compression is eliminated, but reverse plane-depth/orientation drift remains.

For **centered versus origin-pivot fits in both directions**, see
[ATLAS-CENTERING.md](ATLAS-CENTERING.md): same initial mapping and fitting settings,
with actual metric traces and2×2 overlays. Centering changes the trajectory but
does not fix the reverse fit; it exposes strong through-plane affine compression.

For **why the atlas poses differ**, see
[the direct ITK / matched-freedom investigation](ATLAS-ALIGNMENT-INVESTIGATION.md):
CF's original affine fit preserves manual plane depth/tilt, while voxreg refines
those coordinates; gradient construction and capture also matter.

For **actual atlas metric histories, longer-run controls and rotation/translation
diagnosis**, see [ATLAS-CONVERGENCE.md](ATLAS-CONVERGENCE.md). Native stopping is
not proof of a settled affine solution; the new optional metric callback supports
live CPU/CUDA reporting without changing default numerical execution.

For **affine atlas registration in both argument directions**, see
[ATLAS-DIRECTIONS.md](ATLAS-DIRECTIONS.md): voxreg is direction-invariant here;
CF changes by17.23 native pixels RMS. Both direction-matched CF acceptance checks
still fail. Actual template and official Allen-colored overlays are retained.

The **[finite singleton-extent correction](../slice-geometry/THICKNESS.md)** fixes
the zero-thickness mask discovered by the [pre-fix initial-placement control](ATLAS-SEED-CONTROL.md).
The fresh zero-update report now has nonempty volume output; active-axis borders
and the fitted atlas-pose disagreement remain separate, unresolved differences.
The original affine panels already ran registration, not just manual placement.

For **all three executed notebook/HTML comparisons with actual plots and official
Allen-colored overlays**, see [NOTEBOOKS.md](NOTEBOOKS.md). It also audits the
installed CF matrix-seeded requested-rigid stage, which returns an affine-valued
matrix; the earlier chain is not a six-DOF-versus-six-DOF comparison.

For **rigid→affine atlas comparison with current installed ConfUSIus designated
as ground truth**, see [RIGID-AFFINE-ATLAS.md](RIGID-AFFINE-ATLAS.md). Voxreg fails
the reference-placement gate despite passing CPU/CUDA agreement; results and
atlas-label overlays are retained separately from this historical baseline.

For **current public-API timings, motion-correction comparisons and alignment
figures**, see [BENCHMARK.md](BENCHMARK.md). It preserves all successful and failed
precision screens and does not claim atlas/B-spline support.

**Historical baseline at `c2effa2`, using the former 90% coverage rule.**
The [coverage-policy follow-up](../overlap-policy/README.md) removes that cutoff and
reruns the same comparison. The failures and results below are preserved, not
current-policy claims. Check out `c2effa2` to reproduce these exact baseline numbers;
the current script records source hashes and uses the currently checked-out engine.

## Verdict

**Example 01 is supported geometrically, but voxreg defaults do not reproduce its
alignment quality on the full images.** The main blockers are the 100-update
capture budget and the unchanged 90% fixed-domain overlap guard, not CPU/CUDA
numerical disagreement. A separately labeled central fitting crop plus 400 updates
produces much closer visual alignment and comparable common-support correlation.
This is a diagnostic workaround, **not** a fix to the default full-image workflow.
No registration implementation, threshold or ConfUSIus file was changed.

| Example | Evidence / status |
| --- | --- |
| 01: same subject across sessions | New 36-run rigid/affine comparison below. B-spline stage not attempted. |
| 02: motion correction | Reuse the bounded 120-frame comparison in [VALIDATION.md](../../VALIDATION.md#real-recording-quality). The current engine's four default-rigid CPU/CUDA fixed/convergence controls remain bitwise equal to frozen AC outputs; this is not a fresh full-recording ConfUSIus run. |
| 03: recording to Allen fUSI template | Blocked as written: single-slice recording versus a cropped 3D template is slice-to-volume registration, which voxreg rejects. No new downloads or atlas registrations were attempted. |

## Inputs and protocol

The linear stage of `docs/examples/03_registration/01_register_volume_same_subject.py`:
Cybis Pereira dataset, rat75, slice32, `susi` / `rec-minframe2d_pwd`, sessions
20220523 (fixed) and 20220524 (moving). Both are original float64 `(1,112,128)`
angiographies, already present in the local ConfUSIus cache. Data reference:
Cybis Pereira et al., https://doi.org/10.1016/j.celrep.2025.116791 .

Follow the example's exact representation: per-image maximum-referenced dB, then
apply `world_to_qform`; no additional fitting intensity scaling. This deliberately
matches its dB outputs, not original linear-power resampling. Geometry initialization
is used, never intensity moments on pre-scaled data. The qform planes are coplanar
within 1e-5 native pixels. Uniform j/i spacings are approximately 0.09856/0.11 mm.

Run voxreg engine `ad335c9` (CPU/CUDA) against installed ConfUSIus 0.8.0 / ITK 5.4.7
using its public `register_volume`. ConfUSIus uses native GD with automatic learning
rate and a thin-expanded 3D slice; voxreg fits its identifiable in-plane model with
Adam0.1. Native defaults differ (CF window10 versus voxreg window50, both caps100).
Fixed100/fixed400 disable convergence with a window one greater than the budget;
optimizer/scaling differences remain. These are **not matched optimizer trajectories
or timing comparisons**. Normal mapping remains within 1.1e-18 native k pixels, so
normal displacement does not invalidate the shared planar score.

All arms retain the full moving image. The original full fixed image is retained
as the primary arm. The diagnostic fixed crop removes 16 pixels on all active
edges: j[16:-16], i[16:-16], shape `(1,80,96)`. It is a caller-side crop, not a new
mask API or relaxed support guard. Both methods use that same crop when fitting.

**Every comparison is independently rescored on the original full fixed grid**, not
just the crop. A single intersection of valid sample-center support across all 36
runs and identity covers **84.1588%** of the original grid. SciPy64 bilinear pulls
and squared NCC use exactly that same domain for every number below. Per-arm joint
scores/support, original backend scores, all matrices, settings, file/sidecar hashes,
source hashes, metric histories, affine singular values and mapping disagreements
are retained in `results.json`. CPU uses one thread; no runtime claim is made.

## Results

Identity NCC² on the common domain is **0.217985**.

| Fitting condition | Model | voxreg CPU NCC² | ConfUSIus NCC² | voxreg outcome |
| --- | --- | ---: | ---: | --- |
| Full image, defaults | Rigid | 0.549362 | 0.799413 | 100 updates, limit |
| Full image, defaults | Affine | 0.641963 | 0.844788 | 100 updates, limit |
| Full image, fixed400 | Rigid | 0.549362 | 0.836226 | overlap failure at 101 updates |
| Full image, fixed400 | Affine | 0.722159 | 0.846557 | overlap failure at 184 updates |
| Central fitting crop, fixed400 | Rigid | 0.837378 | 0.814755 | 400 updates, limit |
| Central fitting crop, fixed400 | Affine | 0.863565 | 0.845232 | 400 updates, limit |

The central crop **alone** does not fix the default budget: its 100-update voxreg
scores are 0.469023 rigid / 0.591093 affine. Both more capture updates and an adequate
fitting domain matter. The crop400 results retain 100% support on the fitting crop,
but only approximately 87.81% / 86.29% on the original fixed image. Likewise,
ConfUSIus's successful full-image solutions retain only about 87–88% support—below
the full-image guard imposed by voxreg. Simply increasing the full-image cap cannot
make those poses admissible under the existing rule.

The default full-image affine estimate contracts one principal direction to
**0.8738**; after crop400 its singular values are approximately **1.0336 / 0.9903**,
close to ConfUSIus crop400's **1.0318 / 0.9893**. The early contraction is an optimizer
tradeoff on the under-captured path, not a trusted estimate of anatomical scaling.
The guard is a hard stop, not a scale regularizer or a differentiable overlap penalty.

Red/cyan overlays visibly show residual doubled vessels with default voxreg fits
and much better vessel overlap after crop400. Common-domain difference maps tell
the same story. Crop400 transforms still differ from ConfUSIus by roughly
**0.563 pixels RMS rigid / 0.395 pixels RMS affine** across the full fixed grid.
These are algorithm disagreements, **not biological truth errors**. Higher NCC or
modest affine stretch does not prove correct anatomical correspondences.

### Numerical checks

- All 12 CPU/CUDA comparisons pass the unchanged **0.001-pixel maximum pointwise**
  gate and have identical update counts/statuses; worst observed maximum is
  **2.23e-7 pixels**.
- All 36 outputs retain input float64 dtype and their fitted target shape.
- Backend output agrees with independent SciPy64 interpolation on shared interior
  points to **6.74e-13 dB** worst absolute error (assertion bound 1e-8).
- No known biological transform exists, so the synthetic 0.25-pixel **truth** gate
  cannot be asserted against ConfUSIus as if it were ground truth.

## Reproduce and inspect

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python research/confusius-examples/compare_same_subject.py \
  --output artifacts/confusius-examples/same-subject-reproduction
```

The script requires cached inputs, does not download, and refuses to overwrite a
completed output. It saves strict JSON, registered arrays/matrices/native affines
(including the cropped affine), joint support, and plots. Self-checks cover geometry,
output dtype/shape, independent interpolation and CPU/CUDA precision. The current
inspection outputs are under `artifacts/confusius-examples/same-subject-final/`:

- `defaults.png`: full/crop native-default red/cyan overlays.
- `fixed100.png`, `fixed400.png`: the same panels with convergence disabled.
- `differences.png`: voxreg−fixed, ConfUSIus−fixed and voxreg−ConfUSIus, on the same
  joint mask with a shared ±10 dB color scale.
- `arrays.npz`, `results.json`: machine-readable outputs and provenance.

## Next gate

Before presenting voxreg as a drop-in replacement for the same-subject example,
resolve the **capture/fitting-domain/partial-overlap policy** and qualify it on the
unchanged full-image case. Options such as caller-supplied ROI/masks, staged
initialization, pyramids, or an explicitly chosen overlap policy need a separate
implementation decision. Do not silently lower the guard or increase defaults
based on this single pair. Default affine/3D convergence, natural 3D pairs and the
atlas slice-to-volume workflow remain open; adding deformable models is not the
next validation priority.
