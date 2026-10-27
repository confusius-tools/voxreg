# Atlas affine registration in both argument directions

Open **`artifacts/atlas-directions/affine/directions.html`** for the executed
comparison, template composites and official Allen-colored overlays. The `.ipynb`,
standalone PNGs, original-grid outputs and canonical matrices are alongside it.
[atlas-direction-results.json](atlas-direction-results.json) records all arms.

## Controlled experiment

- Original cached example03 recording mean, qform/dB template slab and manual pose.
- **Affine-only**, avoiding the separately audited CF matrix-seeded requested-rigid
  model issue. CF uses its full affine model; voxreg retains its nine-parameter
  native-slice affine model. No optimizer/model settings are changed.
- Both supplied directions: moving recording slice/fixed template volume, and moving
  template volume/fixed recording slice. Seeds are literal inverse-equivalent
  matrices; no pre-resampling or extra composition.
- CF0.8.0: learning_rate1, max500/window50, eight native threads.
- Voxreg: learning_rate0.1, max500/window50, CPU8 or CUDA1 host thread.
- Every returned matrix is normalized to **template-world→recording-world pull**.
  Point errors compare its inverse on all10000 native recording pixel centers.
- Native template display/scoring uses the same SciPy sampler for every pose.
  Official Allen labels use the same CF nearest-neighbor helper and official CSV
  palette for every pose. All panels have matched display scales.

## Results

All arms converged before their500-update cap. CPU/CUDA voxreg rows agree to the
shown digits. NCC² uses90.86% joint native support across all six arms; it is not
an acceptance substitute for agreement with the operator-designated CF reference.

| Supplied moving → fixed | Backend | Updates | Common NCC² | RMS px versus CF in that direction |
| --- | --- | ---: | ---: | ---: |
| Slice → volume | CF | 320 | 0.104891 | Reference |
| Slice → volume | voxreg CPU/CUDA | 91 | 0.138450 | **6.217505 — fail** |
| Volume → slice | CF | 127 | 0.111166 | Reference |
| Volume → slice | voxreg CPU/CUDA | 91 | 0.138450 | **13.651405 — fail** |

**Reversing arguments changes CF substantially, but not voxreg:**

| Backend | Direction-change RMS | Direction-change maximum |
| --- | ---: | ---: |
| CF | **0.832644mm /17.231877px** | 24.204353px |
| voxreg CPU | **0** | 0 |
| voxreg CUDA | **0** | 0 |

Voxreg's canonical matrices are bitwise identical across the two supplied directions
on each device. CPU/CUDA maximum placement disagreement is9.76e-7px in both
cases, passing the unchanged0.001px numerical gate.

The new CF direction is **not silently substituted for the original reference**.
The report separately retains each pose's disagreement from CF's original
slice→volume fit: both voxreg directions remain6.217505px away. The original
reference's0.25px RMS acceptance gate still fails.

Foreground label agreement with the direction-matched CF result is45.65% for
slice→volume and14.76% for volume→slice. Voxreg CPU/CUDA labels are identical.
Both sets of actual colored overlays are retained for operator inspection.

## Interpretation

This supports testing role reversal as a legitimate formulation, rather than
calling it inherently wrong. Voxreg's internal native-slice canonicalization
provides exactly the direction invariance it was designed to provide here.

However, that does **not** establish parity with CF, nor explain the remaining
disagreement. CF's supplied direction changes the fixed sampling domain; it also
expands thin images while preserving extent. Even when both callers supply a
fixed slice, CF's expanded finite-slab domain and full affine parameterization
are not identical to voxreg's native-center/nine-parameter objective. Optimizers
and their scaling also differ. This experiment does not isolate any single cause,
or prove that either new direction is more anatomically correct.

No engine change, new support policy, convergence tuning or performance claim.
The finite singleton-extent correction remains in place for voxreg output.

## Reproduce and checks

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/atlas_directions.py \
  --output artifacts/atlas-directions/new-affine
```

Destination must be new; cached data/labels/palette required, no downloads.
Four notebook code cells executed, producing four comparison figures. Three new
normalization/copy tests and existing atlas-chain, exporter and benchmark tests
pass (**17 focused tests**). Source fingerprints, canonical direction equality,
embedded plots and local HTML links are checked. The prior full453 engine suite
is not presented as a new full456 run; this change is research/tests/docs only.
