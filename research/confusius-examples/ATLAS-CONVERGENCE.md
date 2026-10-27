# Atlas convergence, rotation/translation discrepancy, and live metrics

Open **`artifacts/atlas-convergence/qualified/convergence.html`** for the actual
metric histories, stopping-region zoom and native template overlays. Executed
notebook, PNGs, trial histories and output matrices are alongside it.

## Did the91-update fit converge properly?

**It met the implemented stopping rule, but did not settle at the best metric
reachable on that trajectory.** No optimizer, learning rate or stopping rule was
changed in this diagnostic.

Same cached inputs/manual seed and affine learning_rate0.1:

| Run | Updates | Best internal NCC² | Best at update | Last trial NCC² | RMS px vs original CF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native CPU/CUDA, max500/window50 | 91 | 0.153627 | **56** | 0.150341 | **6.217505** |
| CUDA fixed500/window501 | 500 | 0.163318 | **151** | 0.120369 | **12.726265** |
| CUDA fixed1000/window1001 | 1000 | 0.163318 | **151** | 0.120369 | **12.726265** |

The native CUDA trajectory is an **exact prefix** of both no-stopping trajectories
when repeated inactive-frame clocks are removed. Observing every metric does not
change the native CPU/CUDA matrices: they match the prior no-callback fits bitwise.

Replay of the native50-value stopping monitor crosses the1e-6 threshold at
update91 with value **−6.747e-6**. This ITK-style window criterion can terminate
an unstable/non-improving window; it does not measure transform stationarity or
prove a global/local optimum. The returned pose is the earlier best at56, not the
trial at91. Later, without stopping, the metric improves again and reaches a new
best at151 before deteriorating substantially. Both longer runs return that151
pose, not their poorer final trial.

This is evidence against treating the native "converged" status as proof of a
well-settled affine registration. It is **not proof of an incorrectly implemented
ITK window rule**; that threshold was qualified on rigid slice motion, not this
affine atlas trajectory. The trajectory is not monotonic, and simply extending
the budget is not a CF-parity fix.

Metric support also changes:91.78% at the native retained pose versus90.49% at
the longer-run retained pose. The common final-image score uses90.49% joint native
support; internal losses use each pose's own support. CF's saved raw curve is shown
on a **separate panel** because its finite-slab metric domain differs from voxreg's
native-slice domain. Neither a higher raw score nor an earlier/later stop proves
better anatomy or reference agreement.

## Is the discrepancy mostly a rotation and translation?

Yes, **relative to the original CF affine pose**. A diagnostic proper rigid fit
between the two predicted placements of all10000 recording centers explains
**84.63% of the squared native-pose discrepancy**:

- Best rigid correction angle: **3.836°** (a3D rotation, not necessarily pure
  in-plane rotation).
- Recording-center correction in template world z/y/x:
  **[−0.221657,+0.080076,−0.021887]mm**, magnitude **0.236692mm**.
- RMS disagreement falls from **0.300430mm /6.2175px** to
  **0.117793mm /2.4378px** after that diagnostic rigid correction.

The residual remains above the0.25px reference gate: it is not solely rigid pose;
affine/tangent-shape differences remain. This correction is measured for diagnosis
only, **not applied to the returned registration or presented as a successful fit**.
The fixed500/1000 retained pose is even farther from CF despite its higher metric.

## Reporting during registration

`voxreg.register` now accepts optional **`metric_callback`** and **`metric_interval`**.
The callback gets owned NumPy arrays `(frame_indices, updates, objective)` for each
resident chunk. The objective is raw negative NCC², NaN for unusable metrics;
update counts are per frame, not a shared global loop index. Initial/final trials
are always emitted; stopped frames can repeat. The returned best-valid metric is
separately available in diagnostics.

The default interval is10 rounds; use1 for a deliberate diagnostic trace. Callback
copies/synchronizes CUDA only on reporting rounds. There is **no added host transfer
when disabled**, and no automatic history retention or plotting dependency. A plot
consumer should redraw less often than it collects samples. Frame-based `progress`
keeps its original contract. Callback exceptions propagate and callback work is
included in optimization timings; no speed conclusions are taken from traced runs.

## Scope and verification

- Four genuine fits: native CPU/CUDA, fixed500 and fixed1000 CUDA. Same rate/seed;
  reference is the saved, source/input-hash-checked CF original-direction fit and
  its recorded metric history, not a fresh CF optimization.
- The earlier first diagnostic completed fitting but failed a harness assertion
  that incorrectly treated repeated CUDA inactive-frame clocks as extra updates.
  It is preserved at `artifacts/atlas-convergence/diagnostic/`; the corrected
  qualified report deduplicates clocks for prefix comparison, not metric values.
- Eleven new live-metric tests cover CPU/CUDA, rigid/affine, cadence, per-frame
  clocks, invalid metrics, owned/mutable arrays, validation and error propagation.
  Three rigid-residual tests distinguish genuine rigid correction from scale.
- **470 full tests pass**,1092 warnings; all four frozen motion controls remain
  bitwise identical for images, masks, matrices, status and update counts.
- No convergence/optimizer/default policy changes, pose correction, new GPU
  dispatch or automatic plot window. CF-reference gates still fail.

[atlas-convergence-results.json](atlas-convergence-results.json) stores the complete
traces, final states, reconstructed monitor crossing, rigid residuals and source
fingerprints. The original direction report is required as a reference input.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/atlas_convergence.py \
  --output artifacts/atlas-convergence/new-diagnostic
```

No next fix is selected here. The useful next distinction is optimizer stability
versus objective/support preference; longer runs alone demonstrably do not restore
CF placement agreement.
