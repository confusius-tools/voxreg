# Roadmap

Goal: a small, physically correct CPU/CUDA registration engine for ConfUSIus.
[README](README.md) owns supported behavior; [VALIDATION](VALIDATION.md) owns
numerical contracts, evidence and accuracy limits. Work one roadmap point at a
time, then stop for user review and a separate commit.

## Performance and qualification priorities

1. **Review coruscant benchmark results when available.** Establish the current
   precision-policy baseline through [one pipeline](BENCHMARK.md). Job completion
   is not tracked here; preserve frozen sources/settings and avoid competing heavy
   jobs during measurement. Broad benchmark expansion remains deferred.
2. **Profile the SyN runtime gap to FireANTs.** Preserve the shared
   [0.30-voxel registration gate](research/syn/REFERENCE.md) and recorded errors;
   defer further accuracy tuning while investigating runtime.
3. **Optimize one measured bottleneck.** Compare workload/device-specific batching
   or a source-guided change without weakening precision/safety checks. Label
   non-passing cases, preserve timing boundaries and report memory; see
   [PERFORMANCE](PERFORMANCE.md).
4. **Broaden real-data qualification.** Revisit pair, motion and atlas workflows
   with original images/native geometry and explicit quality limits. Qualify MI
   bandwidth/convergence/runtime and useful pyramid schedules/capture range/whole-call
   performance. Reusable workflow configurations await real-data qualification.

## Planned capabilities and investigations

### Qualify large lazy recordings

The bounded Dask path is implemented and tested on small inputs. Run the
[streaming benchmark](BENCHMARK.md#lazy-streaming-benchmark) with its 32/128 MiB
chunk targets to measure peak RAM/VRAM and scheduler/storage prefetch on recordings
larger than physical RAM
before claiming fully qualified out-of-core execution. One eager spatial pair and
working state must fit each worker; source storage blocks must be bounded too.
Qualify multi-host/GPU deployment and workload-specific concurrency/performance;
spatial-block registration and distributed fitting of one volume remain separate.

**Complex IQ is separate:** voxreg currently requires real intensities. Define a
real fitting representation (e.g. amplitude/power) and phase-preserving resampling
of original complex data; spatial interpolation is not temporal phase correction.
Time chunking alone does not provide complex-data support.

## Implemented — not pending features

Details live in the linked docs rather than being duplicated here:

- **Lazy recordings/resampling:** shared bounded eager time tasks return lazy
  images, masks, optional linear/dense maps and stage/level diagnostics. Local and
  distributed schedulers, worker-time devices/callbacks and joint DataTree Zarr
  storage are covered. Local schedulers share voxreg's eager Rich progress style;
  distributed clients use Dask's progress tools. Tasks follow temporal source chunks; worker fitting batches
  are selected independently or explicitly overridden. Per-image conditioning,
  labels/geometry and implicit references are retained. Full-size memory and
  deployment qualification remains above; see [contracts](README.md#lazy-recordings-and-resampling).

- **Optional fitting winsorization:** independent moving/fixed 0–100 percentile
  pairs, disabled by default. Per-image signed clipping precedes intensity
  transforms and is shared across stages; native moments/output stay unchanged.
  See [semantics](README.md#fitting-winsorization); lazy task parity is tested.

- **Figure 2 DVARS investigation:** valid-support QC resolves the reported spikes;
  no registration change needed. Evidence and remaining interpretation limits are
  in [validation](VALIDATION.md#first-real-data-motion-quality-case).

- **Separate intensity scaling:** moving/fixed controls default independently to
  `None`; `"log1p"` uses ln(1 + abs(x)), while `"sqrt"` and positive finite
  powers use magnitudes. Native moments/output intensities stay untouched;
  stages/pyramids/masks share the documented per-image conditioning policy.
  Recording-wide numerical scaling and the `"db"` spelling are removed; implicit
  references use moving scaling. Synthetic default/precision regressions pass; real-data qualification
  of the changed defaults remains open. See [scaling](README.md#fitting-intensity-scaling)
  and [validation](VALIDATION.md#intensity-scaling).
- **Automatic device selection:** `register`/`resample` default to `device="auto"`;
  explicit devices remain supported, with no CPU retry on CUDA/OOM errors. Eager
  resolution occurs once per call and is recorded in diagnostics; benchmark
  adapters retain explicit devices. See [quick start](README.md#quick-start) and
  [device validation](VALIDATION.md#checks).
- **Cubic final-image interpolation:** optional 2D/3D `"bspline"` for registration
  (`resample_interpolation`) and offline resampling; float64 coefficients/evaluation,
  mirror boundaries, unchanged support and linear fitting/default interpolation.
  CPU/CUDA oracle checks pass. See [interpolation](README.md#final-image-interpolation).
- **Registration module refactor:** entry validation/preparation, typed engine
  inputs, immutable settings, shared results/statuses/batch policy; root imports
  and numerical behavior preserved. See [engine boundaries](DESIGN.md#registration-entry-and-engine-boundaries).
- **Physical length units:** m, cm, mm, um/µm and nm, including mixed inputs;
  internal geometry/exported transforms/physical diagnostics/smoothing use mm,
  outputs retain reference-grid units, and labelled maps normalize explicitly.
  See [units](README.md#physical-units).
- **Mutual information:** per-frame Gaussian-Parzen MI, explicit bins/bandwidth,
  fixed native ranges, mask-aware histograms and bounded workspace, for all
  models/pyramids/stages; NCC defaults unchanged. See [MI](README.md#mutual-information).
- **Fixed/moving masks:** native boolean ROIs sampled through full physical/halfway
  maps; local NCC excludes masked neighbours. Coverage differs from output
  validity; interpolation/smoothing stay independent. See [masks](README.md#roi-masks).
- **Staged registration:** model sequences/`Stage` overrides, initialization once,
  frame failure isolation, stage/level diagnostics, one final resampling and full
  composed maps. See [staging](README.md#staged-registration).
- **Multi-resolution:** shrink factors, Gaussian image smoothing in mm and
  per-level budgets for all models; physical transforms/fields carry between
  levels with fresh optimizer/convergence state and one final resampling. No
  default preset. See [pyramids](README.md#multi-resolution-fitting).

**Still unqualified/deferred:** auto-device/cubic performance, label-aware
resampling (use nearest for categorical labels), and real-data qualification/
configurations listed above. CPU success alone does not qualify CUDA; eager
batching does not provide out-of-core execution.

## Candidates, not commitments or enabled defaults

- **Metric/optimizer controls:** Gaussian-weighted local CC, MSE/custom losses,
  optimizer choice/hyperparameters and regularization hooks. SyN already exposes
  rectangular CC radius and update/total smoothing. Keep current optimizers and
  SyN sigmas/defaults: image-pyramid sigmas are mm, field regularization is in
  voxel units, and ITK field parameters are variances, not interchangeable values.
- **Other options:** velocity-based deformation/integration, compute-dtype
  selection and rigid-plus-scale are not parity requirements. Reimplement
  mechanisms independently; do not transplant licensed vendor code.
- **Multistart:** requires selection that rejects tiny-overlap or anatomically
  wrong optima, especially for slice placement.
- **Line search:** first measure overshoot and extra evaluation cost.

Mixed-dimensional deformation remains out of scope. Add scope only for a
demonstrated need, not as a prerequisite to benchmarks or accuracy fixes. Preserve
sources/settings and minimal failure evidence before cleanup; never retune truth
thresholds or shorten work to manufacture speedups. Historical experiments and
limitations remain in [research](research/README.md).
