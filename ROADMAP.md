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

### Optional fitting winsorization

Add independent moving/fixed percentile-clipping arguments to `register`, disabled
by default and shared across stages. Define percentile units/order, per-image
bounds, masks, zero/constant inputs and ordering before intensity transforms;
keep native moment initialization and final resampling unchanged. Validate against
NumPy percentile/clip, including lazy chunk invariance; expose no ANTs/FireANTs
compatibility aliases.

### Lazy recordings and resampling with Dask

Support `register(recording)` and `resample(recording, ..., transforms)` with lazy
xarray-backed outputs that can be computed or written incrementally via `to_zarr`.
Preserve eager NumPy behavior. Each spatial pair fits eagerly within its task;
one fixed/moving volume pair and working state must fit worker RAM/VRAM.

- **Task boundaries:** bounded eager time batches; shared dependencies for the
  fixed image or implicit first-frame reference; complete stage/pyramid pipeline
  per frame within its task. Preserve time labels, reference behavior and
  independent frame failures.
- **Chunking:** require or arrange complete spatial volumes per fitting task.
  Define time chunks versus `batch_size` without inadvertently loading oversized
  chunks/full recordings. Arbitrary spatial-block or distributed single-volume
  fitting is separate work.
- **Lazy outputs:** intensities, validity masks, optional forward/inverse maps and
  per-frame/stage/level diagnostics. Do not accumulate dense SyN fields across
  frames; linear matrices also need a lazy path for long recordings.
- **Validation:** remove whole-recording `.values`/`np.asarray` conversions and
  eager finite scans from graph construction. Validate geometry, labels and
  options eagerly; check data within tasks and document deferred failures.
- **Conditioning:** use the same per-image policy as eager registration, never
  per-chunk normalization or a recording-wide intensity reference. Results must
  be invariant to time chunking and execution order.
- **Shared execution:** share each batch fit across images, maps and diagnostics.
  Document joint computation/storage without repeated fitting or whole-recording
  persistence. Progress, callbacks and timings must describe execution, not graph
  construction. Support distributed workers as well as local schedulers; execute
  callbacks within tasks with global frame indices and document repeated calls
  on recomputation.
- **Resampling:** support lazy images and dense fields, including eager/lazy
  combinations. Shared transforms apply to all frames; time-labelled transforms
  select per frame. Load only required image/transform frames, allow incremental
  Zarr writes, and retain geometry, units, interpolation, support and time alignment.
- **Resources and proof:** bound concurrent CPU/GPU work and task memory; resolve
  `device="auto"` within workers at execution time and prevent unbounded jobs on
  one GPU. Qualify chunk invariance, masks/stages/pyramids, shared/per-frame maps,
  deferred failures, Zarr round trips and peak memory on recordings larger than
  RAM before claiming out-of-core support.

**Complex IQ is separate:** voxreg currently requires real intensities. Define a
real fitting representation (e.g. amplitude/power) and phase-preserving resampling
of original complex data; spatial interpolation is not temporal phase correction.
Time chunking alone does not provide complex-data support.

## Implemented — not pending features

Details live in the linked docs rather than being duplicated here:

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

**Still unqualified/deferred:** auto-device/cubic performance, a clean full CUDA
test gate ([existing failures](VALIDATION.md#intensity-scaling)), label-aware
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
