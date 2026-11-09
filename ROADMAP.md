# Next steps

Goal: a small, physically correct CPU/CUDA registration engine for ConfUSIus.
Current model support is documented in the [README](README.md); numerical
contracts and unresolved accuracy limits belong in [VALIDATION.md](VALIDATION.md).

## Current work status

Work proceeds one roadmap point at a time, stopping after each for user review
and a separate commit. The registration-module refactor is implemented and
reviewed. No higher-order interpolation or device default has been
enabled. Broad benchmark expansion remains deferred to its roadmap point;
preserve existing artifacts and frozen sources/settings.

## Planned capabilities

### Final image interpolation

Clarify interpolation naming at the public boundary: current interpolation
selects final original-image resampling, while fitting always uses linear
interpolation. Consider an explicit resample_interpolation name with a backwards
compatible alias; add fitting-interpolator choices only for a demonstrated need.

Add an optional higher-order final interpolator, prioritizing cubic B-spline
support consistently in 2D and 3D for both register and resample. Specify the
kernel and any coefficient prefilter explicitly: PyTorch bicubic convolution is
not interchangeable with cubic B-spline interpolation, and grid_sample does not
offer general volumetric cubic interpolation. Keep nearest for categorical
labels and linear as the existing default. Qualify CPU/CUDA behavior, precision,
boundary support, overshoot and memory cost against independent interpolation
oracles, with one final sampling of original intensities.

Label-aware resampling is deferred at the user's request; retain nearest-neighbour
resampling for categorical labels for now.

### Device selection

Consider device="auto" for registration and resampling, resolving to CUDA when
available and CPU otherwise, while preserving explicit device="cpu"/"cuda".
Record the resolved device in diagnostics and document execution on lazy workers.
Resolve whether auto becomes the default after CPU/CUDA quality qualification;
benchmarks retain explicit devices. Do not silently retry on CPU after CUDA
execution failures or out-of-memory errors.

### First real-data motion-quality case: paper Figure 2

Use the existing rat75 / 20220523 / slice32 power-Doppler window, frames 220–339,
from the related confusius-paper repository. The entry script is
/home/sdiebolt/Documents/Work/cortexlab/confusius-tools/confusius-paper/paper/figures/fig2/scripts/03_plot_volumewise_motion_correction.py.
The user reports some voxreg-corrected DVARS above raw DVARS while ConfUSIus's
register_volumewise performs better. Treat this as a quality investigation,
not an established general ranking or a timing benchmark.

- Preserve the paper script, original figure outputs and existing caches. Check
  cache provenance; use separate, explicitly configured outputs for new runs.
- Reproduce raw and corrected per-frame DVARS with the same brain mask, time
  window and definition. Compare the raw-support region and the common valid
  spatial support separately so padding/boundaries cannot masquerade as motion.
- Match the reference frame, fitting intensity transform, model, interpolation,
  initialization, pyramid schedule and iteration/convergence policy explicitly;
  disclose optimizer differences rather than equating numeric learning rates.
  The current script uses ConfUSIus correlation with unscaled fitting intensities
  and learning_rate=1.0; voxreg's default fitting intensities are dB-scaled.
- Inspect the offending adjacent frame pairs, fitted transforms, metric traces,
  status/steps, registration residuals and anatomical alignment. DVARS is a
  temporal QC measure, not a spatial correspondence objective or ground truth;
  smaller DVARS alone cannot qualify a correction.
- Identify a minimal reproducible cause before changing defaults or optimizers.
  Retain bounded evidence and a regression case when a defect is established.

Initial reproduction is recorded in [VALIDATION.md](VALIDATION.md#first-real-data-motion-quality-case).
The fresh default-voxreg DVARS spikes disappear after excluding 31 ROI voxels
that become unsupported. All affected default-run samples lie within half-voxel
cell extents, exposing a center-range versus ITK cell-support difference to audit
before further optimizer tuning. The older voxreg cache is not reproduced by
the tested fresh settings; preserve that provenance uncertainty.

### Lazy recordings and resampling with Dask

Support recordings whose time dimension is too large to fit in RAM. The intended
workflow is register(recording) or resample(recording, ..., transforms) returning
lazy xarray-backed results that can be computed or written incrementally with
to_zarr. Preserve the eager NumPy path; fitting one volume to another remains
eager, including inside each recording task. One fixed/moving spatial pair and
its working state must fit in the worker's RAM/VRAM.

- Partition work along time into bounded eager frame batches. Load the shared
  fixed image, or implicit first-frame reference, through shared dependencies;
  keep every frame's complete stage/pyramid pipeline within its task. Preserve
  time labels, reference behavior and independent frame failure handling.
- Require or explicitly arrange complete spatial volumes within each fitting
  task. Arbitrary spatial-block registration and distributed fitting of one
  volume are separate work. Define how Dask time chunks and batch_size interact
  without loading an oversized time chunk or the full recording inadvertently.
- Keep registered intensities, validity masks, optional forward/inverse maps and
  per-frame/stage/level diagnostics lazy along time. Dense SyN fields must not
  accumulate across all frames in RAM. Linear matrices are smaller, but also
  need a lazy path for sufficiently long recordings.
- Remove whole-recording .values/np.asarray conversions and eager finite-value
  scans from graph construction. Validate geometry, labels and options eagerly;
  perform data-dependent checks within tasks and document deferred failures.
- Resolve recording-wide intensity conditioning explicitly: either preserve its
  semantics with documented lazy, bounded-memory global reductions, or introduce
  a documented per-frame policy. Do not silently normalize independently by
  chunk. Results must be invariant to time chunking and task execution order.
- Share each batch's registration task across its images, maps and diagnostics.
  Document joint computation/storage so saving several outputs does not repeat
  fitting, and so users need not persist the entire recording in memory.
  Progress, callbacks and timings must describe execution rather than graph
  construction, with a defined worker/scheduler behavior.
- Extend resampling to lazy images and lazy dense displacement fields, including
  eager/lazy combinations. A shared transform applies to every time point;
  time-labelled transforms select one per frame. Preserve existing geometry,
  unit, interpolation, support and time-alignment contracts. Load only the image
  and transform frames required by each task, then permit incremental Zarr writes.
- Bound concurrent CPU/GPU work as well as individual task memory. A Dask
  scheduler must not launch unbounded registration tasks on one GPU. Qualify
  chunk-invariant numerical results, masks/stages/pyramids, shared/per-frame maps,
  deferred failures, Zarr round trips and peak memory on recordings larger than
  RAM before claiming out-of-core support.

Beamformed IQ recordings may be complex-valued, whereas voxreg currently
requires real intensities. Track this alongside lazy execution: define a real
fitting representation (for example amplitude or power) and phase-preserving
resampling of the original complex data, without conflating spatial interpolation
with temporal phase correction. Complex-data support is a separate capability;
time chunking by itself does not provide it.

## Priorities

1. **Review the coruscant benchmark when complete.** Establish the current
   precision-policy baseline using [one pipeline](BENCHMARK.md), without competing
   heavy jobs or changing the running job's frozen sources/settings.
2. **Profile the SyN runtime gap to FireANTs.** Keep the shared
   [0.30-voxel registration gate](research/syn/REFERENCE.md) and recorded errors;
   defer further accuracy tuning while investigating runtime.
3. **Optimize one measured bottleneck.** Compare workload/device-specific
   batching or a source-guided implementation change without weakening precision
   or safety checks; label non-passing cases rather than hiding them.
   Preserve timing boundaries and report memory; see [PERFORMANCE.md](PERFORMANCE.md).
4. **Broaden real-data qualification.** Revisit pair, motion and atlas workflows
   with original images, native geometry and explicit quality limitations.

## Implemented

- **Registration module organization:** registration.py owns the public entry
  point and common validation/preparation. Linear and SyN engines consume typed
  prepared inputs and shared immutable settings; results/statuses and batch
  policy have shared owners, independent of fitting engines. Public root imports
  and numerical behavior are retained. See [engine boundaries](DESIGN.md#registration-entry-and-engine-boundaries).

- **Physical length units:** m, cm, mm, um/µm and nm for images/masks, including
  mixed-unit inputs. Internal geometry, exported transforms, physical diagnostics
  and physical smoothing arguments remain in mm; image outputs retain native
  reference coordinates. Labelled matrix/field units are normalized explicitly.
  See [unit conventions](README.md#physical-units).

- **Mutual information:** a shared per-frame metric interface with Gaussian-Parzen
  MI, explicit bins/bandwidth, fixed native-image ranges, mask-aware histograms
  and bounded bin-weight workspace. Works with every model, pyramid and stage;
  NCC defaults are unchanged. See [MI semantics](README.md#mutual-information).
  Real-data bandwidth, convergence and runtime qualification remain open.
- **Fixed and moving masks:** boolean ROIs retain native geometry and are sampled
  through the complete physical/halfway maps at every stage and level. Local NCC
  excludes masked neighbours. ROI coverage is distinct from output validity;
  image interpolation and smoothing remain independent. See [mask semantics](README.md#roi-masks).
- **Staged registration:** explicit model sequences with optional `Stage`
  overrides, initialization once, per-frame failure isolation and separate
  stage/pyramid diagnostics. Original intensities are resampled once; final maps
  include every stage. See [staging](README.md#staged-registration). Reusable
  workflow configurations remain deferred until real-data qualification.
- **Multi-resolution:** explicit coarse-to-fine shrink factors, Gaussian image
  smoothing sigmas in mm and per-level iteration budgets for every model.
  Physical transforms/halfway fields carry between levels with fresh optimizer
  and convergence state; original images are resampled once. See the
  [API and diagnostic conventions](README.md#multi-resolution-fitting).
  Qualifying useful schedules, capture range and whole-call performance on real
  workflows remains future work; no multi-resolution default preset is enabled.

## Candidates, not enabled defaults

- **Further metric/optimizer controls:** Gaussian-weighted local-CC kernels,
  MSE/custom losses, optimizer choice/hyperparameters and regularization hooks.
  SyN's rectangular CC radius and update/total smoothing strengths are public
  options already. Follow ITK's unit distinction: image-pyramid smoothing
  sigmas in mm, deformation regularization in voxel units. Keep existing SyN
  sigmas/defaults; ITK's field parameters are variances, not interchangeable values.
  Keep current optimizers for now. Velocity-based deformation/integration,
  compute-dtype selection and rigid-plus-scale are further options, not
  requirements for parity.
  Reimplement mechanisms independently; do not transplant licensed vendor code.
- **Multistart:** only with a selection rule that rejects tiny-overlap or
  anatomically wrong metric optima, especially for slice placement.
- **Line search:** investigate measured overshoot and extra evaluation cost before
  adding another optimizer policy.

These are candidates, not implemented features or a commitment to expose every
vendor option. Streaming/Dask execution is now planned above but remains
unimplemented; mixed-dimensional deformation remains outside the current scope.
Add other scope for a demonstrated need, not as a prerequisite to benchmarking
or fixing existing accuracy failures.

Keep source/settings and minimal failure evidence before cleanup. Never retune a
truth threshold or shorten work merely to manufacture a speedup. Historical
experiments and their original limitations remain in [research](research/README.md).
