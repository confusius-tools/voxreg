# Registration and resampling benchmarks

Compare **Python APIs only**: voxreg, FireANTs and ANTsPy. No ANTs command-line
executables or `ANTSPATH` are used. ANTsPy calls the ANTs library in process.

## Setup and commands

Two commands cover eager and lazy execution. Run sequentially into fresh output
directories; defaults are five timed calls and eight threads:

```bash
just benchmark-register artifacts/register-eager
just benchmark-resample artifacts/resample-eager
just benchmark-register artifacts/register-lazy --lazy
just benchmark-resample artifacts/resample-lazy --lazy
```

Eager mode compares Python APIs with resident inputs. Lazy mode measures voxreg's
cold-input read/compute/buffered-Zarr writes on Linux disk-backed stores; its
default is an eight-frame smoke case. Explicit `--size-gib 1 4` runs the routine
input-size/memory-scaling comparison. Eager recipes create/reuse an isolated
Python 3.13 environment with ANTsPy 0.6.3; lazy mode does not install ANTsPy.

Live entry points and helpers live in `benchmarks/`; examples and historical
research/artifacts remain separate. Inspect modes without running work:

```bash
uv run --extra benchmarks python benchmarks/benchmark_register.py --list
uv run --extra benchmarks python benchmarks/benchmark_resample.py --list
uv run --extra benchmarks python benchmarks/benchmark_register.py --lazy --help
```

Common optional controls are `--runs` and `--threads`; eager subsets use
`--configs`, while lazy devices use `--devices`. Use `--help` (or `--lazy --help`)
for advanced workload/memory overrides. A one-call run is diagnostic, not a
replacement for the five-call protocol.

Each matrix has **45 configurations**: three workloads × rigid/affine/SyN ×
voxreg CPU/CUDA, FireANTs CPU/CUDA and ANTsPy CPU. Resampling's SyN configuration
applies a known displacement field; it does not fit one.

Select configuration ID globs with `--configs`, for example
`--configs 'resident-2d+t-*'` or `--configs '*-antspyx'`. The isolated interpreter
is required only when selecting ANTsPy cases. Other options are `--runs`,
`--threads`, `--batch-size` and `--output`.

Do not compete with an existing heavy job. Keep the machine awake and on stable
AC/performance settings. No automatic resume, OOM retry or batch reduction occurs.
Unavailable backends, errors and partial runs are recorded rather than hidden.

## Workloads and matching

The analytic phantom is not anatomy. Workloads are a `(1,150,128)` planar
recording, one `(80,150,80)` volume pair and a 100-frame volume recording. Spacing
is world z/y/x `[0.2,0.1,0.1]` mm. Recordings preserve frame 0 and fit the remaining
99 frames against it. A declared 16-pose input cycle does not reduce the number
of fits. Known analytic physical pulls, not another library's output, are truth.

The supervisor stages the same original float32 images, float64 physical geometry,
time labels and truth amplitudes/maps once per workload/model. Every worker
verifies the serialized input hash. Generation and file loading are outside
measured calls and worker memory preparation.

voxreg and FireANTs use one effective batch per workload/model/device, resolved
by voxreg's automatic policy or `--batch-size`, then pinned across all phases and
both backends. ANTsPy processes one frame at a time; this is disclosed, not called
batch equivalence. Equal iteration budgets do not imply equivalent optimizers.

CUDA workers disable both cuDNN and matmul TF32 for both backends and record the
flags. Public voxreg APIs do not change application-wide precision settings.

### Registration settings

All fits use a supplied fixed image, identity initialization and one resolution.
All three backends request **200 outer updates**, with metric-convergence stopping
disabled. Exhausting this fixed work budget is expected, not proof of convergence.
Their linear objectives use masked squared global NCC; SyN uses radius-2 local CC with update
sigma 1.5 and total sigma 0.5 voxels. FireANTs uses Adam, learning rate 0.01 for
linear fits and 0.1 for SyN. Native learning rates are retained: voxreg uses 0.1,
FireANTs uses 0.01 for linear models and 0.1 for SyN, and ANTsPy uses 0.25 for
linear models and its default 0.2 for SyN. These rates have different optimizer
and coordinate meanings; other native presets are not substituted.
voxreg retains its production optimizer and
[precision contract](DESIGN.md#precision-output-and-execution).

ANTsPy uses the `ants.registration` Python argument-list interface, not an
external executable. Native image pointers avoid adding image-file writes:

- **Rigid/affine:** `Rigid[0.25]` / `Affine[0.25]`, global correlation (`GC`),
  regular sampling rate 1.0, shrink 1, image smoothing 0.
- **SyN:** `SyN[0.2,2.25,0.25]`, dense `CC` radius 2, shrink 1, image smoothing 0,
  no affine-fitting stage. The variances correspond nominally to field smoothing
  sigmas 1.5 and 0.5 voxels; the implementations are not identical.

All ANTs stages use single precision and a convergence window of 201, larger than
200 updates, so the native metric-convergence test cannot stop fitting early.
Accuracy-phase native logs must contain every update in order for every fitted
frame; a missing/short count fails the case. Timing/memory calls do not enable
verbose count logging, and do not claim directly observed counts. The configured
fixed-count policy is identical in every phase. Its GC/CC, boundary rules,
regularization and optimizer remain approximate counterparts, not identical methods.
FireANTs also uses its configured schedule for final inverse optimization; ANTs
refines inverse fields within outer iterations. Equal outer counts are not equal
computational work. The production `register` default remains 100 updates.

Complete fixed→moving pulls are scored through `ants.apply_transforms_to_points`
over **every fixed-grid centre**, explicitly applying the returned forward list
without implicit inversion. Scoring is outside timing. Transform files written
by `ants.registration` are part of the API call and **included** in timing;
cleanup is excluded. Accuracy transforms and failed-call files are retained.

### Resampling settings

All engines apply the same known physical map to original data using linear
interpolation. ANTsPy uses `ANTsTransform.apply_to_image`, not a registration or
CLI subprocess. Its transforms/normalized computation are float32; support is
computed by the adapter in float64 before quantization. voxreg preserves original
data dtype and its physical interpolation contract.

## Scorecard and protocol

| Metric | Registration | Resampling |
| --- | --- | --- |
| Runtime | Median **[min–max]** over five complete calls, first included | Same |
| Accuracy | Worst fitted-frame physical pull RMS **≤ 0.30 minimum-spacing voxels** | Worst-frame intensity RMSE / reference RMS **< 1e-5** |
| Memory | Peak process-group RSS and Torch-allocated CUDA VRAM | Same |

Three fresh workers perform timing, memory and accuracy independently. Imports
and initial CUDA context setup are outside timing; no operation is warmed up.
Backend conversion, fitting/application and CPU output return are timed. Memory
polling occurs only in its worker (20 ms; short peaks may be missed). Peak RSS
includes imports and loaded resident inputs; CUDA figures exclude driver/context,
reserved memory and non-Torch allocations. Accuracy covers the full workload.

The shared registration gate is operator-selected, supported by the
[matched single-resolution ANTsPy reference](research/syn/REFERENCE.md), not a
universal correctness standard. It applies to every registration model. The
resampling oracle is independent float64 SciPy interpolation on common physical
centre support, with validity-mask checks. No old/new or cross-device agreement
is used as an acceptance gate.

Current protocols are `voxreg-benchmark-v4` and `voxreg-resample-benchmark-v3`.
Registration v4 uses 200 fixed updates on all backends; earlier v3 registration
used 100 with native ANTs convergence enabled. Those results are not interchangeable.
Non-five-call runs are explicitly diagnostic. Settings, source/input hashes,
versions, host, power boundaries, actual batches, failures and phase logs are
recorded. A changed preparation/measurement policy requires new results, not
relabeling existing measurements.

## Lazy streaming benchmark

Lazy mode is voxreg-only rigid registration or known-map resampling, not SyN or
cross-backend rankings. Seeded float32 volumes are staged incrementally into
uncompressed one-volume Zarr blocks. A 16-pose cycle bounds preparation memory;
every frame is still processed. Sizes are explicitly opt-in:

```bash
# Eight-frame diagnostic smoke case; no ANTs environment required.
just benchmark-resample artifacts/lazy-smoke --lazy --devices cpu --runs 1

# Routine fixed-policy size sweeps; five timing calls per size by default.
# Use cpu instead, or omit --devices to run both CPU and CUDA.
just benchmark-register artifacts/lazy-register --lazy --size-gib 1 4 --devices cuda
just benchmark-resample artifacts/lazy-resample --lazy --size-gib 1 4 --devices cuda
```

Sized inputs default to **128 MiB** temporal chunks, aligned to whole volumes;
smoke chunks are smaller. Chunk size, fitting batch, threads and worker limits
stay fixed across compared input sizes. Worker-selected batches are pinned from
the first timing phase. Registration defaults to 200 updates without early stopping.
Advanced overrides remain available through `--lazy --help`; chunk sizing follows
[Dask's memory/storage/task-duration guidance](https://docs.dask.org/en/stable/array-chunks.html),
not a universal optimal-size claim.

Before **every pass**, completed Dask task data must be released. The runner
fsyncs its own input files, requests `POSIX_FADV_DONTNEED`, then verifies zero
resident file pages using Linux `mincore`. Eviction and verification are outside
timing, with per-pass evidence recorded. Unsupported or incomplete eviction
fails the case; there is no warm-cache fallback. Symlinks and unowned files are
rejected. No root privileges or global cache clearing are used. **tmpfs is not a
cold-input store**; use a disk-backed output directory. This verifies input file
pages, not cold directory metadata, CPU/GPU caches or storage-controller caches.

Fresh timing/memory/accuracy processes measure **open → graph → read → compute →
one DataTree Zarr write**. Cluster/device initialization, input preparation/cache
handling and successful-output deletion are excluded. No untimed operation warm-up
is added. Memory reports baseline, peak and peak-minus-baseline process-tree RSS;
20 ms sampling covers the operation, not cache preparation, and can miss short peaks. Per-worker allocated/reserved VRAM peaks are not
simultaneous device-wide totals. Every stored frame is checked against physical
truth (registration RMS ≤0.30 voxels) and SciPy pulls (relative RMSE <1e-5), with
support/geometry/time checks and selected eager comparisons. Failed outputs and
bounded failure evidence are retained; successful temporary outputs are removed
to avoid retaining five copies. Sources, inputs, settings, logs, `results.json`
and a separate `report.html` remain. The resident reporter does not combine this
`voxreg-lazy-benchmark-v3` protocol with its eager matrices. Lazy v3 defaults to
200 fitting updates; cold-input v2 results used the earlier default of 100.
Existing buffered-input
v1 results are preserved and must not be relabelled as cold-input measurements.

Compare 1/4 GiB RSS growth under the same execution policy; reports summarize
passing cases without inferring an automatic memory-scaling pass/fail. A small
size sweep provides evidence of bounded processing, not proof of execution
beyond physical RAM. An explicitly oversized input is an optional stronger
qualification, not a routine requirement. Reports record host RAM and worker
limits. Retained inputs plus one temporary result must fit disk. Avoid competing
jobs or concurrent reads of the input stores. Output writes remain buffered,
without forced durable-media fsync. Worker-loss retries are disabled; no OOM
retry or automatic chunk shrinking occurs.

## Reports

Each output directory contains `results.json`, a self-contained graphical
`report.html`, frozen sources/inputs and per-configuration phase logs. Failure
and incomplete outcomes remain visible. Graphs show all timing ranges and the
first call; speed ratios require completed accuracy-passing cases with identical
inputs and matched voxreg/FireANTs batches.

To render current results together without pooling hosts, revisions or operations:

```bash
just benchmark-report artifacts/performance-report.html \
  artifacts/benchmark-register-01 artifacts/benchmark-resample-01
```

Or run the reporter directly:

```bash
uv run --extra benchmarks python benchmarks/benchmark_report.py \
  artifacts/benchmark-register-01 artifacts/benchmark-resample-01 \
  --output artifacts/performance-report.html
```

The reporter accepts only the current protocols. Existing saved results are not
modified by the benchmark or reporter.
