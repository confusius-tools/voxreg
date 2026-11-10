# Registration and resampling benchmarks

Compare **Python APIs only**: voxreg, FireANTs and ANTsPy. No ANTs command-line
executables or `ANTSPATH` are used. ANTsPy calls the ANTs library in process.

## Setup and commands

Use the project environment for supervision, voxreg and FireANTs. ANTsPy 0.6.3
needs its own Python 3.13 environment; do not downgrade project dependencies.

On Linux, the Just recipes create or reuse that environment automatically:

```bash
just antspy-venv  # Optional: each benchmark also runs this dependency.
just benchmark-resample artifacts/benchmark-resample-01
just benchmark-register artifacts/benchmark-register-01
```

Run the benchmarks sequentially and use a fresh output directory for each run.
Additional options can follow the output directory, for example
`just benchmark-register artifacts/register-diagnostic --runs 1 --configs '*-antspyx'`.
The defaults are five timed calls and eight threads. The equivalent manual setup
and commands are:

```bash
uv sync --extra examples
uv venv --python 3.13 .venv-antspy
uv pip install --python .venv-antspy/bin/python 'antspyx==0.6.3'

# Inspect configurations without running them.
uv run --extra examples python examples/benchmark_register.py --list
uv run --extra examples python examples/benchmark_resample.py --list

# Run serially, with a fresh output directory for each invocation.
uv run --extra examples python examples/benchmark_resample.py \
  --antspy-python "$PWD/.venv-antspy/bin/python" \
  --runs 5 --threads 8 --output artifacts/benchmark-resample-01

uv run --extra examples python examples/benchmark_register.py \
  --antspy-python "$PWD/.venv-antspy/bin/python" \
  --runs 5 --threads 8 --output artifacts/benchmark-register-01
```

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
voxreg and FireANTs request 100 updates with early stopping disabled. Their linear
objectives use masked squared global NCC; SyN uses radius-2 local CC with update
sigma 1.5 and total sigma 0.5 voxels. FireANTs uses Adam, learning rate 0.01 for
linear fits and 0.1 for SyN. voxreg retains its production optimizer and
[precision contract](DESIGN.md#precision-output-and-execution).

ANTsPy calls `ants.registration` with `singleprecision=True`:

- **Rigid/affine:** `Rigid` / `Affine`, global correlation (`GC`), sampling rate
  1.0, `aff_iterations=(100,)`, shrink `(1,)`, image smoothing `(0,)`.
- **SyN:** `SyNOnly`, `reg_iterations=(100,)`, `grad_step=0.1`, `flow_sigma=2.25`,
  `total_sigma=0.25`, `syn_metric="CC"`, `syn_sampling=2`. The variances correspond
  to field smoothing sigmas 1.5 and 0.5 voxels; there is no affine prefix.

ANTsPy convergence remains enabled: its budget is **at most 100 updates**, not a
claim of 100 observed updates. The Python API does not return iteration counts;
workers record unknown counts rather than invent them. Its GC/CC, boundary rules,
regularization and optimizer are approximate counterparts, not identical methods.

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

Current protocols are `voxreg-benchmark-v3` and `voxreg-resample-benchmark-v3`.
Non-five-call runs are explicitly diagnostic. Settings, source/input hashes,
versions, host, power boundaries, actual batches, failures and phase logs are
recorded. A changed preparation/measurement policy requires new results, not
relabeling existing measurements.

## Lazy streaming benchmark

`benchmark-lazy` is voxreg-only and initially tests rigid motion, not SyN or
cross-backend rankings. It writes seeded analytic float32 volumes incrementally
into uncompressed, one-volume Zarr blocks; a declared 16-pose cycle bounds
preparation memory without skipping any frame. Large files are opt-in:

```bash
# Eight-frame diagnostic smoke case; no ANTs environment required.
just benchmark-lazy artifacts/lazy-smoke --operation resample --devices cpu --runs 1

# Run separately in fresh directories; defaults to five complete calls.
just benchmark-lazy artifacts/lazy-register-10g --operation register --size-gib 10
just benchmark-lazy artifacts/lazy-resample-10g --operation resample --size-gib 10
```

Default large-input Dask chunks target **32/128 MiB**, rounded to whole volumes
and aligned with storage. This follows [Dask's guidance](https://docs.dask.org/en/stable/array-chunks.html):
fit comfortably in memory, commonly 10 MB–1 GB, and tasks substantially longer
than 100 ms. Reports include actual chunk bytes and bounded task-duration
samples, not a claim that these sizes are optimal. Smoke chunks are deliberately
smaller. Overrides: `--chunk-mib`, `--shape K J I`, `--workers`,
`--worker-memory-gib`, `--threads`, `--batch-size`, `--devices` and `--iterations`.
Fitting batches are selected in workers, then pinned across phases; default
registration is 100 updates without early stopping.

Fresh timing/memory/accuracy processes measure **open → graph → read → compute →
one DataTree Zarr write**. Cluster/device initialization, generation and deletion
of successful temporary outputs are excluded; peak process-tree RSS is sampled
only during the memory phase. Per-worker allocated/reserved VRAM peaks are not
simultaneous device-wide totals. Every stored frame is checked against physical
truth (registration RMS ≤0.30 voxels) and SciPy pulls (relative RMSE <1e-5), with
support/geometry/time checks and selected eager comparisons. Failed outputs and
bounded failure evidence are retained; successful temporary outputs are removed
to avoid retaining five copies. Sources, inputs, settings, logs, `results.json`
and a separate `report.html` remain. The resident reporter does not combine this
`voxreg-lazy-benchmark-v1` protocol with its older matrices.

**10 GiB is a streaming test, not necessarily larger than host RAM.** Use an
explicit larger size (e.g. `--size-gib 40` on a 32 GiB host) for that qualification;
the report records both host RAM and worker memory limits. Keep sufficient disk
for inputs plus one result, avoid competing heavy jobs, and do not describe these
runs as cold-cache or durable-media latency measurements: OS caches are not
flushed and writes use normal OS buffering without forced fsync. Worker-loss
retries are disabled; no OOM retry or automatic chunk shrinking occurs.

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
uv run --extra examples python examples/benchmark_report.py \
  artifacts/benchmark-register-01 artifacts/benchmark-resample-01 \
  --output artifacts/performance-report.html
```

The reporter accepts only the current protocols. Existing saved results are not
modified by the benchmark or reporter.
