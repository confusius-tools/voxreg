# Registration and resampling benchmarks

Two entry points, one serial supervision/reporting pipeline. Minimize runtime
subject to accuracy and fitting in memory; no composite score or cross-device
agreement gate.

## Run

```bash
# Inspect configurations without running anything.
uv run --extra examples python examples/benchmark_register.py --list
uv run --extra examples python examples/benchmark_resample.py --list

# ANTsPy needs its own environment; do not downgrade the project's dependencies.
uv venv --python 3.13 .venv-antspy
uv pip install --python .venv-antspy/bin/python 'antspyx==0.6.3'

# Selected workload; quote configuration globs. Output directories must be new.
uv run --extra examples python examples/benchmark_register.py \
  --configs 'resident-2d+t-*' --output artifacts/register-planar-01
uv run --extra examples python examples/benchmark_resample.py \
  --antspy-python .venv-antspy/bin/python \
  --configs 'resident-2d+t-*' --output artifacts/resample-planar-01

# Omit --configs for the default matrix (resampling: 45 resident API cases).
# CLI resampling is opt-in: --configs 'native-*'. Registration can take many hours.
export ANTSPATH="$PWD/artifacts/ants-install-2.6.5/ants-2.6.5/bin"
```

Requires Linux, FireANTs for its cases, and [ANTs 2.6.5](https://github.com/ANTsX/ANTs/releases/tag/v2.6.5)
on `PATH` or in `ANTSPATH` for native cases. No dataset preparation is needed.
Public options: `--output`, `--configs`, `--runs`, `--threads`, `--batch-size`,
`--list`, and resampling-only `--antspy-python`. Defaults: five timed calls and
eight CPU threads. Resampling gives voxreg/FireANTs the **same effective batch**
per workload/model/device, using voxreg's conservative cap unless overridden;
the first timing phase fixes it for later phases/backends. This is a controlled
comparison, not a claim of optimal throughput. ANTsPy applies one frame at a time.
Registration retains its existing backend batch policies. No resume, OOM retry
or runtime autotuning. Native paths cannot contain whitespace or shell/bracket
metacharacters. Run heavy cases serially; do not compete with an existing run.

## Scorecard and protocol

| Metric | Registration | Resampling |
| --- | --- | --- |
| Runtime | Median **[min–max]** over five complete calls, first included | Same |
| Accuracy | Worst fitted-frame physical pull RMS **< 0.25 minimum-spacing voxels** | Worst-frame intensity RMSE / reference intensity RMS **< 1e-5** |
| Memory | Peak process-group RAM and Torch-allocated CUDA VRAM; must fit | Same |

Each configuration uses **three fresh workers**: timing, memory and accuracy.
Memory polling and independent scoring do not perturb timing. Accuracy covers
the full workload with the same batching, not selected frames. Other `--runs`
values are explicitly **diagnostic**. Imports, synthetic/map generation and QC
are outside timing; initial CUDA context setup is excluded, but no operation is
warmed up. CUDA calls are synchronized.

Registration truth compares complete physical pulls over every fixed-grid voxel
center, not image similarity or maximum point error. Resampling uses independent
float64 SciPy linear interpolation at the supplied physical map's coordinates.
Intensity errors use common voxel-center support, excluding out-of-bounds padding;
resident validity masks are also checked against that support. Native CLI masks
are not exposed and are not claimed checked. This interpolation threshold is not
a registration-truth threshold.

Failures retain logs/timings. Execution or accuracy failure gives a nonzero exit;
unavailable CUDA is marked unavailable, not an execution failure. Invalid shapes,
dtypes and nonfinite outputs fail. Ctrl-C stops workers/native children, writes
partial reports and exits 130. A fast inaccurate case is not a winner.

## Workloads and boundaries

Shapes: `(1,150,128)` and `(80,150,80)`, in z/y/x order; spacing
`[0.2,0.1,0.1]` mm. Recordings have **100 frames**, with a declared 16-pose cycle
that reduces preparation cost, not evaluated frame count.

| Entry point | Resident voxreg/FireANTs CPU/CUDA | Native ANTs CPU | Total |
| --- | --- | --- | ---: |
| `benchmark_register.py` | 3 workloads × rigid/affine/SyN × 4 backends | 4 rigid-motion controls + staged pair + 3 direct SyN controls | 44 |
| `benchmark_resample.py` | Same 36 + 9 resident ANTsPy CPU cases | 9 opt-in CLI cases | 45 default; 54 including CLI |

**Registration:** resident calls include preprocessing, geometry, transfers,
fitting, original-image resampling and CPU return. They use one resolution and
100 updates with early stopping disabled; recordings preserve frame zero and
fit 99 frames. FireANTs retains float32 and Adam 0.01 for linear fits, 0.1 for SyN
with radius-2 CC; voxreg uses [production precision](DESIGN.md#precision-output-and-execution).

Native rigid motion fits all frames to the temporal mean with its 100×50×20
preset, dense or explicit random 5%. The staged 3D pair uses combined motion and
installed rigid→affine→SyN wrapper defaults: it is not the resident SyN task.
Direct native SyN uses the residents' nonlinear-only inputs, supplied fixed
anatomy, dense radius-2 CC, `SyN[0.1,3,0]`, `-f 1 -s 0vox` and 100 maximum updates.
Native convergence remains enabled; actual SyN update counts are unknown.

**Resampling:** known world-mm matrices or smooth displacement fields are prepared
once per workload/model, serialized outside measured workers; **no fitting occurs**.
Both Python environments load the same hashed original float32 images, float64
physical maps, lattice and time labels; file loading is outside timing. Resident timing includes uploads, map/grid
conversion, linear interpolation, validity and CPU return. Voxreg calls its public
`resample`; the FireANTs adapter calls its public interpolator with float32
normalized coordinates, deriving support **before float32 quantization**. It
records optional compiled-interpolator availability/use; without that extension
it still uses CUDA through PyTorch `grid_sample`, not a CPU fallback.

ANTsPy uses in-memory `ANTsTransform.apply_to_image`; nonlinear maps become
in-memory transforms through `transform_from_displacement_field`. Image/map
conversion, application, physical support masks, original-minimum fill and NumPy
return are timed. No transform filenames, CLI calls or per-frame subprocesses.
ANTsPy 0.6.3 uses float32 transforms here (including its mandatory float32 field
conversion); package/Python versions are recorded separately. Its NumPy `<2.4`
and SciPy `<1.16` constraints and lack of Python 3.14 wheels require isolation.

Native resampling invokes `antsApplyTransforms` once per frame, including CLI
startup and input/output file I/O; it uses double computation and float32 output.
Prepared files/map serialization and output loading/QC are outside timing.
ITK's voxel-extent boundary behavior differs from resident center support, hence
common-center intensity scoring. Random sampling and staging have no resampling
counterparts. No inverse-field computation is included: maps are already supplied.

**Resident and native tables stay separate**, not a mixed speed ranking. Native
registration also differs in reference, schedules and stopping. Torch/ITK use the
requested thread limit; other BLAS/OpenMP pools use one. `antsMotionCorr -n` means
image count, not threads.

## Outputs and limits

- Rich terminal scorecards show actual resident batch sizes. `benchmark.log` saves
  the transcript; `report.html` is self-contained offline HTML.
- `results.json` records policy, every timing, status, input/map/source hashes,
  revision/dirty state, versions, hardware and settings.
- `<configuration>/` contains exact config and separate timing/memory/accuracy
  JSON/logs, native commands and failure details.
- `source/` freezes production code, the invoked entry point and internal helpers
  used by workers. The underscore-prefixed helpers are not additional runners.

RAM is sampled every 20 ms, including imports, loaded inputs and children;
shared pages may count twice and short peaks can be missed. Resampling excludes
synthetic/map generation from measured workers; it also records loaded-input
baseline RSS and sampled peak excess. The excess is not an exact workspace size. VRAM is Torch allocated,
not driver/context/reserved memory or non-Torch allocations. Native accuracy
outputs remain; successful timing/memory image outputs are removed. Leave disk/RAM
headroom: 3D+t maps are large (native registration SyN accuracy fields alone ~6.14 GB).

Historical reports keep their original source/settings, entry-point names and
protocols (`voxreg-benchmark-v2` for registration; current resampling is
`voxreg-resample-benchmark-v2`). Resampling v1 used backend-specific batches,
in-worker generation and float32 FireANTs support checks; do not merge/relabel
those results as v2. Correctness tests remain separate;
use bounded direct runner checks, not another benchmark/example test framework.
