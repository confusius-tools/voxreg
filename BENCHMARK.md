# Registration benchmark

**One command, three metrics:** minimize runtime, subject to sufficient truth
accuracy and fitting in memory. CPU/CUDA map agreement, old/new map agreement and
image correlation are **not benchmark acceptance gates**.

## Run

```bash
# Inspect the 44 configuration IDs without running anything.
uv run --extra examples python examples/benchmark.py --list

# Start with a selected workload; quote configuration globs.
uv run --extra examples python examples/benchmark.py \
  --configs 'resident-2d+t-*' --output artifacts/benchmark-planar-01

# Full matrix, including native ANTs (can take many hours).
export ANTSPATH="$PWD/artifacts/ants-install-2.6.5/ants-2.6.5/bin"
uv run --extra examples python examples/benchmark.py --output artifacts/benchmark-01
```

Requires Linux and, for native cases, [ANTs 2.6.5](https://github.com/ANTsX/ANTs/releases/tag/v2.6.5)
on `PATH` or in `ANTSPATH`. FireANTs must be installed for its cases. No dataset
preparation is needed. The output directory **must be new**.

Public options are only `--output`, `--configs`, `--runs`, `--threads`,
`--batch-size`, and `--list`. Defaults are five timed calls and eight CPU threads.
`--batch-size N` overrides both resident engines; otherwise each engine/device
uses its own batch policy. There is no resume, OOM retry or runtime autotuning.
Native output paths cannot contain whitespace or shell/bracket metacharacters.

## Scorecard

| Metric | Definition | Role |
| --- | --- | --- |
| Runtime | Median **[min–max]** seconds over five complete registrations, first included | Minimize among accuracy-passing cases |
| Truth error | **Worst fitted frame's RMS** mapped-point error, divided by minimum active voxel spacing | Must be **< 0.25 voxels** |
| Memory | Peak process-group RAM and peak Torch-allocated CUDA VRAM | Must fit; allocation/execution failures fail the case |

Truth RMS compares physical fixed→moving pulls against the independent generating
transform over every fixed-grid voxel center—not the largest point error or image
similarity. Linear scoring uses exact grid covariance; nonlinear scoring evaluates
the full grid. Devices are scored independently; there is no composite score.

Each configuration runs serially in **three fresh workers**: five timing calls,
one memory call and one accuracy call. Memory polling and transform export/truth
scoring therefore do not perturb timing. The accuracy call fits the whole
workload, with the same batching—not a few selected frames. Other `--runs` values
are explicitly labeled **diagnostic**, not five-run benchmark results.

Failures remain visible with their timings and logs; a fast but inaccurate case
is not a winner. Execution/accuracy failures produce a nonzero exit code;
unavailable CUDA is explicitly marked unavailable. Invalid/nonfinite outputs and
production transform guards remain checked. Ctrl-C stops the active worker and
native children, saves partial results and exits 130.

## Workloads and comparison boundaries

| Suite | Workloads | Models/workflows | Configurations |
| --- | --- | --- | ---: |
| Resident voxreg and FireANTs, CPU/CUDA | 2D+t, 3D pair, 3D+t | rigid, affine, SyN | 36 |
| Native ANTs CLI, CPU | 2D+t, 3D+t | temporal mean → rigid, dense and explicit random 5% | 4 |
| Native ANTs CLI, CPU | 3D pair | installed rigid → affine → SyN wrapper defaults | 1 |
| Native ANTs CLI, CPU | 2D+t, 3D pair, 3D+t | direct, single-resolution SyN | 3 |

Spatial shapes are `(1,150,128)` for planar data and `(80,150,80)` for volumes,
in z/y/x order, with spacing `[0.2,0.1,0.1]` mm. Recordings contain **100 frames**;
the analytic phantom's 16-pose cycle does not reduce the fitted frame count.
Resident recordings preserve frame zero and fit the other 99 frames. Resident
fits use one resolution and **100 updates**, with early stopping disabled.

Resident time includes preprocessing, geometry, transfers, fitting, original-image
resampling and CPU return; imports, synthetic generation, QC and scoring are
outside the timer. CUDA is synchronized, with initial context setup outside the
timer but no warm-up registration. Native time includes CLI startup and image
I/O/compression. **Resident and native workflows have separate tables**, not a
mixed speed ranking: their references, fitting schedules and I/O differ.

Native rigid motion fits all frames to their temporal mean with its 100×50×20
preset; dense is the default, random 5% an explicit additional control. The staged
pair contains combined affine/nonlinear motion and is not the resident SyN task.
Direct native SyN uses the same nonlinear-only inputs as residents, dense radius-2
CC, `SyN[0.1,3,0]`, `-f 1 -s 0vox` and 100 **maximum** iterations. Native convergence
remains enabled; actual SyN update counts are unknown. Its supplied fixed anatomy
is frame zero, but it fits/scores all 100 frames. Native pull direction, geometry
and composition are validated before scoring.

Voxreg uses its [production precision](DESIGN.md#precision-output-and-execution).
FireANTs retains native float32, Adam 0.01 for linear fits and 0.1/radius-2 CC for
SyN; algorithms are not identical. Thread limits apply to Torch/ITK; other
BLAS/OpenMP pools are limited to one. `antsMotionCorr -n` means images, **not threads**.

## Outputs and logs

- **Terminal:** Rich phase progress and color-coded scorecards, including the
  actual resident batch size. **`benchmark.log`** retains the terminal transcript;
  **`report.html`** is its self-contained, offline Rich HTML rendering.
- **`results.json`:** incremental results, all timing samples, policy, statuses,
  input/source hashes, revision/dirty state, versions, hardware, tools and settings.
- **`<configuration>/`:** exact config and separate `timing`, `memory`, `accuracy`
  JSON/log files. Native commands and tool logs remain, including failures.
- **`source/`:** frozen production code and benchmark helpers used by every worker.
  The underscore-prefixed helpers are implementation details, not extra runners.

RAM is sampled every 20 ms, including imports/inputs/children; shared pages may be
counted twice and very short peaks missed. VRAM is Torch **allocated**, not driver
or allocator-reserved memory. Native accuracy images/fields remain; successful
timing/memory image outputs are removed. Native 3D+t SyN accuracy fields alone are
about **6.14 GB uncompressed**: leave disk/RAM headroom.

Completed historical reports retain their original source/settings and must not
be relabeled as current measurements. Preliminary probes are compactly archived
under `artifacts/archive/`; their generated bulk arrays/images are discarded.
Library correctness tests stay separate from benchmarking. Use bounded direct
checks for this script, not another benchmark/example test framework.
