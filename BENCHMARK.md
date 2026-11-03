# Registration benchmark

**One entry point: `examples/benchmark.py`.** No dataset download or preparation
runner is needed. Linux is required for process-tree memory accounting.

```bash
# After installing native ANTs 2.6.5, expose its bin directory:
export PATH="$PWD/artifacts/ants-install-2.6.5/ants-2.6.5/bin:$PATH"

uv run --extra examples python examples/benchmark.py \
  --output artifacts/benchmark-01
```

The output directory **must not exist**. Five timed registrations per configuration
is the default. Other `--runs` values are explicitly labeled **diagnostic** in the
terminal and JSON; they are not the standard five-run scorecard. The full suite is intentionally serial and can take many hours:
the historical dense 3D+t ANTs call alone took about 36 minutes. There is no
resume, hidden retry, work reduction or runtime autotuning. Ctrl-C terminates the
active worker and its native children and records an interrupted run.

## Configurations

The routine matrix contains **41 configurations**:

| Suite | Workloads | Models / workflows | Backends | Configurations |
|---|---|---|---|---:|
| Resident arrays | 2D+t, 3D pair, 3D+t | rigid, affine, SyN | voxreg CPU/CUDA, FireANTs CPU/CUDA | 36 |
| Native ANTs CLI | 2D+t, 3D+t | temporal mean → rigid; dense/default and explicit random 5% | CPU | 4 |
| Native ANTs CLI | 3D pair | installed rigid → affine → SyN wrapper defaults | CPU | 1 |

Translation and the old single-resolution ITK adapter are not in this matrix.
This does not remove any production registration model or numerical test oracle.
Native ANTs **dense is the default motion workflow**; random 5% is an additional,
explicitly named configuration, never relabeled as the default.

Native voxel order is elevation × axial × lateral; spacing is
`[0.2, 0.1, 0.1]` mm. The fixed inputs are:

- **2D+t:** 100 × 1 × 150 × 128.
- **3D pair:** 80 × 150 × 80.
- **3D+t:** 100 × 80 × 150 × 80.

A continuous analytic Gaussian/Fourier phantom is painted independently at each
known pose, including reverse-order Newton inversion of nonlinear truth. The
16-pose cycle bounds generation cost, **not the number of fitted frames**. A
recording is stored in full: the 3D+t float32 input alone occupies 384 MB. Dask
inputs and streamed outputs are not supported.

Resident recordings preserve frame zero exactly and fit all remaining frames.
Native motion correction instead fits all frames to its temporal mean. The staged
native pair contains both known affine/rigid motion and known nonlinear motion;
it is not the same fitting task as the resident nonlinear-only pair.

## Three reported metrics

1. **Wall time: median [min–max] of five complete registrations.** The first call
   is included; there is no separate warm-up or first-call score. Every config has
   a fresh worker; its repetitions reuse that worker and the same resident inputs.
2. **Peak RAM and CUDA VRAM from a separate registration.** RAM is the sampled
   process-group RSS peak (20 ms), including worker imports, input preparation,
   resident inputs and native children. Shared pages can be counted more than
   once, and short-lived peaks can be missed. VRAM is peak **Torch allocated**
   memory, including input transfers and intermediates, not reserved allocator
   memory, driver/context overhead or non-Torch allocations. CPU VRAM is `—`.
3. **Worst fitted-frame physical transformation RMS against independent truth.**
   A separate complete registration exports its actual workload transforms with
   the same batch policy; it does not refit selected probe frames. The RMS is over
   every fixed-grid voxel center, in mm, divided by minimum active spacing. It is
   not a per-axis index error or an image-correlation surrogate. The preserved
   resident reference is excluded from the worst fitted-frame score.

The accuracy screen remains **RMS < 0.25 minimum-spacing voxels**. Linear RMS is
computed analytically using the full-grid covariance; nonlinear RMS evaluates
all voxel centers. Native motion pulls are reconstructed from their Euler
parameters and raw mean moment center, then composed as `Ti @ inverse(T0)` to
compare in the common frame-zero reference. Native pair pulls compose the fixed-
domain displacement followed by the affine. This composition was independently
checked against the retained native resampled output.

Each config therefore performs **seven registrations**: five timing, one memory,
one accuracy. Timing/memory use `return_transforms=False`; export/scoring overhead
is not charged to them. A change in selected batch size between repetitions or
phases is an error rather than a quietly different configuration. Execution and
accuracy failures remain in the output; timings are not turned into passing
results when accuracy fails. The command exits nonzero for execution/accuracy
failures, while unavailable CUDA configurations are explicitly marked unavailable.

CPU/CUDA discrepancy, SyN diagnostics, throughput and phase/operator timings are
not scorecard columns. Production numerical checks, inversion and regularity
guards remain enabled and unchanged.

## Timing and numerical boundaries

Resident time includes preprocessing, geometry/image construction, transfers,
optimization, original-image resampling and CPU return. It excludes synthetic
input generation, library import, output checking, truth scoring and report
writing. CUDA is synchronized around the complete call; initial CUDA context
setup is outside the first timer, without running a warm-up registration.

Native time includes fresh CLI startup, image reads, preprocessing/optimization,
resampling and output/transform writes/compression. Input generation, QC and
reporting are excluded. Motion time sums the temporal-mean and rigid-motion CLI
calls. The native and resident suites have **separate labeled tables**: different
references, models, schedules, I/O and precision prevent a mixed speed ranking.

Default resident controls are eight CPU threads, one resolution and 100 updates,
with early stopping disabled for this fixed-budget experiment. Native motion
retains its 100×50×20 multiresolution preset; native staged registration retains
the installed wrapper's defaults. Exact commands are saved. Native ANTs 2.6.5
is required for the qualified Euler reconstruction and staged workflow.

Voxreg's production precision is unchanged: float32 fitting images and float64
geometry, state, NCC and final interpolation, except the previously qualified CPU
volumetric SyN smoother. FireANTs retains its native float32 computations and
optimizer settings: Adam 0.01 for linear models, 0.1 and radius-2 CC for SyN. Its
adapter defaults to 32 linear/planar fits or four volumetric SyN fits. Voxreg
retains its production automatic batching. `--batch-size N` overrides both resident
adapters with the same explicit batch limit for controlled comparisons; it does
not change native ANTs workflows. Spatial pairs still contain only one image and
tail batches can be smaller. Without the override, existing backend policies are
unchanged. A batch that does not fit fails explicitly: no smaller-batch retry.
These are not precision-matched algorithms.

## Select or inspect without running

```bash
# IDs only; no registrations and no output directory needed.
uv run --extra examples python examples/benchmark.py --list

# Match voxreg and FireANTs CUDA batch limits; quote the glob.
uv run --extra examples python examples/benchmark.py \
  --configs 'resident-3d+t-rigid-*cuda' --batch-size 8 \
  --output artifacts/benchmark-rigid-b8-01
```

The public controls are only `--output`, `--configs`, `--runs`, `--threads`,
`--batch-size` and `--list`. Workloads stay at 100 recording frames and the
established iteration presets; these are not CLI tuning knobs. `--threads`
defaults to eight. `--runs` defaults to five, plus the separate memory/accuracy
calls, even for diagnostics. Internal `--worker`/`--mode` arguments are hidden.

Native ANTs binaries come from `PATH` or a bin directory specified by `ANTSPATH`.
Native output paths must not contain spaces or shell/bracket metacharacters.
The former `--frames`, `--iterations`, `--ants-bin` and `--plan-only` flags were
removed in protocol `voxreg-benchmark-v2`. Use `--list` for non-executing inspection;
previous running jobs continue using their frozen v1 source without modification.

## Evidence and historical material

`results.json` contains the three metrics plus status and provenance: source and
input hashes, commit/dirty state, exact command, package versions, hardware,
power state, chosen batching and native tool hashes. Production Python sources
and this entry point's internal helpers are frozen under `source/`; every worker
uses that frozen source. Timing values are updated after each call in each
configuration's `timing.json`. Logs retain tracebacks/native failures. Native
accuracy outputs are retained; successful timing/memory image outputs are deleted
to avoid accumulating large recordings. Inputs remain available for inspection.

The old benchmark runners, profiler experiments and tests of those removed
scripts have been removed. Production regression coverage and the independent
stopping-policy calibration reference remain. **No new benchmark/example tests
are required:** use bounded direct checks for this runner and the production
suite for library contracts.

[Research records](research/README.md), [CPU volume qualification](research/performance/CPU-VOLUME.md),
[completed native ANTs evidence](research/registration/ANTS.md) and
[known synthetic SyN failures](research/syn/SYNTHETIC-PHANTOM.md) remain historical
evidence, not results from this new pipeline. Do not compare their one-call
numbers as if they were the new five-run scorecard.
