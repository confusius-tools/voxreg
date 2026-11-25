# Performance

Minimize whole-call runtime while passing the accuracy gates and fitting in
memory. [BENCHMARK.md](BENCHMARK.md) defines the protocol and commands for
**voxreg, FireANTs and ANTsPy**. Run registration and resampling serially in
fresh output directories; each matrix has 45 configurations.

Report five complete calls (first included), median [min–max], first-call time,
peak RAM/VRAM and accuracy. Timing, memory and accuracy use separate workers
with identical staged inputs. Pin voxreg/FireANTs batching across phases;
ANTsPy processes one frame at a time. Keep hosts, revisions and operations
separate. Failed accuracy gates cannot qualify as speed improvements.

Registration benchmarks now use **200 fixed outer updates** on all backends
(eager registration v4; lazy v3). ANTs accuracy logs verify actual counts with a
convergence window larger than the budget; timing/memory omit verbose logging.
The coruscant baseline below used the earlier 100-update policy and is not a
measurement of this new configuration. Production registration defaults are unchanged.

## Current implementation

- Compact physical sampling maps, cached CPU queries and single-volume partitioning.
- Float32 SyN fitting images, NCC and working fields with native `grid_sample`.
- Float64 master geometry, linear parameters, final inversion and field export.
- Direct Jacobians for exact axis permutations/reversals; general oblique/sheared
  fallback and candidate rejection checks.
- Final sampling of original intensities with dtype preservation.

See the [precision contract](DESIGN.md#precision-output-and-execution).

The separate [lazy streaming benchmark](BENCHMARK.md#lazy-streaming-benchmark)
measures verified cold-input read/compute/buffered-Zarr writes, with isolated
timing, memory and accuracy phases. Routine `--size-gib 1 4` sweeps hold the
128 MiB chunk and fitting/worker policy fixed to compare baseline/peak RAM.
GiB-scale inputs remain opt-in; oversized inputs are a separate stronger check.

## Mac CPU comparison

On 2026-11-24 we compared `d25645d` (before the twelve roadmap fixes) with
`051f60d` on the same arm64 Mac. Both revisions used Python 3.14.10, PyTorch
2.14.1 and four CPU threads. Times below are milliseconds per call, reported as
median [min–max] across five warm calls. All four workloads passed their accuracy
checks on both revisions. [Commands and workload details](BENCHMARK.md#quick-cpu-comparison-across-commits)

| Workload | Before (ms) | After (ms) | Change in median |
| --- | ---: | ---: | ---: |
| Rigid motion correction, four slices | 276 [269–287] | 290 [277–364] | +4.9% |
| SyN, small volume pair | 1,534 [1,431–1,682] | 1,507 [1,440–1,820] | −1.8% |
| Linear resampling, float32 volume | 25.45 [24.81–26.72] | 28.28 [26.59–30.90] | +11.1% |
| Linear resampling, int64 volume | 26.88 [26.26–29.47] | 46.58 [44.56–49.95] | +73.3% |

The registration timing ranges overlap. The clearest slowdown was fractional
linear resampling of int64 images. From the parent of `b177ca6` to that commit
alone, the median rose from 25.53 to 46.36 ms (+81.5%). That commit changed how
integer images are sampled to preserve precision. Both revisions produced the
expected rounded values and support mask on this fractional translation.

Repeating the comparison with two CPU threads gave +12.2% for float32
resampling and +71.9% for int64. Registration varied more. The two-thread
medians rose 9.9% for rigid and 15.4% for SyN, with overlapping timing ranges
for both models.

We also checked 3D SyN accuracy separately on the standard 80 × 150 × 80 grid
at `051f60d`. With one frame per batch and 200 updates per fit, the pair's
root-mean-square transform error was **0.2318 voxels**. The recording's first
frame stayed unchanged, and the next two frames measured **0.2036** and
**0.2180**. Each passed the existing 0.30-voxel limit. The full 100-frame
recording, CUDA, memory use and the other libraries were outside this Mac run.
[Individual timings and accuracy results](research/performance/mac-cpu-051f60d.json)

### Linear resampling follow-up

We reran the same four workloads after optimizing CPU resampling. Each column
below compares a baseline with the working tree on the same Mac. Values are
medians of five calls in milliseconds. Both revisions passed every accuracy check
in each run.

| Workload | Before the roadmap fixes → optimized (4 threads) | Before the speed fix → optimized (2 threads) |
| --- | ---: | ---: |
| Rigid motion correction, four slices | 267.28 → 269.07 | 273.01 → 272.82 |
| SyN, small volume pair | 1,384.34 → 1,387.61 | 1,402.82 → 1,415.71 |
| Linear resampling, float32 | 24.55 → 7.05 | 31.54 → 9.60 |
| Linear resampling, int64 | 25.48 → 10.86 | 45.47 → 11.49 |

The four-thread baseline was `d25645d`. The two-thread baseline was `051f60d`,
which already included the precision fixes. Affine resampling now builds
coordinates from three short axis vectors. For the benchmark's one-axis shift,
integer interpolation reads only the two contributing neighbors. It still uses
voxel coordinates for exact rounding. [Individual timings and accuracy results](research/performance/mac-cpu-linear-resampling.json)

In a separate four-thread spot check, we timed int64 pulls that combined rotation
and translation. The median went from 25.85 to 22.07 ms for a matrix pull and
from 40.29 to 37.13 ms for a dense displacement field, comparing
`d25645d` with the optimized working tree.

For a three-axis fractional shift of an 80 × 150 × 80 int64 volume, a single
fresh process per version reached 403, 500 and 407 MiB of peak memory for the
matrix pull at `d25645d`, `051f60d` and the optimized working tree respectively.
The corresponding displacement-field runs reached 470, 565 and 447 MiB. These
whole-process readings include the Python environment and input data.

### CPU profile of a dense warp on 100 slices

At `9574708`, applying the existing benchmark's displacement field to 100
float32 slices (150 × 128, batches of 32, four Mac CPU threads) took 26.16 ms
in voxreg and 13.05 ms in the FireANTs CPU adapter. Both passed the SciPy
intensity check and matched its validity mask. These are medians of 14 warm
calls per backend. Alternating instrumented calls took 26.05 and 13.04 ms.

| Measured work | voxreg (ms) | FireANTs adapter (ms) |
| --- | ---: | ---: |
| Construct sampling positions | 7.73 | 8.99 |
| Build validity mask | 3.17 | 1.77 |
| Detect and restore exact voxel centers | 4.92 | — |
| Prepare sampler grid and image dtype | 4.01 | Included below |
| Interpolate | 1.81 | 1.59 |

Each row is a separate median. Whole-call totals also include setup, input
checks, copies and output assembly. The FireANTs interpolation measurement
includes casting its grid to float32. Voxreg samples with float64 coordinates.

In a separate experiment, forming all the x positions together, then y and z,
reduced the slice replay from 26.31 to 19.24 ms. The same change **slowed a
3D volume** (80 × 150 × 80) from 27.33 to 40.78 ms. These are medians of 12
paired warm calls per implementation. Both produced identical images and masks
on the two workloads. The measured opportunity is specific to planar CPU replay.

## Coruscant baseline

Reviewed on 2026-11-12: i9-14900K, RTX 5070, eight CPU threads. Eager revision
`052488f` passes 37/45 registration cases and all 45 resampling cases. The eight
registration failures are 3D SyN on voxreg/FireANTs CPU/CUDA, for pairs and
recordings; these are accuracy failures, not execution errors. ANTsPy passes
those cases. Planar CUDA SyN remains qualified: voxreg 1.436 s versus FireANTs
1.130 s (about 27% slower). Do not combine this with the local profile below.

Lazy revision `2b1c787` fixes broadcast-zero templates expanding into input-sized
serialized graphs. Both complete sweeps pass under protocol v2, with 64-frame
128 MiB tasks, fitting batches of 16, one 4 GiB worker and unchanged policy across
sizes. Times include metadata/graph construction, cold reads, compute and one
buffered output write; cache handling is excluded.

| Operation | Input GiB | Median seconds | Peak process-tree RAM GiB | GPU allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Rigid registration | 1 | 41.147 | 3.566 | 0.744 |
| Rigid registration | 4 | 164.921 | 3.966 | 0.744 |
| Known-map resampling | 1 | 1.765 | 2.319 | 0.676 |
| Known-map resampling | 4 | 6.753 | 2.376 | 0.676 |

Baseline process-tree RSS is about 1.45 GiB. Fourfold input growth costs 4.01x/3.83x
runtime and 11.2%/2.5% more peak RAM for registration/resampling; GPU allocation
is unchanged. Brief registration worker pauses still occur at 4 GiB and resume;
there are no input-sized graph warnings or worker restarts in the completed runs.
This supports bounded working-state scaling over the measured range, not a
universal or larger-than-RAM qualification. See [validation](VALIDATION.md#coruscant-benchmark-qualification).

Evidence under `artifacts/from-coruscant/`:
- `coruscant-20261110-190108-{register,resample}-eager/`
- `coruscant-lazy-fixed-20261112-025518-{register,resample}/`

The original stalled lazy run and the interrupted `025404` fixed attempt remain
preserved but are excluded from completed-run statistics.

## Latest local SyN profile

The [2026-11-11 investigation](research/syn/PROFILE-20261111.md) measures a
qualified planar CUDA gap: voxreg 1.867 s versus FireANTs 1.420 s, with matched
batching and accuracy gates. Both volume CUDA arms fail the 0.30-voxel gate and
remain unqualified. Small cache/batching/buffer prototypes did not establish a
worthwhile safe improvement; no production optimization was promoted.

## Fixed-work pilot

The 200-update CPU 3D-pair pilot verifies all three actual accuracy counts as 200.
With native SyN rates (voxreg/FireANTs 0.1, ANTsPy 0.2), truth RMS is 0.232014
for voxreg, 0.303709 for FireANTs and 0.183072 for ANTsPy. The ANTs-only accuracy
recheck is in `artifacts/fixed-count-200-native-rate-pilot/`; its previous
step-0.1 pilot reached 0.204105. FireANTs remains above the unchanged 0.30 gate. One timed call per backend is
only diagnostic; no new speed ranking or full-matrix qualification is claimed.
See `artifacts/fixed-count-200-pilot/report.html` and
[validation](VALIDATION.md#fixed-work-200-update-benchmark-pilot).

## Next measurements

Resolve 3D SyN accuracy before claiming qualified volume performance. Attribute
remaining SyN costs with controlled diagnostics, then retain changes only when
repeated measurements improve runtime without weakening accuracy, support or
precision. Rerun affected workloads after changes rather than repeating the
completed matrices without a new question.
ANTsPy uses its Python API and retains its own optimizer, with metric-convergence
stopping disabled for the fixed-work benchmark. Matching outer iteration counts
does not establish algorithmic equivalence or equal inner inversion work.
