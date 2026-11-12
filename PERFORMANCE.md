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

## Next measurements

Resolve 3D SyN accuracy before claiming qualified volume performance. Attribute
remaining SyN costs with controlled diagnostics, then retain changes only when
repeated measurements improve runtime without weakening accuracy, support or
precision. Rerun affected workloads after changes rather than repeating the
completed matrices without a new question.
ANTsPy uses its Python API and retains its own optimizer and convergence;
matching iteration budgets does not establish algorithmic equivalence.
