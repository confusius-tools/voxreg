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

## Next measurements

Rerun both complete matrices to establish the scorecard for this revision.
Profile the largest whole-call gap, then retain changes only when repeated
measurements improve runtime without weakening accuracy, support or precision.
ANTsPy uses its Python API and retains its own optimizer and convergence;
matching iteration budgets does not establish algorithmic equivalence.
