# Preliminary performance research

These are **diagnostic harnesses, not a second registration API**. They temporarily
substitute a sampler or attach profiler scopes inside their own process. Nothing
here is imported by the engine. See [PERFORMANCE.md](../../PERFORMANCE.md) for the
findings and [BENCHMARK.md](../../BENCHMARK.md) for the backend comparison.

[`results.json`](results.json) preserves compact historical measurements, dataset
hashes, hardware/versions, kernel counts, quality, batch agreement and test outcomes.
It deliberately omits original recordings, registered arrays, large traces,
process IDs and machine-local paths. Its 43-test counts describe the suite when
the measurements were captured, before the research CLI cleanup tests were added.
The snapshot is not regenerated automatically when a command runs.

## Reproduce from the repository root

Install the locked environment with `uv sync --extra examples`. Prepare
`artifacts/notebook/input.npz` with the [NIfTI example](../../README.md), or provide
another compatible power NPZ via `--input`. The benchmark input loader currently
requires positive, axis-aligned single-slice geometry. Synthetic unit tests cover
more general geometry separately.

Run configurations **sequentially on an otherwise idle machine**. Commands below
use a fresh output directory so they do not overwrite the original local evidence:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Matched warm-call timing; default is baseline versus elementwise float64.
uv run python research/performance/affine_probe.py cpu --threads 1 \
    --output artifacts/performance-reproduction
uv run python research/performance/affine_probe.py cpu --threads 8 \
    --output artifacts/performance-reproduction
uv run python research/performance/affine_probe.py cuda \
    --output artifacts/performance-reproduction

# Optional rejected affine_grid shortcuts; not expected to pass every test.
uv run python research/performance/affine_probe.py cuda \
    --variants affine64 affine32 elementwise64 \
    --output artifacts/affine-grid-reproduction

# Run the existing test suite with the proposed sampler substituted.
uv run python research/performance/validate_probe.py elementwise64

# Generate fresh baseline/variant outputs, test batch invariance, score common support.
# No dependency on a previous probe or an ITK/FireANTs result directory.
uv run python research/performance/check_elementwise.py --devices cpu cuda \
    --output artifacts/performance-reproduction

# Instrumented steady-state captures: never use their wall times as benchmarks.
uv run python research/performance/profile_baseline.py voxreg cpu --threads 8 --batch 32 \
    --output artifacts/profile-reproduction
uv run python research/performance/profile_baseline.py fireants cpu --threads 8 --batch 32 \
    --output artifacts/profile-reproduction
uv run python research/performance/profile_baseline.py voxreg cuda --batch 119 \
    --output artifacts/profile-reproduction
uv run python research/performance/profile_baseline.py fireants cuda --batch 119 \
    --output artifacts/profile-reproduction
uv run python research/performance/profile_baseline.py voxreg cuda --batch 119 \
    --sampler elementwise64 --output artifacts/profile-reproduction
```

For CPU-only hosts, omit CUDA commands and use `check_elementwise.py --devices cpu`.
Each command supports `--help`; input/output paths can be explicit even when invoked
outside the repository. `validate_probe.py` locates the repository tests itself.
The timing probe also supports `--repeats` (default 3). Its output summaries identify
which variants were actually run; all large outputs remain under ignored `artifacts/`.

The cleanup makes the checker self-contained: it now regenerates baseline results
and compares only those with the prototype, rather than requiring undocumented
historical affine-grid, ITK and FireANTs NPZs. The broader historical common-support
comparison is retained in `results.json`; rerun the backend comparison separately
using the command in `BENCHMARK.md`.

No production optimization, compilation or new precision policy is enabled here.
