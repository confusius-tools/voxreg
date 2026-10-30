# Preliminary performance research

These are **diagnostic harnesses, not a second registration API**. They temporarily
substitute a sampler or attach profiler scopes inside their own process. Nothing
here is imported by the engine. The engine now includes the scalar-affine rewrite;
[sampler-results.json](sampler-results.json) records its production benchmarks.
`results.json` remains the historical pre-optimization snapshot. See [PERFORMANCE.md](../../PERFORMANCE.md) for the
findings and [BENCHMARK.md](../../BENCHMARK.md) for the backend comparison.

[`results.json`](results.json) preserves compact historical measurements, dataset
hashes, hardware/versions, kernel counts, quality, batch agreement and test outcomes.
It deliberately omits original recordings, registered arrays, large traces,
process IDs and machine-local paths. Its 43-test counts describe the suite when
the measurements were captured, before the research CLI cleanup tests were added.
The snapshot is not regenerated automatically when a command runs. A harness's
"baseline" means the currently imported engine: comparisons to the old engine
require commit `83b6dd6` or the archived-source `PYTHONPATH` recipe in BENCHMARK.md.
Comparing the current engine with the elementwise prototype should no longer
reproduce the historical 5× improvement.

## Current batch-size policy

[`BATCH-SIZE.md`](BATCH-SIZE.md), [`batch_size.py`](batch_size.py) and
[`batch-size-results.json`](batch-size-results.json) record a fresh fixed-budget
ref20/3D sweep with transform export off. Larger batches help linear slice fits,
not the current SyN or volume controls; 3D SyN batch 256 ran out of VRAM. The new
default chooses conservatively from available VRAM, image size and measured
performance caps, with explicit overrides and no optimizer probes/OOM retries.
The [whole-ref20 SyN run](../syn/LANDEMARD.md) separately measures the remaining
FireANTs throughput gap after removing dense-result storage.

## Current dtype/precision result

[DTYPE.md](DTYPE.md) and [dtype-results.json](dtype-results.json) record the shipped
dtype-preserving output contract, eager NCC cleanup, rejected float32 fitting
candidates, broader real-window gates, fresh backend time/memory comparison and
package structure review. `validate_probe.py production --fit32` and `--ncc32`
are deliberately unqualified research variants, not public precision modes.

## CPU follow-up

[CPU-current.md](CPU-current.md) adds shared-NCC controls inside FireANTs and
fresh CPU profiles/timings. Those measurements predate the ConfUSIus integration;
the updated reproducer explicitly disables Rich progress.

[CPU.md](CPU.md) investigates the remaining gap at `857e792` with fresh profiles
and precision ablations. `cpu_ablation.py` runs one variant per interpreter and
retains checked generated source, timings and transforms.
[cpu-results.json](cpu-results.json) is its compact snapshot. Precision-changing
variants are diagnostics, not production optimizations; the engine is unchanged.
[NCC.md](NCC.md) and [ncc-results.json](ncc-results.json) record the subsequent
float64-preserving eager/compiled NCC tests, cold-cache distinctions, limitations
and pinned FireANTs dtype checks. The same harness now accepts `--device` and
`--batch-size` for matched CUDA follow-ups.

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

# Check the shipped sampler directly (same fixed reference, no substitution).
uv run python research/performance/check_elementwise.py --sampler production \
    --devices cpu cuda --output artifacts/production-batch-gate

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
