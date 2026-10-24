# Precision controls and simpler NCC experiments

Historical follow-up to [CPU.md](CPU.md), using production source `857e792`
unchanged during those experiments. [DTYPE.md](DTYPE.md) records the subsequent
shipped eager NCC cleanup and dtype contract; compilation remains experimental.
Results and failures are retained in [ncc-results.json](ncc-results.json).

## Dtype is a numerical policy, not an input-file property

The pinned FireANTs 1.5.1 exposes `Image(..., dtype=...)` and registration
`dtype=...`, defaulting to float32. These are separate controls. A minimal CPU
2D rigid test with one update gave:

| Image dtype | Registration dtype | Result |
|---|---|---|
| float32 | float32 | Works |
| float64 | float32 | Works; geometry and parameters remain float32 |
| float32 | float64 | Float/Double mismatch |
| float64 | float64 | Float/Double mismatch |

`Image` constructs its physical-coordinate matrices with explicit `.float()`
calls, and rigid initialization contains other default-float32 tensors. Thus the
presence of a dtype argument does not establish a clean full-double rigid path.
This is a narrow observation about the pinned version and default 2D path, not a
claim about every FireANTs registration class/version. No upstream source was
patched to make the experiment work.

For voxreg, do not choose optimizer/NCC/geometry precision from the input NumPy
dtype. NIfTI loaders often return float64 independent of measurement precision;
physical-coordinate range and loss sensitivity are also independent of image
storage. Keep the existing single mixed policy for now: float32 image sampling,
float64 geometry/state/NCC. No runtime dtype option was added. If a second policy
is eventually validated, one explicit compute-precision choice is preferable to
independent knobs or automatic dtype inference.

## Experiments retaining float64

All variants retain guards, independent clocks, best-valid transforms and stopping
rules. The simple eager candidate copies FireANTs' loss organization: apply the
binary validity mask to the centered images once, then share those arrays between
variance and covariance. This removes a redundant image-wide multiplication
without dropping any validation or using unstable raw-moment subtraction.

The compiled candidate applies `torch.compile(fullgraph=True, dynamic=True)` only
to registration's NCC call, not to the optimizer or sampler. The per-axis bounds
candidate is the earlier behavior-preserving sampler substitution.

Fresh interpreters, one first call and three warm calls per configuration:

| Candidate | CPU 1 thread, s | CPU 8 threads, s | CUDA batch32, s | CUDA batch120, s |
|---|---:|---:|---:|---:|
| Production eager | 6.002 | 1.982 | 0.848 | 0.229 |
| Eager masked-centered NCC | 5.735 | 1.858 | 0.812 | 0.213 |
| Eager masked NCC + per-axis bounds | 4.576 | 1.761 | 0.856 | 0.214 |
| Compiled NCC | 3.828 | 1.407 | 0.660 | 0.196 |
| Compiled NCC + per-axis bounds | 3.320 | 1.388 | 0.735 | 0.193 |
| Native FireANTs | 2.649 | 0.935 | 0.826 | 0.352 |

These are medians, not fixed-clock guarantees or precision-matched solver
comparisons. The snapshot retains every warm sample. Small eager gains need
paired timing: some CPU ranges overlap. Per-axis bounds help CPU, but adding them
does not consistently help CUDA batch32. Avoid declaring one universal winner.

### Correctness

- Eager masked NCC, alone and with per-axis bounds, gives **bitwise-identical real
  transforms** to production at the measured CPU/thread/CUDA batch configurations.
- Compiled results change by at most 5.89e-10 pixels RMS on CPU and 6.95e-9 pixels
  on CUDA relative to eager. Common-support NCC is unchanged at reported precision.
- The compiled registration call passes all **59 existing tests**. The eager
  masked-NCC replacement also passes **59 tests**, including direct metric gradient
  tests. Tests and tolerances were not weakened.
- Compiled real-recording batches 1/32/119 with a separate fixed image agree within
  3e-16 pixels RMS on CPU and 5.23e-16 on CUDA, below the 0.001-pixel gate.
- Sixteen independent one-shot loss/flag/gradient comparisons agree with eager for
  CPU/CUDA, float32/float64 inputs, normal/empty/constant/near-identical images.

### Compilation is not free simplification

The measured first CPU8 compiled registration call took **9.93 s**, versus 1.41 s
warm. That first call excludes construction/import setup of the `torch.compile`
wrapper, so it is not complete process startup. The combined candidate then took
2.55 s on its first call with a compiler **disk-cache hit**. First-call numbers
must not be mixed as though all caches were cold. Dedicated CPU1/CPU8 cache
folders started empty; CUDA120 reused CUDA32's cache.

There are compatibility limits in this environment:

1. Default parallel CUDA compilation failed with an unclean subprocess-pool exit.
   `TORCHINDUCTOR_COMPILE_THREADS=1` worked. Root cause is unknown; no OOM evidence
   was observed. No package/library source workaround was installed.
2. Replacing the standalone NCC helper everywhere across the heterogeneous test
   suite hit Dynamo's default eight-specialization limit (57 passed, two failed).
   The intended registration-only substitution passes all 59 tests.
3. Direct compiled-helper gradient testing hit AOT donated-buffer restrictions:
   its backward requires `retain_graph=False`, whereas gradcheck retains the
   graph. That isolated helper suite gave 11 passes and one failure. The ordinary
   one-shot backward used by registration agrees with eager in explicit tests.
   We did not disable donation, increase cache limits or relax the gradient test
   to conceal this limitation.

Compilation therefore remains a research optimization, not a default package
behavior or new public mode. CPU compiler availability, cold latency, peak memory
and maintenance costs need consideration alongside warm throughput.

## Recommendation

Keep one validated precision policy and the eager production path for now.
The masked-centered NCC is the smallest promising cleanup: ordinary PyTorch,
no new option, compiler requirement or changed safety contract. Confirm its small
gains with paired time/memory passes before landing. Treat per-axis bounds as a
separate tradeoff because of CUDA batch32, and compiled NCC as a separate opt-in
research direction rather than forcing compiler complexity into the core.

No production change or commit was made during these experiments. Peak memory
and broader recordings were not tested for these new candidates.

## Reproduction

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Serial compilation avoids the observed CUDA compiler-pool failure.
export TORCHINDUCTOR_COMPILE_THREADS=1
# Use a fresh directory for a genuinely cold compiler-cache experiment.
export TORCHINDUCTOR_CACHE_DIR="$PWD/artifacts/ncc-compile/reproduction-cache"

uv run python research/performance/cpu_ablation.py voxreg-compiled-ncc --threads 8 \
    --output artifacts/ncc-compile/reproduction
uv run python research/performance/cpu_ablation.py voxreg-masked-ncc --threads 8 \
    --output artifacts/ncc-compile/reproduction
uv run python research/performance/cpu_ablation.py voxreg-compiled-ncc \
    --device cuda --batch-size 120 --output artifacts/ncc-compile/reproduction
uv run python research/performance/validate_probe.py production --compiled-ncc
uv run python research/performance/validate_probe.py production --masked-ncc
```

Use `voxreg`, `fireants`, or the `*-split-bounds` variants for matching controls.
As before, configurations run sequentially, never while another benchmark/build
is active. The real-data batch check uses `check_elementwise.py --sampler
production` with registration's NCC alias temporarily compiled in that process.
Large outputs, generated substitution source and failed compiler logs remain
ignored under `artifacts/ncc-compile/`.
