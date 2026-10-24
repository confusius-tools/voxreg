# CPU follow-up with the same NCC function in both engines

This investigation ran on the numerical engine at `8ba7ec0`, **before** the
ConfUSIus dependency migration and default Rich reporter. It complements the
historical [CPU.md](CPU.md) and [NCC.md](NCC.md) reports. Data and hashes are in
[cpu-current-results.json](cpu-current-results.json); raw outputs remain ignored
under `artifacts/cpu-current/`. No production optimization was made here.

## What FireANTs actually does in this comparison

The pinned FireANTs CPU path uses ordinary PyTorch `affine_grid`, `grid_sample`
and Adam—not CUDA fused operations, a compiler or an ITK CPU kernel. Its native
geometry/optimizer state is float32. Both methods take 100 updates per moving
frame on this recording; fewer iterations do not explain the difference.

Importantly, the benchmark uses **our custom global squared NCC loss** inside
FireANTs, not its built-in local NCC. Native FireANTs timings therefore reflect
that float32 loss and its native rigid loop, not a special upstream NCC kernel.

The new `fireants-shared32` and `fireants-shared64` variants call voxreg's exact
centered/masked NCC implementation, including meaningfulness/finite guards, at
float32 or float64. All fixed batch images repeat the same reference, so the
shared loss broadcasts one fixed plane as voxreg does rather than converting a
whole repeated fixed batch. Masks still come from FireANTs' warped ones channel;
geometry, optimizer precision and stopping policies are **not** fully matched.

## Fresh unmonitored timings

Sequential fresh interpreters; one cold and three warm calls; batch32 on the same
120-frame neutral float32 input. Separate profiler runs. Clocks/affinity were not
fixed and desktop activity remained present: these are same-session comparisons,
not interchangeable with previous absolute timings.

| Engine / loss | 1 thread, median s | 8 threads, median s |
|---|---:|---:|
| voxreg production, NCC64 | 6.786 | 1.749 |
| voxreg, NCC32 only | 4.366 | 1.106 |
| FireANTs benchmark custom NCC32 | 2.988 | 1.162 |
| FireANTs, shared voxreg NCC32 | 2.731 | 1.005 |
| FireANTs, shared voxreg NCC64 | 4.964 | 1.394 |
| voxreg, separate-axis validity bounds | 6.024 | 1.780 |

**The bulk NCC precision cost is corroborated in both directions.** Switching
only voxreg's NCC to float32 saves about 36–37% here. Moving the same shared
FireANTs loss from float32 to float64 adds about 2.23 s single-threaded and
0.39 s eight-threaded. The small safety checks are not the dominant explanation:
FireANTs remains fast with the shared guarded loss at float32.

At eight threads, NCC32 nearly closes the gap to the shared-loss FireANTs control.
At one thread a substantial difference remains. Neither observation qualifies
NCC32 for production: the prior CPU/CUDA agreement failures are not waived; see
[DTYPE.md](DTYPE.md). Similar speed or NCC scores would not override those gates.

Separate-axis bounds show a single-thread trend but no eight-thread benefit in
this run. There is no evidence for adding a CPU-specific sampler branch now.

## Matched-size CPU profile

Five steady iterations of exactly 32 moving frames, eight threads, inclusive ms:

| Region | voxreg | FireANTs |
|---|---:|---:|
| Grid / sampling / support work* | 5.652 | 2.686 |
| NCC forward | 4.047 | 1.748 |
| Backward, including grid and metric | 7.929 | 7.131 |

*Scope boundaries differ: voxreg's bounds checks are inside sampling; FireANTs
samples an extra validity channel and thresholds it inside NCC. Small physical
matrix assembly is also scoped differently. Do not interpret these as isolated,
precision-matched kernels or sum them into benchmark wall-time predictions.

The remaining evidence points to **large-tensor precision, memory passes and grid
construction**, not xarray container overhead. Voxreg constructs dense normalized
coordinates with eager float64 broadcasts; FireANTs uses the native float32
`affine_grid` operation. Wider intermediates and extra passes are credible costs,
but this is not hardware-counter proof of DRAM bandwidth saturation.

## Next action, not a new default

Keep fitting policy unchanged. Once the ConfUSIus integration is settled, reprofile
that environment and try reducing float64 NCC/grid intermediates without losing
the centered formulation or independent optimizer guards. Do not replace the
optimizer or lower precision merely to match a headline CPU time. Compilation
remains a separate experiment with already documented cold/toolchain limits.

```bash
for threads in 1 8; do
  for variant in voxreg fireants voxreg-ncc32 fireants-shared32 fireants-shared64; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      uv run python research/performance/cpu_ablation.py "$variant" \
        --threads "$threads" --output artifacts/cpu-current
  done
done
```

The reproducer now explicitly disables progress. Reproducing the historical
measurement environment additionally requires the pre-integration source and
lockfile; running on current dependencies is a new measurement, not an exact replay.
