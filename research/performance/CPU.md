# Remaining CPU gap after the scalar-affine sampler

Investigation of production commit `857e792`. **No production code was changed.**
The subsequent [NCC.md](NCC.md) tests compilation and simpler eager NCC while
retaining float64, and checks the pinned FireANTs dtype controls.
The remaining gap is mainly large-image NCC arithmetic/dataflow, followed by grid
and validity construction—not xarray or per-frame Adam clocks.

## Controlled precision ablations

Fresh interpreter per variant/thread count; one cold call and three unmonitored
warm calls; 120 input frames, batch 32, 100 updates per moving frame. Same input,
locked environment and original-power output contract as BENCHMARK.md. Profiler
runs were separate. Normal desktop applications remained running; clocks were
not fixed. Absolute timings drifted from the earlier production session: compare
these fresh controls to each other, not to earlier table entries.

| Backend / isolated change | 1 thread, median s | 8 threads, median s |
|---|---:|---:|
| Production voxreg | 6.480 | 2.497 |
| voxreg, **NCC arithmetic only float32** | 3.891 | 1.395 |
| Native FireANTs | 2.762 | 1.059 |
| FireANTs, **NCC arithmetic only float64** | 4.719 | 1.818 |

Changing only voxreg's NCC dtype removes about 40–44% of its warm runtime and
roughly 70–77% of the observed gap to native FireANTs in these runs. The reverse
FireANTs experiment corroborates the precision cost. This is **not a complete
precision/optimizer match**: FireANTs still has float32 geometry/parameters/state,
and its NCC has a different eager expression layout and fewer safety checks.

The voxreg NCC promotes float32 sampled pixels and the fixed image to float64,
then forms float64 masks, means, centered images, variances and covariance.
Backward propagation also traverses these wider intermediates and casts the
sample gradient back to float32. FireANTs' benchmark custom loss performs that
work in float32. Twice the element width and numerous full-image tensor passes
are credible mechanisms; this investigation did not collect hardware counters
or prove DRAM bandwidth saturation.

The float32-NCC experiment changes transforms by up to **0.00532 minimum-spacing
pixels RMS** from production. CPU batch sizes 1/32/120 still agree within 3e-16
pixels on this recording. That is encouraging, not a new accepted precision
policy: GPU behavior, synthetic recovery and broader recordings were not checked
for this variant. Similar NCC scores alone do not prove equivalent accuracy.

## Where the CPU profile spends time

Five steady iterations, exactly 32 moving frames for both backends, eight threads:

| Profile region | voxreg, ms | FireANTs, ms |
|---|---:|---:|
| Grid construction, validity and sampling* | 8.06 | 2.35 |
| NCC forward | 6.87 | 1.62 |
| Backward (sampling + NCC + geometry) | 13.62 | 7.82 |
| Other work, including optimizer/bookkeeping** | 3.01 | 3.34 |
| Total profiled steps | 31.56 | 15.13 |

*FireANTs' scope starts at its interpolator; small physical matrix assembly is
outside it. Voxreg's sampling scope includes its small matrix assembly. FireANTs
samples the validity channel here but thresholds it inside NCC; voxreg's bounds
test is inside sampling. These are useful coarse regions, not identical
instruction boundaries.

**Voxreg's unscoped remainder versus FireANTs' explicit Adam region plus its
unscoped remainder. This is not an isolated Adam microbenchmark. Nevertheless,
even eliminating voxreg's entire remainder cannot close this gap.

Voxreg has float64 normalized grid arithmetic implemented as several eager
broadcast operations. FireANTs uses a float32 `affine_grid` path and samples a
second ones channel for validity. Voxreg instead compares the two coordinate
components with physical-support bounds and reduces the interleaved boolean grid.
In these five iterations voxreg executes 185 `mul` and 95 `sum` calls, versus
95 and 40 for FireANTs. Operator counts alone are not costs, but the profiler
shows substantial time in multiplication, copying, reductions and comparisons.
The previous image-sized float64 GEMM bottleneck is gone.

Preprocessing, transfers and final resampling are small: production's median
optimization stage is 2.458 s of a 2.497 s call. Xarray is not responsible for
most of the CPU gap. Both backends take 100 updates per moving frame here, so
this is not an iteration-count advantage. The API also processes one frozen
reference row that the FireANTs adapter skips (120 versus 119 rows); the matched
32-moving-frame profiles remove that small difference.

## Other ablations

- **Separate per-axis validity comparisons**, retaining float64 and all policies:
  5.758 s / 2.341 s at 1/8 threads. Real transforms are bitwise equal to production;
  all 59 tests pass with this sampler temporarily substituted, as well as against
  unchanged production. This is a smaller promising candidate, not a confirmed
  general speedup; the
  eight-thread samples range from 1.920 to 2.411 s. More paired CPU timing and
  GPU performance checks are needed before considering production.
- **Float32 C++ affine grid**, keeping float64 parameters/state/NCC:
  5.934 s / 2.226 s. It shifts transforms by up to **0.13022 pixels RMS**.
  It does not justify replacing the scalar-affine production sampler.
- Combining affine32 with NCC32 gives 3.928 s / 1.594 s, not a reliable improvement
  over NCC32 alone. Changing grid generation is not automatically a CPU win.

Mapping disagreements above are not ground-truth errors. Common-support scores
in this investigation use its fourteen variants, not the earlier ITK-inclusive
support mask, so scores across those reports are not interchangeable.

## Recommended next experiment

1. Keep production precision unchanged and try a **fused/compiled NCC forward and
   backward** to reduce full-image intermediates, casts and memory passes. Measure
   compilation/cold cost separately and rerun all numerical/batch gates.
2. Investigate the smaller precision-preserving per-axis validity change with
   paired timing. Do not rewrite independent Adam or remove safety checks first.
3. Treat metric-only float32 as a separate numerical-policy experiment, not the
   default optimization. No runtime precision option was added here.

No compilation/fused implementation or production fix was attempted in this
initial investigation; the subsequent [NCC follow-up](NCC.md) records those
experiments and their limitations.

## Reproduction and evidence

The checked source substitutions are saved beside raw outputs. All measured
transforms, phase logs and traces remain ignored under
`artifacts/cpu-investigation/`. The compact tracked snapshot is
[cpu-results.json](cpu-results.json).

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for threads in 1 8; do
  for variant in voxreg fireants voxreg-ncc32 fireants-ncc64 \
                 voxreg-split-bounds voxreg-affine32 voxreg-affine32-ncc32; do
    uv run python research/performance/cpu_ablation.py "$variant" --threads "$threads"
  done
  for backend in voxreg fireants; do
    uv run python research/performance/profile_baseline.py "$backend" cpu \
        --threads "$threads" --batch 32 --output artifacts/cpu-investigation/profile
  done
done
```
