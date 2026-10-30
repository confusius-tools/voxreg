# CPU smoothing: backend and precision experiment

## Question and boundary

Can Torch filtering improve CPU SyN without loosening the existing 0.25-pixel
RMS truth or 0.001-pixel maximum CPU/CUDA point gates? This is research only;
production defaults, input/output dtype contracts and batching caps are unchanged.

Smoothing operates on vector fields, not the returned intensity image:

- Update-gradient smoothing (sigma 1.5 voxels) suppresses noisy local steps.
- Accumulated-field smoothing (sigma 0.5 voxels) regularizes the evolving warp.
- Positive sampled Jacobians and inverse residuals remain separate checks.

FireANTs was consulted, not copied. This experiment uses voxreg's separable Torch
convolution pattern with **unchanged replicate padding, round(3*sigma) radius,
axis order and fixed active boundaries**. Adopting FireANTs' different padding or
kernel truncation would confound a backend/precision comparison.

## Variants

| Mode | Update filter | Accumulated-field filter |
|---|---|---|
| baseline | Current SciPy float64 | Current SciPy float64 |
| torch64 | Torch float64 | Torch float64 |
| gradient32 | Torch float32 | Current SciPy float64 |
| all32 | Torch float32 | Torch float32 |

`all32` means all **filters**, not all registration computations. Every variant
retains float64 coordinates, displacement storage, CC statistics, inverse work,
and Jacobian evaluation. Float32 filters normalize each frame by a double maximum
before casting, then restore its magnitude in double. Kernels are cached within
the experiment-owned smoother, not in a process-global mutable cache.

Supporting float64 images is separate from selecting these intermediate dtypes.
The current API already preserves original image/output dtype while using
normalized float32 fitting samples; float64/integer preprocessing and final
resampling paths retain their existing behavior. These experiments do not test
float32 coordinates, float32 displacement storage, or float32 CC statistics.

## Numerical qualification

Each nonbaseline variant passed **85 unchanged registration/output controls** at
eight CPU threads, including native 2D/3D recovery, original float64 intensity
outputs and CPU/CUDA physical-point gates. **51 additional independent filter
checks** passed, covering zero, 1e-200, ordinary and 1e200 amplitudes, exact fixed
boundaries, input immutability, float64 returned fields, sigma validation and
unchanged differentiable fallbacks.

Completed native comparisons use 120 updates and two warm repeats. Every arm
passes the truth gate. Largest CPU/CUDA point error across the three cases:

| Variant | Maximum point difference (pixels) |
|---|---:|
| baseline | 0.000083312 |
| torch64 | 0.000083312 |
| gradient32 | 0.000215621 |
| all32 | 0.000139373 |

These are distinct from independent anatomical truth RMS and sampled inverse
residuals. Passing these fixtures does not establish safety for arbitrary images.

## Performance: exploratory battery-powered session

**The laptop was observed discharging, AC disconnected, in balanced mode during
this experiment.** Do not compare these absolute times with the previous AC
benchmarks. The observation does not prove a constant power state throughout
every arm. FireANTs reference timings also varied; a paired AC repeat is required
before claiming a production speedup or choosing a dispatch policy.

Eight CPU threads, warm median wall seconds:

| Variant | 2D pair | 3D pair | Eight 3D volumes | FireANTs CPU, eight volumes in that run |
|---|---:|---:|---:|---:|
| baseline | 0.409 | 3.594 | 11.994 | 5.932 |
| torch64 | 0.422 | 3.346 | 12.296 | 5.863 |
| gradient32 | 0.443 | 3.244 | 10.173 | 5.388 |
| all32 | 0.461 | 3.095 | 8.655 | 5.059 |

The promising direction is float32 filtering for larger CPU workloads, **not an
unconditional switch to Torch**: small 2D calls regressed and Torch float64 was
not better on the volume batch. The apparent 11.994-to-8.655-second improvement
cannot be called a controlled 28% speedup when the reference also changed.

On the eight-volume isolated filter, baseline/Torch-float32 update times were
8.382/1.492 ms; total-field times were 6.415/1.188 ms. Approximately 5–6x faster
filtering did not translate into a 5–6x registration speedup. Float32 conversion,
scaling and the remaining double-precision registration work still cost time.

## Reproduction and evidence

```bash
uv run python research/syn/cpu_smoothing.py --micro \
  --output artifacts/syn-speed/cpu-smoothing-micro-reproduction --threads 8 --repeats 15
uv run python research/syn/cpu_smoothing.py --mode gradient32 --tests --threads 8
uv run python research/syn/cpu_smoothing.py --mode all32 \
  --output artifacts/syn-speed/cpu-smoothing-native-reproduction --threads 8
uv run pytest -q tests/test_syn_cpu_smoothing.py
```

Raw runs, exits and the observed power-state note remain under
`artifacts/syn-speed/cpu-smoothing-*` and `CPU-SMOOTHING-POWER-STATE.txt`.
[Recorded results](cpu-smoothing-results.json) include complete source hashes,
backend arms, truth results, precision checks and filter microbenchmarks.

Keep current batch caps for now. These filters do not change CUDA arithmetic or
make free VRAM a throughput oracle. Revisit CPU batch-size crossover after a
smoothing policy is selected and verified on AC, multiple sizes and thread counts.
