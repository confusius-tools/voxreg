# ITK, FireANTs and voxreg: time and memory

## Current shared pair/series protocol

The new [standardized 3D and motion-correction report](research/registration/README.md)
uses one runner, timing boundary, input/precision manifest, independent quality
scoring and result schema for both workloads. It includes separate fixed-budget
and backend-native convergence comparisons, fresh-process memory passes and
verified AC/performance endpoint snapshots. Native FireANTs early stopping fails
the truth screen on two 3D controls; short times alone are not quality equivalence.

> **Historical boundary below:** the tables below predate the temporary ConfUSIus
> VoxelData integration and retain their original settings/dependency boundary.
> The new report retimes the current stack separately. Do not combine old and new
> medians: fixed-image caching, ITK stopping/return-best settings, common-support
> scoring and power-state provenance differ. See [VALIDATION.md](VALIDATION.md).

## Historical: dtype-preserving output and simpler eager NCC

Source at this historical boundary retains image dtype and adds final-output linear/nearest sampling.
The fitting precision policy remains unchanged: float32 images/interpolation and
float64 geometry/state/NCC. Full float32 fitting and metric-only float32 failed
existing CPU/CUDA agreement gates; no public precision/compiler switch is added.
See [DTYPE.md](research/performance/DTYPE.md) for validation and package review,
and [dtype-results.json](research/performance/dtype-results.json) for the compact
snapshot, source hashes, ranges and before/after results.

The fresh comparison below uses the **same neutral float32 120-frame input** as
previous backend runs. Each case has fresh interpreters, three unmonitored warm
calls and a separate memory pass; all phases completed sequentially. Native
float64/int output correctness is tested separately, not claimed by these timings.
Normal desktop activity and unfixed laptop clocks remain sources of variation.

| Configuration | Median warm, s | Peak RSS, MiB | Peak PSS, MiB | Process VRAM, MiB |
|---|---:|---:|---:|---:|
| ITK, 1 process × 1 thread | 6.052 | 1425 | 1407 | — |
| ITK, 1 process × 8 threads | 2.145 | 1426 | 1408 | — |
| ITK, 8 processes × 1 thread | 1.599 | 11849 | 6821 | — |
| voxreg CPU, batch32, 1 thread | 5.691 | 726 | 695 | — |
| voxreg CPU, batch32, 8 threads | 2.128 | 730 | 702 | — |
| voxreg CUDA, batch32 | 0.780 | 1549 | 1464 | 370 |
| voxreg CUDA, batch120 | **0.219** | 1548 | 1463 | 464 |
| FireANTs CPU, batch32, 1 thread | 3.014 | 1072 | 987 | — |
| FireANTs CPU, batch32, 8 threads | **0.963** | 1071 | 986 | — |
| FireANTs CUDA, batch32 | 0.840 | 1935 | 1849 | 430 |
| FireANTs CUDA, batch120 | 0.355 | 1945 | 1859 | 510 |

Host figures are sampled whole-process-tree peaks including initialization;
process VRAM includes context. CUDA counters overlap these figures, not add to
these figures: voxreg full-batch allocated/reserved **175.2/204 MiB**, FireANTs
**173.5/194 MiB**. Native solver precision and optimizer policies still differ;
these are not biological-accuracy-matched speedup guarantees.

Against a fresh frozen `857e792` control, voxreg medians change:
CPU1-thread **5.935→5.691 s**, CPU8-thread **2.458→2.128 s**, CUDA32
**0.829→0.780 s**, CUDA120 **0.234→0.219 s**. Small CPU differences overlap across
repetitions; interpret them as a trend, not a universal speed ratio. Full-batch
CUDA allocation **189.2→175.2 MiB**, reservation **218→204 MiB**, process VRAM
**478→464 MiB**. All four neutral-input cases retain bitwise-identical transforms,
registered samples and iteration counts.

Before/after and backend controls were rescored together on common support:
median NCC² **0.961168 voxreg**, **0.961191 FireANTs**, **0.961099 ITK**.
Raw runs are in ignored `artifacts/benchmark-dtype-baseline/` and
`artifacts/benchmark-dtype-policy/`. This section supersedes timings below.

## Historical: scalar-affine sampler checkpoint (`857e792`)

The engine now uses the validated scalar-affine sampler. Float64 geometry,
optimizer state, NCC and world transforms are retained; interpolation remains
float32. There is no compilation, convergence change or new public option.

Fresh sequential runs compare the frozen `83b6dd6` engine with the new source on
**the same 120-frame recording**. Baseline code was extracted with `git archive`
and selected via `PYTHONPATH`, using the same locked environment and benchmark
script. Every case uses fresh interpreters, three unmonitored warm calls and a
separate memory pass. Normal desktop applications remained running; processor
clocks were not fixed. Raw phase outputs are in
`artifacts/benchmark-sampler-baseline/` and `artifacts/benchmark-sampler-production/`.
A compact snapshot, source hashes and quality/gate results are tracked in
[research/performance/sampler-results.json](research/performance/sampler-results.json).

### Before / after

| Configuration | Frozen baseline, s | Production, s | Median speed ratio |
|---|---:|---:|---:|
| CPU, batch 32, 1 thread | 8.400 | 5.869 | 1.4× |
| CPU, batch 32, 8 threads | 3.087 | 1.979 | 1.6× |
| CUDA, batch 32 | 1.311 | 0.818 | 1.6× |
| CUDA, batch 120 | 1.178 | **0.230** | **5.1×** |

### Fresh backend controls

RAM is sampled whole-process-tree PSS including cold initialization; VRAM is
sampled full-process device memory, not just live tensors.

| Configuration | Median time, s | Range of 3 calls, s | Host PSS, GiB | Process VRAM, MiB |
|---|---:|---:|---:|---:|
| ITK, 1 process × 1 thread | 5.895 | 5.887–5.935 | 1.381 | — |
| ITK, 1 process × 8 threads | 2.142 | 2.138–2.143 | 1.381 | — |
| ITK, 8 processes × 1 thread | 1.476 | 1.475–1.482 | 6.764 | — |
| voxreg CPU, batch 32, 1 thread | 5.869 | 5.713–5.972 | 0.686 | — |
| voxreg CPU, batch 32, 8 threads | 1.979 | 1.935–2.211 | 0.688 | — |
| voxreg CUDA, batch 32 | 0.818 | 0.817–0.823 | 1.435 | 370 |
| voxreg CUDA, batch 120 | **0.230** | 0.227–0.309 | 1.434 | 478 |
| FireANTs CPU, batch 32, 1 thread | 2.712 | 2.553–2.713 | 0.969 | — |
| FireANTs CPU, batch 32, 8 threads | **0.932** | 0.801–0.940 | 0.968 | — |
| FireANTs CUDA, batch 32 | **0.713** | 0.710–0.717 | 3.133* | 430 |
| FireANTs CUDA, batch 120 | 0.349 | 0.320–0.369 | 1.818 | 510 |

*The FireANTs CUDA batch-32 pass had a cold transient PSS peak of 3,208 MiB;
post-warmup PSS was 1,851 MiB. The peak is retained as measured, not interpreted as
steady-state memory. Sampled peaks and laptop timings vary across runs.

Full-batch CUDA is about 1.5× faster than native FireANTs in this run. At batch 32,
FireANTs remains slightly faster. CPU now approaches single-process threaded ITK,
but native float32 FireANTs is still faster; its statistics/optimizer policies are
not precision-matched to voxreg. Neither backend comparison proves equivalent
biological accuracy or performance across other recordings.

### Correctness and memory gates

All cases were rescored together with the independent SciPy sampler on one common
support: median NCC² is **0.961168 before and after**, **0.961191 FireANTs** and
**0.961099 ITK**. Reference identity/intensities/time labels, status policy and the
100-update moving-frame counts remain unchanged.

- Worst same-device mapping RMS change from baseline: **6.69e-6 minimum-spacing
  pixels**. No optimizer/stopping policy was changed to obtain the speedup.
- Production real-data batch sizes 1/32/120 agree within **3e-16 pixels RMS on CPU**
  and **6.95e-9 on CUDA**, below the unchanged 0.001-pixel gate. The separate-fixed
  reference checker also passes with batch sizes 1/32/119 (119 moving frames).
- **58 tests pass**, including sheared/oblique/strided SciPy sampling, gradients,
  support-boundary tolerance, failure/convergence isolation and CUDA batch checks.
- Full-batch CUDA allocator peak: **212.5 → 189.2 MiB allocated**,
  **254 → 218 MiB reserved**; process VRAM **514 → 478 MiB**.
- A steady-state production profile has only small 2×2 geometry GEMMs; the
  image-sized GEMMs are gone. Compilation and broader-data validation remain open.

Reproduce the production cases with the existing benchmark CLI:

```bash
uv run python examples/benchmark_registration.py \
    --output artifacts/benchmark-sampler-production \
    --cases itk-1 itk-1-t8 itk-8 cpu-32 cpu-32-t8 cuda-32 cuda-120 \
            fireants-cpu-32 fireants-cpu-32-t8 fireants-cuda-32 fireants-cuda-120
uv run python research/performance/check_elementwise.py --sampler production \
    --devices cpu cuda --output artifacts/production-batch-gate
```

The original baseline remains available at commit `83b6dd6`. To reproduce that
control without changing the working tree:

```bash
mkdir -p artifacts/sampler-baseline-source
git archive 83b6dd6 src | tar -x -C artifacts/sampler-baseline-source
PYTHONPATH="$PWD/artifacts/sampler-baseline-source/src" uv run python \
    examples/benchmark_registration.py --output artifacts/benchmark-sampler-baseline \
    --cases cpu-32 cpu-32-t8 cuda-32 cuda-120
```

## Historical baseline comparison (`83b6dd6`)

The following measurements describe the unoptimized baseline, not current engine
performance. They are retained to explain the investigation and its limitations.

Measured on the same **120-frame** rat75/slice32 window, shape
`(120, 1, 112, 128)`. All eleven configurations below were remeasured sequentially
in one session after adding FireANTs. This compares **direct 2D rigid registration**,
not the ConfUSIus pipeline. ITK and FireANTs are **development-only dependencies**;
voxreg's runtime dependencies remain NumPy, xarray and PyTorch.

Hardware: Intel Core Ultra 9 275HX, 32 GB RAM, RTX 5080 Laptop GPU. Python 3.14.8,
PyTorch 2.14.1+cu130, ITK 5.4.7, FireANTs 1.5.1, SimpleITK 2.5.6, Linux.
Full hardware/version information and raw measurements are in
`artifacts/benchmark-fireants/summary.json`. A compact historical snapshot is
tracked in [research/performance/results.json](research/performance/results.json);
raw arrays and generated outputs remain ignored.

## Results

Times are **median warm end-to-end seconds**, with the range of three runs.
RAM is sampled **whole-process-tree PSS** in GiB, including libraries, immutable
inputs, initialization, allocator caches and any ITK workers. PSS apportions
shared pages rather than counting every shared mapping repeatedly.
GPU memory is sampled per-process VRAM, including CUDA context/library overhead.

| Configuration | Time, s | Range, s | Peak host RAM, GiB PSS | Peak process VRAM, MiB |
|---|---:|---:|---:|---:|
| ITK, 1 process × 1 thread | 5.927 | 5.916–5.933 | 1.386 | — |
| ITK, 1 process × 8 threads | 2.561 | 2.442–2.569 | 1.387 | — |
| ITK, 8 processes × 1 thread | 1.397 | 1.372–1.401 | 6.769 | — |
| voxreg CPU, batch 32, 8 threads | 4.302 | 4.288–4.310 | **0.722** | — |
| voxreg CUDA, batch 32 | 1.632 | 1.615–1.931 | 1.452 | 390 |
| voxreg CUDA, batch 120 | 1.386 | 1.370–1.405 | 1.453 | 514 |
| FireANTs CPU, batch 1, 1 thread | 7.989 | 7.519–8.102 | 0.918 | — |
| FireANTs CPU, batch 32, 1 thread | 3.134 | 2.851–3.211 | 0.967 | — |
| FireANTs CPU, batch 32, 8 threads | **1.279** | 1.138–1.355 | 0.964 | — |
| FireANTs CUDA, batch 32 | 1.718 | 1.712–1.722 | 1.812 | 430 |
| FireANTs CUDA, batch 120 | **0.514** | 0.503–0.521 | 1.821 | 510 |

### What this says

- **At the baseline checkpoint, voxreg CPU was slower than ITK.** Eight-thread
  voxreg takes 4.30 s versus 2.56 s for eight-thread ITK here. This eager prototype
  is a correctness baseline, not a demonstrated CPU performance win.
- FireANTs shows that PyTorch is not inherently the problem: its eight-thread CPU
  case takes 1.28 s at similar NCC, versus voxreg's 4.30 s—about 3.4× faster.
- Full-batch FireANTs CUDA takes 0.51 s, about 2.7× faster than either voxreg CUDA
  or eight-process ITK here. At batch 32, however, the GPU engines are comparable;
  full batching matters. No optional FireANTs fused extension was installed.
- Voxreg has the lowest host RAM of the practical CPU controls. FireANTs CUDA uses
  about 0.37 GiB more host RAM than voxreg CUDA, with almost the same process VRAM.
- ITK's 6.77 GiB PSS is a process-duplication tradeoff, **not an inherent ITK
  requirement**: the single-process eight-thread control stays near 1.39 GiB.

Voxreg performs 100 Adam updates with float64 geometry/state/NCC and per-frame
best-valid/stopping bookkeeping. ITK usually stops much earlier; native FireANTs
uses float32 and a different optimizer parameterization. The benchmark does not
separate precision cost from eager dispatch/bookkeeping or optimizer policy.
Those contributions require profiling—not an assumption that xarray or PyTorch
itself explains the slowdown. No engine optimization was made for this comparison.

The earlier ITK/voxreg-only session remains in `artifacts/benchmark/`. It measured
voxreg CPU at 3.35 s and full-batch CUDA at 1.26 s, versus 4.30 s and 1.39 s here.
These are workstation results, not stable cross-session guarantees. That earlier
session also includes the deliberately slow unbatched CUDA control (43.56 s),
which was not remeasured here. FireANTs serial CUDA was not benchmarked.

## Quality and stopping rules

Every method sees the same original float32 samples and resamples original power
back to the fixed grid. One independent SciPy bilinear sampler evaluates the
returned matrices on shared dB inputs. Scores use the **same intersection of
valid sample centers across all eleven configurations**, excluding the unchanged
reference frame. Median common support is 98.33%.

| Engine | Median squared NCC | Minimum squared NCC |
|---|---:|---:|
| ITK | 0.961099 | 0.917559 |
| voxreg | 0.961168 | 0.917576 |
| FireANTs | 0.961191 | 0.917576 |

- ITK uses Euler2D, double-precision fitting/interpolation, global correlation,
  physical-shift optimizer scales and gradient descent at rate **0.1**, up to 100
  iterations, convergence minimum 1e-6/window 10. Rate 1 overshot the synthetic
  recovery cases, so rate 0.1 was validated before measuring the real recording.
  ITK reports 10–39 iterations, median 31.
- Voxreg uses its existing rate 0.1 Adam policy, float64 physical geometry/state
  and NCC statistics, with float32 interpolation. All non-reference frames reach
  its 100-step limit; they are not certified converged.
- FireANTs uses native float32 2D rigid Adam at rate **0.01**, with scaling and
  multiresolution disabled. Early stopping is disabled, and an explicit callback
  count checks **100 updates per moving frame**. A custom independent global
  squared NCC replaces the default local NCC. A second constant image channel
  supplies warped validity, excluding out-of-grid samples without patching the
  library. Only original power is resampled for the final output, with each
  frame's minimum fill outside the grid. No best-iterate snapshot is substituted
  for FireANTs's native final parameters.

Thus these are native-engine practical comparisons with near-identical image NCC,
**not precision-matched solvers, identical stopping rules or proof of equal
biological accuracy**. Direct-ITK settings also differ from the earlier frozen
ConfUSIus reference's 3D singleton/rate-1 configuration.

Synthetic CPU and CUDA tests independently verify physical pull matrices against
known motion (unchanged 0.25 minimum-spacing-pixel RMS gate), reference identity,
100 FireANTs updates and original-power interpolation against SciPy.
On this real recording, FireANTs matrices are identical across CPU batching/thread
controls, and CUDA batch 32 and full batch also agree exactly. CPU–CUDA differences
reach 0.00117 minimum-spacing pixels RMS in the worst frame. Relative to full-batch
FireANTs, voxreg's median/worst mapping disagreements are 0.00247/0.150 pixels;
ITK's are 0.0216/0.106 pixels. These are disagreements, not ground-truth errors.

## Memory accounting

Host RAM numbers are full process footprints, not just image-array working sets.
Loaded-input root RSS baselines and post-warmup tree RSS/PSS baselines are stored
in each raw JSON. Summed RSS is also available: for eight-process ITK it is
**11.77 GiB RSS versus 6.77 GiB PSS**, because RSS repeatedly counts shared pages.
The comparison launcher/monitor itself is outside the measured process tree.

For CUDA, three counters overlap and must **not** be added:

| Engine / batch | PyTorch peak allocated, MiB | Peak reserved, MiB | Sampled process VRAM, MiB |
|---|---:|---:|---:|
| voxreg / 32 | 110.9 | 130 | 390 |
| voxreg / 120 | 212.5 | 254 | 514 |
| FireANTs / 32 | 100.4 | 114 | 430 |
| FireANTs / 120 | 173.5 | 194 | 510 |

Allocated/reserved peaks are exact PyTorch counters since warmup. Reserved memory
includes cached allocator blocks. Process VRAM additionally includes allocations
outside that allocator, including CUDA context/library overhead. Host RSS/PSS and
process VRAM are sampled peaks and may miss very brief transients.

## Installation and reproduction

The PyPI FireANTs release pins an old SimpleITK incompatible with Python 3.14.
Current upstream 1.5.1 supports modern SimpleITK, but its Git checkout references
an inaccessible cookbook submodule. The dev dependency therefore uses the
**immutable upstream source archive** at commit
`f8a9cc259ab9113fc43bd99797000fa8fc34eb59`, with its content hash in `uv.lock`.
No dependency override or FireANTs source modification is used.

Prepare inputs without ConfUSIus by running the NIfTI example from README.md;
it writes `artifacts/notebook/input.npz`. Then:

```bash
uv sync
uv run python examples/benchmark_registration.py \
    --output artifacts/benchmark-fireants \
    --cases itk-1 itk-1-t8 itk-8 cpu-32-t8 cuda-32 cuda-120 \
            fireants-cpu-1 fireants-cpu-32 fireants-cpu-32-t8 \
            fireants-cuda-32 fireants-cuda-120
```

Protocol:

1. No competing benchmark/build was observed in manual checks before launch;
   ordinary desktop CPU/GPU applications remained running. The script does not
   enforce idleness or fix processor clocks.
2. Each configuration uses a **fresh interpreter**. ITK process workers use spawn,
   caching immutable inputs/fixed fitting images per worker.
3. A cold full-recording call warms templates/context/allocators, followed by
   three **unmonitored** warm calls. Preprocessing, fitting, original-power
   resampling and host return are included; file loading/compression are excluded.
   ITK pool scheduling and result IPC are timed. FireANTs constructs/transfers
   fitting images per call; it does not receive preuploaded resident images.
   Setup/cold timings are saved separately, not mixed into the table.
4. **Memory is a separate fresh-process pass**, covering initialization plus cold
   and warm registration. RSS/PSS is sampled around every 20 ms; GPU process VRAM
   around every 250 ms. Monitoring can substantially perturb execution, so its
   wall times are not used as benchmark timings. CUDA allocator peaks are reset
   after warmup.
5. All eleven configurations and both phases completed without interruption.
   Quality uses independent interpolation and one common support across methods.

Raw JSON/NPZ outputs, quality, iteration summaries and mapping comparisons are in
`artifacts/benchmark-fireants/`. `comparison.png` visualizes the results.
The prior run is retained separately, not mixed into this table. Profiling and
real-data convergence calibration remain next work before extending the model.
