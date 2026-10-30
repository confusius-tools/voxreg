# CUDA batching: ref20 and native 3D

Measured on the RTX 5080 Laptop GPU (15.54 GiB total), Torch 2.14.1+cu130,
with desktop applications still using VRAM. One CPU thread; native resolution;
100 fixed-budget updates, convergence window 101; default transform export off.
Inputs were loaded before timing. Preparation, uploads, fitting and corrected
original-intensity output were timed with CUDA synchronization. Each arm was
warmed with two updates, followed by two timed full-input calls. Medians below;
allocated peaks exclude unused Torch allocator cache. No concurrent registration
benchmarks ran. These are bounded measurements, not a universal hardware tuner.

Reproduce from the repository root:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python research/performance/batch_size.py \
  --output artifacts/batch-size/new-run
```

Local raw evidence, failures and source snapshots:
`artifacts/batch-size/sweep-01/`, `sweep-01.log`, `sweep-01-source/` and
`sweep-01-script.py` in that parent directory. Full rows are also recorded in
[`batch-size-results.json`](batch-size-results.json).

## Real ALD001/ref20: first 1,024 frames, 1×145×80 float32

| Model | Batch 32: seconds / peak MiB | Batch 128 | Batch 256 |
|---|---:|---:|---:|
| Rigid | 6.591 / 92 | **1.688 / 181** | 1.721 / 293 |
| Affine | 7.126 / 93 | 1.818 / 181 | **1.710 / 293** |
| SyN | **58.511 / 377** | 72.504 / 1,303 | 79.536 / 2,531 |

Batch 256 is approximately **3.83× faster for rigid and 4.17× for affine** than
32. It is **1.36× slower for SyN**. There was ample VRAM for every slice arm:
maximum capacity is not the same thing as maximum throughput.

Linear corrected outputs were bitwise identical across these batch choices.
SyN batches 128/256 differed from batch 32 by image RMS 5.026 original-power units
(first-reference maximum 1,210,628.5). This is an output-intensity observation,
**not a physical point-precision gate or anatomical accuracy claim**. Native
optimizer trajectories can differ numerically; no accuracy/precision gate was
relaxed to select a default. The package truth and CPU/CUDA gates still pass in
the complete regression suite.

With transforms already disabled, more than 99% of the measured batch-32 SyN
call was optimization, including inverse/regularity work. Optional export fixes
the full-result CPU-memory problem; it does not by itself fix core throughput.

## Shared native 3D control: 256 frames of 32×40×48

The package-wide nonlinear fixture is repeated on the same independent fixed
grid. It is a throughput/memory control, not a real volumetric recording.

| Model | Batch 32: seconds / peak MiB | Batch 128 | Batch 256 |
|---|---:|---:|---:|
| Rigid | **7.262 / 283** | 8.172 / 928 | 8.381 / 1,641 |
| Affine | **7.212 / 283** | 8.147 / 928 | 8.393 / 1,641 |
| SyN | **152.448 / 2,594** | 170.027 / 10,139 | **CUDA OOM during warm-up** |

The OOM is retained in the raw report. No retry silently changed the timed
configuration. Successful 3D corrected outputs were identical in this control.

## Automatic policy

`register(..., batch_size=None)` is now the default. CPU retains a maximum of
32 frames without touching CUDA. On CUDA:

- Count voxels in the larger moving/fixed image, including independent grids.
- Available capacity is CUDA free memory **plus unused reusable Torch cache**.
- Reserve half that capacity; estimate 512 working bytes/voxel for linear models,
  1,536 for slice SyN, 3,072 for volume SyN. These conservative estimates exceed
  observed allocated costs in the controls above; they are not exact formulas
  for every allocator, geometry, dtype or convolution workspace.
- Cap at **256 for linear slice pairs**, **32 for SyN or volumes**, then round the
  CUDA cap down to a power of two and limit it to the recording's frame count.
- Record the effective size in `result.diagnostics.attrs["batch_size"]`.

An explicit positive `batch_size` bypasses the policy. There are no optimizer
probes, trial registrations, silent OOM retries or repeated callbacks. Other
processes can allocate memory after selection, and even one frame might not fit;
this estimate does not guarantee OOM immunity or optimal speed on every GPU.
Returned images/masks and opt-in fields still need full-recording CPU storage.

The cap intentionally does **not** raise the SyN default to 256: this experiment
contradicts the assumption that doing so speeds up the current implementation.
The separate [whole-ref20 SyN comparison](../syn/LANDEMARD.md) measures the
remaining practical gap to FireANTs after disabling dense output export.
