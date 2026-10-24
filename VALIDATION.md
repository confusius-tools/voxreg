# Initial rigid validation

Status: **numerical and real-data quality checks pass; isolated ITK/FireANTs
performance/memory comparisons are now available** in [BENCHMARK.md](BENCHMARK.md).
Throughput measurements cover the documented recording only; this is not a
general-purpose registration library.

## Environment and reproducibility

- Python 3.14.8, NumPy 2.5.4, xarray 2026.9.0.
- PyTorch 2.14.1+cu130, RTX 5080 Laptop GPU, Linux.
- Reference: ConfUSIus 0.9.0.dev0 and ITK 5.4.7, in a separate interpreter.
- Voxreg: float32 interpolation/output; float64 physical geometry, optimizer state,
  NCC statistics and exported matrices.
- Input: Cybis Pereira 2026 rawdata, rat75 / 20220523 / slice32, frames 220:340,
  shape `(120, 1, 112, 128)`.
- Source NIfTI SHA-256:
  `f2d0a5e4daa10afc979b47d926af4e1ba3b2368c1384e32944ef9d3d8687e9b3`.
- Original irregular time labels retained, approximately 88.4–136.0 s.

The ignored `artifacts/notebook/` directory contains frozen reference inputs,
source-code fingerprints, resolved ITK settings, matrices, original-power output,
framewise-displacement references, and machine-readable comparison results. The
reference retains source float64 intensities; voxreg converts them to float32.
The example verifies equality after that declared conversion and checks the
source hash. No optimizer parameters were tuned against this recording.

## Automated tests

```bash
uv run --extra examples pytest -q
# 58 passed, including CUDA, sheared/strided SciPy sampling, gradients and research CLI checks.

uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

Coverage of the behavior, not just shapes:

- Native-coordinate offsets, reverse/strided crops, oblique world affines and
  preservation of lazy geometry through slicing and time reduction.
- Rejection of missing/stale indexes, invalid affines/units/coordinates, invalid
  shapes/options/timing, nonfinite data, constants and incompatible grids.
- Bilinear sampling against independent SciPy interpolation and homogeneous
  mapping/inverse/rotation-center checks.
- CPU float64 gradient checks and independent masked NCC calculations.
- Analytic asymmetric phantom recovery for rotations up to two degrees and
  translations up to two minimum-spacing pixels, with no fitting resampling
  used to generate the ground truth.
- Opposing motions, chunk composition, independent failure handling and frozen
  converged optimizer state.
- CPU/CUDA agreement and NIfTI geometry/intensity/time round trips.

The synthetic recovery gate is RMS physical mapping error below **0.25 minimum
in-plane pixel spacings** over the declared interior region. The worst measured
error across six orientation/stride/intensity combinations is **0.00424 pixels**
(rounded upward). See `synthetic_checks.json` for individual cases.

## Real recording: quality

Both sets of physical matrices are evaluated by an independent SciPy bilinear
sampler applied to the **same dB input arrays**. Each frame uses the intersection
of both methods' valid sample-center support. These are common scoring values,
not a comparison of unlike internal optimizer reports. Summaries include the
reference frame.

| Measure | Unregistered | voxreg | ITK reference |
|---|---:|---:|---:|
| Median squared NCC | 0.94085 | 0.96119 | 0.96087 |
| Minimum squared NCC | 0.87026 | 0.91758 | 0.91746 |

- Median paired voxreg-minus-ITK squared NCC: **+0.000332**.
- Common valid support: median **98.33%**, minimum **97.35%**.
- RMS mapping disagreement with ITK over all fixed-grid points:
  median **0.00726 mm**, 95th percentile **0.01284 mm**, maximum **0.01616 mm**.
  With minimum spacing 0.09856 mm, these are approximately **0.074, 0.130 and
  0.164 pixels**, respectively.

These results show comparable alignment on this recording, not recovery of a
known biological ground truth or universal superiority over ITK. A higher NCC
alone does not prove that genuine fUSI signal changes are preserved.

### Batch stability

A full-recording CUDA comparison of batch sizes **1 versus 32**, with identical
100-step settings, measured worst per-frame grid RMS disagreement of
**6.85e-10 mm**, or **6.95e-9 minimum-spacing pixels** (rounded upward). The
functional check used a 0.005-pixel tolerance. See `batch_invariance.json` and
`serial_transforms.npz`.

During development, float32 geometry/optimizer state produced approximately
0.008-pixel disagreement on this check, and float32 NCC quantization destabilized
best-iterate selection near correlation one. The current float64 geometry/state
and NCC policy removes those observed instabilities. This precision policy needs
profiling before any throughput claim.

### Inspection artifacts

- `motion.png`: angle, physical center displacement, FD and optimizer summaries.
- `trace.png`: the same preselected maximum-standard-deviation voxel before/after.
- `raster.png`: the same native column and dB color limits before/after and for ITK.
- `comparison.png`: the frame with greatest geometric disagreement, including a
  power difference image on common valid support.
- `diagnostics.csv`, `voxreg_result.npz`, `summary.json`: labeled diagnostics,
  matrices, validity and framewise quality comparisons.
- `input.nii.gz`, `voxreg_registered.nii.gz`, `voxreg_valid.nii.gz`,
  `voxreg_minus_fixed.nii.gz`, `itk_registered.nii.gz`,
  `voxreg_minus_itk.nii.gz`: napari-readable spatially placed recordings.

NIfTI intensity/mask/time/affine round trips were checked on the actual outputs.
The FD calculation was also checked against the frozen reference: it evaluates
`norm(T[t+1] @ p - T[t] @ p)` on fixed-grid points, associates the statistic with
time t, and ends with zero. It is not a sum of Euler-angle derivatives.

## Open gates and limits

1. **Convergence calibration:** all 119 non-reference frames reached the configured
   100-step limit. None failed the overlap/metric guards, and quality matches the
   reference closely, but the strict convergence rule did not certify them.
   Their status is `max_iterations`, not `converged`.
2. **Runtime comparison:** the original contested timings were discarded. New
   isolated ITK/FireANTs comparisons include single-thread, eight-thread and
   eight-process ITK controls, CPU/CUDA FireANTs batching, separate timing/memory
   passes, and common quality scoring. The production scalar-affine sampler
   improves full-batch CUDA from 1.18 to 0.23 s and eight-thread CPU from 3.09 to
   1.98 s in matched runs. CPU remains slower than native float32 FireANTs. See [BENCHMARK.md](BENCHMARK.md). These do not establish a speedup
   over ConfUSIus's full public pipeline, which has not been cleanly retimed.
   The notebook/reference exporters still default to quality-only mode.
3. **Profiling/optimization:** the production scalar-affine sampler removes dense
   grid GEMMs while retaining float64 geometry/state/NCC. All 58 tests pass,
   including expanded sheared geometry, native support tolerance and CUDA batch
   checks. Real recording batch sizes 1/32/120 differ by at most 6.95e-9 pixels RMS
   on CUDA; same-device mapping change from baseline is below 6.70e-6 pixels RMS.
   Fresh-process full-batch CUDA peak allocation falls from 212.5 to 189.2 MiB,
   with process VRAM 514 to 478 MiB. See [BENCHMARK.md](BENCHMARK.md) and its tracked
   snapshot. Further fusion/compilation and broader-data validation remain open.
4. **Broader scope:** no full 3D, multi-pose, multiresolution, arbitrary-mask,
   differing-grid, affine or deformable validation has been attempted. These
   inputs are deliberately unsupported rather than silently approximated.

Do not extend the transform model before resolving the remaining rigid gates.
