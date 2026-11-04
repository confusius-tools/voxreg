# Validation and known limits

## Recorded checkpoint

At the float32/PyTorch policy checkpoint (`40b4c01`), the production suite passed
**717 tests** on CPU/CUDA and all five hooks passed. This is a recorded result,
not proof that every dataset, platform or benchmark configuration is qualified.

```bash
uv run pytest
uv run prek run --all-files
```

CUDA tests skip when unavailable; CPU-only success does not qualify CUDA. See
[shared fixtures](tests/conftest.py) and [oracles](tests/oracles.py).

## What is checked

| Area | Independent evidence |
| --- | --- |
| Physical geometry | Known world transforms, native ITK/SimpleITK point checks, unequal/oblique/cropped/strided grids and large origins |
| Sampling and output | SciPy interpolation, finite singleton support, original dtype/intensities, integer rounding, metadata, input immutability and exact reference preservation |
| Metrics and derivatives | Explicit NCC calculations, float64 gradient checks, float32 constant-neighborhood/row-alignment regressions |
| Per-frame state | Batch/failure isolation, best-valid fallback, frozen stopped rows, seed validation and callback ownership |
| Nonlinear fields | Independent composition/interpolation, inverse residuals, Jacobians, smoothing range/boundaries/autograd and analytic nonlinear recovery |
| Convergence | Stored C++ ITK reference values and an independent stopping-policy calculation |
| Optional outputs | Default-off/opt-in parity and checks that dense export does not run when disabled |

Focused CPU/CUDA numerical checks remain useful regressions. End-to-end map
agreement with another device, older voxreg or another library is **not truth**
and is not a benchmark acceptance gate.

## Benchmark accuracy

[BENCHMARK.md](BENCHMARK.md) defines the single measurement pipeline: minimize
median runtime subject to **worst fitted-frame truth RMS < 0.25 minimum-spacing
voxels** and fitting in memory. Library unit tests and benchmark accuracy are
separate checks; passing the former does not imply passing every full workload.

The current precision policy has these **one-call, 100-update diagnostic** results
(eight threads, batch limit 32), not five-run performance measurements:

| Workload | CPU truth RMS (vox) | CUDA truth RMS (vox) | Accuracy |
| --- | ---: | ---: | --- |
| Planar series, reference + 32 fits | 0.264253 | 0.265066 | **Fail** |
| Volume pair, one fit | 0.306317 | 0.306314 | **Fail** |

These workloads already failed the truth target before the precision change.
No accuracy improvement or universal speedup is claimed. A full five-run matrix
for the current policy has not been reviewed here; older completed reports retain
their original source/settings.

The benchmark runner has a real three-phase CPU smoke check and bounded checks
for unchanged configuration IDs, existing-output protection, accuracy/execution
failure, unavailable backends, interruption, unexpected parent errors and saved
text/HTML reports. These are direct runner checks, not another test framework.

## Remaining limits

- **Anatomical accuracy:** inspected real examples and higher NCC do not establish
  correct correspondence. Historical same-subject/atlas failures remain evidence,
  not successes retroactively inferred from the new policy.
- **Capture and stopping:** initialization, overlap and the fitting budget matter.
  Default stopping was calibrated on bounded rigid-slice windows, not all affine,
  SyN, 3D or partial-overlap problems. `max_iterations` is not convergence.
- **Invertibility:** positive sampled Jacobians do not prove global invertibility;
  sampled SyN inverses have nonzero residuals.
- **Memory:** batching does not bound full CPU inputs/outputs or dense exports.
  Large-volume and full-recording limits require workload-specific measurement.
- **Compatibility:** Python 3.11–3.14 syntax/resolution were checked, but full
  numerical/CUDA qualification was on Python 3.14/Linux. Older wheel checks do
  not qualify the latest source on every supported environment.

## Evidence

- [Linear geometry/recovery](research/linear/README.md),
  [slice/volume sampling](research/slice-geometry/NATIVE-FIXED.md) and
  [SyN checks](research/syn/README.md).
- [Stopping calibration](research/convergence/CALIBRATION.md) and
  [real-example investigations](research/confusius-examples/README.md).
- Local current-policy runner checks: `artifacts/benchmark-policy-01/`.
- Compact preliminary source/hash/log evidence:
  `artifacts/archive/preliminary-evidence.tar.gz`; generated bulk arrays were
  discarded. Completed reports and installed tools were retained.
- The detailed historical validation timeline is linked in the
  [historical guide index](research/README.md#pre-cleanup-guides).
