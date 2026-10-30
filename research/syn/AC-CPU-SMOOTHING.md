# AC-qualified CPU volumetric SyN smoothing

## Decision

Stay with PyTorch and retain the batching caps. Promote scaled float32 Gaussian
convolutions **only for multithreaded, nondifferentiable CPU 3D SyN**, with the two
existing sigma values (update 1.5 and total-field 0.5 voxels). Planar, one-thread,
differentiable, compiled and CUDA paths retain their previous smoothing behavior.
The standalone `fields.smooth_field` primitive remains float64.

No precision/backend option, optimizer probe, thread-setting policy or OOM retry
was added. Kernel weights are bounded, read-only CPU constants for those two
sigmas. Fields are normalized in double before casting and restored in double
after filtering. Float64 physical coordinates, stored displacement fields,
composition, CC statistics, inversion and Jacobians remain unchanged. Images are
resampled from their original intensities; float64/integer outputs and calibration
retain their existing contracts.

## Controlled comparison

The preceding [battery experiment](CPU-SMOOTHING.md) was exploratory. This follow-up
ran serially, with AC/performance power-state samples every two seconds and a stop
on power/profile changes. Seven arms completed: baseline, gradient-only float32,
both-filter float32, Torch float64, then reverse-order repeats of both-filter,
gradient-only and baseline. Each uses eight CPU threads, 120 updates, one cold and
two warm calls, the same independent fixtures, seeds, original resampling and
requested transform output. Saved inputs were independently checked for exact
identity between all runs.

Warm CPU wall-time ranges over the two runs:

| Filtering | 2D pair | 3D pair | Eight 3D volumes |
|---|---:|---:|---:|
| SciPy float64 baseline | 0.421–0.482 s | 3.947–3.964 s | 11.521–12.458 s |
| Float32 update filter only | 0.463–0.464 s | 3.679–3.902 s | 10.362–10.978 s |
| Float32 update and total filters | 0.456–0.459 s | 3.421–3.567 s | 8.531–9.674 s |

The single Torch-float64 control took 0.441/3.375/11.659 s. This is not evidence
for an unconditional Torch backend switch. The two corresponding FireANTs CPU
batch controls were 5.804/6.322 s for float32 filtering and 6.103/6.457 s for the
baseline runs: clock/thermal variation remains, but the volumetric benefit
survives reverse ordering and reference controls.

A further thread-count filter check explains the serial fallback: for eight
volumes at one thread, update/total filters cost 7.857/5.872 ms with SciPy versus
8.642/6.666 ms with the float32 prototype. At 24 threads they cost 12.186/9.298 ms
versus 1.622/2.129 ms. These are isolated filters, **not full-call 24-thread
qualification or an automatic thread-count recommendation**.

## Fresh production three-way checkpoint

All reference arms were rerun after the production change on AC, with the same
8-thread/120-update/two-warm protocol. Warm wall seconds:

| Case | ITK CPU | FireANTs CPU | voxreg CPU | FireANTs CUDA | voxreg CUDA |
|---|---:|---:|---:|---:|---:|
| 2D medium pair | 1.028 | 0.317 | 0.431 | 0.353 | 0.581 |
| 3D medium pair | 10.083 | 4.668 | 2.956 | 0.416 | 1.013 |
| Eight 3D volumes | 50.866 | 6.380 | 8.133 | 0.585 | 1.293 |

These include complete requested transforms and original-intensity resampling,
not imports or disk export. ITK runs sequential frames with eight threads, not
eight concurrent registration workers. Batch volumes are 32 cubed; the medium
3D pair is 32 by 40 by 48. CUDA code was not optimized by this change; varying GPU
times must not be attributed to the CPU smoothing patch.

## Accuracy is separate from timing

All native truth gates pass. Worst per-frame independent truth RMS in pixels:

| Case | ITK CPU | FireANTs CPU | voxreg CPU |
|---|---:|---:|---:|
| 2D medium pair | 0.0664113 | 0.0809614 | 0.0761818 |
| 3D medium pair | 0.0797510 | 0.1012686 | 0.1001070 |
| Eight 3D volumes | 0.0626645 | 0.0802709 | 0.0763044 |

The medium-3D voxreg baseline was 0.100106967 pixels; the new result is
0.100107008 pixels. The improvement does not move accuracy toward FireANTs at the
expense of the previous baseline. It also **does not make voxreg numerically
identical to ITK**: ITK remains more accurate on these fixtures.

The fresh production maximum CPU/CUDA physical-point difference is
**0.000139373 pixels**, below the unchanged **0.001-pixel** gate. Across all seven
prototype runs, exported CPU maps were also compared directly with the saved
reference CUDA maps; every case passes that same gate. The independent truth RMS
gate remains **0.25 pixels**. No boundary, radius, test or tolerance was weakened.
These synthetic controls are not anatomical validation.

## What FireANTs does better, and where the cost went

Scoped profiles use eight volumes, eight CPU threads and 20 updates. Inclusive
phase times overlap and must not be added indiscriminately.

| voxreg phase | Float64 filters | Float32 filters |
|---|---:|---:|
| Complete profiled fit | 2.363 s | 1.715 s |
| 80 smoothing calls | 0.867 s | 0.171 s |
| Candidate updates, including smoothing/guards | 1.255 s | 0.585 s |
| Image-loss backward | 0.423 s | 0.400 s |
| 43 sampled-Jacobian evaluations | 0.247 s | 0.261 s |
| Two final halfway inverses | 0.194 s | 0.223 s |

This identifies an avoidable cost rather than attributing the whole gap to a
stronger contract. Smoothing falls from about 37% to 10% of profiled fit time.
Per-candidate Jacobian guards now account for about 15%; they are retained.

FireANTs' profile took 1.246 s. Its update/total field filters used 7-/3-tap kernels
and took 0.037/0.061 s; ours preserve 11-/5-tap kernels, replicate padding and
scaled conversions. FireANTs also uses float32 grids/statistics and native Torch
interpolation, with in-place optimizer work and different optimization/boundary
semantics. Its fused optional extensions remain unavailable in this environment.

**FireANTs does invert its output:** the recorded public inversion took 0.318 s.
Its backward/optimizer totals include that nested inverse optimization and cannot
be compared directly with voxreg's objective-only backward totals. No claim is
made that avoiding inversion explains its CPU lead.

Next cost targets are precision-preserving sampling/backward and guard arithmetic,
not removing regularity/inverse checks, shortening kernels blindly, changing caps
or introducing another backend.

## Proof and provenance

- Full default suite: **920 passed**, 1,098 existing warnings, 124.01 s.
- Real eight-thread registration/output qualification: **85 passed**, 28.14 s.
- **73 filter checks**, including production-path scaled ranges, exact boundaries,
  ownership, autograd and fallback selection.
- Native-grid/Newton compiled requalification: **85 passed**, 216.11 s;
  compilation remains research-only, not the default production path.
- Offline wheel/sdist build and **32 actual-wheel CPU/CUDA checks** pass at eight
  CPU threads, with exact equality of all eight shipped source modules. Wheel
  SHA256: `69432b2125dd9f76c3bdcce8a235340c58f5722ab2828f8cdb9a5f78264a0efb`.
- [Paired AC results, telemetry summary, cross-map checks and phase profiles](ac-cpu-smoothing-results.json).
- [Fresh production three-way results and source hashes](ac-production-three-way-results.json).

Raw artifacts remain under `artifacts/syn-speed/ac-smoothing-01/`,
`ac-production-three-way-01/`, `ac-profile-*-01/` and `ac-package-01/`.
The initial compiled-qualification invocation supplied an unsupported `--output`
argument and exited before running tests; that log is retained separately from
the corrected native-grid compiler qualification. It is not a compiler or
algorithm failure.
