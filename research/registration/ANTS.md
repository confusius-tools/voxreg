# Historical native ANTs workflow evidence

These are completed **single first invocations**, not the current benchmark's
five-run median/range or precision-matched resident-array comparisons. Use
[BENCHMARK.md](../../BENCHMARK.md) for the one current pipeline.

Official native ANTs **2.6.5-gfdce4d2**, eight CPU threads; release archive SHA256
`504b6ab94dc09fa414874fdb847f6fced03f143763501e60cd51d247344641cd`.
Both 100-frame motion workloads completed for dense/default sampling and the
additional, explicitly random 5% sampling arm:

| Workload | Dense/default seconds | Random 5% seconds |
|---|---:|---:|
| 2D+t | 42.084194912 | 23.528618537 |
| 3D+t | 2176.488927526 | 157.285187323 |

Timing includes temporal-mean creation, rigid motion correction, native CLI
startup/I/O and outputs. Native motion uses a temporal-mean reference and fits
all frames; resident recordings instead preserve their exact first frame.
Native motion uses GC, Rigid[0.01], 100×50×20 iterations, shrink 4×2×1,
smoothing 2×1×0 voxels, physical scales, linear interpolation and seed 1. Only the
random arm adds `Random,0.05` metric sampling.

All 100 motion transforms were independently reconstructed in each arm. Worst
full-grid RMS against common frame-zero truth, in minimum-spacing voxels:

| Workload | Dense/default RMS | Random 5% RMS |
|---|---:|---:|
| 2D+t | 0.0418236495 | 0.0440782706 |
| 3D+t | 0.0367504159 | 0.0365004780 |

The staged `antsRegistrationSyN.sh -t s` volume pair completed in
**26.668199485 seconds**. Output shape, lattice, finiteness and image QC passed;
image NCC² was approximately 0.99515. This was not originally independently
qualified against nonlinear physical truth. During benchmark cleanup, its
retained forward field/affine were scored without running a new registration:
**0.6782088627 voxels RMS**, failing the unchanged 0.25 screen. The affine-after-
displacement composition reproduces the retained native resampled image to
8.9e-15 absolute intensity error on the declared interior. Good image correlation
is therefore not a substitute for the physical-truth gate. This old staged
result is not silently presented as an accuracy pass.

Evidence (ignored, local files):

- `artifacts/ants-native-100-dense-01/results.json`.
- `artifacts/ants-native-100-random5-01/results.json`.
- `artifacts/ants-native-pair-defaults-01/results.json`.
- `artifacts/ants-native-100-comparison-01.json`.
- `artifacts/thread-scaling-ants-source-02/`: frozen source, command, truth
  reconstruction and environment provenance; supervisor exited zero.
- `artifacts/benchmark-cleanup-source-01/`: removed runner source and bounded
  validation records for the consolidated pipeline.

Those native jobs completed; cleanup did not resume them. Historical dense and
random arms remain separate evidence; the dense default was not replaced by the
faster sample.
