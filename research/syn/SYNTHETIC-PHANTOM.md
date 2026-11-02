# New synthetic phantom: numerical failure versus visible alignment

Bounded follow-up to the single synthetic benchmark. No optimizer, precision,
boundary, iteration default or acceptance tolerance was changed in production.
These are diagnostic registrations, **not performance measurements**.

## What was reproduced

The previous short-run artifact paths had been removed before this follow-up.
The textured 2D failure was reproduced independently using the same deterministic
8-frame phantom and probes 1/4/7. Probe 4 has the largest motion. Every map is
scored at all 150×128 fixed voxel centers, divided by minimum spacing 0.1 mm.

| Method | Worst probe truth RMS, px | Frame-4 maximum truth error, px |
|---|---:|---:|
| Identity | 0.695 | — |
| voxreg CPU, 100 updates | 0.2642 | 1.031 |
| voxreg CUDA, 100 updates | 0.2642 | 1.031 |
| ITK, 100 updates | 0.2284 | 1.000 |
| FireANTs CUDA, 100 updates | 0.2998 | 1.274 |
| voxreg CPU, 300 updates | 0.1790 | 0.804 |
| voxreg CUDA, 300 updates | 0.1806 | 0.809 |

The extra-update control supports **partial underconvergence at the 100-update
budget**, not a gross physical-coordinate error. It does not establish that more
iterations alone recover the exact deformation. The 300-update rows are separate
diagnostic controls, never substituted into fixed-100 timing comparisons.

## What the images show

- [Whole field: before, independent truth, voxreg CPU/CUDA, ITK and FireANTs](../../artifacts/synthetic-investigation-01/syn-overview.png).
- [25-pixel crop selected at the worst foreground CPU truth error](../../artifacts/synthetic-investigation-01/syn-zoom.png).
- [Worst CPU/CUDA-disagreement probe, with explicit mapping/intensity scales](../../artifacts/synthetic-investigation-01/cpu-cuda.png).

Images share scaling: fixed is red, registered moving is cyan; agreement appears
gray. Mapping-error heatmaps use a deliberately sensitive 0–0.8-pixel scale and
must not be read as intensity differences. All displayed registrations resample
the original source with the same independent SciPy float64 interpolator. Even
truth resampling has ordinary sampled-image interpolation error.

**The registered images look close to truth at normal scale and at the selected
zoom.** That agrees with the view that 0.2–0.3-pixel RMS is not automatically a
practically bad registration. But RMS hides local excursions: this example has
about a one-pixel maximum foreground mapping error. Smooth blobs and weak texture
can look aligned while their full geometric motion remains poorly identified.
That is consistent with aperture/conditioning ambiguity, not proof of the exact
optimizer-level cause. Thin anatomical structures could behave differently.

## CPU/CUDA reproducibility is a different issue

At 100 updates across all three probes:

- Maximum physical-point disagreement: **0.08975 px**.
- RMS disagreement: **0.00236 px**; 99th percentile **0.00433 px**.
- **2.13%** of sampled points exceed the strict 0.001-pixel gate.
- The largest differences are localized in weak image regions, as the map shows.
- For the worst-disagreement frame (frame 1), resampled image RMSE is
  **0.000258% of the shared display range**, and maximum intensity difference is
  **0.00850%**. Those image differences are not visually obvious at ordinary scale.

One-thread and eight-thread CPU maps are **bitwise identical** in this control.
This is planar SyN, which does not use the recently promoted CPU 3D float32
smoother. That change therefore does not explain this particular discrepancy.
At 300 updates the maximum CPU/CUDA map discrepancy grows to **0.12896 px**, even
while truth RMS improves. Increasing iterations is not a numerical-parity fix.

Small backend arithmetic differences amplified in weakly constrained local-CC
regions are a plausible explanation; the exact first divergent operation was
not traced. No low-level root cause is claimed. The existing strict gates remain
useful regression screens and remain unchanged; failing one is not synonymous
with visibly unacceptable anatomy.

## Reproduction and limits

Raw maps, metrics, figures and the diagnostic script are under
`artifacts/synthetic-investigation-01/`. `metrics.json` includes per-probe RMS,
95th-percentile/max errors, foreground errors and thread/backend comparisons.
The initial diagnostic named `inspect.py` shadowed Python's standard library;
its failed log is retained, and the successful script is `diagnose.py`.

This is one synthetic phantom, not anatomical validation or a universal error
budget. No new test files were added. The benchmark's separate 100-frame matrix
uses the unchanged fixed-100 budget, both FireANTs devices by default, and its own
cold/warm measurements and accuracy probes.

## Regular ANTs

A native ANTs comparison is worth adding, with **two distinct labels**:

1. **Controlled single-resolution ANTs:** same input, metric family, declared
   update budget and original-image resampling. This helps distinguish backend
   implementation cost from workflow differences.
2. **Standard ANTs workflow:** the usual rigid→affine→SyN multiresolution recipe,
   with native stopping and a declared output/timing boundary. This addresses
   practical time-to-quality, not equal iterations.

The official [`antsRegistrationSyN.sh`](https://github.com/ANTsX/ANTs/blob/main/Scripts/antsRegistrationSyN.sh)
combines stages, metrics and resolution schedules; it is not equivalent to this
benchmark's identity-start single-resolution native ITK SyN arm. See also
[Anatomy of an antsRegistration call](https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call).
ANTs is an important neuroimaging reference, not evidence of universal superiority
on every fUSI workload. Neither `antsRegistration` nor ANTsPy is installed in this
environment; no ANTs timing is claimed and no heavy new installation was made.
The separate [native workflow timing script](../../BENCHMARK-ANTS.md) now provides
command planning and execution for temporal-mean/rigid motion correction and the
installed rigid → affine → SyN registration wrapper, without changing this matrix.
