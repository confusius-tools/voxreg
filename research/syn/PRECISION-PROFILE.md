# SyN CUDA: precision versus implementation cost

## Conclusion

**Float64 field work explains a large part of the runtime gap, but not all of it.**
Image/NCC statistics already use float32 in the current benchmark. The expensive
float64 work includes field state, composition, inverse solves, Jacobians and
sampling-coordinate arithmetic. A native sampler implementation also matters
independently of coordinate precision.

There is a larger **FP64-preserving candidate**: native-double fitting sampling
plus the direct Jacobian specialization reduces volume-pair runtime by about
**23%** in these diagnostics. It retains all regularity checks and float64 field
math. Batching the two halfway updates together was tried and **rejected**: extra
buffer/packing costs outweighed launch savings.

These diagnostic measurements predate the operator's decision to adopt native
float32 image fitting and float32 working fields. Production now implements that
policy plus the direct Jacobian, while retaining double physical master geometry,
final inversion/validation and original-image resampling. That finalization differs
from the all-float32-field diagnostic arm; its timings are not interchangeable.
The shared registration gate remains ≤ 0.30 voxels, without further accuracy tuning.

## Controlled factorial

RTX 5080 Laptop; eight CPU threads; AC/performance; identical hashed inputs;
100 updates and the same public API workload in every arm. The planar diagnostic
is **32 frames / 31 fits**, not the full 100-frame benchmark. The volume is one
`(80,150,80)` pair. Both TF32 flags are explicitly **off** in the primary table:
this compares ordinary float32 and float64, not TensorFloat-32 acceleration.

**Warmed diagnostics, median [min–max] of three uninstrumented complete calls**,
all samples retained. Not formal five-call timing/memory/accuracy qualification.

| Variant | 32 planes, s | Volume pair, s |
| --- | ---: | ---: |
| Current implementation | 0.895 [0.893–0.900] | 3.405 [3.393–3.437] |
| Float32 field path; original image sampler | 0.636 [0.609–0.683] | 2.340 [2.333–2.368] |
| Native **float64** image sampler; float64 fields | 0.871 [0.866–0.872] | 2.968 [2.966–2.973] |
| Native float32 image sampler; float64 fields | 0.829 [0.819–0.836] | 2.850 [2.850–2.858] |
| Float32 field path + native float64 image sampler | 0.628 [0.576–0.647] | 1.925 [1.916–1.948] |
| Float32 field path + native float32 image sampler | 0.596 [0.516–0.602] | 1.787 [1.764–1.806] |
| FireANTs, native float32 | 0.372 [0.359–0.389] | 1.509 [1.486–1.511] |

Lowering only the field path saves approximately **29% / 31%** (planes/volume).
The complete mixed-precision variant saves **33% / 48%**, but remains approximately
**1.60× / 1.18×** the FireANTs time under these matched TF32-off settings.

The native-float64 versus native-float32 sampler arms use the same coordinate
preprocessing and native operation; only image/query dtype at that call differs.
Changing fixed-order sampling to native-float64 sampling is an **implementation**
comparison, not a pure precision comparison.

The float32-field arms lower the field lattice, state and field-operation math.
The affine seed/image preparation stays float64, and physical export/scoring is
float64. They are therefore **not an entirely float32 pipeline**. They also do
not preserve the current general-geometry contract and must not be promoted from
this experiment alone. TF32 override environment variables were unset.

## Where the time goes

Separate 100-update CUDA-event diagnostics, inclusive nested spans: do not add
these rows together.

| Volume phase | Current, ms | Float32 fields only, ms |
| --- | ---: | ---: |
| Image interpolation | 649 | 653 |
| Objective backward | 757 | 626 |
| Candidate updates, including composition/smoothing/guards | 1,347 | 636 |
| ↳ Sampled Jacobians (also includes final checks) | 585 | 192 |
| ↳ All compositions (also includes finalization) | 299 | 80 |
| Two halfway inverse solves | 278 | 108 |
| All field smoothing | 367 | 324 |

The major precision effect is **not just NCC or Gaussian filtering**. Composition,
Jacobian work, backward/state handling and inverse solves are larger contributors.
Native-double sampling separately reduces the volume image-interpolation span
from approximately **649 to 310 ms** without lowering coordinate precision.

Six bounded **20-update** raw CUDA traces provide kernel-only counts and active
kernel interval unions (not occupancy). For the volume: current **32,116 kernels,
848 ms summed kernel time**; native-double sampler **21,728 / 733 ms**; float32
fields + native-float32 sampler **22,021 / 360 ms**. Similar launch counts in the
last two arms but very different device times show a real compute/memory/dtype
cost, not only Python launch overhead. The short profile has relatively more
finalization; do not extrapolate its phase shares to 100 updates.

## Larger improvement without dropping FP64

A separate warmed experiment combines native-double image sampling and the
[direct FP64 Jacobian specialization](JACOBIAN.md). Three complete calls per arm:

| Variant | 32 planes, median s | Volume pair, median s |
| --- | ---: | ---: |
| Native-double sampler only | 0.859 | 2.953 |
| + direct Jacobian | 0.809 | 2.627 |
| + paired halfway updates | 0.851 | 3.011 |

The paired update retains both halfway regularity checks and produces bitwise
identical measured outputs, but is slower; keep it rejected. The useful combined
candidate is approximately **10% / 23%** below the original implementation in
the primary study. Its outputs match the native-double-only control bitwise in
all measured calls. The native sampler itself changes the original fitting
trajectory, so the combined candidate is not claimed bitwise-equivalent to the
original fixed-order sampler.

## Backend flags substantially affect the FireANTs comparison

Independent volume controls separate the two TF32 flags:

| Matmul TF32 | cuDNN TF32 | FireANTs median s |
| --- | --- | ---: |
| Off | Off | 1.486 (repeat of primary 1.509) |
| On | Off | 1.578 |
| Off | On (installed PyTorch defaults) | 2.250 |
| On | On | 2.351 |

Allowing cuDNN TF32 is **slower** for this FireANTs workload. This is a backend
path-selection observation, not evidence that TF32 is generally slower. The
forward NCC spans are similar (~204–207 ms) in the two single-flag controls;
optimizer spans differ (~1,265 vs 2,011 ms). Exact backward-kernel attribution
remains unmeasured; do not call this merely a faster forward NCC kernel.

The FP64-preserving combined voxreg prototype was rechecked with default flags
(matmul off, cuDNN on): **2.619 s**, versus **2.250 s** for FireANTs under the same
flags. This narrows the volume gap from roughly **51% to 16%**. Under TF32-off
controls FireANTs is faster still, so there is no blanket parity claim. New
registration benchmark workers record both TF32 flags and related cuDNN settings.

## Numerical observations and evidence

The same analytic scorer was applied outside timing, without tuning settings.
Worst-frame RMS remains approximately **0.264–0.269 voxels for planes** and
**0.30631–0.30646 for the volume** across the six voxreg arms. Similar RMS is not
numerical equivalence: planar local field changes reach approximately **0.18
voxels**, and RMS field differences from the original are ~0.016–0.020 voxels.
Float32/general geometry and large-origin accuracy are not qualified here.

`artifacts/syn-precision-investigation-01/` holds frozen source clones, the
hash-matched `worker-main.py`, later control workers, exact settings, all samples,
phase events, six raw traces, fields, independent scores and source checks in
`summary.json`. `f64-combo*.json` records the FP64 combination and rejected paired
update. Two harness failures (module-reload class identity; missing log directory)
are retained separately and excluded; completed fits were not rerun silently.
The generic raw FireANTs `field_dtype` label was unused and incorrectly defaulted
to the voxreg float64 label; actual float32 initialization is verified in logs and
called out in the analysis rather than rewriting raw measurements.

The initial recommendation was to qualify the FP64-preserving combined candidate.
The subsequent operator-approved implementation instead adopts float32 fitting
with double finalization. Its regression and matched benchmark evidence lives in
`artifacts/syn-float32-production-01/` and `artifacts/benchmark-syn-float32-01/`.
Benchmark workers disable TF32 for both backends; public APIs leave global flags
unchanged. Original-image and offline resampling retain their precision contracts.
