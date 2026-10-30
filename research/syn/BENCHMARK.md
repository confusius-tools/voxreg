# SyN: accuracy and speed versus ITK and FireANTs

These are fresh **native symmetric SyN** runs, not the historical rigid/slab
benchmarks. Raw measurements, versions and source hashes are in
[benchmark-results.json](benchmark-results.json). All completed synthetic arms pass
the unchanged **0.25 native-pixel RMS physical truth gate**. Voxreg forward and
inverse CPU/CUDA points pass the unchanged **0.001-pixel maximum** gate.

## Latest default/eager three-way comparison

The `a7e2f45` production checkout was measured again with **all three libraries**,
without compilation or research patches. CPU thread count is **eight**, including
the host configuration for CUDA arms. Each arm uses 120 updates, one cold call and
**two warm repeats**. Raw times, versions, source hashes and every truth result
are in [three-way-eager-results.json](three-way-eager-results.json).

Warm wall time, seconds (lower is better):

| Case | ITK CPU | FireANTs CPU | voxreg CPU | FireANTs CUDA | voxreg CUDA |
|---|---:|---:|---:|---:|---:|
| 2D medium | 1.046 | 0.284 | 0.421 | 0.337 | 0.625 |
| 3D medium | 10.043 | 4.453 | 3.605 | 0.425 | 1.032 |
| 3D batch eight | 48.405 | 5.210 | 11.769 | 0.671 | 1.774 |

The subsequent small coordinate-reorder/smoothing-buffer changes were measured
separately at the same thread/update settings: voxreg CPU/CUDA is now
**0.412 / 0.613 s**, **3.354 / 0.991 s**, and **10.582 / 1.531 s** on these cases.
All truth/point gates pass; reference arms were not rerun for that isolated
follow-up. See [layout-eager-results.json](layout-eager-results.json) and
[SPEED.md](SPEED.md). The table above remains the last complete three-way run,
not a mixture of rows silently taken from different runs.

ITK here is native CPU SyN; there is no ITK CUDA arm. Its eight-volume adapter
runs frames sequentially with eight ITK threads, not an additional eight-worker
frame scheduler. These accuracy controls request dense transforms and original
image resampling; the recording throughput runs separately use transforms off.
Imports are outside timing; preparation, optimization, output construction and
resampling are inside. See the output-cost/optimizer differences below.

All truth RMS values pass 0.25 pixels. Medium 3D RMS is **0.079751 / 0.101269 /
0.100107 pixels** for ITK / FireANTs CPU / voxreg CPU: faster is not identical
accuracy. The largest voxreg CPU/CUDA point difference in these three cases is
**0.000083312 pixels**, below the unchanged 0.001-pixel gate.

Thus default voxreg beats ITK on these cases and FireANTs CPU on the single 3D
pair, but **still loses to FireANTs CUDA and to FireANTs CPU on the batch**. Small
2D CUDA is also slower than voxreg CPU. No universal parity claim is justified.
Compiled warm wins and their substantial cold costs are reported separately in
[SPEED.md](SPEED.md), never substituted into this default-path table.

```bash
uv run python research/syn/compare.py \
  --output artifacts/syn-speed/three-way-reproduction \
  --cases 2d-medium 3d-medium 3d-batch8 \
  --backends voxreg-cpu voxreg-cuda fireants-cpu fireants-cuda itk \
  --threads 8 --iterations 120 --repeats 2
```

The earlier baseline sections below retain their original source state and
repetition counts; they are not presented as current production measurements.

## Protocol and boundaries

- Hardware: Core Ultra 9 275HX, RTX 5080 Laptop GPU; Linux, Python 3.14.
  Torch 2.14.1, ITK 5.4.7, FireANTs 1.5.1 at the pinned development revision.
- Shared package-wide `nonlinear_pair_factory`, with genuine multi-axis smooth
  compositive deformation and independent reverse-order Newton truth. No
  production field routines generate truth. The oblique orthogonal-lattice
  variant is used for native reference compatibility; unit tests also cover shear.
- Same input arrays, prealignment, native resolution and **120 outer updates**;
  convergence windows exceed that budget. Radius-2 squared neighborhood CC;
  nominal update/total sigmas 1.5/0.5 voxels, step size 0.1. All inputs fit in
  float32. Voxreg/ITK fields use float64; FireANTs uses its float32 default.
- Voxreg uses normalized compositive gradient descent, ITK its native SyN method,
  FireANTs its native compositive SyN with **Adam**, including its public final
  reverse-half inversion. These are native implementations, not identical
  optimizers. Gaussian kernels/boundaries, inversion, NCC details and best/final
  state selection differ. Settings are fixed, not a hyperparameter search.
- Package imports are outside timing. Every timed call includes host preparation,
  image upload where needed, optimization, final forward-field construction and
  download, and original-intensity resampling. Reference methods use the same
  SciPy endpoint resampler. Voxreg additionally returns labeled metadata and the
  complete inverse on the original moving grid. Native ITK also computes inverse
  fields; FireANTs' public inverse accessor is unimplemented. Thus these are
  end-to-end workflow timings, not identical output-cost or pure-kernel timings.
- CUDA is synchronized before/after timing. First-call and every warm time are
  retained. Tables use warm medians: **three** repetitions for pair controls,
  **two** for the 8-thread and real-pair controls, and **one** for the eight-volume
  batch (a limited timing observation, not a robust distribution).
- FireANTs fused operations are **unavailable** in this environment; its actual
  Torch fallback is measured. No installed package or sibling source is patched.
  No replicated slices, inflated singleton slabs or hidden direction reversal.

## Pair truth accuracy

RMS physical displacement error divided by the smallest active native spacing;
all native grid points are scored, not just the best-looking supported subset.
Each entry is a single returned pair result; warm repeat variation is retained in
JSON. Lower is better; gate **<0.25 pixels**.

| Native shape | voxreg CPU | voxreg CUDA | ITK CPU | FireANTs CPU | FireANTs CUDA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D 25×29 | 0.06920 | 0.06920 | 0.06811 | unavailable | unavailable |
| 2D 64×80 | 0.07618 | 0.07618 | 0.06641 | 0.08096 | 0.08096 |
| 3D 17×19×21 | 0.08323 | 0.08323 | 0.08158 | unavailable | unavailable |
| 3D 32×40×48 | 0.10011 | 0.10011 | 0.07975 | 0.10127 | 0.09951 |

ITK is slightly more accurate on these controls, but all completed methods meet
the same gate. All completed outputs have positive sampled residual Jacobians;
that is a regularity diagnostic, not a global diffeomorphism certificate.
Voxreg/ITK inverse truth RMS is also retained in JSON (approximately 0.066–0.101
pixels across these pairs).

**Unavailable means an actual public-output failure, not a slow time omitted.**
On 25×29 FireANTs raises `NotImplementedError: Got 4D input, but trilinear mode
needs 5D input`; on 17×19×21 its inverse helper raises a scale-factor assertion
when its minimum grid is larger than the native image. Its optimizer running
before that failure does not count as a completed registration. The genuine
64×80 and 32×40×48 inputs allow complete native outputs without a workaround.

## Pair speed

Warm seconds, **one CPU thread** for every CPU arm; lower is better.

| Native shape | voxreg CPU | voxreg CUDA | ITK CPU | FireANTs CPU | FireANTs CUDA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D 25×29 | 0.263 | 0.647 | 0.151 | unavailable | unavailable |
| 2D 64×80 | 0.555 | 0.664 | 0.859 | 0.210 | 0.354 |
| 3D 17×19×21 | 0.927 | 0.897 | 4.524 | unavailable | unavailable |
| 3D 32×40×48 | 7.978 | 1.085 | 41.652 | 5.864 | 0.425 |

The 32×40×48 **eight-thread CPU control** measures ITK **11.418 s**, voxreg
**5.596 s**, FireANTs **5.032 s**, with unchanged RMS accuracy. Thus voxreg CUDA
is about **10.5× faster than 8-thread ITK** and **5.2× faster than 8-thread voxreg
CPU** here. FireANTs CUDA is still about **2.6× faster than voxreg CUDA**.
Small 2D GPU execution loses to CPU overhead; no universal GPU speedup is claimed.

## Eight-volume motion-style batch

Eight independently deformed **32³** frames, explicit shared fixed image, varying
amplitudes; all frames are optimized, rather than treating frame zero as an
unchanged reference. Same 120-step budget, **eight CPU threads** for CPU arms;
ITK processes the eight frames sequentially, GPU engines batch them.

| Backend | Worst frame RMS, pixels | Warm batch seconds |
| --- | ---: | ---: |
| voxreg CPU | 0.07630 | 17.539 |
| voxreg CUDA | 0.07630 | 2.819 |
| ITK CPU | 0.06266 | 50.833 |
| FireANTs CPU | 0.08027 | 6.222 |
| FireANTs CUDA | 0.07847 | 0.699 |

Voxreg CUDA is about **18.0× faster than this ITK workflow**, **6.2× faster than
voxreg CPU**, and **4.0× slower than FireANTs CUDA**. These are one warm batch
per backend, not a throughput saturation or large-recording memory study.

## Cached natural same-subject pair

The original ConfUSIus rat75 20220523/20220524 slice32 example, **1×112×128**,
max-referenced dB/qform representation. Raw input/sidecar hashes are retained.
A shared voxreg CPU rigid seed (400-update budget) is computed once **outside
SyN timing**, from original images. Its endpoint NCC² is **0.825157**.

| Backend | Endpoint NCC² | Warm SyN seconds |
| --- | ---: | ---: |
| voxreg CPU | 0.906820 | 1.157 |
| voxreg CUDA | 0.906820 | 0.665 |
| ITK CPU, one thread | 0.903779 | 2.268 |
| FireANTs CPU, one thread | 0.907778 | 0.333 |
| FireANTs CUDA | 0.907778 | 0.357 |

This demonstrates a real-data workflow, **not accuracy against anatomical truth**.
The whole-field minimum sampled residual Jacobian is 0.269 for voxreg, 0.381 for
ITK and 0.473 for FireANTs. Different deformations can increase correlation;
those numbers do not establish which anatomical mapping is correct.

## Numerical agreement and honest limits

Across all voxreg pair/batch/real controls, maximum forward CPU/CUDA discrepancy
is **0.000135172 pixels**; maximum inverse discrepancy is **0.000139742 pixels**.
Both remain below **0.001**, normalized by each output domain's smallest spacing.
The corresponding FireANTs forward discrepancies range from approximately
**0.00757 to 0.13932 pixels**. These are observations, not a claim that FireANTs
promises voxreg's precision contract; its faster default float32 execution does
not meet that same numerical gate in these runs.

Sampled inverse residuals, inter-library disagreement, independent truth accuracy
and CPU/CUDA numerical agreement are different quantities. No gate was relaxed.
The old fixed-100 float64 linear-motion failure remains a historical failure.
Pyramids, broad capture range, anatomy, large-brain memory and fused FireANTs
performance are not qualified here.

## Reproduce

From the repository root; every output directory must be new:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
uv run python research/syn/compare.py --output artifacts/syn-replay/pairs \
  --iterations 120 --repeats 3 --cases 2d-small 2d-medium 3d-small 3d-medium
uv run python research/syn/compare.py --output artifacts/syn-replay/threads8 \
  --iterations 120 --repeats 2 --threads 8 --cases 3d-medium \
  --backends itk voxreg-cpu fireants-cpu
uv run python research/syn/compare.py --output artifacts/syn-replay/real \
  --iterations 120 --repeats 2 --cases real-pair
uv run python research/syn/compare.py --output artifacts/syn-replay/batch8 \
  --iterations 120 --repeats 1 --threads 8 --cases 3d-batch8
```

Original arrays, returned fields/masks, logs and frozen own-source files stay
local under `artifacts/syn-implementation/`. No large generated arrays are tracked.
