# CPU volumes: shared coordinates and sampler work partitioning

This follow-up uses the installed PyTorch 2.14.1 CPU/CUDA backend on the Core
Ultra 9 275HX / RTX 5080 Laptop workstation, on AC/performance power. All heavy
runs were serial, **after** the native ANTs queue finished. No registration
precision, physical convention, stopping guard, production batch cap, or public
API was changed. Nothing was copied from FireANTs implementation code.

## Changes retained

- **CPU multi-volume linear sampling:** cache one normalized fixed lattice and
  apply the batched float64 physical mapping with `bmm`. Generate native sampler
  x/y/z directly, avoiding a full-volume component reversal and its backward.
  The cache is about **23 MB** for the native `(80,150,80)` volume, shared across
  the batch. It is not a repeated or prewarped fitting image.
- **CPU single-volume sampling:** split the existing query points into independent
  sampler batch tasks, using an expanded **zero-stride view of the same image**.
  No moving-volume copies or interpolation changes. Padding needed for indivisible
  query counts is removed before returning values or computing the loss.
- Keep the original single-volume coordinate expression: the cached skinny GEMM
  regressed there at high thread counts. Keep nearest-neighbor coordinate
  arithmetic unchanged, including tie behavior. Image-gradient calls bypass the
  sampler partitioning. Planar, mixed-grid, compiled and CUDA paths retain their
  previous coordinate construction; displacement/prepared sampling is unchanged.

Geometry/state/NCC and final interpolation remain float64; fitting images and
native fitting interpolation keep their existing policy. NCC organization was
already shared with the FireANTs comparison. Float32 NCC was **not** promoted.

## Full-size 100-update checks

**One first invocation per fresh interpreter**, eight CPU threads, resident inputs;
not warm medians or statistical speed guarantees. All fits used an explicit fixed
reference. Thus all 32 volumes were fitted, with no inactive reference or tail.
The pair used one native volume. Timings include preparation, fitting, original
image resampling and CPU return; physical-matrix export was enabled in both arms.

| Workload | Frozen prior engine, s | Updated engine, s | Change |
|---|---:|---:|---:|
| 32 volumes, rigid, 100 updates each | 97.861 | 77.694 | 20.6% less time |
| One volume pair, rigid, 100 updates | 5.877 | 2.689 | 2.19x faster |

The 32-frame input hash is
`f03eefd465ec335ae3f8807d306e30a4dd42b9935de0c277d9ba10764b7698a9`.
Independent full-grid affine truth RMS, evaluated analytically for **every fitted
frame**, is at worst **0.003284 native pixels** for the batch and **0.002576** for
the pair, below the unchanged **0.25-pixel** screen.

Maximum old/new CPU dense-grid point discrepancy is **1.90e-11 pixels**.
Updated CPU versus unchanged CUDA discrepancy is **2.26e-8 pixels**, below the
unchanged **0.001-pixel maximum** gate. For affine differences, the maximum norm
on the native grid is attained at a grid corner; all eight corners are checked.
Original output shape/dtype/finiteness and all requested update counts were checked.

## FireANTs adapter: four volumes was not a native limit

The old runner conservatively selected four volumes for all models, without a
measured claim that four was best. Fresh 4/8/16/32 CPU screens motivated the
following same-input, full-100-update check, using native float32 state and the
existing custom masked float32 NCC. These are **within-FireANTs batching controls**,
not precision-matched timing comparisons with voxreg.

| Backend, 32 fitted rigid volumes | Adapter batch 4, s | Adapter batch 32, s |
|---|---:|---:|
| CPU, eight threads | 75.311 | 67.669 |
| CUDA | 25.811 | 9.385 |

All four full runs pass the 0.25-pixel truth screen: worst RMS is approximately
**0.00932 pixels**. However, the fitted-identity stress reveals a native precision
limit: batch 32's CPU/CUDA maximum is **0.008632 pixels**, and CUDA batch 4 versus
32 differs by **0.009089 pixels**. Both **fail 0.001-pixel equivalence**. The
outlier is the already-aligned frame zero, which these explicit-fixed controls
fit rather than preserve. CPU batch 4/32 matrices are bitwise identical.

The recommended recording benchmark has always **preserved frame zero**.
An additional, separately retained full-100-update check of that exact contract
used 33 input frames, preserved the reference exactly and fitted the other 32.
Batch 32 passed all-frame truth and **0.001-pixel CPU/CUDA parity**, with maximum
**0.000181 pixels**. This does not erase or replace the failed fitted-identity
stress, or establish generic native FireANTs numerical invariance. Voxreg passes
that identity stress as well; no precision gate was relaxed for it.

Batch 32's CUDA peak Torch allocation is **2.02 GiB** on the 16-GiB GPU.
Translation and affine additionally passed direct full-size
32-fitted-frame CPU/CUDA scheduling, output and reference checks with one update;
those are bounded smokes, not full-budget truth qualification for those models.

The recommended runner now uses **32 planes or 32 linear-model volumes**.
Volumetric SyN keeps its conservative four-volume setting; it was not included in
this batch sweep. The protocol is `synthetic-v2`. Historical `synthetic-v1`
FireANTs timings used four volumes and must not be silently mixed with v2 results.
Voxreg production caps are unchanged.

## Thread scaling: improved, not solved by adding threads

The initial unprofiled screen used 32 explicit-fixed volumes and eight rigid
updates, one first and one warm call. Warm seconds:

| CPU threads | Prior engine | Combined coordinate candidate |
|---|---:|---:|
| 8 | 8.758 | 7.386 |
| 24 | 11.323 | 8.694 |

These are short diagnostic samples, not confidence intervals. Twenty-four threads
still do not beat eight here; the engine does not change the user's thread budget.

Separate **profiled** two-update calls corroborate the dataflow improvement.
Across the call, coordinate-forward scopes fell from **488 to 277 ms** at eight
threads, and **660 to 306 ms** at 24. Total `aten::flip` inclusive time fell from
**354 to 0.5 ms** at eight threads: only tiny geometry reversals remain.

The batched sampler itself was faster at 24 threads in these profiles. The
unchanged NCC-forward scope, however, took about **942 ms at 24 versus 527 ms at
eight** in the updated engine. Therefore the CPU sampler's batch-only partition
is **not by itself an explanation of the net slowdown**. Large tensor arithmetic,
reductions and synchronization matter. These are instrumented operator timings,
not unprofiled runtime estimates. No hardware bandwidth counters, pinned-placement
experiment or measured P/E-core attribution was collected; DRAM saturation is not
established.

## Rejected candidates and scope

- Plain float64 `affine_grid`: no demonstrated whole-call benefit.
- Float64 `einsum` NCC dots: slower on the 32-volume control; no promotion.
- Tiling every volume through separate sampler calls: no eight-thread batch win,
  and a substantial 24-thread regression. Only the single-volume case is retained.
- Float32 NCC: diagnostic only; not a new precision policy.

The full 60-case synthetic matrix, anatomical data, older Python/SciPy registration
suites and wheel qualification were not rerun for this change. Existing synthetic
SyN strict-screen failures are not fixed or reclassified here.

## Evidence and reproduction

`artifacts/cpu-volume-improvement-01/` retains frozen old/current sources, source
hashes, exact commands, endpoint power, physical matrices, JSON, operator tables
and logs. `source/` freezes the prior engine; `source-production/` captures the
linear implementation used for the full batch check. The final nearest-neighbor
fallback adds a guard to that snapshot without changing the qualified linear path.

```bash
# Local retained reproducer; always choose fresh output directories.
.venv/bin/python artifacts/cpu-volume-improvement-01/full.py baseline \
  --output artifacts/cpu-volume-improvement-01/reproduce-baseline
.venv/bin/python artifacts/cpu-volume-improvement-01/full.py production \
  --output artifacts/cpu-volume-improvement-01/reproduce-production
# Native adapter arm: fire4 / fire32, --device cpu / cuda.
# Separate attribution: operator_profile.py baseline / production --threads 8 / 24.
```

Initial invalid intensity-scaling harness calls and four FireANTs import failures
are preserved. A local `profile.py` shadowed Python's standard module before
FireANTs fitting began; it was renamed `operator_profile.py`, and only those four
arms were explicitly rerun into fresh `*-full100-r2` directories. No vendor patch,
precision retry, tolerance relaxation or hidden workload reduction was used.

The focused geometry/sampling/initialization/linear/mixed-grid gate passed **220
tests**, including 16 new production-sampler regression cases. The complete
CPU/CUDA suite passed **936 tests**, with 1,098 warnings in 132.94 s. No benchmark
or example unit tests were added. Full-suite and hook logs are retained alongside
the benchmark evidence.
