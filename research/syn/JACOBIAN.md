# SyN CUDA: avoid packed Jacobian intermediates

## Finding

The existing sampled-Jacobian check spends substantial time packing full derivative
matrices. A standalone FP64 prototype avoids those buffers for exact axis-aligned,
permuted or reversed lattices. It reads component derivatives directly, using the
same signed inverse-lattice coefficients and the same determinant expression.
Oblique/sheared grids retain the existing general path.

**These measurements used an isolated prototype, not a production edit.** They
retained FP64 geometry, fitting budgets and regularity guards, without custom
kernels, a compiler or vendor transplantation. The direct Jacobian specialization
has since been implemented in production, alongside float32 SyN working fields;
final inversion and export retain double. See [the design contract](../../DESIGN.md).
The shared registration threshold is now **≤ 0.30 voxels**; this investigation
prioritizes runtime, without tuning accuracy or relabelling historical outcomes.

## Measurements

RTX 5080 Laptop, eight CPU threads, AC/performance. Frozen fitting source from
`artifacts/benchmark-syn-cuda-matched-01/source`, verified against its manifest.
Both variants use 100 updates, batch 32 for a 32-frame planar series (31 fits)
and batch 1 for a full `(80,150,80)` volume pair. This is **not** the complete
100-frame planar benchmark or a 100-volume recording.

**Warmed, instrumented diagnostics**, three complete calls per variant with
alternating order, all three retained. These are not formal five-call benchmark
results or separate-worker memory/accuracy qualifications.

| Workload | Current voxreg median [min–max], s | Direct FP64 prototype, s | Reduction |
| --- | ---: | ---: | ---: |
| 32 planes | 0.8760 [0.8693–0.8914] | 0.8522 [0.8326–0.8568] | 2.7% |
| Volume pair | 3.4112 [3.4111–3.4148] | 3.0804 [3.0632–3.0807] | 9.7% |

The inclusive Jacobian CUDA-event spans fall from **65.0 to 42.0 ms** for planes,
and **587.6 to 261.6 ms** for the volume pair. Each call still executes **203**
Jacobian evaluations: both candidate guards per update plus final diagnostics.
These spans include dispatch gaps and are not isolated GPU kernel times.

Separate warmed Jacobian microloops (five samples per variant, twenty evaluations
per sample) measure median per-evaluation spans of **0.287→0.170 ms** for planes
and **3.059→1.297 ms** for volumes. Sampled excess Torch allocation drops from
**198.0 to 161.2 MiB** for the volume microloop. Whole-call Torch allocation peaks
only fall from approximately **1.027 to 0.999 GiB**; other phases still dominate
some peaks. No process RSS or driver-memory claim.

A fresh warmed FireANTs Python peer, identical inputs/batches/update count, gives
**0.3812 s** for the 32-plane batch and **2.2689 s** for the volume pair. It has
no internal Jacobian event instrumentation, unlike the voxreg calls, so this is
context rather than a formal paired speed ranking. The volume gap narrows from
roughly **1.50× to 1.36×**; the planar gap remains roughly **2.2×**. FireANTs'
optional fused extension remains unavailable.

## Preservation checks and limits

- Bitwise-identical CUDA Jacobians for original and permuted/reversed lattices.
- Bitwise-identical results through the sheared fallback.
- Small CPU autograd checks match exactly in both dimensions.
- All twelve measured voxreg calls have bitwise-identical registered outputs to
  the baseline, unchanged update/status records and unchanged Jacobian call counts.
- Inputs, source hashes, batch sizes and AC/performance are recorded. The
  prototype is independent of the earlier native-double sampler experiment;
  their gains have **not** been assumed additive.

The prototype is **unpromoted**. Next: freeze an implementation candidate and run
normal full-workload five-call timing, separate memory/accuracy workers and the
field/geometry regression suite before changing production. Broader CPU/CUDA
qualification and a full 100-volume workload remain unmeasured.

Evidence: `artifacts/syn-jacobian-investigation-01/` contains `probe.py`,
`results.json`, `fireants_probe.py`, `fireants.json`, logs and verification.
