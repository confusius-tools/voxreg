# Removing the fixed coverage cutoff

> Historical qualification record. See [archive scope](../README.md) and the
> [current benchmark](../../BENCHMARK.md); production overlap checks remain.

The former requirement that a moving image cover at least 90% of the fixed image
was a voxreg policy, not a requirement of the ITK correlation metric. It rejected
valid smaller moving fields of view and blocked the real same-subject alignment.

The shared optimizer now uses **more than one valid sample center**, not a coverage
percentage, at both iterative evaluation and final-candidate evaluation. The NCC
checks for finite arithmetic and nonzero variance remain unchanged. Fewer than two
valid centers keeps the existing `insufficient_overlap` status; unusable variance
or arithmetic is `invalid_metric`. The best valid pose/initial fallback, affine
singularity/orientation guards and reported `valid_fraction` remain intact.

This mathematical sample minimum is not an identifiability or confidence test.
Two points can yield perfect correlation without determining a rigid/affine map;
small overlaps may still produce misleading fits. Coverage is reported for
inspection, not used as a universal acceptance percentage. No replacement
percentage, configurable threshold or new masking API was added.

## Regression proof

`tests/test_overlap.py` adds 21 cases, including all three models on CPU/CUDA with
aligned smaller moving images in both 2D and 3D. The moving image covers only
approximately 10–29% of the fixed grid; it must not be rejected for that fraction.
Tests also distinguish zero/one/two valid centers, constant target values within
partial coverage, zero-overlap seed fallback, output fills/dtypes and input
preservation. Before the patch, 16 of these assertions failed; all 21 now pass.

- Full suite: **373 passed**.
- All six 2D/3D native ITK linear-model controls were rerun unchanged: 18 backend
  truth checks still pass RMS <0.25 px, six CPU/CUDA comparisons still pass maximum
  pointwise <0.001 px. Numerical results match the prior linear qualification.
- Four frozen real120-frame CPU/CUDA fixed/convergence motion-correction controls
  remain bitwise identical in every stored output field.
- Ruff, format, lock, source/wheel build and isolated small-moving-image wheel
  checks pass. No timing or memory claim is inferred from these checks.

## Real ConfUSIus example follow-up

The existing [same-subject reproducer](../confusius-examples/compare_same_subject.py)
was rerun with the same cached rat75 session images and all 36 settings. The
[guarded baseline](../confusius-examples/README.md) remains frozen at `c2effa2`.
Input arrays, all native ConfUSIus matrices, and the global joint scoring mask are
**bitwise identical across revisions**. Thus the before/after scores below use
exactly the same full fixed-grid points (84.1588% of the original grid), not a
changed mask or a selected crop. This remains a natural 2D pair without biological
ground truth; NCC is a comparison measure, not proof of anatomical correctness.

| Full-image fit | Guarded voxreg NCC² | Current voxreg NCC² | ConfUSIus NCC² |
| --- | ---: | ---: | ---: |
| Default100, rigid | 0.549362 | 0.549362 | 0.799413 |
| Default100, affine | 0.641963 | 0.641963 | 0.844788 |
| Fixed400, rigid | 0.549362 | **0.837418** | 0.836226 |
| Fixed400, affine | 0.722159 | **0.871176** | 0.846557 |

The full-image rigid/affine fits now reach 400 updates instead of failing the
coverage cutoff at 101/184. Their final full-grid coverage is approximately
87.75% / 86.59%, with no manual crop required. The full-image affine singular
values are approximately 1.0265 / 0.9918, compared with its large transient
contraction at the 100-update cap. Overlays show materially better vessel alignment.
All 12 real-pair CPU/CUDA comparisons still pass the 0.001-pixel screen, with
matching update counts (worst observed difference 2.23e-7 pixels).

**The default100 budget still under-captures this displacement.** This patch does
not change iteration/learning-rate defaults, calibrate affine/3D early stopping,
guarantee partial-overlap capture, or establish a universal affine-quality claim.
Higher NCC than ConfUSIus is not a ground-truth accuracy result. Both fixed400
voxreg runs remain `max_iterations`, not certified convergence.

`results.json` retains the current full-image matrices/settings/scores, all
CPU/CUDA screens, input/source hashes and cross-revision consistency checks.
Full metric histories, cropped arms, arrays and figures remain under
`artifacts/overlap-policy/same-subject/`; the old baseline artifacts are untouched.
The native ITK rerun is under `artifacts/overlap-policy/linear-probe.*/research/linear/`.
The motion-correction bitwise record is `artifacts/overlap-policy/default-regression.json`.

Reproduce the real comparison from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python research/confusius-examples/compare_same_subject.py \
  --output artifacts/overlap-policy/reproduction
```

Inspect `fixed400.png` (full-image and cropped overlays), `defaults.png`, and
`differences.png`. Output dB representation, preprocessing, scoring and the native
optimizer differences are described in the original report. Neither ConfUSIus nor
historical measured results were modified.
