# Whole ALD001/ref20: SyN versus FireANTs

Runner: [`landemard.py`](landemard.py). Uses **all 11,310 native 1×145×80 float32
frames**, the original first-frame reference, and the same per-frame rigid→affine
seeds fitted on the original images. Neither backend receives pre-resampled data.
Dense CPU transform output is disabled; corrected original-power recordings are
retained for inspection.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python research/syn/landemard.py --output artifacts/syn-ref20/new-run
```

## Protocol

- RTX 5080 Laptop GPU, one CPU thread, native single resolution, 100 fixed-budget
  SyN outer updates. Convergence disabled for the measured budget.
- Shared dB fitting preprocessing, radius-2 CC, nominal learning rate 0.1,
  update/total smoothing sigma 1.5/0.5 voxels. These are comparable configurations,
  **not identical optimizers or objectives**: voxreg uses masked local CC,
  float64 geometry/state/statistics, compositive gradient descent and best usable
  halfway state; FireANTs uses its native float32/Adam implementation.
- Voxreg receives the **whole recording in one public call**, default automatic
  batch policy (32 for this SyN recording). FireANTs has no corresponding internal
  recording batch option: the runner calls its public native 2D API in batches of
  256. Batching is documented, not hidden as single-call parity.
- Backend warmed with two updates before one measured full-recording pass.
  CUDA synchronized around wall time. Preparation, transfers, fitting and final
  original-power bilinear resampling/download are included. Imports, dataset I/O,
  shared rigid/affine prealignment, QC and disk export are excluded and/or reported
  separately. One full pass is not a timing distribution or a universal speed claim.
- No complete dense fields are retained on CPU. FireANTs' public coordinate
  accessor stays on GPU and supplies final standard Torch interpolation, with
  native half-open image support and per-frame-minimum fill. The first reference
  is preserved exactly for both outputs. Voxreg additionally performs its sampled
  Jacobian and inverse-consistency diagnostics; output costs are not identical.
- Installed FireANTs uses the **Torch fallback**, not its unavailable fused
  interpolator. No vendor source patches or transplanted optimizer code.
- Report allocated/reserved GPU peaks. CPU `ru_maxrss` is the **whole-process
  high-water mark**, not an isolated per-backend CPU allocation comparison.
- Per-frame Pearson r is measured in dB on the intersection of both output-support
  masks, against the original reference. This is alignment QC, **not an anatomical
  oracle, invertibility proof or evidence that physiology is preserved**.

## Results

Completed the actual whole recording, not an extrapolation from the
1,024-frame [batch sweep](../performance/BATCH-SIZE.md). Machine-readable results:
[`landemard-results.json`](landemard-results.json).

| Backend | SyN end-to-end seconds | Frames/s | Allocated GPU peak | Median Pearson r (dB) |
|---|---:|---:|---:|---:|
| voxreg, automatic batch 32 | **758.861** (12 min 39 s) | 14.90 | 373.8 MiB | 0.984628 |
| FireANTs, batch 256 | **130.172** (2 min 10 s) | 86.89 | 702.1 MiB | 0.985450 |

Raw median correlation was **0.972432**; reference frame is excluded from all
medians. FireANTs was **5.830× faster** for this workflow, with similar correlation
QC. This does not prove anatomical equivalence or identical regularity guarantees.
All 11,309 fitted voxreg frames reached the fixed 100-update budget; one reference
frame was preserved. Native FireANTs update/precision/output-cost differences above
remain relevant; this is not a trajectory-parity benchmark.

Shared rigid/affine prealignment took **24.686 / 23.028 seconds**, outside both SyN
timings. Whole-process CPU high-water RSS was **5,397,651,456 bytes (5.03 GiB)**,
including both backends and QC—not an isolated per-backend peak comparison.

For voxreg, **755.748 s (99.59%)** was in `optimize`, including the final inverse and
regularity work, versus 2.019 s in final resampling/download, 0.819 s preparation
and 0.083 s uploads. Disabling dense export solves the output-memory problem;
**it does not solve the core SyN runtime gap**. Profiling the fitting, smoothing,
interpolation, CC and inversion work is the next evidence-gathering step; this run
does not establish their individual contributions.

Reloaded both full NIfTIs and verified all-frame finite values, original float32
dtype, shape, exact first reference, exact acquisition times and physical lattice
within 1e-8 mm. All 11,310 QC rows are finite and preserve original time labels.
Source hashes match the exact code used for the run. Local run directory:
`artifacts/syn-ref20/full-01/`; its parent contains the log and source snapshots.
The nine-frame, two-update interoperability check is retained separately in
`artifacts/syn-ref20/smoke-02/` and is not a performance result.

The runner writes `results.json`, `qc.csv`, `affine-seeds.npy`, `voxreg.nii` and
`fireants.nii` with ConfUSIus geometry/timing sidecars. Existing output directories
are refused. The raw recording is 524,784,000 uncompressed bytes (472,112,627 bytes
as cached gzip); exporting both float64 three-component fields would instead
require 6,297,408,000 bytes before assembly temporaries. They are not requested.

The earlier [ITK/FireANTs/native SyN controls](BENCHMARK.md) deliberately retain
their original dense-output costs, failures and provenance; they are not silently
replaced by this no-transform workflow. The whole-recording ITK/all-method matrix
has not been run here.
