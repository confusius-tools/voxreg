# ConfUSIus examples: time, alignment and score

**Bottom line:** ConfUSIus captures the between-session displacement better with
short runs. voxreg needs more steps for that pair. On the 120-frame motion-correction
example, both produce very similar alignment, and voxreg is substantially faster.
A GPU helps the batch, not the single small image.

This compares **public ConfUSIus 0.8.0** with voxreg's **b75fc85 engine**. No product
or ConfUSIus source was changed. The actual example loader returns **float64** for
both workloads; input and returned-image precision are retained. Do not mix these
times with historical float32-input benchmarks.

## Example01: same subject, two sessions

The original full fixed/moving images are used, with the example's dB conversion,
qform world and geometry centering. Rigid is the notebook's linear stage; affine
is an explicitly added linear-model comparison, not its B-spline refinement.

Times are **median warm seconds** over three calls. NCC² is independently scored
on the same original fixed-grid points for all 12 arms (85.4143% joint coverage).
Higher is better, up to 1; it is not a biological truth measurement.

| Model / stopping | CF CPU time | vox CPU time | vox GPU time | CF NCC² | vox NCC² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rigid, native stopping | 0.0815 | 0.0778 | 0.2139 | 0.79621 | 0.54749 |
| Affine, native stopping | 0.1389 | 0.0824 | 0.2228 | 0.83940 | 0.63892 |
| Rigid, fixed400 | 1.0753 | 0.2791 | 0.8007 | 0.82874 | 0.83083 |
| Affine, fixed400 | 1.2880 | 0.2959 | 0.8259 | 0.84098 | 0.86808 |

CPU/GPU voxreg scores differ only beyond the shown digits. Native stopping means
CF stops after 27 rigid / 36 affine updates; voxreg reaches its 100-update cap.
The short voxreg runs visibly leave substantial red/cyan vessel separation, so
their shorter CPU time is **not an equal-quality speedup**. At 400 steps, both
align much better; voxreg's affine score is higher on this common domain. Higher
NCC does not prove a more anatomically correct deformation.

Fixed400 disables early stopping for both tools. The CPU speed advantage at that
budget does **not** establish a universal speed advantage: CF's native stopping
needs far fewer updates. The GPU is slower than voxreg's CPU for this single
112×128 image. Defaults and optimizers were not tuned to equal quality.

Visuals: [`pair-overlays.png`](../../artifacts/confusius-examples/timing-b75fc85/pair-overlays.png).
Rows are native rigid, native affine, fixed400 rigid, fixed400 affine. Columns are
unregistered, CF, vox CPU, vox GPU. Red is fixed, cyan moving; gray indicates
alignment. Out-of-support borders are masked, not treated as image information.

The identity score here is 0.19640. Scores differ from the earlier 36-arm report
because this comparison uses a different joint mask (no cropped fitting arms),
not because historical scores were replaced.

## Example02: motion correction, frames220:340

Uses the **freshly loaded example values**, not the older float32 archive.
ConfUSIus retains the notebook's **learning_rate=1.0**, dB fitting and native
stopping. voxreg uses its default Adam rate, 100-update cap and window50 stopping.
Both reference the first frame. Quality excludes that reference, leaving 119
moving frames, and uses one joint mask per frame across all eight arms, including
fixed-step controls. Median common coverage is 98.3329%, minimum 97.3493%; no frame
is silently dropped. Before registration, median NCC² is **0.93994**.

| Public API / configuration | Cold call (s) | Median warm (s) | Median NCC² |
| --- | ---: | ---: | ---: |
| CF, 8 processes × 1 native thread | 9.574 | 7.469 | 0.960855 |
| vox CPU, 8 threads, batch32 | 1.466 | 1.095 | 0.961167 |
| vox GPU, default batch32 | 1.552 | 0.880 | 0.961167 |
| vox GPU, all120 frames in one batch | 0.818 | 0.236 | 0.961167 |

Warm speed ratios versus this CF call are approximately **6.8× CPU, 8.5× default
GPU batching, 31.6× full-window GPU batching**. These are this workload's public-API
results, not a universal GPU or registration-kernel claim. The cold and warm
batch32/batch120 results come from separate fresh interpreters.

CF's moving-frame iteration counts range10–40, median33; voxreg51–95, median79.
CF reports all120 completed, including a fit of the reference; voxreg copies its
reference exactly and reports119 converged plus one reference. This difference
in actual work and native API outputs is retained, not silently harmonized.

Inspected overlays at source frames296 and335, selected as worst and median
**initial** full-grid alignment, not by choosing favorable registered outcomes.
Both corrections visibly improve the original; differences between the tools are
subtle, consistent with the small score difference. The time raster is also
similar and more stable than the original, not evidence of perfect correction.

- [Motion overlays](../../artifacts/confusius-examples/timing-b75fc85/series-example-overlays.png)
- [Motion raster](../../artifacts/confusius-examples/timing-b75fc85/series-raster.png)

Motion figures use actual returned power images with the same display scaling.
Independent quality scoring samples dB source images through the returned poses;
it is not a direct comparison of the two APIs' differently normalized loss fields.

### Fixed100 control and retained precision failure

| Configuration | Median warm (s) | Median NCC² |
| --- | ---: | ---: |
| CF, 8 processes | 21.858 | 0.960856 |
| vox CPU, batch32 | 1.119 | 0.961168 |
| vox GPU, batch32 | 0.832 | 0.961168 |
| vox GPU, batch120 | 0.225 | 0.961168 |

These controls disable stopping, not change the example's learning rate.
**The fixed100 vox CPU/GPU comparison fails the unchanged 0.001-pixel maximum
pointwise agreement gate:** observed **0.00196507 px**, at source frame258
(offset38), for both GPU batch sizes. Tiny NCC differences do not excuse this
failure. The native-stopping series comparison passes, worst **4.75e-8 px**;
all four pair comparisons pass, worst **4.47e-8 px**. There is no synthetic truth
for these biological images, so no truth-accuracy pass is claimed.

[Fixed-step overlays](../../artifacts/confusius-examples/timing-b75fc85/series-fixed100-overlays.png)
are retained. No optimizer, default, test tolerance or numerical gate was changed
to address the precision failure in this comparison-only task.

## Unsupported portions

- Example01's **B-spline refinement** has no voxreg equivalent; not run or timed.
- Example03 is **single-slice-to-3D-atlas registration**, unsupported in voxreg.
  There is no honest head-to-head timing, image or metric for that workflow yet.
  No easier synthetic 3D task was substituted and labeled as the atlas example.

## Timing boundary and reproducibility

`benchmark_examples.py` launches **20 sequential fresh worker interpreters**.
Each receives resident canonical VoxelData, makes one cold public-API call and
three warm calls, and saves the last output. Repeated outputs are bitwise identical
within **all20 cases**, including statuses/updates, not just the scores. Loading,
explicit example preprocessing, decoding the motion table, scoring, plotting and
compression are outside timing. API fitting preprocessing, original-dtype final
resampling, CPU/host return, container/diagnostic construction and CF's process
scheduling/IPC are inside. CF's native motion dataframe/FD overhead is not stripped;
voxreg's output schema is different. This measures the public workflows.

CPU budget: CF pair8 SimpleITK threads, CF series8 processes with one native thread
each, vox CPU8 PyTorch threads. GPU calls have one host PyTorch thread. NumPy/BLAS
and inherited OpenMP limits are1. Explicit resource caps override the notebooks'
all-CPU defaults. Progress and plots are disabled. CUDA is synchronized after cold
calls and before/after warm calls; cold includes its first-call setup. Joblib's
own warm pool is reused within a CF worker; no pool is reused across cases.

Hardware: Core Ultra9 275HX laptop / RTX5080 Laptop, ordinary desktop activity.
Other compute workers were gone before timing; no desktop or external process was
stopped. All40 power snapshots show mains connected and performance profile.
No fixed clocks, thermal stabilization, affinity pinning, continuous idleness or
confidence-interval claim. Warm sample ranges and cold calls are retained in JSON.
The timing pass finished at00:03UTC on2026-10-26; new external benchmark workers
observed at01:03UTC started after it, not during measurement.

Reproduce with cached inputs only, from the repository root (the output must not
already exist):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run python research/confusius-examples/benchmark_examples.py \
  --output artifacts/confusius-examples/timing-reproduction --repeats 3
```

`benchmark-results.json` retains the bounded report: all sample times, options,
iteration/status summaries, input/source hashes, quality distributions and failed
precision-gate cases. Full per-frame records, output hashes, images and arrays are
under `artifacts/confusius-examples/timing-b75fc85/`. Old measured reports and
ConfUSIus remain untouched.

### Analysis-check provenance

The first analysis attempt rejected ~2e-10 raw-power sampling differences with an
8-output-ULP bound that failed to account for world/index coordinate arithmetic.
The same scale of residual occurred in **every** backend, including vox CPU/GPU,
not just reconstruction of CF's public motion table. A scale-aware check requires
raw-power error below1e-12 of the input peak; observed worst relative residual is
**2.38e-14** across both workloads. The prior dB absolute check1e-8 is unchanged.
The 0.001-pixel transform gate is a separate check and its failure remains visible.

Measurements were **not replaced or rerun** for this analysis correction. The
actual timed script is preserved as `timed-source.py`, its SHA256 is reported,
and the timed options/call/decoding functions are unchanged from that snapshot.
The current script records its snapshot and input manifest before measurement so
analysis can be rerun independently. Tests cover public motion-table pose recovery,
example options and missing-score accounting. Product-code qualification is reused;
this work adds no production changes.
