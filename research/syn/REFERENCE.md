# Single-resolution ANTsPy SyN accuracy reference

The benchmark gate for **all registration models** is **RMS ≤ 0.30
minimum-spacing voxels**, following an explicit operator decision. The largest
measured ANTsPy SyN reference error rounds upward to this one-decimal value.
It is an explicit benchmark policy, not an anatomical accuracy standard. The
synthetic deformation remains ground truth; ANTsPy supplies a reference error.
Rigid, affine, SyN and staged native workflows now share the same 0.30 gate.
Historical reports keep their recorded policies and verdicts.

## Reference settings

Only the existing isolated Python package is used: **ANTsPy/antspyx 0.6.3** in
`.venv-antspy` (Python 3.13), eight ITK threads. No external ANTs executable,
shell preset, rigid/affine fitting or multi-resolution registration is involved.

```python
result = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform="SyNOnly",
    initial_transform="Identity",
    reg_iterations=(100,),
    grad_step=0.1,
    flow_sigma=2.25,
    total_sigma=0.25,
    syn_metric="CC",
    syn_sampling=2,
    singleprecision=True,
    outprefix=output_prefix,
)
```

The captured library arguments verify **one level, shrink factor 1, image
smoothing 0**, dense radius-2 CC and `SyN[0.1,2.25,0.25]`. Despite the Python
parameter names, the SyN field arguments are **voxel-space variances**: 2.25/0.25
correspond to the residents' update/total sigmas **1.5/0.5 voxels**.

These settings are approximate matches, not identical algorithms. ANTsPy uses
its own optimizer, Gaussian kernels and boundaries, and float32 fitting.
**Its convergence remains enabled** (`[100,1e-7,8]`, at most 100 updates); the
voxreg/FireANTs benchmark disables early stopping and performs 100 updates.
The reference does not change voxreg's geometry, precision or production API.

## Inputs and scoring

The same deterministic float32 phantom pixels as the resident benchmarks:
24 smooth Gaussian blobs plus weak sinusoidal texture; z/y/x shapes
`(1,150,128)` and `(80,150,80)`, spacing `[0.2,0.1,0.1]` mm, axis-aligned grids.
Each 100-frame series has ten distinct image arrays in its declared pose cycle,
including the identical-image reference. Each distinct input is fitted once;
all 100 source frames are scored with their own declared deformation amplitude.
This is not 100 independent registration calls.

The public `ants.apply_transforms(..., compose=...)` exports the **complete
fixed→moving image pull**, including any identity initialization transform.
The independent benchmark oracle scores the exported map over every fixed-grid
voxel centre, including background, in float64:

`RMS = sqrt(mean(||estimated_pull - analytic_pull||²)) / minimum_active_spacing`.

No intensity-error proxy, fitted mask, selected-point score or ANTs output as
truth. The field composition/direction is checked against the public
`ants.apply_transforms_to_points` API. Input pixel hashes match the earlier
matched voxreg/FireANTs workers for the planar series and maximum-amplitude
volume pair.

| Input family | ANTsPy worst RMS (voxels) | Identity/no-op worst RMS |
| --- | ---: | ---: |
| Planar series | 0.227284927031305 | 0.694626404944423 |
| Volume series (includes the volume-pair deformation) | 0.282142985278761 | 0.508400226056131 |

The larger reference error is **0.282142985278761 → 0.30**. The operator chose
this common threshold for all registration models and prioritized runtime
profiling over further accuracy tuning.
The calibration artifact retains its initial hundredth-rounded proposal (0.29);
the checked-in policy records the subsequent choice of one-decimal rounding.
Both identity baselines remain above this threshold. This calibration does not
measure runtime or memory, establish anatomical correctness, or qualify every
registration configuration under the revised gate.

## Evidence

- Checked-in settings, per-pose errors, input/source hashes and measurement digest:
  [`examples/_benchmark_syn_reference.py`](../../examples/_benchmark_syn_reference.py).
  New benchmark runs embed this record in their policy and freeze the module
  with their other source files.
- Local immutable-fit evidence: `artifacts/antspy-matched-syn-calibration-01/`:
  `run.py`, `worker.py`, `finish.py`, `results.json`, `verification.json`, and
  per-pose effective arguments, transform files, hashes and composition checks.
- An outer 900-second command timeout killed the initial supervisor, not the
  active Python worker. Its completed export was recovered; no fit was restarted.
  Remaining poses ran serially, one at a time. The interruption is recorded in
  `results.json`.
- The earlier literal ANTsPy default and native CLI calibrations are separate
  experiments and **do not set this gate**.
