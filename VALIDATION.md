# Initial rigid validation

Status: **numerical and real-data quality checks pass; isolated ITK/FireANTs
performance/memory comparisons are now available** in [BENCHMARK.md](BENCHMARK.md).
Throughput measurements cover the documented pair/series controls, not general
registration accuracy or performance.

## Real ConfUSIus example 01: same-subject session pair

The [real-example report](research/confusius-examples/README.md) checks the cached
rat75 slice32 angiographies from 20220523/24, with the example's dB/qform
preprocessing, against public ConfUSIus 0.8.0. This is a new natural **2D pair**,
not natural 3D validation. No product or ConfUSIus code was changed.

**Default full-image alignment is not comparable to the reference on this pair.**
On a single common full-grid support mask (84.1588%), voxreg/default NCC² is
0.54936 rigid / 0.64196 affine versus ConfUSIus 0.79941 / 0.84479. Raising the cap
to 400 alone does not solve it: the unchanged 90% overlap guard stops rigid at
101 updates and affine at 184; reference-like poses retain only about 87–88%
original fixed-grid support. At 100 updates affine has a large transient
principal contraction (~0.874), not a trustworthy anatomical scale estimate.

A separately labeled, predeclared central fixed crop (16 pixels per edge, moving
unchanged) plus fixed400 reaches NCC² 0.83738 rigid / 0.86356 affine, versus
ConfUSIus's same-crop 0.81476 / 0.84523. Scores still use the original full grid and
the same joint mask, not crop-only scoring. Vessel overlays/difference images show
substantial improvement, but the crop is a caller-side diagnostic, **not a fix to
the original default example**. Both voxreg fits remain at the iteration limit.
Biological truth is unavailable; mapping disagreement with ConfUSIus is not a truth
error, and higher NCC alone does not establish correct tissue correspondence.

All 12 CPU/CUDA comparisons pass the unchanged 0.001-pixel maximum pointwise gate
(worst 2.23e-7 px), with matching counts/statuses. All 36 float64 backend outputs
match independent SciPy64 interpolation on interior support to <=6.74e-13 dB.
The report retains matrices, scores, support, model scales, complete settings,
file/source hashes and visual artifacts. Script self-checks pass; the production
sources are unchanged from the 352-test linear-engine qualification below.

Example 02 retains its earlier bounded 120-frame evidence, not a new full-recording
run. Example 03 remains unsupported slice-to-volume; the B-spline part of example01
was not attempted. The next general-registration gate is capture/fitting-domain/
partial-overlap policy, before claiming drop-in example parity or adding models.

## Translation/rigid/affine shared engine

The current suite passes **352 tests**, including 44 new linear-model checks.
The sole fitting entry point is `register(..., transform="translation" | "rigid" |
"affine")`; `register_rigid` was removed rather than retained as a redundant
pre-release wrapper. Existing live callers/tests were migrated. Default rigid
sampling, optimization and output conventions remain unchanged.

Translation and affine share the original preprocessing, squared NCC, masked
per-frame Adam/convergence, progress, fixed-grid output and original-dtype final
interpolation. Affine seeds optimize the complete centered linear map, not a
frozen prefix. Seed validation preserves scale/shear and accepts negative
orientation when explicit, but disallows numerical singularity. Runtime candidates
must retain seed orientation and normalized |determinant| > 1e-8; failures mark
only that frame invalid_transform and retain its best valid full matrix.

The new suite checks 2D/3D parameter counts and physical scales, native ITK point
oracles, finite-difference gradients, double SciPy pulls, non-geometric pivots,
oblique/sheared/strided/cropped grids, both sampling kernels, per-frame/CPU/CUDA
batch independence, reference identity, explicit seeded affine refinement after
large displacement, reflection seeds, malformed/singular/incompatible matrices,
per-frame affine failure isolation, and genuine low-contrast float64 output.
Affine diagnostics expose actual linear coefficients and centered mm translations.

The independent [linear-model qualification](research/linear/README.md) runs all
three models on 2D/3D continuous-field phantoms with unequal orthogonal grids.
Native ITK uses double images/GD/physical-shift scales; voxreg uses its qualified
mixed precision/Adam/RMS scales. All run fixed160 updates; timing is not measured.
SciPy64 joint-support NCC² is >=0.9990019. All 18 truth checks pass unchanged
RMS <0.25 px (worst 0.06512 px, ITK 2D affine); all six CPU/CUDA maximum pointwise
comparisons pass <0.001 px (worst 0.000388 px, 3D rigid).

Four real120-frame fixed/convergence CPU/CUDA controls with default rigid fitting
remain bitwise equal to every stored field in the frozen AC benchmark arrays.
Historical time/memory reports and source snapshots were not rewritten. Ruff,
format, lock, source/wheel builds and isolated wheel checks pass. New affine
performance, natural independent-session data, affine early-stop calibration,
regularization, pyramids and non-coplanar/slice-to-volume fitting remain unqualified.

## Initialization/progress ergonomics (0df4bb8)

At this boundary, the suite passed **308 tests**. Matrix-valued `initialization` is tested
against `initial_transform` for NumPy arrays, nested lists, labeled matrices and
batched matrices. The napari moving-to-fixed example is inverted into the existing
physical fixed-to-moving pull convention. Scale/shear rejection and mutual
exclusion are preserved; no new transform family or frozen affine chain is added.

Progress defaults to `"auto"`: spatial pairs and single-frame recordings are quiet;
multiple-frame recordings retain the Rich frame bar. Explicit True/False/None and
callbacks retain their semantics. Slice/volume pair checks verify unchanged numeric
results and explicit opt-in rendering; callbacks still receive (1, 1) for a pair.
The existing CPU/CUDA progress synchronization and numerical checks pass. Ruff,
format, lock, build and isolated wheel smoke checks pass.

## Automatic geometry/moments initialization (aae4918)

At this boundary, the suite passed **288 tests**, including 26 automatic-centering tests.
All changes are in voxreg; ConfUSIus was used only as an API/reference comparison.
Both `center_geometry` and `center_moments` match installed ITK 5.4.7 centering on
physical centers, translation direction and fixed rotation pivot. Geometry uses
grid centers; moments uses the original nonnegative intensities, before dB scaling
or NCC conditioning. Negative/zero-mass input is rejected for moments with advice
to pass unscaled data and use `intensity_scaling`, or select geometry/explicit seeds.
Other manual nonlinear transforms cannot be reliably detected and are documented
as unsuitable for preserving original-intensity moments.

CPU/CUDA checks cover a 20/-12/8 mm offset with no identity overlap, fitting in
linear and dB modes, per-frame initialization and implicit reference identity.
Finite-difference gradients and original-float64 SciPy pulls verify off-geometric
pivots on same-grid oblique/strided slices, unequal-grid slices and volumes.
Physical scales are compared against dense unweighted fixed-grid RMS lever arms
about the intensity centroid. Output retains original intensity units/dtypes,
and fitting-only scaling never mutates the raw caller arrays.

On a rotated independent/cropped 3D control, the fixed160-update precision control
retains the unchanged **0.25-pixel RMS truth** and **0.001-pixel maximum pointwise
CPU/CUDA agreement** gates. The observed maximum CPU/CUDA difference is approximately
0.000061 minimum-spacing pixels.

**Default 3D convergence remains uncalibrated.** On that same small control,
window50/minimum1e-6 stops both devices at 70 updates: truth RMS is approximately
0.00381 CPU / 0.00455 CUDA pixels, but their maximum pointwise difference is
**0.00248 pixels**, exceeding the 0.001-pixel precision screen. Fixed160 improves
truth RMS to approximately 0.000032 / 0.000029 pixels and passes agreement;
fixed400 reduces the maximum difference below 0.00000046 pixels. The default
threshold and error gates were not retuned or relaxed. This result bounds the
qualification: fixed-budget numerical behavior passes, not universal cross-device
agreement under default early stopping. Raw observations are retained in
`artifacts/auto-initialization/precision-probe.json` for later 3D calibration.

The existing real-series fixed/convergence controls are checked against the frozen
AC outputs with automatic initialization omitted; no historical benchmark result
is rewritten as an automatic-centering measurement. Ruff, format, lock, source/wheel
builds and isolated wheel smoke checks pass. No masks, pyramids, rotation search,
non-coplanar plane alignment or general partial-overlap policy were added.

## Explicit initial transforms (2242bc5)

At this boundary, the suite passed **262 tests**, including 33 new initialization tests.
`initial_transform` follows ITK's optimizable InitialTransform / InPlaceOff model,
not its frozen MovingInitialTransform chain. A real ITK 5.4.7 zero-update v4
registration is the direction/copy oracle. Initialization takes a full world
z/y/x rigid pull matrix; it is copied and the full refined transform is returned.
No ITK runtime dependency or extra final interpolation is introduced.

Checks cover shared/per-frame seeds, labeled axis/time/units/convention validation,
nonrigid/malformed inputs, immutable caller arrays, slice-plane restrictions,
reference identity, fresh optimizer state, and failure fallback to the supplied
pose rather than identity. Matrix-to-parameter round trips include zero, tiny
angles, all principal axes and an arbitrary axis at/near pi, including billion-mm
common world origins under the unchanged pointwise gate. Float32 rotational
roundoff normalization preserves the mapped fixed center. Original float64 seeded
resampling agrees with an independent SciPy pull.

An oblique anisotropic independent-grid phantom with a known approximately
18-degree rotation and center displacement `[20, -12, 8]` mm has no initial overlap
under identity. A perturbed supplied pose enables CPU/CUDA refinement to below the
unchanged **0.25 minimum-spacing pixel RMS truth screen**. Three-dimensional
per-frame batching and CPU/CUDA comparison retain the unchanged **0.001-pixel
maximum pointwise gate**, with identical update counts. This is seeded local
recovery, not a new uninitialized capture-range or general 3D convergence claim.

On the frozen real 120-frame input, fresh CPU32-thread8/CUDA120 processes under
both fixed100 and default convergence policies produce **bitwise-identical every
stored output field** to the `e263aa0` AC measurements with initial_transform
omitted. Explicit identity versus omission is also bitwise checked on CPU/CUDA.
Raw checks are under ignored `artifacts/initial-transforms/`. The older benchmark
matrices are not retimed or rewritten as seeded-registration measurements.

Ruff/format/lock/build and wheel-import checks pass. At this boundary automatic initialization was unsupported; geometry/moments modes
are added by the section above. Initialization search, fixed/moving pre-transform
chains, non-coplanar slice alignment and a general partial-overlap policy remain
unsupported.

## Standardized pair/series benchmark (e263aa0)

At this benchmark boundary, the suite passed **229 tests**, including 18 new benchmark-protocol tests.
The existing runner now measures 3D pairs and motion correction with shared timing,
memory, provenance and independent SciPy quality boundaries. Five fixed-budget
3D controls (64³/128³ same/independent grids and a real-template known-pose control)
pass the unchanged 0.25 minimum-spacing pixel RMS truth screen for ITK, voxreg and
FireANTs. The worst fixed-budget RMS error is 0.077451 pixels (rounded upward).

Backend-native convergence is not equivalent: FireANTs stops at 19 updates on the
64³ independent and template controls, with approximately 0.483/0.540 pixel RMS
error, failing that screen. This is recorded rather than changing tolerances or
claiming equal-quality early-stop speedups. The real-template control contains
real intensity structure with synthetic known pose, not natural session motion.

After the operator reported plugging in the laptop, the entire same suite was
rerun using the same frozen input bytes. All 196 before/after worker snapshots
verify AC online and the performance platform profile. Earlier power-uncontrolled
runs are retained separately, not mixed into the new report. Clocks and thermals
remain uncontrolled. The final timed output and separate memory-pass output are
bitwise identical for all 49 configurations, including images, masks, transforms,
updates and statuses. Existing solver tests/precision gates are unchanged.

Ruff/format/lock/build and wheel smoke checks pass. See the
[shared report](research/registration/README.md) and source-hashed
[results](research/registration/results.json). Natural 3D session pairs,
process-parallel ITK under the new boundary and controlled thermal statistics
remain outside this qualification.

## Independent fixed/moving grids and initial 3D rigid support (cc0b4de)

At this geometry boundary, the suite passed **211 tests**, including 31 new grid/volume tests on
CPU/CUDA. Ruff, formatting and lock checks pass; source/wheel builds and a
wheel-import smoke test exercising unequal-grid 3D registration pass.
Unequal-grid identity and rigid pulls agree with independent SciPy
interpolation on anisotropic, oblique, sheared, cropped/strided/reversed grids.
Tests also cover physical rotation-vector handedness, float64 gradients at zero
rotation, large origins, elongated-grid scaling, dtype/rounding, native output
geometry, source time/auxiliary coordinates, reference identity and failure fill.

On one analytically generated 3D pair with independent physical grids, RMS truth
error is 0.010373 minimum-spacing pixels on both devices (<0.25 gate). Maximum
CPU/CUDA pointwise disagreement is below 1.07e-6 pixels (<0.001 unchanged gate).
Independent SciPy NCC² on common support improves from 0.977054935 at identity
to 0.995404515. Both devices converge after 81 updates of a 180-update budget.
This is small-motion synthetic recovery, not biological validation or a broad
3D convergence calibration.

Fresh-process parent/current checks on the float32 120-frame rat75 input are
bitwise identical for registered intensities, masks, transforms, objectives,
update counts, statuses and motion diagnostics on CPU32/CUDA120. No old timing
report is reused as a 3D result. See [geometry evidence](research/geometry/README.md)
and its source-hashed [results](research/geometry/results.json).

Initial transforms, center/plane initialization and a general partial-overlap
policy were absent at this boundary. Explicit initial transforms are added by the
later section above; automatic center/plane initialization and general partial
overlap remain unsupported. Non-coplanar slices and slice-to-volume fitting are
still rejected. The later benchmark section supersedes the performance omission
only for its declared controls.

## Temporary ConfUSIus VoxelData integration

The duplicated voxel-to-world classes have been removed. ConfUSIus is now a
runtime dependency (`confusius>=0.8.0`, installed from the released package). Direct ConfUSIus
constructor inputs and voxreg's registered/valid outputs are checked with both
libraries' validators, preserving geometry, coordinate metadata and reference
identity. Native voxel indices now follow ConfUSIus's integer-coordinate contract.

The user's ALD001 ref24 run-2 NIfTI was loaded with `cf.load(...).compute()` for a
four-frame, 100-iteration smoke check on CPU and CUDA, without any conversion.
Both output arrays validate as ConfUSIus VoxelData; the input and reference are
unchanged. This is a geometry/integration check, not a full-recording quality gate.

`uv sync` now includes ConfUSIus's larger dependency tree; the tested environment
uses ConfUSIus 0.8.0 and xarray 2026.9.0. Numerical synthetic, dtype and registration
gates were unchanged; **180 tests passed at that boundary**. Batch progress tests cover CPU/CUDA,
partial/oversized batches, failed/reference frames, default Rich rendering, silent
modes, callback overrides, errors and display cleanup. Outputs/diagnostics remain
identical and CUDA synchronization counts match quiet calls. Timing/profiling
harnesses explicitly disable the default display.
The previous timing/memory reports below describe the pre-integration environment
and have not been retimed with this dependency stack.

## VoxelData API cleanup

Construction and general validation now use ConfUSIus directly in every tracked
example, research harness and test fixture. Voxreg's constructor and public
validator exports are removed; the same solver-specific geometry checks remain
private and automatic. Tests cover otherwise-valid ConfUSIus Dask/pose/non-mm
inputs being rejected by the narrower solver boundary, plus the existing native
coordinate, stale-index and affine checks. NIfTI example tests include singleton
windows and declared acquisition duration for ConfUSIus's timing requirements.

The focused pre-refactor baseline passed 33 tests. After migration, the full suite
passes 167 tests. On the existing 120-frame real window, transforms, registered
output bytes, iteration counts and statuses match the pre-cleanup results exactly
on CPU (batch32) and CUDA (batch120); outputs pass ConfUSIus validation. This is
behavior-preservation proof, not a new performance benchmark or full-recording
comparison. Local logs are under `artifacts/api-cleanup/`.

## Calibrated convergence and terminal checks

Current defaults are window 50 / minimum 1e-6. Four 120-frame windows (three rat75
windows, including native float64, and ALD001) were compared against fixed 100/400
controls on CPU/CUDA. Worst full-grid RMS mapping disagreement with either control
was below 0.007 minimum-spacing pixels; worst NCC² loss was below 7e-6. Median updates
fell to 78–79, with 475/476 moving frames converged. These are regression comparisons,
not biological truth. The prior 167-test suite passed with the candidate threshold
injected; the final expanded suite passes 180 without changing numerical gates.

Whole-batch termination is separately proven identical to continuing the frozen
loop on all four windows/devices and on mixed-failure synthetic cases. CPU polls
once per eligible iteration; CUDA polls one batch-wide scalar every ten iterations
after the first full window (at most five reads at default settings). The polls
synchronize CUDA and can add overhead when no batch finishes early. Fixed-budget
controls introduce no polls. See [CALIBRATION.md](research/convergence/CALIBRATION.md)
for fresh-process timings, limitations, input/source hashes and reproducible CLIs.

## Initial ITK-style convergence (historical, c46c07c)

The PyTorch monitor is checked on CPU/CUDA against stored C++ ITK reference
values for windows 2/10/20/50, including constant, zero, improving, worsening,
oscillating, plateau and random losses. Absolute tolerance is 2e-15 with relative
1e-12; startup sentinel values match exactly. Frozen histories, per-frame
independence, pre-update stopping and fixed-budget controls are covered. The
existing synthetic recovery and CPU/CUDA mapping tolerances are unchanged.
The existing freeze-stress test now explicitly uses 1e-6 to exercise convergence
within its 400-update budget; it does not change a transform-error tolerance.

Initial defaults followed ITK gradient descent (50/1e-8). On the existing 120-frame real
window, transforms and registered output bytes are identical to the previous
policy on each device. CPU/CUDA median/minimum common-support NCC differ by
less than 4e-12. Only six moving frames
certify convergence within 100 updates. Reporting that limitation is preferable
to relaxing the threshold silently. Separate CPU/CUDA timings and the reference
fixture provenance are in [research/convergence/README.md](research/convergence/README.md).
Historical fixed-budget benchmark/profiling commands now explicitly disable the
new monitor with a window of 101, preserving their 100-update protocol.

## Historical benchmark environment and reproducibility

- Python 3.14.8, NumPy 2.5.4, xarray 2026.9.0.
- PyTorch 2.14.1+cu130, RTX 5080 Laptop GPU, Linux.
- Reference: ConfUSIus 0.9.0.dev0 and ITK 5.4.7, in a separate interpreter.
- Voxreg: float32 fitting images/interpolation; float64 geometry/state/NCC and
  exported matrices. Final interpolation/output retain original image precision
  and dtype, with documented rounding/range checks for integers.
- Input: Cybis Pereira 2026 rawdata, rat75 / 20220523 / slice32, frames 220:340,
  shape `(120, 1, 112, 128)`.
- Source NIfTI SHA-256:
  `f2d0a5e4daa10afc979b47d926af4e1ba3b2368c1384e32944ef9d3d8687e9b3`.
- Original irregular time labels retained, approximately 88.4–136.0 s.

The ignored `artifacts/notebook/` directory contains frozen reference inputs,
source-code fingerprints, resolved ITK settings, matrices, original-power output,
framewise-displacement references, and machine-readable comparison results. The
initial comparison used float32 voxreg samples against float64 reference samples.
The current example retains decoded source dtype and checks exact reference-input
values plus source hash; the neutral backend benchmark still deliberately uses
common float32 input. No optimizer parameters were tuned against this recording.

## Current dtype/precision validation

[DTYPE.md](research/performance/DTYPE.md) records the new input/output contract,
float32 qualification failures, fresh time/memory benchmark and package review.
All supported output dtypes are checked against independent fractional sampling,
with exact reference identity and integer rounding/clipping tests. Float64 detail
below float32 resolution survives final output, including low-contrast fitting.
The optional NIfTI example also round-trips float64 and signed/unsigned integers.

Full float32 fitting and metric-only float32 each fail the unchanged synthetic
CPU/CUDA agreement gate. The production mixed policy is retained; no precision
selector, compiler requirement or metric framework was added. Additional slice32
and slice37 windows pass batch gates, including native decoded float64 output.
Real CPU/CUDA mapping disagreements are reported separately, not claimed as truth
errors or forced below the synthetic tolerance.

## Automated tests

```bash
uv run --extra examples pytest -q
# 352 passed, including translation/rigid/affine, initialization, 3D and CUDA.

uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

Coverage of the behavior, not just shapes:

- Native-coordinate offsets, reverse/strided crops, oblique world affines and
  preservation of lazy geometry through slicing and time reduction.
- Rejection of missing/stale indexes, invalid affines/units/coordinates, invalid
  shapes/options/timing, nonfinite data, constants and unsupported plane pairings.
- Bilinear sampling against independent SciPy interpolation and homogeneous
  mapping/inverse/rotation-center checks.
- CPU float64 gradient checks and independent masked NCC calculations.
- Analytic asymmetric phantom recovery for rotations up to two degrees and
  translations up to two minimum-spacing pixels, with no fitting resampling
  used to generate the ground truth.
- Opposing motions, chunk composition, independent failure handling and frozen
  converged optimizer state.
- CPU/CUDA agreement and NIfTI geometry/intensity/time round trips.

The synthetic recovery gate is RMS physical mapping error below **0.25 minimum
in-plane pixel spacings** over the declared interior region. The worst measured
error across six orientation/stride/intensity combinations is **0.00424 pixels**
(rounded upward). See `synthetic_checks.json` for individual cases.

## Real recording: quality

Both sets of physical matrices are evaluated by an independent SciPy bilinear
sampler applied to the **same dB input arrays**. Each frame uses the intersection
of both methods' valid sample-center support. These are common scoring values,
not a comparison of unlike internal optimizer reports. Summaries include the
reference frame.

| Measure | Unregistered | voxreg | ITK reference |
|---|---:|---:|---:|
| Median squared NCC | 0.94085 | 0.96119 | 0.96087 |
| Minimum squared NCC | 0.87026 | 0.91758 | 0.91746 |

- Median paired voxreg-minus-ITK squared NCC: **+0.000332**.
- Common valid support: median **98.33%**, minimum **97.35%**.
- RMS mapping disagreement with ITK over all fixed-grid points:
  median **0.00726 mm**, 95th percentile **0.01284 mm**, maximum **0.01616 mm**.
  With minimum spacing 0.09856 mm, these are approximately **0.074, 0.130 and
  0.164 pixels**, respectively.

These results show comparable alignment on this recording, not recovery of a
known biological ground truth or universal superiority over ITK. A higher NCC
alone does not prove that genuine fUSI signal changes are preserved.

### Batch stability

A full-recording CUDA comparison of batch sizes **1 versus 32**, with identical
100-step settings, measured worst per-frame grid RMS disagreement of
**6.85e-10 mm**, or **6.95e-9 minimum-spacing pixels** (rounded upward). The
functional check used a 0.005-pixel tolerance. See `batch_invariance.json` and
`serial_transforms.npz`.

During development, float32 geometry/optimizer state produced approximately
0.008-pixel disagreement on this check, and float32 NCC quantization destabilized
best-iterate selection near correlation one. The current float64 geometry/state
and NCC policy removes those observed instabilities. This precision policy needs
profiling before any throughput claim.

### Inspection artifacts

- `motion.png`: angle, physical center displacement, FD and optimizer summaries.
- `trace.png`: the same preselected maximum-standard-deviation voxel before/after.
- `raster.png`: the same native column and dB color limits before/after and for ITK.
- `comparison.png`: the frame with greatest geometric disagreement, including a
  power difference image on common valid support.
- `diagnostics.csv`, `voxreg_result.npz`, `summary.json`: labeled diagnostics,
  matrices, validity and framewise quality comparisons.
- `input.nii.gz`, `voxreg_registered.nii.gz`, `voxreg_valid.nii.gz`,
  `voxreg_minus_fixed.nii.gz`, `itk_registered.nii.gz`,
  `voxreg_minus_itk.nii.gz`: napari-readable spatially placed recordings.

NIfTI intensity/mask/time/affine round trips were checked on the actual outputs.
The FD calculation was also checked against the frozen reference: it evaluates
`norm(T[t+1] @ p - T[t] @ p)` on fixed-grid points, associates the statistic with
time t, and ends with zero. It is not a sum of Euler-angle derivatives.

## Historical open gates at the original slice milestone

This list records the original milestone, not current support. Current model,
convergence and real-example gates are described above.

1. **Convergence calibration:** all 119 non-reference frames reached the configured
   100-step limit. None failed the overlap/metric guards, and quality matches the
   reference closely, but the strict convergence rule did not certify them.
   Their status is `max_iterations`, not `converged`.
2. **Runtime comparison:** the original contested timings were discarded. New
   isolated ITK/FireANTs comparisons include single-thread, eight-thread and
   eight-process ITK controls, CPU/CUDA FireANTs batching, separate timing/memory
   passes, and common quality scoring. The production scalar-affine sampler
   improves full-batch CUDA from 1.18 to 0.23 s and eight-thread CPU from 3.09 to
   1.98 s in matched runs. CPU remains slower than native float32 FireANTs. See [BENCHMARK.md](BENCHMARK.md). These do not establish a speedup
   over ConfUSIus's full public pipeline, which has not been cleanly retimed.
   The notebook/reference exporters still default to quality-only mode.
3. **Profiling/optimization:** the production scalar-affine sampler removes dense
   grid GEMMs while retaining float64 geometry/state/NCC. All 144 tests pass,
   including expanded sheared geometry, native support tolerance and CUDA batch
   checks. Real recording batch sizes 1/32/120 differ by at most 6.95e-9 pixels RMS
   on CUDA; same-device mapping change from baseline is below 6.70e-6 pixels RMS.
   Fresh-process full-batch CUDA peak allocation falls from 212.5 to 189.2 MiB,
   with process VRAM 514 to 478 MiB. See [BENCHMARK.md](BENCHMARK.md) and its tracked
   snapshot. Further fusion/compilation and broader-data validation remain open.
4. **Broader scope:** no full 3D, multi-pose, multiresolution, arbitrary-mask,
   differing-grid, affine or deformable validation has been attempted. These
   inputs are deliberately unsupported rather than silently approximated.

The original milestone deferred model extension pending rigid qualification.
Current model support and its remaining qualification limits are described above.
