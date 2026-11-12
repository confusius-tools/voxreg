# SyN geometry, symmetry, inverse maintenance and convergence curves

## Scope

Source audit plus CPU diagnostics. The initial investigation did not change
production implementation, defaults or benchmark policy. A later 200-update
fixed-work benchmark policy supersedes the earlier native-stopping experiments;
see [BENCHMARK](../../BENCHMARK.md). Production registration remains unchanged. The directional-preconditioning proposal remains
rejected. Local references inspected: ITK `5859e27`, ANTs `73b37ce`, FireANTs
`f8a9cc2`, and installed ANTsPy 0.6.3. Source hashes are retained with the plots.
This is not an audit of every code path in the installed ANTs native binary.

## 1. Non-axis-aligned step normalization

For a physical vector `u` and native lattice `L`, voxreg measures
`norm(inv(L) @ u)`. The full matrix includes orientation and shear; planes use
their orthonormal physical basis. A single scalar normalizes the maximum over
each frame. This changes magnitude, not gradient direction.

ITK's [`ScaleUpdateField`](https://github.com/InsightSoftwareConsortium/ITK/blob/5859e27db6db62d3d60bfa2774b18d9a207431e0/Modules/Registration/RegistrationMethodsv4/include/itkSyNImageRegistrationMethod.hxx#L645)
literally measures `sqrt(sum((u[d] / spacing[d])**2))`. That function contains no
inverse direction matrix. For the same physical vector, these formulas match
canonical aligned axes, and orthogonal rotations with isotropic spacing, but
not general oblique anisotropic lattices. Do not infer identical vector-component
conventions or full registration trajectories from a scaling-function comparison.

A checked example uses spacing 0.2/0.1/0.1 mm, a 45-degree rotation in the first
two axes and physical vector (1, 0, 0). The full-lattice norm is 7.905694; the
literal spacing-only norm is 5. At rate 0.1, voxreg's normalized update has native
voxel norm 0.1, while applying the spacing-only rule to that same vector gives
0.158114. Public `make_field_grid`/`measure_norm` calls match independent NumPy
calculations for aligned anisotropic, oblique anisotropic and oblique isotropic
cases. Preserve full-lattice normalization rather than blindly copying the
spacing-only expression into voxreg's geometry contract.

## 2. Why swap image roles for derivatives?

ITK's NCC derivative machinery computes a moving-transform update in a given
metric call. Swapping image/transform roles supplies the companion update needed
for the other halfway field. This swaps the roles in the calculation, not image
intensities and not simply inversion of the final transform.

There is a more concrete difference than previously stated: ITK's
[NCC documentation](https://github.com/InsightSoftwareConsortium/ITK/blob/5859e27db6db62d3d60bfa2774b18d9a207431e0/Modules/Registration/Metricsv4/include/itkANTSNeighborhoodCorrelationImageToImageMetricv4.h#L40)
explicitly says its fast derivative assumes changes affect only the center of
its neighborhood window. The implementation uses center-image gradients and
local window statistics. voxreg differentiates the implemented joint halfway
NCC objective through all overlapping neighborhoods and both pulls. Spatial
image-gradient and boundary/numerical conventions also differ. ANTs explicitly
sets `DownsampleImagesForMetricDerivatives(true)` and `AverageMidPointGradients(false)`:
ITK pulls the images onto the virtual grid and uses identity displacement
transforms for that metric-derivative evaluation. This is not the same evaluation
path as autograd through our original-image pulls. Neither matching window radius
nor scalar step normalization establishes derivative equivalence.

Our joint objective produces both halfway gradients in one autograd call;
FireANTs likewise backpropagates one halfway-image loss into both warps. They do
not need a second swapped-role call to obtain those two gradients. This is a
different derivative implementation, not evidence that one halfway update was
forgotten.

## 3. Symmetry is not the same as per-iteration inverse consistency

Write our halfway pulls as `F: middle -> fixed` and `M: middle -> moving`.
They are both optimized every iteration. They are not, in general, a field and
its inverse: they have different destination spaces. The complete endpoint pulls
are `M compose inverse(F)` and `F compose inverse(M)` (plus the fixed affine seed).

ITK stores image-to-middle fields and their inverses. Its
[outer optimization loop](https://github.com/InsightSoftwareConsortium/ITK/blob/5859e27db6db62d3d60bfa2774b18d9a207431e0/Modules/Registration/RegistrationMethodsv4/include/itkSyNImageRegistrationMethod.hxx#L226)
composes and smooths each update, then:
1. Estimates the inverse using the previous inverse as a seed.
2. Inverts that inverse again to refine the corresponding forward field.
3. Repeats for the other image-to-middle field.

Thus both halves and both corresponding inverse fields are refreshed at every
outer iteration. Each inversion permits up to 20 iterations with residual-based
stopping. The directions differ from our stored pulls, so reproducing this
requires following the complete composition/derivative convention. In exact
arithmetic, inverting an inverse returns the original map; the correction matters
because these are discrete interpolated fields with approximate inverses. It
reduces inconsistency in the paired representation, which ITK then uses for the
next metric evaluation. Our direct halfway pulls do not need inverse fields for
that evaluation. Whether adopting this correction helps our workload enough to
justify the extra work has not been established by these comparisons.

voxreg instead fits both halfway pulls, then computes their inverses during
finalization (80 iterations each), constructs both endpoint maps and reports
Jacobian/inverse residual diagnostics. It has no ITK-style per-iteration paired
inverse correction. This is a substantive algorithmic distinction, not merely an
export implementation detail. Positive sampled Jacobians do not replace inverse
consistency or prove global diffeomorphism.

FireANTs is closer to voxreg on this point: its
[SyN loop](https://github.com/rohitrango/FireANTs/blob/f8a9cc259ab9113fc43bd99797000fa8fc34eb59/fireants/registration/syn.py#L294)
optimizes two halfway pulls with Adam without maintaining inverse pairs at each
iteration. Its output accessor later inverts the reverse halfway pull and
composes it with the forward pull. The
[inversion helper](https://github.com/rohitrango/FireANTs/blob/f8a9cc259ab9113fc43bd99797000fa8fc34eb59/fireants/utils/warputils.py#L135)
uses another optimization of an inverse-consistency penalty, initialized with
the negative displacement. This is not ITK's paired per-iteration inversion.
The inspected FireANTs SyN accessor does not implement inverse-output parameters.

## 4. Original ANTs benchmark settings and current fixed-work policy

At the original 100-update checkpoint, `benchmarks/_benchmark_register_antspy.py`
called ANTsPy `SyNOnly`, identity seed,
CC radius 2, `grad_step=0.1`, update/total variances 2.25/0.25, one level,
shrink 1, image smoothing 0 and single precision. Installed ANTsPy translates the
budget to convergence `[100,1e-7,8]`: at most 100 outer updates with convergence
enabled, not a forced count of 100. Observed native iteration counts are not
reported by that adapter. voxreg/FireANTs benchmark fitting forced 100 updates.
The current eager registration v4 benchmark instead requests 200 fixed updates on
all backends, with ANTs convergence window 201 and accuracy-phase count checks.
Its timing and memory phases do not enable verbose iteration logging.

ANTsPy 0.6.3 defaults to `grad_step=0.2`; FireANTs SyN defaults to
`optimizer_lr=0.1`. The current benchmark uses these native SyN rates (ANTsPy 0.2,
FireANTs/voxreg 0.1); earlier matched-rate investigations explicitly used 0.1.
Their numerical learning-rate units and optimizers are not equivalent. A current
fixed-200 ANTs-only accuracy recheck at step 0.2 reaches RMS 0.183072 with all 200
updates verified; see `artifacts/fixed-count-200-native-rate-pilot/`.

If **EachIteration** means inverse maintenance, yes: native ITK SyN performs it
each outer iteration. If it means a generic optimizer's learning-rate estimator,
that is different. ANTs' helper enables each-iteration estimation on its generic
optimizers, but its
[SyN branch](https://github.com/ANTsX/ANTs/blob/73b37cee7c569d55ab0fa7fa7842b01f908b6f15/Examples/itkantsRegistrationHelper.hxx#L2243)
sets SyN's learning rate directly and uses its own `ScaleUpdateField`. It does
not attach that generic optimizer to the SyN stage. The configured gradient step
is constant; the field-dependent normalization scalar is recomputed every update.
This is not a line search or a decaying learning-rate schedule.

## 5. Plots and what they establish

Open `artifacts/syn-semantics-curves-20261112-01/index.html`. PNG/SVG figures,
every-update CSV/JSON, scripts, input hashes and source-audit hashes are retained
beside it. Raw metric histories are unsmoothed. Because the objective is negative
mean squared local NCC, the plots use `1 + midpoint_loss` on a logarithmic axis:
zero would mean perfect metric score, not necessarily correct correspondence.
The running minimum is best-so-far observed loss, not a convergence test.

- `learning-rates.png`: rates 0.1/0.2/0.4/0.8 through 300 updates, raw and best
  curves. Rate 0.1 continues improving beyond 100. Larger rates show increasingly
  oscillatory raw curves; best-so-far plots alone hide that behavior.
- `update-budgets.png`: rate 0.1 at limits 25/100/300, plus exact finalized truth
  RMS 0.400014/0.306434/0.184296. Shared metric prefixes are bitwise identical;
  increasing the limit does not change the earlier trajectory.
- `quality-and-residuals.png`: finalized truth error and two-sided inverse residual
  at 100 and 300. At 300, RMS is 0.184296/0.172250/0.176243/0.187123 for the four
  rates, but rate 0.8's inverse residual is about 0.124 native voxels versus 0.010
  at rate 0.1. Accuracy and inverse consistency are distinct observations.

All these runs use the frozen 3D pair, CPU and two Torch threads, with early
stopping disabled. Every outer trial is retained; the earlier final selected-state
re-evaluation is not appended to raw curves. The new 300-update runs' first 100
updates match the prior learning-rate sweep at all of its sampled points.
All runs report `max_iterations`, full support and positive final sampled
Jacobians. They have not been demonstrated to mathematically converge by 300.
Held-out learning-rate results remain mixed; these plots do not justify a new
default, universal convergence, CUDA performance or anatomical-quality claim.

## 6. Completed cross-backend learning-rate/update comparison

All 16 new native cases complete without harness errors; adding the eight
previous voxreg points gives 24 configurations, with 12 passing and 12 failing
the unchanged truth gate. All share the same verified input hash. ANTs retains
native convergence and uses eight ITK threads, matching its original benchmark;
FireANTs and reused voxreg diagnostics use two Torch threads. These are accuracy
comparisons, not timing rankings or equivalent numerical learning-rate units.

| Rate | voxreg 100 | voxreg 300 | FireANTs 100 | FireANTs 300 | ANTsPy limit 100 | ANTsPy limit 300 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.306434 | 0.184296 | 0.358709 | 0.267483 | 0.282613 | 0.153358 |
| 0.2 | 0.299970 | 0.172250 | 0.348453 | 0.261007 | 0.337221 | 0.337221 |
| 0.4 | 0.296346 | 0.176243 | 0.342229 | 0.257765 | 0.408977 | 0.408977 |
| 0.8 | 0.331506 | 0.187123 | 0.336176 | 0.257813 | 0.421117 | 0.421117 |

Values are full-grid truth RMS in minimum-spacing voxels. ANTs performs 100/300
updates at rate 0.1, but stops at **49/15/12** updates at rates 0.2/0.4/0.8 under
both ceilings. The final native convergence values at the larger rates are
negative and satisfy its signed-trend stopping test; the transforms still fail
the truth gate. A larger ceiling cannot help when the same native criterion
stops the run first. This does not establish what those rates would do with
convergence disabled. voxreg/FireANTs execute the full configured outer counts.
All eight native budget pairs have exactly matching metric prefixes.

At rate 0.1, 300 improves all three backends and passes this pair's truth gate.
Larger rates modestly improve FireANTs here, are non-monotonic for voxreg, and
worsen ANTs under native stopping. These findings do not justify changing a
universal default or claiming all 3D recordings/CUDA cases are qualified.

Open `artifacts/syn-semantics-curves-20261112-01/backend-comparison-shared-axis.html`
for the full report. `backend-quality-shared-y.png` uses the same truth-error axis
for both ceilings; `backend-metrics.png` separates native objectives by backend.
Metric heights are not cross-backend accuracy scores: derivative, window and
numerical conventions differ. The best-so-far curves are plotting summaries,
not a claim that every backend returns that state. JSON/CSV, every native log,
source hashes and validation checks remain alongside the figures. An interrupted
two-thread ANTs pilot is retained separately and excluded.

Counts refer to outer image-registration updates: ITK's inner inversion iterations
and FireANTs' output-inversion optimization are additional work. FireANTs passes
its outer schedule to its output-inversion helper, so this sweep changes both
budgets. Native progress descriptions verify that it also executes 100/300
inverse-optimization updates in these runs. voxreg keeps its final inversion
budget at 80. Therefore these are native-configuration comparisons, not a pure
ablation of fitting iterations or equal computational work. No speedup or
mathematical-convergence claim follows from these diagnostic runs.
