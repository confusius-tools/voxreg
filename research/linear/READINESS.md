# ConfUSIus package-readiness check

**Status: linear contracts pass; real-workflow qualification has explicit limits.**
This checks the usable VoxelData API, not a general capture-range study. No runtime
code, optimizer defaults, public exports or numerical gates changed. SyN is not
implemented by this work. Offline sdist/wheel builds pass; wheel Python sources
match the working tree byte-for-byte, and six CPU smoke cases import the actual
wheel (all three models, 2D/3D), not the editable checkout.

## Linear model coverage

Baseline: 557 passing tests. After extending existing contracts: **585 passing
tests**, including CPU/CUDA. New coverage is 28 parameterized cases, not 28 new
test functions. Ruff/format and whitespace checks pass.

| Contract | Coverage |
| --- | --- |
| Known-pose translation / rigid / affine recovery | 2D and 3D; 0.25 native-pixel RMS truth and 0.001 pixel maximum CPU/CUDA gates |
| Centered seed / native ITK point agreement | All three models, 2D/3D |
| Real optimizer → output contract | All three models × 2D/3D × CPU/CUDA × float64/uint16 |
| Output semantics | Fixed grid/index, source name/units/attributes, original dtype, independent SciPy interpolation/rounding/mask, neither input mutated |
| 3D batching | All three models; known truth, chunk independence, time/acquisition coordinates, CPU/CUDA gate |
| Implicit 3D reference and float64 detail | All three models |
| Native mixed grids | Existing both-direction constraints, gradients, finite-cell masks, genuine refinement, native SITK metric oracles and batch failure isolation |

The audit found that the old `test_linear.py` helper's rigid case was actually
translation-only. The shared fixture now includes a genuine rotation large enough
that even the best translation-only fit fails the unchanged truth gate. The shared
mixed fixture also no longer constructs truth through production parameter
encoding. Translation/affine inputs moved from the old linear helper are verified
bitwise identical. Other existing dedicated rigid recovery tests already rotated;
this was a gap in that helper/contract, not proof that all old rigid tests were weak.

Reusable datasets now live in package-wide `tests/conftest.py`, with fresh arrays
per call. Shared pure oracles live in `tests/oracles.py`; the mixed-grid tests no
longer import a fixture builder from another test module. The previous mocked
identity optimizer in the output-contract check is replaced with a real step.
See [test conventions](../../tests/README.md). No test-suite-wide rewrite was done.

Useful reference patterns were inspected in:
- syntx `tests/test_affine_known_transform.py`: analytic physical truth/TRE rather
  than only image correlation.
- ANTs `Examples/TestSuite/CMakeLists.txt`: distinct translation/rotation/affine
  and initialization scenarios, without adopting its stored image baselines.
- FireANTs `tests/test_moments_registration.py`: explicit dimensionality and known
  pose, without importing its CUDA-only fixture assumptions or ellipse ambiguity.
- ITK `Modules/Core/Transform/test/itkCenteredAffineTransformTest.cxx`: physical
  point and centered-transform semantics, complementing existing native oracles.

## Fresh ConfUSIus workflows

All **20 pair/motion arms** completed using cached original data and installed
ConfUSIus: rigid/affine pairs at example/default and fixed-400 budgets; 120-frame
rigid motion correction at example/default and fixed-100 budgets, CPU/CUDA batch
sizes 32/120 where applicable. Every arm preserves original dtype and repeats
bitwise across its cold/warm calls. One warm repetition was used to check execution,
not qualify new speedups. Plotting/loading/scoring remain outside timed calls.

Same-subject common-domain NCC² (diagnostic, not anatomical truth):

| Pair model | CF example settings | voxreg default 100 | CF fixed 400 | voxreg fixed 400 |
| --- | ---: | ---: | ---: | ---: |
| Rigid | 0.796210 | 0.547489 | 0.828742 | 0.830833 |
| Affine | 0.839402 | 0.638920 | 0.840980 | 0.868076 |

**The default pair budget underfits this example.** Use an explicit 400-update
budget for this demonstrated workflow; this is not a universal optimal budget or
an optimizer-default change. CF and voxreg use different optimizers, stopping and
parameter scaling; equal nominal settings do not imply matched trajectories.

All pair and default/example motion CPU/CUDA comparisons pass the unchanged
0.001-pixel maximum gate. The **fixed-100 float64 motion stress control fails**:
0.0019650704977373073 pixels for both CUDA batch sizes. This exactly reproduces the
previously recorded failure; it is neither hidden nor newly fixed. The report's
`cpu_cuda_precision_gate.all_pass` remains false. Do not call the entire real-data
qualification a pass simply because all calls completed and unit tests pass.

Eight additional native atlas fits completed: rigid/affine × recording/template
moving × CPU/CUDA, original shapes 1×125×80 and 20×91×116, no replicated slab.
All preserve output dtype/masks/fill and the physical model constraints. All four
CPU/CUDA gates pass; worst maximum discrepancy is **4.889916799e-6 pixels**.
CPU update counts are 50/49 for rigid and 143/90 for affine (recording/template
moving). These are runnable atlas workflows, **not established anatomical truth**.
No old CF reference artifact is required by the current `explore_atlas.py` entry
point. CF's slab-expanded objective is not silently substituted for native fitting.

## Reproduce and evidence

```sh
uv run pytest -q
uv run python research/confusius-examples/benchmark_examples.py \
  --output artifacts/package-readiness/new-examples --repeats 1
uv run python research/confusius-examples/explore_atlas.py \
  --headless --moving recording --transform affine --device cuda \
  --output artifacts/package-readiness/new-atlas
```

Use fresh destinations. Repeat the atlas command for the other model/direction/
device combinations. Cached inputs must already exist; no implicit downloads.

Local evidence under `artifacts/package-readiness/`:
- `linear-01/{baseline-tests.log,expanded-tests.log,verified-tests.log}`.
- `linear-01/{build.log,wheel-smoke.log}`: offline packaging and wheel-import checks.
- `linear-01/{check_workflows.py,workflow-checks.json,verified-workflows.log}`:
  fixture-preservation assertions, all workflow results, native constraints and
  pointwise precision gates, including the failed stress cases.
- `examples-01/results.json`: source/input hashes and all 20 records and scores.
- `atlas-01/<model>-<moving>-<device>/`: source/input hashes, arrays and plots.

The interrupted first focused run is retained as `fixtures-tests.log`; it is not
counted as a completed check. Generated artifacts are local, not newly committed
large data. No installed package or sibling repository was modified. BrainGlobe
mesh export and desktop GUI interaction are outside this verification. Deformable
registration remains unsupported; the next implementation milestone is SyN, with
these linear-workflow limitations kept explicit rather than declared solved.
