# ITK-style, independent-frame convergence

The production monitor is `src/voxreg/convergence.py`. It implements ITK v4's
`WindowConvergenceMonitoringFunction<double>` calculation, not a generic linear
regression or absolute change test. Registration still uses voxreg's Adam:
matching the monitor does not imply matching ITK's optimization trajectory.

## Calculation and reference

ITK keeps the most recent N metric values, divided by the accumulated absolute
energy **since the monitor started**, not just the current window. It fits a
first-order B-spline with two control points and returns the negative endpoint
gradient. The optimizer checks `value <= minimum` before the next update. Until
N observations exist, the monitor returns double max.

The spline fitter uses float32 point coordinates, basis weights and accumulated
weight denominators even with double energies. Its endpoint is shifted inside
the domain by spacing 0.1 × epsilon 0.001. These details matter: even a constant
nonzero loss can produce a small nonzero trend. We preserve them. Zero-energy
histories return zero, matching the reference's nonfinite-coefficient cleanup.
The fixed fitting/gradient operator is precomputed once per batch; runtime uses
a double weighted reduction. This is numerically equivalent, not bitwise identical.

`tests/data/convergence_itk.json` records 80 observations for each of eight loss
streams, with windows 2/10/20/50: constant positive/negative, zero, improving,
worsening, oscillating, plateau and seeded random. CPU/CUDA comparisons use
atol 2e-15 / rtol 1e-12; startup sentinels match exactly. Tests also prove frozen
histories, independent frames, pre-update stopping and fixed-budget controls.
Existing image recovery and CPU/CUDA mapping tolerances were not relaxed.

Fixture provenance: the C++ oracle used the **v5.4.7 monitor and spline fitter /
control-point function headers**, whose SHA256s are stored in the fixture. It
linked against the available local ITK 6.0 core SDK. This is not a claim that the
entire oracle was built against a clean 5.4.7 binary distribution. Normal Python
unit tests read the fixture; no ITK calls or C++ build are needed by the runtime.

For an independent SDK build (prefer ITK 5.4.7):

```bash
cmake -S research/convergence -B artifacts/convergence/oracle-build \
  -DITK_DIR=/path/to/ITK/lib/cmake/ITK-5.4
cmake --build artifacts/convergence/oracle-build -j2
printf '%s\n' -0.7 -0.8 -0.9 | artifacts/convergence/oracle-build/convergence_oracle 2
```

`ITK_CONVERGENCE_HEADERS` can override the monitor/fitter headers if linking a
different local SDK. To verify every stored sequence against a rebuilt oracle:

```python
import json
import subprocess
from pathlib import Path
import numpy as np

fixture = json.loads(Path("tests/data/convergence_itk.json").read_text())
for window, cases in fixture["expected"].items():
    for name, expected in cases.items():
        output = subprocess.check_output(
            ["artifacts/convergence/oracle-build/convergence_oracle", window],
            input="\n".join(map(str, fixture["losses"][name])).encode(),
        )
        actual = [float(line) for line in output.splitlines()]
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=1e-12)
```

## Integration and defaults

Defaults follow ITK gradient descent's **window 50 / minimum 1e-8**, rather than
blindly copying the historical comparison's explicitly configured 10 / 1e-6.
Histories, normalization, Adam clocks/moments and stopping decisions are independent
per frame. The existing overlap/metric/gradient guards take precedence. Best-valid
transforms and the implicit fixed reference are preserved. Convergence is checked
before the next update; n_iterations counts updates, not metric evaluations.
A window larger than max_iterations disables the monitor without allocating that
history. Historical performance/precision harnesses now explicitly use 101 for their
100-update controls; the notebook example uses the new default policy.

No `.item()`, host tensor inspection or global batch-loss stopping is introduced.
As before, fixed-shape batches still execute the full iteration loop after frames
freeze. Thus there is **no automatic wall-time saving** from lower update counts.
Coarse batch termination would be a separate, measured synchronization tradeoff.

## Real window and measured cost

Same 120-frame neutral float32 rat75 window used in earlier reports; original
output dtype and fitting precision unchanged. `results.json` stores timings,
status/iteration summaries, common-support scores and fitting source hashes.
Unmonitored fresh processes; one cold and three warm calls; separate from tests.
Desktop activity/unfixed clocks remained, and CPU ranges are particularly noisy.
No new memory profiling or biological accuracy claim.

| Policy | Median NCC² | Minimum NCC² | Moving-frame updates, min/median/max |
|---|---:|---:|---:|
| Previous policy, CPU32 | .961191725335 | .917576168845 |100/100/100|
| Trial10 /1e-6, CUDA120 | .960610301675 | .915302938923 |10/14/46|
| Trial20 /1e-6, CUDA120 | .961191725331 | .917572366096 |23/54/69|
| Default50 /1e-8, CPU32 | .961191725335 | .917576168845 |51/100/100|
| Default50 /1e-8, CUDA120 | .961191725331 | .917576168843 |51/100/100|

All methods share the same scoring support (median 98.33%). With the final
50/1e-8 policy, **transforms and registered output bytes are identical to the
previous policy on each device**, despite six frames freezing sooner. The 10-value
trial stops earlier but sacrifices score, so it is not the default. With conservative
ITK defaults, **6/119 moving frames converge; 113 hit the iteration limit**.
Do not mistake that limit for convergence or claim the plateau problem is solved.
Broader threshold calibration remains necessary.

| Device / batch | Previous warm median s (range) | New warm median s (range) |
|---|---:|---:|
| CPU8 threads /32 |2.958 (2.915–3.032)|3.267 (2.823–3.770)|
| CUDA /120 |.4073 (.3983–.4131)|.4230 (.4228–.4602)|

The observed CUDA median overhead is about **4%**, not zero. CPU ranges overlap;
this run cannot isolate a CPU overhead reliably. Earlier same-session timings
were roughly half these absolute values, reinforcing why they must not be mixed
with historical benchmark numbers. The current implementation prioritizes a
verified stopping calculation and no iteration-loop host reads over speculative
throughput claims.
