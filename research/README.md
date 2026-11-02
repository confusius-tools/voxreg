# Historical registration investigations

These documents and compact result files preserve prior numerical, accuracy and
performance evidence. They are **not additional benchmark pipelines**. Historical
commands, script paths, environment details and test counts describe the time the
measurements were captured, not the current checkout.

The exploratory Python runners/profilers and their script-specific tests have
been removed. Their committed versions remain in Git history; uncommitted/latest
runner snapshots and large raw evidence remain under ignored local `artifacts/`,
including `benchmark-cleanup-source-01`, `synthetic-100-source-01`,
`thread-scaling-ants-source-02` and `cpu-volume-improvement-01`. These local data
are not published by a push. The native convergence C++ oracle is retained as
independent numerical/calibration source, not a benchmark entry point.

Use **[BENCHMARK.md](../BENCHMARK.md)** and `examples/benchmark.py` for all new runs.
The production suite lives in `tests/`; its common-support stopping-policy
reference was moved to `tests/calibration_reference.py` without its old CLI.

Useful evidence:

- [CPU volume optimization and linear FireANTs batching](performance/CPU-VOLUME.md).
- [Performance investigation index](performance/README.md).
- [Completed native ANTs workflows](registration/ANTS.md).
- [Known synthetic nonlinear accuracy/precision failures](syn/SYNTHETIC-PHANTOM.md).
- [SyN production acceleration qualification](syn/AC-CPU-SMOOTHING.md).
- [Validation timeline](../VALIDATION.md) and [historical performance](../PERFORMANCE.md).
