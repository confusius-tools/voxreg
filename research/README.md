# Historical investigations

These reports preserve **checkpoint-specific evidence**, not additional benchmark
pipelines or current API guidance. Their commands, tolerances, test counts and
claims apply to the recorded source/settings. Use [BENCHMARK.md](../BENCHMARK.md)
for new measurements and [VALIDATION.md](../VALIDATION.md) for current limits.

## Evidence index

- [Linear recovery](linear/README.md) and [slice/volume geometry](slice-geometry/README.md).
- [Performance studies](performance/README.md), including [CPU volume layout](performance/CPU-VOLUME.md).
- [Native ANTs workflows](registration/ANTS.md).
- [SyN implementation/comparisons](syn/README.md) and [synthetic failures](syn/SYNTHETIC-PHANTOM.md).
- [Convergence calibration](convergence/CALIBRATION.md).
- [Real ConfUSIus examples](confusius-examples/README.md).

## Local artifacts

Preliminary source/hash/log evidence is in ignored
`artifacts/archive/preliminary-evidence.tar.gz`, with a verified manifest.
Generated bulk arrays/images/caches were discarded; completed benchmark reports
and installed tools remain. Restore old `artifacts/...` text paths selectively
from the archive. These local files are **not published by a push**.

Removed runners remain in Git history or archived snapshots. The retained native
convergence oracle is calibration source, not another benchmark entry point.
Production checks reuse [shared fixtures](../tests/conftest.py) and
[independent oracles](../tests/oracles.py).

## Pre-cleanup guides

The long milestone narratives remain available at the last pre-cleanup revision,
rather than being duplicated in the maintained guides:

- [Design and original milestones](https://github.com/sdiebolt/voxreg/blob/b396107a903fae62201594e38a54ee51fe11e7e5/DESIGN.md).
- [Detailed validation timeline](https://github.com/sdiebolt/voxreg/blob/b396107a903fae62201594e38a54ee51fe11e7e5/VALIDATION.md).
- [Performance investigation narrative](https://github.com/sdiebolt/voxreg/blob/b396107a903fae62201594e38a54ee51fe11e7e5/PERFORMANCE.md).
