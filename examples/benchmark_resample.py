"""Benchmark offline resampling with known physical maps, without fitting.

See BENCHMARK.md. Uses the registration benchmark's supervision and reports.
"""

from _benchmark_resample import _configs, _prepare_inputs, _worker
from _benchmark_runner import _main

if __name__ == "__main__":
    raise SystemExit(
        _main(__file__, _configs, _worker, resampling=True, prepare=_prepare_inputs)
    )
