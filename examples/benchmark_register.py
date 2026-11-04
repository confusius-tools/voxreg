"""Benchmark registration: wall time, peak memory and independent truth RMS.

See BENCHMARK.md. Use --list to inspect configurations without running them.
"""

from _benchmark_runner import _main

if __name__ == "__main__":
    raise SystemExit(_main(__file__))
