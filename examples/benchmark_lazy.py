"""Benchmark lazy read → register/resample → Zarr storage; see BENCHMARK.md."""

from _benchmark_lazy import main

if __name__ == "__main__":
    raise SystemExit(main())
