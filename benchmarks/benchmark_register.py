"""Benchmark registration; add --lazy for streaming read/fit/Zarr writes."""

import sys

if __name__ == "__main__":
    if "--lazy" in sys.argv:
        from _benchmark_lazy import main

        arguments = sys.argv[1:]
        arguments.remove("--lazy")
        raise SystemExit(main(arguments, operation="register"))
    from _benchmark_runner import _main

    raise SystemExit(_main(__file__))
