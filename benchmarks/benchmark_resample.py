"""Benchmark resampling; add --lazy for streaming read/sample/Zarr writes."""

import sys

if __name__ == "__main__":
    if "--lazy" in sys.argv:
        from _benchmark_lazy import main

        arguments = sys.argv[1:]
        arguments.remove("--lazy")
        raise SystemExit(main(arguments, operation="resample"))
    from _benchmark_data import _configs
    from _benchmark_resample import _prepare_inputs, _worker
    from _benchmark_runner import _main

    raise SystemExit(
        _main(__file__, _configs, _worker, resampling=True, prepare=_prepare_inputs)
    )
