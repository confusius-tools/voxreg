"""Profile the actual CPU SyN path on shared independent comparison fixtures."""

import argparse
import json
from contextlib import ExitStack, nullcontext
from functools import wraps
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch
from compare import fit_voxreg, inputs
from cpu_fast import patches
from torch.profiler import ProfilerActivity, profile, record_function

from voxreg import fields, syn


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    fixed, moving, _, _, seed, _ = inputs(args.case)
    torch.set_num_threads(args.threads)
    with patches() if args.cpu_fast else nullcontext():
        fit_voxreg(moving, fixed, seed, 2, "cpu", args.threads)
        start = perf_counter()
        fit_voxreg(moving, fixed, seed, args.iterations, "cpu", args.threads)
        seconds = perf_counter() - start

        def observed(function, name):
            @wraps(function)
            def call(*a, **kw):
                with record_function("voxreg::" + name):
                    return function(*a, **kw)

            return call

        with ExitStack() as stack:
            for module, names in [
                (syn, ["_midpoint_loss", "_candidate_fields", "smooth_field"]),
                (fields, ["invert_field", "_inverse_update"]),
            ]:
                for name in names:
                    stack.enter_context(
                        patch.object(
                            module, name, observed(getattr(module, name), name)
                        )
                    )
            with profile(
                activities=[ProfilerActivity.CPU], record_shapes=True
            ) as trace:
                fit_voxreg(moving, fixed, seed, args.iterations, "cpu", args.threads)
    (args.output / "operators.txt").write_text(
        trace.key_averages().table(sort_by="self_cpu_time_total", row_limit=60)
    )
    (args.output / "results.json").write_text(
        json.dumps(
            {
                "case": args.case,
                "threads": args.threads,
                "iterations": args.iterations,
                "cpu_fast": args.cpu_fast,
                "seconds": seconds,
            },
            indent=2,
        )
        + "\n"
    )
    print(seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case", choices=["2d-medium", "3d-medium", "3d-batch8"], default="3d-medium"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cpu-fast", action="store_true")
    run(parser.parse_args())
