"""Profile the actual CPU SyN path on shared independent comparison fixtures."""

import argparse
import json
from contextlib import ExitStack, nullcontext
from functools import wraps
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch
from compare import fit_fireants, fit_voxreg, inputs
from cpu_fast import patches
from torch.profiler import ProfilerActivity, profile, record_function

from voxreg import fields, syn


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    fixed, moving, _, _, seed, _ = inputs(args.case)
    torch.set_num_threads(args.threads)
    fit = fit_voxreg if args.backend == "voxreg" else fit_fireants
    from cpu_smoothing import CPUSmoother

    smoother = CPUSmoother(args.smoothing)
    with (
        patches() if args.cpu_fast else nullcontext(),
        patch.object(syn, "smooth_field", smoother)
        if args.smoothing != "baseline"
        else nullcontext(),
    ):
        fit(moving, fixed, seed, 2, "cpu", args.threads)
        start = perf_counter()
        fit(moving, fixed, seed, args.iterations, "cpu", args.threads)
        seconds = perf_counter() - start

        def observed(function, name):
            @wraps(function)
            def call(*a, **kw):
                label = args.backend + "::" + name
                if name == "_separable_filtering_conv":
                    label += ".kernel" + str(a[1][0].numel())
                with record_function(label):
                    return function(*a, **kw)

            return call

        observations = [
            (
                syn,
                [
                    "_midpoint_loss",
                    "local_ncc",
                    "_candidate_fields",
                    "smooth_field",
                    "jacobian_determinant",
                    "_final_inverse",
                ],
            ),
            (fields, ["invert_field", "_inverse_update", "_contracted_cpu_inverse"]),
        ]
        if args.backend == "fireants":
            from fireants.losses import cc
            from fireants.registration import syn as fire_syn
            from fireants.registration.optimizers.adam import WarpAdam

            observations = [
                (cc, ["_separable_filtering_conv"]),
                (cc.LocalNormalizedCrossCorrelationLoss, ["forward_util"]),
                (WarpAdam, ["step"]),
                (fire_syn, ["compositive_warp_inverse"]),
            ]
        observations.append((torch.autograd, ["grad", "backward"]))
        with ExitStack() as stack:
            for module, names in observations:
                for name in names:
                    stack.enter_context(
                        patch.object(
                            module, name, observed(getattr(module, name), name)
                        )
                    )
            with profile(
                activities=[ProfilerActivity.CPU], record_shapes=True
            ) as trace:
                profiled_start = perf_counter()
                fit(moving, fixed, seed, args.iterations, "cpu", args.threads)
                profiled_seconds = perf_counter() - profiled_start
    assert any(
        event.key.startswith(args.backend + "::") for event in trace.key_averages()
    )
    (args.output / "by-shape.txt").write_text(
        trace.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total", row_limit=80
        )
    )
    (args.output / "operators.txt").write_text(
        trace.key_averages().table(sort_by="self_cpu_time_total", row_limit=60)
    )
    (args.output / "results.json").write_text(
        json.dumps(
            {
                "case": args.case,
                "backend": args.backend,
                "smoothing": args.smoothing,
                "threads": args.threads,
                "iterations": args.iterations,
                "cpu_fast": args.cpu_fast,
                "seconds": seconds,
                "profiled_fit_seconds": profiled_seconds,
                "phases": [
                    {
                        "name": event.key,
                        "calls": event.count,
                        "inclusive_seconds": event.cpu_time_total / 1e6,
                        "exclusive_seconds": event.self_cpu_time_total / 1e6,
                    }
                    for event in trace.key_averages()
                    if event.key.startswith(args.backend + "::")
                ],
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
    parser.add_argument("--backend", choices=["voxreg", "fireants"], default="voxreg")
    parser.add_argument(
        "--smoothing",
        choices=["baseline", "torch64", "gradient32", "all32"],
        default="baseline",
    )
    args = parser.parse_args()
    if args.backend == "fireants" and (args.cpu_fast or args.smoothing != "baseline"):
        parser.error("voxreg experiment switches cannot modify the FireANTs reference")
    run(args)
