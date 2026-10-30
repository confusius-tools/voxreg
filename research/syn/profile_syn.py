"""Profile the actual ref20 SyN call without changing optimizer behavior."""

import argparse
import hashlib
import json
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import confusius as cf
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from voxreg import fields, register, sampling, syn


def run(args):
    targets = {
        "map": [(sampling, "_map_displacement"), (fields, "_map_displacement")],
        "image": [(sampling, "_sample_displaced")],
        "ncc": [(syn, "local_ncc")],
        "loss": [(syn, "_midpoint_loss")],
        "update": [(syn, "_candidate_fields")],
        "inverse": [(fields, "_inverse_update")],
    }
    compiled = {}
    backend_name = getattr(args, "backend", "inductor")
    if backend_name == "inductor-native-grid":
        from torch._inductor.compile_fx import compile_fx
        from torch._inductor.decomposition import select_decomp_table

        decompositions = select_decomp_table().copy()
        decompositions.pop(torch.ops.aten.grid_sampler_2d.default, None)

        def native_grid_backend(graph, inputs):
            return compile_fx(
                graph,
                inputs,
                decompositions=decompositions,
                config_patches={"compile_threads": 1, "emulate_precision_casts": True},
            )

    with ExitStack() as stack:
        if args.compile:
            stack.enter_context(torch._dynamo.config.patch(recompile_limit=64))
        if getattr(args, "gradient32", False):
            from gradient_filter import gradient_filter

            stack.enter_context(patch.object(syn, "smooth_field", gradient_filter))
        if getattr(args, "inverse_rsqrt", False):
            from inverse_rsqrt import inverse_update

            stack.enter_context(patch.object(fields, "_inverse_update", inverse_update))
        for name in args.compile:
            for module, attribute in targets[name]:
                function = getattr(module, attribute)
                if function not in compiled:
                    backend = getattr(args, "backend", "inductor")
                    if backend == "cudagraphs":
                        compiled[function] = torch.compile(
                            function,
                            fullgraph=True,
                            dynamic=False,
                            backend=backend,
                        )
                    elif backend == "inductor-native-grid":
                        compiled[function] = torch.compile(
                            function,
                            fullgraph=True,
                            dynamic=None if name == "update" else True,
                            backend=native_grid_backend,
                        )
                    else:
                        compiled[function] = torch.compile(
                            function,
                            fullgraph=True,
                            dynamic=None if name == "update" else True,
                            options={
                                "compile_threads": 1,
                                "emulate_precision_casts": True,
                            },
                        )

                def cuda_only(
                    *a,
                    _original=function,
                    _compiled=compiled[function],
                    _phase=name,
                    _backend=backend,
                    **kw,
                ):
                    if not a[0].is_cuda:
                        return _original(*a, **kw)
                    if _backend == "cudagraphs" and _phase == "update":
                        # Backward graph buffers must outlive the next capture.
                        a = (*a[:2], a[2].clone(), a[3].clone(), *a[4:])
                    result = _compiled(*a, **kw)
                    if _backend == "cudagraphs":
                        if isinstance(result, torch.Tensor):
                            return result.clone()
                        return tuple(value.clone() for value in result)
                    return result

                stack.enter_context(patch.object(module, attribute, cuda_only))
        if args.benchmark:
            from landemard import run as benchmark

            args.batch_size = None
            args.fireants_batch_size = 256
            benchmark(args)
            report_path = args.output / "results.json"
            report = json.loads(report_path.read_text())
            report["experimental_compile"] = {
                "targets": args.compile,
                "recompile_limit": 64,
                "backend": backend_name,
                "gradient32": getattr(args, "gradient32", False),
                "inverse_rsqrt": getattr(args, "inverse_rsqrt", False),
                "dynamic": False
                if backend_name == "cudagraphs"
                else {"update": None, "other_targets": True},
                "compile_threads": None if backend_name == "cudagraphs" else 1,
                "emulate_precision_casts": None
                if backend_name == "cudagraphs"
                else True,
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        else:
            _run(args)


def _run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    previous = Path("artifacts/syn-ref20/full-01")
    metadata = json.loads((previous / "results.json").read_text())
    data = cf.load(metadata["source"]).isel(time=slice(0, args.frames)).compute()
    seed = np.load(previous / "affine-seeds.npy")[: args.frames]
    options = {
        "transform": "syn",
        "initial_transform": seed,
        "device": "cuda",
        "batch_size": args.frames,
        "max_iterations": args.iterations,
        "convergence_window_size": args.iterations + 1,
        "progress": False,
    }
    torch.set_num_threads(1)
    cold_start = perf_counter()
    warm = register(data, **(options | {"max_iterations": 2}))
    del warm
    torch.cuda.synchronize()
    cold_warmup = perf_counter() - cold_start
    start = perf_counter()
    result = register(data, **options)
    torch.cuda.synchronize()
    wall = perf_counter() - start
    assert result.transforms is None and result.inverse_transforms is None
    del result
    (args.output / "timing.json").write_text(
        json.dumps(
            {
                "frames": args.frames,
                "iterations": args.iterations,
                "compile": args.compile,
                "backend": getattr(args, "backend", "inductor"),
                "gradient32": getattr(args, "gradient32", False),
                "inverse_rsqrt": getattr(args, "inverse_rsqrt", False),
                "cold_two_update_seconds": cold_warmup,
                "unprofiled_seconds": wall,
            },
            indent=2,
        )
        + "\n"
    )

    def labeled(function, name):
        @wraps(function)
        def observed(*a, **kw):
            with record_function("voxreg::" + name):
                return function(*a, **kw)

        return observed

    with ExitStack() as stack:
        for module, names in [
            (
                syn,
                [
                    "local_ncc",
                    "smooth_field",
                    "compose_fields",
                    "jacobian_determinant",
                    "invert_field",
                ],
            ),
            (fields, ["compose_fields", "sample_field_at", "_pixel_norm"]),
            (sampling, ["_sample_displaced"]),
            (torch.autograd, ["grad"]),
        ]:
            if args.compile:
                continue  # Changing helper globals invalidates Dynamo guards.
            for name in names:
                stack.enter_context(
                    patch.object(module, name, labeled(getattr(module, name), name))
                )
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            result = register(data, **options)
            torch.cuda.synchronize()
        assert result.transforms is None
    prof.export_chrome_trace(str(args.output / "trace.json"))
    averages = prof.key_averages()
    (args.output / "operators.txt").write_text(
        averages.table(sort_by="self_device_time_total", row_limit=60)
    )
    rows = [
        {
            "name": e.key,
            "calls": e.count,
            "cpu_ms": e.cpu_time_total / 1000,
            "cuda_ms": e.device_time_total / 1000,
            "self_cuda_ms": e.self_device_time_total / 1000,
        }
        for e in averages
        if e.key.startswith("voxreg::") and e.cpu_time_total > 0
    ]
    report = {
        "frames": args.frames,
        "iterations": args.iterations,
        "unprofiled_seconds": wall,
        "compile": args.compile,
        "backend": getattr(args, "backend", "inductor"),
        "gradient32": getattr(args, "gradient32", False),
        "inverse_rsqrt": getattr(args, "inverse_rsqrt", False),
        "compile_options": {
            "dynamic": False
            if getattr(args, "backend", "inductor") == "cudagraphs"
            else {"update": None, "other_targets": True},
            "compile_threads": None
            if getattr(args, "backend", "inductor") == "cudagraphs"
            else 1,
            "emulate_precision_casts": None
            if getattr(args, "backend", "inductor") == "cudagraphs"
            else True,
        },
        "cold_two_update_seconds": cold_warmup,
        "scopes": sorted(rows, key=lambda e: e["cuda_ms"], reverse=True),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/voxreg").glob("*.py")
        },
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument(
        "--compile",
        nargs="*",
        choices=["map", "image", "ncc", "loss", "update", "inverse"],
        default=[],
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--gradient32", action="store_true")
    parser.add_argument("--inverse-rsqrt", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["inductor", "inductor-native-grid", "cudagraphs"],
        default="inductor",
    )
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if min(args.frames, args.iterations) < 1:
        parser.error("counts must be positive")
    run(args)
