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
    research_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
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
    graphs = backend_name == "cudagraphs" or getattr(args, "inductor_graphs", False)
    compile_options = {"compile_threads": 1, "emulate_precision_casts": True}
    if getattr(args, "inductor_graphs", False):
        compile_options["triton.cudagraphs"] = True
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
                config_patches=compile_options,
            )

    with ExitStack() as stack:
        if args.compile:
            stack.enter_context(torch._dynamo.config.patch(recompile_limit=64))
        if getattr(args, "cpu_fast", False):
            from cpu_fast import patches

            stack.enter_context(patches())
        if getattr(args, "fused_arithmetic", False):
            from fused_arithmetic import geometry_map, inverse_update

            stack.enter_context(
                patch.object(sampling, "_map_displacement", geometry_map)
            )
            stack.enter_context(patch.object(fields, "_map_displacement", geometry_map))
            stack.enter_context(patch.object(fields, "_inverse_update", inverse_update))
        if getattr(args, "box_filter", False):
            from box_filter import box_sum

            stack.enter_context(patch.object(syn, "_box_sum", box_sum))
        if getattr(args, "cpu_map", False):
            from cpu_map import cpu_map

            stack.enter_context(patch.object(sampling, "_map_displacement", cpu_map))
            stack.enter_context(patch.object(fields, "_map_displacement", cpu_map))
        if getattr(args, "jacobian_direct", False):
            from jacobian_direct import jacobian_direct

            stack.enter_context(
                patch.object(syn, "jacobian_determinant", jacobian_direct)
            )
            stack.enter_context(
                patch.object(fields, "jacobian_determinant", jacobian_direct)
            )
        if getattr(args, "smooth_batch", False):
            from smooth_layout import batch_smooth

            original_smooth = syn.smooth_field

            def layout_smooth(field, grid, *, sigma):
                if field.is_cuda and len(grid.shape) == 2:
                    return batch_smooth(field, grid, sigma=sigma)
                return original_smooth(field, grid, sigma=sigma)

            stack.enter_context(patch.object(syn, "smooth_field", layout_smooth))
        if getattr(args, "contract_inverse", False):
            from contract_inverse import contract_inverse

            stack.enter_context(patch.object(syn, "invert_field", contract_inverse))
        if getattr(args, "gradient32", False):
            from gradient_filter import gradient_filter

            stack.enter_context(patch.object(syn, "smooth_field", gradient_filter))
        if getattr(args, "inverse_rsqrt", False):
            from inverse_rsqrt import inverse_update

            stack.enter_context(patch.object(fields, "_inverse_update", inverse_update))
        if getattr(args, "newton_inverse", False) or getattr(args, "newton3d", False):
            import newton_inverse

            inverse_module = newton_inverse
            if getattr(args, "newton3d", False):
                import newton3d

                inverse_module = newton3d
            stack.enter_context(
                patch.object(syn, "_final_inverse", inverse_module.final_inverse)
            )
            if args.compile:
                targets["newton"] = [(newton_inverse, "_newton_update")]
                args.compile = [*args.compile, "newton"]
                if getattr(args, "newton3d", False):
                    targets["newton3d"] = [(newton3d, "_newton_update3d")]
                    args.compile = [*args.compile, "newton3d"]
        if getattr(args, "maxnorm_squared", False):
            from maxnorm import candidate_fields

            stack.enter_context(
                patch.object(syn, "_candidate_fields", candidate_fields)
            )
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
                            options=compile_options,
                        )

                def cuda_only(
                    *a,
                    _original=function,
                    _compiled=compiled[function],
                    _phase=name,
                    _graphs=graphs,
                    **kw,
                ):
                    if not a[0].is_cuda:
                        return _original(*a, **kw)
                    if _graphs and _phase == "update":
                        # Backward graph buffers must outlive the next capture.
                        a = (*a[:2], a[2].clone(), a[3].clone(), *a[4:])
                    result = _compiled(*a, **kw)
                    if _graphs:
                        if isinstance(result, torch.Tensor):
                            return result.clone()
                        return tuple(value.clone() for value in result)
                    return result

                stack.enter_context(patch.object(module, attribute, cuda_only))
        if args.benchmark:
            from landemard import run as benchmark

            args.batch_size = getattr(args, "batch_size", None)
            args.fireants_batch_size = 256
            benchmark(args)
            report_path = args.output / "results.json"
            report = json.loads(report_path.read_text())
            report["experimental_compile"] = {
                "targets": args.compile,
                "research_source_sha256": research_hashes,
                "contract_inverse": getattr(args, "contract_inverse", False),
                "smooth_batch": getattr(args, "smooth_batch", False),
                "jacobian_direct": getattr(args, "jacobian_direct", False),
                "inductor_graphs": getattr(args, "inductor_graphs", False),
                "cpu_fast": getattr(args, "cpu_fast", False),
                "cpu_map": getattr(args, "cpu_map", False),
                "maxnorm_squared": getattr(args, "maxnorm_squared", False),
                "box_filter": getattr(args, "box_filter", False),
                "newton_inverse": getattr(args, "newton_inverse", False),
                "newton3d": getattr(args, "newton3d", False),
                "fused_arithmetic": getattr(args, "fused_arithmetic", False),
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
            if getattr(args, "contract_inverse", False):
                from contract_inverse import COUNTS

                report["inverse_solver_counts"] = COUNTS
            if getattr(args, "newton_inverse", False) or getattr(
                args, "newton3d", False
            ):
                from newton_inverse import COUNTS

                report["newton_solver_counts"] = COUNTS
                if getattr(args, "newton3d", False):
                    from newton3d import COUNTS

                    report["newton_volume_solver_counts"] = COUNTS
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
    parser.add_argument("--contract-inverse", action="store_true")
    parser.add_argument("--cpu-fast", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--smooth-batch", action="store_true")
    parser.add_argument("--inductor-graphs", action="store_true")
    parser.add_argument("--jacobian-direct", action="store_true")
    parser.add_argument("--cpu-map", action="store_true")
    parser.add_argument("--maxnorm-squared", action="store_true")
    parser.add_argument("--box-filter", action="store_true")
    parser.add_argument("--newton-inverse", action="store_true")
    parser.add_argument("--newton3d", action="store_true")
    parser.add_argument("--fused-arithmetic", action="store_true")
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
