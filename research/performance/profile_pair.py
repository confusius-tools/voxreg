"""Diagnose a warm single-image fit; profiler runs are not timing benchmarks."""

import argparse
import json
import statistics
import sys
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/confusius-examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_same_subject import digest, load_inputs
from profile_baseline import scoped

import voxreg.rigid as engine
from voxreg import register


def trace_summary(events):
    kernels = [e for e in events if e.get("cat") == "kernel"]
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    union = 0.0
    last = -float("inf")
    for start, end in intervals:
        union += max(0, end - max(start, last))
        last = max(last, end)
    span = last - intervals[0][0] if intervals else 0.0
    launches = [
        e
        for e in events
        if e.get("cat") in ["cuda_runtime", "cuda_driver"]
        and "Launch" in e.get("name", "")
    ]
    sync = [
        e
        for e in events
        if e.get("cat") in ["cuda_runtime", "cuda_driver"]
        and "Synchronize" in e.get("name", "")
    ]
    copies = [e for e in events if "Memcpy" in e.get("name", "")]
    return {
        "kernel_count": len(kernels),
        "kernel_sum_ms": sum(e["dur"] for e in kernels) / 1000,
        "kernel_union_ms": union / 1000,
        "kernel_span_ms": span / 1000,
        "kernel_busy_fraction": union / span if span else None,
        "median_kernel_us": statistics.median([e["dur"] for e in kernels])
        if kernels
        else None,
        "launch_call_count": len(launches),
        "launch_host_ms": sum(e["dur"] for e in launches) / 1000,
        "synchronize_call_count": len(sync),
        "synchronize_host_ms": sum(e["dur"] for e in sync) / 1000,
        "memcpy_event_count": len(copies),
    }


def run(device, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8 if device == "cpu" else 1)
    fixed, moving, inputs = load_inputs()
    kwargs = {
        "intensity_scaling": "none",
        "initialization": "center_geometry",
        "device": device,
        "progress": False,
        "max_iterations": 100,
        "convergence_window_size": 101,
    }
    captured = {}
    original_optimize = engine._optimize

    def keep(*args, **kw):
        captured.update(args=args, kwargs=kw)
        return original_optimize(*args, **kw)

    with patch.object(engine, "_optimize", keep):
        register(moving, fixed, **kwargs)

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    measurements = []
    for _ in range(3):
        synchronize()
        start = perf_counter()
        result = register(moving, fixed, **kwargs)
        synchronize()
        measurements.append(
            {
                "wall_seconds": perf_counter() - start,
                "stages": result.diagnostics.attrs["timings_seconds"],
            }
        )
    # Unmodified resident optimization separates host preparation and I/O from fitting.
    resident = []
    for _ in range(3):
        synchronize()
        start = perf_counter()
        gold = original_optimize(*captured["args"], **captured["kwargs"])
        synchronize()
        resident.append(perf_counter() - start)
    activities = [ProfilerActivity.CPU] + (
        [ProfilerActivity.CUDA] if device == "cuda" else []
    )
    with (
        ExitStack() as patches,
        profile(
            activities=activities,
            schedule=schedule(wait=20, warmup=2, active=5, repeat=1),
            record_shapes=True,
        ) as prof,
    ):
        original_sample = engine.sample_rigid

        def sampled(*args, **kw):
            prof.step()
            with record_function("voxreg::sample"):
                return original_sample(*args, **kw)

        patches.enter_context(patch.object(engine, "sample_rigid", sampled))
        patches.enter_context(
            patch.object(engine, "global_ncc", scoped("voxreg::ncc", engine.global_ncc))
        )
        patches.enter_context(
            patch.object(
                torch.autograd, "grad", scoped("voxreg::backward", torch.autograd.grad)
            )
        )
        register(moving, fixed, **kwargs)
    trace = output / "trace.json"
    prof.export_chrome_trace(str(trace))
    averages = [
        {
            "name": e.key,
            "count": e.count,
            "self_cpu_us": e.self_cpu_time_total,
            "self_device_us": e.self_device_time_total,
        }
        for e in prof.key_averages()
    ]
    report = {
        "device": device,
        "threads": torch.get_num_threads(),
        "iterations": 100,
        "inputs": inputs,
        "warm_calls": measurements,
        "resident_optimizer_seconds": resident,
        "trace": trace_summary(json.loads(trace.read_text())["traceEvents"]),
        "operators": averages,
        "source_sha256": {
            str(p): digest(p)
            for p in [Path(__file__).resolve(), *Path("src/voxreg").glob("*.py")]
        },
    }
    if device == "cuda":
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    original_optimize(*captured["args"], **captured["kwargs"])
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = original_optimize(*captured["args"], **captured["kwargs"])
            graph.replay()
            torch.cuda.synchronize()
            equal = all(torch.equal(a, b) for a, b in zip(gold, replayed, strict=True))
            assert equal, "Graph changes optimizer output"
            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                start = perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                times.append(perf_counter() - start)
            report["graph_diagnostic"] = {
                "replay_seconds": times,
                "outputs_bitwise_equal": equal,
                "scope": "100 resident optimizer steps only; static graph construction cost excluded, not a product feature or end-to-end timing",
            }
        except RuntimeError as error:
            report["graph_diagnostic"] = {"unavailable": str(error)}
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        device,
        "warm",
        statistics.median([m["wall_seconds"] for m in measurements]),
        "resident",
        statistics.median(resident),
        report["trace"],
        report.get("graph_diagnostic"),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=["cpu", "cuda"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.device, args.output)
