"""Capture five steady optimizer iterations; never treat profiles as benchmarks."""

import argparse
import importlib.util
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from affine_probe import make_sampler
from confusius import create_voxeldata
from torch.profiler import ProfilerActivity, profile, record_function, schedule

import voxreg.rigid as vr
from voxreg import register


def scoped(label, function):
    """Attach a profiler scope without changing the wrapped computation."""

    def wrapped(*args, **kwargs):
        with record_function(label):
            return function(*args, **kwargs)

    return wrapped


def main():
    """Profile one resident chunk after an unrecorded full-call warmup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["voxreg", "fireants"])
    parser.add_argument("device", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--sampler", choices=["baseline", "elementwise64"], default="baseline"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/notebook/input.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/performance-investigation")
    )
    args = parser.parse_args()
    if args.threads < 1 or args.batch < 1:
        parser.error("threads and batch must be positive")
    if args.backend != "voxreg" and args.sampler != "baseline":
        parser.error("sampler substitutions apply only to voxreg")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"{args.backend}-{args.device}-b{args.batch}-t{args.threads}"
    if args.sampler != "baseline":
        name += "-" + args.sampler
    spec = importlib.util.spec_from_file_location(
        "benchmark",
        Path(__file__).resolve().parents[2] / "examples/benchmark_registration.py",
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    raw, affine, position, coords = benchmark.load_input(args.input)
    raw = raw[: args.batch + 1]
    coords["time"] = coords["time"][: len(raw)]
    data = create_voxeldata(
        raw, dims=("time", "k", "j", "i"), **coords, voxel_to_world=affine
    )
    fixed, moving = data.isel(time=0, drop=True), data.isel(time=slice(1, None))

    def run():
        if args.backend == "voxreg":
            return register(
                moving,
                fixed=fixed,
                batch_size=args.batch,
                device=args.device,
                progress=False,
                convergence_window_size=101,
            )
        return benchmark.run_fireants(raw, position, args.device, args.batch)

    def capture(prof):
        prof.export_chrome_trace(str(args.output / f"{name}-trace.json"))
        events = [
            {
                "name": event.key,
                "count": event.count,
                "device_type": str(event.device_type),
                "self_cpu_us": event.self_cpu_time_total,
                "cpu_us": event.cpu_time_total,
                "self_device_us": event.self_device_time_total,
                "device_us": event.device_time_total,
            }
            for event in prof.key_averages()
        ]
        (args.output / f"{name}.json").write_text(json.dumps(events, indent=2) + "\n")
        (args.output / f"{name}.txt").write_text(
            prof.key_averages().table(
                sort_by="self_device_time_total"
                if args.device == "cuda"
                else "self_cpu_time_total",
                row_limit=25,
            )
        )

    activities = [ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with ExitStack() as patches:
        if args.sampler != "baseline":
            patches.enter_context(
                patch.object(vr, "sample_rigid", make_sampler(elementwise=True))
            )
        run()
        if args.device == "cuda":
            torch.cuda.synchronize()
        with profile(
            activities=activities,
            schedule=schedule(wait=20, warmup=2, active=5, repeat=1),
            on_trace_ready=capture,
            record_shapes=True,
        ) as prof:
            if args.backend == "voxreg":
                original_sample = vr.sample_rigid

                def sample(*a, **kw):
                    prof.step()
                    with record_function("voxreg::sample"):
                        return original_sample(*a, **kw)

                patches.enter_context(patch.object(vr, "sample_rigid", sample))
                patches.enter_context(
                    patch.object(vr, "global_ncc", scoped("voxreg::ncc", vr.global_ncc))
                )
                patches.enter_context(
                    patch.object(
                        torch.autograd,
                        "grad",
                        scoped("voxreg::backward", torch.autograd.grad),
                    )
                )
            else:
                import fireants.registration.rigid as fa

                original_step = torch.optim.Adam.step

                def step(*a, **kw):
                    with record_function("fireants::adam"):
                        result = original_step(*a, **kw)
                    prof.step()
                    return result

                original_init = fa.RigidRegistration.__init__

                def initialize(self, *a, **kw):
                    kw["custom_loss"] = scoped("fireants::ncc", kw["custom_loss"])
                    original_init(self, *a, **kw)

                patches.enter_context(patch.object(torch.optim.Adam, "step", step))
                patches.enter_context(
                    patch.object(
                        torch.Tensor,
                        "backward",
                        scoped("fireants::backward", torch.Tensor.backward),
                    )
                )
                patches.enter_context(
                    patch.object(fa.RigidRegistration, "__init__", initialize)
                )
                patches.enter_context(
                    patch.object(
                        fa,
                        "fireants_interpolator",
                        scoped("fireants::sample", fa.fireants_interpolator),
                    )
                )
            run()
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(name, "saved")


if __name__ == "__main__":
    main()
