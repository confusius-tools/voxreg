"""Probe CPU costs and precision-preserving NCC changes without engine edits.

Precision-changing variants are diagnostics, not proposed drop-in replacements.
Run one variant per fresh interpreter and keep other workloads idle.
"""

import argparse
import importlib.util
import inspect
import json
from contextlib import ExitStack
from dataclasses import fields, replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from voxreg import create_voxeldata, register_rigid, rigid, sampling

VARIANTS = [
    "voxreg",
    "voxreg-ncc32",
    "voxreg-fit32",
    "voxreg-split-bounds",
    "voxreg-compiled-ncc",
    "voxreg-compiled-ncc-split-bounds",
    "voxreg-masked-ncc",
    "voxreg-masked-ncc-split-bounds",
    "voxreg-affine32",
    "voxreg-affine32-ncc32",
    "fireants",
    "fireants-ncc64",
]


def clone_function(function, replacements, namespace, output):
    """Apply explicit, checked substitutions and retain the generated source."""
    source = inspect.getsource(function)
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"Expected exactly one source match: {old!r}")
        source = source.replace(old, new)
    output.write_text(source)
    scope = dict(namespace)
    exec(compile(source, str(output), "exec"), scope)  # noqa: S102 -- checked local source, not user code
    return scope[function.__name__]


def masked_ncc(output):
    """Share masked centered images between covariance and variance terms."""
    if "vx = x.square().sum(dim=axes)" in inspect.getsource(sampling.global_ncc):
        return sampling.global_ncc
    return clone_function(
        sampling.global_ncc,
        [
            (
                "x = moved - (moved * weights).sum(dim=axes, keepdim=True) / count",
                "x = (moved - (moved * weights).sum(dim=axes, keepdim=True) / count) * weights",
            ),
            (
                "y = fixed - (fixed * weights).sum(dim=axes, keepdim=True) / count",
                "y = (fixed - (fixed * weights).sum(dim=axes, keepdim=True) / count) * weights",
            ),
            ("(x.square() * weights).sum(dim=axes)", "x.square().sum(dim=axes)"),
            ("(y.square() * weights).sum(dim=axes)", "y.square().sum(dim=axes)"),
            ("(x * y * weights).sum(dim=axes)", "(x * y).sum(dim=axes)"),
        ],
        vars(sampling),
        output,
    )


def fit32_optimizer(output):
    """Fit in float32, but return parameters for the unchanged output geometry."""
    ncc = clone_function(
        sampling.global_ncc,
        [
            (
                "moved.to(torch.float64), fixed.to(torch.float64)",
                "moved.to(torch.float32), fixed.to(torch.float32)",
            )
        ],
        vars(sampling),
        output,
    )
    original = rigid._optimize

    def optimize(images, fixed, plane, reference, max_iterations, learning_rate):
        fit_plane = replace(
            plane,
            **{
                field.name: getattr(plane, field.name).float()
                for field in fields(plane)
                if isinstance(getattr(plane, field.name), torch.Tensor)
            },
        )
        with patch.object(rigid, "global_ncc", ncc):
            parameters, steps, status = original(
                images, fixed, fit_plane, reference, max_iterations, learning_rate
            )
        return parameters.to(plane.parameter_scale.dtype), steps, status

    return optimize


def affine32(images, parameters, plane, *, interpolation="linear"):
    """Use the C++ affine-grid generator, with float32 dense grid arithmetic."""
    rotation = sampling.rotation_matrices(parameters[:, 0])
    linear = plane.plane_to_normalized @ rotation @ plane.normalized_to_plane
    translation = parameters[:, 1:] @ plane.plane_to_normalized.T
    theta = torch.cat((linear, translation[:, :, None]), dim=2).to(images.dtype)
    grid = F.affine_grid(theta, (len(images), 1, *images.shape[1:]), align_corners=True)
    tolerance = plane.support_tolerance.to(grid.dtype)
    valid = ((grid >= -1 - tolerance) & (grid <= 1 + tolerance)).all(dim=-1)
    moved = F.grid_sample(
        images[:, None],
        grid,
        mode={"linear": "bilinear", "nearest": "nearest"}[interpolation],
        padding_mode="border",
        align_corners=True,
    )[:, 0]
    return moved, valid


def main():
    """Measure full recording calls, retaining transforms for common scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/notebook/input.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/cpu-investigation")
    )
    args = parser.parse_args()
    if min(args.threads, args.repeats, args.batch_size) < 1:
        parser.error("threads, repeats and batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    name = f"{args.variant}-t{args.threads}"
    if args.device != "cpu" or args.batch_size != 32:
        name += f"-{args.device}-b{args.batch_size}"
    spec = importlib.util.spec_from_file_location(
        "benchmark",
        Path(__file__).resolve().parents[2] / "examples/benchmark_registration.py",
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    raw, affine, position, coords = benchmark.load_input(args.input)
    data = create_voxeldata(
        raw, dims=("time", "k", "j", "i"), coords=coords, voxel_to_world=affine
    )
    fireants = args.variant.startswith("fireants")
    fireants_run = benchmark.run_fireants
    with ExitStack() as patches:
        if args.variant == "voxreg-fit32":
            patches.enter_context(
                patch.object(
                    rigid, "_optimize", fit32_optimizer(args.output / f"{name}-ncc.py")
                )
            )
        if args.variant.endswith("ncc32"):
            ncc = clone_function(
                sampling.global_ncc,
                [
                    (
                        "moved.to(torch.float64), fixed.to(torch.float64)",
                        "moved.to(torch.float32), fixed.to(torch.float32)",
                    )
                ],
                vars(sampling),
                args.output / f"{name}-ncc.py",
            )
            patches.enter_context(patch.object(rigid, "global_ncc", ncc))
        if "affine32" in args.variant:
            patches.enter_context(patch.object(rigid, "sample_rigid", affine32))
        if "masked-ncc" in args.variant:
            patches.enter_context(
                patch.object(
                    rigid, "global_ncc", masked_ncc(args.output / f"{name}-ncc.py")
                )
            )
        if "compiled-ncc" in args.variant:
            patches.enter_context(
                patch.object(
                    rigid,
                    "global_ncc",
                    torch.compile(sampling.global_ncc, fullgraph=True, dynamic=True),
                )
            )
        if "split-bounds" in args.variant:
            sampler = clone_function(
                sampling.sample_rigid,
                [
                    (
                        "valid = ((grid >= -1 - tolerance) & (grid <= 1 + tolerance)).all(dim=-1)",
                        (
                            "valid = ((gx >= -1 - tolerance[0]) & (gx <= 1 + tolerance[0])\n"
                            "             & (gy >= -1 - tolerance[1]) & (gy <= 1 + tolerance[1]))"
                        ),
                    )
                ],
                vars(sampling),
                args.output / f"{name}-sampler.py",
            )
            patches.enter_context(patch.object(rigid, "sample_rigid", sampler))
        if args.variant == "fireants-ncc64":
            fireants_run = clone_function(
                benchmark.run_fireants,
                [
                    (
                        "weights = (moved[:, 1] >= 1 - 1e-5).to(moved.dtype)",
                        "weights = (moved[:, 1] >= 1 - 1e-5).to(torch.float64)",
                    ),
                    (
                        "x, y = moved[:, 0], target[:, 0]",
                        "x, y = moved[:, 0].to(torch.float64), target[:, 0].to(torch.float64)",
                    ),
                ],
                vars(benchmark),
                args.output / f"{name}-backend.py",
            )

        def run():
            if fireants:
                _, matrices, steps = fireants_run(
                    raw, position, args.device, args.batch_size
                )
                return matrices, steps, None
            result = register_rigid(
                data, batch_size=args.batch_size, device=args.device, progress=False
            )
            return (
                result.transforms.values,
                result.diagnostics.n_iterations.values,
                result.diagnostics.attrs["timings_seconds"],
            )

        if args.device == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        run()
        if args.device == "cuda":
            torch.cuda.synchronize()
        cold = perf_counter() - start
        seconds, stages = [], []
        for _ in range(args.repeats):
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            matrices, steps, timing = run()
            if args.device == "cuda":
                torch.cuda.synchronize()
            seconds.append(perf_counter() - start)
            stages.append(timing)
        if not np.array_equal(steps, np.r_[0, np.repeat(100, len(raw) - 1)]):
            raise AssertionError(
                "The real recording must take 100 updates per moving frame."
            )
        np.testing.assert_array_equal(matrices[0], np.eye(4))
        np.savez_compressed(args.output / f"{name}.npz", transforms=matrices)
        record = {
            "variant": args.variant,
            "threads": args.threads,
            "batch_size": args.batch_size,
            "device": args.device,
            "shape": list(raw.shape),
            "cold_seconds": cold,
            "warm_seconds": seconds,
            "stages": stages,
            "torch": torch.__version__,
            "note": "Timing-only pass, no profiler/memory sampler. Precision diagnostics do not establish preserved behavior.",
        }
        (args.output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(name, seconds)


if __name__ == "__main__":
    main()
