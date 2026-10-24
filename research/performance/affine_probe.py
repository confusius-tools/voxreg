"""Compare diagnostic sampler substitutions without modifying production code."""

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import voxreg.rigid as vr
from voxreg import create_voxeldata, register_rigid
from voxreg.sampling import rotation_matrices


def make_sampler(grid32=False, elementwise=False):
    """Build an isolated sampler, caching constants only for the current plane."""
    cached_plane = None
    left = right = tolerance = nx = ny = None

    def sample(images, parameters, plane, *, interpolation="linear"):
        """Evaluate a composed affine grid with the original support tolerance."""
        nonlocal cached_plane, left, right, tolerance, nx, ny
        if cached_plane is not plane:
            bounds = parameters.new_tensor([images.shape[2] - 1, images.shape[1] - 1])
            half = bounds / 2
            left = plane.inverse_lattice / half[:, None]
            right = torch.linalg.inv(plane.inverse_lattice) * half[None, :]
            tolerance = 2e-5 / bounds
            nx = (
                torch.arange(
                    images.shape[2], device=parameters.device, dtype=parameters.dtype
                )[None, None, :]
                / bounds[0]
                * 2
                - 1
            )
            ny = (
                torch.arange(
                    images.shape[1], device=parameters.device, dtype=parameters.dtype
                )[None, :, None]
                / bounds[1]
                * 2
                - 1
            )
            cached_plane = plane
        rotation = rotation_matrices(parameters[:, 0])
        translation = parameters[:, 1:] @ left.T
        if elementwise:
            delta = (
                left
                @ (
                    rotation
                    - torch.eye(2, device=parameters.device, dtype=parameters.dtype)
                )
                @ right
            )
            gx = nx + (
                delta[:, 0, 0, None, None] * nx
                + delta[:, 0, 1, None, None] * ny
                + translation[:, 0, None, None]
            )
            gy = ny + (
                delta[:, 1, 0, None, None] * nx
                + delta[:, 1, 1, None, None] * ny
                + translation[:, 1, None, None]
            )
            grid = torch.stack((gx, gy), dim=-1)
        else:
            linear = left @ rotation @ right
            theta = torch.cat((linear, translation[:, :, None]), dim=2)
            grid = F.affine_grid(
                theta.to(images.dtype) if grid32 else theta,
                (len(images), 1, *images.shape[1:]),
                align_corners=True,
            )
        tol = tolerance.to(grid.dtype)
        valid = ((grid >= -1 - tol) & (grid <= 1 + tol)).all(dim=-1)
        sampled = F.grid_sample(
            images[:, None],
            grid.to(images.dtype),
            mode={"linear": "bilinear", "nearest": "nearest"}[interpolation],
            padding_mode="border",
            align_corners=True,
        )[:, 0]
        return sampled, valid

    return sample


def main():
    """Measure warm calls and mapping disagreement against a local baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/notebook/input.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/performance-investigation")
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["elementwise64", "affine64", "affine32"],
        default=["elementwise64"],
    )
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
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
    fixed, moving = data.isel(time=0, drop=True), data.isel(time=slice(1, None))
    batch = 32 if args.device == "cpu" else len(moving)
    samplers = {"baseline": vr.sample_rigid}
    for name in args.variants:
        samplers[name] = make_sampler(
            grid32=name == "affine32", elementwise=name == "elementwise64"
        )
    results = {}
    reference = None
    for name, sampler in samplers.items():
        with patch.object(vr, "sample_rigid", sampler):

            def run():
                return register_rigid(
                    moving,
                    fixed=fixed,
                    batch_size=batch,
                    device=args.device,
                    progress=False,
                )

            warm = run()
            if reference is None:
                reference = warm.transforms.values.copy()
            seconds, stages = [], []
            for _ in range(args.repeats):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                start = perf_counter()
                result = run()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                seconds.append(perf_counter() - start)
                stages.append(result.diagnostics.attrs["timings_seconds"])
            j, i = np.indices(raw.shape[-2:])
            points = position @ np.stack(
                (np.zeros_like(j), j, i, np.ones_like(j))
            ).reshape(4, -1)
            delta = np.einsum(
                "tij,jp->tip", result.transforms.values - reference, points
            )[:, :3]
            rms = np.sqrt(np.mean(np.sum(delta**2, axis=1), axis=1)) / min(
                position[1, 1], position[2, 2]
            )
            results[name] = {
                "seconds": seconds,
                "stages": stages,
                "rms_vs_baseline_pixels_max": float(rms.max()),
                "rms_vs_baseline_pixels_median": float(np.median(rms)),
            }
            np.savez_compressed(
                args.output / f"probe-{args.device}-t{args.threads}-{name}.npz",
                transforms=result.transforms.values,
                registered=result.registered.values,
            )
            print(name, seconds, "max_delta_pixels", rms.max(), flush=True)
    (args.output / f"probe-{args.device}-t{args.threads}.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
