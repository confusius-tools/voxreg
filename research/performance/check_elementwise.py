"""Check real-recording batch invariance for production or the sampler prototype."""

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from affine_probe import make_sampler
from confusius import create_voxeldata

import voxreg.rigid as vr
from voxreg import register


def main():
    """Generate all required baseline/variant outputs instead of reading old runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/notebook/input.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/performance-investigation")
    )
    parser.add_argument(
        "--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"]
    )
    parser.add_argument(
        "--sampler", choices=["production", "elementwise64"], default="elementwise64"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "benchmark",
        Path(__file__).resolve().parents[2] / "examples/benchmark_registration.py",
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    raw, affine, position, coords = benchmark.load_input(args.input)
    data = create_voxeldata(
        raw, dims=("time", "k", "j", "i"), **coords, voxel_to_world=affine
    )
    fixed, moving = data.isel(time=0, drop=True), data.isel(time=slice(1, None))
    j, i = np.indices(raw.shape[-2:])
    points = position @ np.stack((np.zeros_like(j), j, i, np.ones_like(j))).reshape(
        4, -1
    )
    ell = min(position[1, 1], position[2, 2])
    report, files = {}, {}

    def save(name, matrices):
        path = args.output / f"{name}-quality.npz"
        np.savez_compressed(
            path, transforms=np.concatenate([np.eye(4)[None], matrices])
        )
        files[name] = path

    for device in args.devices:
        baseline = register(
            moving,
            fixed=fixed,
            device=device,
            batch_size=len(moving),
            progress=False,
            convergence_window_size=101,
        )
        save(f"baseline-{device}", baseline.transforms.values)
        reference = None
        for batch in sorted({1, 32, len(moving)}):
            name = f"{args.sampler}-{device}-b{batch}"
            sampler = (
                vr.sample_rigid
                if args.sampler == "production"
                else make_sampler(elementwise=True)
            )
            with patch.object(vr, "sample_rigid", sampler):
                start = perf_counter()
                result = register(
                    moving,
                    fixed=fixed,
                    device=device,
                    batch_size=batch,
                    progress=False,
                    convergence_window_size=101,
                )
                print(name, perf_counter() - start, flush=True)
            matrices = result.transforms.values
            save(name, matrices)
            if reference is None:
                reference = matrices
            delta = np.einsum("tij,jp->tip", matrices - reference, points)[:, :3]
            rms = np.sqrt(np.mean(np.sum(delta**2, axis=1), axis=1)) / ell
            report[f"{device}-b1-vs-b{batch}"] = {"rms_pixels_max": float(rms.max())}
            if rms.max() >= 0.001:
                raise AssertionError(f"{name} exceeds the 0.001-pixel batch gate.")
    report["quality"] = benchmark.score(args.input, files)
    filename = (
        "production-validation.json"
        if args.sampler == "production"
        else "elementwise-validation.json"
    )
    (args.output / filename).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
