"""Compare stopping policies with fixed-budget controls; quality only, not timings."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from confusius import create_voxeldata

from voxreg import position_to_world, register_rigid

POLICIES = {
    "fixed100": (100, 101, 1e-8),
    "fixed400": (400, 401, 1e-8),
    "itk50-e8": (100, 50, 1e-8),
    "window50-e7": (100, 50, 1e-7),
    "window50-e6": (100, 50, 1e-6),
    "window20-e6": (100, 20, 1e-6),
    "window10-e6": (100, 10, 1e-6),
}


def compare(data, *, device, batch_size):
    """Score all policies on joint support and report physical mapping changes."""
    spec = importlib.util.spec_from_file_location(
        "motion_example",
        Path(__file__).resolve().parents[2] / "examples/notebook_motion_correction.py",
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    affine = position_to_world(data)
    raw = data.values[:, 0].astype(float)
    db = 10 * np.log10(np.maximum(raw / raw.max(), 1e-12))
    j, i = np.indices(raw.shape[1:])
    points = affine @ np.stack((np.zeros_like(j), j, i, np.ones_like(j))).reshape(4, -1)
    spacing = np.linalg.norm(affine[:3, [1, 2]], axis=0).min()
    common = np.ones(raw.shape, dtype=bool)
    results, sampled = {}, {}
    for name, (iterations, window, minimum) in POLICIES.items():
        result = register_rigid(
            data,
            device=device,
            batch_size=batch_size,
            max_iterations=iterations,
            convergence_window_size=window,
            convergence_minimum_value=minimum,
            progress=False,
        )
        matrices = result.transforms.values
        np.testing.assert_array_equal(matrices[0], np.eye(4))
        sampled[name], valid = example.sample_matrices(db, matrices, affine)
        common &= valid
        results[name] = {
            "matrices": matrices,
            "iterations": result.diagnostics.n_iterations.values[1:].tolist(),
            "status": result.diagnostics.status.values[1:].tolist(),
        }
    if np.any(common.sum(axis=(1, 2)) < 2):
        raise ValueError("Insufficient joint support for scoring.")
    scores = {
        name: np.array(
            [
                np.corrcoef(db[0][mask], frame[mask])[0, 1] ** 2
                for frame, mask in zip(frames, common, strict=True)
            ]
        )[1:]
        for name, frames in sampled.items()
    }
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise ValueError("Nonfinite common-support NCC scores.")
    report = {}
    for name, result in results.items():
        record = {key: value for key, value in result.items() if key != "matrices"}
        record["ncc2"] = scores[name].tolist()
        record["median_ncc2"] = float(np.median(scores[name]))
        record["vs_control"] = {}
        for control in ["fixed100", "fixed400"]:
            displacement = np.einsum(
                "tij,jp->tip", result["matrices"] - results[control]["matrices"], points
            )[:, :3]
            rms = np.sqrt(np.mean(np.sum(displacement**2, axis=1), axis=1)) / spacing
            record["vs_control"][control] = {
                "max_rms_pixels": float(rms.max()),
                "worst_ncc2_drop": float((scores[control] - scores[name]).max()),
                "median_ncc2_change": float(np.median(scores[name] - scores[control])),
            }
        report[name] = record
    return {
        "device": device,
        "batch_size": batch_size,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "common_fraction_median": float(np.median(common.mean(axis=(1, 2)))),
        "policies": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="NPZ: values, voxel_to_world, time/k/j/i."
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.output.resolve() == args.input.resolve():
        parser.error("output must not overwrite the input archive")
    torch.set_num_threads(8)
    with np.load(args.input, allow_pickle=False) as archive:
        if len(archive["time"]) < 2:
            parser.error("at least two frames are required")
        data = create_voxeldata(
            archive["values"],
            dims=("time", "k", "j", "i"),
            voxel_to_world=archive["voxel_to_world"],
            **{dim: archive[dim] for dim in ["time", "k", "j", "i"]},
        )
    report = compare(data, device=args.device, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, record in report["policies"].items():
        print(name, np.median(record["iterations"]), record["vs_control"]["fixed400"])


if __name__ == "__main__":
    main()
