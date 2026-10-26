"""Record mixed-domain truth and CPU/CUDA gates; unit fixtures supply one oracle."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from voxreg import position_to_world, register

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "slice_fixtures", ROOT / "tests/test_slice_volume.py"
)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def pixels(matrix, truth, data):
    points = position_to_world(data) @ np.vstack(
        (np.indices(data.shape).reshape(3, -1), np.ones((1, data.size)))
    )
    ell = np.linalg.norm(
        position_to_world(data)[:3, :3][:, np.array(data.shape) > 1], axis=0
    ).min()
    error = np.linalg.norm(((matrix - truth) @ points)[:3], axis=0) / ell
    return {
        "rms_pixels": float(np.sqrt(np.mean(error**2))),
        "max_pixels": float(error.max()),
    }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    records = []
    agreement = []
    for model in ["translation", "rigid", "affine"]:
        target, volume, truth = fixtures.pair(model)
        matrices = {}
        for device in fixtures.DEVICES:
            result = register(
                volume,
                target,
                transform=model,
                intensity_scaling="none",
                device=device,
                max_iterations=400,
                convergence_window_size=401,
                progress=False,
            )
            matrix = result.transforms.values
            matrices[device] = matrix
            error = pixels(matrix, truth, target)
            records.append(
                {
                    "model": model,
                    "device": device,
                    "truth": error,
                    "truth_gate_pass": error["rms_pixels"] < 0.25,
                    "status": str(result.diagnostics.status.values),
                    "iterations": int(result.diagnostics.n_iterations),
                    "slice_objective": float(result.diagnostics.objective),
                    "matrix": matrix.tolist(),
                }
            )
        if "cuda" in matrices:
            measured = {
                key: pixels(matrices["cpu"], matrices["cuda"], data)["max_pixels"]
                for key, data in [
                    ("observed_slice", target),
                    ("canonical_volume_extension", volume),
                ]
            }
            agreement.append(
                {
                    "model": model,
                    "max_pointwise_pixels": measured,
                    "precision_gate_pass": max(measured.values()) < 0.001,
                }
            )
    report = {
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).resolve(),
                ROOT / "tests/test_slice_volume.py",
                *sorted((ROOT / "src/voxreg").glob("*.py")),
            ]
        },
        "cases": records,
        "cpu_cuda": agreement,
        "truth_gate_rms_pixels": 0.25,
        "precision_gate_max_pixels": 0.001,
        "scope": "Known SciPy-interpolated 3D volume, independent slice grid; fixed400 from identity. Canonical normal extension comparison is a numerical check, not anatomical truth outside the slice.",
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    assert all(case["truth_gate_pass"] for case in records)
    assert all(case["precision_gate_pass"] for case in agreement)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
