"""Complete resident-array registration calls and full-workload truth checks."""

import json

import numpy as np
from _benchmark_data import SHAPES, _dataset, _grid, _run_fireants, _write_json
from _benchmark_metrics import _linear_rms, _point_rms
from _benchmark_resample_io import _digest

from voxreg import position_to_world, register


def _prepare_inputs(config_path):
    """Stage the same original arrays and physical geometry for every backend."""
    config = json.loads(config_path.read_text())
    directory = (
        config_path.parent.parent / "inputs" / f"{config['workload']}-{config['model']}"
    )
    if not (directory / "inputs.json").exists():
        moving, fixed, amplitudes = _dataset(
            SHAPES[config["workload"]],
            config["frames"],
            config["model"],
            config["workload"].endswith("+t"),
        )
        recording = "time" in moving.dims
        arrays = {
            "moving": moving.values,
            "fixed": fixed.values,
            "amplitudes": amplitudes,
            "time": moving.time.values if recording else np.array([0]),
        }
        metadata = {
            "operation": "registration",
            "model": config["model"],
            "workload": config["workload"],
            "recording": recording,
            "affine": position_to_world(fixed).tolist(),
        }
        metadata["input_sha256"] = _digest(arrays, metadata)
        directory.mkdir(parents=True)
        for name, values in arrays.items():
            np.save(directory / f"{name}.npy", values, allow_pickle=False)
        _write_json(directory / "inputs.json", metadata)
    config["inputs"] = str(directory)
    _write_json(config_path, config)


def _voxel_inputs(arrays, metadata):
    from confusius import create_voxeldata

    dimensions = ("k", "j", "i")
    moving = create_voxeldata(
        arrays["moving"],
        dims=(("time",) if metadata["recording"] else ()) + dimensions,
        voxel_to_world=np.array(metadata["affine"]),
        **({"time": arrays["time"]} if metadata["recording"] else {}),
    )
    fixed = create_voxeldata(
        arrays["fixed"], dims=dimensions, voxel_to_world=np.array(metadata["affine"])
    )
    return moving, fixed, arrays["amplitudes"]


def _run(moving, fixed, amplitudes, config, accuracy):
    model, backend = config["model"], config["backend"]
    reference = config["workload"].endswith("+t")
    shape = SHAPES[config["workload"]]
    if backend.startswith("fireants-"):
        output, errors, details = _run_fireants(
            moving,
            fixed,
            model,
            config["iterations"],
            backend.split("-")[1],
            accuracy,
            reference,
            amplitudes,
            batch_size=config.get("batch_size"),
        )
    else:
        result = register(
            moving,
            None if reference else fixed,
            transform=model,
            intensity_scaling="none",
            device=backend,
            max_iterations=config["iterations"],
            convergence_window_size=config["iterations"] + 1,
            return_transforms=accuracy,
            batch_size=config.get("batch_size"),
            progress=False,
        )
        output = result.registered.values
        if "time" not in moving.dims:
            output = output[None]
        details = {
            "batch_size": result.diagnostics.attrs["batch_size"],
            "updates": np.atleast_1d(result.diagnostics.n_iterations.values).tolist(),
            "statuses": np.atleast_1d(result.diagnostics.status.values).tolist(),
        }
        errors = None
        if accuracy:
            transforms = result.forward_transforms.values
            if model != "syn":
                errors = _linear_rms(
                    transforms.reshape(-1, 4, 4), amplitudes, model, shape
                )
            else:
                points = _grid(shape)
                fields = transforms.reshape(len(amplitudes), -1, 3)
                errors = [
                    _point_rms(points + field.T, points, amplitude, model, shape)
                    for field, amplitude in zip(fields, amplitudes, strict=True)
                ]
    return output, errors, details


def _check(output, moving, config, details):
    reference = config["workload"].endswith("+t")
    expected = moving.shape if reference else (1, *moving.shape)
    if output.shape != expected or output.dtype != np.float32:
        raise AssertionError("Registered output shape/dtype changed")
    if not all(np.isfinite(frame).all() for frame in output):
        raise AssertionError("Nonfinite registered output")
    if reference and not np.array_equal(output[0], moving.values[0]):
        raise AssertionError("Reference frame changed")
    if not set(details["statuses"]) <= {
        "reference",
        "converged",
        "max_iterations",
        "backend_stop",
    }:
        raise AssertionError(f"Registration failure: {details['statuses']}")
    expected_count = len(moving) if reference else 1
    if len(details["updates"]) != expected_count:
        raise AssertionError("Incomplete per-frame registration results")
