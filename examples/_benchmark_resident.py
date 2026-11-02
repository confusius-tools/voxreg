"""Complete resident-array registration calls and full-workload truth checks."""

import numpy as np
from _benchmark_data import SHAPES, _grid, _run_fireants
from _benchmark_metrics import _linear_rms, _point_rms

from voxreg import register


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
            transforms = result.transforms.values
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
