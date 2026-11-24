"""Complete resident-array registration calls and full-workload truth checks."""

import json

import numpy as np
from _benchmark_data import SHAPES, _grid, _run_fireants, _write_json, make_dataset
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
        moving, fixed, amplitudes = make_dataset(
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
    from voxeldata import create_voxeldata

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


def run_registration(moving, fixed, amplitudes, config, accuracy):
    """Run one resident registration call and optionally score the fitted transform.

    Parameters
    ----------
    moving, fixed : xarray.DataArray
        Moving data and reference grid.
    amplitudes : numpy.ndarray
        Per-frame motion amplitudes used to generate the inputs.
    config : dict
        Workload, model, backend and fitting settings.
    accuracy : bool
        Whether to export transforms and score their error against known motion.

    Returns
    -------
    output : numpy.ndarray
        Registered frames.
    errors : list of float or None
        Per-frame transform errors in voxels when scored.
    details : dict
        Fitting batch size, per-frame update counts and termination statuses.
    """
    model, backend = config["model"], config["backend"]
    reference = config["workload"].endswith("+t")
    shape = tuple(fixed.shape)
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
            moving_intensity_scaling=None,
            fixed_intensity_scaling=None,
            device=backend,
            max_iterations=config["iterations"],
            convergence_window_size=config["iterations"] + 1,
            return_transforms=accuracy,
            batch_size=config.get("batch_size"),
            progress=False,
        )
        output = result["images"].to_dataset()["registered"].values
        if "time" not in moving.dims:
            output = output[None]
        details = {
            "batch_size": result["diagnostics"].to_dataset().attrs["batch_size"],
            "updates": np.atleast_1d(
                result["diagnostics"].to_dataset().n_iterations.values
            ).tolist(),
            "statuses": np.atleast_1d(
                result["diagnostics"].to_dataset().status.values
            ).tolist(),
        }
        errors = None
        if accuracy:
            transforms = result["forward"].to_dataset()["transform"]
            if model != "syn":
                errors = _linear_rms(
                    transforms.values.reshape(-1, 4, 4), amplitudes, model, shape
                )
            else:
                points = _grid(shape)
                fields = transforms.transpose(..., "world_component").values.reshape(
                    len(amplitudes), -1, 3
                )
                errors = [
                    _point_rms(points + field.T, points, amplitude, model, shape)
                    for field, amplitude in zip(fields, amplitudes, strict=True)
                ]
    return output, errors, details


def check_registration(output, moving, config, details):
    """Validate registered pixels and per-frame diagnostics.

    Parameters
    ----------
    output : numpy.ndarray
        Registered frames.
    moving : xarray.DataArray
        Original moving image or recording.
    config : dict
        Workload settings.
    details : dict
        Diagnostics returned by the registration backend.

    Raises
    ------
    AssertionError
        If the pixels, reference frame or diagnostics violate the benchmark checks.
    """
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
