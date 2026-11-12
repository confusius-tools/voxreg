"""Isolated ANTsPy registration: Python API only, no Torch or CLI executables."""

import platform
import shutil
import sys
import traceback
from pathlib import Path
from time import perf_counter

import ants
import numpy as np
import pandas as pd
from _benchmark_antspy import _main, _power, _versions, _write
from _benchmark_data import _grid
from _benchmark_iterations import run_ants_registration
from _benchmark_metrics import _point_rms
from _benchmark_resample_io import _load
from ants.internal import get_pointer_string


def _settings(model, iterations):
    return {
        "initial_transform": "Identity",
        "singleprecision": True,
        "transform": {
            "rigid": "Rigid[0.25]",
            "affine": "Affine[0.25]",
            "syn": "SyN[0.2,2.25,0.25]",
        }[model],
        "metric": "CC" if model == "syn" else "GC",
        "metric_parameters": "2" if model == "syn" else "32,regular,1.0",
        # A window larger than the budget cannot trigger the native stopping test.
        "convergence": f"[{iterations},{'1e-7' if model == 'syn' else '1e-6'},{iterations + 1}]",
        "shrink_factors": "1",
        "smoothing_sigmas": "0",
    }


def _register(arrays, metadata, config, directory, *, verify_iterations=False):
    planar = arrays["fixed"].shape[0] == 1
    axes = [2, 1] if planar else [2, 1, 0]
    affine = np.array(metadata["affine"])
    geometry = {
        "spacing": np.diag(affine)[axes].tolist(),
        "origin": affine[axes, 3].tolist(),
    }
    fixed = arrays["fixed"][0] if planar else arrays["fixed"]
    target = ants.from_numpy(np.ascontiguousarray(fixed.T), **geometry)
    raw = arrays["moving"] if metadata["recording"] else arrays["moving"][None]
    output = np.empty_like(raw)
    transforms, updates = [], []
    for index, original in enumerate(raw):
        if metadata["recording"] and index == 0:
            output[index] = original
            transforms.append(None)
            updates.append(0)
            continue
        values = original[0] if planar else original
        moving = ants.from_numpy(np.ascontiguousarray(values.T), **geometry)
        fixed_image, moving_image = target.clone("float"), moving.clone("float")
        warped, reverse = fixed_image.clone(), moving_image.clone()
        prefix = directory / f"frame-{index:03d}_"
        settings = _settings(config["model"], config["iterations"])
        metric = (
            f"{settings['metric']}[{get_pointer_string(fixed_image)},"
            f"{get_pointer_string(moving_image)},1,{settings['metric_parameters']}]"
        )
        arguments = [
            "--dimensionality",
            str(target.dimension),
            "--initial-moving-transform",
            settings["initial_transform"],
            "--metric",
            metric,
            "--transform",
            settings["transform"],
            "--convergence",
            settings["convergence"],
            "--smoothing-sigmas",
            settings["smoothing_sigmas"],
            "--shrink-factors",
            settings["shrink_factors"],
            "--use-histogram-matching",
            "0",
            "--collapse-output-transforms",
            "1",
            "--output",
            f"[{prefix},{get_pointer_string(warped)},{get_pointer_string(reverse)}]",
            "--masks",
            "[NA,NA]",
            "--float",
            "1",
            "--write-composite-transform",
            "0",
        ]
        updates.append(
            run_ants_registration(
                arguments,
                config["iterations"],
                prefix.with_name(prefix.name + "iterations.log")
                if verify_iterations
                else None,
            )
        )
        sampled = warped.numpy().T
        output[index] = sampled[None] if planar else sampled
        paths = [
            str(path)
            for path in sorted(directory.glob(prefix.name + "[0-9]*"), reverse=True)
            if path.name.endswith("GenericAffine.mat")
            or (
                path.name.endswith("Warp.nii.gz")
                and not path.name.endswith("InverseWarp.nii.gz")
            )
        ]
        required = "Warp.nii.gz" if config["model"] == "syn" else "GenericAffine.mat"
        if not any(path.endswith(required) for path in paths):
            raise RuntimeError(
                f"ANTsPy did not write the expected {required} transform"
            )
        transforms.append(paths)
    return output, transforms, updates


def _errors(transforms, arrays, metadata):
    shape = arrays["fixed"].shape
    points = _grid(shape)
    axes = [2, 1] if shape[0] == 1 else [2, 1, 0]
    columns = ["x", "y", "z"][: len(axes)]
    queries = pd.DataFrame(points[axes].T, columns=columns)
    errors = []
    for paths, amplitude in zip(transforms, arrays["amplitudes"], strict=True):
        if paths is None:
            if not metadata["recording"] or errors or amplitude != 0:
                raise AssertionError(
                    "Only the unfitted identity reference may omit a transform"
                )
            errors.append(0.0)
            continue
        mapped = points.copy()
        mapped[axes] = (
            ants.apply_transforms_to_points(
                len(axes), queries, paths, whichtoinvert=[False] * len(paths)
            )[columns]
            .to_numpy()
            .T
        )
        errors.append(_point_rms(mapped, points, amplitude, metadata["model"], shape))
    if len(errors) != len(arrays["amplitudes"]) or not np.isfinite(errors).all():
        raise AssertionError("Missing/nonfinite full-grid registration truth errors")
    return errors


def _worker(config_path, mode):
    import json

    config = json.loads(config_path.read_text())
    directory = config_path.parent
    path = directory / f"{mode}.json"
    record = {
        "mode": mode,
        "status": "running",
        "versions": _versions(),
        "executable": sys.executable,
        "python": platform.python_version(),
        "power_start": _power(),
    }
    _write(path, record)
    try:
        try:
            Path("/proc/self/oom_score_adj").write_text("500")
        except OSError:
            pass
        arrays, metadata = _load(config["inputs"], config)
        count = len(arrays["amplitudes"])
        reference = metadata["recording"]
        record.update(
            input_sha256=metadata["input_sha256"],
            resident_input_bytes=sum(a.nbytes for a in arrays.values()),
            backend_settings={
                "batch_size": 1,
                "api": "ants.registration argument-list API; no external executable",
                "cli_binaries_used": False,
                "settings": _settings(config["model"], config["iterations"]),
                "threads": config["threads"],
                "image_dtype": "float32",
                "transform_precision": "float32 (ANTsPy API)",
                "updates": ([0] if reference else [])
                + [None] * (count - int(reference)),
                "statuses": (["reference"] if reference else [])
                + ["backend_stop"] * (count - int(reference)),
                "requested_updates": ([0] if reference else [])
                + [config["iterations"]] * (count - int(reference)),
                "stopping_policy": "Convergence window exceeds the update budget; accuracy logs must verify every frame's fixed count",
                "observed_updates_reason": "Native logs verify accuracy-phase updates; timing/memory omit verbose count instrumentation",
                "file_io_in_timing": "ANTsPy transform-file writes included; input staging, scoring and cleanup excluded",
            },
            wall_seconds=[],
        )
        if mode == "memory":
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    record["baseline_ram_bytes"] = int(line.split()[1]) * 1024
        for index in range(config["runs"] if mode == "timing" else 1):
            run = directory / f"{mode}-{index + 1}"
            run.mkdir()
            start = perf_counter()
            output, transforms, updates = _register(
                arrays, metadata, config, run, verify_iterations=mode == "accuracy"
            )
            elapsed = perf_counter() - start
            expected = (
                arrays["moving"].shape if reference else (1, *arrays["moving"].shape)
            )
            if (
                output.shape != expected
                or output.dtype != np.float32
                or not np.isfinite(output).all()
            ):
                raise AssertionError(
                    "Invalid ANTsPy registered shape, dtype or intensities"
                )
            if reference and not np.array_equal(output[0], arrays["moving"][0]):
                raise AssertionError("ANTsPy changed the unfitted reference frame")
            if mode == "timing":
                record["wall_seconds"].append(elapsed)
                print(
                    f"Completed timed registration {index + 1}/{config['runs']}: {elapsed:.6f} s",
                    flush=True,
                )
            if mode == "accuracy":
                record["backend_settings"]["updates"] = updates
                record["backend_settings"]["statuses"] = (
                    ["reference"] if reference else []
                ) + ["max_iterations"] * (count - int(reference))
                errors = _errors(transforms, arrays, metadata)
                worst = max(errors[1:] if reference else errors)
                limit = config["truth_rms_limit_voxels"]
                record.update(
                    rms_per_frame_voxels=errors,
                    worst_frame_rms_voxels=worst,
                    truth_rms_limit_voxels=limit,
                    truth_rms_comparison="<=",
                    accuracy_pass=worst <= limit,
                )
            _write(path, record)
            if mode != "accuracy":
                shutil.rmtree(run)
            del output, transforms
        record["status"] = (
            "quality_failed"
            if mode == "accuracy" and not record["accuracy_pass"]
            else "passed"
        )
    except Exception:
        record.update(status="error", error=traceback.format_exc())
        raise
    finally:
        record["power_end"] = _power()
        _write(path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(_worker))
