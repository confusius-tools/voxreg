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
from _benchmark_metrics import _point_rms
from _benchmark_resample_io import _load


def _settings(model, iterations):
    options = {"initial_transform": "Identity", "singleprecision": True}
    if model == "syn":
        options.update(
            type_of_transform="SyNOnly",
            reg_iterations=(iterations,),
            grad_step=0.1,
            flow_sigma=2.25,
            total_sigma=0.25,
            syn_metric="CC",
            syn_sampling=2,
        )
    else:
        options.update(
            type_of_transform={"rigid": "Rigid", "affine": "Affine"}[model],
            aff_iterations=(iterations,),
            aff_shrink_factors=(1,),
            aff_smoothing_sigmas=(0,),
            aff_metric="GC",
            aff_random_sampling_rate=1.0,
        )
    return options


def _register(arrays, metadata, config, directory):
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
    transforms = []
    for index, original in enumerate(raw):
        if metadata["recording"] and index == 0:
            output[index] = original
            transforms.append(None)
            continue
        values = original[0] if planar else original
        moving = ants.from_numpy(np.ascontiguousarray(values.T), **geometry)
        result = ants.registration(
            fixed=target,
            moving=moving,
            outprefix=str(directory / f"frame-{index:03d}_"),
            **_settings(config["model"], config["iterations"]),
        )
        sampled = result["warpedmovout"].numpy().T
        output[index] = sampled[None] if planar else sampled
        transforms.append(result["fwdtransforms"])
    return output, transforms


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
                "api": "ants.registration",
                "cli_binaries_used": False,
                "settings": _settings(config["model"], config["iterations"]),
                "threads": config["threads"],
                "image_dtype": "float32",
                "transform_precision": "float32 (ANTsPy API)",
                "updates": ([0] if reference else [])
                + [None] * (count - int(reference)),
                "statuses": (["reference"] if reference else [])
                + ["backend_stop"] * (count - int(reference)),
                "observed_updates_reason": "ANTsPy does not return iteration counts; convergence remains enabled",
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
            output, transforms = _register(arrays, metadata, config, run)
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
