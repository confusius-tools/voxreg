"""Isolated ANTsPy worker: no Torch imports, CLI calls or timed file I/O."""

import argparse
import importlib.metadata
import json
import platform
import sys
import traceback
from pathlib import Path
from time import perf_counter

import ants
import numpy as np
from _benchmark_resample_io import LIMIT, _errors, _load

PACKAGES = (
    "antspyx",
    "numpy",
    "scipy",
    "torch",
    "xarray",
    "fireants",
    "SimpleITK",
    "nibabel",
    "rich",
)


def _versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _write(path, record):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _power():
    paths = {
        "ac": "/sys/class/power_supply/ADP0/online",
        "battery": "/sys/class/power_supply/BAT0/status",
        "profile": "/sys/firmware/acpi/platform_profile",
    }
    return {
        key: Path(path).read_text().strip() if Path(path).exists() else None
        for key, path in paths.items()
    }


def _resample(arrays, metadata):
    planar = arrays["fixed"].shape[0] == 1
    axes = [2, 1] if planar else [2, 1, 0]
    affine = np.array(metadata["affine"])
    geometry = {
        "spacing": np.diag(affine)[axes].tolist(),
        "origin": affine[axes, 3].tolist(),
    }
    fixed = arrays["fixed"][0] if planar else arrays["fixed"]
    reference = ants.from_numpy(np.ascontiguousarray(fixed.T), **geometry)
    recording = metadata["recording"]
    raw = arrays["moving"] if recording else arrays["moving"][None]
    maps = arrays["transforms"] if recording else arrays["transforms"][None]
    output = np.empty_like(raw)
    masks = np.empty(raw.shape, dtype=bool)
    positions = np.indices(arrays["fixed"].shape).reshape(3, -1)
    points = affine[:3, :3] @ positions + affine[:3, 3, None]
    inverse = np.linalg.inv(affine[:3, :3])
    for index, (original, transform) in enumerate(zip(raw, maps, strict=True)):
        values = original[0] if planar else original
        moving = ants.from_numpy(np.ascontiguousarray(values.T), **geometry)
        if metadata["model"] == "syn":
            field = transform[0] if planar else transform
            vectors = field[..., axes].transpose(
                *range(len(axes) - 1, -1, -1), len(axes)
            )
            vector_image = ants.from_numpy(
                np.ascontiguousarray(vectors, dtype=np.float32),
                has_components=True,
                **geometry,
            )
            mapping = ants.transform_from_displacement_field(vector_image)
        else:
            mapping = ants.create_ants_transform(
                transform_type="AffineTransform",
                precision="float",
                dimension=len(axes),
                matrix=transform[np.ix_(axes, axes)],
                translation=transform[axes, 3],
            )
        sampled = (
            mapping.apply_to_image(moving, reference=reference, interpolation="linear")
            .numpy()
            .T
        )
        mapped = (
            points + transform.reshape(-1, 3).T
            if metadata["model"] == "syn"
            else transform[:3, :3] @ points + transform[:3, 3, None]
        )
        query = inverse @ (mapped - affine[:3, 3, None])
        valid = (
            (
                (query >= -1e-10)
                & (query <= np.array(arrays["fixed"].shape)[:, None] - 1 + 1e-10)
            )
            .all(axis=0)
            .reshape(arrays["fixed"].shape)
        )
        output[index] = sampled[None] if planar else sampled
        output[index][~valid] = original.min()
        masks[index] = valid
    return (output if recording else output[0]), (masks if recording else masks[0])


def _worker(config_path, mode):
    config = json.loads(config_path.read_text())
    path = config_path.parent / f"{mode}.json"
    record = {
        "mode": mode,
        "status": "running",
        "versions": _versions(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "power_start": _power(),
    }
    _write(path, record)
    try:
        try:
            Path("/proc/self/oom_score_adj").write_text("500")
        except OSError:
            pass
        arrays, metadata = _load(config["inputs"], config)
        record["input_sha256"] = metadata["input_sha256"]
        record["resident_input_bytes"] = sum(a.nbytes for a in arrays.values())
        record["backend_settings"] = {
            "batch_size": 1,
            "sampler": "ANTsTransform.apply_to_image",
            "image_dtype": "float32",
            "transform_precision": "float32 (ANTsPy API)",
            "boundary": "float64 physical center support computed by adapter",
            "geometry_dtype": "float64 support; float32 ANTsPy transform",
            "threads": config["threads"],
            "file_io_in_timing": False,
        }
        if mode == "memory":
            # External worker is the process-group leader; no child interpreters run.
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    record["baseline_ram_bytes"] = int(line.split()[1]) * 1024
        record["wall_seconds"] = []
        calls = config["runs"] if mode == "timing" else 1
        for index in range(calls):
            start = perf_counter()
            output, validity = _resample(arrays, metadata)
            seconds = perf_counter() - start
            if (
                output.shape != arrays["moving"].shape
                or output.dtype != arrays["moving"].dtype
                or not np.isfinite(output).all()
            ):
                raise AssertionError(
                    "Invalid ANTsPy output shape, dtype or intensities"
                )
            if mode == "timing":
                record["wall_seconds"].append(seconds)
                _write(path, record)
                print(
                    f"Completed timed resampling {index + 1}/{calls}: {seconds:.6f} s",
                    flush=True,
                )
            if mode == "accuracy":
                errors = _errors(output, arrays, metadata, validity)
                record.update(
                    relative_rmse_per_frame=errors,
                    worst_frame_relative_rmse=max(errors),
                    accuracy_pass=max(errors) < LIMIT,
                    support_check="exact physical center mask (adapter)",
                )
            del output, validity
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
        print(f"{mode}: {record['status']}", flush=True)
    return 0


def _main(worker=_worker):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--mode", choices=("timing", "memory", "accuracy"))
    args = parser.parse_args()
    if args.versions:
        print(json.dumps(_versions()))
    elif args.worker and args.mode:
        return worker(args.worker, args.mode)
    else:
        parser.error("--worker and --mode are required")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
