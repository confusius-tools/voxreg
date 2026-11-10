"""Known physical pulls, backend adapters and the resampling phase worker."""

import hashlib
import json
import traceback
from pathlib import Path
from time import perf_counter

import _benchmark_data as data
import numpy as np
import torch
import xarray as xr
from _benchmark_resample_io import _digest, _load
from _benchmark_resample_io import _errors as _array_errors
from _benchmark_runner import RESAMPLE_LIMIT, _group_rss, _versions, _write
from torch.nn import functional

from voxreg import position_to_world, resample


def _prepare(config):
    shape = data.SHAPES[config["workload"]]
    moving, fixed, amplitudes = data._dataset(
        shape, config["frames"], config["model"], config["workload"].endswith("+t")
    )
    recording = "time" in moving.dims
    if config["model"] == "syn":
        points = data._grid(shape)
        values = np.empty((len(amplitudes), *shape, 3), dtype=np.float64)
        cache = {}
        for index, amplitude in enumerate(amplitudes):
            key = index % 16 if index else -1
            if key not in cache:
                cache[key] = (
                    data._truth(points, amplitude, "syn", shape) - points
                ).T.reshape(*shape, 3)
            values[index] = cache[key]
        transforms = xr.DataArray(
            values if recording else values[0],
            dims=(*moving.dims, "world_component"),
            coords=moving.coords,
            attrs={
                "representation": "displacement",
                "spatial_units": "mm",
                "convention": "fixed-world-to-moving-world pull",
            },
        ).assign_coords(world_component=["z", "y", "x"])
    else:
        values = np.stack([data._matrix(config["model"], a, shape) for a in amplitudes])
        transforms = xr.DataArray(
            values if recording else values[0],
            dims=(("time",) if recording else ()) + ("world_out", "world_in"),
            coords={
                "world_out": ["z", "y", "x", "1"],
                "world_in": ["z", "y", "x", "1"],
                **({"time": moving.time} if recording else {}),
            },
            attrs={
                "spatial_units": "mm",
                "convention": "fixed-world-to-moving-world pull",
            },
        )
    digest = hashlib.sha256(memoryview(moving.values))
    digest.update(memoryview(transforms.values))
    return moving, fixed, transforms, amplitudes, digest.hexdigest()


def _prepare_inputs(config_path):
    """Stage identical arrays once, outside every backend's measured workers."""
    config = json.loads(config_path.read_text())
    if config["backend"].endswith("cuda") and not torch.cuda.is_available():
        return
    directory = (
        config_path.parent.parent / "inputs" / f"{config['workload']}-{config['model']}"
    )
    if not (directory / "inputs.json").exists():
        moving, fixed, transforms, _, _ = _prepare(config)
        arrays = {
            "moving": moving.values,
            "fixed": fixed.values,
            "transforms": transforms.values,
            "time": moving.time.values if "time" in moving.dims else np.array([0]),
        }
        metadata = {
            "operation": "resampling",
            "recording": "time" in moving.dims,
            "model": config["model"],
            "workload": config["workload"],
            "affine": position_to_world(fixed).tolist(),
        }
        metadata["input_sha256"] = _digest(arrays, metadata)
        directory.mkdir(parents=True)
        for name, values in arrays.items():
            np.save(directory / f"{name}.npy", values, allow_pickle=False)
        _write(directory / "inputs.json", metadata)
    config["inputs"] = str(directory)
    _write(config_path, config)


def _voxel_inputs(arrays, metadata):
    from voxeldata import create_voxeldata

    dims = ("k", "j", "i")
    moving = create_voxeldata(
        arrays["moving"],
        dims=(("time",) if metadata["recording"] else ()) + dims,
        voxel_to_world=np.array(metadata["affine"]),
        **({"time": arrays["time"]} if metadata["recording"] else {}),
    )
    fixed = create_voxeldata(
        arrays["fixed"], dims=dims, voxel_to_world=np.array(metadata["affine"])
    )
    if metadata["model"] == "syn":
        transforms = xr.DataArray(
            arrays["transforms"],
            dims=(*moving.dims, "world_component"),
            coords=moving.coords,
        ).assign_coords(world_component=["z", "y", "x"])
        transforms.attrs = {"representation": "displacement", "spatial_units": "mm"}
    else:
        transforms = xr.DataArray(
            arrays["transforms"],
            dims=(("time",) if metadata["recording"] else ())
            + ("world_out", "world_in"),
            coords={
                "world_out": ["z", "y", "x", "1"],
                "world_in": ["z", "y", "x", "1"],
                **({"time": arrays["time"]} if metadata["recording"] else {}),
            },
        )
    return moving, fixed, transforms


@torch.no_grad()
def _resident(moving, fixed, transforms, config):
    backend = config["backend"]
    device = "cuda" if backend.endswith("cuda") else "cpu"
    from voxreg.preparation import choose_batch_size

    series = moving if "time" in moving.dims else moving.expand_dims(time=[0])
    batch = min(
        series.sizes["time"],
        config["batch_size"]
        or choose_batch_size(series, fixed, config["model"], device),
    )
    if not backend.startswith("fireants"):
        output, valid = resample(
            moving, fixed, transforms, device=device, batch_size=batch
        )
        return (
            output.values,
            valid.values,
            {
                "batch_size": batch,
                "sampler": "voxreg.resample",
                "image_dtype": "float32",
                "geometry_dtype": "float64",
            },
        )

    from fireants.interpolator import FFO_AVAILABLE, USE_FFO, fireants_interpolator

    planar = moving.sizes["k"] == 1
    axes = [2, 1] if planar else [2, 1, 0]
    shape = tuple(fixed.shape)
    extent = ((np.array(shape) - 1) * data.SPACING / 2)[axes]
    raw = moving.values if "time" in moving.dims else moving.values[None]
    maps = transforms.values if "time" in transforms.dims else transforms.values[None]
    output = np.empty_like(raw)
    masks = np.empty(raw.shape, dtype=bool)
    identity = torch.eye(len(axes), len(axes) + 1, device=device, dtype=torch.float64)
    tolerance = torch.tensor(2e-10 / (np.array(shape)[axes] - 1), device=device)
    for first in range(0, len(raw), batch):
        last = min(first + batch, len(raw))
        images = torch.tensor(
            raw[first:last, 0] if planar else raw[first:last], device=device
        )
        field = None
        if config["model"] == "syn":
            values = maps[first:last, 0] if planar else maps[first:last]
            field = torch.tensor(
                values[..., axes] / extent, device=device, dtype=torch.float64
            )
            affine = identity[None].expand(last - first, -1, -1)
        else:
            matrices = maps[first:last]
            linear = (
                matrices[:, axes][:, :, axes]
                * extent[None, None, :]
                / extent[None, :, None]
            )
            shift = matrices[:, axes, 3] / extent
            affine = torch.tensor(
                np.concatenate((linear, shift[..., None]), axis=-1),
                device=device,
                dtype=torch.float64,
            )
        query = functional.affine_grid(
            affine, (last - first, 1, *images.shape[1:]), align_corners=True
        )
        if field is not None:
            query += field
        # Support belongs to the physical map, not its rounded sampler coordinates.
        valid = (query.abs() <= 1 + tolerance).all(dim=-1)
        sampled = fireants_interpolator(
            images[:, None],
            grid=query.float(),
            is_displacement=False,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[:, 0]
        sampled = torch.where(
            valid,
            sampled,
            images.flatten(1).amin(1).reshape(-1, *([1] * (images.ndim - 1))),
        )
        output[first:last] = (
            sampled.cpu().numpy()[:, None] if planar else sampled.cpu().numpy()
        )
        masks[first:last] = (
            valid.cpu().numpy()[:, None] if planar else valid.cpu().numpy()
        )
    return (
        (output if "time" in moving.dims else output[0]),
        (masks if "time" in moving.dims else masks[0]),
        {
            "batch_size": batch,
            "sampler": "FireANTs interpolator (adapter includes map conversion, validity and CPU return)",
            "image_dtype": "float32",
            "geometry_dtype": "float64 support; float32 sampling coordinates",
            "fused_ops_available": FFO_AVAILABLE,
            "fused_interpolator_enabled": USE_FFO
            and FFO_AVAILABLE
            and device == "cuda",
        },
    )


def _worker(config_path, mode):
    config = json.loads(config_path.read_text())
    directory = config_path.parent
    record = {
        "mode": mode,
        "status": "running",
        "versions": _versions(),
        "power_start": data._power(),
    }
    path = directory / f"{mode}.json"
    _write(path, record)
    print(f"{config['id']} / {mode}; config: {config_path}; result: {path}", flush=True)
    try:
        try:
            Path("/proc/self/oom_score_adj").write_text("500")
        except OSError:
            pass
        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)
        cuda = config["backend"].endswith("cuda")
        if cuda and not torch.cuda.is_available():
            record.update(status="unavailable", reason="CUDA unavailable")
            return 0
        arrays, metadata = _load(config["inputs"], config)
        moving, fixed, transforms = _voxel_inputs(arrays, metadata)
        record["input_sha256"] = metadata["input_sha256"]
        record["resident_input_bytes"] = sum(a.nbytes for a in arrays.values())
        if config["backend"].startswith("fireants"):
            import fireants.interpolator  # noqa: F401 -- exclude import/dispatcher setup from timing.
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.synchronize()
            record["cuda_device"] = {
                "name": torch.cuda.get_device_name(),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            }
            if mode == "memory":
                torch.cuda.reset_peak_memory_stats()
        if mode == "memory":
            import os

            record["baseline_ram_bytes"] = _group_rss(os.getpgrp())
        record["wall_seconds"] = []
        calls = config["runs"] if mode == "timing" else 1
        for index in range(calls):
            if cuda:
                torch.cuda.synchronize()
            start = perf_counter()
            output, validity, details = _resident(moving, fixed, transforms, config)
            if cuda:
                torch.cuda.synchronize()
            seconds = perf_counter() - start
            if (
                record.get("backend_settings", details)["batch_size"]
                != details["batch_size"]
            ):
                raise RuntimeError("Automatic batch size changed between calls")
            record["backend_settings"] = details
            if (
                output.shape != moving.shape
                or output.dtype != moving.dtype
                or not np.isfinite(output).all()
            ):
                raise AssertionError(
                    f"Invalid output: {output.shape}/{output.dtype}; expected {moving.shape}/{moving.dtype}; finite={np.isfinite(output).all()}"
                )
            if mode == "timing":
                record["wall_seconds"].append(seconds)
                print(
                    f"Completed timed resampling {index + 1}/{calls}: {seconds:.6f} s",
                    flush=True,
                )
                _write(path, record)
            if mode == "accuracy":
                errors = _array_errors(output, arrays, metadata, validity)
                record["support_check"] = "exact physical center mask"
                expected = len(arrays["time"]) if metadata["recording"] else 1
                if len(errors) != expected or not np.isfinite(errors).all():
                    raise AssertionError("Missing/nonfinite interpolation errors")
                record.update(
                    relative_rmse_per_frame=errors,
                    worst_frame_relative_rmse=max(errors),
                    accuracy_pass=max(errors) < RESAMPLE_LIMIT,
                )
                print(
                    f"Worst-frame relative RMS: {max(errors):.9g}; required < {RESAMPLE_LIMIT}",
                    flush=True,
                )
            del output, validity
        if mode == "memory" and cuda:
            record["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        record["status"] = (
            "quality_failed"
            if mode == "accuracy" and not record["accuracy_pass"]
            else "passed"
        )
    except Exception:
        record.update(status="error", error=traceback.format_exc())
        raise
    finally:
        record["power_end"] = data._power()
        _write(path, record)
        print(f"{mode}: {record['status']}", flush=True)
    return 0
