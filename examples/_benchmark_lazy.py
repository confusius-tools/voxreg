"""Disk-backed synthetic lazy benchmarks with isolated measurement phases."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import dask
import numpy as np
import psutil
import torch
import xarray as xr
import zarr
from distributed import Client, LocalCluster, get_task_stream
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
"""Live or frozen source root used by isolated benchmark workers."""
sys.path.insert(0, str(ROOT / "src"))

from confusius import create_voxeldata
from confusius.io import load

from voxreg import register, resample
from voxreg.preparation import choose_batch_size

PROTOCOL = "voxreg-lazy-benchmark-v1"
"""Streaming protocol, separate from resident cross-backend comparisons."""
SPACING = np.array([0.2, 0.1, 0.1])
"""Synthetic world z/y/x voxel spacing in millimetres."""


def write_json(path: Path, value: Any) -> None:
    """Atomically write a result without nonfinite JSON numbers.

    Parameters
    ----------
    path : pathlib.Path
        Result path.
    value : object
        JSON-safe result.

    Returns
    -------
    None
        The result is written to disk.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def truth_matrix(frame: int, shape: tuple[int, ...]) -> np.ndarray:
    """Define a known rigid pull independent of registration code.

    Parameters
    ----------
    frame : int
        Absolute recording frame, with frame zero exactly identity.
    shape : tuple of int
        Native spatial shape.

    Returns
    -------
    numpy.ndarray
        Homogeneous fixed-world-to-moving-world pull in z/y/x mm.
    """
    matrix = np.eye(4)
    if frame:
        amplitude = 0.8 + 0.2 * np.sin(2 * np.pi * frame / 16)
        vector = np.array([0.012, 0, 0] if shape[0] == 1 else [0.008, -0.01, 0.006])
        matrix[:3, :3] = Rotation.from_rotvec(amplitude * vector).as_matrix()
        matrix[:3, 3] = amplitude * SPACING * [0 if shape[0] == 1 else 0.8, -1.2, 1.5]
    return matrix


def phantom(shape: tuple[int, ...], affine: np.ndarray, pull: np.ndarray) -> np.ndarray:
    """Sample seeded analytic texture under a prescribed physical motion.

    Parameters
    ----------
    shape : tuple of int
        Spatial volume shape.
    affine : numpy.ndarray
        Native index-to-world map.
    pull : numpy.ndarray
        Known fixed-to-moving world transform.

    Returns
    -------
    numpy.ndarray
        Original float32 image, not an interpolation of another volume.
    """
    points = affine @ np.vstack(
        (np.indices(shape).reshape(3, -1), np.ones((1, math.prod(shape))))
    )
    points = (np.linalg.inv(pull) @ points)[:3]
    extent = np.maximum((np.array(shape) - 1) * SPACING, SPACING)
    axes = range(1 if shape[0] == 1 else 0, 3)
    rng = np.random.default_rng(871)
    values = np.full(points.shape[1], 0.05)
    for _ in range(16):
        center = rng.uniform(-0.35, 0.35, 3) * extent
        width = np.maximum(rng.uniform(0.04, 0.09, 3) * extent, 1.5 * SPACING)
        distance = sum(
            ((points[axis] - center[axis]) / width[axis]) ** 2 for axis in axes
        )
        values += rng.uniform(0.5, 2) * np.exp(-0.5 * distance)
    return values.reshape(shape).astype(np.float32)


def stage_inputs(
    directory: Path, shape: tuple[int, ...], frames: int
) -> dict[str, Any]:
    """Write real uncompressed Zarr files with bounded synthetic preparation.

    Parameters
    ----------
    directory : pathlib.Path
        Fresh input directory.
    shape : tuple of int
        Native volume shape.
    frames : int
        Number of independently fitted recording frames.

    Returns
    -------
    dict
        Affine, byte counts, payload digest and input provenance.
    """
    directory.mkdir()
    affine = np.diag([*SPACING, 1.0])
    affine[:3, 3] = -(np.array(shape) - 1) * SPACING / 2
    group = zarr.open_group(directory / "recording.zarr", mode="w", zarr_format=2)
    image = group.create_array(
        "image",
        shape=(frames, *shape),
        chunks=(1, *shape),
        dtype="float32",
        compressor=None,
        fill_value=None,
    )
    image.attrs.update(
        _ARRAY_DIMENSIONS=["time", "k", "j", "i"],
        voxel_to_world=affine.tolist(),
        voxel_to_world_units="mm",
        units="synthetic power",
    )
    matrices = group.create_array(
        "matrix",
        shape=(frames, 4, 4),
        chunks=(1, 4, 4),
        dtype="float64",
        compressor=None,
        fill_value=None,
    )
    matrices.attrs.update(
        _ARRAY_DIMENSIONS=["time", "world_out", "world_in"],
        spatial_units="mm",
        convention="fixed-world-to-moving-world pull",
    )
    for dim, values in (
        ("time", np.arange(frames) * 0.1),
        ("k", np.arange(shape[0])),
        ("j", np.arange(shape[1])),
        ("i", np.arange(shape[2])),
        ("world_out", np.array(["z", "y", "x", "1"])),
        ("world_in", np.array(["z", "y", "x", "1"])),
    ):
        coordinate = group.create_array(
            dim, data=values, compressor=None, fill_value=None
        )
        coordinate.attrs["_ARRAY_DIMENSIONS"] = [dim]
    digest = hashlib.sha256()
    cache: dict[int, np.ndarray] = {}
    for frame in range(frames):
        key = frame % 16 if frame else -1
        matrix = truth_matrix(frame, shape)
        if key not in cache:
            cache[key] = phantom(shape, affine, matrix)
        values = cache[key]
        image[frame] = values
        matrices[frame] = matrix
        digest.update(values.tobytes())
        digest.update(matrix.tobytes())
    fixed = create_voxeldata(cache[-1], dims=("k", "j", "i"), voxel_to_world=affine)
    fixed.drop_vars(["z", "y", "x"]).assign_attrs(
        voxel_to_world=affine.tolist(), voxel_to_world_units="mm"
    ).to_dataset(name="image").to_zarr(
        directory / "fixed.zarr",
        zarr_format=2,
        consolidated=False,
        encoding={"image": {"compressors": None}},
    )
    metadata = {
        "shape": list(shape),
        "frames": frames,
        "affine": affine.tolist(),
        "spacing_mm": SPACING.tolist(),
        "dtype": "float32",
        "image_bytes": frames * math.prod(shape) * 4,
        "storage_chunk_frames": 1,
        "compression": "none",
        "payload_sha256": digest.hexdigest(),
        "pose_cycle": "16 poses after identity frame zero; every frame is processed",
        "seed": 871,
        "maximum_preparation_cached_volumes": 17,
    }
    metadata["stored_input_bytes"] = sum(
        path.stat().st_size for path in directory.rglob("*") if path.is_file()
    )
    metadata["allocated_disk_bytes"] = sum(
        getattr(path.stat(), "st_blocks", 0) * 512
        for path in directory.rglob("*")
        if path.is_file()
    )
    write_json(directory / "manifest.json", metadata)
    return metadata


def configure_worker(config: dict[str, Any]) -> dict[str, Any]:
    """Initialize device/threads and choose a pinned fitting batch outside timing.

    Parameters
    ----------
    config : dict
        Frozen case settings.

    Returns
    -------
    dict
        Worker device information and candidate fitting batch.
    """
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    if config["device"] == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    moving = load(
        Path(config["inputs"]) / "recording.zarr",
        variable="image",
        chunks={"time": config["chunk_frames"]},
        consolidated=False,
    )
    fixed = load(
        Path(config["inputs"]) / "fixed.zarr", variable="image", consolidated=False
    )
    batch = (
        config.get("effective_batch_size")
        or config["batch_size"]
        or choose_batch_size(moving, fixed, "rigid", config["device"])
    )
    result = {
        "batch_size": min(batch, config["chunk_frames"]),
        "threads": torch.get_num_threads(),
        "device": config["device"],
    }
    if config["device"] == "cuda":
        props = torch.cuda.get_device_properties(0)
        result.update(
            gpu_uuid=str(props.uuid),
            gpu_name=props.name,
            total_vram_bytes=props.total_memory,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        )
    return result


def gpu_peaks() -> dict[str, Any]:
    """Read each worker's allocated/reserved peaks after the memory phase.

    Returns
    -------
    dict
        CUDA process peaks, or no CUDA allocation for CPU execution.
    """
    if not torch.cuda.is_initialized():
        return {}
    return {
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def storage_view(image: xr.DataArray) -> xr.DataArray:
    """Store a spatial array without duplicating dense world coordinates.

    Parameters
    ----------
    image : xarray.DataArray
        VoxelData image or mask.

    Returns
    -------
    xarray.DataArray
        Zarr-ready values with affine metadata for ConfUSIus loading.
    """
    index = image.xindexes["z"]
    return image.drop_vars(["z", "y", "x"]).assign_attrs(
        voxel_to_world=index.voxel_to_world.tolist(), voxel_to_world_units=index.units
    )


def execute(config: dict[str, Any], output: Path) -> None:
    """Read, construct, execute and store the complete lazy operation.

    Parameters
    ----------
    config : dict
        Frozen settings including the pinned fitting batch.
    output : pathlib.Path
        Fresh Zarr result directory.

    Returns
    -------
    None
        A single DataTree write completes all related outputs jointly.
    """
    inputs = Path(config["inputs"])
    moving = load(
        inputs / "recording.zarr",
        variable="image",
        chunks={"time": config["chunk_frames"]},
        consolidated=False,
    )
    fixed = load(inputs / "fixed.zarr", variable="image", consolidated=False)
    options = {"device": config["device"], "batch_size": config["effective_batch_size"]}
    if config["operation"] == "register":
        result = register(
            moving,
            fixed,
            transform="rigid",
            max_iterations=config["iterations"],
            convergence_window_size=config["iterations"] + 1,
            return_transforms=True,
            progress=False,
            **options,
        )
        assert (
            result.forward_transforms is not None
            and result.inverse_transforms is not None
        )
        tree = {
            "images": storage_view(result.registered).to_dataset(name="image"),
            "validity": storage_view(result.valid).to_dataset(name="valid"),
            "diagnostics": result.diagnostics,
            "forward": result.forward_transforms.to_dataset(name="matrix"),
            "inverse": result.inverse_transforms.to_dataset(name="matrix"),
        }
    else:
        maps = xr.open_zarr(
            inputs / "recording.zarr",
            chunks={"time": config["chunk_frames"]},
            consolidated=False,
        ).matrix
        image, valid = resample(moving, fixed, maps, **options)
        tree = {
            "images": storage_view(image).to_dataset(name="image"),
            "validity": storage_view(valid).to_dataset(name="valid"),
        }
    encoding = {
        f"/{group}": {name: {"compressors": None} for name in dataset.variables}
        for group, dataset in tree.items()
    }
    xr.DataTree.from_dict(tree).to_zarr(
        output, compute=False, zarr_format=2, consolidated=False, encoding=encoding
    ).compute()


def verify(config: dict[str, Any], output: Path, directory: Path) -> dict[str, Any]:
    """Stream every frame against physical truth and an independent SciPy pull.

    Parameters
    ----------
    config : dict
        Frozen case settings.
    output : pathlib.Path
        Completed Zarr output.
    directory : pathlib.Path
        Place for bounded failure evidence.

    Returns
    -------
    dict
        Full-workload accuracy, support and eager spot-check results.

    Raises
    ------
    AssertionError
        If stored geometry, labels, frame counts or input hashes disagree.
    """
    inputs = Path(config["inputs"])
    metadata = json.loads((inputs / "manifest.json").read_text())
    shape, frames = tuple(metadata["shape"]), metadata["frames"]
    affine = np.array(metadata["affine"])
    points = affine @ np.vstack(
        (np.indices(shape).reshape(3, -1), np.ones((1, math.prod(shape))))
    )
    source = zarr.open_group(inputs / "recording.zarr", mode="r")
    saved = zarr.open_group(output, mode="r")
    image, valid = saved["images/image"], saved["validity/valid"]
    assert image.shape == valid.shape == (frames, *shape)
    assert image.dtype == np.dtype("float32") and valid.dtype == np.dtype(bool)
    assert np.array_equal(np.array(image.attrs["voxel_to_world"]), affine)
    assert image.attrs["voxel_to_world_units"] == "mm"
    assert np.array_equal(np.array(valid.attrs["voxel_to_world"]), affine)
    assert valid.attrs["voxel_to_world_units"] == "mm"
    for group in ("images", "validity"):
        assert np.array_equal(saved[f"{group}/time"][:], source["time"][:])
        for axis, size in zip(("k", "j", "i"), shape, strict=True):
            assert np.array_equal(saved[f"{group}/{axis}"][:], np.arange(size))
    if config["operation"] == "register":
        for group, convention in (
            ("forward", "fixed-world-to-moving-world pull"),
            ("inverse", "moving-world-to-fixed-world pull"),
        ):
            assert saved[f"{group}/matrix"].attrs["spatial_units"] == "mm"
            assert saved[f"{group}/matrix"].attrs["convention"] == convention
            assert np.array_equal(saved[f"{group}/time"][:], source["time"][:])
    digest = hashlib.sha256()
    worst_truth = worst_intensity = 0.0
    support_mismatches = 0
    invalid_statuses = 0
    observed_batches: set[int] = set()
    observed_tasks: set[int] = set()
    samples = sorted({0, frames // 2, frames - 1})
    eager_worst = 0.0
    fixed = load(inputs / "fixed.zarr", variable="image", consolidated=False).compute()
    inverse_affine = np.linalg.inv(affine)
    for frame in range(frames):
        raw = source["image"][frame]
        truth = source["matrix"][frame]
        digest.update(raw.tobytes())
        digest.update(truth.tobytes())
        fitted = (
            saved["forward/matrix"][frame]
            if config["operation"] == "register"
            else truth
        )
        assert np.isfinite(raw).all() and np.isfinite(fitted).all(), (
            "Source intensities and fitted maps must be finite"
        )
        if config["operation"] == "register":
            status = str(saved["diagnostics/status"][frame])
            invalid_statuses += status != "max_iterations"
            invalid_statuses += (
                int(saved["diagnostics/n_iterations"][frame]) != config["iterations"]
            )
            observed_batches.add(int(saved["diagnostics/execution_batch_size"][frame]))
            observed_tasks.add(int(saved["diagnostics/execution_task_size"][frame]))
            errors = (
                np.linalg.norm(((fitted - truth) @ points)[:3], axis=0) / SPACING.min()
            )
            worst_truth = max(worst_truth, float(np.sqrt(np.mean(errors**2))))
            if not np.allclose(
                saved["inverse/matrix"][frame] @ fitted, np.eye(4), atol=1e-10, rtol=0
            ):
                invalid_statuses += 1
        coordinates = (inverse_affine @ fitted @ points)[:3]
        tolerance = 1e-5 if config["operation"] == "register" else 1e-10
        support = np.ones(math.prod(shape), dtype=bool)
        for axis, size in enumerate(shape):
            support &= (
                (coordinates[axis] >= -0.5) & (coordinates[axis] < 0.5)
                if size == 1
                else (coordinates[axis] >= -tolerance)
                & (coordinates[axis] <= size - 1 + tolerance)
            )
        expected = map_coordinates(
            raw.astype(np.float64),
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
        ).reshape(shape)
        support = support.reshape(shape)
        expected[~support] = raw.min()
        actual, actual_valid = image[frame], valid[frame]
        assert np.isfinite(actual).all(), "Stored intensities must be finite"
        mismatches = int(np.count_nonzero(actual_valid != support))
        support_mismatches += mismatches
        relative = float(
            np.sqrt(np.mean((actual.astype(np.float64) - expected) ** 2))
            / max(np.sqrt(np.mean(expected**2)), np.finfo(float).tiny)
        )
        worst_intensity = max(worst_intensity, relative)
        if (mismatches or relative >= 1e-5 or worst_truth > 0.30) and not (
            directory / "failure.npz"
        ).exists():
            np.savez(
                directory / "failure.npz",
                frame=frame,
                raw=raw,
                actual=actual,
                expected=expected,
                valid=actual_valid,
                expected_valid=support,
                fitted=fitted,
                truth=truth,
            )
        if frame in samples:
            native = create_voxeldata(raw, dims=("k", "j", "i"), voxel_to_world=affine)
            if config["operation"] == "register":
                eager = register(
                    native,
                    fixed,
                    max_iterations=config["iterations"],
                    convergence_window_size=config["iterations"] + 1,
                    device=config["device"],
                    progress=False,
                    return_transforms=True,
                )
                assert eager.transforms is not None
                delta = (
                    np.linalg.norm(
                        ((eager.transforms.values - fitted) @ points)[:3], axis=0
                    ).max()
                    / SPACING.min()
                )
                eager_worst = max(eager_worst, float(delta))
            else:
                eager_image, eager_valid = resample(
                    native, fixed, truth, device=config["device"]
                )
                assert np.array_equal(eager_valid, support)
                eager_worst = max(
                    eager_worst, float(np.max(np.abs(eager_image.values - actual)))
                )
    assert digest.hexdigest() == metadata["payload_sha256"]
    return {
        "accuracy_pass": worst_truth <= 0.30
        and worst_intensity < 1e-5
        and support_mismatches == 0
        and invalid_statuses == 0
        and eager_worst < (0.01 if config["operation"] == "register" else 1e-5),
        "checked_frames": frames,
        "worst_truth_rms_voxels": worst_truth,
        "truth_limit_voxels": 0.30,
        "worst_relative_intensity_rmse": worst_intensity,
        "intensity_limit": 1e-5,
        "support_mismatches": support_mismatches,
        "invalid_statuses": invalid_statuses,
        "observed_fitting_batch_sizes": sorted(observed_batches),
        "observed_task_sizes": sorted(observed_tasks),
        "eager_checked_frames": samples,
        "eager_worst_difference": eager_worst,
        "eager_difference_units": "minimum-spacing voxels"
        if config["operation"] == "register"
        else "intensity",
        "input_sha256_verified": digest.hexdigest(),
    }


def worker(config_path: Path, phase: str) -> int:
    """Run one isolated timing, memory or accuracy phase.

    Parameters
    ----------
    config_path : pathlib.Path
        Frozen case config.
    phase : str
        Measurement phase.

    Returns
    -------
    int
        Nonzero for errors or quality failures.
    """
    config = json.loads(config_path.read_text())
    directory = config_path.parent
    record: dict[str, Any] = {"phase": phase, "status": "running"}
    write_json(directory / f"{phase}.json", record)
    output = directory / f"{phase}.zarr"
    try:
        if config["device"] == "cuda" and not torch.cuda.is_available():
            record.update(status="unavailable", reason="CUDA unavailable")
            return 0
        with (
            dask.config.set({"distributed.scheduler.allowed-failures": 0}),
            LocalCluster(
                n_workers=config["workers"],
                threads_per_worker=1,
                processes=True,
                memory_limit=config["worker_memory_bytes"],
                dashboard_address=None,
            ) as cluster,
            Client(cluster) as client,
        ):
            initialization = client.run(configure_worker, config)
            config["effective_batch_size"] = min(
                value["batch_size"] for value in initialization.values()
            )
            record.update(
                worker_initialization=initialization,
                effective_batch_size=config["effective_batch_size"],
            )
            calls = config["runs"] if phase == "timing" else 1
            times = []
            for index in range(calls):
                started = time.perf_counter()
                if phase == "accuracy":
                    with get_task_stream(client=client) as stream:
                        execute(config, output)
                else:
                    execute(config, output)
                seconds = time.perf_counter() - started
                if phase == "timing":
                    times.append(seconds)
                    record["wall_seconds"] = times
                    write_json(directory / f"{phase}.json", record)
                if phase == "memory":
                    record["worker_gpu_peaks"] = client.run(gpu_peaks)
                if phase == "accuracy":
                    durations = [
                        part["stop"] - part["start"]
                        for event in stream.data
                        if str(event["key"]).startswith(
                            ("_run_registration_batch-", "_run_resampling_batch-")
                        )
                        for part in event["startstops"]
                        if part["action"] == "compute"
                    ]
                    record["task_duration_sample"] = {
                        "count": len(durations),
                        "median_seconds": float(np.median(durations))
                        if durations
                        else None,
                        "minimum_seconds": min(durations) if durations else None,
                        "guidance_seconds": 0.1,
                        "note": "Bounded Dask task-stream sample; includes task execution/lease waits, not an exhaustive profile.",
                    }
                    record.update(verify(config, output, directory))
                    record["stored_output_bytes"] = sum(
                        path.stat().st_size
                        for path in output.rglob("*")
                        if path.is_file()
                    )
                if phase != "accuracy" or record["accuracy_pass"]:
                    # Only this fresh worker-owned store is removed, outside timing.
                    shutil.rmtree(output)
            record["status"] = (
                "passed"
                if phase != "accuracy" or record["accuracy_pass"]
                else "quality_failed"
            )
    except Exception:
        record.update(status="error", error=traceback.format_exc())
        raise
    finally:
        write_json(directory / f"{phase}.json", record)
    return int(record["status"] in ("error", "quality_failed"))


def process_tree_rss(pid: int) -> int:
    """Sample aggregate RSS of a phase supervisor and all its worker processes.

    Parameters
    ----------
    pid : int
        Root phase PID.

    Returns
    -------
    int
        Process-tree RSS sum, including shared pages counted in each process.
    """
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def measure(config_path: Path, phase: str, script: Path) -> dict[str, Any]:
    """Supervise a fresh phase process, polling memory only in its memory phase.

    Parameters
    ----------
    config_path : pathlib.Path
        Frozen case settings.
    phase : str
        Timing, memory or accuracy selector.
    script : pathlib.Path
        Frozen CLI entry point.

    Returns
    -------
    dict
        Phase result with exit status and optional process-tree RSS peak.
    """
    config = json.loads(config_path.read_text())
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": str(config["threads"]),
        "MKL_NUM_THREADS": str(config["threads"]),
    }
    peak = 0
    with (config_path.parent / f"{phase}.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(script),
                "--worker",
                str(config_path),
                "--phase",
                phase,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if phase == "memory":
                    peak = max(peak, process_tree_rss(process.pid))
                time.sleep(0.02 if phase == "memory" else 0.2)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise
    result_path = config_path.parent / f"{phase}.json"
    record = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {"status": "error", "error": "Phase ended without a result"}
    )
    record["exit_code"] = process.returncode
    if phase == "memory":
        record["peak_process_tree_rss_bytes"] = peak
    if process.returncode and record["status"] != "quality_failed":
        record["status"] = "error"
    write_json(result_path, record)
    return record


def report(directory: Path, result: dict[str, Any]) -> None:
    """Write concise machine-readable and standalone human-readable results.

    Parameters
    ----------
    directory : pathlib.Path
        Fresh benchmark directory.
    result : dict
        Complete run state, including nonpassing cases.

    Returns
    -------
    None
        JSON and HTML reports are written without changing earlier artifacts.
    """
    write_json(directory / "results.json", result)
    rows = []
    for case in result["cases"]:
        timing = case.get("timing", {}).get("wall_seconds", [])
        duration = (
            f"{np.median(timing):.3f} [{min(timing):.3f}–{max(timing):.3f}]"
            if timing
            else "—"
        )
        peak_bytes = case.get("memory", {}).get("peak_process_tree_rss_bytes")
        peak = f"{peak_bytes / 2**30:.3f}" if peak_bytes is not None else "—"
        rate = (
            result["inputs"]["image_bytes"] / 2**30 / float(np.median(timing))
            if timing
            else 0
        )
        rate_label = f"{rate:.3f}" if timing else "—"
        first = f"{timing[0]:.3f}" if timing else "—"
        chunk = case["config"]["chunk_bytes"] / 2**20
        rows.append(
            f"<tr><td>{html.escape(case['id'])}</td><td>{case['status']}</td><td>{chunk:.2f}</td><td>{duration}</td><td>{first}</td><td>{rate_label}</td><td>{peak}</td></tr>"
        )
    (directory / "report.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Lazy voxreg benchmark</title><style>body{font:16px system-ui;margin:2em}td,th{padding:.6em;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>Lazy voxreg benchmark</h1><p>Read → compute → one DataTree Zarr write. No cold-cache claim. Nonpassing cases are not qualified throughput.</p><table><tr><th>Case</th><th>Status</th><th>Chunk MiB</th><th>Median [min–max], s</th><th>First call, s</th><th>Input GiB/s</th><th>Peak process-tree RSS, GiB</th></tr>"
        + "".join(rows)
        + f"</table><p>Overall status: {html.escape(result['status'])}. Diagnostic: {result['diagnostic']}. Logical input: {result['inputs']['image_bytes'] / 2**30:.3f} GiB.</p>"
        + "<p>See results.json for first-call times, input/output bytes, GPU peaks, settings, accuracy and task-duration samples. RSS sums count shared pages repeatedly; GPU peaks are per worker, not a concurrent device-wide sum.</p>"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the opt-in disk-backed synthetic registration/resampling benchmark.

    Parameters
    ----------
    argv : list of str, optional
        Command arguments; not provided reads the command line.

    Returns
    -------
    int
        Zero for passing or unavailable cases, nonzero for errors/quality failures.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--operation", choices=("register", "resample"), default="register"
    )
    parser.add_argument(
        "--size-gib",
        type=float,
        help="Opt into large files; omitted runs an eight-frame smoke case",
    )
    parser.add_argument("--shape", type=int, nargs=3)
    parser.add_argument(
        "--chunk-mib",
        type=float,
        nargs="+",
        help="Dask chunk targets; large default: 32 128 MiB, rounded to volumes",
    )
    parser.add_argument(
        "--devices", choices=("cpu", "cuda"), nargs="+", default=["cpu", "cuda"]
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-memory-gib", type=float, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--phase", choices=("timing", "memory", "accuracy"), help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker, args.phase)
    if args.output is None:
        parser.error("--output is required")
    for name in ("runs", "iterations", "workers", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for name in ("size_gib", "worker_memory_gib"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
        if value is not None and value > sys.maxsize / 2**30:
            parser.error(f"--{name.replace('_', '-')} exceeds supported byte sizes")
    shape = tuple(args.shape or ((32, 128, 128) if args.size_gib else (1, 24, 32)))
    if any(size < 2 for size in shape[1:]) or shape[0] < 1:
        parser.error("shape must be a k=1 plane or 3D volume with spatial axes >= 2")
    volume_bytes = math.prod(shape) * 4
    if volume_bytes > sys.maxsize:
        parser.error("shape exceeds supported volume byte sizes")
    frames = (
        max(2, math.ceil(args.size_gib * 2**30 / volume_bytes)) if args.size_gib else 8
    )
    targets = args.chunk_mib or (
        [32, 128]
        if args.size_gib
        else [2 * volume_bytes / 2**20, 4 * volume_bytes / 2**20]
    )
    if any(
        not math.isfinite(value) or value <= 0 or value > sys.maxsize / 2**20
        for value in targets
    ):
        parser.error("--chunk-mib targets must be finite and positive")
    chunk_frames = sorted(
        {
            min(frames, max(1, math.ceil(value * 2**20 / volume_bytes)))
            for value in targets
        }
    )
    if max(chunk_frames) * volume_bytes > args.worker_memory_gib * 2**30 / 4:
        parser.error(
            "input chunks must use at most one quarter of worker memory; leave room for fixed/output/working state"
        )
    output = args.output.resolve()
    if output.exists():
        parser.error(
            "output directory must be fresh; existing files are never overwritten"
        )
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < frames * volume_bytes * 2.5 + 256 * 2**20:
        parser.error(
            "insufficient free disk for input plus one temporary result and safety margin"
        )
    output.mkdir(parents=True)
    frozen = output / "sources"
    shutil.copytree(
        ROOT / "src", frozen / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    (frozen / "examples").mkdir()
    for name in ("benchmark_lazy.py", "_benchmark_lazy.py"):
        shutil.copy2(ROOT / "examples" / name, frozen / "examples" / name)
    hashes = {
        str(path.relative_to(frozen)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in frozen.rglob("*")
        if path.is_file()
    }
    result = {
        "protocol": PROTOCOL,
        "operation": args.operation,
        "scope": "voxreg rigid synthetic disk-backed recordings; no cross-backend ranking",
        "diagnostic": args.runs != 5 or args.size_gib is None,
        "inputs": {"image_bytes": frames * volume_bytes},
        "preparation_seconds": None,
        "source_sha256": hashes,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "voxreg",
                "torch",
                "numpy",
                "scipy",
                "xarray",
                "confusius",
                "dask",
                "distributed",
                "zarr",
            )
        },
        "host": {
            "platform": platform.platform(),
            "physical_ram_bytes": psutil.virtual_memory().total,
        },
        "cache_policy": "OS caches are not flushed; source generation and earlier phases may warm caches. Writes use OS buffering, without forced durable-media fsync.",
        "timing_boundary": "open input metadata, build graph, read, compute and finish one DataTree Zarr write; imports/cluster/device initialization and output deletion excluded",
        "chunk_guidance_source": "https://docs.dask.org/en/stable/array-chunks.html",
        "gpu_memory_note": "Per-worker allocated/reserved peaks exclude CUDA driver/context and non-Torch allocations; peaks from different processes are not simultaneous device totals.",
        "larger_than_host_ram": frames * volume_bytes > psutil.virtual_memory().total,
        "larger_than_total_worker_limits": frames * volume_bytes
        > args.workers * args.worker_memory_gib * 2**30,
        "cases": [],
        "status": "preparing",
    }
    report(output, result)
    try:
        started = time.perf_counter()
        result["inputs"] = stage_inputs(output / "inputs", shape, frames)
        result["preparation_seconds"] = time.perf_counter() - started
        result["status"] = "running"
        report(output, result)
        for device in dict.fromkeys(args.devices):
            for chunk in chunk_frames:
                name = f"{args.operation}-rigid-{device}-t{chunk}"
                directory = output / name
                directory.mkdir()
                config = {
                    "id": name,
                    "operation": args.operation,
                    "device": device,
                    "inputs": str(output / "inputs"),
                    "runs": args.runs,
                    "iterations": args.iterations,
                    "threads": args.threads,
                    "workers": args.workers,
                    "worker_memory_bytes": int(args.worker_memory_gib * 2**30),
                    "batch_size": args.batch_size,
                    "chunk_frames": chunk,
                    "chunk_bytes": chunk * volume_bytes,
                    "chunk_targets_mib": targets,
                    "guidance": "Dask: commonly 10MB–1GB, tasks >100ms, fit comfortably in RAM, align storage; smoke sizes are exceptions",
                }
                config_path = directory / "config.json"
                write_json(config_path, config)
                case: dict[str, Any] = {
                    "id": name,
                    "config": config,
                    "status": "running",
                }
                result["cases"].append(case)
                report(output, result)
                for phase in ("timing", "memory", "accuracy"):
                    case[phase] = measure(
                        config_path, phase, frozen / "examples" / "benchmark_lazy.py"
                    )
                    state = case[phase]["status"]
                    if state != "passed":
                        case["status"] = state
                        break
                    if phase == "timing":
                        config["effective_batch_size"] = case[phase][
                            "effective_batch_size"
                        ]
                        write_json(config_path, config)
                else:
                    case["status"] = "passed"
                print(f"{name}: {case['status']}", flush=True)
                report(output, result)
                if case["status"] in ("error", "quality_failed"):
                    break
            if result["cases"][-1]["status"] in ("error", "quality_failed"):
                break
        result["status"] = (
            "failed"
            if any(
                case["status"] in ("error", "quality_failed")
                for case in result["cases"]
            )
            else "passed"
            if any(case["status"] == "passed" for case in result["cases"])
            else "unavailable"
        )
    except BaseException:
        result["status"] = "interrupted"
        result["error"] = traceback.format_exc()
        report(output, result)
        raise
    report(output, result)
    return int(result["status"] not in ("passed", "unavailable"))
