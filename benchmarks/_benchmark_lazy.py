"""Disk-backed synthetic lazy benchmarks with isolated measurement phases."""

from __future__ import annotations

import argparse
import gc
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
from itertools import product
from pathlib import Path
from typing import Any

import dask
import numpy as np
import psutil
import torch
import xarray as xr
import zarr
from _benchmark_cache import evict_input_cache
from distributed import Client, LocalCluster, get_task_stream
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
"""Live or frozen source root used by isolated benchmark workers."""
sys.path.insert(0, str(ROOT / "src"))

from voxeldata import create_voxeldata, load_array

from voxreg import register, resample
from voxreg.preparation import choose_batch_size

PROTOCOL = "voxreg-lazy-benchmark-v3"
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
    directory.mkdir(parents=True)
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
    moving = load_array(
        Path(config["inputs"]) / "recording.zarr",
        variable="image",
        chunks={"time": config["chunk_frames"]},
        consolidated=False,
    )
    fixed = load_array(
        Path(config["inputs"]) / "fixed.zarr",
        variable="image",
        chunks=None,
        consolidated=False,
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


def count_worker_tasks(dask_worker: Any) -> int:
    """Collect completed results and count retained Dask task data.

    Parameters
    ----------
    dask_worker : distributed.Worker
        Benchmark worker supplied by `Client.run`.

    Returns
    -------
    int
        Retained in-memory or spilled task keys; zero is required before eviction.
    """
    gc.collect()
    return len(dask_worker.data)


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
        Zarr-ready values with affine metadata for VoxelData loading.
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
    moving = load_array(
        inputs / "recording.zarr",
        variable="image",
        chunks={"time": config["chunk_frames"]},
        consolidated=False,
    )
    fixed = load_array(
        inputs / "fixed.zarr", variable="image", chunks=None, consolidated=False
    )
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
        assert "forward" in result and "inverse" in result
        tree = {
            "images": storage_view(
                result["images"].to_dataset()["registered"]
            ).to_dataset(name="image"),
            "validity": storage_view(result["images"].to_dataset()["valid"]).to_dataset(
                name="valid"
            ),
            "diagnostics": result["diagnostics"].to_dataset(),
            "forward": result["forward"]
            .to_dataset()["transform"]
            .to_dataset(name="matrix"),
            "inverse": result["inverse"]
            .to_dataset()["transform"]
            .to_dataset(name="matrix"),
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
    fixed = load_array(
        inputs / "fixed.zarr", variable="image", chunks=None, consolidated=False
    ).compute()
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
                assert "forward" in eager
                delta = (
                    np.linalg.norm(
                        (
                            (eager["forward"].to_dataset()["transform"].values - fitted)
                            @ points
                        )[:3],
                        axis=0,
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
    record: dict[str, Any] = {
        "phase": phase,
        "status": "running",
        "cache_evictions": [],
        "measurement_active": False,
    }
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
                # Released futures may take a moment to disappear from worker caches.
                for _ in range(100):
                    cached = client.run(count_worker_tasks)
                    if not any(cached.values()):
                        break
                    time.sleep(0.05)
                else:
                    record["retained_worker_tasks"] = cached
                    raise RuntimeError(
                        "Cold-input check failed: previous Dask task data is still retained."
                    )
                eviction = {
                    **evict_input_cache(Path(config["inputs"])),
                    "worker_cached_tasks": cached,
                }
                record["cache_evictions"].append(eviction)
                write_json(directory / f"{phase}.json", record)
                if not eviction["verified"]:
                    raise RuntimeError(
                        f"Cold-input verification failed: {eviction['resident_pages_after']} resident pages remain. "
                        "Use disk-backed inputs; tmpfs or pinned pages cannot be qualified as cold."
                    )
                if phase == "memory":
                    record["baseline_process_tree_rss_bytes"] = process_tree_rss(
                        os.getpid()
                    )
                record["measurement_active"] = True
                write_json(directory / f"{phase}.json", record)
                started = time.perf_counter()
                try:
                    if phase == "accuracy":
                        with get_task_stream(client=client) as stream:
                            execute(config, output)
                    else:
                        execute(config, output)
                finally:
                    seconds = time.perf_counter() - started
                    record["measurement_active"] = False
                    if phase == "memory":
                        record["end_process_tree_rss_bytes"] = process_tree_rss(
                            os.getpid()
                        )
                    write_json(directory / f"{phase}.json", record)
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
                            ("voxreg-fit-", "_run_resampling_batch-")
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
    result_path = config_path.parent / f"{phase}.json"
    with (config_path.parent / f"{phase}.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(script),
                "--lazy",
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
                if phase == "memory" and result_path.exists():
                    state = json.loads(result_path.read_text())
                    if state.get("measurement_active"):
                        peak = max(peak, process_tree_rss(process.pid))
                time.sleep(0.02 if phase == "memory" else 0.2)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise
    record = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {"status": "error", "error": "Phase ended without a result"}
    )
    record["exit_code"] = process.returncode
    if phase == "memory":
        baseline = record.get("baseline_process_tree_rss_bytes", 0)
        peak = max(peak, baseline, record.get("end_process_tree_rss_bytes", 0))
        record["peak_process_tree_rss_bytes"] = peak
        record["peak_rss_increment_bytes"] = peak - baseline
    if process.returncode and record["status"] != "quality_failed":
        record["status"] = "error"
    write_json(result_path, record)
    return record


def summarize_scaling(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare passing input sizes only within an identical execution policy.

    Parameters
    ----------
    cases : list of dict
        Completed or in-progress benchmark cases.

    Returns
    -------
    list of dict
        Descriptive input/absolute/incremental RSS scaling; no automatic memory
        qualification or larger-than-RAM claim is inferred from these ratios.
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    policy_fields = (
        "device",
        "chunk_frames",
        "effective_batch_size",
        "workers",
        "worker_memory_bytes",
        "threads",
    )
    for case in cases:
        memory = case.get("memory", {})
        if case["status"] != "passed" or any(
            memory.get(name) is None
            for name in (
                "peak_process_tree_rss_bytes",
                "baseline_process_tree_rss_bytes",
            )
        ):
            continue
        policy = tuple(case["config"].get(name) for name in policy_fields)
        groups.setdefault(policy, []).append(case)
    summaries = []
    for policy, group in groups.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda case: case["config"]["input_image_bytes"])
        sizes = [case["config"]["input_image_bytes"] for case in group]
        peaks = [case["memory"]["peak_process_tree_rss_bytes"] for case in group]
        baselines = [
            case["memory"]["baseline_process_tree_rss_bytes"] for case in group
        ]
        increments = [case["memory"]["peak_rss_increment_bytes"] for case in group]
        summaries.append(
            {
                **dict(zip(policy_fields, policy, strict=True)),
                "input_bytes": sizes,
                "baseline_rss_bytes": baselines,
                "peak_rss_bytes": peaks,
                "peak_rss_increment_bytes": increments,
                "input_growth_ratio": sizes[-1] / sizes[0],
                "peak_rss_growth_ratio": peaks[-1] / peaks[0] if peaks[0] else None,
                "rss_increment_growth_bytes": increments[-1] - increments[0],
            }
        )
    return summaries


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
    result["scaling"] = summarize_scaling(result["cases"])
    write_json(directory / "results.json", result)
    rows = []
    for case in result["cases"]:
        timing = case.get("timing", {}).get("wall_seconds", [])
        duration = (
            f"{np.median(timing):.3f} [{min(timing):.3f}–{max(timing):.3f}]"
            if timing
            else "—"
        )
        memory = case.get("memory", {})
        peak = memory.get("peak_process_tree_rss_bytes")
        baseline = memory.get("baseline_process_tree_rss_bytes")
        rss = (
            f"{baseline / 2**30:.3f} → {peak / 2**30:.3f} (+{memory['peak_rss_increment_bytes'] / 2**30:.3f})"
            if peak is not None and baseline is not None
            else "—"
        )
        size = case["config"]["input_image_bytes"] / 2**30
        rate_label = (
            f"{size / float(np.median(timing)):.3f}"
            if timing and case["status"] == "passed"
            else "—"
        )
        first = f"{timing[0]:.3f}" if timing else "—"
        chunk = case["config"]["chunk_bytes"] / 2**20
        rows.append(
            f"<tr><td>{html.escape(case['id'])}</td><td>{case['status']}</td><td>{size:.3f}</td><td>{chunk:.2f}</td><td>{duration}</td><td>{first}</td><td>{rate_label}</td><td>{rss}</td></tr>"
        )
    scaling = "".join(
        f"<li>{html.escape(summary['device'])}, chunk {summary['chunk_frames']} frames, batch {summary['effective_batch_size']}: input {summary['input_growth_ratio']:.2f}×; peak RSS {summary['peak_rss_growth_ratio']:.2f}×; incremental RSS growth {summary['rss_increment_growth_bytes'] / 2**30:.3f} GiB.</li>"
        for summary in result["scaling"]
        if summary["peak_rss_growth_ratio"] is not None
    )
    (directory / "report.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Lazy voxreg benchmark</title><style>body{font:16px system-ui;margin:2em}td,th{padding:.6em;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>Cold-input lazy voxreg benchmark</h1><p>Verified cold input file pages → compute → one buffered DataTree Zarr write. Eviction/verification is outside timing. Nonpassing cases are not qualified throughput.</p><table><tr><th>Case</th><th>Status</th><th>Input GiB</th><th>Chunk MiB</th><th>Median [min–max], s</th><th>First call, s</th><th>Input GiB/s</th><th>RSS baseline → peak (+increment), GiB</th></tr>"
        + "".join(rows)
        + f"</table><p>Overall status: {html.escape(result['status'])}. Diagnostic: {result['diagnostic']}.</p><h2>Fixed-policy size scaling</h2><ul>{scaling}</ul>"
        + "<p>Scaling is descriptive, not proof of larger-than-RAM operation. See results.json for inputs, cache verification, GPU peaks, accuracy and task durations. RSS sums count shared pages repeatedly; GPU peaks are per worker, not a concurrent device-wide sum. File-page eviction does not clear storage-controller caches or make output writes durable.</p>"
    )


def main(argv: list[str] | None = None, *, operation: str = "register") -> int:
    """Run the opt-in disk-backed synthetic registration/resampling benchmark.

    Parameters
    ----------
    argv : list of str, optional
        Command arguments; not provided reads the command line.
    operation : {"register", "resample"}, default: "register"
        Operation selected by the public benchmark entry point.

    Returns
    -------
    int
        Zero for passing or unavailable cases, nonzero for errors/quality failures.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage="%(prog)s --lazy --output OUTPUT [--size-gib 1 4] [--devices cpu cuda] [--runs N]",
    )
    advanced = parser.add_argument_group("advanced workload/memory overrides")
    parser.add_argument("--output", type=Path)
    parser.set_defaults(operation=operation)
    parser.add_argument(
        "--size-gib",
        type=float,
        nargs="+",
        help="Explicit input-size sweep, e.g. 1 4 GiB; omitted runs eight smoke frames",
    )
    advanced.add_argument("--shape", type=int, nargs=3)
    advanced.add_argument(
        "--chunk-mib",
        type=float,
        nargs="+",
        help="Dask chunk targets; sized-input default: 128 MiB, fixed across sizes",
    )
    parser.add_argument(
        "--devices", choices=("cpu", "cuda"), nargs="+", default=["cpu", "cuda"]
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    advanced.add_argument("--iterations", type=int, default=200)
    advanced.add_argument("--workers", type=int, default=1)
    advanced.add_argument("--worker-memory-gib", type=float, default=4)
    advanced.add_argument("--batch-size", type=int)
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
    for name, values in (
        ("size-gib", args.size_gib or []),
        ("worker-memory-gib", [args.worker_memory_gib]),
    ):
        if any(not math.isfinite(value) or value <= 0 for value in values):
            parser.error(f"--{name} must be finite and positive")
        if any(value > sys.maxsize / 2**30 for value in values):
            parser.error(f"--{name} exceeds supported byte sizes")
    if sys.platform != "linux" or not hasattr(os, "posix_fadvise"):
        parser.error(
            "Verified cold-input lazy benchmarks require Linux fadvise/mincore."
        )
    shape = tuple(args.shape or ((32, 128, 128) if args.size_gib else (1, 24, 32)))
    if any(size < 2 for size in shape[1:]) or shape[0] < 1:
        parser.error("shape must be a k=1 plane or 3D volume with spatial axes >= 2")
    volume_bytes = math.prod(shape) * 4
    if volume_bytes > sys.maxsize:
        parser.error("shape exceeds supported volume byte sizes")
    frame_counts = (
        sorted(
            {max(2, math.ceil(size * 2**30 / volume_bytes)) for size in args.size_gib}
        )
        if args.size_gib
        else [8]
    )
    targets = args.chunk_mib or (
        [128] if args.size_gib else [2 * volume_bytes / 2**20, 4 * volume_bytes / 2**20]
    )
    if any(
        not math.isfinite(value) or value <= 0 or value > sys.maxsize / 2**20
        for value in targets
    ):
        parser.error("--chunk-mib targets must be finite and positive")
    chunk_frames = sorted(
        {
            min(frame_counts[0], max(1, math.ceil(value * 2**20 / volume_bytes)))
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
    required_disk = (
        sum(frame_counts) + 1.5 * max(frame_counts)
    ) * volume_bytes + 256 * 2**20
    if shutil.disk_usage(ancestor).free < required_disk:
        parser.error(
            "insufficient free disk for retained inputs plus one temporary result and safety margin"
        )
    output.mkdir(parents=True)
    frozen = output / "sources"
    shutil.copytree(
        ROOT / "src", frozen / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    (frozen / "benchmarks").mkdir()
    for name in (
        f"benchmark_{operation}.py",
        "_benchmark_lazy.py",
        "_benchmark_cache.py",
    ):
        shutil.copy2(ROOT / "benchmarks" / name, frozen / "benchmarks" / name)
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
        "inputs": [
            {
                "id": f"n{frames}",
                "frames": frames,
                "image_bytes": frames * volume_bytes,
                "larger_than_host_ram": frames * volume_bytes
                > psutil.virtual_memory().total,
                "larger_than_total_worker_limits": frames * volume_bytes
                > args.workers * args.worker_memory_gib * 2**30,
            }
            for frames in frame_counts
        ],
        "preparation_seconds": 0.0,
        "source_sha256": hashes,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "voxreg",
                "torch",
                "numpy",
                "scipy",
                "xarray",
                "voxeldata",
                "dask",
                "distributed",
                "zarr",
            )
        },
        "host": {
            "platform": platform.platform(),
            "physical_ram_bytes": psutil.virtual_memory().total,
        },
        "cache_policy": "Before every pass: fsync benchmark-owned input files, advise DONTNEED, verify zero resident file pages with Linux mincore; fail closed. No global cache clearing. Output writes remain OS-buffered, without forced durable-media fsync.",
        "timing_boundary": "open input metadata, build graph, read, compute and finish one DataTree Zarr write; imports/cluster/device initialization, input flushing/eviction/verification and output deletion excluded",
        "chunk_guidance_source": "https://docs.dask.org/en/stable/array-chunks.html",
        "gpu_memory_note": "Per-worker allocated/reserved peaks exclude CUDA driver/context and non-Torch allocations; peaks from different processes are not simultaneous device totals.",
        "cases": [],
        "status": "preparing",
    }
    report(output, result)
    try:
        pinned_batches: dict[tuple[str, int], int] = {}
        for input_metadata in result["inputs"]:
            frames = input_metadata["frames"]
            inputs = output / "inputs" / input_metadata["id"]
            started = time.perf_counter()
            input_metadata.update(stage_inputs(inputs, shape, frames))
            result["preparation_seconds"] += time.perf_counter() - started
            result["status"] = "running"
            report(output, result)
            for device, chunk in product(dict.fromkeys(args.devices), chunk_frames):
                policy = (device, chunk)
                name = f"{args.operation}-rigid-{device}-t{chunk}-n{frames}"
                directory = output / name
                directory.mkdir()
                config = {
                    "id": name,
                    "operation": args.operation,
                    "device": device,
                    "inputs": str(inputs),
                    "input_frames": frames,
                    "input_image_bytes": input_metadata["image_bytes"],
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
                if policy in pinned_batches:
                    config["effective_batch_size"] = pinned_batches[policy]
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
                        config_path,
                        phase,
                        frozen / "benchmarks" / f"benchmark_{operation}.py",
                    )
                    state = case[phase]["status"]
                    if state != "passed":
                        case["status"] = state
                        break
                    if phase == "timing":
                        batch = case[phase]["effective_batch_size"]
                        if policy in pinned_batches and pinned_batches[policy] != batch:
                            case.update(
                                status="error",
                                error="Fitting batch changed across input sizes",
                            )
                            break
                        pinned_batches[policy] = batch
                        config["effective_batch_size"] = batch
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
