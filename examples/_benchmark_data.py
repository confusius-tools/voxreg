"""Shared benchmark inputs and resident registration adapters; no CLI."""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SHAPES = {"2d+t": (1, 150, 128), "3d": (80, 150, 80), "3d+t": (80, 150, 80)}
SPACING = np.array([0.2, 0.1, 0.1])


def _configs():
    cases = [
        {
            "id": f"resident-{workload}-{model}-{backend}",
            "suite": "resident",
            "workload": workload,
            "model": model,
            "backend": backend,
        }
        for workload, model, backend in itertools.product(
            SHAPES,
            ("rigid", "affine", "syn"),
            ("cpu", "cuda", "fireants-cpu", "fireants-cuda", "antspyx"),
        )
    ]
    return cases


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _power():
    paths = {
        "ac": "/sys/class/power_supply/ADP0/online",
        "battery": "/sys/class/power_supply/BAT0/status",
        "profile": "/sys/firmware/acpi/platform_profile",
    }
    return {
        name: Path(p).read_text().strip() if Path(p).exists() else None
        for name, p in paths.items()
    }


def _grid(shape):
    return (
        (np.indices(shape).reshape(3, -1).T - (np.array(shape) - 1) / 2) * SPACING
    ).T


def _matrix(model, amplitude, shape):
    planar = shape[0] == 1
    matrix = np.eye(4)
    if model == "rigid":
        vector = np.array([0.008, 0, 0] if planar else [0.005, -0.006, 0.004])
        matrix[:3, :3] = Rotation.from_rotvec(amplitude * vector).as_matrix()
    elif model == "affine":
        strain = np.array(
            [[0.004, 0.002, 0], [-0.002, -0.004, 0.003], [0.002, 0.001, 0.003]]
        )
        if planar:
            strain[0] = 0
            strain[:, 0] = 0
        matrix[:3, :3] += amplitude * strain
    else:
        raise ValueError(f"Unsupported benchmark linear model: {model}")
    matrix[:3, 3] = amplitude * np.array([0 if planar else 0.04, -0.04, 0.06])
    return matrix


def _warp(points, amplitude, shape, inverse=False):
    # Independent analytic compositions; reverse-order scalar Newton inverts truth.
    result = points.copy()
    axes = list(range(1 if shape[0] == 1 else 0, 3))
    extent = (np.array(shape) - 1) * SPACING
    rates = amplitude * 1.2 * SPACING.min() * np.array([1, -0.6, 0.35])[: len(axes)]
    stages = list(zip(axes, rates, strict=True))
    for axis, rate in reversed(stages) if inverse else stages:
        window = np.ones(points.shape[1])
        for other in axes:
            if other != axis:
                window *= np.cos(np.pi * result[other] / extent[other])
        if inverse:
            target = result[axis].copy()
            for _ in range(12):
                angle = np.pi * result[axis] / extent[axis]
                residual = result[axis] + rate * np.cos(angle) * window - target
                derivative = 1 - rate * np.pi / extent[axis] * np.sin(angle) * window
                result[axis] -= residual / derivative
        else:
            result[axis] += rate * np.cos(np.pi * result[axis] / extent[axis]) * window
    return result


def _truth(points, amplitude, model, shape, inverse=False):
    if model == "syn":
        return _warp(points, amplitude, shape, inverse)
    matrix = _matrix(model, amplitude, shape)
    if inverse:
        matrix = np.linalg.inv(matrix)
    return matrix[:3, :3] @ points + matrix[:3, 3, None]


def _phantom(points, shape):
    rng = np.random.default_rng(871)
    extent = np.maximum((np.array(shape) - 1) * SPACING, SPACING)
    values = np.full(points.shape[1], 0.05)
    axes = range(1 if shape[0] == 1 else 0, 3)
    for _ in range(24):
        location = rng.uniform(-0.35, 0.35, 3) * extent
        widths = np.maximum(rng.uniform(0.025, 0.06, 3) * extent, 3 * SPACING)
        distance = sum(((points[a] - location[a]) / widths[a]) ** 2 for a in axes)
        values += rng.uniform(0.5, 2) * np.exp(-0.5 * distance)
    # Continuous texture avoids uninformative constant windows in local correlation.
    for _ in range(4):
        frequencies = rng.uniform(0.15, 0.45, 3)
        phase = sum(points[a] / SPACING[a] * frequencies[a] for a in axes)
        values += 0.01 * np.sin(phase + rng.uniform(-np.pi, np.pi))
    return values.reshape(shape).astype(np.float32)


def _dataset(shape, frames, model, recording):
    from confusius import create_voxeldata

    points = _grid(shape)
    affine = np.diag([*SPACING, 1.0])
    affine[:3, 3] = -(np.array(shape) - 1) * SPACING / 2
    fixed_values = _phantom(points, shape)
    fixed = create_voxeldata(fixed_values, dims=("k", "j", "i"), voxel_to_world=affine)
    count = frames if recording else 1
    amplitudes = 0.8 + 0.2 * np.sin(2 * np.pi * np.arange(count) / 16)
    if recording:
        amplitudes[0] = 0
    else:
        amplitudes[0] = 1
    values = np.empty((count, *shape), dtype=np.float32)
    # A declared 16-pose cycle bounds preparation cost, not the number of fitted frames.
    cache = {}
    for index, amplitude in enumerate(amplitudes):
        key = index % 16 if recording and index else -1
        if key not in cache:
            cache[key] = _phantom(
                _truth(points, amplitude, model, shape, inverse=True), shape
            )
        values[index] = cache[key]
    if recording:
        moving = create_voxeldata(
            values,
            dims=("time", "k", "j", "i"),
            time=np.arange(count) * 0.1,
            voxel_to_world=affine,
        )
    else:
        moving = create_voxeldata(
            values[0], dims=("k", "j", "i"), voxel_to_world=affine
        )
    return moving, fixed, amplitudes


def _run_fireants(
    moving,
    fixed,
    model,
    iterations,
    device,
    export,
    reference,
    amplitudes,
    *,
    batch_size,
):
    import SimpleITK as sitk
    import torch
    from _benchmark_metrics import _point_rms
    from fireants.interpolator import FFO_AVAILABLE, USE_FFO
    from fireants.io.image import BatchedImages, Image
    from fireants.registration.affine import AffineRegistration
    from fireants.registration.rigid import RigidRegistration
    from fireants.registration.syn import SyNRegistration
    from torch.nn import functional

    shape = tuple(fixed.shape)
    planar = shape[0] == 1
    axes = [2, 1] if planar else [2, 1, 0]
    raw = moving.values if "time" in moving.dims else moving.values[None]
    scale = max(float(raw.max()), float(fixed.values.max()))
    fitting = raw / scale
    linear = model != "syn"
    batch = batch_size

    def image(values):
        values = values[0] if planar else values
        if linear:
            values = np.stack((values, np.ones_like(values)), axis=-1)
        result = sitk.GetImageFromArray(np.ascontiguousarray(values), isVector=linear)
        result.SetSpacing(tuple(SPACING[axes]))
        result.SetOrigin(tuple(-((np.array(shape) - 1) * SPACING / 2)[axes]))
        return Image(result, device=device)

    target = image(fixed.values / scale)
    output = np.empty_like(raw)
    maps, counts, statuses = [], [], []
    first = int(reference)
    if first:
        output[0] = raw[0]
        counts.append(0)
        statuses.append("reference")
        if export:
            maps.append(0.0)
    for start in range(first, len(raw), batch):
        stop = min(start + batch, len(raw))
        count = stop - start
        targets = BatchedImages([target] * count, optimize_memory=False)
        sources = BatchedImages([image(frame) for frame in fitting[start:stop]])
        calls = 0

        def global_ncc(moved, stationary):
            nonlocal calls
            calls += 1
            spatial = tuple(range(1, moved.ndim - 1))
            weights = (moved[:, 1] >= 1 - 1e-5).to(moved.dtype)
            n = weights.sum(dim=spatial, keepdim=True).clamp_min(1)
            x, y = moved[:, 0], stationary[:, 0]
            x = (x - (x * weights).sum(dim=spatial, keepdim=True) / n) * weights
            y = (y - (y * weights).sum(dim=spatial, keepdim=True) / n) * weights
            denominator = x.square().sum(dim=spatial) * y.square().sum(dim=spatial)
            usable = torch.isfinite(denominator) & (
                denominator > torch.finfo(moved.dtype).tiny
            )
            safe = torch.where(usable, denominator, 1)
            return -torch.where(
                usable, (x * y).sum(dim=spatial).square() / safe, 0
            ).sum()

        cls = {
            "rigid": RigidRegistration,
            "affine": AffineRegistration,
            "syn": SyNRegistration,
        }[model]
        options = {"scaling": False} if model == "rigid" else {}
        if linear:
            options.update(loss_type="custom", custom_loss=global_ncc)
        else:
            options.update(
                loss_type="cc",
                cc_kernel_size=5,
                smooth_grad_sigma=1.5,
                smooth_warp_sigma=0.5,
            )
        method = cls(
            scales=[1],
            iterations=[iterations],
            fixed_images=targets,
            moving_images=sources,
            optimizer="Adam",
            optimizer_lr=0.01 if linear else 0.1,
            max_tolerance_iters=iterations + 1,
            blur=False,
            progress_bar=False,
            **options,
        )
        if not linear:

            def count_loss(*_):
                nonlocal calls
                calls += 1

            method.loss_fn.register_forward_hook(count_loss)
        method.optimize()
        if calls != iterations:
            raise RuntimeError(
                f"FireANTs performed {calls}/{iterations} requested updates"
            )
        # The public SyN accessor includes its own reverse-half inversion.
        with torch.no_grad():
            grid = method.get_warped_coordinates(targets, sources)
            original = torch.tensor(
                raw[start:stop, 0, None] if planar else raw[start:stop, None],
                device=device,
            )
            sampled = functional.grid_sample(
                original, grid, align_corners=True, padding_mode="border"
            )[:, 0]
            size = torch.tensor(
                shape[1:][::-1] if planar else shape[::-1], device=device
            )
            pixels = (grid + 1) * ((size - 1) / 2)
            valid = ((pixels >= -0.5) & (pixels < size - 0.5)).all(dim=-1)
            fill = (
                original.flatten(1).amin(1).reshape(count, *([1] * (sampled.ndim - 1)))
            )
            sampled = torch.where(valid, sampled, fill).cpu().numpy()
            output[start:stop] = sampled[:, None] if planar else sampled
            if export:
                points = _grid(shape)
                active = [1, 2] if planar else [0, 1, 2]
                for local in range(count):
                    native = pixels[local].flip(-1).cpu().numpy().reshape(-1, len(axes))
                    world = points.copy()
                    world[active] = (
                        (native.astype(np.float64) - (np.array(shape)[active] - 1) / 2)
                        * SPACING[active]
                    ).T
                    maps.append(
                        _point_rms(
                            world, points, amplitudes[start + local], model, shape
                        )
                    )
        counts.extend([calls] * count)
        statuses.extend(["max_iterations"] * count)
    return (
        output,
        maps if export else None,
        {
            "updates": counts,
            "statuses": statuses,
            "batch_size": batch,
            "metric": "masked global NCC" if linear else "native radius-2 CC",
            "optimizer": "Adam",
            "learning_rate": 0.01 if linear else 0.1,
            "fused_ops_available": FFO_AVAILABLE,
            "fused_interpolator_enabled": USE_FFO and FFO_AVAILABLE,
        },
    )
