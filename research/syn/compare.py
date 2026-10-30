"""Compare native SyN against ITK and FireANTs; shared fixtures, fresh outputs only."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import map_coordinates

from voxreg import position_to_world, register
from voxreg.rigid import _fitting_images

ROOT = Path(__file__).resolve().parents[2]


def _basis(data):
    if data.sizes["k"] != 1:
        return np.eye(3)
    a = position_to_world(data)
    i = a[:3, 2] / np.linalg.norm(a[:3, 2])
    j = a[:3, 1] - i * (i @ a[:3, 1])
    return np.column_stack((i, j / np.linalg.norm(j)))


def _header(affine, fixed, basis):
    planar = basis.shape[1] == 2
    lattice = basis.T @ affine[:3, 1 if planar else 0 : 3]
    origin = basis.T @ (affine[:3, 3] - position_to_world(fixed)[:3, 3])
    rows = np.arange(len(origin)) if planar else np.arange(3)[::-1]
    lattice = lattice[rows, ::-1]
    origin = origin[rows]
    spacing = np.linalg.norm(lattice, axis=0)
    return origin, spacing, lattice / spacing


def resample(moving, fixed, fields):
    """Common original-intensity float64 resampling, not a vendor metric score."""
    af, am = position_to_world(fixed), position_to_world(moving)
    points = (
        af
        @ np.vstack((np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size))))
    )[:3]
    frames = moving.values if "time" in moving.dims else moving.values[None]
    sampled = []
    masks = []
    for source, field in zip(frames, fields, strict=True):
        query = np.linalg.solve(
            am[:3, :3], points + field.reshape(-1, 3).T - am[:3, 3, None]
        )
        valid = (
            (query >= -1e-5) & (query <= np.array(source.shape)[:, None] - 1 + 1e-5)
        ).all(axis=0)
        values = map_coordinates(
            source.astype(float), query, order=1, mode="nearest", prefilter=False
        )
        sampled.append(
            np.where(valid, values, source.min())
            .reshape(fixed.shape)
            .astype(source.dtype)
        )
        masks.append(valid.reshape(fixed.shape))
    return np.array(sampled), np.array(masks)


def _complete(vectors, basis, fixed, seed):
    af = position_to_world(fixed)
    planar = basis.shape[1] == 2
    world = vectors @ basis.T
    if planar:
        world = world[None]
    points = (
        af
        @ np.vstack((np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size))))
    )[:3].T.reshape(*fixed.shape, 3)
    return world @ seed[:3, :3].T + points @ (seed[:3, :3] - np.eye(3)).T + seed[:3, 3]


def fit_itk(moving, fixed, seed, iterations, threads):
    import itk

    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(threads)
    planar = fixed.sizes["k"] == 1
    basis = _basis(fixed)
    d = basis.shape[1]
    raw = moving.values if "time" in moving.dims else moving.values[None]
    fit, target = _fitting_images(
        raw[:, 0] if planar else raw,
        fixed.values[0] if planar else fixed.values,
        "none",
    )
    I = itk.Image[itk.F, d]

    def image(values, affine):
        result = itk.image_from_array(np.ascontiguousarray(values))
        origin, spacing, direction = _header(affine, fixed, basis)
        result.SetOrigin(origin.tolist())
        result.SetSpacing(spacing.tolist())
        result.SetDirection(itk.matrix_from_array(direction))
        return result

    target_image = image(target, position_to_world(fixed))
    forwards = []
    inverses = []
    for frame in fit:
        source = image(frame, np.linalg.inv(seed) @ position_to_world(moving))
        transform = itk.DisplacementFieldTransform[itk.D, d].New()
        for inverse in (False, True):
            field = itk.Image[itk.Vector[itk.D, d], d].New()
            field.CopyInformation(target_image)
            field.SetRegions(target_image.GetLargestPossibleRegion())
            field.Allocate()
            zero = itk.Vector[itk.D, d]()
            zero.Fill(0)
            field.FillBuffer(zero)
            (
                transform.SetInverseDisplacementField
                if inverse
                else transform.SetDisplacementField
            )(field)
        method = itk.SyNImageRegistrationMethod[I, I].New()
        method.SetInitialTransform(transform)
        method.InPlaceOn()
        method.SetFixedImage(target_image)
        method.SetMovingImage(source)
        metric = itk.ANTSNeighborhoodCorrelationImageToImageMetricv4[I, I].New()
        radius = itk.Size[d]()
        radius.Fill(2)
        metric.SetRadius(radius)
        metric.SetUseMovingImageGradientFilter(False)
        metric.SetUseFixedImageGradientFilter(False)
        method.SetMetric(metric)
        method.SetNumberOfLevels(1)
        method.SetNumberOfIterationsPerLevel([iterations])
        method.SetShrinkFactorsPerLevel([1])
        method.SetSmoothingSigmasPerLevel([0])
        method.SetConvergenceWindowSize(iterations + 1)
        method.SetLearningRate(0.1)
        method.SetGaussianSmoothingVarianceForTheUpdateField(1.5**2)
        method.SetGaussianSmoothingVarianceForTheTotalField(0.5**2)
        method.Update()
        transform = method.GetTransform()
        vectors = itk.array_from_image(transform.GetDisplacementField())
        if not planar:
            vectors = vectors[..., ::-1]
        forwards.append(_complete(vectors, basis, fixed, seed))
        if np.array_equal(seed, np.eye(4)) and moving.shape[-3:] == fixed.shape:
            vectors = itk.array_from_image(transform.GetInverseDisplacementField())
            if not planar:
                vectors = vectors[..., ::-1]
            inv = vectors @ basis.T
            inverses.append(inv[None] if planar else inv)
    fields = np.array(forwards)
    registered, valid = resample(moving, fixed, fields)
    return fields, np.array(inverses) if inverses else None, registered, valid


def fit_fireants(moving, fixed, seed, iterations, device, threads):
    import SimpleITK as sitk
    from fireants.io.image import BatchedImages, Image
    from fireants.registration.syn import SyNRegistration

    torch.set_num_threads(threads)
    planar = fixed.sizes["k"] == 1
    basis = _basis(fixed)
    raw = moving.values if "time" in moving.dims else moving.values[None]
    fit, target = _fitting_images(
        raw[:, 0] if planar else raw,
        fixed.values[0] if planar else fixed.values,
        "none",
    )

    def image(values, affine):
        result = sitk.GetImageFromArray(np.ascontiguousarray(values))
        origin, spacing, direction = _header(affine, fixed, basis)
        result.SetOrigin(origin.tolist())
        result.SetSpacing(spacing.tolist())
        result.SetDirection(direction.ravel().tolist())
        return Image(result, device=device)

    stationary = BatchedImages([image(target, position_to_world(fixed)) for _ in fit])
    sources = BatchedImages(
        [image(frame, np.linalg.inv(seed) @ position_to_world(moving)) for frame in fit]
    )
    method = SyNRegistration(
        scales=[1],
        iterations=[iterations],
        fixed_images=stationary,
        moving_images=sources,
        loss_type="cc",
        cc_kernel_size=5,
        optimizer="Adam",
        optimizer_lr=0.1,
        smooth_grad_sigma=1.5,
        smooth_warp_sigma=0.5,
        blur=False,
        max_tolerance_iters=iterations + 1,
        progress_bar=False,
    )
    method.optimize()
    # The public forward accessor performs FireANTs' own reverse-warp inversion.
    coordinates = (
        method.get_warped_coordinates(stationary, sources).detach().cpu().numpy()
    )
    shape = np.array(raw.shape[2:] if planar else raw.shape[1:])
    pixels = (coordinates[..., ::-1] + 1) * ((shape - 1) / 2)
    af = position_to_world(moving)
    # For nonidentity shared seeds, normalized coordinates already refer to the
    # original moving voxels; account for unequal native grids/origins explicitly.
    fixed_points = (
        position_to_world(fixed)
        @ np.vstack((np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size))))
    )[:3].T.reshape(*fixed.shape, 3)
    if planar:
        pixels = pixels[:, None]
    native = (
        np.concatenate((np.zeros((*pixels.shape[:-1], 1)), pixels), axis=-1)
        if planar
        else pixels
    )
    fields = native @ af[:3, :3].T + af[:3, 3] - fixed_points
    registered, valid = resample(moving, fixed, fields)
    return fields, None, registered, valid


def fit_voxreg(moving, fixed, seed, iterations, device, threads):
    torch.set_num_threads(threads)
    result = register(
        moving,
        fixed,
        return_transforms=True,
        transform="syn",
        initial_transform=seed,
        intensity_scaling="none",
        device=device,
        max_iterations=iterations,
        convergence_window_size=iterations + 1,
        progress=False,
    )
    spatial = "time" not in moving.dims
    return (
        result.transforms.values[None] if spatial else result.transforms.values,
        result.inverse_transforms.values[None]
        if spatial
        else result.inverse_transforms.values,
        result.registered.values[None] if spatial else result.registered.values,
        result.valid.values[None] if spatial else result.valid.values,
    )


def score(moving, fixed, fields, truth, seed):
    values, valid = resample(moving, fixed, fields)
    correlations = []
    for data, mask in zip(values, valid, strict=True):
        x, y = data[mask].astype(float), fixed.values[mask].astype(float)
        if len(x) < 2:
            correlations.append(None)
            continue
        x -= x.mean()
        y -= y.mean()
        denominator = (x @ x) * (y @ y)
        correlations.append(
            float((x @ y) ** 2 / denominator) if denominator > 0 else None
        )
    basis = _basis(fixed)
    affine = position_to_world(fixed)
    planar = fixed.sizes["k"] == 1
    lattice = basis.T @ affine[:3, 1 if planar else 0 : 3]
    points = (
        affine
        @ np.vstack((np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size))))
    )[:3].T.reshape(*fixed.shape, 3)
    seed_displacement = points @ (seed[:3, :3] - np.eye(3)).T + seed[:3, 3]
    minima = []
    for field in fields:
        residual = (field - seed_displacement) @ np.linalg.inv(seed[:3, :3]).T @ basis
        if planar:
            residual = residual[0]
        components = np.moveaxis(residual, -1, 0)
        derivatives = np.stack(
            np.gradient(components, axis=tuple(range(1, components.ndim))), axis=-1
        )
        jacobian = np.moveaxis(derivatives, 0, -2) @ np.linalg.inv(lattice) + np.eye(
            len(lattice)
        )
        minima.append(float(np.linalg.det(jacobian).min()))
    report = {"ncc_squared": correlations, "min_residual_jacobian": minima}
    if truth is not None:
        ell = np.linalg.norm(
            position_to_world(fixed)[:3, 1 if fixed.sizes["k"] == 1 else 0 : 3], axis=0
        ).min()
        error = np.linalg.norm(fields - truth, axis=-1) / ell
        report["truth_rms_px"] = np.sqrt(np.mean(error**2, axis=(1, 2, 3))).tolist()
        report["truth_max_px"] = error.reshape(len(fields), -1).max(axis=1).tolist()
        report["truth_gate_pass"] = bool(
            np.all(np.array(report["truth_rms_px"]) < 0.25)
        )
    return report


def inputs(case):
    # Research-only access to the package-wide factories: no copied phantom.
    sys.path.insert(0, str(ROOT / "tests"))
    import conftest

    factory = conftest.nonlinear_pair_factory.__wrapped__(
        conftest.displacement_factory.__wrapped__(
            conftest.linear_pair_factory.__wrapped__()
        )
    )
    planar = case in ("2d-small", "2d-medium")
    shape = (
        (64, 80)
        if case == "2d-medium"
        else (32, 40, 48)
        if case == "3d-medium"
        else (32, 32, 32)
        if case == "3d-batch8"
        else None
    )
    if case == "real-pair":
        sys.path.insert(0, str(ROOT / "research/confusius-examples"))
        from compare_same_subject import load_inputs

        fixed, moving, provenance = load_inputs()
        linear = register(
            moving,
            fixed,
            return_transforms=True,
            transform="rigid",
            initialization="center_geometry",
            intensity_scaling="none",
            max_iterations=400,
            progress=False,
        )
        return fixed, moving, None, None, linear.transforms.values, provenance
    fixed, moving, truth, inverse = factory(planar, shape=shape, sheared=False)
    if case == "3d-batch8":
        frames = []
        truths = []
        inverses = []
        for amplitude in np.linspace(1.4, 2.1, 8):
            _, m, t, i = factory(False, shape=shape, amplitude=amplitude, sheared=False)
            frames.append(m)
            truths.append(t)
            inverses.append(i)
        moving = xr.concat(frames, dim=xr.IndexVariable("time", np.arange(8)))
        truth, inverse = np.array(truths), np.array(inverses)
    else:
        truth, inverse = truth[None], inverse[None]
    return fixed, moving, truth, inverse, np.eye(4), None


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    importlib.import_module("itk")
    from fireants.interpolator import fireants_interpolator

    importlib.import_module("fireants.registration.syn")  # load before timing

    report = {
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["voxreg", "torch", "itk", "fireants", "numpy"]
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "fireants_fused": fireants_interpolator.use_ffo,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "cpu_threads": args.threads,
        "cases": {},
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *sorted((ROOT / "src/voxreg").glob("*.py")),
                ROOT / "tests/conftest.py",
                Path(__file__),
            ]
        },
    }
    arms = args.backends
    for case in args.cases:
        fixed, moving, truth, inverse, seed, provenance = inputs(case)
        folder = output / case
        folder.mkdir()
        np.savez_compressed(
            folder / "inputs.npz",
            fixed=fixed.values,
            moving=moving.values,
            affine_fixed=position_to_world(fixed),
            affine_moving=position_to_world(moving),
            seed=seed,
            truth=truth if truth is not None else np.array([]),
        )
        result = {
            "shape": list(fixed.shape),
            "frames": moving.sizes.get("time", 1),
            "provenance": provenance,
            "backends": {},
        }
        report["cases"][case] = result
        for arm in arms:
            print(case, arm, flush=True)
            device = "cuda" if arm.endswith("cuda") else "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                continue
            try:
                function = (
                    fit_itk
                    if arm == "itk"
                    else fit_fireants
                    if arm.startswith("fireants")
                    else fit_voxreg
                )
                durations = []
                samples = []
                for repeat in range(args.repeats + 1):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    start = perf_counter()
                    options = (
                        {"threads": args.threads}
                        if arm == "itk"
                        else {"threads": args.threads, "device": device}
                    )
                    fields, inv, registered, valid = function(
                        moving, fixed, seed, args.iterations, **options
                    )
                    if device == "cuda":
                        torch.cuda.synchronize()
                    durations.append(perf_counter() - start)
                    samples.append(fields.copy())
                row = score(moving, fixed, fields, truth, seed)
                row.update(
                    cold_seconds=durations[0],
                    warm_seconds=durations[1:],
                    warm_median_seconds=float(np.median(durations[1:])),
                    repeat_max_world_mm=float(
                        max(np.abs(v - fields).max() for v in samples)
                    ),
                    finite=bool(np.isfinite(fields).all()),
                )
                if inv is not None and inverse is not None:
                    ell = np.linalg.norm(
                        position_to_world(fixed)[
                            :3, 1 if fixed.sizes["k"] == 1 else 0 : 3
                        ],
                        axis=0,
                    ).min()
                    row["inverse_truth_rms_px"] = (
                        np.sqrt(
                            np.mean(
                                np.sum((inv - inverse) ** 2, axis=-1), axis=(1, 2, 3)
                            )
                        )
                        / ell
                    ).tolist()
                np.savez_compressed(
                    folder / f"{arm}.npz",
                    fields=fields,
                    inverse=inv if inv is not None else np.array([]),
                    registered=registered,
                    valid=valid,
                )
                result["backends"][arm] = row
                print(row, flush=True)
            except (RuntimeError, ValueError, AssertionError) as error:
                result["backends"][arm] = {
                    "error": type(error).__name__,
                    "message": str(error),
                }
                print(type(error).__name__, str(error), flush=True)
            (output / "results.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
        if {"voxreg-cpu", "voxreg-cuda"}.issubset(result["backends"]) and all(
            "error" not in result["backends"][a] for a in ("voxreg-cpu", "voxreg-cuda")
        ):
            cpu = np.load(folder / "voxreg-cpu.npz")["fields"]
            gpu = np.load(folder / "voxreg-cuda.npz")["fields"]
            ell = np.linalg.norm(
                position_to_world(fixed)[:3, 1 if fixed.sizes["k"] == 1 else 0 : 3],
                axis=0,
            ).min()
            result["cpu_cuda_max_point_px"] = float(
                np.linalg.norm(cpu - gpu, axis=-1).max() / ell
            )
            result["precision_gate_pass"] = result["cpu_cuda_max_point_px"] < 0.001
        (output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[
            "2d-small",
            "2d-medium",
            "3d-small",
            "3d-medium",
            "3d-batch8",
            "real-pair",
        ],
        default=["2d-small", "2d-medium", "3d-small", "3d-medium", "3d-batch8"],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["voxreg-cpu", "voxreg-cuda", "itk", "fireants-cpu", "fireants-cuda"],
        default=["voxreg-cpu", "voxreg-cuda", "itk", "fireants-cpu", "fireants-cuda"],
    )
    arguments = parser.parse_args()
    if min(arguments.iterations, arguments.repeats, arguments.threads) < 1:
        parser.error("iterations, repeats and threads must be positive")
    run(arguments)
