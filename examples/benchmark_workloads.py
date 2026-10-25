"""Standardized pair/series workloads for benchmark_registration.py, not a second runner."""

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np


def fingerprint(path):
    """Hash inputs/source without retaining a second copy of large archives."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_dataset(path, workload):
    """Load common float32 inputs; all affines map array positions to world z/y/x."""
    if workload not in ("pair", "series"):
        raise ValueError("workload must be pair or series")
    with np.load(path, allow_pickle=False) as z:
        provenance = (
            json.loads(str(z["provenance"].item())) if "provenance" in z else {}
        )
        if "values" in z:
            if workload != "series":
                raise ValueError(
                    "The legacy values/time archive is a series, not a pair."
                )
            moving = z["values"]
            conversion = np.eye(4)
            for axis, dim in enumerate(("k", "j", "i")):
                coords = z[dim]
                if (
                    coords.ndim != 1
                    or len(coords) != moving.shape[axis + 1]
                    or not np.issubdtype(coords.dtype, np.integer)
                ):
                    raise ValueError(
                        "Expected integer native coordinates matching the spatial shape."
                    )
                conversion[axis, 3] = int(coords[0])
                if len(coords) > 1:
                    steps = np.diff(coords.astype(float))
                    if steps[0] == 0 or not np.allclose(
                        steps, steps[0], rtol=0, atol=1e-10
                    ):
                        raise ValueError(
                            "Native coordinates must be uniform and monotonic."
                        )
                    conversion[axis, axis] = steps[0]
            am = af = z["voxel_to_world"] @ conversion
            fixed = moving[0]
        else:
            moving, fixed = z["moving"], z["fixed"]
            am, af = z["moving_affine"], z["fixed_affine"]
        time = z["time"].copy() if "time" in z else None
        truth = z["truth"].copy() if "truth" in z else None
    if workload == "pair":
        if moving.ndim != 3:
            raise ValueError("Pair inputs require spatial-only moving/fixed arrays.")
        moving = moving[None]
    if moving.ndim != 4 or fixed.ndim != 3 or 0 in moving.shape or 0 in fixed.shape:
        raise ValueError(
            "Expected moving (time,k,j,i) and fixed (k,j,i), without empty axes."
        )
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (moving, fixed)):
        raise ValueError(
            "Standardized inputs must be finite float32; conversion belongs in input preparation."
        )
    if np.ptp(fixed) == 0 or np.any(np.ptp(moving, axis=(1, 2, 3)) == 0):
        raise ValueError("Benchmark images must have nonzero intensity variance.")
    planar = moving.shape[1] == fixed.shape[0] == 1
    for shape, affine in ((moving.shape[1:], am), (fixed.shape, af)):
        if (
            affine.shape != (4, 4)
            or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
        ):
            raise ValueError("Expected finite homogeneous position-to-world affines.")
        linear = affine[:3, :3]
        spacing = np.linalg.norm(linear, axis=0)
        if np.any(spacing == 0) or not np.allclose(
            (linear / spacing).T @ (linear / spacing), np.eye(3), atol=1e-6, rtol=0
        ):
            raise ValueError(
                "This cross-backend protocol requires orthogonal image axes, not shear."
            )
        if min(shape[1:] if planar else shape) < 32:
            raise ValueError(
                "At least 32 samples per active axis are required (FireANTs minimum scale size)."
            )
        if planar and (
            not np.allclose(linear, np.diag(np.diag(linear)))
            or np.any(np.diag(linear) <= 0)
        ):
            raise ValueError(
                "The 2D comparison requires positive axis-aligned planes; 3D supports obliquity."
            )
    if planar and not np.isclose(am[0, 3], af[0, 3], atol=1e-8, rtol=0):
        raise ValueError("Slice pairs must be coplanar.")
    if workload == "series":
        if (
            len(moving) < 2
            or time is None
            or time.shape != (len(moving),)
            or len(np.unique(time)) != len(time)
        ):
            raise ValueError("Series need at least two frames and unique time labels.")
        if not np.array_equal(fixed, moving[0]) or not np.array_equal(af, am):
            raise ValueError(
                "The series protocol uses the first frame as an exact reference."
            )
    if truth is not None:
        if truth.shape == (4, 4):
            truth = truth[None]
        if (
            truth.shape != (len(moving), 4, 4)
            or not np.isfinite(truth).all()
            or not np.allclose(truth[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or not np.allclose(
                truth[:, :3, :3].transpose(0, 2, 1) @ truth[:, :3, :3],
                np.eye(3),
                rtol=0,
                atol=1e-8,
            )
            or not np.allclose(np.linalg.det(truth[:, :3, :3]), 1, rtol=0, atol=1e-8)
        ):
            raise ValueError(
                "Truth must contain one finite rigid world pull matrix per frame."
            )
    return {
        "moving": moving,
        "fixed": fixed,
        "am": am,
        "af": af,
        "time": time,
        "truth": truth,
        "planar": planar,
        "reference": workload == "series",
        "provenance": provenance,
    }


def fit_images(data, scaling):
    """Common float32 fitting preprocessing, included in every timed call."""
    moving, fixed = data["moving"], data["fixed"]
    scale = max(float(np.abs(moving).max()), float(np.abs(fixed).max()))
    if scaling == "db":
        if moving.min() < 0 or fixed.min() < 0:
            raise ValueError(
                "dB fitting requires nonnegative inputs; use --intensity-scaling none."
            )
        return tuple(
            10 * np.log10(np.maximum(a / scale, 1e-12)) for a in (moving, fixed)
        )
    return moving / scale, fixed / scale


def image_geometry(affine, planar):
    """Array-position z/y/x geometry to ITK/SimpleITK physical x/y[/z]."""
    axes = [2, 1] if planar else [2, 1, 0]
    linear = affine[np.ix_(axes, axes)]
    spacing = np.linalg.norm(linear, axis=0)
    return spacing, affine[axes, 3], linear / spacing


def coordinates(data, matrix):
    """Independent broadcast affine map and pixel-center support, no dense GEMM."""
    transform = np.linalg.solve(data["am"], matrix @ data["af"])
    lattice = np.ogrid[tuple(slice(0, n) for n in data["fixed"].shape)]
    mapped, valid = [], np.ones(data["fixed"].shape, dtype=bool)
    for axis, size in enumerate(data["moving"].shape[1:]):
        value = transform[axis, 3] + sum(
            transform[axis, col] * lattice[col] for col in range(3)
        )
        mapped.append(value)
        valid &= (value >= -1e-5) & (value <= size - 1 + 1e-5)
    return np.stack(mapped), valid


def run_itk_standard(data, args, threads):
    """Native 2D/3D ITK GD; no cached fitting image outside the timed boundary."""
    import itk

    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(threads)
    planar = data["planar"]
    moving_fit, fixed_fit = fit_images(data, args.intensity_scaling)

    def image(values, affine):
        result = itk.image_from_array(
            np.asarray(values[0] if planar else values, dtype=np.float64)
        )
        spacing, origin, direction = image_geometry(affine, planar)
        result.SetSpacing(tuple(spacing))
        result.SetOrigin(tuple(origin))
        result.SetDirection(itk.matrix_from_array(direction.copy()))
        return result

    fixed = image(fixed_fit, data["af"])
    outputs, matrices, updates, statuses, validity, stops = [], [], [], [], [], []
    for index, raw in enumerate(data["moving"]):
        if data["reference"] and index == 0:
            output, matrix, count, status, stop = (
                raw.copy(),
                np.eye(4),
                0,
                "reference",
                "reference",
            )
        else:
            moving = image(moving_fit[index], data["am"])
            image_type = type(moving)
            metric = itk.CorrelationImageToImageMetricv4[image_type, image_type].New()
            metric.SetUseFixedImageGradientFilter(False)
            metric.SetUseMovingImageGradientFilter(False)
            metric.SetMaximumNumberOfWorkUnits(threads)
            transform = (itk.Euler2DTransform if planar else itk.Euler3DTransform)[
                itk.D
            ].New()
            center = (data["af"] @ np.r_[(np.array(data["fixed"].shape) - 1) / 2, 1])[
                :3
            ]
            transform.SetCenter(tuple(center[[2, 1] if planar else [2, 1, 0]]))
            optimizer = itk.GradientDescentOptimizerv4Template[itk.D].New(
                LearningRate=0.1,
                NumberOfIterations=args.max_iterations,
                MinimumConvergenceValue=1e-6,
                ConvergenceWindowSize=args.max_iterations + 1
                if args.policy == "fixed"
                else 50,
                DoEstimateLearningRateOnce=False,
                DoEstimateLearningRateAtEachIteration=False,
                ReturnBestParametersAndValue=True,
            )
            optimizer.SetScalesEstimator(
                itk.RegistrationParameterScalesFromPhysicalShift[type(metric)].New(
                    Metric=metric, TransformForward=True
                )
            )
            registration = itk.ImageRegistrationMethodv4[image_type, image_type].New(
                FixedImage=fixed,
                MovingImage=moving,
                Metric=metric,
                Optimizer=optimizer,
                InitialTransform=transform,
                InPlace=True,
            )
            registration.SetNumberOfWorkUnits(threads)
            registration.SetNumberOfLevels(1)
            registration.SetShrinkFactorsPerLevel([1])
            registration.SetSmoothingSigmasPerLevel([0.0])
            registration.Update()
            resampler = itk.ResampleImageFilter[image_type, image_type].New(
                Input=image(raw, data["am"]),
                ReferenceImage=fixed,
                UseReferenceImage=True,
                Transform=transform,
                DefaultPixelValue=float(raw.min()),
                Interpolator=itk.LinearInterpolateImageFunction[
                    image_type, itk.D
                ].New(),
            )
            resampler.SetNumberOfWorkUnits(threads)
            resampler.Update()
            output = itk.array_from_image(resampler.GetOutput()).astype(np.float32)
            if planar:
                output = output[None]
            axes = [2, 1] if planar else [2, 1, 0]
            matrix = np.eye(4)
            matrix[np.ix_(axes, axes)] = np.asarray(transform.GetMatrix())
            matrix[axes, 3] = transform.TransformPoint((0.0,) * len(axes))
            count = int(optimizer.GetCurrentIteration())
            stop = optimizer.GetStopConditionDescription()
            status = (
                "max_iterations" if count >= args.max_iterations else "backend_stop"
            )
        _, valid = coordinates(data, matrix)
        outputs.append(np.where(valid, output, raw.min()))
        matrices.append(matrix)
        updates.append(count)
        statuses.append(status)
        validity.append(valid)
        stops.append(stop)
    return {
        "registered": np.stack(outputs),
        "transforms": np.stack(matrices),
        "n_iterations": np.array(updates),
        "status": np.array(statuses),
        "valid": np.stack(validity),
        "stop_condition": np.array(stops),
    }


def run_fireants_standard(data, args, device, batch):
    """Native float32 FireANTs; masked squared global NCC, no pyramids or scaling."""
    import SimpleITK as sitk
    import torch
    import torch.nn.functional as F
    from fireants.io.image import BatchedImages, Image
    from fireants.registration.rigid import RigidRegistration

    planar = data["planar"]
    moving_fit, fixed_fit = fit_images(data, args.intensity_scaling)

    def image(values, affine):
        values = values[0] if planar else values
        result = sitk.GetImageFromArray(
            np.stack((values, np.ones_like(values)), axis=-1), isVector=True
        )
        spacing, origin, direction = image_geometry(affine, planar)
        result.SetSpacing(tuple(spacing))
        result.SetOrigin(tuple(origin))
        result.SetDirection(tuple(direction.ravel()))
        return Image(result, device=device)

    reference = image(fixed_fit, data["af"])
    outputs, matrices, updates, statuses, masks = [], [], [], [], []
    first = int(data["reference"])
    if first:
        outputs.append(data["moving"][:1].copy())
        matrices.append(np.eye(4)[None])
        updates.append(0)
        statuses.append("reference")
        masks.append(np.ones_like(data["moving"][:1], dtype=bool))
    for start in range(first, len(data["moving"]), batch):
        end = min(start + batch, len(data["moving"]))
        count = end - start
        fixed = BatchedImages([reference] * count, optimize_memory=False)
        moving = BatchedImages([image(v, data["am"]) for v in moving_fit[start:end]])
        calls = 0

        def loss(moved, target):
            nonlocal calls
            calls += 1
            axes = tuple(range(1, moved.ndim - 1))
            weights = (moved[:, 1] >= 1 - 1e-5).to(moved.dtype)
            n = weights.sum(dim=axes, keepdim=True).clamp_min(1)
            x, y = moved[:, 0], target[:, 0]
            x = (x - (x * weights).sum(dim=axes, keepdim=True) / n) * weights
            y = (y - (y * weights).sum(dim=axes, keepdim=True) / n) * weights
            vx, vy = x.square().sum(dim=axes), y.square().sum(dim=axes)
            denominator = vx * vy
            meaningful = (
                (n.reshape(len(moved)) > 1)
                & (vx > 0)
                & (vy > 0)
                & torch.isfinite(denominator)
                & (denominator > torch.finfo(moved.dtype).tiny)
            )
            safe = torch.where(meaningful, denominator, torch.ones_like(denominator))
            values = -(x * y).sum(dim=axes).square() / safe
            return torch.where(meaningful, values, 0).sum()

        registration = RigidRegistration(
            scales=[1],
            iterations=[args.max_iterations],
            fixed_images=fixed,
            moving_images=moving,
            loss_type="custom",
            custom_loss=loss,
            optimizer="Adam",
            optimizer_lr=0.01,
            scaling=False,
            blur=False,
            max_tolerance_iters=args.max_iterations + 1
            if args.policy == "fixed"
            else 10,
            tolerance=1e-6,
            progress_bar=False,
        )
        registration.optimize()
        if args.policy == "fixed" and calls != args.max_iterations:
            raise RuntimeError(f"FireANTs fixed-budget mismatch: {calls}")
        with torch.no_grad():
            native = registration.get_rigid_matrix().cpu().numpy()
            axes = [2, 1] if planar else [2, 1, 0]
            world = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            world[np.ix_(np.arange(count), axes, axes)] = native[
                :, : len(axes), : len(axes)
            ]
            world[:, axes, 3] = native[:, : len(axes), len(axes)]
            grid = registration.get_warped_coordinates(fixed, moving)
            raw = data["moving"][start:end]
            original = torch.tensor(
                raw[:, 0, None] if planar else raw[:, None], device=device
            )
            sampled = F.grid_sample(
                original, grid, align_corners=True, padding_mode="border"
            )[:, 0]
            size = np.array(raw.shape[2:] if planar else raw.shape[1:])[::-1]
            tolerance = torch.tensor(2e-5 / (size - 1), device=device, dtype=grid.dtype)
            valid = ((grid >= -1 - tolerance) & (grid <= 1 + tolerance)).all(dim=-1)
            fill = (
                original.flatten(1)
                .amin(dim=1)
                .reshape((count,) + (1,) * (sampled.ndim - 1))
            )
            sampled = torch.where(valid, sampled, fill)
            if planar:
                sampled, valid = sampled[:, None], valid[:, None]
            outputs.append(sampled.cpu().numpy())
            masks.append(valid.cpu().numpy())
            matrices.append(world)
        updates.extend([calls] * count)
        statuses.extend(
            ["max_iterations" if calls == args.max_iterations else "backend_monitor"]
            * count
        )
    return {
        "registered": np.concatenate(outputs),
        "transforms": np.concatenate(matrices),
        "n_iterations": np.array(updates),
        "status": np.array(statuses),
        "valid": np.concatenate(masks),
    }


def standard_worker(args, measure):
    """Construct one resident-host workload and use the shared measurement runner."""
    setup = perf_counter()
    if args.input.resolve() in (
        args.output.resolve(),
        args.output.with_suffix(".npz").resolve(),
    ):
        raise ValueError("Output must not overwrite the input.")
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise ValueError("Refusing to overwrite existing worker outputs.")
    data = load_dataset(args.input, args.workload)
    fit_images(data, args.intensity_scaling)  # Fail before starting measurements.
    threads = 8 if args.case.endswith("-t8") else 1
    metadata = {
        "case": args.case,
        "pid": os.getpid(),
        "protocol": "standardized-v1",
        "workload": args.workload,
        "policy": args.policy,
        "max_iterations": args.max_iterations,
        "intensity_scaling": args.intensity_scaling,
        "input_shape": list(data["moving"].shape),
        "fixed_shape": list(data["fixed"].shape),
        "cpu_threads": threads,
        "input_sha256": fingerprint(args.input),
        "numpy": np.__version__,
        "confusius": version("confusius"),
        "xarray": version("xarray"),
    }
    if args.case.startswith("itk"):
        import itk

        itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(threads)
        metadata.update(
            itk=itk.Version.GetITKVersion(),
            precision="float32 common fitting samples; double ITK images/metric/transform/interpolation; float32 output",
            optimizer="GD with physical-shift scales, rate .1, return best",
            stopping="disabled until budget+1"
            if args.policy == "fixed"
            else "ITK window50/minimum1e-6",
            batch_size=1,
        )

        def run():
            return run_itk_standard(data, args, threads)
    else:
        import torch

        torch.set_num_threads(threads)
        device = "cuda" if "cuda" in args.case.split("-") else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        fireants = args.case.startswith("fireants")
        batch = int(args.case.split("-")[2 if fireants else 1])
        metadata.update(torch=torch.__version__, batch_size=batch)
        if fireants:
            metadata.update(
                fireants=version("fireants"),
                simpleitk=version("SimpleITK"),
                precision="native float32 images/geometry/state/NCC/interpolation",
                optimizer="Adam rate .01; quaternion in 3D",
                stopping="disabled until budget+1"
                if args.policy == "fixed"
                else "native scalar batch-loss slope monitor10/tolerance1e-6; not ITK convergence",
            )

            def run():
                return run_fireants_standard(data, args, device, batch)
        else:
            from confusius import create_voxeldata

            from voxreg import register_rigid

            kwargs = {"voxel_to_world": data["am"]}
            if data["reference"]:
                kwargs["time"] = data["time"]
            moving = create_voxeldata(
                data["moving"] if data["reference"] else data["moving"][0],
                dims=("time", "k", "j", "i") if data["reference"] else ("k", "j", "i"),
                **kwargs,
            )
            fixed = (
                None
                if data["reference"]
                else create_voxeldata(
                    data["fixed"], dims=("k", "j", "i"), voxel_to_world=data["af"]
                )
            )
            metadata.update(
                precision="float32 fitting/final interpolation; float64 geometry/state/NCC/matrices",
                optimizer="Adam physically scaled, rate .1; rotation vector in 3D",
                stopping="disabled until budget+1"
                if args.policy == "fixed"
                else "independent ITK-formula window50/minimum1e-6; coarse whole-batch polling",
            )

            def run():
                r = register_rigid(
                    moving,
                    fixed,
                    device=device,
                    batch_size=batch,
                    max_iterations=args.max_iterations,
                    intensity_scaling=args.intensity_scaling,
                    convergence_window_size=args.max_iterations + 1
                    if args.policy == "fixed"
                    else 50,
                    progress=False,
                )
                return {
                    "registered": r.registered.values
                    if data["reference"]
                    else r.registered.values[None],
                    "transforms": r.transforms.values
                    if data["reference"]
                    else r.transforms.values[None],
                    "n_iterations": np.atleast_1d(r.diagnostics.n_iterations.values),
                    "status": np.atleast_1d(r.diagnostics.status.values),
                    "valid": r.valid.values
                    if data["reference"]
                    else r.valid.values[None],
                }

    metadata["setup_seconds"] = perf_counter() - setup
    measure(args, run, metadata)


def score_standard(input_path, files, workload, scaling):
    """Independent float64 SciPy scores on common support; missing truth is not error."""
    from scipy.ndimage import map_coordinates

    data = load_dataset(input_path, workload)
    moving, fixed = fit_images(data, scaling)
    common = np.broadcast_to(
        coordinates(data, np.eye(4))[1], (len(moving), *fixed.shape)
    ).copy()
    matrices = {}
    for case, path in files.items():
        with np.load(path, allow_pickle=False) as z:
            matrices[case] = z["transforms"]
        for frame, matrix in enumerate(matrices[case]):
            common[frame] &= coordinates(data, matrix)[1]
    # Baseline uses exactly the same support as every returned transform.
    if data["truth"] is not None:
        for frame, matrix in enumerate(data["truth"]):
            common[frame] &= coordinates(data, matrix)[1]
        matrices = {"truth": data["truth"], **matrices}
    quality = {}
    for case, transforms in {
        "identity": np.broadcast_to(np.eye(4), (len(moving), 4, 4)),
        **matrices,
    }.items():
        scores, rms, maxima = [], [], []
        for frame, matrix in enumerate(transforms):
            mapped, _ = coordinates(data, matrix)
            mask = common[frame]
            sampled = map_coordinates(
                moving[frame].astype(float),
                mapped,
                order=1,
                mode="nearest",
                prefilter=False,
            )
            x, y = sampled[mask], fixed.astype(float)[mask]
            x, y = x - x.mean() if len(x) else x, y - y.mean() if len(y) else y
            denominator = float(x @ x) * float(y @ y)
            scores.append(
                float((x @ y) ** 2 / denominator)
                if len(x) > 1 and denominator > 0
                else None
            )
            if data["truth"] is not None:
                # Affine displacement RMS is exact from fixed-grid mean/covariance.
                difference = (matrix - data["truth"][frame])[:3] @ data["af"]
                shape = np.array(fixed.shape)
                mean = difference[:, :3] @ ((shape - 1) / 2) + difference[:, 3]
                mean_square = np.sum(mean**2) + np.sum(
                    difference[:, :3] ** 2 @ ((shape**2 - 1) / 12)
                )
                corners = np.array(
                    np.meshgrid(*[(0, n - 1) for n in shape], indexing="ij")
                ).reshape(3, -1)
                worst = np.linalg.norm(
                    difference[:, :3] @ corners + difference[:, 3, None], axis=0
                ).max()
                ell = np.linalg.norm(
                    data["af"][:3, 1:] if data["planar"] else data["af"][:3, :3], axis=0
                ).min()
                rms.append(float(np.sqrt(mean_square) / ell))
                maxima.append(float(worst / ell))
        selection = slice(1, None) if data["reference"] else slice(None)
        chosen = scores[selection]
        finite = [v for v in chosen if v is not None]
        quality[case] = {
            "ncc2_per_frame": scores,
            "median_ncc2": float(np.median(finite)) if finite else None,
            "min_ncc2": min(finite) if finite else None,
            "invalid_score_frames": len(chosen) - len(finite),
            "common_fraction_median": float(
                np.median(
                    common[selection].reshape(len(common[selection]), -1).mean(axis=1)
                )
            ),
        }
        if rms:
            quality[case].update(
                rms_truth_pixels=rms, max_pointwise_truth_pixels=maxima
            )
    return quality
