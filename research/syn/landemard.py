"""Whole-ref20 SyN: one voxreg call versus public FireANTs, no retained dense outputs."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import resource
from pathlib import Path
from time import perf_counter

import confusius as cf
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from compare import _basis, _header
from fireants.interpolator import fireants_interpolator
from fireants.io.image import BatchedImages, Image
from fireants.registration.syn import SyNRegistration

from voxreg import position_to_world, register
from voxreg.rigid import _fitting_images


def fit_fireants_recording(data, seed, batch_size, iterations):
    """Public native 2D API; only corrected images/masks reach CPU, never fields."""
    fixed = data.isel(time=0, drop=True)
    basis = _basis(fixed)
    affine = position_to_world(data)
    fit, target = _fitting_images(data.values[:, 0], fixed.values[0], "db")

    def image(values, geometry):
        result = sitk.GetImageFromArray(np.ascontiguousarray(values))
        origin, spacing, direction = _header(geometry, fixed, basis)
        result.SetOrigin(origin.tolist())
        result.SetSpacing(spacing.tolist())
        result.SetDirection(direction.ravel().tolist())
        return Image(result, device="cuda")

    stationary = image(target, affine)
    output = np.empty_like(data.values)
    masks = np.empty(data.shape, dtype=bool)
    for first in range(0, len(data), batch_size):
        last = min(first + batch_size, len(data))
        print(f"FireANTs frames {first}:{last}/{len(data)}", flush=True)
        targets = BatchedImages([stationary] * (last - first))
        sources = BatchedImages(
            [
                image(fit[index], np.linalg.inv(seed[index]) @ affine)
                for index in range(first, last)
            ]
        )
        method = SyNRegistration(
            scales=[1],
            iterations=[iterations],
            fixed_images=targets,
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
        with torch.no_grad():
            coordinates = method.get_warped_coordinates(targets, sources)
            original = torch.tensor(data.values[first:last], device="cuda")
            sampled = F.grid_sample(
                original,
                coordinates,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            size = torch.tensor(data.shape[-2:][::-1], device="cuda")
            pixels = (coordinates + 1) * ((size - 1) / 2)
            valid = ((pixels >= -0.5) & (pixels < size - 0.5)).all(dim=-1)[:, None]
            sampled = torch.where(
                valid, sampled, original.amin(dim=(-2, -1), keepdim=True)
            )
            output[first:last] = sampled.cpu().numpy()
            masks[first:last] = valid.cpu().numpy()
        del method, sources, targets, coordinates, original, sampled, valid
    output[0], masks[0] = data.values[0], True
    assert np.isfinite(output).all() and output.shape == data.shape
    return output, masks


def correlations(data, reference, mask, scale):
    result = np.full(len(data), np.nan)
    target = 10 * np.log10(np.maximum(reference / scale, 1e-12))
    for index, (values, valid) in enumerate(zip(data, mask, strict=True)):
        db = 10 * np.log10(np.maximum(values / scale, 1e-12))
        x, y = db[valid], target[valid]
        if len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0:
            result[index] = np.corrcoef(x, y)[0, 1]
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    root = cf.datasets.fetch_landemard_2026(
        datasets="rawdata", subjects="ALD001", acqs="ref20", datatypes="fusi"
    )
    source = next(root.glob("sub-ALD001/fusi/*_acq-ref20*_pwd.nii.gz"))
    data = cf.load(source).isel(time=slice(0, args.frames)).compute()
    report = {
        "source": str(source),
        "shape": list(data.shape),
        "iterations": args.iterations,
        "voxreg_batch_size_override": args.batch_size,
        "fireants_batch_size": args.fireants_batch_size,
        "gpu": torch.cuda.get_device_name(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("voxreg", "torch", "fireants", "numpy", "confusius")
        },
        "fireants_fused": fireants_interpolator.use_ffo,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *sorted(Path("src/voxreg").glob("*.py")),
                Path(__file__),
                Path(__file__).with_name("compare.py"),
            ]
        },
        "notes": "Same original float32 images/reference, dB preprocessing, shared per-frame affine seeds, single resolution, radius-2 CC, sigma 1.5/0.5 voxels, nominal learning rate 0.1 and fixed update budget. Voxreg uses float64 state/compositive GD/best halfway pose; FireANTs uses native float32/Adam. No dense CPU outputs. FireANTs has no internal recording batch option, so its public API is called per batch. Original-power bilinear output/support/minimum fill included; voxreg also computes inverse/regularity diagnostics. Imports, dataset loading, shared seed fitting, QC and disk export excluded from SyN timing. No anatomical oracle. CPU RSS is whole-process high-water mark, not a per-backend allocation measurement.",
        "backends": {},
    }

    def save_report():
        report["process_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    seed = None
    report["prealignment_seconds"] = {}
    for model in ("rigid", "affine"):
        torch.cuda.synchronize()
        start = perf_counter()
        result = register(
            data,
            transform=model,
            initial_transform=seed,
            return_transforms=True,
            device="cuda",
            batch_size=args.batch_size,
            max_iterations=args.iterations,
            convergence_window_size=args.iterations + 1,
            progress=False,
        )
        torch.cuda.synchronize()
        report["prealignment_seconds"][model] = perf_counter() - start
        seed = result.transforms
        del result
    np.save(args.output / "affine-seeds.npy", seed.values)
    save_report()
    common = np.ones(data.shape, dtype=bool)
    report["warmup_seconds"] = {}
    for backend in ("voxreg", "fireants"):
        warm_start = perf_counter()
        # Warm the backend on a small native slice batch before the complete run.
        warm_batch = (
            (args.batch_size or 32) if backend == "voxreg" else args.fireants_batch_size
        )
        warm = data.isel(time=slice(0, min(warm_batch, len(data))))
        if backend == "voxreg":
            result = register(
                warm,
                transform="syn",
                initial_transform=seed[: len(warm)],
                device="cuda",
                batch_size=args.batch_size,
                max_iterations=2,
                convergence_window_size=3,
                progress=False,
            )
            del result
        else:
            fit_fireants_recording(
                warm, seed.values[: len(warm)], args.fireants_batch_size, 2
            )
        gc.collect()
        torch.cuda.synchronize()
        report["warmup_seconds"][backend] = perf_counter() - warm_start
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        if backend == "voxreg":
            result = register(
                data,
                transform="syn",
                initial_transform=seed,
                device="cuda",
                batch_size=args.batch_size,
                max_iterations=args.iterations,
                convergence_window_size=args.iterations + 1,
                progress=False,
            )
            assert result.transforms is None and result.inverse_transforms is None
            values, valid = result.registered.values, result.valid.values
            status, counts = np.unique(result.diagnostics.status, return_counts=True)
            diagnostic = {
                "batch_size": result.diagnostics.attrs["batch_size"],
                "status_counts": dict(
                    zip(status.tolist(), counts.tolist(), strict=True)
                ),
                "stage_seconds": result.diagnostics.attrs["timings_seconds"],
            }
        else:
            values, valid = fit_fireants_recording(
                data, seed.values, args.fireants_batch_size, args.iterations
            )
            diagnostic = {}
        torch.cuda.synchronize()
        report["backends"][backend] = {
            "seconds": perf_counter() - start,
            "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
            "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
            **diagnostic,
        }
        np.testing.assert_array_equal(values[0], data.values[0])
        common &= valid
        cf.io.save_nifti(data.copy(data=values), args.output / f"{backend}.nii")
        print(backend, report["backends"][backend], flush=True)
        del values, valid
        if backend == "voxreg":
            del result
        save_report()
    scale = float(data.isel(time=0).max())
    scores = [correlations(data.values, data.values[0], common, scale)]
    for backend in ("voxreg", "fireants"):
        corrected = cf.load(args.output / f"{backend}.nii").compute()
        score = correlations(corrected.values, data.values[0], common, scale)
        scores.append(score)
        report["backends"][backend]["median_pearson_r_db_excluding_reference"] = float(
            np.nanmedian(score[1:])
        )
        del corrected
    report["raw_median_pearson_r_db_excluding_reference"] = float(
        np.nanmedian(scores[0][1:])
    )
    np.savetxt(
        args.output / "qc.csv",
        np.column_stack((data.time.values, *scores)),
        delimiter=",",
        header="time_s,uncorrected,voxreg,fireants",
        comments="",
    )
    save_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frames", type=int, default=None, help="Default: whole recording"
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override voxreg automatic batching",
    )
    parser.add_argument("--fireants-batch-size", type=int, default=256)
    args = parser.parse_args()
    if min(args.iterations, args.fireants_batch_size, args.frames or 11310) < 1 or (
        args.batch_size is not None and args.batch_size < 1
    ):
        parser.error("counts must be positive")
    run(args)
