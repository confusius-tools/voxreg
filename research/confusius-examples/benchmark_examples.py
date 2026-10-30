"""Public-API example timing and independent common-support quality comparison.

Each case runs sequentially in a fresh interpreter. Only resident-input public API
calls and CUDA synchronization are timed; loading, decoding, scoring and plotting
are outside the boundary. No product or ConfUSIus code is modified.
"""

import argparse
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "examples"))

import confusius as cf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from benchmark_registration import power_state
from compare_same_subject import cpu_cuda_error, digest, load_inputs, ncc, sample
from notebook_motion_correction import sample_matrices
from scipy.spatial.transform import Rotation

from voxreg import position_to_world, register

SERIES_SOURCE = (
    Path.home()
    / ".cache/confusius/cybis-pereira-2026-bids/1.0.0/sub-rat75/ses-20220523/fusi"
    / "sub-rat75_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz"
)


def options(group, policy, model, backend):
    """Example02 deliberately retains its documented rate 1.0, not the old .1 control."""
    if backend == "confusius":
        if group == "pair":
            result = {
                "transform_type": model,
                "initialization": "center_geometry",
                "sitk_threads": 8,
                "show_progress": False,
                "plot_metric": False,
                "plot_composite": False,
            }
        else:
            result = {
                "transform": model,
                "n_jobs": 8,
                "moving_intensity_scaling": "db",
                "learning_rate": 1.0,
                "show_progress": False,
            }
        if policy != "example":
            budget = 400 if group == "pair" else 100
            result.update(
                number_of_iterations=budget, convergence_window_size=budget + 1
            )
    else:
        result = {
            "transform": model,
            "intensity_scaling": "none" if group == "pair" else "db",
            "progress": False,
            "device": "cuda" if "cuda" in backend else "cpu",
        }
        if group == "pair":
            result["initialization"] = "center_geometry"
        else:
            result["batch_size"] = 120 if backend == "voxreg-cuda120" else 32
        if policy != "example":
            budget = 400 if group == "pair" else 100
            result.update(max_iterations=budget, convergence_window_size=budget + 1)
    return result


def matrices_from_motion(frame):
    """Undo CF's public Euler extraction: raw matrix-index order is world z/y/x."""
    matrices = np.broadcast_to(np.eye(4), (len(frame), 4, 4)).copy()
    matrices[:, :3, :3] = Rotation.from_euler(
        "xyz", frame[["rot_z", "rot_y", "rot_x"]].to_numpy()
    ).as_matrix()
    matrices[:, :3, 3] = frame[["trans_z", "trans_y", "trans_x"]].to_numpy()
    return matrices


def distribution(values):
    a = np.asarray(values, dtype=float)
    valid = a[np.isfinite(a)]
    return {
        "n_valid": int(valid.size),
        "n_invalid": int(a.size - valid.size),
        "minimum": float(valid.min()) if valid.size else None,
        "median": float(np.median(valid)) if valid.size else None,
        "maximum": float(valid.max()) if valid.size else None,
    }


def series_input(output):
    with np.load(output / "series-input.npz", allow_pickle=False) as archive:
        return cf.create_voxeldata(
            archive["values"],
            dims=("time", "k", "j", "i"),
            voxel_to_world=archive["voxel_to_world"],
            **{d: archive[d] for d in ("time", "k", "j", "i")},
        )


def decode(result, group, backend):
    if backend == "confusius":
        if group == "pair":
            image, matrix, diag = result
            return {
                "registered": image.values,
                "transforms": matrix[None],
                "n_iterations": np.array([diag.n_iterations]),
                "status": np.array([diag.status]),
            }
        frame = result.attrs["motion_params"]
        return {
            "registered": result.values,
            "transforms": matrices_from_motion(frame),
            "n_iterations": frame.n_iterations.to_numpy(),
            "status": frame.status.to_numpy(dtype=str),
        }
    return {
        "registered": result.registered.values,
        "transforms": result.transforms.values
        if group == "series"
        else result.transforms.values[None],
        "n_iterations": np.atleast_1d(result.diagnostics.n_iterations.values),
        "status": np.atleast_1d(result.diagnostics.status.values),
    }


def worker(output, key, repeats):
    group, policy, model, backend = key.split("/")
    path = output / (key.replace("/", "__") + ".json")
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8 if backend == "voxreg-cpu" else 1)
    fixed, moving = (
        load_inputs()[:2] if group == "pair" else (None, series_input(output))
    )
    kwargs = options(group, policy, model, backend)
    if backend == "confusius":
        call = (
            (lambda: cf.registration.register_volume(moving, fixed, **kwargs))
            if group == "pair"
            else (lambda: cf.registration.register_volumewise(moving, **kwargs))
        )
    else:
        call = lambda: register(moving, fixed, return_transforms=True, **kwargs)
    cuda = "cuda" in backend
    meta = {
        "key": key,
        "options": kwargs,
        "input_dtype": str(moving.dtype),
        "shape": list(moving.shape),
        "power_before": power_state(),
        "warm_seconds": [],
        "repeat_hashes": [],
    }
    try:
        result = None
        for index in range(repeats + 1):
            result = None
            if cuda and index:
                torch.cuda.synchronize()
            start = perf_counter()
            result = call()
            if cuda:
                torch.cuda.synchronize()
            elapsed = perf_counter() - start
            arrays = decode(result, group, backend)
            assert (
                np.isfinite(arrays["transforms"]).all()
                and np.isfinite(arrays["registered"]).all()
            )
            assert arrays["registered"].dtype == moving.dtype
            import hashlib

            meta["repeat_hashes"].append(
                {
                    name: hashlib.sha256(array.tobytes()).hexdigest()
                    for name, array in arrays.items()
                }
            )
            if index:
                meta["warm_seconds"].append(elapsed)
            else:
                meta["cold_seconds"] = elapsed
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        meta.update(
            run_status="completed",
            warm_median_seconds=statistics.median(meta["warm_seconds"]),
            output_dtype=str(arrays["registered"].dtype),
            n_iterations=arrays["n_iterations"].tolist(),
            status=arrays["status"].tolist(),
            repeated_outputs_bitwise_equal=all(
                h == meta["repeat_hashes"][0] for h in meta["repeat_hashes"]
            ),
            power_after=power_state(),
        )
    except Exception:  # noqa: BLE001 -- Retain the failure and continue other independent cases.
        meta.update(
            run_status="failed", error=traceback.format_exc(), power_after=power_state()
        )
    path.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    print(key, meta["run_status"], meta.get("warm_median_seconds"), flush=True)
    return meta["run_status"] == "completed"


def overlay(path, rows, fixed, images, masks, records):
    """Fixed red/moving cyan; an identical dB display range in every panel."""
    fig, axes = plt.subplots(
        len(rows),
        len(rows[0]),
        figsize=(3.6 * len(rows[0]), 3.3 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    normalize = lambda a: np.clip((a + 30) / 30, 0, 1)
    for axes_row, keys in zip(axes, rows, strict=True):
        for ax, key in zip(axes_row, keys, strict=True):
            target = fixed[key] if isinstance(fixed, dict) else fixed
            rgb = np.stack(
                (normalize(target), normalize(images[key]), normalize(images[key])),
                axis=-1,
            )
            rgb[~masks[key]] = 0.2
            ax.imshow(rgb)
            score = records[key]
            ax.set_title(
                f"{key}\nNCC²={score:.5f}"
                if score is not None
                else f"{key}\nno usable common score",
                fontsize=9,
            )
            ax.set_axis_off()
    fig.suptitle(
        "Fixed red / moving cyan; gray = alignment; masked edges dark gray; display −30…0 dB"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def score(output, records):
    fixed, moving, _ = load_inputs()
    series = series_input(output)
    raw = series.values[:, 0]
    db = 10 * np.log10(np.maximum(raw.astype(float) / raw.max(), 1e-12))
    af = position_to_world(series)
    comparison = {"pair": {}, "series": {}}
    pair_images, pair_masks = {}, {}
    identity, valid, _ = sample(np.eye(4), fixed, moving)
    pair_images["identity"], pair_masks["identity"] = identity[0], valid[0]
    series_images, series_masks, series_display = {}, {}, {}
    matrices = {}
    for key, record in records.items():
        if record["run_status"] != "completed":
            continue
        with np.load(
            output / (key.replace("/", "__") + ".npz"), allow_pickle=False
        ) as archive:
            matrix = archive["transforms"]
            matrices[key] = matrix.copy()
            if key.startswith("pair/"):
                values, mask, _ = sample(matrix[0], fixed, moving)
                pair_images[key], pair_masks[key] = values[0], mask[0]
                reference = values[mask]
                actual = archive["registered"][mask]
            else:
                values, mask = sample_matrices(db, matrix, af)
                series_images[key], series_masks[key] = values, mask
                series_display[key] = 10 * np.log10(
                    np.maximum(archive["registered"][:, 0] / raw.max(), 1e-12)
                )
                sampled_raw, _ = sample_matrices(raw, matrix, af)
                reference = sampled_raw[mask]
                actual = archive["registered"][:, 0][mask]
            error = float(np.max(np.abs(actual - reference))) if actual.size else None
            record["independent_resampling_max_abs"] = error
            # CF motion matrices are reconstructed outside timing from its public table.
            # Their agreement with actual output also validates axis/angle conventions.
            # A world/index round-trip accumulates coordinate error, not just
            # one interpolation ULP. Keep the dB check; scale raw-power checks.
            amplitude = (
                float(np.abs(moving.values).max())
                if key.startswith("pair/")
                else float(np.abs(raw).max())
            )
            record["independent_resampling_max_relative_to_input_peak"] = (
                error / amplitude if error is not None else None
            )
            tolerance = 1e-8 if key.startswith("pair/") else 1e-12 * amplitude
            assert error is not None and error < tolerance, (key, error, tolerance)
    common = np.logical_and.reduce(list(pair_masks.values()))
    pair_scores = {
        key: ncc(fixed.values[0], values, common) for key, values in pair_images.items()
    }
    comparison["pair"].update(common_fraction=float(common.mean()), scores=pair_scores)
    pair_rows = []
    for policy in ["example", "fixed400"]:
        for model in ["rigid", "affine"]:
            pair_rows.append(
                ["identity"]
                + [
                    f"pair/{policy}/{model}/{backend}"
                    for backend in ["confusius", "voxreg-cpu", "voxreg-cuda"]
                    if f"pair/{policy}/{model}/{backend}" in pair_images
                ]
            )
    if len({len(row) for row in pair_rows}) == 1:
        overlay(
            output / "pair-overlays.png",
            pair_rows,
            fixed.values[0],
            pair_images,
            pair_masks,
            pair_scores,
        )
    joint = np.logical_and.reduce(list(series_masks.values()))
    before = [ncc(db[0], frame, mask) for frame, mask in zip(db, joint, strict=True)]
    initial = [ncc(db[0], frame, np.ones_like(frame, dtype=bool)) for frame in db]
    ordered = np.argsort(np.asarray(initial[1:])) + 1
    frames = [int(ordered[0]), int(ordered[len(ordered) // 2])]
    comparison["series"].update(
        common_fraction=distribution(joint[1:].mean(axis=(1, 2))),
        before=distribution(before[1:]),
        frames_selected_by_initial_alignment=frames,
        selection="worst and median initial full-grid NCC², excluding reference; no registered result used for selection",
        scores={},
        per_frame_scores={},
    )
    for key, values in series_images.items():
        scores = [
            ncc(db[0], frame, mask) for frame, mask in zip(values, joint, strict=True)
        ]
        comparison["series"]["scores"][key] = distribution(scores[1:])
        comparison["series"]["per_frame_scores"][key] = scores
        records[key]["full_grid_support_fraction"] = distribution(
            series_masks[key][1:].mean(axis=(1, 2))
        )
    for policy in ["example", "fixed100"]:
        images, masks, targets, scores, rows = {}, {}, {}, {}, []
        keys = [key for key in series_images if f"/{policy}/" in key]
        for frame in frames:
            row = []
            for key in ["identity"] + keys:
                label = f"frame{220 + frame}/{key.removeprefix('series/')}"
                images[label] = (
                    db[frame] if key == "identity" else series_display[key][frame]
                )
                masks[label] = (
                    np.ones_like(joint[frame])
                    if key == "identity"
                    else series_masks[key][frame]
                )
                targets[label] = db[0]
                scores[label] = (
                    before[frame]
                    if key == "identity"
                    else comparison["series"]["per_frame_scores"][key][frame]
                )
                row.append(label)
            rows.append(row)
        overlay(
            output / f"series-{policy}-overlays.png",
            rows,
            targets,
            images,
            masks,
            scores,
        )
    # Use the example's maximum-raw-temporal-SD voxel to choose the raster column.
    _, column = np.unravel_index(raw.std(axis=0).argmax(), raw.shape[1:])
    keys = [key for key in series_images if "/example/" in key]
    fig, axes = plt.subplots(
        1,
        len(keys) + 1,
        figsize=(3.5 * (len(keys) + 1), 5),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, key in zip(axes[0], ["identity"] + keys, strict=True):
        image = db if key == "identity" else series_display[key]
        ax.imshow(image[:, :, column].T, aspect="auto", vmin=-30, vmax=0, cmap="gray")
        ax.set_title(key.removeprefix("series/example/"))
        ax.set_xlabel("frame offset (0–119)")
    axes[0, 0].set_ylabel("j (native image row)")
    fig.suptitle(
        f"Motion raster: column i={column} selected from original maximum temporal-SD voxel"
    )
    fig.savefig(output / "series-raster.png", dpi=150)
    plt.close(fig)
    comparison["series"]["raster_column"] = int(column)
    comparison["cpu_cuda_max_pointwise_pixels"] = {}
    for key, matrix in matrices.items():
        if key.endswith("/voxreg-cpu"):
            prefix = key.removesuffix("voxreg-cpu")
            for gpu in ["voxreg-cuda", "voxreg-cuda32", "voxreg-cuda120"]:
                if prefix + gpu in matrices:
                    target = (
                        fixed
                        if key.startswith("pair/")
                        else series.isel(time=0, drop=True)
                    )
                    differences = [
                        cpu_cuda_error(a, b, target)
                        for a, b in zip(matrix, matrices[prefix + gpu], strict=True)
                    ]
                    comparison["cpu_cuda_max_pointwise_pixels"][prefix + gpu] = max(
                        differences
                    )
    precision = comparison["cpu_cuda_max_pointwise_pixels"]
    comparison["cpu_cuda_precision_gate"] = {
        "maximum_pointwise_limit_pixels": 0.001,
        "failed_cases": [key for key, value in precision.items() if value > 0.001],
        "all_pass": all(value <= 0.001 for value in precision.values()),
    }
    return comparison


def run(output, repeats):
    output.mkdir(parents=True, exist_ok=False)
    (output / "timed-source.py").write_bytes(Path(__file__).read_bytes())
    if not SERIES_SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing cached source: {SERIES_SOURCE}; no implicit download"
        )
    data = cf.load(SERIES_SOURCE).isel(time=slice(220, 340)).compute()
    np.savez_compressed(
        output / "series-input.npz",
        values=data.values,
        voxel_to_world=position_to_world(data),
        **{d: data[d].values for d in ("time", "k", "j", "i")},
    )
    _, _, pair_inputs = load_inputs()
    inputs = {
        "pair": pair_inputs,
        "series": {
            "path": str(SERIES_SOURCE),
            "sha256": digest(SERIES_SOURCE),
            "sidecar_sha256": digest(
                SERIES_SOURCE.with_name(
                    SERIES_SOURCE.name.removesuffix(".nii.gz") + ".json"
                )
            ),
            "window": [220, 340],
            "dtype": str(data.dtype),
            "shape": list(data.shape),
            "position_to_world": position_to_world(data).tolist(),
            "archive_sha256": digest(output / "series-input.npz"),
        },
    }
    (output / "inputs.json").write_text(
        json.dumps(inputs, indent=2, allow_nan=False) + "\n"
    )
    cases = [
        f"pair/{policy}/{model}/{backend}"
        for policy in ["example", "fixed400"]
        for model in ["rigid", "affine"]
        for backend in ["confusius", "voxreg-cpu", "voxreg-cuda"]
    ]
    cases += [
        f"series/{policy}/rigid/{backend}"
        for policy in ["example", "fixed100"]
        for backend in ["confusius", "voxreg-cpu", "voxreg-cuda32", "voxreg-cuda120"]
    ]
    records = {}
    environment = os.environ | {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    }
    for key in cases:
        with (output / (key.replace("/", "__") + ".log")).open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    str(output),
                    "--worker",
                    key,
                    "--repeats",
                    str(repeats),
                ],
                check=False,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        path = output / (key.replace("/", "__") + ".json")
        records[key] = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "run_status": "failed",
                "returncode": result.returncode,
                "error": "Worker exited without results; inspect log.",
            }
        )
        print(
            key,
            records[key]["run_status"],
            records[key].get("warm_median_seconds"),
            flush=True,
        )
    finish(output, inputs, records)


def finish(output, inputs, records):
    """Keep analysis rerunnable without replacing measurements if a check fails."""
    import SimpleITK as sitk
    from confusius.registration import volume, volumewise

    sources = list((ROOT / "src/voxreg").glob("*.py")) + [
        Path(__file__).resolve(),
        Path(__file__).with_name("compare_same_subject.py"),
        Path(__file__).with_name("plot_registration.py"),
        ROOT / "examples/notebook_motion_correction.py",
        ROOT / "examples/benchmark_registration.py",
        ROOT / "uv.lock",
        Path(inspect.getfile(volumewise)),
        Path(inspect.getfile(volume)),
    ]
    example_root = (
        ROOT.parents[1] / "confusius-tools/confusius/docs/examples/03_registration"
    )
    sources += sorted(example_root.glob("0[123]*.py"))
    report = {
        "inputs": inputs,
        "versions": {
            "confusius": cf.__version__,
            "voxreg": "0.1.0",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "SimpleITK": sitk.Version_VersionString(),
            "python": sys.version,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "source_sha256": {str(p): digest(p) for p in sources},
        "timed_script_sha256": digest(output / "timed-source.py"),
        "runs": records,
        "quality": score(output, records),
        "unsupported": {
            "example01_bspline": "No voxreg deformable model; not run or timed.",
        },
        "separate_workflows": {
            "example03_atlas": "Not part of this timing protocol. Native singleton/volume workflows are supported and checked separately with explore_atlas.py; CF expanded-slab fitting is a different objective.",
        },
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.worker:
        sys.exit(0 if worker(args.output, args.worker, args.repeats) else 1)
    run(args.output, args.repeats)
