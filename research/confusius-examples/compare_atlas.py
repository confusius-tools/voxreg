"""Retry example03 on cached real inputs; compare poses on the native slice.

No replicated slice is used by voxreg. CF's expanded-slab objective differs, so
common slice-domain scores are diagnostics, not anatomical ground truth. This is
quality validation, not a fresh-process timing benchmark or annotation download.
"""

import argparse
import dataclasses
import inspect
import json
import warnings
from pathlib import Path
from time import perf_counter

import confusius as cf
import matplotlib
import numpy as np
import torch
from atlas_inputs import inputs

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_same_subject import cpu_cuda_error, digest, ncc, sample

from voxreg import position_to_world, register


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    fixed, moving, placement, paths, shape = inputs()
    assert np.isfinite(fixed.values).all() and np.isfinite(moving.values).all()
    seed = np.linalg.inv(placement)
    samples, masks = {}, {}
    samples["manual"], masks["manual"], _ = sample(placement, moving, fixed)
    records = {"manual": {"matrix": seed.tolist()}}
    saved = {"fixed": fixed.values, "moving": moving.values}
    for backend in ["confusius", "voxreg-cpu", "voxreg-cuda"]:
        if backend == "voxreg-cuda" and not torch.cuda.is_available():
            continue
        torch.set_num_threads(8 if backend == "voxreg-cpu" else 1)
        start = perf_counter()
        if backend == "confusius":
            options = {
                "transform_type": "affine",
                "metric": "correlation",
                "convergence_window_size": 50,
                "number_of_iterations": 500,
                "learning_rate": 1.0,
                "initialization": seed,
                "sitk_threads": 8,
                "show_progress": False,
                "plot_metric": False,
                "plot_composite": False,
            }
            registered, matrix, diagnostics = cf.registration.register_volume(
                moving, fixed, **options
            )
            meta = dataclasses.asdict(diagnostics)
            meta["metric_values"] = meta["metric_values"].tolist()
        else:
            options = {
                "transform": "affine",
                "convergence_window_size": 50,
                "max_iterations": 500,
                "intensity_scaling": "none",
                "initialization": seed,
                "device": backend.removeprefix("voxreg-"),
                "progress": False,
            }
            with warnings.catch_warnings(record=True) as caught:
                result = register(moving, fixed, return_transforms=True, **options)
            registered, matrix = result.registered, result.transforms.values
            meta = {
                "n_iterations": int(result.diagnostics.n_iterations),
                "status": str(result.diagnostics.status.values),
                "objective": float(result.diagnostics.objective),
                "metric_valid_fraction": float(result.diagnostics.valid_fraction),
                "output_valid_fraction": float(
                    result.diagnostics.output_valid_fraction
                ),
                "warnings": [str(w.message) for w in caught],
            }
            # Reverse the argument roles without changing the anatomical problem.
            reverse = register(
                fixed, moving, return_transforms=True, **(options | {"initialization": placement})
            )
            discrepancy = cpu_cuda_error(
                np.linalg.inv(matrix), reverse.transforms.values, moving
            )
            meta["argument_reversal_max_pointwise_pixels"] = discrepancy
            assert discrepancy < 0.001
            assert reverse.registered.dtype == fixed.dtype
            saved[backend + "__volume_on_slice"] = reverse.registered.values
        meta["elapsed_seconds_not_benchmark"] = perf_counter() - start
        assert registered.dtype == moving.dtype and registered.shape == fixed.shape
        meta.update(
            matrix=np.asarray(matrix).tolist(),
            options={
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in options.items()
            },
            returned_dtype=str(registered.dtype),
        )
        records[backend] = meta
        samples[backend], masks[backend], _ = sample(
            np.linalg.inv(matrix), moving, fixed
        )
        saved[backend + "__registered"] = registered.values
        print(backend, meta["status"], meta["n_iterations"], flush=True)
    common = np.logical_and.reduce(list(masks.values()))
    assert common.sum() > 1
    for key, record in records.items():
        record["native_slice_common_ncc2"] = ncc(moving.values, samples[key], common)
        record["native_slice_support_fraction"] = float(masks[key].mean())
        saved[key + "__template_on_slice"] = samples[key]
    precision = (
        cpu_cuda_error(
            np.linalg.inv(np.asarray(records["voxreg-cpu"]["matrix"])),
            np.linalg.inv(np.asarray(records["voxreg-cuda"]["matrix"])),
            moving,
        )
        if "voxreg-cuda" in records
        else None
    )
    report = {
        "records": records,
        "input_provenance": [
            {
                "path": str(path),
                "sha256": digest(path),
                "sidecar_sha256": digest(
                    path.with_name(path.name.removesuffix(".nii.gz") + ".json")
                ),
            }
            for path in paths
        ],
        "recording_shape": shape,
        "fixed_shape": list(fixed.shape),
        "moving_shape": list(moving.shape),
        "position_to_world": {
            "fixed": position_to_world(fixed).tolist(),
            "moving": position_to_world(moving).tolist(),
        },
        "common_slice_fraction": float(common.mean()),
        "cpu_cuda_max_pointwise_pixels": precision,
        "cpu_cuda_precision_limit_pixels": 0.001,
        "cpu_cuda_pass": precision < 0.001 if precision is not None else None,
        "versions": {
            "confusius": cf.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "source_sha256": {
            str(path): digest(path)
            for path in [
                Path(__file__).resolve(),
                *Path("src/voxreg").glob("*.py"),
                Path(inspect.getfile(cf.registration.register_volume)),
            ]
        },
        "limitations": [
            "Native slice-domain NCC, not biological truth or CF expanded-slab loss.",
            "Slice output uses affine-defined singleton voxel extent; fitting remains on native slice centers.",
            "No speed claim: elapsed includes unlike cold setup and reverse verification.",
            "Atlas annotation/mesh fetch and cache export not run.",
        ],
    }
    np.savez_compressed(output / "arrays.npz", **saved, common=common)
    fig, axes = plt.subplots(
        1, len(records), figsize=(4 * len(records), 5), constrained_layout=True
    )
    for ax, key in zip(np.atleast_1d(axes), records, strict=True):
        red = np.clip((samples[key][0] + 30) / 30, 0, 1)
        cyan = np.clip((moving.values[0] + 30) / 30, 0, 1)
        rgb = np.stack((red, cyan, cyan), axis=-1)
        rgb[~masks[key][0]] = 0.2
        ax.imshow(rgb)
        ax.set_axis_off()
        ax.set_title(f"{key}\nNCC²={records[key]['native_slice_common_ncc2']:.5f}")
    fig.suptitle(
        "Atlas template red / recording cyan, on the native recording slice; shared display −30…0 dB"
    )
    fig.savefig(output / "atlas-overlays.png", dpi=150)
    plt.close(fig)
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
