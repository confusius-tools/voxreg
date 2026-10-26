"""Compare rigid→affine against installed ConfUSIus ground truth, without tuning.

Uses example03's cached 3D slab and native recording slice. Each backend's rigid
matrix seeds its own affine fit. No engine changes, downloads, or speed claims.
"""

import argparse
import dataclasses
import hashlib
import inspect
import json
import warnings
from pathlib import Path

import confusius as cf
import matplotlib
import numpy as np
import torch
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_atlas import inputs
from compare_same_subject import digest, ncc, sample

from voxreg import position_to_world, register


def fit_chain(moving, fixed, seed, backend):
    records = {}
    for stage in ["rigid", "affine"]:
        starting = np.asarray(seed).copy()
        if backend == "confusius":
            options = {
                "transform_type": stage,
                "metric": "correlation",
                "convergence_window_size": 50,
                "number_of_iterations": 500,
                "learning_rate": 1.0,
                "initialization": starting.copy(),
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
                "transform": stage,
                "max_iterations": 500,
                "convergence_window_size": 50,
                "intensity_scaling": "none",
                "initialization": starting.copy(),
                "device": backend.removeprefix("voxreg-"),
                "progress": False,
            }
            with warnings.catch_warnings(record=True) as caught:
                result = register(moving, fixed, **options)
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
        assert registered.shape == fixed.shape and registered.dtype == moving.dtype
        matrix = np.asarray(matrix).copy()
        assert matrix.shape == (4, 4) and np.isfinite(matrix).all()
        records[stage] = meta | {
            "matrix": matrix.tolist(),
            "initialization": starting.tolist(),
            "returned_dtype": str(registered.dtype),
            "options": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in options.items()
            },
        }
        seed = matrix
        print(backend, stage, meta["status"], meta["n_iterations"], flush=True)
    assert records["affine"]["initialization"] == records["rigid"]["matrix"]
    return records


def point_error(candidate_pull, reference_pull, native_slice):
    """Compare observed slice placements, not arbitrary off-plane completions."""
    points = position_to_world(native_slice) @ np.vstack(
        (np.indices(native_slice.shape).reshape(3, -1), np.ones((1, native_slice.size)))
    )
    delta = ((np.linalg.inv(candidate_pull) - np.linalg.inv(reference_pull)) @ points)[
        :3
    ]
    distance = np.linalg.norm(delta, axis=0)
    ell = float(np.linalg.norm(position_to_world(native_slice)[:3, 1:3], axis=0).min())
    rms, maximum = float(np.sqrt(np.mean(distance**2))), float(distance.max())
    return {
        "rms_mm": rms,
        "max_mm": maximum,
        "rms_native_pixels": rms / ell,
        "max_native_pixels": maximum / ell,
        "pixel_size_mm": float(ell),
    }


def label_agreement(candidate, reference, valid):
    candidate, reference = candidate[valid], reference[valid]
    union = (candidate != 0) | (reference != 0)
    per_region = {}
    for region in np.union1d(candidate, reference):
        if region == 0:
            continue
        a, b = candidate == region, reference == region
        per_region[str(int(region))] = float(2 * np.sum(a & b) / (a.sum() + b.sum()))
    return {
        "valid_pixels": int(valid.sum()),
        "agreement_fraction": float(np.mean(candidate == reference))
        if candidate.size
        else None,
        "foreground_union_pixels": int(union.sum()),
        "foreground_label_agreement": float(
            np.mean(candidate[union] == reference[union])
        )
        if union.any()
        else None,
        "macro_region_dice": float(np.mean(list(per_region.values())))
        if per_region
        else None,
        "per_region_dice": per_region,
    }


def boundaries(labels):
    edge = np.zeros_like(labels, dtype=bool)
    horizontal, vertical = (
        labels[:, 1:] != labels[:, :-1],
        labels[1:, :] != labels[:-1, :],
    )
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    return edge & (labels != 0)


def cached_annotation():
    # Match ConfUSIus's atlas builder: resolution-diagonal world grid, origin zero.
    root = Path.home() / ".brainglobe/brainglobe-atlasapi"
    manifest = root / "atlases/allen_mouse_100um/3_1/manifest.json"
    store = (
        root
        / "annotation-sets/allen_mouse-annotation/3_1/annotations_compressed.ome.zarr/s3"
    )
    if not manifest.is_file() or not (store / "c").is_dir():
        raise FileNotFoundError(
            "Cached Allen 100um v3.1 annotation chunks required; no downloads."
        )
    meta = json.loads(manifest.read_text())
    assert meta["orientation"] == "asr" and meta["resolution"] == [100.0, 100.0, 100.0]
    array = zarr.open_array(store, mode="r")[:]
    assert list(array.shape) == meta["shape"] and array.dtype == np.uint32
    affine = np.diag([*np.asarray(meta["resolution"]) * 1e-3, 1.0])
    annotation = cf.create_voxeldata(
        array.view(np.int32), dims=("k", "j", "i"), voxel_to_world=affine
    )
    return annotation, {
        "manifest": str(manifest),
        "manifest_sha256": digest(manifest),
        "store": str(store),
        "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "position_to_world": affine.tolist(),
        "loader": "Cached compressed label array; geometry matches current ConfUSIus _build_dataset_from_brainglobe. No structures, meshes or network loading.",
    }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    fixed, moving, placement, paths, shape = inputs()
    original = [image.values.copy() for image in (moving, fixed)]
    annotation, annotation_provenance = cached_annotation()
    atlas_affine = np.asarray(fixed.attrs["affines"]["world_to_sform"])
    backends = ["confusius", "voxreg-cpu"] + (
        ["voxreg-cuda"] if torch.cuda.is_available() else []
    )
    records = {}
    for backend in backends:
        torch.set_num_threads(8 if backend == "voxreg-cpu" else 1)
        records[backend] = fit_chain(moving, fixed, np.linalg.inv(placement), backend)
        (output / "fit-records.json").write_text(
            json.dumps(records, indent=2, allow_nan=False) + "\n"
        )
    samples, masks, labels = {}, {}, {}
    for backend, chain in records.items():
        for stage, record in chain.items():
            key = backend + "__" + stage
            matrix = np.asarray(record["matrix"])
            samples[key], masks[key], _ = sample(np.linalg.inv(matrix), moving, fixed)
            subject_to_atlas = atlas_affine @ np.linalg.inv(matrix)
            labels[key] = cf.registration.resample_like(
                annotation,
                moving,
                subject_to_atlas,
                interpolation="nearest",
                fill_value=0,
                sitk_threads=8,
            ).values
            record["subject_to_atlas"] = subject_to_atlas.tolist()
    common = np.logical_and.reduce(list(masks.values()))
    native_values = moving.values.astype(np.float64)
    precision = {}
    for stage in ["rigid", "affine"]:
        reference = np.asarray(records["confusius"][stage]["matrix"])
        for backend in backends:
            record = records[backend][stage]
            record["reference_error"] = point_error(
                np.asarray(record["matrix"]), reference, moving
            )
            record["reference_truth_gate_pass"] = (
                record["reference_error"]["rms_native_pixels"] < 0.25
            )
            key = backend + "__" + stage
            record["common_native_ncc2"] = ncc(native_values, samples[key], common)
            record["native_slice_support_fraction"] = float(masks[key].mean())
            # The same trusted CF nearest-neighbor resampler is used for all poses.
            record["labels_against_reference"] = label_agreement(
                labels[key],
                labels["confusius__" + stage],
                np.ones(moving.shape, dtype=bool),
            )
        if "voxreg-cuda" in records:
            error = point_error(
                np.asarray(records["voxreg-cuda"][stage]["matrix"]),
                np.asarray(records["voxreg-cpu"][stage]["matrix"]),
                moving,
            )
            precision[stage] = error | {
                "precision_gate_pass": error["max_native_pixels"] < 0.001
            }
    for image, initial in zip((moving, fixed), original, strict=True):
        np.testing.assert_array_equal(image.values, initial)
    report = {
        "reference": "Installed ConfUSIus is ground truth by operator instruction; compare each corresponding rigid/affine stage.",
        "records": records,
        "cpu_cuda": precision,
        "truth_rms_limit_native_pixels": 0.25,
        "precision_max_limit_native_pixels": 0.001,
        "common_native_slice_fraction": float(common.mean()),
        "scoring": "Point error evaluates inverse exported pulls on every native recording pixel center, including out-of-slab points. Image NCC uses common template support; atlas label agreement includes background and separately reports foreground union. No gate is relaxed.",
        "inputs": [
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
        "shapes": {"moving": list(moving.shape), "fixed": list(fixed.shape)},
        "position_to_world": {
            "moving": position_to_world(moving).tolist(),
            "fixed": position_to_world(fixed).tolist(),
        },
        "annotation": annotation_provenance,
        "versions": {
            "confusius": cf.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "source_sha256": {
            str(path): digest(path)
            for path in [
                Path(__file__).resolve(),
                Path(inspect.getfile(inputs)),
                *Path("src/voxreg").glob("*.py"),
                *Path(inspect.getfile(cf.registration.register_volume)).parent.glob(
                    "*.py"
                ),
            ]
        },
        "limitations": [
            "Same original example 3D slab, not an independently changed full-template preprocessing.",
            "CF and voxreg objectives/parameterizations differ; higher NCC does not override CF ground-truth displacement.",
            "Cached label-only atlas adapter avoids the installed BrainGlobe/Pandas structures-table UInt16 conversion failure. No meshes or atlas export.",
            "Quality experiment only; no timings or product/default changes.",
        ],
    }
    np.savez_compressed(
        output / "arrays.npz",
        moving=moving.values,
        fixed=fixed.values,
        common=common,
        **{key + "__image": value for key, value in samples.items()},
        **{key + "__labels": value for key, value in labels.items()},
    )
    fig, axes = plt.subplots(
        2, len(backends), figsize=(5 * len(backends), 10), constrained_layout=True
    )
    for row, stage in enumerate(["rigid", "affine"]):
        for col, backend in enumerate(backends):
            key = backend + "__" + stage
            ax = axes[row, col]
            red = np.clip((samples[key][0] + 30) / 30, 0, 1)
            cyan = np.clip((moving.values[0] + 30) / 30, 0, 1)
            rgb = np.stack((red, cyan, cyan), axis=-1)
            rgb[~masks[key][0]] = 0.2
            ax.imshow(rgb)
            ax.set_axis_off()
            record = records[backend][stage]
            ax.set_title(
                f"{backend} {stage}\nNCC²={record['common_native_ncc2']:.5f}; CF RMS={record['reference_error']['rms_native_pixels']:.3f}px"
            )
    fig.suptitle("Template red / recording cyan; identical display −30…0 dB")
    fig.savefig(output / "template-overlays.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(
        2, len(backends), figsize=(5 * len(backends), 10), constrained_layout=True
    )
    gray = np.clip((moving.values[0] + 30) / 30, 0, 1)
    for row, stage in enumerate(["rigid", "affine"]):
        reference_edge = boundaries(labels["confusius__" + stage][0])
        for col, backend in enumerate(backends):
            ax = axes[row, col]
            rgb = np.stack((gray, gray, gray), axis=-1)
            edge = boundaries(labels[backend + "__" + stage][0])
            rgb[reference_edge] = [0, 1, 1]
            rgb[edge] = [1, 0, 1]
            rgb[edge & reference_edge] = 1
            ax.imshow(rgb)
            ax.set_axis_off()
            ax.set_title(backend + " " + stage)
    fig.suptitle(
        "Allen label boundaries: CF truth cyan / candidate magenta / shared white"
    )
    fig.savefig(output / "atlas-boundaries.png", dpi=150)
    plt.close(fig)
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
