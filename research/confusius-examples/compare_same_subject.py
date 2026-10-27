"""Reproduce ConfUSIus example 01's linear stage, without editing ConfUSIus.

Uses cached rat75 slice32 susi images only. This is a quality/capture diagnostic,
not a timing benchmark. Full-image results are retained even when they underfit.
A separately labeled symmetric central crop diagnoses fitting-domain effects.
Source hashes identify the engine's coverage policy; the original guarded report
requires its frozen c2effa2 revision.
"""

import argparse
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path

import confusius as cf
import itk
import matplotlib
import numpy as np
import torch
from plot_registration import sample

from voxreg import position_to_world, register

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    root = Path.home() / ".cache/confusius/cybis-pereira-2026-bids/1.0.0/sub-rat75"
    images, provenance = [], []
    for session in ["20220523", "20220524"]:
        path = (
            root
            / f"ses-{session}"
            / "susi"
            / f"sub-rat75_ses-{session}_acq-slice32_rec-minframe2d_pwd.nii.gz"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Cached example input missing: {path}; no implicit download."
            )
        raw = cf.load(path).compute()
        # Exactly the example representation: per-image max-referenced dB, qform world.
        data = raw.fusi.scale.db().fusi.affine.apply(raw.affines["world_to_qform"])
        images.append(data)
        provenance.append(
            {
                "path": str(path),
                "sha256": digest(path),
                "sidecar_sha256": digest(
                    path.with_name(path.name.removesuffix(".nii.gz") + ".json")
                ),
                "shape": list(raw.shape),
                "dtype": str(raw.dtype),
                "raw_range": [float(raw.min()), float(raw.max())],
                "position_to_world": position_to_world(data).tolist(),
            }
        )
    return *images, provenance


def ncc(x, y, mask):
    if mask.sum() < 2:
        return None
    a, b = x[mask], y[mask]
    a, b = a - a.mean(), b - b.mean()
    denominator = (a @ a) * (b @ b)
    return float((a @ b) ** 2 / denominator) if denominator > 0 else None


def cpu_cuda_error(a, b, fixed):
    af = position_to_world(fixed)
    points = af @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size)))
    )
    ell = np.linalg.norm(af[:3, 1:], axis=0).min()
    return float(np.linalg.norm(((a - b) @ points)[:3], axis=0).max() / ell)


def plot_group(path, group, fixed, samples, masks, records):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    fixed_values = fixed.values[0]

    def norm(v):
        return np.clip((v + 30) / 30, 0, 1)

    columns = [
        "identity",
        "rigid/voxreg-cpu",
        "rigid/confusius",
        "affine/voxreg-cpu",
        "affine/confusius",
    ]
    for row, domain in enumerate(["full", "central-crop"]):
        prefix = f"{domain}/{group}"
        for col, name in enumerate(columns):
            key = "identity" if name == "identity" else f"{prefix}/{name}"
            moved = samples[key][0]
            valid = masks[key][0]
            rgb = np.stack((norm(fixed_values), norm(moved), norm(moved)), axis=-1)
            rgb[~valid] = 0.2
            axes[row, col].imshow(rgb, origin="upper")
            score = records[key].get("cross_arm_joint_ncc2")
            axes[row, col].set_title(
                f"{domain}: {name}\nNCC²={score:.4f}"
                if score is not None
                else f"{domain}: {name}"
            )
            axes[row, col].set_axis_off()
    fig.suptitle(
        f"{group}: full fixed-grid overlays; fixed red / moving cyan; common display range −30..0 dB\nCentral crop changes fitting only; every panel is evaluated on the full fixed grid."
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_differences(path, fixed, samples, common):
    fig, axes = plt.subplots(4, 3, figsize=(12, 14), constrained_layout=True)
    arms = [
        ("full/defaults", "rigid"),
        ("full/defaults", "affine"),
        ("central-crop/fixed400", "rigid"),
        ("central-crop/fixed400", "affine"),
    ]
    for row, (arm, model) in enumerate(arms):
        vox = samples[f"{arm}/{model}/voxreg-cpu"]
        reference = samples[f"{arm}/{model}/confusius"]
        for col, (difference, label) in enumerate(
            [
                (vox - fixed.values, "voxreg − fixed"),
                (reference - fixed.values, "ConfUSIus − fixed"),
                (vox - reference, "voxreg − ConfUSIus"),
            ]
        ):
            masked = np.ma.array(difference[0], mask=~common[0])
            image = axes[row, col].imshow(
                masked, cmap="coolwarm", vmin=-10, vmax=10, origin="upper"
            )
            axes[row, col].set_title(f"{arm}/{model}\n{label}")
            axes[row, col].set_axis_off()
    fig.colorbar(
        image, ax=axes, label="dB difference (same global joint support)", shrink=0.6
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}/results.json")
    torch.set_num_threads(1)
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(1)
    fixed, moving, inputs = load_inputs()
    # Predeclared diagnostic ROI: remove 16 pixels on every active edge, moving unchanged.
    cropped = fixed.isel(j=slice(16, -16), i=slice(16, -16))
    af, am = position_to_world(fixed), position_to_world(moving)
    native = np.linalg.solve(am[:3, :3], af[:3, :3])
    offset = np.linalg.solve(am[:3, :3], af[:3, 3] - am[:3, 3])
    coplanar_bound = float(
        abs(offset[0]) + np.abs(native[0] * (np.array(fixed.shape) - 1)).sum()
    )
    assert coplanar_bound < 1e-5, (
        "Example planes are not supported by voxreg; do not project them silently."
    )
    records = {}
    samples = {}
    masks = {}
    saved = {}
    identity = np.eye(4)
    values, valid, _ = sample(identity, fixed, moving)
    records["identity"] = {
        "matrix": identity.tolist(),
        "support_fraction": float(valid.mean()),
    }
    samples["identity"], masks["identity"] = values, valid
    protocols = {"defaults": None, "fixed100": 100, "fixed400": 400}
    for domain, target in [("full", fixed), ("central-crop", cropped)]:
        for policy, budget in protocols.items():
            for model in ["rigid", "affine"]:
                for backend in ["voxreg-cpu", "voxreg-cuda", "confusius"]:
                    if backend == "voxreg-cuda" and not torch.cuda.is_available():
                        continue
                    key = f"{domain}/{policy}/{model}/{backend}"
                    if backend == "confusius":
                        options = {
                            "transform_type": model,
                            "initialization": "center_geometry",
                            "show_progress": False,
                            "plot_metric": False,
                            "plot_composite": False,
                            "sitk_threads": 1,
                        }
                        if budget is not None:
                            options.update(
                                number_of_iterations=budget,
                                convergence_window_size=budget + 1,
                            )
                        registered, matrix, diagnostics = (
                            cf.registration.register_volume(moving, target, **options)
                        )
                        meta = dataclasses.asdict(diagnostics)
                        meta["metric_values"] = meta["metric_values"].tolist()
                    else:
                        options = {
                            "transform": model,
                            "initialization": "center_geometry",
                            "intensity_scaling": "none",
                            "progress": False,
                            "device": backend.removeprefix("voxreg-"),
                        }
                        if budget is not None:
                            options.update(
                                max_iterations=budget,
                                convergence_window_size=budget + 1,
                            )
                        result = register(moving, target, **options)
                        registered, matrix = result.registered, result.transforms.values
                        meta = {
                            "n_iterations": int(result.diagnostics.n_iterations),
                            "status": str(result.diagnostics.status.values),
                            "objective": float(result.diagnostics.objective),
                            "valid_fraction": float(result.diagnostics.valid_fraction),
                        }
                    assert (
                        registered.dtype == moving.dtype
                        and registered.shape == target.shape
                    )
                    full_values, full_valid, coordinates = sample(matrix, fixed, moving)
                    target_values, target_valid, _ = sample(matrix, target, moving)
                    # Resampling agreement uses only shared interior points, not differing fill rules.
                    difference = (
                        float(
                            np.max(
                                np.abs(
                                    registered.values[target_valid]
                                    - target_values[target_valid]
                                )
                            )
                        )
                        if target_valid.any()
                        else None
                    )
                    assert difference is not None and difference < 1e-8, (
                        key,
                        difference,
                    )
                    singular = np.linalg.svd(matrix[1:3, 1:3], compute_uv=False)
                    center = af @ np.r_[(np.array(fixed.shape) - 1) / 2, 1]
                    meta.update(
                        center_displacement_zyx_mm=(
                            np.asarray(matrix) @ center - center
                        )[:3].tolist(),
                        matrix=np.asarray(matrix).tolist(),
                        fit_options=options,
                        full_grid_support_fraction=float(full_valid.mean()),
                        fit_grid_support_fraction=float(target_valid.mean()),
                        native_k_max_absolute=float(np.abs(coordinates[0]).max()),
                        active_singular_values=singular.tolist(),
                        active_determinant=float(np.linalg.det(matrix[1:3, 1:3])),
                        independent_resampling_max_abs_db=difference,
                    )
                    records[key] = meta
                    samples[key] = full_values
                    masks[key] = full_valid
                    saved[key.replace("/", "__") + "__registered"] = registered.values
                    saved[key.replace("/", "__") + "__matrix"] = np.asarray(matrix)
                    print(
                        key,
                        meta["status"],
                        meta["n_iterations"],
                        meta["full_grid_support_fraction"],
                        flush=True,
                    )
    # A shared score domain across ALL arms/models/backends prevents crop/affine support gaming.
    all_common = np.logical_and.reduce(list(masks.values()))
    assert all_common.any()
    for key, record in records.items():
        record["cross_arm_joint_ncc2"] = ncc(fixed.values, samples[key], all_common)
    precision = {}
    backend_disagreement = {}
    for domain in ["full", "central-crop"]:
        for policy in protocols:
            keys = [k for k in records if k.startswith(f"{domain}/{policy}/")]
            common = masks["identity"].copy()
            for key in keys:
                common &= masks[key]
            for key in keys:
                records[key]["arm_joint_ncc2"] = ncc(fixed.values, samples[key], common)
                records[key]["arm_joint_support_fraction"] = float(common.mean())
                records[key]["arm_joint_identity_ncc2"] = ncc(
                    fixed.values, samples["identity"], common
                )
            for model in ["rigid", "affine"]:
                cpu = f"{domain}/{policy}/{model}/voxreg-cpu"
                cuda = f"{domain}/{policy}/{model}/voxreg-cuda"
                reference = f"{domain}/{policy}/{model}/confusius"
                points = af @ np.vstack(
                    (np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size)))
                )
                displacement = (
                    (
                        np.array(records[cpu]["matrix"])
                        - np.array(records[reference]["matrix"])
                    )
                    @ points
                )[:3]
                ell = np.linalg.norm(af[:3, 1:], axis=0).min()
                errors = np.linalg.norm(displacement, axis=0) / ell
                backend_disagreement[f"{domain}/{policy}/{model}"] = {
                    "rms_pixels": float(np.sqrt(np.mean(errors**2))),
                    "max_pointwise_pixels": float(errors.max()),
                    "meaning": "algorithm disagreement, not error against biological truth",
                }
                if cuda in records:
                    maximum = cpu_cuda_error(
                        np.array(records[cpu]["matrix"]),
                        np.array(records[cuda]["matrix"]),
                        fixed,
                    )
                    precision[f"{domain}/{policy}/{model}"] = {
                        "max_pointwise_pixels": maximum,
                        "gate_pass": maximum < 0.001,
                        "same_update_count": records[cpu]["n_iterations"]
                        == records[cuda]["n_iterations"],
                    }
    manifest = {
        "example": "ConfUSIus 03_registration/01_register_volume_same_subject.py linear stage",
        "inputs": inputs,
        "preprocessing": "per-image max-referenced dB followed by world_to_qform; fitting scaling none on both backends",
        "fit_domains": {
            "full": list(fixed.shape),
            "central-crop": {
                "shape": list(cropped.shape),
                "slices": "j[16:-16], i[16:-16]; moving remains full",
            },
        },
        "coplanarity_max_native_k": coplanar_bound,
        "protocols": {
            "defaults": "each backend native defaults; CF GD auto LR/window10 vs voxreg Adam0.1/window50, both caps100",
            "fixed100": "native GD auto vs Adam0.1, convergence disabled, cap100",
            "fixed400": "native GD auto vs Adam0.1, convergence disabled, cap400",
        },
        "versions": {
            "confusius": cf.__version__,
            "torch": torch.__version__,
            "itk": itk.Version.GetITKVersion(),
        },
        "source_sha256": {
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path(inspect.getfile(cf.registration.register_volume)),
                *Path("src/voxreg").glob("*.py"),
            ]
        },
        "cross_arm_joint_support_fraction": float(all_common.mean()),
        "precision": precision,
        "voxreg_confusius_mapping_disagreement": backend_disagreement,
        "runs": records,
        "limits": [
            "No biological ground truth; higher NCC does not establish anatomically valid scale/shear.",
            "ConfUSIus fits a thin-expanded 3D image; voxreg fits the constrained physical plane.",
            "Native learning-rate/scaling/stop policies are not matched algorithms. No timing/speedup comparison.",
            "Example 02 uses earlier frozen motion-correction evidence; example 03 remains unsupported slice-to-volume.",
        ],
    }
    (output / "results.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        output / "arrays.npz",
        fixed=fixed.values,
        moving=moving.values,
        fixed_affine=af,
        cropped_fixed_affine=position_to_world(cropped),
        moving_affine=am,
        global_common=all_common,
        **saved,
    )
    plot_differences(output / "differences.png", fixed, samples, all_common)
    for policy in protocols:
        plot_group(output / f"{policy}.png", policy, fixed, samples, masks, records)
    assert all(
        item["gate_pass"] and item["same_update_count"] for item in precision.values()
    ), "CPU/CUDA screen failed; retain the results."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/confusius-examples/same-subject")
    )
    run(parser.parse_args().output)
