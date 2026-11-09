"""Reproduce the 120-frame motion-correction example using ConfUSIus VoxelData."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import xarray as xr
from confusius import create_voxeldata
from scipy.ndimage import map_coordinates

from voxreg import (
    RegistrationResult,
    position_to_world,
    register,
)
from voxreg.sampling import make_plane


def load_recording(path: Path, window: slice | None = None) -> xr.DataArray:
    """Load a NIfTI time window using its physical affine and declared timing.

    Parameters
    ----------
    path : pathlib.Path
        Four-dimensional NIfTI with mm spatial units.
    window : slice, optional
        Time window; all frames are loaded when not provided.

    Returns
    -------
    data : xarray.DataArray
        Independent VoxelData in time/k/j/i order.

    Raises
    ------
    ValueError
        If shape, units, timing or the requested window are invalid.
    """
    image = nib.load(path)
    if len(image.shape) != 4 or image.header.get_xyzt_units() != ("mm", "sec"):
        raise ValueError("Expected a 4D NIfTI with mm/sec units.")
    sidecar = path.with_name(
        path.name.removesuffix(".nii.gz").removesuffix(".nii") + ".json"
    )
    metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if "VolumeTiming" in metadata:
        time = np.asarray(metadata["VolumeTiming"], dtype=float)
    else:
        interval = float(image.header.get_zooms()[3])
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError(
                "A positive NIfTI time step or sidecar VolumeTiming is required."
            )
        time = np.arange(image.shape[3]) * interval + float(image.header["toffset"])
    if (
        time.shape != (image.shape[3],)
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError(
            "VolumeTiming must contain one finite increasing time per frame."
        )
    window = window if window is not None else slice(None)
    values = np.asarray(image.dataobj[..., window]).transpose(3, 2, 1, 0)
    affine = image.affine[np.ix_([2, 1, 0, 3], [2, 1, 0, 3])]
    result = create_voxeldata(
        values,
        dims=("time", "k", "j", "i"),
        time=time[window],
        # Infer before cropping so singleton windows retain source timing.
        volume_acquisition_duration=metadata.get(
            "FrameAcquisitionDuration",
            float(np.median(np.diff(time)))
            if len(time) > 1
            else float(image.header.get_zooms()[3]),
        ),
        voxel_to_world=affine,
        attrs={"source": str(path.resolve())},
    )
    result.time.attrs["units"] = "s"
    return result


def save_nifti(path: Path, data: xr.DataArray) -> None:
    """Export VoxelData to napari-readable NIfTI with an exact timing sidecar.

    Parameters
    ----------
    path : pathlib.Path
        Output .nii.gz path.
    data : xarray.DataArray
        Time-dependent VoxelData; boolean masks are exported as uint8.
    """
    affine = position_to_world(data)[np.ix_([2, 1, 0, 3], [2, 1, 0, 3])]
    values = data.values.astype(np.uint8) if data.dtype == bool else data.values
    image = nib.Nifti1Image(values.transpose(3, 2, 1, 0), affine, dtype=values.dtype)
    image.header.set_xyzt_units("mm", "sec")
    image.header["pixdim"][4] = (
        np.median(np.diff(data.time.values))
        if data.sizes["time"] > 1
        else data.time.attrs.get("volume_acquisition_duration", 1)
    )
    image.header["toffset"] = float(data.time.values[0])
    nib.save(image, path)
    path.with_name(
        path.name.removesuffix(".nii.gz").removesuffix(".nii") + ".json"
    ).write_text(
        json.dumps(
            {
                "VolumeTiming": data.time.values.tolist(),
                "FrameAcquisitionDuration": data.time.attrs.get(
                    "volume_acquisition_duration", float(image.header["pixdim"][4])
                ),
            },
            indent=2,
        )
        + "\n"
    )


def sample_matrices(
    values: np.ndarray, matrices: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Independently sample world pull mappings with SciPy for common scoring.

    Parameters
    ----------
    values : (T, H, W) numpy.ndarray
        Intensities to sample, e.g. logarithmic fitting images.
    matrices : (T, 4, 4) numpy.ndarray
        World z/y/x pull transforms.
    affine : (4, 4) numpy.ndarray
        Array-position k/j/i to world z/y/x map.

    Returns
    -------
    sampled : numpy.ndarray
        Bilinear moving image samples.
    valid : numpy.ndarray
        Sample-center validity on all three native axes.
    """
    j, i = np.indices(values.shape[1:])
    points = np.stack((np.zeros_like(j), j, i, np.ones_like(j)))
    mapped = np.einsum(
        "tab,bhw->tahw", np.linalg.inv(affine) @ matrices @ affine, points
    )
    h, w = values.shape[1:]
    valid = (
        (np.abs(mapped[:, 0]) < 1e-5)
        & (mapped[:, 1] >= -1e-5)
        & (mapped[:, 1] <= h - 1 + 1e-5)
        & (mapped[:, 2] >= -1e-5)
        & (mapped[:, 2] <= w - 1 + 1e-5)
    )
    sampled = np.array(
        [
            map_coordinates(
                frame.astype(float),
                coordinates[1:3],
                order=1,
                mode="nearest",
                prefilter=False,
            )
            for frame, coordinates in zip(values, mapped, strict=True)
        ]
    )
    return sampled, valid


def plot_results(
    data: xr.DataArray, result: RegistrationResult, reference: dict | None, output: Path
) -> None:
    """Save motion, voxel trace, raster and worst-disagreement image plots.

    Parameters
    ----------
    data : xarray.DataArray
        Original recording.
    result : RegistrationResult
        Voxreg outputs.
    reference : dict, optional
        Frozen ITK arrays; not required to inspect voxreg alone.
    output : pathlib.Path
        Existing output directory.
    """
    raw = data.values[:, 0]
    movies = {"Before": raw, "voxreg": result.registered.values[:, 0]}
    mappings = {"voxreg": result.transforms.values}
    if reference is not None:
        movies["ITK"] = reference["registered"][:, 0]
        mappings["ITK"] = reference["transforms"]
    time = data.time.values
    plane = make_plane(data)
    j, i = np.indices(raw.shape[1:])
    homogeneous = np.stack((np.zeros_like(j), j, i, np.ones_like(j))).reshape(4, -1)
    points = (position_to_world(data) @ homogeneous)[:3]
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True, layout="constrained")
    for name, matrices in mappings.items():
        local_r = plane.basis.T @ matrices[:, :3, :3] @ plane.basis
        angle = np.arctan2(local_r[:, 1, 0], local_r[:, 0, 0])
        shift = (
            matrices[:, :3, :3] @ plane.center + matrices[:, :3, 3] - plane.center
        ) @ plane.basis
        axes[0].plot(time, np.rad2deg(angle), label=name)
        axes[1].plot(time, shift[:, 0], label=f"{name} intrinsic i")
        axes[1].plot(time, shift[:, 1], label=f"{name} intrinsic j", linestyle="--")
        displacement = np.linalg.norm(
            np.einsum("tij,jp->tip", np.diff(matrices[:, :3, :3], axis=0), points)
            + np.diff(matrices[:, :3, 3], axis=0)[:, :, None],
            axis=1,
        )
        for label, fd in (
            ("mean", displacement.mean(axis=1)),
            ("max", displacement.max(axis=1)),
            ("rms", np.sqrt(np.mean(displacement**2, axis=1))),
        ):
            axes[2].plot(time, np.r_[fd, 0], label=f"{name} {label}")
    axes[3].plot(time, result.diagnostics.objective, label="voxreg fitting objective")
    count_axis = axes[3].twinx()
    count_axis.plot(
        time,
        result.diagnostics.n_iterations,
        alpha=0.4,
        color="tab:orange",
        label="voxreg steps",
    )
    count_axis.set_ylabel("Steps")
    for ax, label in zip(
        axes,
        [
            "Rotation (deg)",
            "Center displacement (mm)",
            "FD t → t+1 (mm)",
            "Negative NCC²",
        ],
        strict=True,
    ):
        ax.set_ylabel(label)
        ax.legend(fontsize="small", ncols=2)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(output / "motion.png", dpi=160)
    plt.close(fig)
    row, column = np.unravel_index(np.std(raw, axis=0).argmax(), raw.shape[1:])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].imshow(np.mean(raw, axis=0), cmap="gray")
    axes[0].scatter(column, row, color="red", s=12)
    axes[0].set_title(f"Preselected maximum-SD voxel: j={row}, i={column}")
    for name, movie in movies.items():
        axes[1].plot(time, movie[:, row, column], label=name)
    axes[1].legend()
    axes[1].set(xlabel="Time (s)", ylabel="Original power intensity")
    fig.savefig(output / "trace.png", dpi=160)
    plt.close(fig)
    scale = raw.max()
    rasters = {
        name: 10 * np.log10(np.maximum(movie[:, :, column] / scale, 1e-12))
        for name, movie in movies.items()
    }
    low, high = np.percentile(rasters["Before"], [2, 99.8])
    fig, axes = plt.subplots(
        1,
        len(rasters),
        figsize=(5 * len(rasters), 4),
        layout="constrained",
        squeeze=False,
    )
    for ax, (name, raster) in zip(axes[0], rasters.items(), strict=True):
        ax.imshow(
            raster.T,
            origin="upper",
            aspect="auto",
            cmap="gray",
            vmin=low,
            vmax=high,
            extent=[time[0], time[-1], raw.shape[1] - 1, 0],
        )
        ax.set(title=name, xlabel="Time (s)", ylabel="Native j position")
    fig.savefig(output / "raster.png", dpi=160)
    plt.close(fig)
    if reference is not None:
        difference = result.registered.values[:, 0] - movies["ITK"]
        delta = result.transforms.values - reference["transforms"]
        mapping_difference = (
            np.einsum("tij,jp->tip", delta[:, :3, :3], points) + delta[:, :3, 3, None]
        )
        frame = np.mean(mapping_difference**2, axis=(1, 2)).argmax()
        _, itk_valid = sample_matrices(
            raw[frame : frame + 1],
            reference["transforms"][frame : frame + 1],
            position_to_world(data),
        )
        common = itk_valid[0] & result.valid.values[frame, 0]
        visible_difference = np.where(common, difference[frame], np.nan)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
        low, high = np.percentile(raw[0], [2, 99.8])
        for ax, (name, image) in zip(
            axes[:3],
            [
                ("Fixed", raw[0]),
                ("voxreg", movies["voxreg"][frame]),
                ("ITK", movies["ITK"][frame]),
            ],
            strict=True,
        ):
            ax.imshow(image, cmap="gray", vmin=low, vmax=high)
            ax.set_title(f"{name}, comparison frame {frame}")
        limit = np.nanpercentile(np.abs(visible_difference), 99)
        display = axes[3].imshow(
            visible_difference, cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        axes[3].set_title("voxreg − ITK (common valid support)")
        fig.colorbar(display, ax=axes[3], label="Power difference; 99% color limits")
        fig.savefig(output / "comparison.png", dpi=160)
        plt.close(fig)


def main() -> None:
    """Run the fixed notebook window, export outputs and optionally score ITK."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/notebook"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    start = perf_counter()
    data = load_recording(args.input, slice(220, 340))
    load_seconds = perf_counter() - start
    if data.sizes["time"] != 120:
        raise ValueError("The notebook requires frames 220:340.")
    result = register(
        data,
        moving_intensity_scaling="log1p",
        return_transforms=True,
        device=args.device,
        batch_size=args.batch_size,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    reference = None
    if args.reference is not None:
        with np.load(args.reference, allow_pickle=False) as archive:
            reference = dict(archive)
        np.testing.assert_array_equal(
            data.values, reference["values"].astype(data.dtype)
        )
        np.testing.assert_array_equal(data.time, reference["time"])
        np.testing.assert_allclose(
            position_to_world(data), reference["voxel_to_world"], atol=1e-12
        )
    with args.input.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    status, counts = np.unique(result.diagnostics.status.values, return_counts=True)
    summary = {
        "source": str(args.input.resolve()),
        "sha256": digest,
        "window": [220, 340],
        "shape": list(data.shape),
        "input_dtype": str(data.dtype),
        "output_dtype": str(result.registered.dtype),
        "optimizer": {
            "max_iterations": 100,
            "learning_rate": 0.1,
            "moving_intensity_scaling": "log1p",
            "fixed_intensity_scaling": None,
            "fitting_sampling_dtype": "float32",
            "geometry_and_state_dtype": "float64",
            "ncc_dtype": "float64",
        },
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "xarray": xr.__version__,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "batch_size": args.batch_size,
        "torch_threads": args.threads,
        "load_seconds": load_seconds,
        "stage_seconds": result.diagnostics.attrs["timings_seconds"],
        "statuses": dict(zip(status.tolist(), counts.tolist(), strict=True)),
        "timing_note": "Example diagnostics only; use examples/benchmark_register.py for performance measurements.",
    }
    comparison = {}
    if reference is not None:
        if digest != json.loads(str(reference["metadata"]))["sha256"]:
            raise ValueError("Reference source hash does not match input.")
        db = 10 * np.log10(np.maximum(data.values[:, 0] / data.values.max(), 1e-12))
        affine = position_to_world(data)
        sampled, vox_valid = sample_matrices(db, result.transforms.values, affine)
        itk_sampled, itk_valid = sample_matrices(db, reference["transforms"], affine)
        common = vox_valid & itk_valid
        before, vox_scores, itk_scores = [], [], []
        for frame, mask in enumerate(common):
            if np.count_nonzero(mask) < 2:
                raise ValueError(f"No common scoring support in frame {frame}.")
            for images, scores in (
                (db, before),
                (sampled, vox_scores),
                (itk_sampled, itk_scores),
            ):
                scores.append(np.corrcoef(db[0][mask], images[frame][mask])[0, 1] ** 2)
        comparison.update(
            before_ncc2=np.array(before),
            voxreg_ncc2=np.array(vox_scores),
            itk_ncc2=np.array(itk_scores),
            common_fraction=common.mean(axis=(1, 2)),
        )
        j, i = np.indices(db.shape[1:])
        points = affine @ np.stack((np.zeros_like(j), j, i, np.ones_like(j))).reshape(
            4, -1
        )
        shift = np.einsum(
            "tij,jp->tip", result.transforms.values - reference["transforms"], points
        )[:, :3]
        comparison["mapping_rms_mm"] = np.sqrt(
            np.mean(np.sum(shift**2, axis=1), axis=1)
        )
        comparison["power_difference_rms"] = np.array(
            [
                np.sqrt(np.mean((v[mask] - ref[mask]) ** 2))
                for v, ref, mask in zip(
                    result.registered.values[:, 0],
                    reference["registered"][:, 0],
                    common,
                    strict=True,
                )
            ]
        )
        summary["quality"] = {
            key: dict(
                zip(
                    ["min", "median", "p95", "max"],
                    np.percentile(values, [0, 50, 95, 100]).tolist(),
                    strict=True,
                )
            )
            for key, values in comparison.items()
        }
        summary["quality"]["voxreg_minus_itk_ncc2_median"] = float(
            np.median(comparison["voxreg_ncc2"] - comparison["itk_ncc2"])
        )
        summary["reference"] = json.loads(str(reference["metadata"]))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    result.diagnostics.to_dataframe().to_csv(args.output / "diagnostics.csv")
    geometry = {dim: data[dim].values for dim in data.dims}
    geometry["voxel_to_world"] = position_to_world(data)
    np.savez_compressed(args.output / "input.npz", values=data.values, **geometry)
    np.savez_compressed(
        args.output / "voxreg_result.npz",
        registered=result.registered.values,
        valid=result.valid.values,
        transforms=result.transforms.values,
        **geometry,
        **{
            name: variable.values
            for name, variable in result.diagnostics.data_vars.items()
        },
        **comparison,
    )
    for name, image in {
        "input": data,
        "voxreg_registered": result.registered,
        "voxreg_valid": result.valid,
        "voxreg_minus_fixed": result.registered.copy(
            data=result.registered.values - data.values[:1]
        ),
    }.items():
        save_nifti(args.output / f"{name}.nii.gz", image)
    if reference is not None:
        save_nifti(
            args.output / "itk_registered.nii.gz",
            data.copy(data=reference["registered"]),
        )
        save_nifti(
            args.output / "voxreg_minus_itk.nii.gz",
            data.copy(data=result.registered.values - reference["registered"]),
        )
    plot_results(data, result, reference, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
