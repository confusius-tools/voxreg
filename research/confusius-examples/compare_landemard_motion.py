"""ALD001/ref20: whole-recording rigid→affine→SyN, NIfTI outputs and compact QC.

uv run python research/confusius-examples/compare_landemard_motion.py
ConfUSIus fetches ~0.5 GB on first use. Corrected power recordings retain native
geometry, dtype and timing. One register call per method; GPU batching is internal.
Only linear seed matrices are requested; dense SyN outputs are disabled.
Opens a four-recording ConfUSIus/Napari grid after saving; --no-view is headless.
QC is alignment evidence, not anatomical truth or proof physiology is retained.
"""

import argparse
import json
import shutil
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import confusius as cf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from voxreg import position_to_world, register


def show_recordings(data, output):
    """Four calibrated ConfUSIus layers, one synchronized Napari time slider."""
    import napari

    viewer = None
    limits = (0, float(data.isel(time=0).max()))
    for name in ("uncorrected", "rigid", "affine", "syn"):
        recording = data if name == "uncorrected" else cf.load(output / f"{name}.nii")
        viewer, _ = cf.plotting.plot_napari(
            recording,
            viewer=viewer,
            name=name,
            colormap="gray",
            contrast_limits=limits,
            show_colorbar=False,
        )
    viewer.grid.shape = (2, 2)
    viewer.grid.stride = 1
    viewer.grid.enabled = True
    napari.run()
    return viewer


def time_summary(values, edges):
    """Temporal mean and min/max envelope; all-invalid bins remain missing."""
    means, lows, highs = [], [], []
    for first, last in pairwise(edges):
        block = values[first:last]
        finite = np.isfinite(block)
        count = finite.sum(axis=0)
        means.append(
            np.divide(
                np.where(finite, block, 0).sum(axis=0),
                count,
                out=np.full(block.shape[1:], np.nan),
                where=count > 0,
            )
        )
        lows.append(
            np.where(count > 0, np.where(finite, block, np.inf).min(axis=0), np.nan)
        )
        highs.append(
            np.where(count > 0, np.where(finite, block, -np.inf).max(axis=0), np.nan)
        )
    return np.stack(means), np.stack(lows), np.stack(highs)


def compare(data, output, *, device="cpu", iterations=100, batch_size=None):
    if (
        data.sizes["k"] != 1
        or data.sizes["time"] < 2
        or (batch_size is not None and batch_size < 1)
    ):
        raise ValueError("Need a k=1 recording with >=2 frames and batch_size >=1.")
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < 3 * data.nbytes + 10_000_000:
        raise OSError("Not enough disk space for three corrected recordings.")
    names = ("uncorrected", "rigid", "affine", "syn")
    models = names[1:]
    n, _, height, width = data.shape
    middle = width // 2
    time = np.asarray(data.time, dtype=float)
    reference = data.isel(time=0, drop=True)
    scale = float(reference.max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The reference must have a finite positive power maximum.")
    reference_db = 10 * np.log10(np.maximum(reference.values[0] / scale, 1e-12))
    settings = {
        "device": device,
        "iterations_per_stage": iterations,
        "batch_size": batch_size,
        "stages": {},
    }
    common = np.ones((n, height, width), dtype=bool)
    seed = None
    for model in models:
        print(f"Registering all {n} frames: {model}", flush=True)
        start = perf_counter()
        result = register(
            data,
            transform=model,
            return_transforms=model != "syn",
            initial_transform=seed,
            device=device,
            batch_size=batch_size,
            max_iterations=iterations,
        )
        labels, counts = np.unique(result.diagnostics.status.values, return_counts=True)
        settings["stages"][model] = {
            "seconds": perf_counter() - start,
            "batch_size": result.diagnostics.attrs["batch_size"],
            "status_counts": dict(zip(labels.tolist(), counts.tolist(), strict=True)),
        }
        common &= result.valid.values[:, 0]
        if model != "syn":
            seed = result.transforms
        print(f"Saving {model}.nii...", flush=True)
        cf.io.save_nifti(result.registered, output / f"{model}.nii")
        del result  # Release dense SyN outputs as soon as export completes.

    recordings = {"uncorrected": data}
    recordings.update({model: cf.load(output / f"{model}.nii") for model in models})
    rasters = np.full((n, height, 4), np.nan, dtype=np.float32)
    scores = np.full((n, 4), np.nan)
    # Read one recording at a time for QC, after releasing all dense fields.
    for column, recording in enumerate(recordings.values()):
        values = recording.values[:, 0]
        db = 10 * np.log10(np.maximum(values / scale, 1e-12))
        rasters[:, :, column] = np.where(common[:, :, middle], db[:, :, middle], np.nan)
        for row, mask in enumerate(common):
            x, y = db[row][mask], reference_db[mask]
            if len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0:
                scores[row, column] = np.corrcoef(x, y)[0, 1]
    np.savetxt(
        output / "qc.csv",
        np.column_stack((time, scores)),
        delimiter=",",
        header="time_s," + ",".join(names),
        comments="",
    )

    # Wide stacked rasters for the whole run; envelopes preserve brief QC dips.
    edges = np.linspace(0, n, min(n, 1200) + 1, dtype=int)
    raster_mean, _, _ = time_summary(rasters, edges)
    mean, low, high = time_summary(scores, edges)
    time_edges = np.r_[
        time[0] - (time[1] - time[0]) / 2,
        (time[:-1] + time[1:]) / 2,
        time[-1] + (time[-1] - time[-2]) / 2,
    ]
    unit = 60 if time[-1] - time[0] > 180 else 1
    xedges = (time_edges[edges] - time[0]) / unit
    centers = (xedges[:-1] + xedges[1:]) / 2
    dy = np.linalg.norm(position_to_world(data)[:3, 1])
    cmap = plt.get_cmap("gray").with_extremes(bad="0.25")
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(16, 10),
        width_ratios=[3, 1],
        layout="constrained",
        facecolor="black",
    )
    for column, name in enumerate(names):
        ax = axes[column, 0]
        ax.pcolormesh(
            xedges,
            np.arange(height + 1) * dy,
            raster_mean[:, :, column].T,
            cmap=cmap,
            vmin=-30,
            vmax=0,
            shading="flat",
        )
        ax.invert_yaxis()
        ax.set_ylabel(f"{name}\nAlong j (mm)")
        ax = axes[column, 1]
        ax.fill_between(
            centers, low[:, column], high[:, column], color="white", alpha=0.35
        )
        ax.plot(centers, mean[:, column], color="white", lw=0.6)
        ax.set_ylim(max(-1, float(np.nanmin(scores)) - 0.01), 1.001)
        ax.set_ylabel("Pearson r")
        print(
            f"{name}: median r (excluding reference) = {np.nanmedian(scores[1:, column]):.5f}"
        )
    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.set_xlabel(
            f"Time from reference ({'min' if unit == 60 else 's'})", color="white"
        )
        ax.yaxis.label.set_color("white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
    fig.suptitle(
        f"ALD001/ref20 — {n} frames, middle i={middle}, shared support/−30…0 dB\n"
        "Raster: temporal mean (≤1200 bins); QC: mean + min/max envelope; not anatomical truth",
        color="white",
    )
    fig.savefig(output / "qc.png", dpi=140)
    plt.close(fig)
    settings.update(
        middle_i=middle,
        time_seconds=time.tolist(),
        display_power_reference=scale,
        position_to_world=position_to_world(data).tolist(),
        workflow="rigid → affine → SyN; original inputs, same first-frame reference",
        metric="Pearson r in dB on per-frame intersection of all valid masks",
        raster_bins=len(edges) - 1,
    )
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--frames", type=int, default=None, help="Default: all remaining frames"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override voxreg's automatic batch size",
    )
    parser.add_argument(
        "--no-view", action="store_true", help="Skip the final Napari grid"
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/landemard-ref20-full")
    )
    args = parser.parse_args()
    if (
        args.start < 0
        or (args.frames is not None and args.frames < 2)
        or args.iterations < 1
        or (args.batch_size is not None and args.batch_size < 1)
    ):
        parser.error("start >=0; frames >=2 if supplied; iterations and batch-size >=1")
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    root = cf.datasets.fetch_landemard_2026(
        datasets="rawdata", subjects="ALD001", acqs="ref20", datatypes="fusi"
    )
    paths = list(root.glob("sub-ALD001/fusi/*_acq-ref20*_pwd.nii.gz"))
    if len(paths) != 1:
        raise ValueError(f"Expected one ref20 recording, found {paths}")
    stop = None if args.frames is None else args.start + args.frames
    data = cf.load(paths[0]).isel(time=slice(args.start, stop)).compute()
    if args.frames is not None and data.sizes["time"] != args.frames:
        raise ValueError("Requested window exceeds recording length")
    compare(
        data,
        args.output,
        device=args.device,
        iterations=args.iterations,
        batch_size=args.batch_size,
    )
    metadata = json.loads((args.output / "settings.json").read_text())
    metadata.update(source=str(paths[0]), start=args.start)
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved corrected NIfTIs and QC to {args.output.resolve()}")
    if not args.no_view:
        show_recordings(data, args.output)
