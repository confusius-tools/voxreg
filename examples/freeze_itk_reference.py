"""Freeze the notebook reference using an existing ConfUSIus environment.

This comparison-only script is deliberately not an engine dependency. Run it
with ConfUSIus's Python, not voxreg's environment. See README.md for the command.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter

import confusius as cf
import itk
import numpy as np


def main() -> None:
    """Export known inputs, ITK matrices and public-pipeline reference timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--repeats",
        type=int,
        default=0,
        help="Timing repetitions; zero exports quality reference only.",
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.repeats < 0:
        parser.error("jobs must be positive and repeats nonnegative")
    data = cf.load(args.input).isel(time=slice(220, 340)).compute()
    if data.sizes["time"] != 120:
        raise ValueError("The reference recording must contain frames 220:340.")
    fixed = data.isel(time=0, drop=True)
    settings = {
        "transform_type": "rigid",
        "metric": "correlation",
        "moving_intensity_scaling": "db",
        "fixed_intensity_scaling": "db",
        "learning_rate": 1.0,
        "number_of_iterations": 100,
        "itk_threads": 1,
        "show_progress": False,
    }
    resolved = inspect.signature(cf.registration.register_volume).bind_partial(
        **settings
    )
    resolved.apply_defaults()
    settings = dict(resolved.arguments)
    # Warm ITK's Python templates before measuring ready-to-run registration.
    cf.registration.register_volume(data.isel(time=1, drop=True), fixed, **settings)
    outputs, matrices, metrics, iterations = [], [], [], []
    for index, frame in enumerate(data):
        if index % 20 == 0:
            print(f"Freezing frame {index}/120", flush=True)
        registered, matrix, diagnostic = cf.registration.register_volume(
            frame, fixed, **settings
        )
        outputs.append(registered.values)
        matrices.append(matrix)
        metrics.append(diagnostic.final_metric_value)
        iterations.append(diagnostic.n_iterations)
    timings = {}
    pipeline_settings = {
        "transform": "rigid",
        "metric": "correlation",
        "moving_intensity_scaling": "db",
        "learning_rate": 1.0,
        "number_of_iterations": 100,
        "show_progress": False,
    }
    for jobs in sorted({1, args.jobs}) if args.repeats else []:
        cf.registration.register_volumewise(data, n_jobs=jobs, **pipeline_settings)
        elapsed = []
        for _ in range(args.repeats):
            start = perf_counter()
            result = cf.registration.register_volumewise(
                data, n_jobs=jobs, **pipeline_settings
            )
            elapsed.append(perf_counter() - start)
            np.testing.assert_allclose(result.values, outputs, rtol=1e-5, atol=1e-5)
        timings[str(jobs)] = elapsed
    fd = cf.registration.compute_framewise_displacement(matrices, fixed)
    with args.input.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    package = Path(cf.__file__).parent
    source_files = (
        "registration/volume.py",
        "registration/_utils.py",
        "_utils/itk.py",
        "_utils/geometry.py",
        "io/nifti.py",
    )
    metadata = {
        "source": str(args.input.resolve()),
        "sha256": digest,
        "input_dtype": str(data.dtype),
        "reference_source_sha256": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "sidecar_sha256": hashlib.sha256(
            args.input.with_name(
                args.input.name.removesuffix(".nii.gz") + ".json"
            ).read_bytes()
        ).hexdigest(),
        "window": [220, 340],
        "settings": settings,
        "confusius_version": cf.__version__,
        "itk_version": itk.Version.GetITKVersion(),
        "public_pipeline_seconds": timings,
        "public_pipeline_settings": pipeline_settings,
        "timing_notes": "Warm public pipeline; eager inputs; one ITK work unit per frame; default public joblib backend; timings run sequentially. Empty timings means no performance measurement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        values=data.values,
        time=data.time.values,
        k=data.k.values,
        j=data.j.values,
        i=data.i.values,
        voxel_to_world=data.fusi.affine.voxel_to_world,
        registered=np.array(outputs),
        transforms=np.array(matrices),
        objective=np.array(metrics),
        n_iterations=np.array(iterations),
        **fd,
        metadata=json.dumps(metadata),
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
