"""Native ANTs I/O and commands for benchmark.py; no independent runner."""

import hashlib

import _benchmark_data as synthetic
import nibabel as nib
import numpy as np


def _save_image(path, values, shape, recording):
    """Export native planes/volumes in NIfTI x/y[/z]/time, not world z/y/x."""
    planar = shape[0] == 1
    if recording:
        array = (
            values[:, 0].transpose(2, 1, 0) if planar else values.transpose(3, 2, 1, 0)
        )
    else:
        array = values[0].T if planar else values.transpose(2, 1, 0)
    affine = np.diag([*synthetic.SPACING[::-1], 1.0])
    affine[:3, 3] = -((np.array(shape)[::-1] - 1) * synthetic.SPACING[::-1]) / 2
    if planar and recording:
        affine[2, 2] = 0.1
    # LPS-aligned ITK axes also avoid antsMotionCorr's 2D direction-collapse reset.
    affine[:2] *= -1
    image = nib.Nifti1Image(array, affine)
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    zooms = list(image.header.get_zooms())
    if recording:
        # antsMotionCorr -d 2 expects (x,y,time), not an artificial (x,y,1,time).
        zooms[-1] = 0.1
    image.header.set_zooms(zooms)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, path)
    restored = nib.load(path)
    if restored.shape != array.shape or not np.allclose(
        restored.affine, affine, rtol=0, atol=1e-6
    ):
        raise AssertionError("NIfTI shape/physical geometry did not round-trip")
    if recording and not np.isclose(restored.header.get_zooms()[-1], 0.1):
        raise AssertionError("NIfTI time spacing did not round-trip")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "affine_ras_xyz": affine.tolist(),
        "synthetic_world": "LPS, components z/y/x; NIfTI header is RAS x/y/z",
        "axis_order": "xy/time"
        if planar and recording
        else "xyz/time"
        if recording
        else "xy"
        if planar
        else "xyz",
    }


def _prepare(case, directory, frames):
    shape = synthetic.SHAPES[case["workload"]]
    recording = case["workload"].endswith("+t")
    nonlinear = case["workflow"] == "syn"
    moving, fixed, _ = synthetic._dataset(
        shape, frames, "rigid" if case["workflow"] == "motion" else "syn", recording
    )
    if case["workflow"] == "registration":
        # Known pull: affine ∘ rigid ∘ nonlinear; invert in reverse order to paint anatomy.
        linear = synthetic._matrix("affine", 1, shape) @ synthetic._matrix(
            "rigid", 1, shape
        )
        inverse = np.linalg.inv(linear)
        points = synthetic._grid(shape)
        queries = inverse[:3, :3] @ points + inverse[:3, 3, None]
        queries = synthetic._truth(queries, 1, "syn", shape, inverse=True)
        moving.values[...] = synthetic._phantom(queries, shape)
        synthetic._write_json(
            directory / "truth.json",
            {
                "pull": "affine(rigid(syn(x))) in LPS world z/y/x mm",
                "linear_matrix": linear.tolist(),
                "nonlinear_amplitude": 1,
                "shape": list(shape),
                "spacing_mm": synthetic.SPACING.tolist(),
            },
        )
    metadata = {
        "moving": _save_image(
            directory / "moving.nii",
            moving.values[None] if nonlinear and not recording else moving.values,
            shape,
            recording or nonlinear,
        ),
        "moving_array_sha256": hashlib.sha256(memoryview(moving.values)).hexdigest(),
    }
    if not recording or nonlinear:
        metadata["fixed"] = _save_image(
            directory / "fixed.nii", fixed.values, shape, False
        )
    metadata["sha256"] = {}
    for path in directory.glob("*.nii"):
        with path.open("rb") as stream:
            metadata["sha256"][path.name] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return metadata


def _commands(
    case, directory, run, tools, frames, threads, seed, motion_sampling, *, export=False
):
    prefix = run / "ants_"
    moving = directory / "moving.nii"
    if case["workflow"] == "syn":
        count = frames if case["workload"].endswith("+t") else 1
        dimension = "2" if case["workload"] == "2d+t" else "3"
        return [
            (
                "syn-motion",
                [
                    tools["antsMotionCorr"],
                    "-d",
                    dimension,
                    "-m",
                    f"CC[{directory / 'fixed.nii'},{moving},1,2]",
                    "-t",
                    "SyN[0.1,3,0]",
                    "-i",
                    "100",
                    "-f",
                    "1",
                    "-s",
                    "0vox",
                    "-u",
                    "1",
                    "-n",
                    str(count),
                    "--use-histogram-matching",
                    "0",
                    "--random-seed",
                    str(seed),
                    "-p",
                    "Linear",
                    "-v",
                    "1",
                    "-w",
                    str(int(export)),
                    "-o",
                    f"[{prefix},{run / 'corrected.nii.gz'},{run / 'corrected-mean.nii.gz'}]",
                ],
            )
        ]
    if case["workflow"] == "registration":
        return [
            (
                "rigid-affine-syn",
                [
                    tools["antsRegistrationSyN.sh"],
                    "-d",
                    "3",
                    "-f",
                    str(directory / "fixed.nii"),
                    "-m",
                    str(moving),
                    "-o",
                    str(prefix),
                    "-t",
                    "s",
                    "-n",
                    str(threads),
                    "-e",
                    str(seed),
                ],
            )
        ]
    dimension = "2" if case["workload"] == "2d+t" else "3"
    mean = run / "reference-mean.nii.gz"
    sampling = ",Random,0.05" if motion_sampling == "random5" else ""
    return [
        (
            "temporal-mean",
            [
                tools["antsMotionCorr"],
                "-d",
                dimension,
                "-a",
                str(moving),
                "-o",
                str(mean),
            ],
        ),
        (
            "rigid-motion",
            [
                tools["antsMotionCorr"],
                "-d",
                dimension,
                "-m",
                f"GC[{mean},{moving},1,1{sampling}]",
                "-t",
                "Rigid[0.01]",
                "-i",
                "100x50x20",
                "-f",
                "4x2x1",
                "-s",
                "2x1x0vox",
                "-u",
                "1",
                "-e",
                "1",
                "-n",
                str(frames),
                "--use-histogram-matching",
                "0",
                "--random-seed",
                str(seed),
                "-p",
                "Linear",
                "-v",
                "1",
                "-o",
                f"[{prefix},{run / 'corrected.nii.gz'},{run / 'corrected-mean.nii.gz'}]",
            ],
        ),
    ]


def _output_qc(case, directory, run, frames, *, export=False):
    nonlinear = case["workflow"] == "syn"
    if nonlinear and not case["workload"].endswith("+t"):
        frames = 1
    registration = case["workflow"] == "registration"
    corrected = run / "ants_Warped.nii.gz" if registration else run / "corrected.nii.gz"
    reference = (
        directory / "fixed.nii"
        if registration or nonlinear
        else run / "reference-mean.nii.gz"
    )
    required = [corrected, reference]
    if registration:
        required += [
            run / f"ants_{suffix}"
            for suffix in (
                "0GenericAffine.mat",
                "1Warp.nii.gz",
                "1InverseWarp.nii.gz",
                "InverseWarped.nii.gz",
            )
        ]
    else:
        required += [run / "corrected-mean.nii.gz", run / "ants_MOCOparams.csv"]
    if nonlinear:
        import re

        text = (run / "syn-motion.log").read_text()
        completed = [
            int(i) for i in re.findall(r"done resampling timepoint\s*:\s*(\d+)", text)
        ]
        if completed != list(range(frames)):
            raise AssertionError("Incomplete native SyN frame log")
        if export:
            required += [run / "ants_Warp.nii.gz", run / "ants_InverseWarp.nii.gz"]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"ANTs did not produce required outputs: {missing}")
    if not registration:
        parameters = np.loadtxt(
            run / "ants_MOCOparams.csv", delimiter=",", skiprows=1, ndmin=2
        )
        if len(parameters) != frames or not np.isfinite(parameters).all():
            raise AssertionError("ANTs motion parameters are incomplete/nonfinite")
    image, target = nib.load(corrected), nib.load(reference)
    dimension = 2 if case["workload"] == "2d+t" else 3
    expected = target.shape if registration else (*target.shape, frames)
    if image.shape != expected:
        raise AssertionError(f"ANTs output shape {image.shape}, expected {expected}")
    if not np.allclose(
        image.affine[:3, :dimension], target.affine[:3, :dimension], rtol=0, atol=1e-5
    ) or not np.allclose(image.affine[:3, 3], target.affine[:3, 3], rtol=0, atol=1e-5):
        raise AssertionError("ANTs output is not on its reference physical lattice")
    if case["workload"].endswith("+t") and not np.isclose(
        image.header.get_zooms()[-1], 0.1, rtol=0, atol=1e-6
    ):
        raise AssertionError("ANTs output time spacing changed")
    # Decode gzip once; correctness checks are outside timed CLI calls.
    values = np.asarray(image.dataobj)
    if not np.isfinite(values).all():
        raise AssertionError("Nonfinite corrected image")
    return {
        "shape": list(image.shape),
        "dtype": str(image.get_data_dtype()),
        "checked_frames": 1 if registration else frames,
    }
