"""Short, same-host CPU comparisons between two committed voxreg revisions."""

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar

from _benchmark_syn_reference import REFERENCE

if TYPE_CHECKING:
    import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
"""Working repository for resolving and exporting commits."""

PROTOCOL = "voxreg-cpu-history-v1"
"""Identifies comparable workload definitions in saved reports."""

Result = TypeVar("Result")


def measure(
    call: Callable[[], Result],
    repeats: int,
    validate: Callable[[Result], None] | None = None,
) -> list[float]:
    """Time complete calls after the caller has validated a warm-up result.

    Parameters
    ----------
    call : callable
        Operation on the resident inputs.
    repeats : int
        Number of timed calls.
    validate : callable, optional
        Check the result after stopping the timer.

    Returns
    -------
    list of float
        Wall-clock seconds per call, excluding disposal of returned data.
    """
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        samples.append(perf_counter() - start)
        if validate is not None:
            validate(result)
        del result
    return samples


def registration_case(
    shape: tuple[int, int, int], frames: int, model: str, repeats: int, batch_size: int
) -> dict[str, Any]:
    """Check a fitted transform against known motion and time registration.

    Parameters
    ----------
    shape : tuple of int
        Spatial voxel counts in k, j, i order.
    frames : int
        One for an image pair, otherwise the number of recording frames.
    model : str
        Registration model and corresponding synthetic motion.
    repeats : int
        Number of timed calls returning only registered images and diagnostics.
    batch_size : int
        Maximum number of frames fitted together.

    Returns
    -------
    dict
        Fit diagnostics, accuracy verdict and timing samples.
    """
    import numpy as np
    from _benchmark_data import make_dataset
    from _benchmark_resident import check_registration, run_registration

    recording = frames > 1
    workload = ("2d" if shape[0] == 1 else "3d") + ("+t" if recording else "")
    config = {
        "workload": workload,
        "backend": "cpu",
        "model": model,
        "iterations": 200,
        "batch_size": batch_size,
    }
    inputs = make_dataset(shape, frames, model, recording)
    output, errors, details = run_registration(*inputs, config, accuracy=True)
    check_registration(output, inputs[0], config, details)
    expected = [0] + [200] * (frames - 1) if recording else [200]

    def check_timed_fit(result: tuple) -> None:
        """Verify that timing covered the same work as the accuracy check.

        Parameters
        ----------
        result : tuple
            Registered pixels, unused truth score and fit diagnostics.

        Raises
        ------
        AssertionError
            If the output is invalid or the timed call completed different work.
        """
        image, _, diagnostics = result
        check_registration(image, inputs[0], config, diagnostics)
        if not np.array_equal(image, output):
            raise AssertionError(
                "Timed registration pixels differ from the accuracy call"
            )
        if diagnostics != details:
            raise AssertionError("Timed fit diagnostics differ from the accuracy call")

    return {
        "shape": shape,
        "frames": frames,
        "rms_voxels": errors,
        **details,
        "passed": details["updates"] == expected
        and len(errors) == frames
        and all(0 <= error <= REFERENCE["limit_voxels"] for error in errors),
        "seconds": measure(
            lambda: run_registration(*inputs, config, accuracy=False),
            repeats,
            check_timed_fit,
        ),
    }


def resampling_case(
    moving: "xr.DataArray", fixed: "xr.DataArray", dtype: str, repeats: int
) -> dict[str, Any]:
    """Check a fractional translation against its analytic result and time it.

    Parameters
    ----------
    moving, fixed : xarray.DataArray
        Float32 benchmark volume and matching reference grid.
    dtype : str
        Input dtype to benchmark, either float32 or int64.
    repeats : int
        Number of timed calls.

    Returns
    -------
    dict
        Sampling accuracy and timing samples.
    """
    import numpy as np
    from _benchmark_data import SPACING

    from voxreg import resample

    if dtype == "int64":
        moving = moving.copy(data=np.rint(moving.values * 1000).astype(np.int64))
    fraction = 0.3712345
    matrix = np.eye(4)
    matrix[2, 3] = fraction * SPACING[2]

    def run() -> tuple["xr.DataArray", "xr.DataArray"]:
        """Apply the known translation.

        Returns
        -------
        image : xarray.DataArray
            Sampled image.
        valid : xarray.DataArray
            Mask of supported sample positions.
        """
        return resample(
            moving, fixed, matrix, interpolation="linear", device="cpu", batch_size=1
        )

    output, valid = run()
    raw = moving.values.astype(np.float64)
    expected = np.full_like(raw, raw.min())
    expected[..., :-1] = (1 - fraction) * raw[..., :-1] + fraction * raw[..., 1:]
    if dtype == "int64":
        expected = np.rint(expected)
    error = float(np.max(np.abs(output.values - expected)))
    passed = (
        valid.values[..., :-1].all()
        and not valid.values[..., -1].any()
        and output.dtype == moving.dtype
        and error <= (0 if dtype == "int64" else 1e-6)
    )

    def check_timed_sample(result: tuple["xr.DataArray", "xr.DataArray"]) -> None:
        """Check the timed image and support against the verified warm-up."""
        image, support = result
        if image.dtype != moving.dtype or not np.array_equal(
            image.values, output.values
        ):
            raise AssertionError(
                "Timed resampling pixels differ from the accuracy call"
            )
        if support.dtype != np.bool_ or not np.array_equal(
            support.values, valid.values
        ):
            raise AssertionError(
                "Timed resampling support differs from the accuracy call"
            )

    return {
        "shape": fixed.shape,
        "frames": 1,
        "max_intensity_error": error,
        "passed": bool(passed),
        "seconds": measure(run, repeats, check_timed_sample),
    }


def worker(source: Path, repeats: int, threads: int, full_syn: bool) -> dict[str, dict]:
    """Run workloads in a fresh interpreter using the selected library revision.

    Parameters
    ----------
    source : pathlib.Path
        Archived source directory.
    repeats : int
        Timed calls per case.
    threads : int
        PyTorch CPU thread count.
    full_syn : bool
        Whether to run the standard-size SyN accuracy cases.

    Returns
    -------
    dict
        Measurements indexed by workload name.

    Raises
    ------
    RuntimeError
        If the imported package is outside the selected revision.
    """
    sys.path.insert(0, str(source))
    import torch
    from _benchmark_data import SHAPES, make_dataset

    import voxreg

    if not Path(voxreg.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError(
            "Benchmark imported voxreg from outside the selected revision"
        )
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if full_syn:
        return {
            f"3d-syn-{frames}f": registration_case(SHAPES["3d"], frames, "syn", 0, 1)
            for frames in (1, 3)
        }
    cases = {
        "rigid-4-slices": registration_case(SHAPES["2d+t"], 4, "rigid", repeats, 4),
        "syn-small-volume": registration_case((24, 40, 24), 1, "syn", repeats, 1),
    }
    moving, fixed, _ = make_dataset(SHAPES["3d"], 1, "rigid", False)
    for dtype in ("float32", "int64"):
        cases[f"linear-resample-{dtype}"] = resampling_case(
            moving, fixed, dtype, repeats
        )
    return cases


def snapshot(revision: str, directory: Path) -> str:
    """Extract the selected commit's library into a temporary directory.

    Parameters
    ----------
    revision : str
        Git revision to resolve.
    directory : pathlib.Path
        Destination for the archived source.

    Returns
    -------
    str
        Full commit ID of the extracted source.
    """
    sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", revision + "^{commit}"],
        cwd=ROOT,
        text=True,
    ).strip()
    archive = subprocess.check_output(["git", "archive", sha, "src/voxreg"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
        contents.extractall(directory, filter="data")
    return sha


def main() -> int:
    """Run the selected protocol and save a JSON report.

    Returns
    -------
    int
        Exit status, zero when all current accuracy checks pass.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="HEAD^")
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--full-size-syn",
        action="store_true",
        help="Only check 200-update accuracy at the selected revision on a full-size pair and three-frame recording",
    )
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.source:
        cases = worker(args.source, args.repeats, args.threads, args.full_size_syn)
        args.output.write_text(json.dumps(cases, allow_nan=False))
        return 0
    report = {
        "protocol": PROTOCOL,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "threads": args.threads,
        "packages": {
            p: version(p) for p in ("torch", "numpy", "scipy", "voxeldata", "xarray")
        },
        "results": {},
    }
    env = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    with tempfile.TemporaryDirectory() as tmp:
        snapshots = {}
        roles = (
            {"current": args.revision}
            if args.full_size_syn
            else {"baseline": args.baseline, "current": args.revision}
        )
        for role, revision in roles.items():
            directory = Path(tmp) / role
            sha = snapshot(revision, directory)
            snapshots[role] = directory / "src"
            report["results"][role] = {"commit": sha, "cases": {}}
        # Reverse the order in the second block to reduce bias from machine drift.
        blocks = (
            [("current", 0)]
            if args.full_size_syn
            else [("baseline", 3), ("current", 3), ("current", 2), ("baseline", 2)]
        )
        for role, repeats in blocks:
            path = Path(tmp) / "block.json"
            command = [
                sys.executable,
                __file__,
                "--source",
                str(snapshots[role]),
                "--repeats",
                str(repeats),
                "--threads",
                str(args.threads),
                "--output",
                str(path),
            ]
            if args.full_size_syn:
                command.append("--full-size-syn")
            print(
                f"Measuring {role} {report['results'][role]['commit'][:8]} ({repeats} timed repeats)",
                flush=True,
            )
            subprocess.run(command, env=env, check=True)
            cases = report["results"][role]["cases"]
            for name, row in json.loads(path.read_text()).items():
                if name in cases:
                    cases[name]["seconds"].extend(row["seconds"])
                    cases[name]["passed"] &= row["passed"]
                else:
                    cases[name] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    current = report["results"]["current"]
    if args.full_size_syn:
        for name, row in current["cases"].items():
            print(name, row["updates"], row["rms_voxels"])
    else:
        baseline = report["results"]["baseline"]
        lines = [
            f"CPU comparison `{baseline['commit'][:8]}` → `{current['commit'][:8]}` ({args.threads} threads).",
            "",
            "| Workload | Before (ms) | After (ms) | Change | Accuracy (before / after) |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for name, row in current["cases"].items():
            old = baseline["cases"][name]
            before, after = median(old["seconds"]), median(row["seconds"])
            verdict = ["pass" if case["passed"] else "FAIL" for case in (old, row)]
            lines.append(
                f"| {name} | {before * 1000:.2f} | {after * 1000:.2f} | {after / before - 1:+.1%} | {' / '.join(verdict)} |"
            )
        summary = "\n".join(lines) + "\n"
        print(summary)
        if "GITHUB_STEP_SUMMARY" in os.environ:
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
                stream.write(summary)
    return int(not all(case["passed"] for case in current["cases"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
