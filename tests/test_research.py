"""Keep the tracked research commands usable without historical artifacts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

SCRIPTS = Path(__file__).resolve().parents[1] / "research/performance"


@pytest.mark.parametrize(
    "script",
    [
        "affine_probe",
        "profile_baseline",
        "validate_probe",
        "check_elementwise",
        "cpu_ablation",
    ],
)
def test_research_help_needs_no_inputs(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / f"{script}.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage:" in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("sampler", ["production", "elementwise64"])
def test_batch_check_generates_its_own_comparison(recording_factory, tmp_path, sampler):
    data, _, _ = recording_factory()
    path = tmp_path / "input.npz"
    np.savez(
        path,
        values=data.values,
        voxel_to_world=data.xindexes["z"].transform.voxel_to_world,
        **{dim: data[dim].values for dim in data.dims},
    )
    output = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "check_elementwise.py"),
            "--input",
            str(path),
            "--output",
            str(output),
            "--devices",
            "cpu",
            "--sampler",
            sampler,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        env=dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        ),
    )
    filename = (
        "production-validation.json"
        if sampler == "production"
        else "elementwise-validation.json"
    )
    report = json.loads((output / filename).read_text())
    quality = report["quality"]
    baseline = quality["baseline-cpu"]["median_ncc2"]
    assert 0.99 < baseline <= 1
    for batch in (1, 2, 32):
        assert report[f"cpu-b1-vs-b{batch}"]["rms_pixels_max"] < 0.001
        assert_allclose(
            quality[f"{sampler}-cpu-b{batch}"]["median_ncc2"], baseline, atol=1e-7
        )
