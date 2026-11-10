"""Public lazy benchmark CLI and independent streaming scorecard checks."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark_lazy.py"
SPEC = importlib.util.spec_from_file_location(
    "lazy_benchmark", ROOT / "examples" / "_benchmark_lazy.py"
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


@pytest.mark.parametrize("operation", ["register", "resample"])
def test_smoke_cli_writes_qualified_phases_without_large_files(tmp_path, operation):
    output = tmp_path / operation
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--operation",
            operation,
            "--devices",
            "cpu",
            "--runs",
            "1",
            "--threads",
            "1",
            "--chunk-mib",
            "0.01",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads((output / "results.json").read_text())
    assert result["protocol"] == "voxreg-lazy-benchmark-v1"
    assert result["diagnostic"] is True
    assert result["status"] == "passed"
    assert result["inputs"]["frames"] == 8
    assert result["inputs"]["image_bytes"] == 8 * 24 * 32 * 4
    assert result["inputs"]["compression"] == "none"
    assert result["larger_than_host_ram"] is False
    assert result["source_sha256"]["src/voxreg/registration.py"]
    assert (output / "report.html").is_file()
    case = result["cases"][0]
    assert case["status"] == "passed"
    assert case["config"]["chunk_frames"] == 4
    assert len(case["timing"]["wall_seconds"]) == 1
    assert case["timing"]["wall_seconds"][0] > 0
    assert case["memory"]["peak_process_tree_rss_bytes"] > 0
    assert case["accuracy"]["checked_frames"] == 8
    assert case["accuracy"]["accuracy_pass"] is True
    assert case["accuracy"]["worst_relative_intensity_rmse"] < 1e-5
    assert case["accuracy"]["support_mismatches"] == 0
    assert (
        case["accuracy"]["input_sha256_verified"] == result["inputs"]["payload_sha256"]
    )
    assert case["accuracy"]["task_duration_sample"]["count"] > 0
    if operation == "register":
        assert case["accuracy"]["worst_truth_rms_voxels"] <= 0.30
        assert case["accuracy"]["observed_task_sizes"] == [4]
        assert case["accuracy"]["observed_fitting_batch_sizes"] == [4]
    assert not list(output.glob("*/*.zarr")) or all(
        path.parent.name == "inputs" for path in output.glob("*/*.zarr")
    )
    assert (output / "inputs" / "recording.zarr").is_dir()
    # A fresh invocation must refuse to overwrite this completed evidence.
    again = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert again.returncode == 2
    assert "fresh" in again.stderr
    assert json.loads((output / "results.json").read_text()) == result


def test_large_defaults_follow_mib_targets_without_allocating_large_inputs(
    tmp_path, monkeypatch
):
    prepared, measured = [], []

    def stage(directory, shape, frames):
        prepared.append((shape, frames))
        return {"image_bytes": frames * int(np.prod(shape)) * 4}

    def measure(config_path, phase, script):
        measured.append(json.loads(config_path.read_text()))
        return {
            "status": "unavailable",
            "reason": "No actual worker in this configuration test",
        }

    monkeypatch.setattr(BENCHMARK, "stage_inputs", stage)
    monkeypatch.setattr(BENCHMARK, "measure", measure)
    monkeypatch.setattr(
        BENCHMARK.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 2**30)
    )
    output = tmp_path / "large-config"
    assert (
        BENCHMARK.main(
            ["--output", str(output), "--size-gib", "10", "--devices", "cpu"]
        )
        == 0
    )
    assert prepared == [((32, 128, 128), 5120)]
    assert [case["chunk_frames"] for case in measured] == [16, 64]
    assert [case["chunk_bytes"] for case in measured] == [32 * 2**20, 128 * 2**20]
    assert all(case["runs"] == 5 for case in measured)
    result = json.loads((output / "results.json").read_text())
    assert result["larger_than_total_worker_limits"] is True
    assert result["status"] == "unavailable"
    assert not (output / "inputs").exists()


@pytest.mark.parametrize(
    "damage", ["image", "nonfinite", "support", "geometry", "input"]
)
def test_streaming_verifier_rejects_corrupted_results(tmp_path, damage):
    BENCHMARK.stage_inputs(tmp_path / "inputs", (1, 12, 14), 3)
    config = {
        "inputs": str(tmp_path / "inputs"),
        "operation": "resample",
        "chunk_frames": 2,
        "effective_batch_size": 1,
        "device": "cpu",
    }
    output = tmp_path / "result.zarr"
    BENCHMARK.execute(config, output)
    assert BENCHMARK.verify(config, output, tmp_path)["accuracy_pass"] is True
    result = zarr.open_group(output, mode="a")
    if damage == "nonfinite":
        image = result["images/image"]
        assert isinstance(image, zarr.Array)
        image[0] = np.nan
        with pytest.raises(AssertionError, match="finite"):
            BENCHMARK.verify(config, output, tmp_path)
    elif damage == "geometry":
        result["images/image"].attrs["voxel_to_world_units"] = "seconds"
        with pytest.raises(AssertionError):
            BENCHMARK.verify(config, output, tmp_path)
    elif damage == "input":
        source = zarr.open_group(tmp_path / "inputs" / "recording.zarr", mode="a")
        image = source["image"]
        assert isinstance(image, zarr.Array)
        image[1] = np.zeros((1, 12, 14), dtype=np.float32)
        with pytest.raises(AssertionError):
            BENCHMARK.verify(config, output, tmp_path)
    else:
        if damage == "image":
            image = result["images/image"]
            assert isinstance(image, zarr.Array)
            image[:] = 0
        else:
            support = result["validity/valid"]
            assert isinstance(support, zarr.Array)
            support[:] = True
        assert BENCHMARK.verify(config, output, tmp_path)["accuracy_pass"] is False
        assert (tmp_path / "failure.npz").is_file()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--size-gib", "-1"],
        ["--size-gib", "nan"],
        ["--size-gib", "1e308"],
        ["--chunk-mib", "1e308"],
        ["--chunk-mib", "0"],
        ["--runs", "0"],
        ["--workers", "0"],
        ["--shape", "1", "1", "8"],
        ["--worker-memory-gib", "0.000001"],
        ["--batch-size", "0"],
    ],
)
def test_invalid_cli_options_do_not_create_files(tmp_path, arguments):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        BENCHMARK.main(["--output", str(output), *arguments])
    assert error.value.code == 2
    assert not output.exists()
