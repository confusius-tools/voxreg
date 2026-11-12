"""Public lazy benchmark CLI and independent streaming scorecard checks."""

import importlib.util
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
SPEC = importlib.util.spec_from_file_location(
    "lazy_benchmark", ROOT / "benchmarks" / "_benchmark_lazy.py"
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_cold_eviction_precedes_every_timed_call(tmp_path, monkeypatch):
    """Eviction is repeated and its duration is excluded from measured calls."""
    clock = [0.0]
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "device": "cpu",
                "workers": 1,
                "worker_memory_bytes": 2**30,
                "inputs": str(tmp_path / "inputs"),
                "runs": 2,
            }
        )
    )

    class Client:
        def run(self, function, *args):
            if function is BENCHMARK.count_worker_tasks:
                return {"worker": 0}
            return {"worker": {"batch_size": 1}}

    def evict(directory):
        clock[0] += 10
        return {"verified": True, "resident_pages_after": 0}

    def execute(config, output):
        output.mkdir()
        clock[0] += 2

    monkeypatch.setattr(BENCHMARK, "LocalCluster", lambda **kwargs: nullcontext(None))
    monkeypatch.setattr(BENCHMARK, "Client", lambda cluster: nullcontext(Client()))
    monkeypatch.setattr(BENCHMARK, "evict_input_cache", evict)
    monkeypatch.setattr(BENCHMARK, "execute", execute)
    monkeypatch.setattr(BENCHMARK.time, "perf_counter", lambda: clock[0])
    assert BENCHMARK.worker(config_path, "timing") == 0
    record = json.loads((tmp_path / "timing.json").read_text())
    assert record["wall_seconds"] == [2.0, 2.0]
    assert len(record["cache_evictions"]) == 2
    assert not record["measurement_active"]


@pytest.mark.parametrize("failure", ["resident", "unsupported", "retained"])
def test_cache_failure_prevents_execution_and_retains_evidence(
    tmp_path, monkeypatch, failure
):
    """No registration or timing occurs without verified cold inputs."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "device": "cpu",
                "workers": 1,
                "worker_memory_bytes": 2**30,
                "inputs": str(tmp_path / "inputs"),
                "runs": 1,
            }
        )
    )

    class Client:
        def run(self, function, *args):
            if function is BENCHMARK.count_worker_tasks:
                return {"worker": 1 if failure == "retained" else 0}
            return {"worker": {"batch_size": 1}}

    def evict(directory):
        if failure == "unsupported":
            raise OSError("Unsupported cold-input filesystem")
        return {"verified": False, "resident_pages_after": 1}

    def fail(*args):
        pytest.fail("Unverified input must not execute")

    monkeypatch.setattr(BENCHMARK, "LocalCluster", lambda **kwargs: nullcontext(None))
    monkeypatch.setattr(BENCHMARK, "Client", lambda cluster: nullcontext(Client()))
    monkeypatch.setattr(BENCHMARK, "evict_input_cache", evict)
    monkeypatch.setattr(BENCHMARK, "execute", fail)
    monkeypatch.setattr(BENCHMARK.time, "sleep", lambda seconds: None)
    with pytest.raises((OSError, RuntimeError), match="cold-input|Cold-input"):
        BENCHMARK.worker(config_path, "timing")
    record = json.loads((tmp_path / "timing.json").read_text())
    assert record["status"] == "error"
    assert not record["measurement_active"]
    assert "wall_seconds" not in record
    if failure == "resident":
        assert record["cache_evictions"][0]["resident_pages_after"] == 1


@pytest.mark.parametrize("operation", ["register", "resample"])
@pytest.mark.skipif(
    sys.platform != "linux", reason="Cold-input verification requires Linux"
)
def test_smoke_cli_writes_qualified_phases_without_large_files(
    benchmark_disk_dir, operation
):
    output = benchmark_disk_dir / operation
    script = ROOT / "benchmarks" / f"benchmark_{operation}.py"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--lazy",
            "--output",
            str(output),
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
    assert result["protocol"] == "voxreg-lazy-benchmark-v3"
    assert result["diagnostic"] is True
    assert result["status"] == "passed"
    metadata = result["inputs"][0]
    assert metadata["frames"] == 8
    assert metadata["image_bytes"] == 8 * 24 * 32 * 4
    assert metadata["compression"] == "none"
    assert metadata["larger_than_host_ram"] is False
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
    assert case["accuracy"]["input_sha256_verified"] == metadata["payload_sha256"]
    assert case["accuracy"]["task_duration_sample"]["count"] > 0
    if operation == "register":
        assert case["accuracy"]["worst_truth_rms_voxels"] <= 0.30
        assert case["accuracy"]["observed_task_sizes"] == [4]
        assert case["accuracy"]["observed_fitting_batch_sizes"] == [4]
    assert not list(output.glob("*/*.zarr")) or all(
        path.parent.name == "inputs" for path in output.glob("*/*.zarr")
    )
    assert (output / "inputs" / "n8" / "recording.zarr").is_dir()
    assert result["source_sha256"]["benchmarks/_benchmark_cache.py"]
    for phase in ("timing", "memory", "accuracy"):
        evictions = case[phase]["cache_evictions"]
        assert len(evictions) == 1
        assert evictions[0]["verified"]
        assert evictions[0]["resident_pages_after"] == 0
        assert evictions[0]["bytes"] >= metadata["image_bytes"]
    memory = case["memory"]
    assert memory["baseline_process_tree_rss_bytes"] > 0
    assert memory["peak_rss_increment_bytes"] == (
        memory["peak_process_tree_rss_bytes"]
        - memory["baseline_process_tree_rss_bytes"]
    )
    # A fresh invocation must refuse to overwrite this completed evidence.
    again = subprocess.run(
        [sys.executable, str(script), "--lazy", "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert again.returncode == 2
    assert "fresh" in again.stderr
    assert json.loads((output / "results.json").read_text()) == result


@pytest.mark.skipif(
    sys.platform != "linux", reason="Cold-input protocol requires Linux"
)
def test_size_sweep_pins_policy_without_allocating_large_inputs(tmp_path, monkeypatch):
    prepared, measured = [], []

    def stage(directory, shape, frames):
        prepared.append((shape, frames))
        return {"image_bytes": frames * int(np.prod(shape)) * 4}

    def measure(config_path, phase, script):
        measured.append(json.loads(config_path.read_text()))
        if phase == "timing":
            return {
                "status": "passed",
                "effective_batch_size": 7,
                "wall_seconds": [1.0] * 5,
            }
        if phase == "memory":
            return {
                "status": "passed",
                "baseline_process_tree_rss_bytes": 100,
                "peak_process_tree_rss_bytes": 200,
                "peak_rss_increment_bytes": 100,
            }
        return {"status": "passed", "accuracy_pass": True}

    monkeypatch.setattr(BENCHMARK, "stage_inputs", stage)
    monkeypatch.setattr(BENCHMARK, "measure", measure)
    monkeypatch.setattr(
        BENCHMARK.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 2**30)
    )
    output = tmp_path / "large-config"
    assert (
        BENCHMARK.main(
            ["--output", str(output), "--size-gib", "1", "4", "--devices", "cpu"]
        )
        == 0
    )
    assert prepared == [((32, 128, 128), 512), ((32, 128, 128), 2048)]
    assert [case["chunk_frames"] for case in measured] == [64] * 6
    assert [case["chunk_bytes"] for case in measured] == [128 * 2**20] * 6
    assert all(case["runs"] == 5 for case in measured)
    assert {case["iterations"] for case in measured} == {200}
    assert "effective_batch_size" not in measured[0]
    assert all(case["effective_batch_size"] == 7 for case in measured[1:])
    result = json.loads((output / "results.json").read_text())
    assert result["status"] == "passed"
    assert result["scaling"][0]["input_growth_ratio"] == 4
    assert result["scaling"][0]["peak_rss_growth_ratio"] == 1
    assert result["scaling"][0]["rss_increment_growth_bytes"] == 0
    assert not (output / "inputs").exists()


@pytest.mark.skipif(
    sys.platform != "linux", reason="Cold-input protocol requires Linux"
)
def test_tiny_size_sweep_reports_real_cold_passes_and_memory_scaling(
    benchmark_disk_dir,
):
    output = benchmark_disk_dir / "sweep"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/benchmark_resample.py"),
            "--lazy",
            "--output",
            str(output),
            "--size-gib",
            "0.00001",
            "0.00002",
            "--shape",
            "1",
            "24",
            "32",
            "--chunk-mib",
            "0.01",
            "--devices",
            "cpu",
            "--runs",
            "1",
            "--threads",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads((output / "results.json").read_text())
    assert result["status"] == "passed"
    assert [item["frames"] for item in result["inputs"]] == [4, 7]
    assert {case["config"]["chunk_frames"] for case in result["cases"]} == {4}
    assert {case["config"]["effective_batch_size"] for case in result["cases"]} == {4}
    for case in result["cases"]:
        for phase in ("timing", "memory", "accuracy"):
            assert case[phase]["cache_evictions"][0]["resident_pages_after"] == 0
        assert case["accuracy"]["checked_frames"] == case["config"]["input_frames"]
    assert len(result["scaling"]) == 1
    summary = result["scaling"][0]
    assert summary["input_bytes"] == [4 * 3072, 7 * 3072]
    assert summary["input_growth_ratio"] == 7 / 4
    assert len(summary["baseline_rss_bytes"]) == len(summary["peak_rss_bytes"]) == 2
    page = (output / "report.html").read_text()
    assert "Verified cold input file pages" in page
    assert "Fixed-policy size scaling" in page
    assert "not proof of larger-than-RAM" in page


def test_scaling_does_not_pool_failures_or_different_policies():
    def case(size, batch=2, status="passed"):
        return {
            "status": status,
            "config": {
                "input_image_bytes": size,
                "device": "cpu",
                "chunk_frames": 4,
                "effective_batch_size": batch,
                "workers": 1,
                "worker_memory_bytes": 2**30,
                "threads": 1,
            },
            "memory": {
                "baseline_process_tree_rss_bytes": 100,
                "peak_process_tree_rss_bytes": 200,
                "peak_rss_increment_bytes": 100,
            },
        }

    assert (
        BENCHMARK.summarize_scaling(
            [case(1), case(4, batch=3), case(8, status="quality_failed")]
        )
        == []
    )
    summary = BENCHMARK.summarize_scaling([case(4), case(1), case(8, status="error")])
    assert len(summary) == 1
    assert summary[0]["input_bytes"] == [1, 4]
    assert summary[0]["peak_rss_growth_ratio"] == 1


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
