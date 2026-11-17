"""Combined offline report CLI preserves operation and protocol boundaries."""

import base64
import json
import os
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class ReportParser(HTMLParser):
    """Collect embedded plots, tables and detail disclosure state."""

    def __init__(self):
        super().__init__()
        self.images = []
        self.tables_outside_details = 0
        self.open_details = 0
        self.external_resources = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "details":
            assert "open" not in attrs
            self.open_details += 1
        if tag == "table" and self.open_details == 0:
            self.tables_outside_details += 1
        if tag == "img":
            assert attrs.get("alt")
            self.images.append(attrs["src"])
        for key in ("src", "href"):
            value = attrs.get(key, "")
            if value and not value.startswith(("#", "data:")):
                self.external_resources.append(value)

    def handle_endtag(self, tag):
        if tag == "details":
            self.open_details -= 1


def eager_report(operation):
    resampling = operation == "resample"
    error_key = "worst_frame_relative_rmse" if resampling else "worst_frame_rms_voxels"
    return {
        "protocol": "voxreg-resample-benchmark-v3"
        if resampling
        else "voxreg-benchmark-v4",
        "kind": "benchmark",
        "status": "completed",
        "runs": 5,
        "git_commit": "eager-revision",
        "host": {"cpu": "fixture CPU"},
        "policy": {"relative_rmse_limit": 1e-5}
        if resampling
        else {"truth_rms_limit_voxels": 0.3, "truth_rms_comparison": "<="},
        "cases": [
            {
                "id": "resident-3d-rigid-cpu",
                "backend": "cpu",
                "model": "rigid",
                "workload": "3d",
                "status": "passed",
                "timing": {"status": "passed", "wall_seconds": [1, 2, 3, 4, 5]},
                "memory": {"status": "passed", "peak_ram_bytes": 2**30},
                "accuracy": {
                    "status": "passed",
                    "accuracy_pass": True,
                    error_key: 1e-7,
                },
            }
        ],
    }


def lazy_report(operation, protocol="voxreg-lazy-benchmark-v3"):
    cases = []
    for size in (1, 4):
        eviction = {
            "verified": True,
            "resident_pages_after": 0,
            "worker_cached_tasks": {"worker": 0},
        }
        cases.append(
            {
                "id": f"{operation}-cuda-{size}",
                "status": "passed",
                "config": {
                    "operation": operation,
                    "device": "cuda",
                    "runs": 5,
                    "input_image_bytes": size * 2**30,
                    "chunk_frames": 64,
                    "effective_batch_size": 16,
                    "workers": 1,
                    "worker_memory_bytes": 4 * 2**30,
                    "threads": 8,
                    "iterations": 200,
                },
                "timing": {
                    "status": "passed",
                    "wall_seconds": [size * v for v in (2, 3, 4, 5, 6)],
                    "cache_evictions": [eviction] * 5,
                },
                "memory": {
                    "status": "passed",
                    "baseline_process_tree_rss_bytes": 2**30,
                    "peak_process_tree_rss_bytes": 3 * 2**30,
                    "peak_rss_increment_bytes": 2 * 2**30,
                    "worker_gpu_peaks": {
                        "a": {"allocated_bytes": 2**29, "reserved_bytes": 2**30}
                    },
                    "cache_evictions": [eviction],
                },
                "accuracy": {
                    "status": "passed",
                    "accuracy_pass": True,
                    "worst_relative_intensity_rmse": 2e-8,
                    "intensity_limit": 1e-5,
                    "worst_truth_rms_voxels": 0.01,
                    "truth_limit_voxels": 0.3,
                    "cache_evictions": [eviction],
                },
            }
        )
    return {
        "protocol": protocol,
        "operation": operation,
        "status": "passed",
        "diagnostic": False,
        "host": {"platform": "fixture platform"},
        "cases": cases,
        "timing_boundary": "cold read/compute/buffered write",
        "source_sha256": {"example.py": "abc"},
    }


def render(tmp_path, reports):
    inputs = []
    for index, report in enumerate(reports):
        directory = tmp_path / f"experiment-{index}"
        directory.mkdir()
        (directory / "results.json").write_text(json.dumps(report))
        inputs.append(str(directory))
    output = tmp_path / "combined.html"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/benchmark_report.py"),
            "--output",
            str(output),
            *inputs,
        ],
        env={
            **os.environ,
            "PYTHONUTF8": "0",
            "PYTHONCOERCECLOCALE": "0",
            "LC_ALL": "C",
        },
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    return result, output


def test_combined_report_is_graph_first_self_contained_and_separated(tmp_path):
    reports = [
        eager_report("register"),
        eager_report("resample"),
        lazy_report("register"),
        lazy_report("resample"),
    ]
    result, output = render(tmp_path, reports)
    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    parser = ReportParser()
    parser.feed(text)
    assert len(parser.images) == 15
    assert parser.tables_outside_details == 0
    assert parser.external_resources == []
    assert parser.open_details == 0
    assert 'id="run-4"' in text
    assert "Overview" in text and "Failures and incomplete cases (0)" in text
    assert "Lazy register" in text and "Lazy resample" in text
    assert "14/14" in text and "14 expected passes" in text
    assert "eager-revision" in text and "abc" in text
    plots = [
        base64.b64decode(image.split(",", 1)[1]).decode() for image in parser.images
    ]
    assert all("<svg" in plot for plot in plots)
    joined = "\n".join(plots)
    for label in (
        "median",
        "baseline",
        "increment",
        "allocated",
        "reserved",
    ):
        assert label.lower() in joined.lower()
    assert "independent SciPy reference" in text
    assert "<script" not in text
    assert "●" in text and "GiB" in text


@pytest.mark.parametrize(
    "protocol",
    [
        "voxreg-lazy-benchmark-v1",
        "voxreg-lazy-benchmark-v2",
        "voxreg-lazy-benchmark-v3",
    ],
)
def test_lazy_partial_and_failed_cases_remain_unqualified(tmp_path, protocol):
    report = lazy_report("register", protocol)
    report["status"] = "interrupted"
    complete, partial = report["cases"]
    complete["status"] = "quality_failed"
    complete["accuracy"]["status"] = "quality_failed"
    complete["accuracy"]["accuracy_pass"] = False
    complete["accuracy"]["error"] = "bad <signal>"
    partial["status"] = "running"
    partial.pop("memory")
    partial.pop("accuracy")
    partial["timing"] = {"status": "running"}
    result, output = render(tmp_path, [report])
    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "Failures and incomplete cases (2)" in text
    assert "bad &lt;signal&gt;" in text and "bad <signal>" not in text
    assert "Partial run" in text
    assert "7/7" in text and "14 expected passes" in text
    parser = ReportParser()
    parser.feed(text)
    plots = "\n".join(
        base64.b64decode(image.split(",", 1)[1]).decode() for image in parser.images
    )
    assert "<!-- None -->" not in plots
    assert "cuda · chunk 64" not in plots  # No qualified scaling line or legend.


def test_missing_lazy_memory_is_omitted_not_zero_and_cache_claims_are_checked(tmp_path):
    report = lazy_report("resample")
    for case in report["cases"]:
        case["memory"].pop("worker_gpu_peaks")
        case["memory"].pop("peak_process_tree_rss_bytes")
        for phase in ("timing", "memory", "accuracy"):
            for eviction in case[phase]["cache_evictions"]:
                eviction["worker_cached_tasks"] = {"worker": 1}
    result, output = render(tmp_path, [report])
    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "0/14" in text
    assert "Largest per-worker GPU peak" not in text
    assert "<td>—</td>" in text


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf")])
def test_invalid_lazy_measurements_fail_closed(tmp_path, value):
    report = lazy_report("register")
    report["cases"][0]["memory"]["peak_process_tree_rss_bytes"] = value
    result, output = render(tmp_path, [report])
    assert result.returncode != 0
    assert "Invalid lazy measurement" in result.stderr
    assert not output.exists()


def test_unsupported_protocol_does_not_overwrite_a_result(tmp_path):
    result, output = render(tmp_path, [{"protocol": "unknown", "cases": []}])
    assert result.returncode != 0
    assert "Unsupported protocol" in result.stderr
    assert not output.exists()
