"""Fixed-work registration benchmark CLI and native update-count validation."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

SPEC = importlib.util.spec_from_file_location(
    "benchmark_iterations", ROOT / "benchmarks/_benchmark_iterations.py"
)
assert SPEC is not None and SPEC.loader is not None
NATIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NATIVE)


@pytest.mark.parametrize("prefix", ["1DIAGNOSTIC", "2DIAGNOSTIC"])
def test_native_iteration_log_requires_every_update(tmp_path, prefix):
    path = tmp_path / "native.log"
    path.write_text(
        "DIAGNOSTIC,Iteration,metricValue\n"
        + "".join(f" {prefix}, {i}, -0.9, inf,\n" for i in range(1, 4))
    )
    assert NATIVE.read_iteration_count(path, 3) == 3


@pytest.mark.parametrize("ids", [[], [1, 2], [1, 2, 3, 4], [1, 1, 3], [1, 3]])
def test_native_iteration_log_rejects_short_or_invalid_sequences(tmp_path, ids):
    path = tmp_path / "native.log"
    path.write_text("".join(f"1DIAGNOSTIC, {i}, -0.9, inf,\n" for i in ids))
    with pytest.raises(ValueError, match="iteration count mismatch"):
        NATIVE.read_iteration_count(path, 3)


@pytest.mark.parametrize("capture", [False, True])
def test_native_count_logging_is_accuracy_only(tmp_path, monkeypatch, capfd, capture):
    calls = []
    fake = ModuleType("ants")

    def registration(*, fixed, moving):
        calls.append(fixed)
        assert moving is None
        if "--verbose" in fixed:
            os.write(1, b"1DIAGNOSTIC, 1, -0.9, inf,\n")

    fake.__dict__["registration"] = registration
    monkeypatch.setitem(sys.modules, "ants", fake)
    path = tmp_path / "native.log"
    observed = NATIVE.run_ants_registration(
        ["--convergence", "[1,1e-7,2]"], 1, path if capture else None
    )
    assert observed == (1 if capture else None)
    assert ("--verbose" in calls[0]) == capture
    assert path.exists() == capture
    os.write(1, b"stdout restored\n")
    assert capfd.readouterr().out == "stdout restored\n"


def test_native_registration_error_restores_stdout(tmp_path, monkeypatch, capfd):
    fake = ModuleType("ants")

    def registration(**kwargs):
        os.write(1, b"native error evidence\n")
        raise RuntimeError("native failed")

    fake.__dict__["registration"] = registration
    monkeypatch.setitem(sys.modules, "ants", fake)
    path = tmp_path / "native.log"
    with pytest.raises(RuntimeError, match="native failed"):
        NATIVE.run_ants_registration([], 1, path)
    os.write(1, b"stdout restored\n")
    assert capfd.readouterr().out == "stdout restored\n"
    assert path.read_text() == "native error evidence\n"


@pytest.fixture
def fake_antspy_environment(tmp_path):
    package = tmp_path / "python" / "ants"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        """import os
from pathlib import Path
import numpy as np
REGISTRY = {}
class Image:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32).copy()
        self.dimension = self.values.ndim
    def clone(self, pixeltype=None):
        return Image(self.values)
    def numpy(self):
        return self.values.copy()
def from_numpy(values, **geometry):
    return Image(values)
def apply_transforms_to_points(dimensions, queries, paths, **kwargs):
    assert all(Path(p).is_file() for p in paths)
    return queries.copy()
def registration(*, fixed, moving):
    assert isinstance(fixed, list) and moving is None
    args = fixed
    get = lambda key: args[args.index(key) + 1]
    iterations, tolerance, window = get('--convergence').strip('[]').split(',')
    iterations, window = int(iterations), int(window)
    assert window > iterations
    assert get('--shrink-factors') == '1'
    assert get('--smoothing-sigmas') == '0'
    assert get('--initial-moving-transform') == 'Identity'
    assert get('--float') == '1'
    prefix, warped, reverse = get('--output').strip('[]').split(',')
    target, source = get('--metric').split('[', 1)[1].split(',')[:2]
    REGISTRY[warped].values = REGISTRY[source].values.copy()
    REGISTRY[reverse].values = REGISTRY[target].values.copy()
    Path(prefix + '0GenericAffine.mat').write_text('identity')
    if get('--transform').startswith('SyN['):
        Path(prefix + '1Warp.nii.gz').write_text('identity')
        Path(prefix + '1InverseWarp.nii.gz').write_text('identity')
    verbose = '--verbose' in args
    assert verbose == ('accuracy-' in prefix)
    if verbose:
        count = iterations - int(os.environ.get('SHORT_NATIVE_COUNT', '0'))
        for i in range(1, count + 1):
            os.write(1, f'1DIAGNOSTIC, {i}, -0.9, inf,\\n'.encode())
    return 0
"""
    )
    (package / "internal.py").write_text(
        "from ants import REGISTRY\n"
        "def get_pointer_string(image):\n"
        "    key = str(id(image))\n"
        "    REGISTRY[key] = image\n"
        "    return key\n"
    )
    metadata = package.parent / "antspyx-0.6.3.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: antspyx\nVersion: 0.6.3\n")
    return {**os.environ, "PYTHONPATH": str(package.parent)}


@pytest.mark.parametrize("model", ["rigid", "affine", "syn"])
def test_cli_requests_200_fixed_updates_without_timing_logs(
    tmp_path, fake_antspy_environment, model
):
    output = tmp_path / "benchmark"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/benchmark_register.py"),
            "--output",
            str(output),
            "--configs",
            f"resident-3d-{model}-antspyx",
            "--antspy-python",
            sys.executable,
            "--runs",
            "1",
            "--threads",
            "1",
        ],
        cwd=ROOT,
        env=fake_antspy_environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    report = json.loads((output / "results.json").read_text())
    assert report["protocol"] == "voxreg-benchmark-v4"
    assert report["resident_iterations"] == 200
    case = report["cases"][0]
    assert case["status"] == "quality_failed"
    expected_transform = {
        "rigid": "Rigid[0.25]",
        "affine": "Affine[0.25]",
        "syn": "SyN[0.2,2.25,0.25]",
    }[model]
    for phase in ("timing", "memory", "accuracy"):
        assert (
            case[phase]["backend_settings"]["settings"]["transform"]
            == expected_transform
        )
    for phase in ("timing", "memory"):
        assert case[phase]["status"] == "passed"
        assert case[phase]["backend_settings"]["updates"] == [None]
    assert case["accuracy"]["backend_settings"]["updates"] == [200]
    assert case["accuracy"]["backend_settings"]["statuses"] == ["max_iterations"]
    assert case["accuracy"]["worst_frame_rms_voxels"] > 0.3
    assert "DIAGNOSTIC" not in (output / case["id"] / "timing.log").read_text()


def test_cli_fails_if_native_updates_are_short(tmp_path, fake_antspy_environment):
    output = tmp_path / "benchmark"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/benchmark_register.py"),
            "--output",
            str(output),
            "--configs",
            "resident-3d-syn-antspyx",
            "--antspy-python",
            sys.executable,
            "--runs",
            "1",
            "--threads",
            "1",
        ],
        cwd=ROOT,
        env={**fake_antspy_environment, "SHORT_NATIVE_COUNT": "1"},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    report = json.loads((output / "results.json").read_text())
    case = report["cases"][0]
    assert case["accuracy"]["status"] == "error"
    assert "iteration count mismatch" in case["accuracy"]["error"]
