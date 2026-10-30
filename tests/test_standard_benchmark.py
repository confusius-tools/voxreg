"""Shared pair/series protocol: real backend pulls, independent scoring and safe CLI."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / filename)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


workloads = module("workloads", "benchmark_workloads.py")
prepare = module("prepare_benchmark", "prepare_registration_benchmark.py")


def pair(tmp_path):
    arrays, provenance = prepare.synthetic(32, True)
    path = tmp_path / "pair.npz"
    np.savez(path, **arrays, provenance=json.dumps(provenance))
    return path, workloads.load_dataset(path, "pair")


def options(**kwargs):
    return SimpleNamespace(
        max_iterations=100, policy="fixed", intensity_scaling="none", **kwargs
    )


@pytest.mark.parametrize("backend", ["itk", "fireants-cpu", "fireants-cuda"])
def test_direct_3d_backends_recover_known_physical_motion(tmp_path, backend):
    import torch

    _, data = pair(tmp_path)
    if backend.endswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("itk" if backend == "itk" else "fireants")
    result = (
        workloads.run_itk_standard(data, options(), 1)
        if backend == "itk"
        else workloads.run_fireants_standard(data, options(), backend.split("-")[1], 1)
    )
    path = tmp_path / "result.npz"
    np.savez(path, **result)
    quality = workloads.score_standard(
        tmp_path / "pair.npz", {backend: path}, "pair", "none"
    )
    assert quality[backend]["rms_truth_pixels"][0] < 0.25
    assert quality[backend]["median_ncc2"] > 0.995
    assert quality[backend]["median_ncc2"] > quality["identity"]["median_ncc2"]
    np.testing.assert_array_equal(result["n_iterations"], [100])
    assert result["registered"].shape == (1, *data["fixed"].shape)
    assert result["registered"].dtype == np.float32
    from scipy.ndimage import map_coordinates

    mapped, valid = workloads.coordinates(data, result["transforms"][0])
    expected = map_coordinates(
        data["moving"][0].astype(float),
        mapped,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    np.testing.assert_allclose(
        result["registered"][0][valid], expected[valid], atol=4e-5, rtol=0
    )


@pytest.mark.parametrize("backend", ["itk", "fireants"])
def test_standard_series_reference_and_batching(recording_factory, tmp_path, backend):
    data, _, _ = recording_factory()
    path = tmp_path / "series.npz"
    np.savez(
        path,
        values=data.values,
        voxel_to_world=data.xindexes["z"].voxel_to_world,
        **{dim: data[dim].values for dim in data.dims},
    )
    loaded = workloads.load_dataset(path, "series")
    args = options()
    args.intensity_scaling = "db"
    result = (
        workloads.run_itk_standard(loaded, args, 1)
        if backend == "itk"
        else workloads.run_fireants_standard(loaded, args, "cpu", 2)
    )
    np.testing.assert_array_equal(result["registered"][0], data.values[0])
    np.testing.assert_array_equal(result["transforms"][0], np.eye(4))
    np.testing.assert_array_equal(result["n_iterations"], [0, 100, 100])
    assert result["status"][0] == "reference"
    assert result["valid"][0].all()


@pytest.mark.parametrize("backend", ["itk", "fireants"])
def test_native_convergence_mode_is_bounded_and_scored(tmp_path, backend):
    path, data = pair(tmp_path)
    args = options()
    args.policy = "convergence"
    result = (
        workloads.run_itk_standard(data, args, 1)
        if backend == "itk"
        else workloads.run_fireants_standard(data, args, "cpu", 1)
    )
    assert 0 < result["n_iterations"][0] <= 100
    output = tmp_path / "converged.npz"
    np.savez(output, **result)
    quality = workloads.score_standard(path, {backend: output}, "pair", "none")
    assert quality[backend]["rms_truth_pixels"][0] < 0.25


def test_common_support_and_exact_truth_error_scoring(tmp_path):
    path, data = pair(tmp_path)
    output = tmp_path / "truth.npz"
    np.savez(output, transforms=data["truth"])
    quality = workloads.score_standard(path, {"exact": output}, "pair", "none")
    assert quality["exact"]["rms_truth_pixels"] == [0]
    assert quality["exact"]["max_pointwise_truth_pixels"] == [0]
    assert quality["exact"]["median_ncc2"] == quality["truth"]["median_ncc2"]
    lattice = np.indices(data["fixed"].shape).reshape(3, -1)
    points = data["af"] @ np.vstack((lattice, np.ones((1, lattice.shape[1]))))
    error = ((np.eye(4) - data["truth"][0]) @ points)[:3]
    distance = (
        np.linalg.norm(error, axis=0) / np.linalg.norm(data["af"][:3, :3], axis=0).min()
    )
    np.testing.assert_allclose(
        quality["identity"]["rms_truth_pixels"],
        [np.sqrt(np.mean(distance**2))],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        quality["identity"]["max_pointwise_truth_pixels"], [distance.max()], rtol=1e-12
    )
    bad = np.eye(4)
    bad[0, 3] = 1000
    np.savez(output, transforms=bad[None])
    missing = workloads.score_standard(path, {"outside": output}, "pair", "none")
    assert missing["outside"]["median_ncc2"] is None
    assert missing["outside"]["common_fraction_median"] == 0
    json.dumps(missing, allow_nan=False)


@pytest.mark.parametrize(
    "defect", ["complex", "nonfinite", "constant", "affine", "shear", "small", "truth"]
)
def test_standard_input_rejects_unsupported_data(tmp_path, defect):
    arrays, _ = prepare.synthetic(32, False)
    if defect == "complex":
        arrays["moving"] = arrays["moving"].astype(complex)
    if defect == "nonfinite":
        arrays["moving"][0, 0, 0] = np.nan
    if defect == "constant":
        arrays["fixed"][:] = 1
    if defect == "affine":
        arrays["moving_affine"][3, 0] = 1
    if defect == "shear":
        arrays["moving_affine"][0, 1] += 0.1
    if defect == "small":
        arrays["moving"] = arrays["moving"][:31]
    if defect == "truth":
        arrays["truth"][0, 0] = np.nan
    path = tmp_path / "bad.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        workloads.load_dataset(path, "pair")


def test_template_control_has_real_values_but_synthetic_pose(tmp_path):
    import nibabel as nib

    path = tmp_path / "template.nii.gz"
    values = np.arange(32**3, dtype=np.float64).reshape((32,) * 3)
    image = nib.Nifti1Image(values, np.diag([0.1, 0.2, 0.3, 1]))
    image.set_qform(image.affine, code=1)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    arrays, provenance = prepare.template_pair(path)
    np.testing.assert_array_equal(arrays["moving"], arrays["fixed"])
    np.testing.assert_allclose(
        arrays["moving_affine"], arrays["truth"] @ arrays["fixed_affine"]
    )
    assert arrays["moving"].dtype == np.float32
    assert (
        provenance["kind"] == "template-known-pose"
        and "not independent sessions" in provenance["warning"]
    )


def test_power_snapshots_tolerate_unavailable_sysfs(monkeypatch):
    runner = module("benchmark_power", "benchmark_registration.py")
    assert "/sys/firmware/acpi/platform_profile" in runner.power_state()

    def unavailable(self):
        raise PermissionError("not readable")

    monkeypatch.setattr(Path, "read_text", unavailable)
    assert all(value is None for value in runner.power_state().values())


def test_shared_runner_cli_and_refusal_to_overwrite(tmp_path):
    path, _ = pair(tmp_path)
    output = tmp_path / "run"
    command = [
        sys.executable,
        str(ROOT / "examples/benchmark_registration.py"),
        "--workload",
        "pair",
        "--input",
        str(path),
        "--output",
        str(output),
        "--cases",
        "cpu-1",
        "--max-iterations",
        "2",
        "--repeats",
        "1",
    ]
    env = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    run = subprocess.run(
        command, check=False, env=env, capture_output=True, text=True, timeout=90
    )
    assert run.returncode == 0, run.stderr
    result = json.loads((output / "summary.json").read_text())
    assert result["protocol"] == "standardized-v1" and result["workload"] == "pair"
    assert result["results"]["cpu-1"]["time"]["n_iterations"] == [2]
    assert "power_before_measurements" in result["results"]["cpu-1"]["time"]
    assert "power_after_measurements" in result["results"]["cpu-1"]["time"]
    assert result["results"]["cpu-1"]["memory"]["tree_peak_pss_mib"] > 0
    assert result["source_sha256"] and result["input_sha256"]
    repeated = subprocess.run(
        command, check=False, env=env, capture_output=True, text=True, timeout=30
    )
    assert repeated.returncode != 0 and "must be empty" in repeated.stderr
    existing = tmp_path / "existing.npz"
    existing.write_bytes(b"user input")
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/prepare_registration_benchmark.py"),
            "--output",
            str(existing),
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode != 0 and existing.read_bytes() == b"user input"
