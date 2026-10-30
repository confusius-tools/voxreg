"""Reference comparison must hand off full rigid pulls, not compose residuals."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from confusius import create_voxeldata
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture
def harness():
    path = (
        Path(__file__).resolve().parents[1]
        / "research/confusius-examples/rigid_affine_atlas.py"
    )
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("atlas_chain", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(path.parent))


@pytest.mark.parametrize("backend", ["confusius", "voxreg-cpu", "voxreg-cuda"])
def test_affine_receives_the_full_rigid_result(harness, monkeypatch, backend):
    moving = np.zeros((1, 3, 4), dtype=np.float32)
    fixed = np.zeros((2, 3, 4), dtype=np.float64)
    seed = np.eye(4)
    seed[:3, 3] = [1, 2, 3]
    rigid = np.eye(4)
    rigid[:2, :2] = [[np.cos(0.2), -np.sin(0.2)], [np.sin(0.2), np.cos(0.2)]]
    rigid[:3, 3] = [4, 5, 6]
    affine = rigid.copy()
    affine[0, 1] += 0.02
    calls = []

    @dataclasses.dataclass
    class Diagnostics:
        metric_values: np.ndarray
        n_iterations: int = 4
        status: str = "completed"

    def fake(moving, fixed, **options):
        stage = options.get("transform_type", options.get("transform"))
        calls.append((stage, options["initialization"].copy()))
        matrix = rigid if stage == "rigid" else affine
        registered = np.zeros(fixed.shape, dtype=moving.dtype)
        if backend == "confusius":
            return registered, matrix, Diagnostics(np.array([-0.9]))
        return SimpleNamespace(
            registered=registered,
            transforms=SimpleNamespace(values=matrix),
            diagnostics=SimpleNamespace(
                n_iterations=4,
                status=SimpleNamespace(values=np.array("converged")),
                objective=-0.9,
                valid_fraction=0.9,
                output_valid_fraction=0.0,
            ),
        )

    if backend == "confusius":
        monkeypatch.setattr(harness.cf.registration, "register_volume", fake)
    else:
        monkeypatch.setattr(harness, "register", fake)
    records = harness.fit_chain(moving, fixed, seed, backend)
    assert [stage for stage, _ in calls] == ["rigid", "affine"]
    assert_array_equal(calls[0][1], seed)
    assert_array_equal(calls[1][1], rigid)
    assert_array_equal(records["affine"]["matrix"], affine)
    assert_array_equal(records["affine"]["initialization"], rigid)
    assert_array_equal(seed[:3, 3], [1, 2, 3])


def test_reference_error_uses_inverse_pulls_and_native_pixels(harness):
    native = create_voxeldata(
        np.ones((1, 2, 3)),
        dims=("k", "j", "i"),
        voxel_to_world=np.diag([0.4, 0.1, 0.2, 1]),
    )
    candidate = np.eye(4)
    candidate[0, 3] = -0.3
    result = harness.point_error(candidate, np.eye(4), native)
    assert_allclose([result["rms_mm"], result["max_mm"]], [0.3, 0.3])
    assert_allclose([result["rms_native_pixels"], result["max_native_pixels"]], [3, 3])
    assert result["rms_native_pixels"] > 0.25
    json.dumps(
        result | {"reference_truth_gate_pass": result["rms_native_pixels"] < 0.25},
        allow_nan=False,
    )


def test_label_agreement_does_not_hide_foreground_mismatch(harness):
    reference = np.array([[0, 1], [2, 2]])
    candidate = np.array([[0, 1], [2, 3]])
    result = harness.label_agreement(candidate, reference, np.ones((2, 2), dtype=bool))
    assert result["agreement_fraction"] == 0.75
    assert result["foreground_union_pixels"] == 3
    assert result["foreground_label_agreement"] == 2 / 3
    assert_allclose(result["per_region_dice"]["2"], 2 / 3)
    assert result["per_region_dice"]["3"] == 0
    assert_array_equal(
        harness.boundaries(np.array([[0, 0, 1], [0, 0, 1]])),
        [[False, False, True], [False, False, True]],
    )
