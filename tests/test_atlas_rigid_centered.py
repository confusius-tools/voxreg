"""Research controls preserve tilt while preventing off-plane collapse."""

import importlib
from pathlib import Path

import confusius as cf
import numpy as np
import pytest
import SimpleITK as sitk
from numpy.testing import assert_allclose


@pytest.fixture
def control(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "research/confusius-examples")
    )
    return importlib.import_module("atlas_rigid_centered")


@pytest.mark.parametrize("builder", ["centered_rigid", "centered_unit_normal"])
def test_seed_rotation_translation_and_center_preserved(control, builder):
    rotation = sitk.Euler3DTransform()
    rotation.SetRotation(0.12, -0.2, 0.08)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(rotation.GetMatrix()).reshape(3, 3)
    matrix[:3, 3] = [2, -3, 4]
    center = np.array([0.4, 5.99, -0.2])
    transform = getattr(control, builder)(matrix, center)
    assert_allclose(control.linear_matrix(transform), matrix, atol=1e-12, rtol=0)
    assert_allclose(transform.GetCenter(), center, atol=0, rtol=0)
    for point in [[0.0, 0.0, 0.0], [2.0, -1.0, 3.0]]:
        assert_allclose(
            transform.TransformPoint(point),
            (matrix @ np.r_[point, 1])[:3],
            atol=1e-12,
            rtol=0,
        )
    assert transform.GetNumberOfParameters() == (
        6 if builder == "centered_rigid" else 12
    )


def test_reject_nonrigid_seed(control):
    matrix = np.eye(4)
    matrix[0, 0] = 0.02
    with pytest.raises(ValueError, match="proper orthogonal"):
        control.centered_rigid(matrix, np.zeros(3))


def test_unit_normal_allows_tilt_and_in_plane_scale_shear(control):
    transform = control.centered_unit_normal(np.eye(4), [0.4, 5.99, 0.0])
    params = list(transform.GetParameters())
    params[:3] = [0.08, -0.12, 0.05]
    params[7:9] = [0.85, 1.15]
    params[11] = 0.2
    transform.SetParameters(params)
    matrix = control.linear_matrix(transform)
    normal = matrix[:3, 0]
    tangents = matrix[:3, 1:3]
    assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-12, rtol=0)
    assert_allclose(normal @ tangents, [0, 0], atol=1e-12, rtol=0)
    assert (
        abs(normal[0]) < 0.99
    )  # Tilt is genuinely free, not a fixed world row/column.
    assert not np.allclose(tangents.T @ tangents, np.eye(2))
    assert abs(tangents[:, 0] @ tangents[:, 1]) > 0.01
    np.testing.assert_allclose(
        np.cross(tangents[:, 0], tangents[:, 1])
        / np.linalg.norm(np.cross(tangents[:, 0], tangents[:, 1])),
        normal,
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("model", ["rigid", "unit_normal_affine", "affine"])
def test_actual_cf_transform_and_metric_pairing(control, model):
    data = cf.create_voxeldata(
        np.random.default_rng(41).uniform(1, 2, (5, 7, 9)).astype(np.float32),
        dims=("k", "j", "i"),
        voxel_to_world=np.diag([0.1, 0.1, 0.1, 1.0]),
    )
    record, states = control.fit_control(
        data, data, np.eye(4), model=model, max_iterations=5, window=6
    )
    assert (
        record["free_parameters"]
        == {"rigid": 6, "unit_normal_affine": 9, "affine": 12}[model]
    )
    assert len(record["parameter_states"]) == len(states) == 6
    assert record["settings"]["window"] > record["settings"]["max_iterations"]
    assert record["event_pairing_verified"]
    assert record["max_normal_scale_error_all_states"] < 1e-10
    assert record["max_normal_tangent_dot_all_states"] < 1e-10
    assert states.shape[0] == len(record["raw_metric"])


@pytest.mark.parametrize("model", ["rigid", "unit_normal_affine", "affine"])
def test_all_patches_restored_on_failure(control, monkeypatch, model):
    volume = importlib.import_module("confusius.registration.volume")
    original = (
        volume.affine_to_sitk_linear_transform,
        volume.sitk_linear_transform_to_affine,
        sitk.ImageRegistrationMethod,
        sitk.Transform.Downcast,
    )

    def fail(*args, **kwargs):
        assert kwargs["transform_type"] == ("rigid" if model == "rigid" else "affine")
        if model == "unit_normal_affine":
            assert kwargs["optimizer_weights"] == [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
            ]
        raise RuntimeError("fit failed")

    monkeypatch.setattr(cf.registration, "register_volume", fail)
    data = cf.create_voxeldata(
        np.ones((1, 5, 7)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    )
    with pytest.raises(RuntimeError, match="fit failed"):
        control.fit_control(data, data, np.eye(4), model=model)
    assert (
        volume.affine_to_sitk_linear_transform,
        volume.sitk_linear_transform_to_affine,
        sitk.ImageRegistrationMethod,
        sitk.Transform.Downcast,
    ) == original
