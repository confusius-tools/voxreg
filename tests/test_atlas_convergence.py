"""A rigid correction diagnostic must not hide remaining affine differences."""

import importlib
from pathlib import Path

import confusius as cf
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "research/confusius-examples")
    )
    return importlib.import_module("atlas_convergence")


def native_slice():
    affine = np.eye(4)
    affine[:3, 3] = [2.25, -1.0, 3.0]
    return cf.create_voxeldata(
        np.ones((1, 5, 6)), dims=("k", "j", "i"), voxel_to_world=affine
    )


def test_known_rigid_difference_is_removed(diagnostic):
    pose = np.eye(4)
    vector = np.array([0.1, -0.2, 0.05])
    pose[:3, :3] = Rotation.from_rotvec(vector).as_matrix()
    pose[:3, 3] = [0.3, -0.2, 0.1]
    result = diagnostic.rigid_residual(np.linalg.inv(pose), np.eye(4), native_slice())
    assert_allclose(
        result["rigid_correction_angle_degrees"],
        np.degrees(np.linalg.norm(vector)),
        atol=1e-12,
    )
    assert result["rms_after_rigid_correction_mm"] < 1e-12
    assert_allclose(result["fraction_squared_error_explained"], 1.0, atol=1e-12)


def test_affine_scale_difference_is_not_removed_by_rigid_correction(diagnostic):
    pose = np.diag([1.0, 1.2, 0.8, 1.0])
    result = diagnostic.rigid_residual(np.linalg.inv(pose), np.eye(4), native_slice())
    assert 0 < result["rms_after_rigid_correction_mm"] < result["rms_before_mm"]
    assert 0 < result["fraction_squared_error_explained"] < 1


def test_identical_poses_have_zero_error(diagnostic):
    result = diagnostic.rigid_residual(np.eye(4), np.eye(4), native_slice())
    assert result["rms_before_mm"] == 0
    assert result["rms_after_rigid_correction_mm"] < 1e-12
    assert result["fraction_squared_error_explained"] is None
