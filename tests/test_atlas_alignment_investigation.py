"""Validate the geometry projection and derivative check used by the diagnosis."""

import importlib
from pathlib import Path

import confusius as cf
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from voxreg import position_to_world


@pytest.fixture
def investigation(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "research/confusius-examples")
    )
    return importlib.import_module("atlas_alignment_investigation")


@pytest.mark.parametrize("oblique", [False, True])
def test_normal_completion_preserves_every_native_center(investigation, oblique):
    affine = np.eye(4)
    rotation = (
        Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix() if oblique else np.eye(3)
    )
    affine[:3, :3] = rotation @ np.diag([0.4, 0.13, 0.09])
    affine[:3, 3] = [1.2, -3, 4]
    native = cf.create_voxeldata(
        np.ones((1, 5, 6)), dims=("k", "j", "i"), voxel_to_world=affine
    )
    pose = np.array(
        [
            [1.1, 0.04, 0.02, 2],
            [-0.03, 0.9, 0.02, -1],
            [0.01, 0, 1.05, 0.3],
            [0, 0, 0, 1.0],
        ]
    )
    pull = np.linalg.inv(pose)
    before = pull.copy()
    projected = np.linalg.inv(investigation.project_native_pull(pull, native))
    points = position_to_world(native) @ np.vstack(
        (np.indices(native.shape).reshape(3, -1), np.ones((1, native.size)))
    )
    assert_allclose(projected @ points, pose @ points, atol=1e-12, rtol=0)
    basis, _ = np.linalg.qr(affine[:3, 1:3])
    normal = np.cross(basis[:, 0], basis[:, 1])
    mapped = projected[:3, :3] @ normal
    tangents = projected[:3, :3] @ basis
    assert_allclose(np.linalg.norm(mapped), 1, atol=1e-12)
    assert_allclose(mapped @ tangents, 0, atol=1e-12)
    assert_array_equal(pull, before)


def test_production_gradient_matches_finite_differences(investigation):
    def image(shape, spacing, origin, offset):
        affine = np.eye(4)
        affine[:3, :3] = np.diag(spacing)
        affine[:3, 3] = origin
        z, y, x = (
            affine
            @ np.vstack(
                (np.indices(shape).reshape(3, -1), np.ones((1, np.prod(shape))))
            )
        )[:3] + np.asarray(offset)[:, None]
        values = np.sin(6 * z + 1.1 * y) + np.cos(5 * x - 3 * y) + 0.2 * z * y
        return cf.create_voxeldata(
            values.reshape(shape), dims=("k", "j", "i"), voxel_to_world=affine
        )

    volume = image((7, 9, 11), [0.15, 0.12, 0.1], [-0.5, -0.6, -0.8], [0, 0, 0])
    native = image(
        (1, 5, 6), [0.4, 0.073, 0.085], [0.0311, -0.2173, -0.2731], [0.03, 0.02, 0.04]
    )
    engine = importlib.import_module("voxreg.rigid")
    before = engine._optimize
    result = investigation.gradient_check(native, volume, np.eye(4))
    assert result["double_FD_relative_L2_error"] < 1e-5
    assert result["mixed_double_relative_L2_error"] < 1e-3
    assert engine._optimize is before
