"""Changing an affine center must not change the initial physical mapping."""

import importlib
from pathlib import Path

import confusius as cf
import numpy as np
import pytest
from confusius.registration.affines import sitk_linear_transform_to_affine
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture
def centering(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "research/confusius-examples")
    )
    return importlib.import_module("atlas_centering")


def test_recenter_nontrivial_affine_preserves_mapping(centering):
    matrix = np.array(
        [
            [1.1, 0.03, -0.02, 2.0],
            [-0.04, 0.9, 0.02, -3.0],
            [0.01, 0.02, 1.04, 4.0],
            [0, 0, 0, 1.0],
        ]
    )
    before = matrix.copy()
    center = np.array([0.4, 5.99, -0.3])
    transform = centering.centered_affine(matrix, center)
    assert_allclose(
        sitk_linear_transform_to_affine(transform), matrix, atol=1e-12, rtol=0
    )
    for point in [[0, 0, 0], [0.4, 5.99, -0.3], [2, -1, 3]]:
        assert_allclose(
            transform.TransformPoint(point),
            (matrix @ np.r_[point, 1])[:3],
            atol=1e-12,
            rtol=0,
        )
    assert_array_equal(matrix, before)
    assert_array_equal(transform.GetCenter(), center)


def test_center_uses_physical_coordinates(centering):
    affine = np.array([[0, 0, 0.1, 4], [0, 0.2, 0, -2], [0.4, 0, 0, 7], [0, 0, 0, 1.0]])
    data = cf.create_voxeldata(
        np.ones((1, 5, 7)), dims=("k", "j", "i"), voxel_to_world=affine
    )
    assert_allclose(centering.physical_center(data), [4.3, -1.6, 7])


def test_seed_converter_restored_on_failure(centering, monkeypatch):
    module = importlib.import_module("confusius.registration.volume")
    original = module.affine_to_sitk_linear_transform

    def fail(*args, **kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(centering, "fit_curve", fail)
    with pytest.raises(RuntimeError, match="fit failed"):
        centering.fit_centered(None, None, np.eye(4), np.ones(3))
    assert module.affine_to_sitk_linear_transform is original
