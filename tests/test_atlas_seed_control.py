"""Zero-update control separates transform placement from differing FOV support."""

import importlib
from pathlib import Path

import confusius as cf
import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture
def control(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "research/confusius-examples"
    monkeypatch.syspath_prepend(str(path))
    return importlib.import_module("atlas_seed_control")


def image(values, origin):
    affine = np.eye(4)
    affine[:3, 3] = origin
    return cf.create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=affine)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_hold_preserves_seed_resamples_and_restores_optimizer(control, device):
    volume = image(np.arange(125, dtype=float).reshape(5, 5, 5), [0, 0, 0])
    target = image(np.arange(9, dtype=float).reshape(1, 3, 3), [2, 1, 1])
    engine = importlib.import_module("voxreg.rigid")
    original = engine._optimize
    seed = np.eye(4)
    result, warnings = control.held_register(volume, target, seed, device=device)
    assert engine._optimize is original
    assert not warnings
    assert int(result.diagnostics.n_iterations) == 0
    assert_array_equal(result.transforms, seed)
    assert_array_equal(seed, np.eye(4))
    assert result.valid.values.all()
    assert_allclose(
        result.registered.values, volume.values[2:3, 1:4, 1:4], atol=1e-12, rtol=0
    )


def test_mixed_half_voxel_border_matches_itk_support(control):
    volume = image(np.arange(125, dtype=float).reshape(5, 5, 5), [0, 0, 0])
    target = image(np.arange(9, dtype=float).reshape(1, 3, 3), [2, -0.25, 1])
    result, _ = control.held_register(volume, target, np.eye(4))
    reference = cf.registration.resample_like(
        volume, target, np.eye(4), interpolation="linear", sitk_threads=1
    )
    assert result.valid.values.all()
    assert_allclose(reference.values[0, 0], volume.values[2, 0, 1:4], atol=1e-12)
    assert_allclose(result.registered.values, reference.values, atol=1e-12)
    error = control.mapping_error(result.transforms.values, np.eye(4), target, volume)
    assert error["max_source_pixel_mapping_error"] == 0


def test_empty_common_support_is_not_reported_as_agreement(control):
    result = control.compare_arrays(
        np.ones((2, 2)),
        np.ones((2, 2), dtype=bool),
        np.zeros((2, 2)),
        np.zeros((2, 2), dtype=bool),
    )
    assert result["common_centers"] == 0
    assert result["common_max_abs"] is None
    assert result["common_rms"] is None
    assert result["support_disagreement_centers"] == 4
    assert result["whole_output_max_abs"] == 1
