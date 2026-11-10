"""Coverage is diagnostic, not a minimum fraction of the fixed image."""

import numpy as np
import pytest
import torch
from voxeldata import create_voxeldata

from voxreg import register
from voxreg.linear import _optimize
from voxreg.sampling import make_grid

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("shape", [(1, 12, 14), (8, 10, 12)])
@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
@pytest.mark.parametrize("device", DEVICES)
def test_aligned_smaller_moving_image_is_not_rejected(shape, model, device):
    values = np.random.default_rng(14).uniform(0.1, 2, shape)
    fixed = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    moving = fixed.isel(
        k=slice(None) if shape[0] == 1 else slice(2, -2), j=slice(3, -3), i=slice(3, -3)
    )
    original = moving.values.copy()
    result = register(
        moving,
        fixed,
        return_transforms=True,
        transform=model,
        max_iterations=1,
        progress=False,
        device=device,
    )
    assert "forward" in result
    assert result["diagnostics"].to_dataset().status == "max_iterations"
    assert 0 < float(result["diagnostics"].to_dataset().valid_fraction) < 0.5
    assert result["diagnostics"].to_dataset().objective < -0.999999
    np.testing.assert_allclose(
        result["forward"].to_dataset()["transform"], np.eye(4), atol=1e-12, rtol=0
    )
    assert result["images"].to_dataset()["registered"].dtype == np.float64
    np.testing.assert_allclose(
        result["images"]
        .to_dataset()["registered"]
        .values[result["images"].to_dataset()["valid"].values],
        fixed.values[result["images"].to_dataset()["valid"].values],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_array_equal(moving.values, original)


@pytest.mark.parametrize("valid_count", [0, 1, 2])
def test_mathematical_sample_minimum_not_coverage_fraction(monkeypatch, valid_count):
    from voxreg import linear

    values = np.arange(16, dtype=float).reshape(1, 4, 4)
    data = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    grid = make_grid(data, data)
    images = torch.tensor(values)
    valid = torch.zeros_like(images, dtype=torch.bool)
    valid.reshape(-1)[:valid_count] = True

    def sampled(images, parameters, plane):
        return images + parameters.sum(dim=1)[:, None, None] * 0, valid

    monkeypatch.setattr(linear, "sample_rigid", sampled)
    parameters, updates, status = _optimize(
        images, images[0], grid, torch.tensor([False]), 1, 0.1
    )
    assert status.item() == (5 if valid_count == 2 else 2)
    assert updates.item() == (1 if valid_count == 2 else 0)
    assert not parameters.any()


@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
def test_partial_coverage_constant_metric_is_still_invalid(model):
    values = np.arange(12 * 14, dtype=float).reshape(1, 12, 14)
    fixed = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    moving = fixed.isel(j=slice(3, -3), i=slice(3, -3))
    # Fixed varies, but it is constant over the entire moving footprint.
    target = fixed.copy(deep=True)
    target.values[:, 3:-3, 3:-3] = 1
    result = register(
        moving,
        target,
        transform=model,
        max_iterations=2,
        progress=False,
    )
    assert 0 < result["diagnostics"].to_dataset().valid_fraction < 0.5
    assert result["diagnostics"].to_dataset().status == "invalid_metric"
    assert result["diagnostics"].to_dataset().n_iterations == 0


@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
def test_no_overlap_keeps_seed_and_original_fill(model):
    values = np.arange(12 * 14, dtype=float).reshape(1, 12, 14)
    fixed = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    seed = np.eye(4)
    seed[2, 3] = 100
    result = register(
        fixed,
        fixed,
        return_transforms=True,
        transform=model,
        initialization=seed,
        max_iterations=2,
        progress=False,
    )
    assert "forward" in result
    assert result["diagnostics"].to_dataset().status == "insufficient_overlap"
    assert result["diagnostics"].to_dataset().n_iterations == 0
    assert not result["images"].to_dataset()["valid"].any()
    np.testing.assert_allclose(
        result["forward"].to_dataset()["transform"], seed, atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(
        result["images"].to_dataset()["registered"], np.full_like(values, values.min())
    )
