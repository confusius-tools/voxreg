"""FireANTs physical seeding, loss ownership and finite slab geometry controls."""

import importlib
from pathlib import Path

import confusius as cf
import numpy as np
import pytest
import SimpleITK as sitk
import torch
from numpy.testing import assert_allclose


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "research/confusius-examples")
    )
    return importlib.import_module("atlas_fireants")


@pytest.fixture
def volume():
    return cf.create_voxeldata(
        np.random.default_rng(48).uniform(-20, -5, (4, 7, 9)).astype(np.float32),
        dims=("k", "j", "i"),
        voxel_to_world=np.diag([0.1, 0.1, 0.1, 1.0]),
    )


def test_expanded_geometry_preserves_cell_extent_and_center(experiment):
    a = np.diag([0.4, 0.05, 0.1, 1.0])
    a[:3, 3] = [0.4, 3.0, -4.0]
    values = np.random.default_rng(47).normal(size=(1, 7, 9)).astype(np.float32)
    data = cf.create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=a)
    expanded = experiment.expanded_voxeldata(data)
    from voxreg import position_to_world

    b = position_to_world(expanded)
    assert expanded.shape == (4, 7, 9)
    assert_allclose(b[:3, 0] * 4, a[:3, 0], atol=1e-15, rtol=0)
    assert_allclose(b[:3, 3] + b[:3, 0] * 1.5, a[:3, 3], atol=1e-15, rtol=0)
    assert_allclose(
        expanded.values, np.broadcast_to(values, expanded.shape), atol=0, rtol=0
    )


def test_global_ncc_masks_outside_without_nan(experiment):
    image = torch.randn(1, 2, 4, 5, 6)
    image[:, 1] = 1
    assert_allclose(
        experiment.masked_global_ncc(image, image).item(), -1, atol=1e-6, rtol=0
    )
    image[:, 1] = 0
    image.requires_grad_(True)
    value = experiment.masked_global_ncc(image, image.detach())
    value.backward()
    assert value.item() == 0 and torch.isfinite(image.grad).all()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize("model", ["rigid", "affine"])
@pytest.mark.parametrize("metric", ["local_cc", "global_ncc"])
def test_real_fit_preserves_seed_grid_and_pairs_metrics(
    experiment, volume, device, model, metric
):
    rotation = sitk.Euler3DTransform()
    rotation.SetRotation(0.04, -0.03, 0.02)
    seed = np.eye(4)
    seed[:3, :3] = np.array(rotation.GetMatrix()).reshape(3, 3)
    if model == "affine":
        seed[:3, :3] = seed[:3, :3] @ np.diag([1.0, 1.04, 0.97])
    seed[:3, 3] = [0.01, -0.02, 0.03]
    row, states = experiment.fit_fireants(
        volume, volume, seed, model=model, metric=metric, device=device, budget=2
    )
    assert len(row["raw_metric"]) == len(states) == 3
    assert row["initial_grid_max_normalized_error"] < 1e-4
    assert_allclose(row["initial_pull"], seed, atol=2e-6, rtol=0)
    assert_allclose(row["best_pull"], states[row["best_evaluation"]], atol=0, rtol=0)
    if model == "rigid":
        assert_allclose(
            np.linalg.svd(states[:, :3, :3], compute_uv=False), 1, atol=1e-5, rtol=0
        )
