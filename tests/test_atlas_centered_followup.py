"""Matched raw state/metric observation and genuinely model-preserving step probes."""

import importlib
from pathlib import Path

import confusius as cf
import numpy as np
import pytest
import torch


@pytest.fixture
def followup(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "research/confusius-examples")
    )
    return importlib.import_module("atlas_centered_followup")


@pytest.fixture
def pair():
    values = np.random.default_rng(43).uniform(1, 2, (5, 7, 9)).astype(np.float32)
    volume = cf.create_voxeldata(
        values, dims=("k", "j", "i"), voxel_to_world=np.diag([0.1, 0.1, 0.1, 1.0])
    )
    affine = np.diag([0.4, 0.1, 0.1, 1.0])
    affine[0, 3] = 0.2
    plane = cf.create_voxeldata(
        values[2:3], dims=("k", "j", "i"), voxel_to_world=affine
    )
    return volume, plane


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
@pytest.mark.parametrize("moving_is_slice", [False, True])
def test_vox_capture_preserves_seed_pivot_and_best_return(
    followup, pair, model, moving_is_slice, device
):
    volume, plane = pair
    source, target = (plane, volume) if moving_is_slice else (volume, plane)
    row, states = followup.vox_trace(
        source, target, np.eye(4), model=model, device=device, budget=3, window=4
    )
    assert row["optimized_parameters"] == (6 if model == "rigid" else 9)
    assert row["n_iterations"] == 3
    assert len(states) == len(row["raw_metric"]) == len(row["clocks"]) == 4
    np.testing.assert_allclose(
        row["center_zyx_mm"], [0.2, 0.3, 0.4], atol=1e-15, rtol=0
    )
    np.testing.assert_array_equal(row["decoded_seed"], np.eye(4))
    assert row["returned_metric"] <= row["raw_final_metric"] + 1e-12


def test_unsupported_display_is_gray_not_nearest_extrapolation(followup, pair):
    import matplotlib.pyplot as plt

    volume, plane = pair
    matrix = np.eye(4)
    matrix[0, 3] = 100.0
    values, valid, _ = followup.display_sample(matrix, plane, volume)
    assert not valid.any() and np.isnan(values).all()
    assert followup.metric_support(volume, plane, matrix) == 0.0
    followup.supported_composites(
        plane.values, {"Unsupported": values}, "Support check"
    )
    image = plt.gcf().axes[0].images[0].get_array()
    assert image is not None
    assert not np.ma.getmaskarray(image).any()
    np.testing.assert_allclose(np.asarray(image), 0.25, rtol=0, atol=0)
    plt.close(plt.gcf())


@pytest.mark.parametrize("model", ["rigid", "unit_normal_affine"])
def test_probe_uses_parameter_path_not_matrix_chord(followup, pair, monkeypatch, model):
    volume, _ = pair
    builder = (
        followup.centered_rigid if model == "rigid" else followup.centered_unit_normal
    )
    transform = builder(np.eye(4), [0.0, 0.0, 0.0])
    p0 = np.array(transform.GetParameters())
    p1 = p0.copy()
    p1[:3] = [0.2, -0.12, 0.05]
    transform.SetParameters(p1)
    end = followup.linear_matrix(transform)
    seen = []

    def score(_source, _target, matrix, **kwargs):
        seen.append(matrix.copy())
        return 0.5

    monkeypatch.setattr(followup, "score", score)
    row = {
        "model": model,
        "parameter_states": [p0.tolist(), p1.tolist()],
        "center_fixed_world_zyx_mm": [0.0, 0.0, 0.0],
        "raw_metric": [-0.5, -0.5],
    }
    probe = followup.step_probe(volume, volume, row, 0)
    assert len(probe["fractions"]) == len(seen) == 8
    for matrix in seen:
        np.testing.assert_allclose(
            np.linalg.norm(matrix[:3, 0]), 1.0, atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            matrix[:3, 0] @ matrix[:3, 1:3], [0, 0], atol=1e-12, rtol=0
        )
    assert not np.allclose(seen[-2], 0.7 * np.eye(4) + 0.3 * end)
