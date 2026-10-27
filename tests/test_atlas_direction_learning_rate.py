"""Metric histories must be paired with evaluated poses, not post-step positions."""

import importlib
from pathlib import Path

import confusius as cf
import numpy as np
import pytest
import SimpleITK as sitk


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "research/confusius-examples")
    )
    return importlib.import_module("atlas_direction_learning_rate")


def test_metric_pose_pairing_and_scoped_observer(diagnostic):
    z, y, x = np.indices((6, 8, 9), dtype=float)

    def values(offset):
        return (
            np.sin(0.37 * (x + offset))
            + 0.3 * np.cos(0.41 * y)
            + np.exp(-((z - 2.1) ** 2 + (y - 3.2) ** 2) / 12)
        )

    source = cf.create_voxeldata(
        values(0.15), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    )
    target = source.copy(data=values(0))
    original = sitk.ImageRegistrationMethod
    result, states = diagnostic.fit_curve(source, target, np.eye(4), learning_rate=0.1)
    assert sitk.ImageRegistrationMethod is original
    assert len(states) == len(result["raw_metric"])
    assert result["event_pairing_verified"]
    assert not np.array_equal(states[0], states[1])
    assert result["best_metric"] == min(result["raw_metric"])
    probe = diagnostic.first_step_probe(source, target, states)
    assert len(probe) == 7
    assert probe[0]["valid_points"] == source.size
    assert abs(probe[0]["objective"] - result["raw_metric"][0]) < 1e-10


def test_constructor_is_restored_if_registration_fails(diagnostic, monkeypatch):
    original = sitk.ImageRegistrationMethod

    def fail(*args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(cf.registration, "register_volume", fail)
    with pytest.raises(RuntimeError, match="registration failed"):
        diagnostic.fit_curve(None, None, np.eye(4))
    assert sitk.ImageRegistrationMethod is original
