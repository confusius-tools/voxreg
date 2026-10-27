"""Different supplied directions must describe the same physical seed."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture
def directions(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "research/confusius-examples")
    )
    return importlib.import_module("atlas_directions")


@pytest.mark.parametrize("moving_is_slice", [True, False])
def test_direction_normalization_preserves_template_to_recording_pull(
    directions, moving_is_slice
):
    expected = np.array(
        [
            [1.0, 0.2, 0.0, -5.0],
            [0.0, 0.9, 0.1, 2.0],
            [0.1, 0.0, 1.1, -6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    returned = expected.copy() if moving_is_slice else np.linalg.inv(expected)
    actual = directions.canonical_pull(returned, moving_is_slice=moving_is_slice)
    points = np.array([[1.0, 3.0], [2.0, -4.0], [5.0, 2.0], [1.0, 1.0]])
    assert_allclose(actual @ points, expected @ points, atol=1e-14, rtol=0)


def test_canonical_matrix_does_not_alias_backend_output(directions):
    returned = np.eye(4)
    actual = directions.canonical_pull(returned, moving_is_slice=True)
    actual[0, 3] = 42
    assert_array_equal(returned, np.eye(4))
