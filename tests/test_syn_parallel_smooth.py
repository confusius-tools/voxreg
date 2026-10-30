"""Independent checks for the research-only threaded CPU Gaussian candidate."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from voxreg.fields import make_field_grid, smooth_field


def candidate():
    path = Path(__file__).resolve().parents[1] / "research/syn/parallel_smooth.py"
    spec = importlib.util.spec_from_file_location("parallel_smooth_research", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.threaded_smooth


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("sigma", [0, 0.5, 1.5])
def test_threaded_smoothing_matches_independent_filter(
    displacement_factory, monkeypatch, planar, sigma
):
    data, _, _, values = displacement_factory(planar)
    grid = make_field_grid(data)
    field = torch.tensor(values)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    expected = values.copy()
    if sigma:
        expected = gaussian_filter(
            expected,
            sigma=sigma,
            radius=int(3 * sigma + 0.5),
            axes=tuple(range(2, values.ndim)),
            mode="nearest",
        )
        expected = np.where(grid.boundary.numpy(), 0, expected)
    actual = candidate()(field, grid, sigma=sigma)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(field, values)
    assert actual.dtype == torch.float64


def test_threaded_smoothing_retains_autograd(displacement_factory, monkeypatch):
    data, _, _, values = displacement_factory(False)
    grid = make_field_grid(data)
    field = torch.tensor(values, requires_grad=True)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    actual = candidate()(field, grid, sigma=0.5)
    expected = smooth_field(field, grid, sigma=0.5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.autograd.grad(actual.square().sum(), field)[0]
    reference = torch.autograd.grad(expected.square().sum(), field)[0]
    torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
