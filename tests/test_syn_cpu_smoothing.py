"""Independent range, boundary and ownership checks for CPU filter experiments."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from voxreg.fields import make_field_grid, smooth_field


def smoother(mode):
    if mode == "production":
        from voxreg import syn

        return syn.smooth_field
    path = Path(__file__).resolve().parents[1] / "research/syn/cpu_smoothing.py"
    spec = importlib.util.spec_from_file_location("cpu_smoothing_research", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CPUSmoother(mode)


@pytest.mark.parametrize("mode", ["torch64", "gradient32", "all32", "production"])
@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("sigma", [0.5, 1.5])
@pytest.mark.parametrize("scale", [0, 1, 1e-200, 1e200])
def test_cpu_filter_range_and_boundaries(
    displacement_factory, monkeypatch, mode, planar, sigma, scale
):
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    data, _, _, values = displacement_factory(planar)
    grid = make_field_grid(data)
    source = values * scale
    field = torch.tensor(source)
    actual = smoother(mode)(field, grid, sigma=sigma)
    expected = gaussian_filter(
        values if scale else np.zeros_like(values),
        sigma=sigma,
        radius=int(3 * sigma + 0.5),
        axes=tuple(range(2, values.ndim)),
        mode="nearest",
    )
    expected = np.where(grid.boundary.numpy(), 0, expected)
    low_precision = (
        mode == "all32"
        or (mode == "gradient32" and sigma == 1.5)
        or (mode == "production" and not planar)
    )
    tolerance = np.abs(values).max() * (2e-6 if low_precision else 2e-14)
    np.testing.assert_allclose(
        actual.numpy() / (scale or 1), expected, rtol=0, atol=tolerance
    )
    np.testing.assert_array_equal(field, source)
    assert actual.dtype == torch.float64 and torch.isfinite(actual).all()
    assert not actual.masked_select(grid.boundary).any()


@pytest.mark.parametrize("mode", ["torch64", "gradient32", "all32", "production"])
def test_cpu_filter_preserves_bypass_and_autograd(displacement_factory, mode):
    data, _, _, values = displacement_factory(False)
    grid = make_field_grid(data)
    field = torch.tensor(values, requires_grad=True)
    function = smoother(mode)
    assert function(field, grid, sigma=0) is field
    actual = function(field, grid, sigma=0.5)
    expected = smooth_field(field, grid, sigma=0.5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.autograd.grad(actual.square().sum(), field)[0]
    reference = torch.autograd.grad(expected.square().sum(), field)[0]
    torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
    for sigma in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="sigma"):
            function(field, grid, sigma=sigma)


@pytest.mark.parametrize(
    "reason", ["planar", "serial", "autograd", "compiled", "sigma"]
)
def test_syn_smoothing_preserves_unqualified_paths(
    displacement_factory, monkeypatch, reason
):
    from voxreg import syn

    data, _, _, values = displacement_factory(reason == "planar")
    grid = make_field_grid(data)
    field = torch.tensor(values, requires_grad=reason == "autograd")
    monkeypatch.setattr(
        torch, "get_num_threads", lambda: 1 if reason == "serial" else 8
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: reason == "compiled")
    calls = []

    def original(value, geometry, *, sigma):
        calls.append((value, geometry, sigma))
        return value

    monkeypatch.setattr(syn, "_smooth_field", original)
    sigma = 1.0 if reason == "sigma" else 0.5
    assert syn.smooth_field(field, grid, sigma=sigma) is field
    assert len(calls) == 1 and calls[0][0] is field and calls[0][1] is grid
    assert calls[0][2] == sigma
