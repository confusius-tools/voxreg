"""Independent double-precision controls for norm reduction and box sums."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import voxel
from numpy.testing import assert_allclose

from voxreg.fields import make_field_grid


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("scale", [0.0, 1e-100, 1.0, 1e100])
def test_maximum_squared_norm_matches_numpy(
    displacement_factory, device, planar, sparse, scale
):
    path = Path(__file__).resolve().parents[1] / "research/syn/maxnorm.py"
    spec = importlib.util.spec_from_file_location("maximum_norm_research", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data, _, _, values = displacement_factory(planar)
    if sparse:
        data = voxel(data.values, np.diag([0.31, 0.17, 0.23, 1.0]))
    grid = make_field_grid(data, device=device)
    field = torch.tensor(values * scale, device=device)
    voxel_field = np.einsum(
        "ij,bj...->bi...", grid.inverse_lattice.cpu().numpy(), field.cpu().numpy()
    )
    expected = (
        np.sqrt(np.square(voxel_field).sum(axis=1)).reshape(len(values), -1).max(axis=1)
    )
    actual = module.maximum_norm(field, grid)
    assert actual.dtype == torch.float64
    assert_allclose(actual.cpu(), expected, rtol=3e-15, atol=0)


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("radius", [0, 2])
def test_native_box_sum_and_adjoint_match_scipy(device, dimensions, radius):
    from scipy.ndimage import correlate

    path = Path(__file__).resolve().parents[1] / "research/syn/box_filter.py"
    spec = importlib.util.spec_from_file_location("box_filter_research", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shape = (2, 3, *([7, 9, 11][-dimensions:]))
    rng = np.random.default_rng(982)
    values = rng.normal(size=shape)
    weights = rng.normal(size=shape)
    kernel = np.ones((1, 1, *([2 * radius + 1] * dimensions)))
    expected = correlate(values, kernel, mode="constant", cval=0)
    expected_gradient = correlate(weights, kernel, mode="constant", cval=0)
    tensor = torch.tensor(values, device=device, requires_grad=True)
    output = module.box_sum(tensor, radius)
    gradient = torch.autograd.grad(
        (output * torch.tensor(weights, device=device)).sum(), tensor
    )[0]
    assert output.dtype == torch.float64
    assert_allclose(output.detach().cpu(), expected, rtol=2e-13, atol=1e-13)
    assert_allclose(gradient.cpu(), expected_gradient, rtol=2e-13, atol=1e-13)
