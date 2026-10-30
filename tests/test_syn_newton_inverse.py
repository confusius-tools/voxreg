"""Independent two-sided physical checks for the bounded inverse experiment."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from oracles import compose_displacements_reference

from voxreg.fields import make_field_grid


@pytest.mark.parametrize("planar", [False, True])
def test_newton_inverse_preserves_physical_residuals(
    displacement_factory, device, monkeypatch, planar
):
    directory = Path(__file__).resolve().parents[1] / "research/syn"
    monkeypatch.syspath_prepend(str(directory))
    path = directory / ("newton_inverse.py" if planar else "newton3d.py")
    spec = importlib.util.spec_from_file_location("newton_inverse_research", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data, _, lattice, values = displacement_factory(planar)
    grid = make_field_grid(data, device=device)
    field = torch.tensor(values, device=device)
    with torch.no_grad():
        inverse, error = module.final_inverse(field, grid)
    assert inverse.dtype == torch.float64
    assert_array_equal(field.cpu(), values)
    result = inverse.cpu().numpy()
    left = compose_displacements_reference(values, result, lattice)
    right = compose_displacements_reference(result, values, lattice)

    def maximum(residual):
        pixels = np.einsum("ij,bj...->bi...", np.linalg.inv(lattice), residual)
        return np.linalg.norm(pixels, axis=1).reshape(len(values), -1).max(1)

    assert maximum(left).max() < 1e-10
    assert_allclose(
        error.cpu(), np.maximum(maximum(left), maximum(right)), rtol=0, atol=1e-13
    )
    if device == "cuda":
        assert module.COUNTS["accepted"] == 1 and module.COUNTS["fallback"] == 0
