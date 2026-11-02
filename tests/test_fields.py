"""Physical SyN field primitives against analytic and independent SciPy oracles."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from oracles import compose_displacements_reference, itk_displacement_reference, voxel
from scipy.ndimage import gaussian_filter

from voxreg.fields import (
    compose_fields,
    invert_field,
    jacobian_determinant,
    make_field_grid,
    smooth_field,
)


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_physical_composition_matches_scipy(
    displacement_factory, device, planar, strided
):
    data, basis, lattice, values = displacement_factory(planar, strided=strided)
    grid = make_field_grid(data, device=device)
    assert_allclose(grid.basis.cpu(), basis, rtol=0, atol=1e-15)
    assert_allclose(grid.lattice.cpu(), lattice, rtol=0, atol=1e-15)
    outer = torch.tensor(values, device=device)
    inner = torch.tensor(values[:, ::-1].copy() * -0.7, device=device)
    original = outer.clone()
    actual = compose_fields(outer, inner, grid)
    expected = compose_displacements_reference(values, inner.cpu().numpy(), lattice)
    assert_allclose(actual.cpu(), expected, rtol=0, atol=1e-14)
    assert_array_equal(outer.cpu(), original.cpu())
    assert not np.allclose(actual.cpu(), (outer + inner).cpu(), rtol=0, atol=1e-8)
    # Native ITK also checks physical directions and the noncommuting pull order.
    import SimpleITK as sitk

    picks = (
        np.array(
            np.meshgrid(
                *[[0, n // 3, n // 2, n - 2, n - 1] for n in grid.shape], indexing="ij"
            )
        )
        .reshape(len(grid.shape), -1)
        .T
    )
    points = picks @ lattice.T
    rows = np.arange(len(grid.shape))[::-1] if not planar else np.arange(2)
    for frame in range(len(values)):
        itk = sitk.CompositeTransform(
            [
                itk_displacement_reference(values[frame], lattice),
                itk_displacement_reference(inner[frame].cpu().numpy(), lattice),
            ]
        )
        moved = np.array([itk.TransformPoint(p[rows].tolist()) for p in points])[
            :, rows
        ]
        expected_points = (
            points + actual[frame].cpu().numpy()[(slice(None), *picks.T)].T
        )
        assert_allclose(moved, expected_points, rtol=0, atol=1e-13)


@pytest.mark.parametrize("planar", [False, True])
def test_composition_has_finite_half_voxel_support(
    displacement_factory, device, planar
):
    data, _, _, _ = displacement_factory(planar)
    data = voxel(data.values, np.diag([2.0, 4.0, 8.0, 1.0]))
    grid = make_field_grid(data, device=device)
    d = len(grid.shape)
    outer = torch.ones((2, d, *grid.shape), dtype=torch.float64, device=device)
    pixel_shift = torch.full((2, d), 0.5, dtype=torch.float64, device=device)
    pixel_shift[1] *= -1
    shift = pixel_shift @ grid.lattice.T
    inner = shift.reshape(2, d, *([1] * d)).expand_as(outer)
    actual = compose_fields(outer, inner, grid)
    expected = compose_displacements_reference(
        outer.cpu().numpy(), inner.cpu().numpy(), grid.lattice.cpu().numpy()
    )
    assert_allclose(actual.cpu(), expected, rtol=0, atol=1e-14)
    assert_allclose(
        actual[(0, slice(None), *([-1] * d))].cpu(), shift[0].cpu(), rtol=0, atol=0
    )
    assert_allclose(
        actual[(1, slice(None), *([0] * d))].cpu(), shift[1].cpu() + 1, rtol=0, atol=0
    )


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_physical_jacobian_matches_affine_and_detects_reflection(
    displacement_factory, device, planar, strided
):
    data, _, lattice, _ = displacement_factory(planar, strided=strided)
    grid = make_field_grid(data, device=device)
    d = len(grid.shape)
    linear = np.eye(d)
    linear[0, 0] = 1.04
    linear[1, 1] = 0.97
    linear[0, 1] = 0.025
    linear[1, 0] = -0.015
    points = np.einsum("ij,j...->i...", lattice, np.indices(grid.shape))
    for sign in [1, -1]:
        matrix = linear.copy()
        matrix[0] *= sign
        field = np.einsum("ij,j...->i...", matrix - np.eye(d), points)[None]
        actual = jacobian_determinant(torch.tensor(field, device=device), grid)
        assert_allclose(actual.cpu(), np.linalg.det(matrix), rtol=0, atol=2e-14)
        assert bool((actual > 0).all()) == (sign == 1)


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_inverse_is_two_sided_and_batch_independent(
    displacement_factory, device, planar, strided
):
    data, _, lattice, values = displacement_factory(planar, strided=strided)
    grid = make_field_grid(data, device=device)
    field = torch.tensor(values, device=device)
    inverse, error = invert_field(field, grid)
    result = inverse.cpu().numpy()
    left = compose_displacements_reference(values, result, lattice)
    right = compose_displacements_reference(result, values, lattice)

    def maximum_pixels(residual):
        pixel = np.einsum("ij,bj...->bi...", np.linalg.inv(lattice), residual)
        return np.linalg.norm(pixel, axis=1).reshape(len(values), -1).max(axis=1)

    expected = np.maximum(maximum_pixels(left), maximum_pixels(right))
    assert_allclose(error.cpu(), expected, rtol=0, atol=1e-14)
    # Solve the forward equation; reverse interpolation error is reported, not hidden.
    assert maximum_pixels(left).max() < 0.001
    assert (jacobian_determinant(field, grid) > 0).all()
    assert (jacobian_determinant(inverse, grid) > 0).all()
    assert not inverse.masked_select(grid.boundary).any()
    for frame in range(len(values)):
        alone, _ = invert_field(field[frame : frame + 1], grid)
        assert_allclose(alone.cpu(), result[frame : frame + 1], rtol=0, atol=1e-14)
    resumed, resumed_error = invert_field(field, grid, initial=inverse, iterations=5)
    assert torch.isfinite(resumed_error).all()
    improved = compose_displacements_reference(values, resumed.cpu().numpy(), lattice)
    assert (maximum_pixels(improved) < maximum_pixels(left)).all()
    if device == "cuda":
        cpu, _ = invert_field(torch.tensor(values), make_field_grid(data))
        delta = inverse.cpu() - cpu
        ell = np.linalg.norm(lattice, axis=0).min()
        assert torch.linalg.vector_norm(delta, dim=1).max() / ell < 0.001


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("sigma", [0, 0.9])
def test_smoothing_matches_scipy_and_fixes_active_boundaries(
    displacement_factory, device, planar, sigma
):
    data, _, _, values = displacement_factory(planar)
    grid = make_field_grid(data, device=device)
    values = values + 0.1
    field = torch.tensor(values, device=device)
    original = field.clone()
    actual = smooth_field(field, grid, sigma=sigma)
    expected = np.array(
        [
            [gaussian_filter(c, sigma, mode="nearest", truncate=3) for c in b]
            for b in values
        ]
    )
    if sigma:
        expected = np.where(grid.boundary.cpu().numpy(), 0, expected)
    assert_allclose(actual.cpu(), expected, rtol=0, atol=1e-14)
    assert_array_equal(field.cpu(), original.cpu())
    if planar:
        assert actual.shape[-2:] == data.shape[1:]


@pytest.mark.parametrize("planar", [False, True])
def test_field_composition_gradients(displacement_factory, device, planar):
    data, _, _, _ = displacement_factory(planar)
    data = data.isel(j=slice(None, None, 4), i=slice(None, None, 4))
    if not planar:
        data = data.isel(k=slice(None, None, 4))
    grid = make_field_grid(data, device=device)
    d = len(grid.shape)
    rng = np.random.default_rng(4)
    outer = torch.tensor(
        rng.normal(0, 0.01, (1, d, *grid.shape)), device=device, requires_grad=True
    )
    offset = grid.lattice @ torch.full((d,), 0.23, device=device, dtype=torch.float64)
    inner = (
        offset.reshape(1, d, *([1] * d)).expand_as(outer).clone().requires_grad_(True)
    )
    # CUDA resampling/replicate-padding backward use atomic adds: roundoff only.
    nondet_tol = 32 * torch.finfo(torch.float64).eps if device == "cuda" else 0
    assert torch.autograd.gradcheck(
        lambda a: smooth_field(a, grid, sigma=0.9),
        (outer,),
        fast_mode=True,
        nondet_tol=nondet_tol,
    )
    assert torch.autograd.gradcheck(
        lambda a, b: compose_fields(a, b, grid),
        (outer, inner),
        fast_mode=True,
        eps=1e-6,
        atol=2e-6,
        rtol=2e-5,
        nondet_tol=nondet_tol,
    )


@pytest.mark.parametrize("planar", [False, True])
def test_identity_inverse_does_not_couple_to_other_frames(
    displacement_factory, device, planar
):
    data, _, _, values = displacement_factory(planar)
    values[0] = 0
    grid = make_field_grid(data, device=device)
    field = torch.tensor(values, device=device)
    inverse, error = invert_field(field, grid)
    assert torch.isfinite(inverse).all() and torch.isfinite(error).all()
    assert not inverse[0].any() and error[0] == 0
    assert inverse[1].any() and error[1] > 0
    assert_array_equal(jacobian_determinant(inverse[:1], grid).cpu(), 1)


@pytest.mark.parametrize("planar", [False, True])
def test_cpu_inverse_certificate_preserves_residual_checks(
    displacement_factory, monkeypatch, planar
):
    from voxreg import fields
    from voxreg.syn import _final_inverse

    data, _, _, values = displacement_factory(planar)
    grid = make_field_grid(data)
    field = torch.tensor(values * 0.01)
    expected, _ = invert_field(
        field, grid, iterations=80, initial=torch.zeros_like(field)
    )
    original = fields._inverse_update
    calls = []

    def observed(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(fields, "_inverse_update", observed)
    inverse, error = _final_inverse(field, grid)
    assert not calls
    pixels = np.einsum(
        "ij,bj...->bi...", grid.inverse_lattice.numpy(), (inverse - expected).numpy()
    )
    assert np.linalg.norm(pixels, axis=1).max() < 1e-9
    left = (
        fields._pixel_norm(compose_fields(field, inverse, grid), grid)
        .flatten(1)
        .amax(1)
    )
    right = (
        fields._pixel_norm(compose_fields(inverse, field, grid), grid)
        .flatten(1)
        .amax(1)
    )
    assert_array_equal(error, torch.maximum(left, right))
    assert (left < 1e-10).all()


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("reason", ["boundary", "gradient", "noncontractive"])
def test_cpu_inverse_preserves_original_fallback(
    displacement_factory, monkeypatch, planar, reason
):
    from voxreg import fields
    from voxreg.syn import _final_inverse

    data, _, _, values = displacement_factory(planar, strided=True)
    grid = make_field_grid(data)
    field = torch.tensor(values * (100 if reason == "noncontractive" else 0.01))
    if reason == "boundary":
        field[:, :, 0] = 0.001
    if reason == "gradient":
        field.requires_grad_()
    original = fields._inverse_update
    calls = []

    def observed(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(fields, "_inverse_update", observed)
    inverse, error = _final_inverse(field, grid)
    assert len(calls) == 80
    assert torch.isfinite(inverse).all() and torch.isfinite(error).all()
    if reason == "gradient":
        inverse.sum().backward()
        assert field.grad is not None
        assert torch.isfinite(field.grad).all()


@pytest.mark.parametrize("planar", [False, True])
def test_field_geometry_ignores_absolute_world_origin(
    displacement_factory, device, planar
):
    data, _, _, values = displacement_factory(planar)
    from voxreg import position_to_world

    affine = position_to_world(data)
    affine[:3, 3] += [1e12, -2e12, 4e12]
    shifted = voxel(data.values.copy(), affine)
    original_grid = make_field_grid(data, device=device)
    shifted_grid = make_field_grid(shifted, device=device)
    field = torch.tensor(values, device=device)
    assert_array_equal(
        compose_fields(field, -field, original_grid).cpu(),
        compose_fields(field, -field, shifted_grid).cpu(),
    )


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("field_broadcast", [False, True])
def test_tiny_geometry_maps_match_numpy(
    displacement_factory, device, planar, broadcast, field_broadcast
):
    from voxreg.sampling import _map_displacement

    _, _, lattice, values = displacement_factory(planar)
    matrix = np.linalg.inv(lattice)
    if planar:
        matrix = np.vstack([matrix, matrix.sum(axis=0) * 0.3])
    matrices = np.stack([matrix * (1 + 0.1 * n) for n in range(len(values))])
    if broadcast:
        matrices = matrices[:1]
    if field_broadcast:
        values = values[:1]
    # A noncontiguous field must not change the component/batch convention.
    field = torch.tensor(values, device=device).transpose(-1, -2)
    expected = np.einsum("boc,bc...->b...o", matrices, values.swapaxes(-1, -2))
    actual = _map_displacement(torch.tensor(matrices, device=device), field)
    assert_allclose(actual.cpu(), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("planar", [False, True])
def test_prepared_seed_queries_preserve_samples_and_gradients(
    nonlinear_pair_factory, displacement_factory, device, planar
):
    from voxreg.sampling import (
        _prepare_displaced_sampling,
        _sample_prepared,
        make_grid,
        sample_rigid,
    )

    _, _, _, values = displacement_factory(planar)
    fixed, moving, *_ = nonlinear_pair_factory(planar, shape=values.shape[2:])
    raw = moving.values[0] if planar else moving.values
    images = torch.tensor(np.repeat(raw[None], len(values), axis=0), device=device)
    from voxreg.transforms import configure_transform

    geometry = configure_transform(
        make_grid(fixed, moving, device=device), fixed, "affine"
    )
    parameters = torch.full(
        (len(values), len(geometry.parameter_scale)),
        0.003,
        dtype=torch.float64,
        device=device,
    )
    prepared = _prepare_displaced_sampling(images, parameters, geometry)
    original = [tensor.clone() for tensor in prepared[:2]]
    for scale in (-1, 0.3, 1.5):
        field = torch.tensor(values * scale, device=device, requires_grad=True)
        cached, valid = _sample_prepared(images, field, prepared)
        direct, direct_valid = sample_rigid(
            images, parameters, geometry, displacement=field
        )
        assert_array_equal(cached.detach().cpu(), direct.detach().cpu())
        assert_array_equal(valid.cpu(), direct_valid.cpu())
        cg = torch.autograd.grad(cached.sum(), field)[0]
        dg = torch.autograd.grad(direct.sum(), field)[0]
        assert_array_equal(cg.cpu(), dg.cpu())
    for actual, expected in zip(prepared[:2], original, strict=True):
        assert_array_equal(actual.cpu(), expected.cpu())


@pytest.mark.parametrize("planar", [False, True])
def test_exact_sparse_lattice_keeps_general_field_math(
    displacement_factory, device, planar
):
    from dataclasses import replace

    from voxreg import position_to_world
    from voxreg.fields import _pixel_norm

    data, _, _, values = displacement_factory(planar)
    affine = position_to_world(data)
    affine[:3, :3] = np.diag([0.24, -0.21, 0.18])
    grid = make_field_grid(voxel(data.values.copy(), affine), device=device)
    assert grid.voxel_axes is not None
    general = replace(grid, voxel_axes=None, voxel_scale=None)
    field = torch.tensor(values, device=device, requires_grad=True)
    expected_norm = np.linalg.norm(
        np.einsum("ij,bj...->bi...", grid.inverse_lattice.cpu(), values), axis=1
    )
    assert_allclose(
        _pixel_norm(field, grid).detach().cpu(), expected_norm, rtol=1e-14, atol=1e-14
    )
    fast = compose_fields(field, field * -0.7, grid)
    slow = compose_fields(field, field * -0.7, general)
    expected = compose_displacements_reference(
        values, values * -0.7, grid.lattice.cpu().numpy()
    )
    assert_allclose(fast.detach().cpu(), expected, rtol=0, atol=1e-14)
    assert_allclose(fast.detach().cpu(), slow.detach().cpu(), rtol=0, atol=1e-14)
    assert_allclose(
        jacobian_determinant(field, grid).detach().cpu(),
        jacobian_determinant(field, general).detach().cpu(),
        rtol=1e-14,
        atol=1e-14,
    )
    fg = torch.autograd.grad(fast.sum(), field)[0]
    sg = torch.autograd.grad(slow.sum(), field)[0]
    assert_allclose(fg.cpu(), sg.cpu(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("planar", [False, True])
def test_smoothing_keeps_differentiable_tensor_path(
    displacement_factory, device, planar
):
    data, _, _, values = displacement_factory(planar)
    grid = make_field_grid(data, device=device)
    base = torch.tensor(values, device=device)
    theta = torch.tensor(1.0, device=device, requires_grad=True, dtype=torch.float64)
    actual = smooth_field(base * theta, grid, sigma=1.5).sum()
    derivative = torch.autograd.grad(actual, theta)[0]
    expected = smooth_field(base, grid, sigma=1.5).sum()
    assert_allclose(derivative.detach().cpu(), expected.cpu(), rtol=1e-12, atol=1e-12)


def test_tiny_geometry_map_gradients(displacement_factory):
    from voxreg.sampling import _map_displacement

    _, _, lattice, values = displacement_factory(True)
    matrix = torch.tensor(np.linalg.inv(lattice)[None], requires_grad=True)
    field = torch.tensor(values[..., :2, :3], requires_grad=True)
    assert torch.autograd.gradcheck(_map_displacement, (matrix, field))


def test_field_options_and_geometry_validation(displacement_factory):
    data, _, _, values = displacement_factory(True)
    grid = make_field_grid(data)
    field = torch.tensor(values)
    for sigma in [-1, np.nan]:
        with pytest.raises(ValueError, match="sigma"):
            smooth_field(field, grid, sigma=sigma)
    for iterations in [0, True, 1.5]:
        with pytest.raises(ValueError, match="iterations"):
            invert_field(field, grid, iterations=iterations)  # ty: ignore[invalid-argument-type] -- deliberate invalid-input coverage.
    with pytest.raises(ValueError, match="shape"):
        compose_fields(field[:1], field, grid)
    with pytest.raises(ValueError, match="float64"):
        compose_fields(field.float(), field.float(), grid)
    with pytest.raises(ValueError, match="axes|sizes"):
        make_field_grid(data.isel(j=slice(0, 1)))
