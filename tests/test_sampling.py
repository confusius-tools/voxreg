"""Independent interpolation, physical mappings, NCC and autodiff checks."""

import numpy as np
import pytest
import torch
from confusius import create_voxeldata
from numpy.testing import assert_allclose, assert_array_equal
from scipy.ndimage import map_coordinates

from voxreg import position_to_world
from voxreg.sampling import (
    build_world_matrices,
    compute_global_ncc,
    make_plane,
    sample_rigid,
)


@pytest.mark.parametrize("planar", [False, True])
def test_fitting_sampler_values_and_right_knot_derivatives(
    nonlinear_pair_factory, device, planar
):
    from voxreg.sampling import _sample_displaced

    shape = (5, 6) if planar else (4, 5, 6)
    _, moving, *_ = nonlinear_pair_factory(planar, shape=shape, amplitude=0.1)
    image = moving.values[0] if planar else moving.values
    dimensions = len(shape)
    knot = np.arange(1, dimensions + 1, dtype=float)
    points = np.stack(
        (
            knot,
            knot + np.linspace(0.2, 0.4, dimensions),
            np.zeros(dimensions),
            np.array(shape) + 0.2,
        )
    )
    half = (np.array(shape) - 1) / 2
    normalized = (points / half - 1)[:, ::-1].copy()
    query_shape = (1, *([1] * (dimensions - 1)), len(points), dimensions)
    query = torch.tensor(
        normalized.reshape(query_shape), device=device, requires_grad=True
    )
    actual = _sample_displaced(
        torch.tensor(image[None], device=device), query, "linear"
    )
    expected = map_coordinates(
        image.astype(float), points.T, order=1, mode="nearest", prefilter=False
    )
    assert_allclose(actual.detach().cpu().reshape(-1), expected, rtol=2e-7, atol=1e-7)
    derivative = (
        torch.autograd.grad(actual.sum(), query)[0]
        .cpu()
        .numpy()
        .reshape(len(points), dimensions)
    )
    expected_derivative = np.zeros_like(points)
    step = 1e-6
    for axis in range(dimensions):
        changed = points.copy()
        changed[:2, axis] += step
        changed_values = map_coordinates(
            image.astype(float), changed.T, order=1, mode="nearest", prefilter=False
        )
        expected_derivative[:2, dimensions - axis - 1] = (
            (changed_values[:2] - expected[:2]) / step * half[axis]
        )
    assert_allclose(derivative, expected_derivative, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("sheared", [False, True])
def test_physical_sampler_against_scipy(recording_factory, strided, sheared):
    data, _, _ = recording_factory(oblique=True, strided=strided)
    if sheared:
        affine = data.xindexes["z"].voxel_to_world.copy()
        affine[:3, 1] += 0.35 * affine[:3, 2]
        data = create_voxeldata(
            data.values,
            dims=data.dims,
            **{dim: data[dim] for dim in data.dims},
            voxel_to_world=affine,
        )
    affine = position_to_world(data)
    fixed = data.isel(time=0, drop=True)
    plane = make_plane(fixed, dtype=torch.float64)
    parameters = torch.tensor(
        [[0, 0, 0], [0.019, 0.15, -0.12], [-0.13, -0.21, 0.17]], dtype=torch.float64
    )
    matrices = build_world_matrices(parameters.numpy(), plane)
    j, i = np.indices(data.shape[-2:])
    points = np.stack((np.zeros_like(i), j, i, np.ones_like(i)))
    mapped = np.einsum(
        "tab,bhw->tahw", np.linalg.inv(affine) @ matrices @ affine, points
    )
    h, w = j.shape
    valid = (
        (mapped[:, 1] >= -1e-5)
        & (mapped[:, 1] <= h - 1 + 1e-5)
        & (mapped[:, 2] >= -1e-5)
        & (mapped[:, 2] <= w - 1 + 1e-5)
    )
    expected = np.stack(
        [
            map_coordinates(
                data.values[0, 0].astype(float),
                coordinates[1:3],
                order=1,
                mode="nearest",
                prefilter=False,
            )
            for coordinates in mapped
        ]
    )
    images = np.repeat(data.values[0], len(parameters), axis=0)
    actual, actual_valid = sample_rigid(
        torch.tensor(images, dtype=torch.float64), parameters, plane
    )
    assert_allclose(actual, expected, atol=1e-12)
    assert_array_equal(actual_valid, valid)
    for matrix, parameter in zip(matrices, parameters.numpy()):
        assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-14)
        assert_allclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-14)
        assert_allclose(np.linalg.inv(matrix) @ matrix, np.eye(4), atol=1e-13)
        center = np.r_[plane.center, 1]
        assert_allclose(
            (matrix @ center - center)[:3], plane.basis @ parameter[1:], atol=1e-13
        )


def test_identity_boundaries_and_metric(recording_factory):
    data, _, _ = recording_factory(oblique=True, strided=True)
    plane = make_plane(data)
    images = torch.tensor(data.values[:, 0])
    moved, valid = sample_rigid(images, torch.zeros((len(images), 3)), plane)
    assert valid.all()
    assert_allclose(moved, images, atol=2e-6)
    loss, meaningful = compute_global_ncc(moved, images[0], valid)
    expected = [
        -(np.corrcoef(image.numpy().ravel(), images[0].numpy().ravel())[0, 1] ** 2)
        for image in moved
    ]
    assert_allclose(loss, expected, atol=3e-7)
    assert meaningful.all()


@pytest.mark.parametrize("shift", [-1.25, -2e-5, -0.5e-5, 0.5e-5, 2e-5, 1.25])
def test_native_pixel_support_tolerance(recording_factory, shift):
    data, _, affine = recording_factory()
    plane = make_plane(data, dtype=torch.float64)
    parameters = torch.tensor(
        [[0, shift * affine[2, 2], -shift * affine[1, 1]]], dtype=torch.float64
    )
    images = torch.tensor(data.values[:1, 0], dtype=torch.float64)
    actual, valid = sample_rigid(images, parameters, plane)
    j, i = np.indices(data.shape[-2:], dtype=float)
    mapped_j, mapped_i = j - shift, i + shift
    expected_valid = (
        (mapped_j >= -1e-5)
        & (mapped_j <= j.max() + 1e-5)
        & (mapped_i >= -1e-5)
        & (mapped_i <= i.max() + 1e-5)
    )
    expected = map_coordinates(
        images[0].numpy(),
        [mapped_j, mapped_i],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    assert_array_equal(valid[0], expected_valid)
    assert_allclose(actual[0], expected, atol=1e-12)


def test_gradients_and_masked_ncc(recording_factory):
    data, _, _ = recording_factory(oblique=True)
    plane = make_plane(data, dtype=torch.float64)
    images = torch.tensor(data.values[1:2, 0], dtype=torch.float64)
    fixed = torch.tensor(data.values[0, 0], dtype=torch.float64)
    parameters = torch.tensor(
        [[0.011, 0.073, -0.041]], dtype=torch.float64, requires_grad=True
    )

    def objective(p):
        sampled, _ = sample_rigid(images, p, plane)
        mask = torch.zeros_like(sampled, dtype=torch.bool)
        mask[:, 5:-5, 5:-5] = True
        return compute_global_ncc(sampled, fixed, mask)[0]

    assert torch.autograd.gradcheck(
        objective, (parameters,), eps=1e-6, atol=1e-5, rtol=1e-4
    )
    moved, valid = sample_rigid(images, parameters, plane)
    loss, _ = compute_global_ncc(moved, fixed, valid)
    expected = -(
        np.corrcoef(
            moved.detach().numpy()[valid],
            np.broadcast_to(fixed.numpy(), moved.shape)[valid],
        )[0, 1]
        ** 2
    )
    assert_allclose(loss.detach(), [expected], atol=1e-12)
    empty_loss, meaningful = compute_global_ncc(
        torch.ones_like(images), fixed, torch.zeros_like(images, dtype=torch.bool)
    )
    assert_array_equal(meaningful, [False])
    assert_array_equal(empty_loss, [0])


def test_float32_ncc_rejects_rounded_constant_means(device):
    moved = torch.full((2, 23, 31), 0.1, device=device, requires_grad=True)
    loss, meaningful = compute_global_ncc(
        moved, moved.detach(), torch.ones_like(moved, dtype=torch.bool)
    )
    assert not meaningful.any()
    assert torch.equal(loss, torch.zeros_like(loss))
    gradient = torch.autograd.grad(loss.sum(), moved)[0]
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_float32_ncc_identical_odd_sized_frames(device):
    source = torch.linspace(0, 1, 3 * 5 * 7, device=device).reshape(1, 3, 5, 7)
    moved = source.sin().repeat(3, 1, 1, 1).requires_grad_()
    fixed = (source[0] * 3).cos()
    valid = torch.ones_like(moved, dtype=torch.bool)
    loss, meaningful = compute_global_ncc(moved, fixed, valid)
    assert meaningful.all() and torch.equal(loss, loss[:1].expand_as(loss))
    gradient = torch.autograd.grad(loss.sum(), moved)[0]
    assert torch.equal(gradient, gradient[:1].expand_as(gradient))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ncc_follows_fitting_tensor_precision(device, dtype):
    values = torch.arange(24, dtype=dtype, device=device).reshape(2, 3, 4) / 24
    moved = values.sin().requires_grad_()
    fixed = (values[0] * 3).cos()
    valid = torch.ones_like(moved, dtype=torch.bool)
    valid[:, 0, 0] = False
    loss, meaningful = compute_global_ncc(moved, fixed, valid)
    assert loss.dtype == dtype and meaningful.all()
    gradient = torch.autograd.grad(loss.sum(), moved)[0]
    assert gradient.dtype == dtype and torch.isfinite(gradient).all()
    expected = [
        -(np.corrcoef(x[mask], fixed.cpu().numpy()[mask])[0, 1] ** 2)
        for x, mask in zip(
            moved.detach().cpu().numpy(), valid.cpu().numpy(), strict=True
        )
    ]
    assert_allclose(
        loss.detach().cpu(), expected, atol=2e-6 if dtype == torch.float32 else 1e-12
    )
