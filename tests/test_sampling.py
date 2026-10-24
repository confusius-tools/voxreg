"""Independent interpolation, physical mappings, NCC and autodiff checks."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from scipy.ndimage import map_coordinates

from voxreg import create_voxeldata, position_to_world
from voxreg.sampling import global_ncc, make_plane, sample_rigid, world_matrices


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("sheared", [False, True])
def test_physical_sampler_against_scipy(recording_factory, strided, sheared):
    data, _, _ = recording_factory(oblique=True, strided=strided)
    if sheared:
        affine = data.xindexes["z"].transform.voxel_to_world.copy()
        affine[:3, 1] += 0.35 * affine[:3, 2]
        data = create_voxeldata(
            data.values,
            dims=data.dims,
            coords={dim: data[dim].values for dim in data.dims},
            voxel_to_world=affine,
        )
    affine = position_to_world(data)
    fixed = data.isel(time=0, drop=True)
    plane = make_plane(fixed, dtype=torch.float64)
    parameters = torch.tensor(
        [[0, 0, 0], [0.019, 0.15, -0.12], [-0.13, -0.21, 0.17]], dtype=torch.float64
    )
    matrices = world_matrices(parameters.numpy(), plane)
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
    loss, meaningful = global_ncc(moved, images[0], valid)
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
        return global_ncc(sampled, fixed, mask)[0]

    assert torch.autograd.gradcheck(
        objective, (parameters,), eps=1e-6, atol=1e-5, rtol=1e-4
    )
    moved, valid = sample_rigid(images, parameters, plane)
    loss, _ = global_ncc(moved, fixed, valid)
    expected = -(
        np.corrcoef(
            moved.detach().numpy()[valid],
            np.broadcast_to(fixed.numpy(), moved.shape)[valid],
        )[0, 1]
        ** 2
    )
    assert_allclose(loss.detach(), [expected], atol=1e-12)
    empty_loss, meaningful = global_ncc(
        torch.ones_like(images), fixed, torch.zeros_like(images, dtype=torch.bool)
    )
    assert_array_equal(meaningful, [False])
    assert_array_equal(empty_loss, [0])
