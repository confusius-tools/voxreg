"""Independent interpolation, physical mappings, NCC and autodiff checks."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from scipy.ndimage import map_coordinates

from voxreg.sampling import global_ncc, make_plane, sample_rigid, world_matrices


@pytest.mark.parametrize("strided", [False, True])
def test_physical_sampler_against_scipy(recording_factory, strided):
    data, _, affine = recording_factory(oblique=True, strided=strided)
    fixed = data.isel(time=0, drop=True)
    plane = make_plane(fixed, dtype=torch.float64)
    parameters = torch.tensor([[0.019, 0.15, -0.12]], dtype=torch.float64)
    matrix = world_matrices(parameters.numpy(), plane)[0]
    j, i = np.indices(data.shape[-2:])
    points = np.stack((np.zeros_like(i), j, i, np.ones_like(i)))
    mapped = np.einsum("ab,bhw->ahw", np.linalg.inv(affine) @ matrix @ affine, points)
    h, w = j.shape
    valid = (
        (mapped[1] >= -1e-5)
        & (mapped[1] <= h - 1 + 1e-5)
        & (mapped[2] >= -1e-5)
        & (mapped[2] <= w - 1 + 1e-5)
    )
    expected = map_coordinates(
        data.values[0, 0].astype(float),
        mapped[1:3],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    actual, actual_valid = sample_rigid(
        torch.tensor(data.values[0], dtype=torch.float64), parameters, plane
    )
    assert_allclose(actual[0], expected, atol=1e-12)
    assert_array_equal(actual_valid[0], valid)
    assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-14)
    assert_allclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-14)
    inverse = np.linalg.inv(matrix)
    assert_allclose(inverse @ matrix, np.eye(4), atol=1e-13)
    center = np.r_[plane.center, 1]
    assert_allclose(
        (matrix @ center - center)[:3],
        plane.basis @ parameters.numpy()[0, 1:],
        atol=1e-13,
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
    expected = (
        -(
            np.corrcoef(
                moved.detach().numpy()[valid],
                np.broadcast_to(fixed.numpy(), moved.shape)[valid],
            )[0, 1]
            ** 2
        )
    )
    assert_allclose(loss.detach(), [expected], atol=1e-12)
    empty_loss, meaningful = global_ncc(
        torch.ones_like(images), fixed, torch.zeros_like(images, dtype=torch.bool)
    )
    assert_array_equal(meaningful, [False])
    assert_array_equal(empty_loss, [0])
