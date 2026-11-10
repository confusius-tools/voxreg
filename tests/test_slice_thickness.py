"""A singleton voxel occupies its affine-defined cell, not a zero-width plane."""

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from numpy.testing import assert_allclose, assert_array_equal
from oracles import to_sitk_image
from scipy.spatial.transform import Rotation
from voxeldata import create_voxeldata

from voxreg import position_to_world
from voxreg.sampling import make_grid, recover_initial_parameters, sample_rigid
from voxreg.transforms import configure_transform

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
@pytest.mark.parametrize("oblique", [False, True])
def test_singleton_slab_matches_sitk_without_replicating_pixels(
    device, dtype, interpolation, oblique
):
    affine = np.diag([0.5, 0.25, 0.125, 1.0])
    values = np.arange(1, 31, dtype=dtype).reshape(1, 5, 6)
    if oblique:
        affine[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix() @ np.diag(
            [0.8, 0.25, 0.125]
        )
        affine[:3, 3] = [3.0, -2.0, 4.0]
    source = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=affine)
    pull = np.eye(4)
    if oblique:
        source = source.isel(j=slice(None, None, -1), i=slice(None, None, 2))
        pull[:3, :3] = Rotation.from_rotvec([-0.15, 0.1, 0.05]).as_matrix()
        pull[:3, 3] = [0.3, -0.2, 0.1]
    # Axis-aligned samples include both exact faces: -.5 is inside, +.5 outside.
    index = np.eye(4)
    index[0, 0], index[0, 3] = (0.2, -0.6) if oblique else (0.25, -0.75)
    target_affine = np.linalg.inv(pull) @ position_to_world(source) @ index
    target = create_voxeldata(
        np.ones((7, *source.shape[1:]), dtype=dtype),
        dims=("k", "j", "i"),
        voxel_to_world=target_affine,
    )
    grid = configure_transform(
        make_grid(target, source, device=device), target, "affine"
    )
    parameters = recover_initial_parameters(pull, grid, 1)
    sampled, valid = sample_rigid(
        torch.tensor(np.ascontiguousarray(source.values[None]), device=device),
        torch.tensor(parameters, device=device),
        grid,
        interpolation=interpolation,
    )
    fill = float(source.min())
    actual = torch.where(valid, sampled, fill).cpu().numpy()[0]
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(pull[:3, :3].ravel())
    transform.SetTranslation(pull[:3, 3])
    sampler = sitk.ResampleImageFilter()
    sampler.SetNumberOfThreads(1)
    sampler.SetReferenceImage(to_sitk_image(target))
    sampler.SetTransform(transform)
    sampler.SetInterpolator(
        sitk.sitkLinear if interpolation == "linear" else sitk.sitkNearestNeighbor
    )
    sampler.SetDefaultPixelValue(fill)
    reference = sitk.GetArrayFromImage(sampler.Execute(to_sitk_image(source))).T
    ones = source.copy(data=np.ones(source.shape))
    sampler.SetDefaultPixelValue(0.0)
    expected_valid = (
        sitk.GetArrayFromImage(sampler.Execute(to_sitk_image(ones))).T > 0.5
    )
    assert_array_equal(valid.cpu().numpy()[0], expected_valid)
    assert_allclose(
        actual, reference, atol=1e-5 if dtype == np.float32 else 1e-12, rtol=0
    )
    assert actual.dtype == dtype
    assert valid.any()
    assert not valid.all()
    if not oblique:
        assert_array_equal(
            expected_valid.any(axis=(1, 2)),
            [False, True, True, True, True, False, False],
        )
