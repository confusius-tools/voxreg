"""Mixed registration samples native caller-fixed grids without slab expansion."""

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from numpy.testing import assert_allclose, assert_array_equal
from oracles import voxel
from scipy.spatial.transform import Rotation

from voxreg import linear, position_to_world, register
from voxreg.sampling import (
    build_world_matrices,
    make_grid,
    recover_initial_parameters,
    sample_rigid,
)
from voxreg.transforms import check_affine_parameters, configure_transform

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def observe_seed(monkeypatch, records):
    def optimize(
        images, target, grid, reference, *args, initial_parameters=None, **kwargs
    ):
        parameters = (
            initial_parameters
            if initial_parameters is not None
            else torch.zeros(
                (len(images), len(grid.parameter_scale)),
                dtype=torch.float64,
                device=images.device,
            )
        )
        sampled, _valid = sample_rigid(images, parameters, grid)
        records.append(
            (
                tuple(images.shape[1:]),
                tuple(target.shape),
                tuple(sampled.shape[1:]),
                grid.center.copy(),
                build_world_matrices(parameters.cpu().numpy(), grid),
            )
        )
        return (
            parameters,
            torch.zeros(len(images), dtype=torch.int64, device=images.device),
            torch.full((len(images),), 5, dtype=torch.int64, device=images.device),
        )

    monkeypatch.setattr(linear, "_optimize", optimize)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
@pytest.mark.parametrize("moving_slice", [False, True])
def test_native_domains_and_nonidentity_seed_match_sitk_value(
    slice_volume_pair_factory, monkeypatch, device, model, moving_slice
):
    from confusius.registration._utils import voxeldata_to_sitk_image

    plane, volume, truth = slice_volume_pair_factory(model)
    common = np.eye(4)
    common[:3, :3] = Rotation.from_rotvec([0.13, -0.2, 0.17]).as_matrix()
    common[:3, 3] = [11, -7, 19]
    plane = voxel(plane.values, common @ position_to_world(plane))
    volume = voxel(volume.values, common @ position_to_world(volume))
    truth = common @ truth @ np.linalg.inv(common)
    moving, fixed, seed = (
        (plane, volume, np.linalg.inv(truth))
        if moving_slice
        else (volume, plane, truth)
    )
    originals = (moving.values.copy(), fixed.values.copy(), seed.copy())
    records = []
    observe_seed(monkeypatch, records)
    result = register(
        moving,
        fixed,
        return_transforms=True,
        transform=model,
        initialization=seed,
        device=device,
        max_iterations=1,
        progress=False,
    )
    assert result.transforms is not None
    source_shape, target_shape, sampled_shape, pivot, decoded = records[0]
    assert source_shape == moving.shape and target_shape == sampled_shape == fixed.shape
    assert_allclose(
        pivot,
        (position_to_world(fixed) @ np.r_[(np.array(fixed.shape) - 1) / 2, 1])[:3],
        atol=1e-13,
        rtol=0,
    )
    assert_allclose(decoded[0], seed, atol=1e-12, rtol=0)
    assert_allclose(result.transforms, seed, atol=1e-12, rtol=0)
    assert result.registered.shape == fixed.shape
    assert result.diagnostics.attrs["metric_domain"] == (
        "fixed_volume" if moving_slice else "fixed_slice"
    )
    assert (
        result.diagnostics.attrs["optimized_parameters"]
        == {"translation": 3, "rigid": 6, "affine": 9}[model]
    )
    evaluator = sitk.ImageRegistrationMethod()
    evaluator.SetMetricAsCorrelation()
    evaluator.SetMetricUseFixedImageGradientFilter(False)
    evaluator.SetMetricUseMovingImageGradientFilter(False)
    evaluator.SetInterpolator(sitk.sitkLinear)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(seed[:3, :3].ravel())
    transform.SetTranslation(seed[:3, 3])
    evaluator.SetInitialTransform(transform)
    expected = evaluator.MetricEvaluate(
        sitk.Cast(voxeldata_to_sitk_image(fixed), sitk.sitkFloat64),
        sitk.Cast(voxeldata_to_sitk_image(moving), sitk.sitkFloat64),
    )
    assert_allclose(result.diagnostics.objective, expected, atol=1e-6, rtol=0)
    assert_array_equal(moving.values, originals[0])
    assert_array_equal(fixed.values, originals[1])
    assert_array_equal(seed, originals[2])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
def test_singleton_and_active_axes_use_half_voxel_support(monkeypatch, device, model):
    affine = np.diag([0.25, 0.5, 0.5, 1.0])
    moving = voxel(np.arange(20, dtype=float).reshape(1, 4, 5) + 1, affine)
    target_affine = affine.copy()
    target_affine[:3, 3] = affine[:3, :3] @ [-0.5, -0.5, -0.5]
    target_affine[:3, 0] /= 2
    fixed = voxel(np.arange(168, dtype=float).reshape(4, 6, 7) + 1, target_affine)
    records = []
    observe_seed(monkeypatch, records)
    result = register(
        moving,
        fixed,
        transform=model,
        device=device,
        max_iterations=1,
        progress=False,
    )
    expected = np.zeros(fixed.shape, dtype=bool)
    expected[:2, :4, :5] = True
    assert_array_equal(result.valid, expected)
    assert result.registered.values[0, 0, 0] == moving.values[0, 0, 0]
    assert result.registered.dtype == moving.dtype


@pytest.mark.parametrize("moving_slice", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_slice_affine_constraint_is_on_recording_to_volume_map(
    slice_volume_pair_factory, moving_slice, device
):
    plane, volume, _ = slice_volume_pair_factory("affine")
    moving, fixed = (plane, volume) if moving_slice else (volume, plane)
    grid = configure_transform(make_grid(fixed, moving, device=device), fixed, "affine")
    p = torch.tensor(
        [[0.04, -0.03, 0.1, 0.02, -0.015, 0.06, 0.12, -0.08, 0.09]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    public = build_world_matrices(p.detach().cpu().numpy(), grid)
    native = np.linalg.inv(public)[0] if moving_slice else public[0]
    frame = np.column_stack(
        (grid.slice_basis, np.cross(grid.slice_basis[:, 0], grid.slice_basis[:, 1]))
    )
    mapped = native[:3, :3] @ frame
    assert_allclose(np.linalg.norm(mapped[:, 2]), 1, atol=1e-13, rtol=0)
    assert_allclose(mapped[:, :2].T @ mapped[:, 2], 0, atol=1e-13, rtol=0)
    assert_allclose(
        recover_initial_parameters(public, grid, 1),
        p.detach().cpu().numpy(),
        atol=1e-12,
        rtol=0,
    )
    sampled, valid = sample_rigid(
        torch.tensor(moving.values[None], device=device), p, grid
    )
    sampled[valid].square().mean().backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all() and p.grad.abs().max() > 0
    collapsed = p.detach().clone()
    collapsed[:, :6] = -torch.tensor(grid.slice_basis.reshape(-1), device=device)
    assert not check_affine_parameters(collapsed, grid)[0].any()
    invalid = native.copy()
    invalid[:3, :3] = mapped @ np.diag([1, 1, 0.7]) @ frame.T
    with pytest.raises(ValueError, match="normal scale/shear/reflection"):
        recover_initial_parameters(
            np.linalg.inv(invalid) if moving_slice else invalid, grid, 1
        )
