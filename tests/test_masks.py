"""Native mask geometry, masked statistics/pyramids and output support contracts."""

import numpy as np
import pytest
import torch
import xarray as xr
from confusius import create_voxeldata
from numpy.testing import assert_allclose, assert_array_equal
from oracles import max_pull_difference
from registration_options import RegistrationOptions

from voxreg import position_to_world, register, resample
from voxreg.masks import _MaskSampler, _MetricMasks, _source
from voxreg.metrics import compute_local_ncc
from voxreg.sampling import _recover_initial_parameters, make_grid
from voxreg.transforms import configure_transform


def _roi(data):
    indices = np.indices(data.shape)
    mask = np.ones(data.shape, dtype=bool)
    for axis, size in enumerate(data.shape):
        if size > 1:
            mask &= (indices[axis] >= size // 4) & (indices[axis] < 3 * size // 4)
    return data.copy(data=mask)


@pytest.mark.parametrize("planar", [True, False])
def test_mask_sampler_matches_world_cell_oracle(linear_pair_factory, device, planar):
    fixed, moving, truth = linear_pair_factory(planar, "affine")
    mask = _roi(moving).isel(j=slice(None, None, -2), i=slice(1, None, 2))
    affine = position_to_world(mask)
    # Avoid exact nearest-neighbor ties in an independent floating-point oracle.
    affine[:3, 3] += affine[:3, :3] @ np.array([0.0 if planar else 0.13, 0.17, 0.19])
    mask = create_voxeldata(
        mask.values.copy(), dims=("k", "j", "i"), voxel_to_world=affine
    )
    geometry = configure_transform(
        make_grid(fixed, moving, device=device), fixed, "affine"
    )
    parameters = torch.tensor(
        _recover_initial_parameters(truth, geometry, 1), device=device
    )
    source = _source(mask, device)
    actual = (
        _MaskSampler(source, fixed, geometry).sample(1, parameters).cpu().numpy()[0]
    )
    world = position_to_world(fixed) @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones(fixed.size))
    )
    mapped = (np.linalg.inv(position_to_world(mask)) @ truth @ world)[:3]
    valid = ((mapped >= -0.5) & (mapped < np.array(mask.shape)[:, None] - 0.5)).all(
        axis=0
    )
    nearest = (
        np.floor(mapped + 0.5).astype(int).clip(0, np.array(mask.shape)[:, None] - 1)
    )
    expected = valid & mask.values[tuple(nearest)]
    assert_array_equal(actual.reshape(-1), expected)


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("model", ["translation", "rigid", "affine", "syn"])
def test_full_masks_preserve_native_fit(linear_pair_factory, device, planar, model):
    fixed, moving, _ = linear_pair_factory(planar, "rigid")
    options: RegistrationOptions = {
        "transform": model,
        "max_iterations": 3,
        "intensity_scaling": "none",
        "device": device,
        "return_transforms": True,
        "progress": False,
    }
    baseline = register(moving, fixed, **options)
    masked = register(
        moving,
        fixed,
        fixed_mask=fixed.copy(data=np.ones_like(fixed, dtype=bool)),
        moving_mask=moving.copy(data=np.ones_like(moving, dtype=bool)),
        **options,
    )
    xr.testing.assert_identical(masked.registered, baseline.registered)
    xr.testing.assert_identical(masked.forward_transforms, baseline.forward_transforms)
    xr.testing.assert_identical(masked.valid, baseline.valid)
    assert_array_equal(masked.diagnostics.objective, baseline.diagnostics.objective)


@pytest.mark.parametrize("model", ["rigid", "syn"])
@pytest.mark.parametrize("pyramid", [False, True])
def test_roi_fitting_keeps_output_support_distinct(linear_pair_factory, model, pyramid):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    result = register(
        moving,
        fixed,
        transform=model,
        fixed_mask=_roi(fixed),
        moving_mask=_roi(moving),
        shrink_factors=(2, 1) if pyramid else (1,),
        smoothing_sigmas=(0.18, 0) if pyramid else (0,),
        max_iterations=3,
        intensity_scaling="none",
        return_transforms=True,
        progress=False,
    )
    assert result.forward_transforms is not None
    sampled, valid = resample(moving, fixed, result.forward_transforms)
    assert_allclose(result.registered, sampled, rtol=0, atol=2e-6)
    assert_array_equal(result.valid, valid)
    assert result.diagnostics.roi_valid_fraction < result.diagnostics.valid_fraction
    assert np.isfinite(result.diagnostics.objective)


@pytest.mark.parametrize("model", ["rigid", "syn"])
def test_empty_roi_fails_explicitly_without_changing_seed(linear_pair_factory, model):
    fixed, moving, seed = linear_pair_factory(True, "rigid")
    result = register(
        moving,
        fixed,
        transform=model,
        initial_transform=seed,
        fixed_mask=fixed.copy(data=np.zeros_like(fixed, dtype=bool)),
        max_iterations=3,
        return_transforms=True,
        progress=False,
    )
    assert result.diagnostics.status == "insufficient_overlap"
    assert result.diagnostics.n_iterations == 0
    assert result.diagnostics.roi_valid_fraction == 0
    if model == "rigid":
        assert result.forward_transforms is not None
        assert_allclose(result.forward_transforms, seed, rtol=0, atol=1e-12)
    assert result.valid.any()


def test_mask_cell_rounding_follows_itk():
    import itk

    values = np.zeros((1, 3, 5), dtype=bool)
    values[:, :, 1::2] = True
    mask = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    affine = np.eye(4)
    affine[2, 3] = 0.5
    domain = create_voxeldata(
        np.ones((1, 3, 4)), dims=("k", "j", "i"), voxel_to_world=affine
    )
    actual = _MaskSampler(_source(mask, "cpu"), domain).sample(1).numpy()[0]
    image = itk.image_from_array(values.astype(np.uint8))
    spatial = getattr(itk, "ImageMaskSpatialObject")[3].New(Image=image)  # noqa: B009 -- generated ITK type
    spatial.Update()
    expected = [spatial.IsInsideInWorldSpace((i + 0.5, 1.0, 0.0)) for i in range(4)]
    assert_array_equal(actual[1], expected)
    assert_array_equal(actual[1], [True, False, True, False])


@pytest.mark.parametrize("terminal", ["affine", "syn"])
def test_per_frame_masks_isolate_staged_motion_failures(linear_pair_factory, terminal):
    fixed, _, _ = linear_pair_factory(True, "rigid")
    moving = xr.concat(
        [fixed, fixed.copy(data=np.roll(fixed.values, 1, axis=-1)), fixed],
        dim=xr.IndexVariable("time", [1, 3, 5]),
    )
    mask = moving.copy(data=np.ones(moving.shape, dtype=bool))
    mask.values[2] = False
    original = mask.copy(deep=True)
    results = []
    for batch in (1, 3):
        result = register(
            moving,
            transform=("rigid", terminal),
            moving_mask=mask,
            fixed_mask=_roi(fixed),
            max_iterations=3,
            shrink_factors=(2, 1),
            smoothing_sigmas=(0.18, 0),
            intensity_scaling="none",
            batch_size=batch,
            return_transforms=True,
            progress=False,
        )
        assert result.stage_diagnostics is not None
        assert_array_equal(
            result.diagnostics.status,
            ["reference", "max_iterations", "insufficient_overlap"],
        )
        assert result.stage_diagnostics[1].status[2] == "skipped"
        assert_array_equal(result.registered[0], moving[0])
        results.append(result)
    xr.testing.assert_identical(mask, original)
    first, second = results
    assert (
        first.forward_transforms is not None and second.forward_transforms is not None
    )
    assert_array_equal(
        first.forward_transforms.isel(time=0),
        second.forward_transforms.isel(time=0),
    )
    assert_allclose(
        first.forward_transforms.isel(time=2),
        second.forward_transforms.isel(time=2),
        rtol=0,
        atol=1e-12,
    )
    assert (
        max_pull_difference(first.forward_transforms, second.forward_transforms, fixed)
        < 0.001
    )


def test_mask_validation(linear_pair_factory):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    with pytest.raises(TypeError, match="boolean"):
        register(moving, fixed, fixed_mask=fixed)
    with pytest.raises(ValueError, match="spatial-only"):
        register(moving, fixed, fixed_mask=_roi(fixed).expand_dims(time=[0]))
    recording = moving.expand_dims(time=[2, 4])
    with pytest.raises(ValueError, match="time labels"):
        register(recording, fixed, moving_mask=_roi(moving).expand_dims(time=[0, 1]))


@pytest.mark.parametrize("planar", [True, False])
def test_masked_global_ncc_agrees_with_itk(linear_pair_factory, planar):
    import SimpleITK as sitk
    from confusius.registration._utils import voxeldata_to_sitk_image

    fixed, moving, _ = linear_pair_factory(planar, "rigid")
    fixed_mask, moving_mask = _roi(fixed), _roi(moving)
    result = register(
        moving,
        fixed,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        intensity_scaling="none",
        max_iterations=3,
        return_transforms=True,
        progress=False,
    )
    assert result.forward_transforms is not None
    matrix = result.forward_transforms.values
    evaluator = sitk.ImageRegistrationMethod()
    evaluator.SetMetricAsCorrelation()
    evaluator.SetMetricUseFixedImageGradientFilter(False)
    evaluator.SetMetricUseMovingImageGradientFilter(False)
    evaluator.SetInterpolator(sitk.sitkLinear)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix[:3, :3].ravel())
    transform.SetTranslation(matrix[:3, 3])
    evaluator.SetInitialTransform(transform)
    evaluator.SetMetricFixedMask(voxeldata_to_sitk_image(fixed_mask.astype(np.uint8)))
    evaluator.SetMetricMovingMask(voxeldata_to_sitk_image(moving_mask.astype(np.uint8)))
    expected = evaluator.MetricEvaluate(
        sitk.Cast(voxeldata_to_sitk_image(fixed), sitk.sitkFloat64),
        sitk.Cast(voxeldata_to_sitk_image(moving), sitk.sitkFloat64),
    )
    assert_allclose(result.diagnostics.objective, expected, rtol=0, atol=2e-6)


@pytest.mark.parametrize("moving_slice", [True, False])
def test_masks_support_native_mixed_domains(slice_volume_pair_factory, moving_slice):
    fixed, moving, truth = slice_volume_pair_factory("rigid")
    if moving_slice:
        fixed, moving, truth = moving, fixed, np.linalg.inv(truth)
    result = register(
        moving,
        fixed,
        initial_transform=truth,
        fixed_mask=_roi(fixed),
        moving_mask=_roi(moving),
        shrink_factors=(2, 1),
        smoothing_sigmas=(0.18, 0),
        max_iterations=2,
        return_transforms=True,
        intensity_scaling="none",
        progress=False,
    )
    assert result.forward_transforms is not None
    assert np.isfinite(result.diagnostics.objective)
    sampled, valid = resample(moving, fixed, result.forward_transforms)
    assert_allclose(sampled, result.registered, rtol=0, atol=2e-6)
    assert_array_equal(valid, result.valid)


def test_local_ncc_excludes_masked_neighbours_from_statistics():
    x = np.arange(42, dtype=np.float64).reshape(6, 7) / 10
    y = np.sin(x) + x / 3
    roi = np.ones(x.shape, dtype=bool)
    roi[1:4, 3:5] = False
    actual, meaningful = compute_local_ncc(
        torch.tensor(x[None]), torch.tensor(y[None]), torch.tensor(roi[None]), radius=1
    )
    correlations = []
    for j, i in np.argwhere(roi):
        region = (
            slice(max(0, j - 1), min(6, j + 2)),
            slice(max(0, i - 1), min(7, i + 2)),
        )
        a, b = x[region][roi[region]], y[region][roi[region]]
        if len(a) > 1 and np.var(a) > 1e-12 and np.var(b) > 1e-12:
            correlations.append(np.corrcoef(a, b)[0, 1] ** 2)
    assert meaningful.item()
    assert_allclose(actual.item(), -np.mean(correlations), rtol=0, atol=1e-12)


@pytest.mark.parametrize("planar", [True, False])
def test_syn_masks_follow_both_halfway_maps(linear_pair_factory, planar):
    fixed, moving, seed = linear_pair_factory(planar, "affine")
    geometry = configure_transform(make_grid(fixed, moving), fixed, "affine")
    parameters = torch.tensor(_recover_initial_parameters(seed, geometry, 1))
    fixed_mask, moving_mask = _roi(fixed), _roi(moving)
    masks = _MetricMasks(
        _source(fixed_mask, "cpu"), _source(moving_mask, "cpu"), fixed, geometry
    )
    shape = fixed.shape[1:] if planar else fixed.shape
    components = len(shape)
    uf, um = torch.zeros((1, components, *shape)), torch.zeros((1, components, *shape))
    uf[:, 0] = 0.23
    um[:, -1] = -0.19
    actual = masks.apply(torch.ones((1, *shape), dtype=torch.bool), parameters, um, uf)
    # Independent full-world pulls: fixed half moves on its own; moving half
    # moves before the complete affine seed, with mm field components.
    points = position_to_world(fixed) @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones(fixed.size))
    )
    fixed_world = points[:3] + geometry.basis @ uf.numpy().reshape(components, -1)
    halfway = points.copy()
    halfway[:3] += geometry.basis @ um.numpy().reshape(components, -1)
    moving_world = (seed @ halfway)[:3]

    def inside(mask, world):
        affine = position_to_world(mask)
        positions = np.linalg.inv(affine[:3, :3]) @ (world - affine[:3, 3, None])
        valid = (
            (positions >= -0.5) & (positions < np.array(mask.shape)[:, None] - 0.5)
        ).all(0)
        indices = (
            np.floor(positions + 0.5)
            .astype(int)
            .clip(0, np.array(mask.shape)[:, None] - 1)
        )
        return valid & mask.values[tuple(indices)]

    expected = inside(fixed_mask, fixed_world) & inside(moving_mask, moving_world)
    assert_array_equal(actual.numpy().reshape(-1), expected)
    baseline = masks.apply(torch.ones_like(actual), parameters)
    assert not torch.equal(actual, baseline)
