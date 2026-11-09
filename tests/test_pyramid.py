"""Pyramid geometry, physical smoothing and end-to-end registration contracts."""

import numpy as np
import pytest
import torch
import xarray as xr
from confusius import create_voxeldata
from numpy.testing import assert_allclose, assert_array_equal
from registration_options import RegistrationOptions
from scipy.ndimage import convolve

from voxreg import position_to_world, register, resample
from voxreg.fields import make_field_grid
from voxreg.pyramid import _smooth_images, make_level, resize_halfway


@pytest.mark.parametrize(
    "options, message",
    [
        ({"shrink_factors": ()}, "shrink_factors"),
        ({"shrink_factors": 2}, "shrink_factors"),
        ({"shrink_factors": (0,)}, "shrink_factors"),
        ({"shrink_factors": (True,)}, "shrink_factors"),
        ({"shrink_factors": (1.5,)}, "shrink_factors"),
        ({"shrink_factors": (1, 2), "smoothing_sigmas": (0, 0)}, "coarse to fine"),
        ({"shrink_factors": (2, 1)}, "equal lengths"),
        ({"smoothing_sigmas": ()}, "smoothing_sigmas"),
        ({"smoothing_sigmas": (True,)}, "smoothing_sigmas"),
        ({"smoothing_sigmas": (-1,)}, "smoothing_sigmas"),
        ({"smoothing_sigmas": (np.nan,)}, "smoothing_sigmas"),
        ({"smoothing_sigmas": (np.inf,)}, "smoothing_sigmas"),
        ({"max_iterations": ()}, "max_iterations"),
        ({"max_iterations": (1, 2)}, "max_iterations"),
        ({"max_iterations": (False,)}, "max_iterations"),
        ({"max_iterations": (0,)}, "max_iterations"),
    ],
)
def test_schedule_validation(linear_pair_factory, options, message):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    with pytest.raises(ValueError, match=message):
        register(moving, fixed, **options)


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("model", ["translation", "rigid", "affine", "syn"])
def test_explicit_single_level_preserves_defaults(linear_pair_factory, planar, model):
    fixed, moving, _ = linear_pair_factory(planar, "rigid")
    options: RegistrationOptions = {
        "transform": model,
        "max_iterations": 2,
        "return_transforms": True,
        "progress": False,
    }
    baseline = register(moving, fixed, **options)
    options["max_iterations"] = (2,)
    explicit = register(
        moving,
        fixed,
        **options,
        shrink_factors=(1,),
        smoothing_sigmas=(0,),
    )
    for name in ("registered", "valid", "forward_transforms", "inverse_transforms"):
        xr.testing.assert_identical(getattr(baseline, name), getattr(explicit, name))
    xr.testing.assert_equal(baseline.diagnostics, explicit.diagnostics)


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("sheared", [False, True])
def test_physical_smoothing_and_native_strides(device, planar, sheared):
    shape = (1, 13, 15) if planar else (7, 13, 15)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([1.5, 0.7, 0.4])
    if sheared:
        affine[2, 1] = 0.2
    affine[:3, 3] = [1e9, -3e8, 4e8]
    data = create_voxeldata(
        np.random.default_rng(4).normal(size=shape),
        dims=("k", "j", "i"),
        voxel_to_world=affine,
    )
    data = data.isel(j=slice(None, None, -2), i=slice(1, None, 2))
    image = data.values[0] if planar else data.values
    tensor = torch.tensor(image[None].copy(), device=device)
    sigma = 0.6
    lattice = position_to_world(data)[:3, 1 if planar else 0 : 3]
    covariance = sigma**2 * np.linalg.inv(lattice.T @ lattice)
    radii = np.ceil(3 * np.sqrt(np.diag(covariance))).astype(int)
    offsets = np.stack(
        np.meshgrid(*(np.arange(-r, r + 1) for r in radii), indexing="ij"), axis=-1
    )
    kernel = np.exp(-np.sum((offsets @ lattice.T) ** 2, axis=-1) / (2 * sigma**2))
    expected = convolve(image, kernel / kernel.sum(), mode="nearest")
    actual, level = make_level(tensor, data, 4, sigma)
    strides = tuple(min(4, max(1, n - 1)) for n in image.shape)
    slices = tuple(slice(None, None, stride) for stride in strides)
    assert_allclose(actual.cpu()[0], expected[slices], rtol=0, atol=2e-14)
    original = position_to_world(data)
    expected_affine = original.copy()
    expected_affine[:3, 1 if planar else 0 : 3] *= strides
    assert_allclose(position_to_world(level), expected_affine, rtol=0, atol=0)
    assert level.sizes["k"] == 1 if planar else level.sizes["k"] >= 2
    assert min(level.sizes["j"], level.sizes["i"]) >= 2
    assert_allclose(_smooth_images(tensor, data, 0).cpu(), tensor.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("planar", [True, False])
def test_halfway_prolongation_keeps_physical_units(displacement_factory, planar):
    data, _, _, _ = displacement_factory(planar, strided=True)
    shape = data.shape[1:] if planar else data.shape
    _, coarse = make_level(torch.zeros((1, *shape)), data, 2, 0)
    source, destination = make_field_grid(coarse), make_field_grid(data)
    # An affine field has an independently known linear interpolation everywhere inside.
    indices = source.indices[0]
    field = (
        (0.03 * indices.sum(-1))[None, None]
        .expand(1, len(shape), *source.shape)
        .clone()
    )
    actual = resize_halfway(field, source, destination)
    expected = (0.03 * (destination.indices[0] / 2).sum(-1))[None, None].expand_as(
        actual
    )
    inside = ~destination.boundary.expand_as(actual)
    for axis, size in enumerate(source.shape):
        inside &= (destination.indices[..., axis][:, None] / 2 <= size - 1).expand_as(
            actual
        )
    assert_allclose(actual[inside], expected[inside], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("model", ["translation", "rigid", "affine", "syn"])
def test_pyramid_native_output_and_level_diagnostics(
    linear_pair_factory, device, planar, model
):
    fixed, moving, truth = linear_pair_factory(planar, "rigid")
    original = moving.copy(deep=True)
    result = register(
        moving,
        fixed,
        transform=model,
        device=device,
        initial_transform=truth if model != "translation" else None,
        shrink_factors=(3, 1),
        smoothing_sigmas=(0.3, 0),
        max_iterations=(2, 3),
        return_transforms=True,
        progress=False,
    )
    assert result.registered.shape == fixed.shape
    assert result.registered.dtype == moving.dtype
    assert result.registered.xindexes["z"].equals(fixed.xindexes["z"])
    assert_array_equal(result.diagnostics.shrink_factor, [3, 1])
    assert_array_equal(result.diagnostics.iteration_budget, [2, 3])
    assert result.diagnostics.smoothing_sigma.attrs["units"] == "mm"
    assert result.diagnostics.level_n_iterations.dims == ("level",)
    assert (
        result.diagnostics.n_iterations == result.diagnostics.level_n_iterations.sum()
    )
    assert result.diagnostics.objective == result.diagnostics.level_objective[-1]
    assert result.diagnostics.status == result.diagnostics.level_status[-1]
    assert result.forward_transforms is not None
    sampled, valid = resample(moving, fixed, result.forward_transforms, device=device)
    assert_allclose(sampled, result.registered, rtol=0, atol=3e-6)
    assert_array_equal(valid, result.valid)
    xr.testing.assert_identical(moving, original)


@pytest.mark.parametrize("model", ["rigid", "syn"])
def test_coarse_final_level_still_outputs_native_grid(linear_pair_factory, model):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    result = register(
        moving,
        fixed,
        transform=model,
        shrink_factors=(3,),
        smoothing_sigmas=(0.2,),
        max_iterations=2,
        return_transforms=True,
        progress=False,
    )
    assert result.registered.shape == fixed.shape
    assert result.forward_transforms is not None
    sampled, valid = resample(moving, fixed, result.forward_transforms)
    assert_allclose(sampled, result.registered, rtol=0, atol=2e-6)
    assert_array_equal(valid, result.valid)


@pytest.mark.parametrize("model", ["rigid", "syn"])
def test_pyramid_reference_failure_isolation_and_batching(linear_pair_factory, model):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    # Use equal lattices for an implicit reference while retaining a nonidentity image.
    moving = fixed.copy(data=np.roll(fixed.values, 1, axis=-1))
    recording = xr.concat(
        [fixed, moving, fixed.copy(data=np.ones_like(fixed))],
        dim=xr.IndexVariable("time", [1, 3, 5]),
    )
    results = []
    for batch_size in (1, 3):
        progress, metrics = [], []
        result = register(
            recording,
            transform=model,
            shrink_factors=(2, 1),
            smoothing_sigmas=(0.2, 0),
            max_iterations=3,
            batch_size=batch_size,
            return_transforms=True,
            progress=lambda n, total, progress=progress: progress.append((n, total)),
            metric_callback=lambda frames, steps, loss, metrics=metrics: metrics.append(
                (frames, steps, loss)
            ),
        )
        assert progress[-1] == (3, 3)
        assert_array_equal(result.registered[0], recording[0])
        assert_array_equal(result.diagnostics.level_n_iterations[0], [0, 0])
        assert_array_equal(
            result.diagnostics.level_status[0], ["reference", "reference"]
        )
        assert_array_equal(
            result.diagnostics.level_status[2], ["invalid_metric", "invalid_metric"]
        )
        assert metrics and all(steps.max() <= 3 for _, steps, _ in metrics)
        fitted_clocks = [
            steps[frames == 1].item() for frames, steps, _ in metrics if 1 in frames
        ]
        assert fitted_clocks == [0, 3, 0, 3]
        results.append(result)
    assert_allclose(results[0].registered, results[1].registered, rtol=0, atol=2e-6)
    assert_array_equal(
        results[0].diagnostics.level_n_iterations,
        results[1].diagnostics.level_n_iterations,
    )


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
def test_linear_pyramid_recovers_independent_truth(
    linear_pair_factory, device, planar, model
):
    fixed, moving, truth = linear_pair_factory(planar, model)
    result = register(
        moving,
        fixed,
        transform=model,
        shrink_factors=(2, 1),
        smoothing_sigmas=(0.18, 0),
        max_iterations=(80, 120),
        convergence_window_size=1000,
        device=device,
        return_transforms=True,
        progress=False,
    )
    assert result.forward_transforms is not None
    points = position_to_world(fixed) @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones(fixed.size))
    )
    difference = ((result.forward_transforms.values - truth) @ points)[:3]
    spacing = np.linalg.norm(
        position_to_world(fixed)[:3, :3][:, np.array(fixed.shape) > 1], axis=0
    ).min()
    assert np.sqrt(np.mean(np.sum(difference**2, axis=0))) / spacing < 0.30
    assert_array_equal(result.diagnostics.level_n_iterations, [80, 120])


@pytest.mark.parametrize("planar", [True, False])
def test_syn_pyramid_recovers_nonlinear_truth(nonlinear_pair_factory, device, planar):
    fixed, moving, truth, _ = nonlinear_pair_factory(planar, axis_aligned=True)
    result = register(
        moving,
        fixed,
        transform="syn",
        shrink_factors=(2, 1),
        smoothing_sigmas=(0.18, 0),
        max_iterations=(60, 120),
        convergence_window_size=1000,
        device=device,
        return_transforms=True,
        progress=False,
    )
    assert result.forward_transforms is not None
    spacing = np.linalg.norm(
        position_to_world(fixed)[:3, :3][:, np.array(fixed.shape) > 1], axis=0
    ).min()
    residual = result.forward_transforms.values - truth
    assert np.sqrt(np.mean(np.sum(residual**2, axis=-1))) / spacing < 0.30
    assert result.diagnostics.min_jacobian > 0
    assert_array_equal(
        result.diagnostics.level_status, ["max_iterations", "max_iterations"]
    )


@pytest.mark.parametrize("moving_slice", [True, False])
@pytest.mark.parametrize("model", ["rigid", "affine"])
def test_mixed_pyramid_preserves_native_domain(
    slice_volume_pair_factory, moving_slice, model
):
    fixed, moving, truth = slice_volume_pair_factory(model)
    if moving_slice:
        fixed, moving, truth = moving, fixed, np.linalg.inv(truth)
    result = register(
        moving,
        fixed,
        transform=model,
        initial_transform=truth,
        shrink_factors=(3, 1),
        smoothing_sigmas=(0.2, 0),
        max_iterations=(2, 3),
        return_transforms=True,
        progress=False,
    )
    assert result.registered.shape == fixed.shape
    assert result.forward_transforms is not None
    sampled, valid = resample(moving, fixed, result.forward_transforms)
    assert_allclose(sampled, result.registered, rtol=0, atol=2e-6)
    assert_array_equal(valid, result.valid)


@pytest.mark.parametrize("model", ["rigid", "affine", "syn"])
@pytest.mark.parametrize("planar", [True, False])
def test_pyramid_moments_uses_original_images_and_preserves_failed_seed(
    linear_pair_factory, model, planar
):
    fixed, moving, _ = linear_pair_factory(planar, "translation")
    fixed = fixed.copy(data=np.rint(fixed.values * 1000).astype(np.uint16))
    moving = moving.copy(data=np.full(moving.shape, 500, dtype=np.uint16))
    indices = np.indices(fixed.shape).reshape(3, -1)
    center = (indices * fixed.values.ravel()).sum(axis=1) / fixed.values.sum()
    fixed_center = (position_to_world(fixed) @ np.r_[center, 1])[:3]
    moving_center = (
        position_to_world(moving) @ np.r_[(np.array(moving.shape) - 1) / 2, 1]
    )[:3]
    expected = moving_center - fixed_center
    result = register(
        moving,
        fixed,
        transform=model,
        initialization="center_moments",
        shrink_factors=(2, 1),
        smoothing_sigmas=(0.18, 0),
        max_iterations=(2, 2),
        return_transforms=True,
        progress=False,
    )
    assert result.forward_transforms is not None
    assert_array_equal(
        result.diagnostics.level_status, ["invalid_metric", "invalid_metric"]
    )
    assert result.diagnostics.n_iterations == 0
    assert result.registered.dtype == np.uint16
    if model == "syn":
        assert_allclose(
            result.forward_transforms,
            np.broadcast_to(expected, (*fixed.shape, 3)),
            rtol=0,
            atol=1e-12,
        )
    else:
        assert_allclose(
            result.forward_transforms.values[:3, 3], expected, rtol=0, atol=1e-12
        )
