"""3D slice placement with affine-defined singleton voxel extent."""

from typing import Literal

import numpy as np
import pytest
from registration_options import RegistrationOptions
import torch
from confusius import create_voxeldata
from numpy.testing import assert_allclose, assert_array_equal
from oracles import sample_mixed as sample
from oracles import voxel
from scipy.spatial.transform import Rotation

from voxreg import position_to_world, register
from voxreg.sampling import _initial_parameters, make_grid, sample_rigid, world_matrices
from voxreg.transforms import configure_transform

DEVICES: list[Literal["cpu", "cuda"]] = (
    ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)


@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
@pytest.mark.parametrize("device", DEVICES)
def test_slice_volume_directions_use_their_own_native_fixed_grid(
    slice_volume_pair_factory, model, device
):
    target, volume, truth = slice_volume_pair_factory(model)
    kwargs: RegistrationOptions = {
        "transform": model,
        "intensity_scaling": "none",
        "max_iterations": 400,
        "convergence_window_size": 401,
        "device": device,
        "progress": False,
    }
    result = register(volume, target, return_transforms=True, **kwargs)
    assert result.transforms is not None
    reverse = register(target, volume, return_transforms=True, **kwargs)
    assert reverse.transforms is not None
    assert result.registered.shape == target.shape
    assert reverse.registered.shape == volume.shape
    assert result.registered.dtype == reverse.registered.dtype == np.float64
    assert result.diagnostics.valid_fraction > 0.95
    assert not np.allclose(
        reverse.transforms, np.linalg.inv(result.transforms), atol=1e-6, rtol=0
    )
    assert result.diagnostics.attrs["metric_domain"] == "fixed_slice"
    assert reverse.diagnostics.attrs["metric_domain"] == "fixed_volume"
    points = position_to_world(target) @ np.vstack(
        (np.indices(target.shape).reshape(3, -1), np.ones((1, target.size)))
    )
    errors = (
        np.linalg.norm(((result.transforms.values - truth) @ points)[:3], axis=0) / 0.35
    )
    assert np.sqrt(np.mean(errors**2)) < 0.25
    expected, valid = sample(volume, target, result.transforms.values)
    assert_array_equal(result.valid, valid)
    assert_allclose(
        result.registered.values[valid], expected[valid], atol=1e-12, rtol=0
    )
    expected, valid = sample(target, volume, reverse.transforms.values)
    assert_array_equal(reverse.valid, valid)
    assert_allclose(
        reverse.registered.values[valid], expected[valid], atol=1e-12, rtol=0
    )
    assert np.all(reverse.registered.values[~valid] == target.values.min())
    assert_allclose(reverse.diagnostics.valid_fraction, valid.mean(), atol=1e-8)
    for fitted in [result, reverse]:
        assert fitted.transforms is not None
        center = np.asarray(fitted.diagnostics.attrs["rotation_center_zyx_mm"])
        matrix = fitted.transforms.values
        shift = matrix[:3, 3] + (matrix[:3, :3] - np.eye(3)) @ center
        assert_allclose(
            [float(fitted.diagnostics[f"translation_{axis}"]) for axis in "zyx"],
            shift,
            atol=1e-12,
            rtol=0,
        )
        if model == "affine":
            actual = np.array(
                [
                    [float(fitted.diagnostics[f"linear_{row}{col}"]) for col in "zyx"]
                    for row in "zyx"
                ]
            )
            assert_allclose(actual, matrix[:3, :3], atol=1e-15, rtol=0)
        if model == "rigid":
            vector = np.array(
                [float(fitted.diagnostics[f"rotation_{axis}"]) for axis in "zyx"]
            )
            actual = Rotation.from_rotvec(vector[::-1]).as_matrix()[::-1, ::-1]
            assert_allclose(actual, matrix[:3, :3], atol=1e-14, rtol=0)


@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
def test_slice_volume_cpu_cuda_unchanged_precision_gate(
    slice_volume_pair_factory, model
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    target, volume, _ = slice_volume_pair_factory(model)
    kwargs: RegistrationOptions = {
        "transform": model,
        "intensity_scaling": "none",
        "max_iterations": 400,
        "convergence_window_size": 401,
        "progress": False,
    }
    a, b = (
        register(volume, target, return_transforms=True, device=device, **kwargs)
        for device in DEVICES
    )
    assert a.transforms is not None and b.transforms is not None
    points = position_to_world(target) @ np.vstack(
        (np.indices(target.shape).reshape(3, -1), np.ones((1, target.size)))
    )
    difference = (
        np.linalg.norm(
            ((a.transforms.values - b.transforms.values) @ points)[:3], axis=0
        )
        / 0.35
    )
    assert difference.max() < 0.01


@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
def test_mixed_seed_roundtrip_and_sampler(slice_volume_pair_factory, model):
    target, volume, truth = slice_volume_pair_factory(model)
    grid = configure_transform(make_grid(target, volume), target, model)
    parameters = _initial_parameters(truth, grid, 1)
    assert_allclose(world_matrices(parameters, grid)[0], truth, atol=1e-14, rtol=0)
    images = torch.tensor(volume.values[None])
    state = torch.tensor(parameters, requires_grad=True)
    sampled, valid = sample_rigid(images, state, grid)
    assert valid.all()
    assert_allclose(sampled.detach().numpy()[0], target.values, atol=1e-13, rtol=0)
    gradient = torch.autograd.grad(sampled.square().sum(), state)[0].numpy()[0]
    for axis in range(state.shape[1]):
        step = np.zeros_like(parameters)
        step[0, axis] = 1e-7
        upper = (
            sample_rigid(images, torch.tensor(parameters + step), grid)[0]
            .square()
            .sum()
            .item()
        )
        lower = (
            sample_rigid(images, torch.tensor(parameters - step), grid)[0]
            .square()
            .sum()
            .item()
        )
        assert_allclose(gradient[axis], (upper - lower) / 2e-7, atol=2e-6, rtol=1e-5)


def test_affine_slice_has_nine_parameters_not_twelve(slice_volume_pair_factory):
    target, volume, truth = slice_volume_pair_factory("affine")
    grid = configure_transform(make_grid(target, volume), target, "affine")
    assert len(grid.parameter_scale) == 9
    bad = truth.copy()
    bad[:3, 0] *= 2
    with pytest.raises(ValueError, match="unidentifiable normal"):
        _initial_parameters(bad, grid, 1)
    collapsed = np.zeros((1, 9))
    collapsed[0, :6] = -grid.slice_basis.reshape(-1)
    from voxreg.transforms import affine_properties

    assert not affine_properties(torch.tensor(collapsed), grid)[0].any()


@pytest.mark.parametrize("model", ["rigid", "affine"])
@pytest.mark.parametrize("mode", ["align_planes", "matrix"])
def test_nonparallel_slice_planes_need_an_explicit_alignment_choice(model, mode):
    rng = np.random.default_rng(4)
    af = np.diag([0.4, 0.3, 0.35, 1.0])
    af[:3, 3] = [0, -2, -2]
    fixed = voxel(rng.uniform(0.1, 2, (1, 13, 15)), af)
    rotation = Rotation.from_rotvec([0, 0.35, 0]).as_matrix()
    center = position_to_world(fixed) @ np.r_[(np.array(fixed.shape) - 1) / 2, 1]
    truth = np.eye(4)
    truth[:3, :3] = rotation
    truth[:3, 3] = [2, -1, 3] + center[:3] - rotation @ center[:3]
    moving = voxel(fixed.values.copy(), truth @ af)
    with pytest.raises(ValueError, match="Noncoplanar"):
        register(
            moving,
            fixed,
            return_transforms=True,
            intensity_scaling="none",
            transform=model,
            progress=False,
        )
    with pytest.raises(ValueError, match="preserve"):
        register(
            moving,
            fixed,
            return_transforms=True,
            intensity_scaling="none",
            transform=model,
            initialization="center_geometry",
            progress=False,
        )
    result = register(
        moving,
        fixed,
        return_transforms=True,
        intensity_scaling="none",
        transform=model,
        initialization="align_planes" if mode == "align_planes" else truth,
        max_iterations=1,
        progress=False,
    )
    assert result.transforms is not None
    assert_allclose(result.transforms, truth, atol=1e-12, rtol=0)
    assert result.diagnostics.objective < -0.999999
    assert_allclose(
        result.registered.values[result.valid.values],
        fixed.values[result.valid.values],
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
@pytest.mark.parametrize("device", DEVICES)
def test_batched_slice_targets_seeds_failures_and_chunk_independence(
    slice_volume_pair_factory, model, device
):
    target, volume, truth = slice_volume_pair_factory(model)
    second = truth.copy()
    second[:3, 3] += [0.03, 0.02, -0.01]
    values, _ = sample(volume, target, second)
    movie = create_voxeldata(
        np.stack((target.values, values, np.ones_like(values))),
        dims=("time", "k", "j", "i"),
        time=[3, 7, 11],
        voxel_to_world=position_to_world(target),
    )
    seeds = np.linalg.inv(np.stack((truth, second, truth)))
    original = movie.values.copy()
    kwargs: RegistrationOptions = {
        "transform": model,
        "intensity_scaling": "none",
        "initialization": seeds,
        "max_iterations": 3,
        "convergence_window_size": 4,
        "device": device,
        "progress": False,
    }
    a = register(movie, volume, return_transforms=True, batch_size=1, **kwargs)
    assert a.transforms is not None
    b = register(movie, volume, return_transforms=True, batch_size=3, **kwargs)
    assert b.transforms is not None
    assert list(a.diagnostics.status.values) == [
        "max_iterations",
        "max_iterations",
        "invalid_metric",
    ]
    assert list(a.diagnostics.n_iterations.values) == [3, 3, 0]
    assert_allclose(a.transforms, b.transforms, atol=1e-10, rtol=0)
    assert_allclose(
        a.diagnostics.objective,
        b.diagnostics.objective,
        atol=4 * np.finfo(np.float32).eps,
        rtol=0,
        equal_nan=True,
    )
    assert_allclose(a.transforms.values[2], seeds[2], atol=1e-9, rtol=0)
    assert_array_equal(movie.values, original)
    assert_array_equal(seeds, np.linalg.inv(np.stack((truth, second, truth))))


@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
def test_mixed_no_overlap_retains_public_seed(slice_volume_pair_factory, model):
    target, volume, _ = slice_volume_pair_factory(model)
    seed = np.eye(4)
    seed[0, 3] = 100
    for moving, fixed in [(volume, target), (target, volume)]:
        result = register(
            moving,
            fixed,
            return_transforms=True,
            transform=model,
            initialization=seed,
            intensity_scaling="none",
            max_iterations=2,
            progress=False,
        )
        assert result.transforms is not None
        assert result.diagnostics.status == "insufficient_overlap"
        assert result.diagnostics.n_iterations == 0
        assert_allclose(result.transforms, seed, atol=1e-12, rtol=0)
        assert not result.valid.any()
        assert np.isnan(result.diagnostics.objective)


@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
def test_reversed_strided_slice_and_volume_grids(slice_volume_pair_factory, model):
    target, volume, truth = slice_volume_pair_factory(model)
    target = target.isel(j=slice(None, None, 2), i=slice(None, None, -1))
    volume = volume.isel(k=slice(None, None, -1), j=slice(None, None, -1))
    result = register(
        volume,
        target,
        return_transforms=True,
        transform=model,
        initialization=truth,
        intensity_scaling="none",
        max_iterations=1,
        progress=False,
    )
    assert result.transforms is not None
    expected, valid = sample(volume, target, result.transforms.values)
    assert_array_equal(result.valid, valid)
    assert_allclose(
        result.registered.values[valid], expected[valid], atol=1e-12, rtol=0
    )
    assert result.diagnostics.objective < -0.999999


@pytest.mark.parametrize("angle", [2.8, np.pi - 5e-7, np.pi])
def test_plane_seed_can_select_the_opposite_normal_orientation(angle):
    af = np.eye(4)
    fixed = voxel(np.random.default_rng(22).uniform(0.1, 2, (1, 9, 11)), af)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0, angle, 0]).as_matrix()
    pose[:3, 3] = [2, 3, 4]
    moving = voxel(fixed.values, pose @ af)
    result = register(
        moving,
        fixed,
        return_transforms=True,
        initialization=pose,
        intensity_scaling="none",
        max_iterations=1,
        progress=False,
    )
    assert result.transforms is not None
    assert_allclose(result.transforms, pose, atol=1e-12, rtol=0)
    assert result.diagnostics.objective < -0.999999


@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
def test_oblique_mixed_grid_seed_and_native_itk_point_oracle(
    slice_volume_pair_factory, model
):
    import itk

    target, volume, truth = slice_volume_pair_factory(model)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0.3, -0.25, 0.1]).as_matrix()
    pose[:3, 3] = [1000, -2000, 3000]
    target = voxel(target.values, pose @ position_to_world(target))
    volume = voxel(volume.values, pose @ position_to_world(volume))
    truth = pose @ truth @ np.linalg.inv(pose)
    grid = configure_transform(make_grid(target, volume), target, model)
    parameters = _initial_parameters(truth, grid, 1)
    actual = world_matrices(parameters, grid)[0]
    native = getattr(itk, "AffineTransform")[itk.D, 3].New()
    native.SetMatrix(itk.matrix_from_array(actual[:3, :3]))
    native.SetOffset(tuple(actual[:3, 3]))
    points = position_to_world(target) @ np.vstack(
        (np.indices(target.shape).reshape(3, -1), np.ones((1, target.size)))
    )
    expected = np.array(
        [native.TransformPoint(tuple(point[:3])) for point in points.T]
    ).T
    assert_allclose((actual @ points)[:3], expected, atol=1e-12, rtol=0)
    result = register(
        volume,
        target,
        return_transforms=True,
        transform=model,
        initialization=truth,
        intensity_scaling="none",
        max_iterations=1,
        progress=False,
    )
    assert result.transforms is not None
    expected, valid = sample(volume, target, result.transforms.values)
    assert_allclose(
        result.registered.values[valid], expected[valid], atol=2e-11, rtol=0
    )
    assert result.diagnostics.objective < -0.999999


def test_integer_mixed_output_retains_rounding_and_original_dtype(
    slice_volume_pair_factory,
):
    target, volume, truth = slice_volume_pair_factory("rigid")
    integer = voxel(
        np.rint(volume.values * 5000).astype(np.uint16), position_to_world(volume)
    )
    values, valid = sample(integer.astype(float), target, truth)
    target = voxel(values, position_to_world(target))
    original = integer.values.copy()
    result = register(
        integer,
        target,
        return_transforms=True,
        initialization=truth,
        intensity_scaling="none",
        max_iterations=1,
        progress=False,
    )
    assert result.transforms is not None
    assert result.registered.dtype == np.uint16
    values, valid = sample(integer.astype(float), target, result.transforms.values)
    assert_array_equal(
        result.registered.values[valid], np.rint(values[valid]).astype(np.uint16)
    )
    assert_array_equal(integer.values, original)


def test_plane_initializer_scope_and_translation_rotation_validation(
    slice_volume_pair_factory,
):
    target, volume, _ = slice_volume_pair_factory("rigid")
    for moving, fixed in [(volume, target), (target, volume), (volume, volume)]:
        with pytest.raises(ValueError, match="two k=1"):
            register(moving, fixed, initialization="align_planes", progress=False)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0, 0.3, 0]).as_matrix()
    tilted = voxel(target.values, pose @ position_to_world(target))
    with pytest.raises(ValueError, match="choose rigid or affine"):
        register(
            tilted,
            target,
            transform="translation",
            initialization="align_planes",
            progress=False,
        )


@pytest.mark.parametrize("mode", ["center_geometry", "center_moments"])
def test_mixed_centering_seed_is_direction_consistent(
    slice_volume_pair_factory, mode, monkeypatch
):
    from voxreg import rigid

    def seed_only(
        images, target, grid, reference, *args, initial_parameters=None, **kwargs
    ):
        return (
            initial_parameters,
            torch.zeros(len(images), dtype=torch.int64, device=images.device),
            torch.full((len(images),), 5, dtype=torch.int64, device=images.device),
        )

    monkeypatch.setattr(rigid, "_optimize", seed_only)
    target, volume, _ = slice_volume_pair_factory("rigid")
    kwargs: RegistrationOptions = {
        "initialization": mode,
        "intensity_scaling": "none",
        "max_iterations": 2,
        "convergence_window_size": 3,
        "progress": False,
    }
    a = register(volume, target, return_transforms=True, **kwargs)
    assert a.transforms is not None
    b = register(target, volume, return_transforms=True, **kwargs)
    assert b.transforms is not None
    assert_allclose(a.transforms, np.linalg.inv(b.transforms), atol=1e-12, rtol=0)


def test_volume_grid_centers_within_singleton_voxel_extent_are_valid(
    slice_volume_pair_factory,
):
    target, volume, _ = slice_volume_pair_factory("translation")
    values, _ = sample(volume, target, np.eye(4))
    target = voxel(values, position_to_world(target))
    result = register(
        target,
        volume,
        return_transforms=True,
        intensity_scaling="none",
        max_iterations=1,
        progress=False,
    )
    assert result.transforms is not None
    assert result.valid.any()
    assert not result.valid.values[:12].any()
    assert not result.valid.values[13:].any()
    assert result.valid.values[12].any()
    expected, valid = sample(target, volume, result.transforms.values)
    assert_array_equal(result.valid, valid)
    assert_allclose(
        result.registered.values[valid], expected[valid], atol=1e-12, rtol=0
    )


def test_parallel_plane_centering_and_translation_model():
    rng = np.random.default_rng(4)
    fixed = voxel(rng.uniform(0.1, 2, (1, 13, 15)), np.eye(4))
    af = np.eye(4)
    af[:3, 3] = [4, 2, 3]
    moving = voxel(fixed.values.copy(), af)
    result = register(
        moving,
        fixed,
        return_transforms=True,
        transform="translation",
        initialization="center_geometry",
        intensity_scaling="none",
        max_iterations=1,
        progress=False,
    )
    assert result.transforms is not None
    assert_allclose(result.transforms, af, atol=1e-12, rtol=0)
    assert result.diagnostics.objective < -0.999999
