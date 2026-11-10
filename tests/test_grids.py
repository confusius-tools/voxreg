"""Independent physical grids, 3D rotations and volume-to-volume registration."""

from dataclasses import replace

import numpy as np
import pytest
import torch
import xarray as xr
from registration_options import RegistrationOptions
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from voxeldata import create_voxeldata, validate_voxeldata

from voxreg import position_to_world, register
from voxreg.sampling import (
    SpatialGrid,
    build_rotations,
    build_world_matrices,
    make_grid,
    sample_rigid,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def grid_data(shape, angles, spacing, *, seed=3):
    affine = np.eye(4)
    affine[:3, :3] = Rotation.from_euler(
        "xyz", angles, degrees=True
    ).as_matrix() @ np.diag(spacing)
    affine[:3, 1] += 0.1 * affine[:3, 2]
    coords = {
        dim: 5 + (2 if axis else -2) * np.arange(size)
        for axis, (dim, size) in enumerate(zip(("k", "j", "i"), shape, strict=True))
    }
    center = np.array([np.mean(coords[dim]) for dim in ("k", "j", "i")])
    affine[:3, 3] = -affine[:3, :3] @ center
    return create_voxeldata(
        np.random.default_rng(seed).uniform(1, 3, shape),
        dims=("k", "j", "i"),
        voxel_to_world=affine,
        k=coords["k"],
        j=coords["j"],
        i=coords["i"],
    )


def pair(planar=False):
    fixed = grid_data(
        (1, 7, 9) if planar else (5, 7, 9),
        [0, 0, 0] if planar else [11, -17, 26],
        [0.3, 0.25, 0.2],
    )
    moving = grid_data(
        (1, 15, 17) if planar else (13, 15, 17),
        [0, 0, 0] if planar else [-9, 13, 70],
        [0.5, 0.4, 0.3],
        seed=9,
    )
    return fixed, moving


def reference_coordinates(moving, fixed, matrices):
    indices = np.indices(fixed.shape)
    homogeneous = np.concatenate((indices, np.ones((1, *fixed.shape))), axis=0).reshape(
        4, -1
    )
    mapped = (
        np.linalg.inv(position_to_world(moving))
        @ matrices
        @ position_to_world(fixed)
        @ homogeneous
    )
    valid = (
        (mapped[:3] >= -1e-5)
        & (mapped[:3] <= np.array(moving.shape)[:, None] - 1 + 1e-5)
    ).all(axis=0)
    return mapped[:3].reshape(3, *fixed.shape), valid.reshape(fixed.shape)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
@pytest.mark.parametrize("identity", [False, True])
def test_independent_grid_sampling_matches_scipy(
    device, planar, interpolation, identity
):
    fixed, moving = pair(planar)
    geometry = make_grid(fixed, moving, device=device)
    p = np.array(
        [0.07, 0.1, -0.08] if planar else [0.03, -0.02, 0.04, 0.05, -0.08, 0.06]
    )
    if identity:
        p[:] = 0
    matrix = build_world_matrices(p[None], geometry)[0]
    if identity:
        np.testing.assert_array_equal(matrix, np.eye(4))
    coords, expected_valid = reference_coordinates(moving, fixed, matrix)
    expected = map_coordinates(
        moving.values,
        coords,
        order=1 if interpolation == "linear" else 0,
        mode="nearest",
        prefilter=False,
    )
    values = moving.values[None, 0] if planar else moving.values[None]
    actual, valid = sample_rigid(
        torch.tensor(values, device=device),
        torch.tensor(p[None], device=device),
        geometry,
        interpolation=interpolation,
    )
    np.testing.assert_allclose(
        actual.cpu()[0], expected[0] if planar else expected, atol=2e-12, rtol=0
    )
    np.testing.assert_array_equal(
        valid.cpu()[0], expected_valid[0] if planar else expected_valid
    )
    assert expected_valid.all()


@pytest.mark.parametrize("zero", [False, True])
def test_volume_sampler_gradients_including_zero_rotation(zero):
    fixed, moving = pair()
    geometry = make_grid(fixed, moving)
    values = torch.tensor(moving.values[None])
    p = torch.tensor(
        [[0.03, -0.02, 0.04, 0.05, -0.08, 0.06]],
        requires_grad=True,
        dtype=torch.float64,
    )
    if zero:
        # Keep translations off interpolation knots while checking the zero-angle limit.
        p = torch.tensor(
            [[0, 0, 0, 0.05, -0.08, 0.06]], dtype=torch.float64, requires_grad=True
        )
    weights = torch.linspace(0.1, 1, fixed.size, dtype=torch.float64).reshape(
        fixed.shape
    )

    def objective(parameters):
        sampled, _ = sample_rigid(values, parameters, geometry)
        return (sampled * weights).sum()

    assert torch.autograd.gradcheck(objective, (p,), eps=1e-6, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("image_grad", [False, True])
@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
def test_cached_cpu_volume_sampler_matches_untiled_geometry(
    batch, dtype, image_grad, interpolation
):
    fixed, moving = pair()
    geometry = make_grid(fixed, moving.expand_dims(time=np.arange(batch)))
    assert isinstance(geometry, SpatialGrid)
    cached = (
        geometry.resident_coordinates.clone()
        if geometry.resident_coordinates is not None
        else None
    )
    assert (cached is not None) == (batch > 1)
    images = torch.tensor(
        np.repeat(moving.values[None], batch, axis=0),
        dtype=dtype,
        requires_grad=image_grad,
    )
    parameters = torch.tensor(
        [[0.03, -0.02, 0.04, 0.05, -0.08, 0.06]] * batch, dtype=torch.float64
    )
    if batch > 1:
        parameters[-1, 3] = 100  # Include unsupported points and border extension.
    parameters.requires_grad_()
    inputs = (parameters, images) if image_grad else (parameters,)
    weights = torch.linspace(0.1, 1, fixed.size).reshape(fixed.shape)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected, expected_valid = sample_rigid(
            images,
            parameters,
            replace(geometry, resident_coordinates=None),
            interpolation=interpolation,
        )
        expected_gradients = torch.autograd.grad((expected * weights).sum(), inputs)
        torch.set_num_threads(4)
        actual, valid = sample_rigid(
            images, parameters, geometry, interpolation=interpolation
        )
        gradients = torch.autograd.grad((actual * weights).sum(), inputs)
    finally:
        torch.set_num_threads(previous)
    torch.testing.assert_close(
        actual, expected, rtol=0, atol=2e-6 if dtype == torch.float32 else 2e-12
    )
    assert torch.equal(valid, expected_valid)
    if interpolation == "nearest":
        assert torch.equal(actual, expected)
    for actual_gradient, expected_gradient in zip(
        gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=2e-6,
            atol=2e-5 if dtype == torch.float32 else 2e-11,
        )
    if geometry.resident_coordinates is not None:
        assert cached is not None
        assert torch.equal(geometry.resident_coordinates, cached)


def test_volume_parameter_scales_use_per_axis_physical_lever_arms():
    data = create_voxeldata(
        np.arange(27, dtype=float).reshape(3, 3, 3),
        dims=("k", "j", "i"),
        voxel_to_world=np.diag([1e10, 1.0, 1.0, 1.0]),
    )
    geometry = make_grid(data, data)
    expected = np.r_[
        1 / np.sqrt(np.array([2.0, 1e20 + 1, 1e20 + 1]) * 2 / 3), [1.0] * 3
    ]
    np.testing.assert_allclose(geometry.parameter_scale, expected, rtol=1e-14)


def test_rotation_vectors_use_physical_xyz_handedness():
    vectors = np.array([[0.2, -0.1, 0.3], [0, 0, 0]])
    expected = Rotation.from_rotvec(vectors[:, ::-1]).as_matrix()[:, ::-1, ::-1]
    np.testing.assert_allclose(
        build_rotations(torch.tensor(vectors)),
        expected,
        atol=1e-15,
    )
    assert torch.autograd.gradcheck(
        build_rotations,
        (torch.zeros((1, 3), dtype=torch.float64, requires_grad=True),),
    )


@pytest.mark.parametrize("dtype", [np.float64, np.uint16])
@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
@pytest.mark.parametrize("device", DEVICES)
def test_single_pair_output_uses_fixed_grid_and_original_dtype(
    linear_pair_factory, dtype, planar, model, device
):
    fixed, moving, _ = linear_pair_factory(planar, model)
    moving = moving.copy(data=(moving.values * 50).astype(dtype))
    moving.name = "signal"
    moving.attrs.update(units="power", subject="moving")
    fixed.attrs["subject"] = "fixed"
    untouched = moving.copy(deep=True)
    fixed_original = fixed.copy(deep=True)
    r = register(
        moving,
        fixed,
        return_transforms=True,
        transform=model,
        device=device,
        max_iterations=1,
        progress=False,
    )
    assert "forward" in r
    assert (
        r["images"].to_dataset()["registered"].dims == fixed.dims
        and r["images"].to_dataset()["registered"].shape == fixed.shape
    )
    assert r["images"].to_dataset()["registered"].dtype == dtype
    assert (
        r["images"].to_dataset()["registered"].xindexes["z"].equals(fixed.xindexes["z"])
    )
    assert r["forward"].to_dataset()["transform"].shape == (4, 4)
    assert r["diagnostics"].to_dataset().status.dims == ()
    validate_voxeldata(r["images"].to_dataset()["registered"])
    validate_voxeldata(r["images"].to_dataset()["valid"])
    xr.testing.assert_identical(moving, untouched)
    xr.testing.assert_identical(fixed, fixed_original)
    assert r["images"].to_dataset()["registered"].name == "registered"
    assert (
        r["images"].to_dataset()["registered"].attrs["units"] == "power"
        and r["images"].to_dataset()["registered"].attrs["subject"] == "moving"
    )
    coords, valid = reference_coordinates(
        moving, fixed, r["forward"].to_dataset()["transform"].values
    )
    expected = map_coordinates(
        moving.values.astype(float), coords, order=1, mode="nearest", prefilter=False
    )
    if dtype == np.uint16:
        expected = np.rint(expected).astype(dtype)
    np.testing.assert_allclose(
        r["images"].to_dataset()["registered"], expected, atol=2e-12, rtol=0
    )
    np.testing.assert_array_equal(r["images"].to_dataset()["valid"], valid)


def volume_phantom():
    # More samples than the interpolation tests, with unequal anisotropic grids.
    fixed = grid_data((17, 19, 21), [11, -17, 26], [0.3, 0.25, 0.2])
    moving = grid_data((29, 31, 33), [-9, 13, 20], [0.3, 0.25, 0.2])
    center = position_to_world(fixed)[:3, 3] + position_to_world(fixed)[:3, :3] @ (
        (np.array(fixed.shape) - 1) / 2
    )
    p = np.array([0.02, -0.015, 0.01, 0.11, -0.08, 0.09])
    rotation = Rotation.from_rotvec(p[:3][::-1]).as_matrix()[::-1, ::-1]
    truth = np.eye(4)
    truth[:3, :3] = rotation
    truth[:3, 3] = center + p[3:] - rotation @ center
    rng = np.random.default_rng(32)
    locations = rng.uniform(-1.6, 1.6, (16, 3))
    widths = rng.uniform(0.4, 0.8, (16, 3))
    amplitudes = rng.uniform(0.5, 2, 16)

    def paint(data, inverse):
        indices = np.indices(data.shape)
        points = (
            inverse
            @ position_to_world(data)
            @ np.concatenate((indices, np.ones((1, *data.shape))), axis=0).reshape(
                4, -1
            )
        )[:3].T
        values = 0.05 + np.sum(
            amplitudes
            * np.exp(
                -0.5 * np.sum(((points[:, None] - locations) / widths) ** 2, axis=-1)
            ),
            axis=1,
        )
        return data.copy(data=values.reshape(data.shape).astype(np.float32))

    return paint(fixed, np.eye(4)), paint(moving, np.linalg.inv(truth)), truth


@pytest.mark.parametrize("device", DEVICES)
def test_3d_known_motion_recovery_on_independent_grids(device):
    fixed, moving, truth = volume_phantom()
    r = register(
        moving,
        fixed,
        return_transforms=True,
        device=device,
        max_iterations=180,
        progress=False,
    )
    assert "forward" in r
    indices = np.indices(fixed.shape)
    points = position_to_world(fixed) @ np.concatenate(
        (indices, np.ones((1, *fixed.shape))), axis=0
    ).reshape(4, -1)
    error = ((r["forward"].to_dataset()["transform"].values - truth) @ points)[:3]
    ell = np.linalg.norm(position_to_world(fixed)[:3, :3], axis=0).min()
    rms = np.sqrt(np.mean(np.sum(error**2, axis=0))) / ell
    assert rms < 0.25, (rms, r["diagnostics"].to_dataset())
    assert r["diagnostics"].to_dataset().objective < -0.995
    assert r["diagnostics"].to_dataset().valid_fraction > 0.99
    assert {"rotation_z", "rotation_y", "rotation_x", "translation_z"} <= set(
        r["diagnostics"].to_dataset()
    )


@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
def test_3d_batch_and_cpu_cuda_independence(linear_pair_factory, model):
    fixed, moving, truth = linear_pair_factory(False, model)
    data = create_voxeldata(
        np.stack([moving.values, moving.values]),
        dims=("time", *moving.dims),
        time=[2.0, 4.0],
        voxel_to_world=moving.xindexes["z"].voxel_to_world,
        **{dim: moving[dim] for dim in moving.dims},
    )
    data = data.assign_coords(acquisition=("time", [70, 73]))
    options: RegistrationOptions = {
        "transform": model,
        "max_iterations": 120,
        "progress": False,
    }
    # Keep the strict CPU batch check separate from CUDA agreement below.
    together = register(
        data, fixed, return_transforms=True, batch_size=2, device="cpu", **options
    )
    assert "forward" in together
    alone = register(
        data, fixed, return_transforms=True, batch_size=1, device="cpu", **options
    )
    assert "forward" in alone
    np.testing.assert_allclose(
        together["forward"].to_dataset()["transform"],
        alone["forward"].to_dataset()["transform"],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_array_equal(
        together["diagnostics"].to_dataset().n_iterations,
        alone["diagnostics"].to_dataset().n_iterations,
    )
    assert together["images"].to_dataset()["registered"].shape == (2, *fixed.shape)
    np.testing.assert_array_equal(
        together["images"].to_dataset()["registered"].time, data.time
    )
    np.testing.assert_array_equal(
        together["images"].to_dataset()["registered"].acquisition, data.acquisition
    )
    validate_voxeldata(together["images"].to_dataset()["registered"])
    validate_voxeldata(together["images"].to_dataset()["valid"])
    points = position_to_world(fixed) @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size)))
    )
    ell = np.linalg.norm(position_to_world(fixed)[:3, :3], axis=0).min()
    truth_error = np.einsum(
        "tij,jp->tip",
        together["forward"].to_dataset()["transform"].values - truth,
        points,
    )[:, :3]
    assert np.sqrt(np.mean(np.sum(truth_error**2, axis=1))) / ell < 0.25
    if torch.cuda.is_available():
        cuda = register(data, fixed, return_transforms=True, device="cuda", **options)
        assert "forward" in cuda
        indices = np.indices(fixed.shape)
        points = position_to_world(fixed) @ np.concatenate(
            (indices, np.ones((1, *fixed.shape))), axis=0
        ).reshape(4, -1)
        error = np.einsum(
            "tij,jp->tip",
            cuda["forward"].to_dataset()["transform"].values
            - together["forward"].to_dataset()["transform"].values,
            points,
        )[:, :3]
        ell = np.linalg.norm(position_to_world(fixed)[:3, :3], axis=0).min()
        assert np.linalg.norm(error, axis=1).max() < 0.01 * ell
        np.testing.assert_array_equal(
            cuda["diagnostics"].to_dataset().n_iterations,
            together["diagnostics"].to_dataset().n_iterations,
        )


@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
def test_3d_reference_and_float64_detail_along_k_are_preserved(model):
    values = np.broadcast_to(
        2.0**30 + 1e-3 * np.arange(5)[:, None, None], (5, 6, 7)
    ).copy()
    data = create_voxeldata(
        np.stack([values, values]),
        dims=("time", "k", "j", "i"),
        dt=1,
        voxel_to_world=np.eye(4),
    )
    r = register(
        data,
        return_transforms=True,
        transform=model,
        max_iterations=1,
        progress=False,
    )
    assert "forward" in r
    assert r["images"].to_dataset()["registered"].dtype == np.float64
    np.testing.assert_array_equal(r["images"].to_dataset()["registered"][0], data[0])
    np.testing.assert_array_equal(r["forward"].to_dataset()["transform"][0], np.eye(4))
    assert r["diagnostics"].to_dataset().status.values[0] == "reference"
    assert r["diagnostics"].to_dataset().objective.values[1] < -0.99999
    assert r["diagnostics"].to_dataset().status.values[1] != "invalid_metric"


def test_common_large_world_origin_does_not_enter_dense_mapping():
    fixed, moving = pair()
    p = torch.tensor([[0.03, -0.02, 0.04, 0.05, -0.08, 0.06]], dtype=torch.float64)
    expected, expected_valid = sample_rigid(
        torch.tensor(moving.values[None]), p, make_grid(fixed, moving)
    )
    shifted = []
    shift = 2.0**26
    for data in (fixed, moving):
        affine = data.xindexes["z"].voxel_to_world.copy()
        affine[:3, 3] += shift
        shifted.append(
            create_voxeldata(
                data.values,
                dims=data.dims,
                voxel_to_world=affine,
                **{dim: data[dim] for dim in data.dims},
            )
        )
    actual, valid = sample_rigid(
        torch.tensor(moving.values[None]), p, make_grid(*shifted)
    )
    np.testing.assert_allclose(actual, expected, atol=16 * np.spacing(shift), rtol=0)
    np.testing.assert_array_equal(valid, expected_valid)


def test_no_overlap_retains_identity_and_uses_original_minimum():
    fixed, moving = pair()
    affine = moving.xindexes["z"].voxel_to_world.copy()
    affine[:3, 3] += 100
    moving = create_voxeldata(
        moving.values,
        dims=moving.dims,
        **{dim: moving[dim] for dim in moving.dims},
        voxel_to_world=affine,
    )
    r = register(
        moving, fixed, return_transforms=True, max_iterations=2, progress=False
    )
    assert r["diagnostics"].to_dataset().status == "insufficient_overlap"
    assert r["diagnostics"].to_dataset().n_iterations == 0
    assert np.isnan(r["diagnostics"].to_dataset().objective)
    assert not r["images"].to_dataset()["valid"].any()
    np.testing.assert_array_equal(r["forward"].to_dataset()["transform"], np.eye(4))
    np.testing.assert_array_equal(
        r["images"].to_dataset()["registered"],
        np.full(fixed.shape, moving.values.min()),
    )


def test_non_coplanar_needs_seed_and_mixed_grids_use_slice_domain():
    fixed, _ = pair(True)
    tilted = grid_data((1, 15, 17), [0, 20, 0], [0.5, 0.4, 0.3])
    with pytest.raises(ValueError, match="coplanar"):
        register(tilted, fixed, progress=False)
    volume = grid_data((3, 15, 17), [0, 0, 0], [0.5, 0.4, 0.3])
    result = register(volume, fixed, max_iterations=1, progress=False)
    assert result["images"].to_dataset()["registered"].shape == fixed.shape
    assert result["diagnostics"].to_dataset().attrs["metric_domain"] == "fixed_slice"
