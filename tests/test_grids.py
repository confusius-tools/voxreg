"""Independent physical grids, 3D rotations and volume-to-volume registration."""

import numpy as np
import pytest
import torch
from confusius import create_voxeldata
from confusius.validation import validate_voxeldata
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from voxreg import position_to_world, register_rigid
from voxreg.sampling import make_grid, rotation_vectors, sample_rigid, world_matrices

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
        **coords,
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
    matrix = world_matrices(p[None], geometry)[0]
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
        rotation_vectors(torch.tensor(vectors)), expected, atol=1e-15
    )
    assert torch.autograd.gradcheck(
        rotation_vectors,
        (torch.zeros((1, 3), dtype=torch.float64, requires_grad=True),),
    )


@pytest.mark.parametrize("dtype", [np.float64, np.uint16])
@pytest.mark.parametrize("planar", [False, True])
def test_single_pair_output_uses_fixed_grid_and_original_dtype(
    monkeypatch, dtype, planar
):
    from voxreg import rigid

    fixed, moving = pair(planar)
    moving = moving.copy(data=(moving.values * 50).astype(dtype))
    untouched = moving.values.copy()

    def identity(images, fixed, plane, reference, *args, **kwargs):
        return (
            plane.parameter_scale.new_zeros((len(images), len(plane.parameter_scale))),
            torch.zeros(len(images), dtype=torch.int64),
            torch.ones(len(images), dtype=torch.int64),
        )

    monkeypatch.setattr(rigid, "_optimize", identity)
    r = register_rigid(moving, fixed, intensity_scaling="none", progress=False)
    assert r.registered.dims == fixed.dims and r.registered.shape == fixed.shape
    assert r.registered.dtype == dtype
    assert r.registered.xindexes["z"].equals(fixed.xindexes["z"])
    assert r.transforms.shape == (4, 4)
    assert r.diagnostics.status.dims == ()
    validate_voxeldata(r.registered)
    validate_voxeldata(r.valid)
    np.testing.assert_array_equal(moving, untouched)
    coords, valid = reference_coordinates(moving, fixed, np.eye(4))
    expected = map_coordinates(
        moving.values.astype(float), coords, order=1, mode="nearest", prefilter=False
    )
    if dtype == np.uint16:
        expected = np.rint(expected).astype(dtype)
    np.testing.assert_allclose(r.registered, expected, atol=2e-12, rtol=0)
    np.testing.assert_array_equal(r.valid, valid)
    np.testing.assert_array_equal(r.transforms, np.eye(4))


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
    r = register_rigid(
        moving,
        fixed,
        device=device,
        intensity_scaling="none",
        max_iterations=180,
        progress=False,
    )
    indices = np.indices(fixed.shape)
    points = position_to_world(fixed) @ np.concatenate(
        (indices, np.ones((1, *fixed.shape))), axis=0
    ).reshape(4, -1)
    error = ((r.transforms.values - truth) @ points)[:3]
    ell = np.linalg.norm(position_to_world(fixed)[:3, :3], axis=0).min()
    rms = np.sqrt(np.mean(np.sum(error**2, axis=0))) / ell
    assert rms < 0.25, (rms, r.diagnostics)
    assert r.diagnostics.objective < -0.995
    assert r.diagnostics.valid_fraction > 0.99
    assert {"rotation_z", "rotation_y", "rotation_x", "translation_z"} <= set(
        r.diagnostics
    )


def test_3d_batch_and_cpu_cuda_independence():
    fixed, moving, _ = volume_phantom()
    data = create_voxeldata(
        np.stack([moving.values, moving.values]),
        dims=("time", *moving.dims),
        time=[2.0, 4.0],
        voxel_to_world=moving.xindexes["z"].voxel_to_world,
        **{dim: moving[dim] for dim in moving.dims},
    )
    data = data.assign_coords(acquisition=("time", [70, 73]))
    options = {"intensity_scaling": "none", "max_iterations": 120, "progress": False}
    together = register_rigid(data, fixed, batch_size=2, **options)
    alone = register_rigid(data, fixed, batch_size=1, **options)
    np.testing.assert_allclose(together.transforms, alone.transforms, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(
        together.diagnostics.n_iterations, alone.diagnostics.n_iterations
    )
    assert together.registered.shape == (2, *fixed.shape)
    np.testing.assert_array_equal(together.registered.time, data.time)
    np.testing.assert_array_equal(together.registered.acquisition, data.acquisition)
    validate_voxeldata(together.registered)
    validate_voxeldata(together.valid)
    if torch.cuda.is_available():
        cuda = register_rigid(data, fixed, device="cuda", **options)
        indices = np.indices(fixed.shape)
        points = position_to_world(fixed) @ np.concatenate(
            (indices, np.ones((1, *fixed.shape))), axis=0
        ).reshape(4, -1)
        error = np.einsum(
            "tij,jp->tip", cuda.transforms.values - together.transforms.values, points
        )[:, :3]
        ell = np.linalg.norm(position_to_world(fixed)[:3, :3], axis=0).min()
        assert np.linalg.norm(error, axis=1).max() < 0.001 * ell
        np.testing.assert_array_equal(
            cuda.diagnostics.n_iterations, together.diagnostics.n_iterations
        )


def test_3d_reference_and_float64_detail_along_k_are_preserved():
    values = np.broadcast_to(
        2.0**30 + 1e-3 * np.arange(5)[:, None, None], (5, 6, 7)
    ).copy()
    data = create_voxeldata(
        np.stack([values, values]),
        dims=("time", "k", "j", "i"),
        dt=1,
        voxel_to_world=np.eye(4),
    )
    r = register_rigid(data, intensity_scaling="none", max_iterations=1, progress=False)
    assert r.registered.dtype == np.float64
    np.testing.assert_array_equal(r.registered[0], data[0])
    np.testing.assert_array_equal(r.transforms[0], np.eye(4))
    assert r.diagnostics.status.values[0] == "reference"
    assert r.diagnostics.objective.values[1] < -0.99999
    assert r.diagnostics.status.values[1] != "invalid_metric"


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
    r = register_rigid(moving, fixed, max_iterations=2, progress=False)
    assert r.diagnostics.status == "insufficient_overlap"
    assert r.diagnostics.n_iterations == 0
    assert np.isnan(r.diagnostics.objective)
    assert not r.valid.any()
    np.testing.assert_array_equal(r.transforms, np.eye(4))
    np.testing.assert_array_equal(
        r.registered, np.full(fixed.shape, moving.values.min())
    )


def test_non_coplanar_slices_and_mixed_dimensions_are_explicit_errors():
    fixed, _ = pair(True)
    tilted = grid_data((1, 15, 17), [0, 20, 0], [0.5, 0.4, 0.3])
    with pytest.raises(ValueError, match="coplanar"):
        register_rigid(tilted, fixed, progress=False)
    volume = grid_data((3, 15, 17), [0, 0, 0], [0.5, 0.4, 0.3])
    with pytest.raises(ValueError, match="coplanar|3D"):
        register_rigid(volume, fixed, progress=False)
