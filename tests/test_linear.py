"""Shared translation/rigid/affine engine: physical pulls, not frozen chains."""

import numpy as np
import pytest
import torch
import xarray as xr
from confusius import create_voxeldata
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from voxreg import position_to_world, register
from voxreg.sampling import _initial_parameters, make_grid, sample_rigid, world_matrices
from voxreg.transforms import affine_properties, configure_transform

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def pair(planar, model="affine", same=False):
    q = Rotation.from_rotvec([0.15, -0.2, 0.1]).as_matrix()
    lattice = q @ np.diag([0.24, 0.21, 0.18])
    basis = q[:, [2, 1]] if planar else np.eye(3)
    center = np.array([10.0, -20.0, 30.0])
    dimensions = 2 if planar else 3
    linear = np.eye(dimensions)
    if model == "affine":
        linear += (
            np.array([[0.04, 0.035], [-0.02, -0.03]])
            if planar
            else np.array(
                [[0.04, 0.025, -0.015], [-0.01, -0.03, 0.02], [0.015, 0, 0.025]]
            )
        )
    translation = np.array([0.10, -0.08]) if planar else np.array([0.10, -0.08, 0.06])
    truth = np.eye(4)
    truth[:3, :3] += basis @ (linear - np.eye(dimensions)) @ basis.T
    truth[:3, 3] = basis @ translation - (truth[:3, :3] - np.eye(3)) @ center
    rng = np.random.default_rng(871)
    locations = rng.uniform(-1.4, 1.4, (16, dimensions))
    widths = rng.uniform(0.3, 0.65, (16, dimensions))
    amplitudes = rng.uniform(0.5, 2, 16)
    fixed_shape = (1, 25, 29) if planar else (17, 19, 21)
    moving_shape = fixed_shape if same else ((1, 33, 37) if planar else (25, 27, 29))
    images = []
    for shape, pose in [(fixed_shape, np.eye(4)), (moving_shape, truth)]:
        af = np.eye(4)
        af[:3, :3] = lattice
        af[:3, 3] = center - lattice @ ((np.array(shape) - 1) / 2)
        points = (
            lattice
            @ (np.indices(shape).reshape(3, -1) - (np.array(shape) - 1)[:, None] / 2)
        ).T @ basis
        values = 0.05 + np.sum(
            amplitudes
            * np.exp(
                -0.5 * np.sum(((points[:, None] - locations) / widths) ** 2, axis=-1)
            ),
            axis=1,
        )
        images.append(
            create_voxeldata(
                values.reshape(shape), dims=("k", "j", "i"), voxel_to_world=pose @ af
            )
        )
    return *images, truth


def points(data):
    return position_to_world(data) @ np.vstack(
        (np.indices(data.shape).reshape(3, -1), np.ones((1, data.size)))
    )


def error(a, b, data):
    ell = np.linalg.norm(
        position_to_world(data)[:3, :3][:, np.array(data.shape) > 1], axis=0
    ).min()
    return (
        np.linalg.norm(((np.asarray(a) - np.asarray(b)) @ points(data))[:3], axis=0)
        / ell
    )


@pytest.mark.parametrize("model", ["rigid", "affine"])
@pytest.mark.parametrize("device", DEVICES)
def test_nonparallel_planes_refine_image_motion_after_geometry_alignment(model, device):
    fixed, moving, truth = pair(True, model)
    placement = np.eye(4)
    placement[:3, :3] = Rotation.from_rotvec([0, 0.3, 0]).as_matrix()
    placement[:3, 3] = [2, 1, -1]
    center = (position_to_world(fixed) @ np.r_[(np.array(fixed.shape) - 1) / 2, 1])[:3]
    spin = np.eye(4)
    spin[:3, :3] = Rotation.from_rotvec([0.07, 0, 0]).as_matrix()
    spin[:3, 3] = center - spin[:3, :3] @ center
    placement = placement @ spin
    moving = create_voxeldata(
        moving.values,
        dims=moving.dims,
        voxel_to_world=placement @ position_to_world(moving),
    )
    truth = placement @ truth
    kwargs = {
        "transform": model,
        "initialization": "align_planes",
        "intensity_scaling": "none",
        "max_iterations": 400,
        "convergence_window_size": 401,
        "progress": False,
    }
    result = register(moving, fixed, device=device, **kwargs)
    assert np.sqrt(np.mean(error(result.transforms, truth, fixed) ** 2)) < 0.25
    assert result.diagnostics.objective < -0.999
    if device == "cuda":
        cpu = register(moving, fixed, device="cpu", **kwargs)
        assert error(result.transforms, cpu.transforms, fixed).max() < 0.001


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("model", ["translation", "affine"])
def test_seed_roundtrip_and_itk_oracle(planar, model):
    import itk

    fixed, moving, truth = pair(planar, model)
    grid = configure_transform(make_grid(fixed, moving), fixed, model)
    parameters = _initial_parameters(truth, grid, 1)
    native = itk.AffineTransform[itk.D, 3].New()
    native.SetCenter(tuple(grid.center))
    native.SetMatrix(itk.matrix_from_array(truth[:3, :3]))
    native.SetTranslation(
        tuple(truth[:3, 3] + (truth[:3, :3] - np.eye(3)) @ grid.center)
    )
    expected = np.array(
        [native.TransformPoint(tuple(p[:3])) for p in points(fixed).T]
    ).T
    actual = world_matrices(parameters, grid)[0]
    np.testing.assert_allclose(
        (actual @ points(fixed))[:3], expected, rtol=0, atol=3e-14
    )
    failed = moving.copy(data=np.ones(moving.shape))
    r = register(
        failed,
        fixed,
        transform=model,
        initialization=truth,
        intensity_scaling="none",
        max_iterations=1,
    )
    np.testing.assert_allclose(r.transforms, actual, atol=3e-14, rtol=0)
    assert r.diagnostics.status == "invalid_metric"
    for key in ("linear_ii", "linear_zz"):
        if key in r.diagnostics:
            assert r.diagnostics[key] == pytest.approx(
                (grid.basis.T @ truth[:3, :3] @ grid.basis)[0, 0]
            )


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("model", ["translation", "affine"])
@pytest.mark.parametrize("device", DEVICES)
def test_seeded_sampling_scipy_and_gradients(planar, model, device):
    fixed, moving, truth = pair(planar, model)
    # Make the pivot visibly different from the geometric midpoint.
    pivot = position_to_world(fixed) @ np.r_[(np.array(fixed.shape) - 1) * 0.35, 1]
    grid = configure_transform(
        make_grid(fixed, moving, device=device, center=pivot[:3]), fixed, model
    )
    state = _initial_parameters(truth, grid, 1)
    state[:, -grid.basis.shape[1] :] += np.linspace(0.013, 0.021, grid.basis.shape[1])
    p = torch.tensor(state, device=device, requires_grad=True)
    values = moving.values[0] if planar else moving.values
    images = torch.tensor(values[None], device=device)
    assert torch.autograd.gradcheck(
        lambda state: sample_rigid(images, state, grid)[0].sum(),
        (p,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )
    sampled, valid = sample_rigid(images, p, grid)
    matrix = world_matrices(p.detach().cpu().numpy(), grid)[0]
    coordinates = (np.linalg.inv(position_to_world(moving)) @ matrix @ points(fixed))[
        :3
    ]
    expected = map_coordinates(
        moving.values, coordinates, order=1, mode="nearest", prefilter=False
    )
    np.testing.assert_allclose(
        sampled.detach().cpu().numpy().reshape(-1), expected, atol=2e-13, rtol=0
    )
    assert valid.all()
    assert (
        len(grid.parameter_scale) == (6 if planar else 12)
        if model == "affine"
        else len(grid.parameter_scale) == (2 if planar else 3)
    )
    # Dense unweighted RMS displacements independently verify entry-wise scales.
    distances = (points(fixed)[:3] - pivot[:3, None]).T @ grid.basis
    ell = np.linalg.norm(
        position_to_world(fixed)[:3, :3][:, np.array(fixed.shape) > 1], axis=0
    ).min()
    scales = np.full(grid.basis.shape[1], ell)
    if model == "affine":
        scales = np.r_[
            np.tile(ell / np.sqrt(np.mean(distances**2, axis=0)), grid.basis.shape[1]),
            scales,
        ]
    np.testing.assert_allclose(grid.parameter_scale.cpu(), scales, rtol=2e-13)


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("model", ["translation", "affine"])
def test_fitting_truth_and_cpu_cuda_precision(planar, model):
    fixed, moving, truth = pair(planar, model)
    results = []
    for device in DEVICES:
        r = register(
            moving,
            fixed,
            transform=model,
            intensity_scaling="none",
            initialization="center_moments",
            max_iterations=160,
            convergence_window_size=161,
            device=device,
            progress=False,
        )
        delta = error(r.transforms, truth, fixed)
        assert np.sqrt(np.mean(delta**2)) < 0.25
        assert r.diagnostics.objective < -0.99
        assert r.diagnostics.status == "max_iterations"
        results.append(r)
    if len(results) > 1:
        assert error(results[0].transforms, results[1].transforms, fixed).max() < 0.001
        np.testing.assert_array_equal(
            results[0].diagnostics.n_iterations, results[1].diagnostics.n_iterations
        )


@pytest.mark.parametrize("model", ["translation", "affine"])
@pytest.mark.parametrize("device", DEVICES)
def test_series_batch_reference_and_explicit_seed(recording_factory, model, device):
    data, _, _ = recording_factory(oblique=True, strided=True)
    options = {
        "transform": model,
        "max_iterations": 100,
        "convergence_window_size": 101,
        "progress": False,
        "initialization": "center_moments",
        "device": device,
    }
    a = register(data, batch_size=1, **options)
    b = register(data, batch_size=3, **options)
    np.testing.assert_allclose(a.transforms, b.transforms, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(
        a.diagnostics.n_iterations, b.diagnostics.n_iterations
    )
    np.testing.assert_array_equal(a.transforms[0], np.eye(4))
    np.testing.assert_array_equal(a.registered[0], data[0])
    c = register(
        data,
        transform=model,
        initialization=a.transforms,
        max_iterations=1,
        progress=False,
    )
    assert c.diagnostics.status[0] == "reference"
    bad = a.transforms.copy(deep=True)
    bad.values[0, :3, 3] += (
        0.1 * make_grid(data.isel(time=0, drop=True), data).basis[:, 0]
    )
    with pytest.raises(ValueError, match="implicit reference"):
        register(data, transform=model, initialization=bad, progress=False)


@pytest.mark.parametrize("model", ["translation", "affine"])
def test_same_grid_slice_kernel_against_independent_path(recording_factory, model):
    data, _, _ = recording_factory(oblique=True, strided=True)
    fixed = data.isel(time=0, drop=True)
    moving = data.isel(time=1, drop=True)
    grid = configure_transform(make_grid(fixed, moving), fixed, model)
    p = grid.parameter_scale.new_zeros((1, len(grid.parameter_scale)))
    p[0, -2:] = p.new_tensor([0.02, -0.03])
    if model == "affine":
        p[0, :4] = p.new_tensor([0.01, -0.02, 0.015, -0.005])
    images = torch.tensor(moving.values.astype(np.float64))
    actual, valid = sample_rigid(images, p, grid)
    matrix = world_matrices(p, grid)[0]
    coords = (np.linalg.inv(position_to_world(moving)) @ matrix @ points(fixed))[:3]
    expected = map_coordinates(
        moving.values.astype(np.float64),
        coords,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    np.testing.assert_allclose(actual.numpy().reshape(-1), expected, atol=2e-13, rtol=0)
    expected_valid = (
        (coords >= -1e-5) & (coords <= np.array(moving.shape)[:, None] - 1 + 1e-5)
    ).all(axis=0)
    np.testing.assert_array_equal(valid.numpy().reshape(-1), expected_valid)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4)),
        np.diag([0, 1, 1, 1]),
        np.diag([1e-10, 1, 1, 1]),
        np.full((4, 4), np.nan),
        np.ones((3, 3)),
    ],
)
def test_invalid_affine_seed(bad):
    fixed, moving, _ = pair(False)
    with pytest.raises(ValueError):
        register(moving, fixed, transform="affine", initialization=bad, progress=False)


@pytest.mark.parametrize("model", ["translation", "rigid"])
def test_restricted_models_reject_nonmatching_seed(model):
    fixed, moving, truth = pair(False)
    with pytest.raises(ValueError, match="identity linear|proper rigid"):
        register(moving, fixed, transform=model, initialization=truth, progress=False)
    with pytest.raises(ValueError, match="proper rigid"):
        register(moving, fixed, initialization=truth, progress=False)


@pytest.mark.parametrize("planar", [False, True])
def test_affine_seed_is_refined_as_full_pose_not_frozen_prefix(planar):
    fixed, moving, truth = pair(planar)
    normal = position_to_world(fixed)[:3, 0]
    normal /= np.linalg.norm(normal)
    shift = np.array([20.0, -12.0, 8.0])
    if planar:
        shift -= normal * np.dot(normal, shift)
    affine = position_to_world(moving)
    affine[:3, 3] += shift
    moving = create_voxeldata(
        moving.values.copy(), dims=moving.dims, voxel_to_world=affine
    )
    truth = truth.copy()
    truth[:3, 3] += shift
    grid = make_grid(fixed, moving)
    seed = truth.copy()
    seed[:3, :3] += grid.basis @ (np.eye(grid.basis.shape[1]) * 0.01) @ grid.basis.T
    seed[:3, 3] -= (seed[:3, :3] - truth[:3, :3]) @ grid.center
    seed[:3, 3] += grid.basis @ np.full(grid.basis.shape[1], 0.03)
    r = register(
        moving,
        fixed,
        transform="affine",
        initialization=seed,
        intensity_scaling="none",
        max_iterations=160,
        convergence_window_size=161,
        progress=False,
    )
    before = np.sqrt(np.mean(error(seed, truth, fixed) ** 2))
    after = np.sqrt(np.mean(error(r.transforms, truth, fixed) ** 2))
    assert after < 0.25 and after < before / 4
    assert np.max(np.abs(r.transforms.values[:3, :3] - seed[:3, :3])) > 1e-4


@pytest.mark.parametrize("planar", [False, True])
def test_affine_final_interpolation_preserves_float64_detail(planar):
    fixed, moving, truth = pair(planar)
    fixed = fixed.copy(data=2.0**30 + fixed.values * 0.001)
    moving = moving.copy(data=2.0**30 + moving.values * 0.001)
    r = register(
        moving,
        fixed,
        transform="affine",
        initialization=truth,
        intensity_scaling="none",
        max_iterations=1,
        progress=False,
    )
    coordinates = (
        np.linalg.inv(position_to_world(moving)) @ r.transforms.values @ points(fixed)
    )[:3]
    expected = map_coordinates(
        moving.values, coordinates, order=1, mode="nearest", prefilter=False
    ).reshape(fixed.shape)
    assert r.registered.dtype == np.float64 and r.valid.all()
    assert np.unique(r.registered).size > 100
    np.testing.assert_allclose(r.registered, expected, atol=1e-6, rtol=0)


def test_reflecting_affine_seed_is_preserved():
    fixed, moving, truth = pair(False)
    truth[:3, 0] *= -1
    failed = moving.copy(data=np.ones(moving.shape))
    r = register(
        failed,
        fixed,
        transform="affine",
        initialization=truth,
        intensity_scaling="none",
        max_iterations=1,
        progress=False,
    )
    np.testing.assert_allclose(r.transforms, truth, atol=3e-14, rtol=0)


@pytest.mark.parametrize("model", ["translation", "affine"])
def test_planar_seed_cannot_change_unidentifiable_normal(model):
    fixed, moving, _ = pair(True)
    af = position_to_world(fixed)
    normal = af[:3, 0] / np.linalg.norm(af[:3, 0])
    bad = np.eye(4)
    bad[:3, 3] = normal
    with pytest.raises(ValueError, match="preserve the fixed plane"):
        register(moving, fixed, transform=model, initialization=bad, progress=False)


def test_affine_candidate_guard_is_per_frame_and_keeps_best(monkeypatch):
    from voxreg import rigid

    fixed, moving, _ = pair(False)
    series = xr.concat([moving, moving], dim=xr.IndexVariable("time", [0, 1]))
    original = rigid.sample_rigid
    seen = []

    def sample(images, parameters, grid, **kwargs):
        # Observe candidate states without changing the production sampler.
        sampled, valid = original(images, parameters, grid, **kwargs)
        seen.append(parameters.detach().clone())
        return sampled, valid

    monkeypatch.setattr(rigid, "sample_rigid", sample)
    # Test the runtime predicate directly on a batch with singular and negative maps.
    grid = configure_transform(make_grid(fixed, moving), fixed, "affine")
    p = grid.parameter_scale.new_zeros((3, 12))
    p[0, 0] = -1
    p[1, 0] = -2
    valid, sign = affine_properties(p, grid)
    np.testing.assert_array_equal(valid, [False, True, True])
    np.testing.assert_array_equal(sign, [0, -1, 1])
    # A large learning rate crosses orientation or loses support; fallback is valid.
    r = register(
        series,
        fixed,
        transform="affine",
        learning_rate=100,
        max_iterations=2,
        intensity_scaling="none",
        progress=False,
    )
    assert np.isfinite(r.transforms).all()
    assert np.all(np.abs(np.linalg.det(r.transforms.values[:, :3, :3])) > 1e-8)
    assert set(r.diagnostics.status.values) <= {
        "invalid_transform",
        "insufficient_overlap",
    }
    assert any((item.abs() > 1).any() for item in seen)


def test_invalid_affine_frame_does_not_stop_companions(monkeypatch):
    from voxreg import rigid

    fixed, moving, _ = pair(False)
    series = xr.concat([moving, moving], dim=xr.IndexVariable("time", [0, 1]))
    original = rigid.affine_properties
    calls = 0

    def properties(parameters, grid):
        nonlocal calls
        valid, sign = original(parameters, grid)
        if calls > 0:
            valid[0] = False
        calls += 1
        return valid, sign

    monkeypatch.setattr(rigid, "affine_properties", properties)
    result = register(
        series,
        fixed,
        transform="affine",
        max_iterations=2,
        intensity_scaling="none",
        progress=False,
    )
    np.testing.assert_array_equal(result.diagnostics.n_iterations, [0, 2])
    np.testing.assert_array_equal(
        result.diagnostics.status, ["invalid_transform", "max_iterations"]
    )
    np.testing.assert_array_equal(result.transforms[0], np.eye(4))


def test_single_public_registration_entry_point():
    import voxreg

    assert set(voxreg.__all__) == {
        "RegistrationResult",
        "position_to_world",
        "register",
    }
    assert not hasattr(voxreg, "register_rigid")


def test_default_is_identical_to_explicit_rigid(recording_factory):
    data, _, _ = recording_factory()
    a = register(data, max_iterations=5, progress=False)
    b = register(data, transform="rigid", max_iterations=5, progress=False)
    for field in ("registered", "valid", "transforms", "diagnostics"):
        xr.testing.assert_equal(getattr(a, field), getattr(b, field))


@pytest.mark.parametrize("bad", ["euler", "nonrigid", None, 3])
def test_invalid_transform_family(bad, recording_factory):
    data, _, _ = recording_factory()
    with pytest.raises(ValueError, match="transform must"):
        register(data, transform=bad, progress=False)
