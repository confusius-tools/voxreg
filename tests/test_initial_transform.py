"""ITK-style optimized initial poses, not frozen pre-transform composition."""

from dataclasses import replace

import numpy as np
import pytest
import torch
import xarray as xr
from confusius import create_voxeldata
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from voxreg import position_to_world, register
from voxreg.sampling import _initial_parameters, make_grid, world_matrices

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def posed_pair():
    center = np.array([10.0, -20.0, 30.0])
    lattice = Rotation.from_euler(
        "xyz", [7, -12, 17], degrees=True
    ).as_matrix() @ np.diag([0.3, 0.25, 0.2])
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_rotvec([0.17, -0.12, 0.24]).as_matrix()[::-1, ::-1]
    truth[:3, 3] = center + [20.0, -12.0, 8.0] - truth[:3, :3] @ center
    rng = np.random.default_rng(32)
    locations = rng.uniform(-1.5, 1.5, (16, 3))
    widths = rng.uniform(0.3, 0.7, (16, 3))
    amplitudes = rng.uniform(0.5, 2, 16)
    images = []
    for shape, pose in [((17, 19, 21), np.eye(4)), ((25, 27, 29), truth)]:
        affine = np.eye(4)
        affine[:3, :3] = lattice
        affine[:3, 3] = center - lattice @ ((np.array(shape) - 1) / 2)
        points = (
            lattice
            @ (np.indices(shape).reshape(3, -1) - (np.array(shape) - 1)[:, None] / 2)
        ).T
        values = 0.05 + np.sum(
            amplitudes
            * np.exp(
                -0.5 * np.sum(((points[:, None] - locations) / widths) ** 2, axis=-1)
            ),
            axis=1,
        )
        images.append(
            create_voxeldata(
                values.reshape(shape).astype(np.float32),
                dims=("k", "j", "i"),
                voxel_to_world=pose @ affine,
            )
        )
    return *images, truth


def world_points(data):
    return position_to_world(data) @ np.vstack(
        (np.indices(data.shape).reshape(3, -1), np.ones((1, data.size)))
    )


@pytest.mark.parametrize(
    "angle", [0.0, 1e-12, 0.3, np.pi - 1e-10, np.pi, np.pi + 1e-10]
)
def test_inverse_parameterization_including_pi(angle):
    fixed, moving, _ = posed_pair()
    grid = make_grid(fixed, moving)
    for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, -2, 3]]:
        matrix = np.eye(4)
        axis = np.array(axis) / np.linalg.norm(axis)
        matrix[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()[::-1, ::-1]
        matrix[:3, 3] = [0.2, -0.3, 0.4]
        parameters = _initial_parameters(matrix, grid, 1)
        np.testing.assert_allclose(
            world_matrices(parameters, grid)[0], matrix, atol=3e-14, rtol=0
        )
        shift = np.array([1e9, -2e9, 3e9])
        shifted = replace(grid, center=grid.center + shift)
        seed = world_matrices(parameters, shifted)[0]
        recovered = world_matrices(_initial_parameters(seed, shifted, 1), shifted)[0]
        points = world_points(fixed)
        points[:3] += shift[:, None]
        ell = np.linalg.norm(position_to_world(fixed)[:3, :3], axis=0).min()
        assert (
            np.linalg.norm(((recovered - seed) @ points)[:3], axis=0).max()
            < 0.001 * ell
        )


@pytest.mark.parametrize("device", DEVICES)
def test_initial_pose_enables_large_displacement_and_is_refined(device):
    fixed, moving, truth = posed_pair()
    options = {
        "intensity_scaling": "none",
        "progress": False,
        "max_iterations": 150,
        "device": device,
    }
    identity = register(moving, fixed, **options)
    assert identity.diagnostics.status == "insufficient_overlap"
    seed = truth.copy()
    seed[:3, :3] = (
        truth[:3, :3] @ Rotation.from_rotvec([0.004, -0.006, 0.003]).as_matrix()
    )
    center = position_to_world(fixed) @ np.r_[(np.array(fixed.shape) - 1) / 2, 1]
    seed[:3, 3] += (truth[:3, :3] - seed[:3, :3]) @ center[:3] + [0.06, -0.04, 0.05]
    untouched = seed.copy()
    result = register(moving, fixed, initial_transform=seed, **options)
    np.testing.assert_array_equal(seed, untouched)
    points = world_points(fixed)
    ell = np.linalg.norm(position_to_world(fixed)[:3, :3], axis=0).min()
    before = np.linalg.norm(((seed - truth) @ points)[:3], axis=0)
    after = np.linalg.norm(((result.transforms.values - truth) @ points)[:3], axis=0)
    assert np.sqrt(np.mean(after**2)) < 0.25 * ell
    assert np.mean(after**2) < np.mean(before**2) / 10
    assert result.diagnostics.objective < -0.995
    assert result.diagnostics.n_iterations > 0
    # Diagnostics parameterize the full returned transform, not a residual.
    p = np.array(
        [
            [
                float(result.diagnostics[name])
                for name in [
                    "rotation_z",
                    "rotation_y",
                    "rotation_x",
                    "translation_z",
                    "translation_y",
                    "translation_x",
                ]
            ]
        ]
    )
    np.testing.assert_allclose(
        world_matrices(p, make_grid(fixed, moving))[0], result.transforms, atol=1e-13
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("failure", ["constant", "overlap"])
def test_failed_fit_retains_initial_not_identity(device, failure):
    fixed, moving, seed = posed_pair()
    if failure == "constant":
        moving = moving.copy(data=np.ones_like(moving.values))
    else:
        seed = seed.copy()
        seed[:3, 3] += 1000
    r = register(
        moving,
        fixed,
        initial_transform=seed,
        intensity_scaling="none",
        max_iterations=2,
        progress=False,
        device=device,
    )
    assert r.diagnostics.status == (
        "invalid_metric" if failure == "constant" else "insufficient_overlap"
    )
    assert r.diagnostics.n_iterations == 0
    np.testing.assert_allclose(r.transforms, seed, atol=2e-14, rtol=0)
    if failure == "overlap":
        assert not r.valid.any()


@pytest.mark.parametrize("oblique,strided", [(False, False), (True, True)])
def test_slice_per_frame_initialization_and_reference(
    recording_factory, oblique, strided
):
    data, truth, _ = recording_factory(oblique=oblique, strided=strided)
    seed = truth.copy()
    options = {
        "initial_transform": seed,
        "intensity_scaling": "none",
        "max_iterations": 100,
        "progress": False,
    }
    combined = register(data, batch_size=3, **options)
    serial = register(data, batch_size=1, **options)
    np.testing.assert_allclose(
        combined.transforms, serial.transforms, atol=1e-9, rtol=0
    )
    np.testing.assert_array_equal(
        combined.diagnostics.n_iterations, serial.diagnostics.n_iterations
    )
    np.testing.assert_array_equal(combined.transforms[0], np.eye(4))
    np.testing.assert_array_equal(combined.registered[0], data[0])
    assert combined.diagnostics.status[0] == "reference"
    np.testing.assert_array_equal(seed, truth)
    labeled = combined.transforms.copy(data=truth)
    again = register(
        data,
        initial_transform=labeled,
        intensity_scaling="none",
        max_iterations=100,
        progress=False,
    )
    np.testing.assert_array_equal(again.transforms, combined.transforms)


@pytest.mark.parametrize("device", DEVICES)
def test_identity_seed_preserves_default_bitwise(recording_factory, device):
    data, _, _ = recording_factory(oblique=True, strided=True)
    options = {"max_iterations": 65, "progress": False, "device": device}
    default = register(data, **options)
    seeded = register(data, initial_transform=np.eye(4), **options)
    for name in ["registered", "valid", "transforms", "diagnostics"]:
        xr.testing.assert_equal(getattr(default, name), getattr(seeded, name))


@pytest.mark.parametrize(
    "seed",
    [
        np.eye(3),
        np.zeros((2, 4, 4)),
        np.eye(4) * np.nan,
        np.diag([1, 1, 1, 2]),
        np.diag([1.01, 1, 1, 1]),
        np.diag([-1, 1, 1, 1]),
        np.array([[1, 0.01, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    ],
)
def test_reject_nonrigid_and_malformed_seeds(seed):
    fixed, moving, _ = posed_pair()
    with pytest.raises(ValueError, match="initial_transform"):
        register(moving, fixed, initial_transform=seed, progress=False)


@pytest.mark.parametrize(
    "seed", [np.eye(4).astype(complex), np.full((4, 4), "x"), np.eye(4).astype(bool)]
)
def test_reject_nonnumeric_seeds(seed):
    fixed, moving, _ = posed_pair()
    with pytest.raises(TypeError, match="initial_transform"):
        register(moving, fixed, initial_transform=seed, progress=False)


def test_slice_seed_rejects_out_of_plane_motion(recording_factory):
    data, _, _ = recording_factory(oblique=False)
    for rotation, translation in [
        (Rotation.from_euler("x", 0.02).as_matrix()[::-1, ::-1], [0, 0, 0]),
        (np.eye(3), [0.01, 0, 0]),
    ]:
        seed = np.eye(4)
        seed[:3, :3] = rotation
        seed[:3, 3] = translation
        with pytest.raises(ValueError, match="preserve the fixed plane"):
            register(
                data,
                data.isel(time=0, drop=True),
                initial_transform=seed,
                progress=False,
            )


def test_reference_and_labeled_seed_validation(recording_factory):
    data, truth, _ = recording_factory()
    with pytest.raises(ValueError, match="implicit reference"):
        register(data, initial_transform=truth[1], progress=False)
    result = register(data, max_iterations=1, progress=False)
    labeled = result.transforms.copy(data=truth)
    for wrong in [
        labeled.isel(time=slice(None, None, -1)),
        labeled.isel(world_in=slice(None, None, -1)),
        labeled.rename(world_out="axis"),
        labeled.assign_attrs(spatial_units="m"),
        labeled.assign_attrs(convention="moving-to-fixed"),
    ]:
        with pytest.raises(ValueError, match="initial_transform"):
            register(data, initial_transform=wrong, progress=False)


def test_seeded_original_float64_resampling_matches_scipy(monkeypatch):
    from voxreg import rigid

    fixed, moving, seed = posed_pair()
    moving = moving.astype(np.float64)
    moving.values[:] += np.arange(moving.size).reshape(moving.shape) * 1e-10
    seed[:3, 3] += [0.025, -0.018, 0.017]

    def unchanged(images, fixed, plane, reference, *args, **kwargs):
        return (
            kwargs["initial_parameters"],
            torch.zeros(len(images), dtype=torch.int64),
            torch.ones(len(images), dtype=torch.int64),
        )

    monkeypatch.setattr(rigid, "_optimize", unchanged)
    r = register(
        moving, fixed, initial_transform=seed, intensity_scaling="none", progress=False
    )
    positions = (np.linalg.inv(position_to_world(moving)) @ seed @ world_points(fixed))[
        :3
    ]
    expected = map_coordinates(
        moving.values, positions, order=1, mode="nearest", prefilter=False
    ).reshape(fixed.shape)
    np.testing.assert_allclose(r.registered, expected, atol=5e-13, rtol=0)
    assert r.valid.all() and r.registered.dtype == np.float64


def test_3d_seed_batch_and_device_independence():
    fixed, moving, seed = posed_pair()
    data = moving.expand_dims(time=[2.0, 4.0])
    seeds = np.stack([seed, seed])
    seeds[0, :3, 3] += [0.06, -0.04, 0.05]
    seeds[1, :3, 3] += [-0.03, 0.05, -0.02]
    options = {"intensity_scaling": "none", "max_iterations": 110, "progress": False}
    together = register(data, fixed, initial_transform=seeds, batch_size=2, **options)
    alone = register(data, fixed, initial_transform=seeds, batch_size=1, **options)
    points = world_points(fixed)
    ell = np.linalg.norm(position_to_world(fixed)[:3, :3], axis=0).min()
    for other in [alone] + (
        [
            register(
                data,
                fixed,
                initial_transform=seeds,
                batch_size=2,
                device="cuda",
                **options,
            )
        ]
        if torch.cuda.is_available()
        else []
    ):
        error = np.einsum(
            "bij,jp->bip", other.transforms.values - together.transforms.values, points
        )[:, :3]
        assert np.linalg.norm(error, axis=1).max() < 0.001 * ell
        np.testing.assert_array_equal(
            other.diagnostics.n_iterations, together.diagnostics.n_iterations
        )
    shared = register(data, fixed, initial_transform=seeds[0], batch_size=2, **options)
    explicit = register(
        data, fixed, initial_transform=np.stack([seeds[0]] * 2), batch_size=2, **options
    )
    np.testing.assert_array_equal(shared.transforms, explicit.transforms)


def test_unlabeled_dataarray_seeds_follow_position(recording_factory):
    data, truth, _ = recording_factory()
    seed = xr.DataArray(truth, dims=("time", "world_out", "world_in"))
    r = register(
        data,
        initial_transform=seed,
        intensity_scaling="none",
        max_iterations=1,
        progress=False,
    )
    np.testing.assert_allclose(r.transforms, truth, atol=1e-12)
    # Selecting one time leaves scalar provenance, not a per-frame index.
    shared = r.transforms.isel(time=1)
    fixed = data.isel(time=0, drop=True)
    options = {"intensity_scaling": "none", "max_iterations": 1, "progress": False}
    labeled = register(data, fixed, initial_transform=shared, **options)
    positional = register(data, fixed, initial_transform=shared.values, **options)
    np.testing.assert_array_equal(labeled.transforms, positional.transforms)


def test_float32_rotation_roundoff_preserves_seed_center():
    fixed, moving, seed = posed_pair()
    seed = seed.astype(np.float32)
    untouched = seed.copy()
    grid = make_grid(fixed, moving)
    matrix = world_matrices(_initial_parameters(seed, grid, 1), grid)[0]
    center = np.r_[grid.center, 1]
    np.testing.assert_allclose(matrix @ center, seed @ center, atol=2e-14, rtol=0)
    np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-15)
    np.testing.assert_array_equal(seed, untouched)


def test_itk_initial_transform_inplaceoff_oracle():
    import itk

    fixed, moving, seed = posed_pair()
    image_type = itk.Image[itk.D, 3]
    images = []
    for data in [fixed, moving]:
        image = itk.image_from_array(data.values.astype(np.float64))
        affine = position_to_world(data)
        linear = affine[:3, :3][::-1, ::-1]
        spacing = np.linalg.norm(linear, axis=0)
        image.SetSpacing(tuple(spacing))
        image.SetOrigin(tuple(affine[:3, 3][::-1]))
        image.SetDirection(itk.matrix_from_array(linear / spacing))
        images.append(image)
    initial = itk.Euler3DTransform[itk.D].New()
    initial.SetCenter((3.0, 2.0, 1.0))
    initial.SetMatrix(itk.matrix_from_array(seed[:3, :3][::-1, ::-1].copy()))
    initial.SetOffset(tuple(seed[:3, 3][::-1]))
    untouched = np.asarray(initial.GetParameters()).copy()
    optimizer = itk.GradientDescentOptimizerv4.New()
    optimizer.SetNumberOfIterations(0)
    registration = itk.ImageRegistrationMethodv4[image_type, image_type].New()
    registration.SetFixedImage(images[0])
    registration.SetMovingImage(images[1])
    registration.SetInitialTransform(initial)
    registration.InPlaceOff()
    registration.SetMetric(
        itk.CorrelationImageToImageMetricv4[image_type, image_type].New()
    )
    registration.SetOptimizer(optimizer)
    registration.SetNumberOfLevels(1)
    registration.SetShrinkFactorsPerLevel([1])
    registration.SetSmoothingSigmasPerLevel([0])
    registration.Update()
    native = registration.GetModifiableTransform()
    failed = moving.copy(data=np.ones_like(moving.values))
    result = register(
        failed,
        fixed,
        initial_transform=seed,
        intensity_scaling="none",
        max_iterations=1,
        progress=False,
    )
    points = world_points(fixed)[:, ::100]
    expected = np.array(
        [
            tuple(native.TransformPoint(tuple(point[:3][::-1])))[::-1]
            for point in points.T
        ]
    ).T
    np.testing.assert_allclose(
        (result.transforms.values @ points)[:3], expected, atol=3e-13, rtol=0
    )
    # ITK's cloned output can be modified without changing the supplied initial object.
    parameters = native.GetParameters()
    parameters[3] += 1
    native.SetParameters(parameters)
    np.testing.assert_array_equal(np.asarray(initial.GetParameters()), untouched)


@pytest.mark.parametrize("representation", ["array", "list", "labeled", "batch"])
def test_initialization_matrix_alias_matches_initial_transform(representation):
    fixed, moving, truth = posed_pair()
    # Napari display pose: moving -> fixed. Registration needs the inverse pull.
    affine = np.linalg.inv(truth)
    seed = np.linalg.inv(affine)
    if representation == "list":
        seed = seed.tolist()
    elif representation == "labeled":
        seed = xr.DataArray(
            seed,
            dims=("world_out", "world_in"),
            coords={d: ["z", "y", "x", "1"] for d in ("world_out", "world_in")},
        )
    elif representation == "batch":
        seed = seed[None]
    untouched = np.array(seed, copy=True)
    options = {"intensity_scaling": "none", "max_iterations": 3, "progress": False}
    expected = register(moving, fixed, initial_transform=seed, **options)
    actual = register(moving, fixed, initialization=seed, **options)
    for name in ["registered", "valid", "transforms", "diagnostics"]:
        xr.testing.assert_equal(getattr(actual, name), getattr(expected, name))
    np.testing.assert_array_equal(seed, untouched)


@pytest.mark.parametrize("kind", ["scale", "shear"])
def test_initialization_matrix_does_not_discard_nonrigid_affine(kind):
    fixed, moving, _ = posed_pair()
    seed = np.eye(4)
    seed[0, 0 if kind == "scale" else 1] += 0.2
    with pytest.raises(ValueError, match="proper rigid"):
        register(
            moving, fixed, initialization=seed, intensity_scaling="none", progress=False
        )


def test_initialization_matrix_and_initial_transform_are_exclusive():
    fixed, moving, truth = posed_pair()
    with pytest.raises(ValueError, match="not both"):
        register(
            moving, fixed, initialization=truth, initial_transform=truth, progress=False
        )
