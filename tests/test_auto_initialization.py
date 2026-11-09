"""Physical centering uses original intensities, not fitting/display scaling."""

# ruff: noqa: B009 -- ITK generates these attributes dynamically; getattr keeps typing accurate.

from typing import Literal

import numpy as np
import pytest
import torch
from confusius import create_voxeldata
from oracles import max_pull_difference
from registration_options import RegistrationOptions
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from voxreg import position_to_world, register

DEVICES: list[Literal["cpu", "cuda"]] = (
    ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)


def volume_pair():
    shape = (9, 11, 13)
    k, j, i = np.indices(shape)
    values = (
        0.01
        + np.exp(-((k - 3) ** 2 + (j - 4) ** 2 + (i - 5) ** 2) / 4)
        + 0.4 * np.exp(-((k - 5) ** 2 + (j - 8) ** 2 + (i - 9) ** 2) / 3)
    ).astype(np.float64)
    affine = np.eye(4)
    affine[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix() @ np.diag(
        [0.4, 0.3, 0.2]
    )
    affine[:3, 3] = [10, -20, 30]
    fixed = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=affine)
    shifted = affine.copy()
    shifted[:3, 3] += [20, -12, 8]
    moving = create_voxeldata(
        values.copy(), dims=("k", "j", "i"), voxel_to_world=shifted
    )
    return fixed, moving


def native_centers(fixed, moving, mode):
    import itk

    images = []
    for data in [fixed, moving]:
        affine = position_to_world(data)
        image = itk.image_from_array(
            np.ascontiguousarray(data.values.T, dtype=np.float64)
        )
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        image.SetSpacing(tuple(spacing))
        image.SetOrigin(tuple(affine[:3, 3]))
        image.SetDirection(itk.matrix_from_array(affine[:3, :3] / spacing))
        images.append(image)
    transform = getattr(itk, "VersorRigid3DTransform")[itk.D].New()
    initializer = getattr(itk, "CenteredTransformInitializer")[
        type(transform), type(images[0]), type(images[1])
    ].New(Transform=transform, FixedImage=images[0], MovingImage=images[1])
    if mode == "center_geometry":
        initializer.GeometryOn()
    else:
        initializer.MomentsOn()
    initializer.InitializeTransform()
    return np.array(transform.GetCenter()), np.array(transform.GetTranslation())


@pytest.mark.parametrize("mode", ["center_geometry", "center_moments"])
@pytest.mark.parametrize("scaling", [None, "db", "sqrt", 2.0])
@pytest.mark.parametrize("device", DEVICES)
def test_auto_pose_recovers_large_offset_and_matches_itk(mode, scaling, device):
    fixed, moving = volume_pair()
    original_fixed, original_moving = fixed.values.copy(), moving.values.copy()
    center, translation = native_centers(fixed, moving, mode)
    before = register(
        moving,
        fixed,
        return_transforms=True,
        moving_intensity_scaling=scaling,
        fixed_intensity_scaling=scaling,
        max_iterations=1,
        progress=False,
        device=device,
    )
    assert before.diagnostics.status == "insufficient_overlap"
    r = register(
        moving,
        fixed,
        return_transforms=True,
        initialization=mode,
        moving_intensity_scaling=scaling,
        fixed_intensity_scaling=scaling,
        max_iterations=1,
        progress=False,
        device=device,
    )
    assert r.transforms is not None
    expected = np.eye(4)
    expected[:3, 3] = translation
    np.testing.assert_allclose(r.transforms, expected, atol=3e-13, rtol=0)
    np.testing.assert_allclose(
        r.diagnostics.attrs["rotation_center_zyx_mm"], center, atol=3e-13, rtol=0
    )
    np.testing.assert_allclose(r.registered, fixed, atol=3e-13, rtol=0)
    assert r.valid.all() and r.diagnostics.objective < -0.999999
    assert r.registered.dtype == np.float64
    np.testing.assert_array_equal(fixed.values, original_fixed)
    np.testing.assert_array_equal(moving.values, original_moving)


@pytest.mark.parametrize("kind", ["same_slice", "independent_slice", "volume"])
@pytest.mark.parametrize("device", DEVICES)
def test_moments_pivot_sampling_and_scales(
    recording_factory, monkeypatch, kind, device
):
    from voxreg import linear

    if kind == "volume":
        fixed, moving = volume_pair()
    else:
        data, _, _ = recording_factory(oblique=True, strided=True)
        fixed = data.isel(time=0, drop=True).astype(np.float64)
        moving = data.isel(time=1, drop=True).astype(np.float64)
        if kind == "independent_slice":
            fixed = fixed.isel(j=slice(6, -6), i=slice(6, -6))
    center, translation = native_centers(fixed, moving, "center_moments")
    captured = {}

    def rotated(images, target, plane, reference, *args, **kwargs):
        """Exercise the new pivot in final sampling independently of fitting."""
        p = kwargs["initial_parameters"].clone()
        p[:, : 3 if kind == "volume" else 1] = torch.tensor(
            [0.04, -0.03, 0.02] if kind == "volume" else [0.04],
            device=images.device,
            dtype=p.dtype,
        )
        from voxreg.sampling import sample_rigid

        assert torch.autograd.gradcheck(
            lambda q: sample_rigid(images.double(), q, plane)[0].sum(),
            (p.detach().requires_grad_(True),),
            eps=1e-6,
            atol=2e-5,
            rtol=2e-4,
        )
        captured["plane"] = plane
        return (
            p,
            torch.zeros(len(images), dtype=torch.int64, device=images.device),
            torch.ones(len(images), dtype=torch.int64, device=images.device),
        )

    monkeypatch.setattr(linear, "_optimize", rotated)
    r = register(
        moving,
        fixed,
        return_transforms=True,
        initialization="center_moments",
        progress=False,
        device=device,
    )
    assert r.transforms is not None
    plane = captured["plane"]
    if kind == "volume":
        rotation = Rotation.from_rotvec([0.02, -0.03, 0.04]).as_matrix()[::-1, ::-1]
    else:
        c, s = np.cos(0.04), np.sin(0.04)
        rotation = (
            np.eye(3)
            + plane.basis @ (np.array([[c, -s], [s, c]]) - np.eye(2)) @ plane.basis.T
        )
    expected = np.eye(4)
    expected[:3, :3] = rotation
    expected[:3, 3] = translation - (rotation - np.eye(3)) @ center
    np.testing.assert_allclose(r.transforms, expected, atol=2e-12, rtol=0)
    af = position_to_world(fixed)
    points = af @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size)))
    )
    coordinates = (np.linalg.inv(position_to_world(moving)) @ expected @ points)[:3]
    valid = (
        (coordinates >= -1e-5)
        & (coordinates <= np.array(moving.shape)[:, None] - 1 + 1e-5)
    ).all(axis=0)
    values = map_coordinates(
        moving.values, coordinates, order=1, mode="nearest", prefilter=False
    )
    values[~valid] = moving.values.min()
    np.testing.assert_allclose(
        r.registered, values.reshape(fixed.shape), atol=2e-12, rtol=0
    )
    np.testing.assert_array_equal(r.valid, valid.reshape(fixed.shape))
    distances = points[:3] - center[:, None]
    ell = np.linalg.norm(af[:3, :3] if kind == "volume" else af[:3, 1:], axis=0).min()
    if kind == "volume":
        variance = np.mean(distances**2, axis=1)
        scales = np.r_[
            ell / np.sqrt(variance[[1, 2, 0]] + variance[[2, 0, 1]]), [ell] * 3
        ]
    else:
        scales = np.r_[ell / np.sqrt(np.mean(np.sum(distances**2, axis=0))), [ell] * 2]
    np.testing.assert_allclose(plane.parameter_scale.cpu(), scales, rtol=2e-12)


@pytest.mark.parametrize("mode", ["center_geometry", "center_moments"])
def test_series_reference_batch_and_fitting_scaling(recording_factory, mode, device):
    data, _, _ = recording_factory(oblique=True, strided=True)
    options: RegistrationOptions = {
        "initialization": mode,
        "device": device,
        "max_iterations": 100,
        "progress": False,
    }
    combined = register(data, return_transforms=True, batch_size=3, **options)
    assert combined.transforms is not None
    serial = register(data, return_transforms=True, batch_size=1, **options)
    assert serial.transforms is not None
    if device == "cpu":
        np.testing.assert_allclose(
            combined.transforms, serial.transforms, atol=1e-9, rtol=0
        )
    else:
        assert (
            max_pull_difference(
                combined.transforms, serial.transforms, data.isel(time=0, drop=True)
            )
            < 0.01
        )
    np.testing.assert_array_equal(
        combined.diagnostics.n_iterations, serial.diagnostics.n_iterations
    )
    np.testing.assert_array_equal(combined.transforms[0], np.eye(4))
    np.testing.assert_array_equal(combined.registered[0], data[0])
    # With a frozen optimizer, fitting scaling cannot alter the initialized pose/pivot.
    from unittest.mock import patch

    def frozen(images, target, plane, reference, *args, **kwargs):
        return (
            kwargs["initial_parameters"],
            torch.zeros(len(images), dtype=torch.int64),
            torch.ones(len(images), dtype=torch.int64),
        )

    with patch("voxreg.linear._optimize", frozen):
        linear = register(data, return_transforms=True, **options)
        db = register(
            data, return_transforms=True, moving_intensity_scaling="db", **options
        )
    np.testing.assert_array_equal(linear.transforms, db.transforms)
    np.testing.assert_array_equal(
        linear.diagnostics.attrs["rotation_center_zyx_mm"],
        db.diagnostics.attrs["rotation_center_zyx_mm"],
    )


@pytest.mark.parametrize("side", ["fixed", "moving"])
@pytest.mark.parametrize("value", [-1.0, 0.0])
def test_moments_reject_invalid_original_weights(side, value):
    fixed, moving = volume_pair()
    bad = (fixed if side == "fixed" else moving).copy(data=np.full(fixed.shape, value))
    with pytest.raises(ValueError, match=f"{side}.*center_moments.*unscaled"):
        register(
            bad if side == "moving" else moving,
            bad if side == "fixed" else fixed,
            initialization="center_moments",
            progress=False,
        )


@pytest.mark.parametrize("mode", ["moments", "geometry", False, 3])
def test_invalid_mode(mode):
    fixed, moving = volume_pair()
    with pytest.raises(ValueError, match="initialization"):
        register(moving, fixed, initialization=mode, progress=False)


def test_moments_refines_rotation_fixed_budget_with_cpu_cuda_agreement():
    fixed, source = volume_pair()
    affine = position_to_world(fixed)
    center = affine @ np.r_[(np.array(fixed.shape) - 1) / 2, 1]
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_rotvec([0.025, -0.018, 0.012]).as_matrix()[::-1, ::-1]
    truth[:3, 3] = [20, -12, 8] - (truth[:3, :3] - np.eye(3)) @ center[:3]
    moving = create_voxeldata(
        source.values.copy(), dims=source.dims, voxel_to_world=truth @ affine
    )
    fixed = fixed.isel(k=slice(1, -1), j=slice(1, -1), i=slice(1, -1))
    af = position_to_world(fixed)
    points = af @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size)))
    )
    ell = np.linalg.norm(af[:3, :3], axis=0).min()
    results = []
    for device in DEVICES:
        r = register(
            moving,
            fixed,
            return_transforms=True,
            initialization="center_moments",
            max_iterations=160,
            convergence_window_size=161,
            progress=False,
            device=device,
        )
        assert r.transforms is not None
        error = (
            np.linalg.norm(((r.transforms.values - truth) @ points)[:3], axis=0) / ell
        )
        assert np.sqrt(np.mean(error**2)) < 0.25
        assert r.diagnostics.objective < -0.999
        results.append(r)
    if len(results) == 2:
        assert results[0].transforms is not None and results[1].transforms is not None
        error = (
            (results[0].transforms.values - results[1].transforms.values) @ points
        )[:3]
        assert np.linalg.norm(error, axis=0).max() < 0.01 * ell
        np.testing.assert_array_equal(
            results[0].diagnostics.n_iterations, results[1].diagnostics.n_iterations
        )


def test_explicit_and_auto_are_exclusive():
    fixed, moving = volume_pair()
    with pytest.raises(ValueError, match="not both"):
        register(
            moving,
            fixed,
            initialization="center_geometry",
            initial_transform=np.eye(4),
            progress=False,
        )
