"""Independent fitting transforms preserve native initialization and output."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from registration_options import RegistrationOptions

from voxreg import Stage, position_to_world, register, resample
from voxreg.preparation import prepare_fitting_images


@pytest.mark.parametrize("metric", [None, "mi"])
@pytest.mark.parametrize(
    "moving_scaling,fixed_scaling",
    [(None, None), ("log1p", "sqrt"), ("sqrt", 2.0), (3, "log1p")],
)
def test_conditioning_matches_numpy_magnitude_oracle(
    metric, moving_scaling, fixed_scaling
):
    moving = np.array([[[0, -0.01, 1], [2, -4, 8]], [[0, 0.5, -3], [4, 8, -16.0]]])
    fixed = np.array([[0, -0.2, 1], [2, -5, 10.0]])
    originals = moving.copy(), fixed.copy()
    expected = []
    for values, scaling in ((moving, moving_scaling), (fixed, fixed_scaling)):
        if scaling == "log1p":
            values = np.log1p(np.abs(values))
        elif scaling is not None:
            values = np.abs(values) ** (0.5 if scaling == "sqrt" else scaling)
        axes = tuple(range(values.ndim - 2, values.ndim))
        low = (
            values.min(axis=axes, keepdims=True)
            if metric == "mi"
            else values[..., :1, :1]
        )
        values = values - low
        amplitude = (
            values.max(axis=axes, keepdims=True)
            if metric == "mi"
            else np.abs(values).max(axis=axes, keepdims=True)
        )
        expected.append(values / amplitude)
    fit, target = prepare_fitting_images(
        moving, fixed, moving_scaling, fixed_scaling, metric
    )
    assert_allclose(fit, expected[0], rtol=1e-6, atol=1e-7)
    assert_allclose(target, expected[1], rtol=1e-6, atol=1e-7)
    assert fit.dtype == target.dtype == np.float32
    assert_array_equal(moving, originals[0])
    assert_array_equal(fixed, originals[1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
@pytest.mark.parametrize("metric", [None, "mi"])
@pytest.mark.parametrize("scaling", [None, "log1p", "sqrt", 2.0])
def test_conditioning_is_independent_of_other_frames(dtype, metric, scaling):
    values = np.array([[[1, 2, 3], [4, 8, 16]]], dtype=dtype)
    fixed = np.array([[2, 3, 5], [7, 11, 13]], dtype=dtype)
    alone, target = prepare_fitting_images(values, fixed, scaling, None, metric)
    combined = np.concatenate((values, values * np.array(100, dtype=dtype)))
    together, combined_target = prepare_fitting_images(
        combined, fixed, scaling, None, metric
    )
    assert_array_equal(alone[0], together[0])
    assert_array_equal(target, combined_target)
    assert np.ptp(alone) > 0
    assert_array_equal(values, np.array([[[1, 2, 3], [4, 8, 16]]], dtype=dtype))


def test_float32_conditioning_preserves_signed_contrast_with_bright_other_frame():
    values = np.array([[[0, -1, 2], [3, -4, 8]]], dtype=np.float32)
    fixed = np.array([[0, -2, 1], [4, 8, -16]], dtype=np.float32)
    combined = np.concatenate((values, values * np.float32(1e30)))
    fit, target = prepare_fitting_images(combined, fixed)
    assert_array_equal(fit[0], values[0] / 8)
    assert_array_equal(target, fixed / 16)
    assert np.isfinite(fit).all()


def test_log1p_reference_and_zeros_match_numpy_without_global_reference():
    values = np.array([[[0, 1e-14], [0.1, 1]], [[0, 1e-10], [10, 100.0]]])
    fit, target = prepare_fitting_images(
        values, values[0], "log1p", implicit_fixed=True
    )
    expected = np.log1p(np.abs(values))
    expected /= expected.max(axis=(1, 2), keepdims=True)
    assert_allclose(fit, expected, atol=1e-7)
    assert_array_equal(target, fit[0])
    _, independently_scaled = prepare_fitting_images(
        values, values[0], "log1p", "log1p"
    )
    assert_array_equal(target, independently_scaled)


@pytest.mark.parametrize("side", ["moving", "fixed"])
@pytest.mark.parametrize(
    "bad",
    [
        "none",
        "db",
        "invalid",
        0,
        -1,
        np.nan,
        np.inf,
        True,
        np.bool_(True),
        1j,
        [],
        np.array([1, 2]),
        10**400,
    ],
)
def test_invalid_scaling_rejected_before_stages(recording_factory, side, bad):
    data, _, _ = recording_factory()
    with pytest.raises(ValueError, match=f"{side}_intensity_scaling"):
        register(
            data,
            data.isel(time=0, drop=True),
            transform=("translation", "rigid"),
            **{f"{side}_intensity_scaling": bad},
        )


def test_obsolete_shared_option_is_removed(recording_factory):
    data, _, _ = recording_factory()
    with pytest.raises(TypeError, match="intensity_scaling"):
        options: RegistrationOptions = {"intensity_scaling": "log1p"}
        register(data, **options)


def test_fixed_scaling_requires_explicit_fixed(recording_factory):
    data, _, _ = recording_factory()
    with pytest.raises(ValueError, match="explicit fixed"):
        register(data, fixed_intensity_scaling="sqrt")


@pytest.mark.parametrize("side", ["moving", "fixed"])
def test_power_overflow_is_descriptive(recording_factory, side):
    data, _, _ = recording_factory()
    data = data.astype(np.float64) * 1e200
    with pytest.raises(ValueError, match=f"{side}.*overflowed float64"):
        options: RegistrationOptions = {f"{side}_intensity_scaling": 2.0}
        register(data, data.isel(time=0, drop=True), **options)


@pytest.mark.parametrize("side", ["moving", "fixed"])
@pytest.mark.parametrize("scaling", ["log1p", "sqrt", 2.0])
def test_moments_reject_negative_native_weights_before_magnitude_scaling(
    recording_factory, side, scaling
):
    data, _, _ = recording_factory()
    fixed = data.isel(time=0, drop=True).copy(deep=True)
    if side == "moving":
        data.values[1, 0, 0, 0] = -1
    else:
        fixed.values[0, 0, 0] = -1
    with pytest.raises(ValueError, match=f"{side}.*center_moments.*unscaled"):
        register(
            data,
            fixed,
            initialization="center_moments",
            moving_intensity_scaling=scaling,
            fixed_intensity_scaling=scaling,
            progress=False,
        )


def test_zero_fixed_log1p_and_total_power_underflow_are_rejected():
    values = np.array([[[0.1, 0.2], [0.3, 0.4]]])
    with pytest.raises(ValueError, match="fixed image must not be constant"):
        prepare_fitting_images(values, np.zeros((2, 2)), None, "log1p")
    with pytest.raises(ValueError, match="all-zero fitting"):
        prepare_fitting_images(values, values[0], 10000.0, 10000.0)


@pytest.mark.parametrize("scaling", ["log1p", "sqrt", np.float64(2.0)])
def test_extreme_finite_magnitudes_and_zeros(scaling):
    values = np.array([[[0, 1e-300], [1e-200, 1e100]]])
    fit, target = prepare_fitting_images(values, values[0], scaling, scaling)
    assert np.isfinite(fit).all() and np.isfinite(target).all()
    assert target[1, 1] == 1
    assert target[0, 0] == 0
    assert target[1, 0] < 1e-10


@pytest.mark.parametrize("metric", ["ncc", "mi"])
@pytest.mark.parametrize(
    "model", ["rigid", "syn", ("translation", Stage("rigid"), "syn")]
)
@pytest.mark.parametrize(
    "moving_scaling,fixed_scaling", [(2.0, None), (None, "sqrt"), ("log1p", "sqrt")]
)
def test_independent_transforms_match_pretransformed_fit_and_native_resampling(
    recording_factory, metric, model, moving_scaling, fixed_scaling, device
):
    data, _, _ = recording_factory(oblique=True, strided=True)
    data = data.astype(np.float64)
    fixed = data.isel(time=0, drop=True).copy(data=data.values[0] ** 2)
    mask = fixed.copy(data=np.ones(fixed.shape, dtype=bool))
    mask.values[:, :4] = False
    options: RegistrationOptions = {
        "transform": model,
        "device": device,
        "metric": metric,
        "fixed_mask": mask,
        "moving_mask": mask,
        "shrink_factors": (2, 1),
        "smoothing_sigmas": (0.1, 0),
        "max_iterations": (2, 2),
        "return_transforms": True,
        "progress": False,
    }
    result = register(
        data,
        fixed,
        moving_intensity_scaling=moving_scaling,
        fixed_intensity_scaling=fixed_scaling,
        **options,
    )
    moving_values = data.values
    if moving_scaling == "log1p":
        moving_values = np.log1p(np.abs(moving_values))
    elif moving_scaling is not None:
        moving_values = moving_values**moving_scaling
    fixed_values = np.sqrt(fixed.values) if fixed_scaling == "sqrt" else fixed.values
    expected = register(
        data.copy(data=moving_values), fixed.copy(data=fixed_values), **options
    )
    assert result.transforms is not None and expected.transforms is not None
    assert_allclose(result.transforms, expected.transforms, rtol=0, atol=1e-11)
    assert_allclose(
        result.diagnostics.objective, expected.diagnostics.objective, rtol=0, atol=1e-6
    )
    assert_array_equal(result.valid, expected.valid)
    native, valid = resample(data, fixed, result.transforms, device=device)
    assert_allclose(result.registered, native, rtol=0, atol=1e-11)
    assert_array_equal(result.valid, valid)
    assert result.registered.dtype == data.dtype
    assert_array_equal(result.registered.time, data.time)
    assert_allclose(
        position_to_world(result.registered), position_to_world(fixed), atol=0
    )


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("model", ["rigid", ("translation", "rigid")])
@pytest.mark.parametrize("scaling", ["log1p", "sqrt", 2.0])
def test_moment_initialization_uses_native_weights(
    recording_factory, monkeypatch, explicit, model, scaling
):
    from voxreg import linear

    def frozen(images, target, plane, reference, *args, **kwargs):
        """Retain the initial pose independently of the fitting objective."""
        return (
            kwargs["initial_parameters"],
            torch.zeros(len(images), dtype=torch.int64),
            torch.where(reference, 4, 5),
        )

    monkeypatch.setattr(linear, "_optimize", frozen)
    data, _, _ = recording_factory(oblique=True, strided=True)
    data = data.astype(np.float64)
    fixed = data.isel(time=0, drop=True) if explicit else None
    options: RegistrationOptions = {
        "initialization": "center_moments",
        "transform": model,
        "max_iterations": 1,
        "return_transforms": True,
        "progress": False,
    }
    native = register(data, fixed, **options)
    assert native.transforms is not None
    positions = np.indices(data.shape[1:]).reshape(3, -1)
    weights = data.values.reshape(len(data), -1)
    centroids = (positions @ weights.T / weights.sum(axis=1)).T
    affine = position_to_world(data)
    expected = np.broadcast_to(np.eye(4), (len(data), 4, 4)).copy()
    expected[:, :3, 3] = (centroids - centroids[0]) @ affine[:3, :3].T
    assert_allclose(native.transforms, expected, rtol=0, atol=1e-12)
    scaled = register(
        data,
        fixed,
        moving_intensity_scaling=scaling,
        fixed_intensity_scaling="sqrt" if explicit else None,
        **options,
    )
    assert scaled.transforms is not None
    assert_array_equal(scaled.transforms, native.transforms)
    assert_array_equal(scaled.registered, native.registered)
    native_first = (
        native.stage_diagnostics[0] if native.stage_diagnostics else native.diagnostics
    )
    scaled_first = (
        scaled.stage_diagnostics[0] if scaled.stage_diagnostics else scaled.diagnostics
    )
    assert_array_equal(
        scaled_first.attrs["rotation_center_zyx_mm"],
        native_first.attrs["rotation_center_zyx_mm"],
    )
    if not explicit:
        assert_array_equal(scaled.registered[0], data[0])
        assert_array_equal(scaled.transforms[0], np.eye(4))


@pytest.mark.parametrize("scaling", ["log1p", "sqrt", 2.0])
def test_batches_do_not_recondition_frames(recording_factory, scaling):
    data, _, _ = recording_factory()
    data = data.copy(data=data.values * np.array([1, 0.001, 100])[:, None, None, None])
    options: RegistrationOptions = {
        "moving_intensity_scaling": scaling,
        "max_iterations": 2,
        "return_transforms": True,
        "progress": False,
    }
    serial = register(data, batch_size=1, **options)
    together = register(data, batch_size=3, **options)
    assert serial.transforms is not None and together.transforms is not None
    assert_allclose(serial.transforms, together.transforms, rtol=0, atol=1e-7)
    assert_allclose(
        serial.diagnostics.objective, together.diagnostics.objective, rtol=0, atol=1e-6
    )
    assert_array_equal(serial.diagnostics.status, together.diagnostics.status)
