"""Per-image fitting winsorization leaves native geometry and intensities intact."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from registration_options import RegistrationOptions

from voxreg import position_to_world, register, resample


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize(
    "model", ["translation", "rigid", "affine", "syn", ("rigid", "syn")]
)
@pytest.mark.parametrize("metric", ["ncc", "mi"])
@pytest.mark.parametrize("sides", ["both", "moving", "fixed"])
def test_winsorization_matches_numpy_before_transforms_and_pyramids(
    linear_pair_factory, planar, model, metric, device, sides
):
    fixed, moving, _ = linear_pair_factory(planar, "rigid")
    # Include outliers outside the ROI so bounds must use the full native image.
    moving = moving.copy(data=moving.values.copy())
    fixed = fixed.copy(data=fixed.values.copy())
    moving.values.flat[0], fixed.values.flat[0] = -100, 200
    moving_original, fixed_original = moving.values.copy(), fixed.values.copy()
    mask = fixed.copy(data=np.ones(fixed.shape, dtype=bool))
    mask.values.flat[0] = False
    moving_bounds = None if sides == "fixed" else (10, 90)
    fixed_bounds = None if sides == "moving" else (5, 95)
    clipped_moving = moving.copy(
        data=np.clip(moving.values, *np.percentile(moving.values, moving_bounds))
        if moving_bounds is not None
        else moving.values
    )
    clipped_fixed = fixed.copy(
        data=np.clip(fixed.values, *np.percentile(fixed.values, fixed_bounds))
        if fixed_bounds is not None
        else fixed.values
    )
    options: RegistrationOptions = {
        "transform": model,
        "metric": metric,
        "moving_intensity_scaling": "log1p",
        "fixed_intensity_scaling": "sqrt",
        "fixed_mask": mask,
        "max_iterations": (2, 2),
        "shrink_factors": (2, 1),
        "smoothing_sigmas": (0.1, 0),
        "device": device,
        "progress": False,
        "return_transforms": True,
    }
    result = register(
        moving,
        fixed,
        moving_winsorize_percentiles=moving_bounds,
        fixed_winsorize_percentiles=fixed_bounds,
        **options,
    )
    expected = register(clipped_moving, clipped_fixed, **options)
    assert result.transforms is not None and expected.transforms is not None
    assert_allclose(result.transforms, expected.transforms, rtol=0, atol=1e-11)
    assert_allclose(
        result.diagnostics.objective, expected.diagnostics.objective, rtol=0, atol=1e-6
    )
    native, valid = resample(moving, fixed, result.transforms, device=device)
    assert_allclose(result.registered, native, rtol=0, atol=1e-11)
    assert_array_equal(result.valid, valid)
    assert_array_equal(moving, moving_original)
    assert_array_equal(fixed, fixed_original)
    assert_allclose(
        position_to_world(result.registered), position_to_world(fixed), rtol=0, atol=0
    )
    if result.stage_diagnostics:
        assert expected.stage_diagnostics is not None
        for actual_stage, expected_stage in zip(
            result.stage_diagnostics, expected.stage_diagnostics, strict=True
        ):
            assert_allclose(
                actual_stage.objective, expected_stage.objective, rtol=0, atol=1e-6
            )


@pytest.mark.parametrize("side", ["moving", "fixed"])
@pytest.mark.parametrize(
    "bad",
    [
        (),
        (1,),
        (1, 2, 3),
        "1,99",
        (True, 99),
        (1, np.bool_(True)),
        (np.nan, 99),
        (1, np.inf),
        (-1, 99),
        (1, 101),
        (99, 1),
        (50, 50),
        (1j, 99),
        (10**400, 99),
    ],
)
def test_invalid_bounds_are_rejected_before_fitting(
    recording_factory, monkeypatch, side, bad
):
    from voxreg import linear

    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid clipping bounds must fail before optimization.")

    monkeypatch.setattr(linear, "_optimize", unexpected)
    data, _, _ = recording_factory()
    options: RegistrationOptions = {f"{side}_winsorize_percentiles": bad}
    with pytest.raises(ValueError, match=f"{side}_winsorize_percentiles"):
        register(
            data,
            data.isel(time=0, drop=True),
            transform=("translation", "rigid"),
            **options,
        )


def test_fixed_clipping_requires_explicit_fixed(recording_factory):
    data, _, _ = recording_factory()
    with pytest.raises(ValueError, match="fixed_winsorize_percentiles.*explicit fixed"):
        register(data, fixed_winsorize_percentiles=(1, 99))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_implicit_reference_and_per_frame_bounds_are_batch_independent(
    recording_factory, dtype, device
):
    data, _, _ = recording_factory()
    values = np.rint(data.values * 100).astype(dtype)
    values[2] *= 10
    data = data.copy(data=values)
    percentiles = [np.float64(10), np.float64(90)]
    clipped = np.clip(
        values.astype(np.float64),
        *np.percentile(values, percentiles, axis=(1, 2, 3), keepdims=True),
    )
    options: RegistrationOptions = {
        "max_iterations": 2,
        "progress": False,
        "return_transforms": True,
        "device": device,
    }
    together = register(
        data, moving_winsorize_percentiles=percentiles, batch_size=3, **options
    )
    serial = register(
        data, moving_winsorize_percentiles=percentiles, batch_size=1, **options
    )
    expected = register(data.copy(data=clipped), **options)
    assert (
        together.transforms is not None
        and serial.transforms is not None
        and expected.transforms is not None
    )
    assert_allclose(together.transforms, expected.transforms, rtol=0, atol=1e-11)
    assert_allclose(together.transforms, serial.transforms, rtol=0, atol=1e-7)
    assert_array_equal(together.registered[0], data[0])
    assert_array_equal(together.transforms[0], np.eye(4))
    assert_array_equal(together.registered.time, data.time)
    assert together.registered.dtype == dtype
    assert percentiles == [10, 90]


@pytest.mark.parametrize("explicit", [False, True])
def test_clipping_preserves_native_moment_initialization(
    recording_factory, monkeypatch, explicit
):
    from voxreg import linear

    def frozen(images, target, plane, reference, *args, **kwargs):
        return (
            kwargs["initial_parameters"],
            torch.zeros(len(images), dtype=torch.int64),
            torch.where(reference, 4, 5),
        )

    monkeypatch.setattr(linear, "_optimize", frozen)
    data, _, _ = recording_factory()
    data = data.astype(np.float64)
    data.values[1, 0, 2, 3] = 10000
    fixed = data.isel(time=0, drop=True) if explicit else None
    options: RegistrationOptions = {
        "initialization": "center_moments",
        "transform": ("translation", "rigid"),
        "return_transforms": True,
        "progress": False,
        "device": "cpu",
    }
    native = register(data, fixed, **options)
    clipped = register(
        data,
        fixed,
        moving_winsorize_percentiles=(10, 90),
        fixed_winsorize_percentiles=(5, 95) if explicit else None,
        **options,
    )
    assert native.transforms is not None and clipped.transforms is not None
    positions = np.indices(data.shape[1:]).reshape(3, -1)
    weights = data.values.reshape(len(data), -1)
    centers = (positions @ weights.T / weights.sum(axis=1)).T
    expected = np.broadcast_to(np.eye(4), (len(data), 4, 4)).copy()
    expected[:, :3, 3] = (centers - centers[0]) @ position_to_world(data)[:3, :3].T
    assert_allclose(native.transforms, expected, rtol=0, atol=1e-12)
    assert_array_equal(clipped.transforms, native.transforms)
    assert_array_equal(clipped.registered, native.registered)


def test_clipped_constant_fixed_and_zero_recording_rejected(recording_factory):
    data, _, _ = recording_factory()
    fixed = data.isel(time=0, drop=True).copy(data=np.zeros(data.shape[1:]))
    fixed.values.flat[0] = 1
    with pytest.raises(ValueError, match="fixed image must not be constant"):
        register(data, fixed, fixed_winsorize_percentiles=(1, 99))
    with pytest.raises(ValueError, match="all-zero"):
        register(
            data.copy(data=np.zeros(data.shape)), moving_winsorize_percentiles=(1, 99)
        )


def test_clipped_constant_moving_frame_retains_failure_isolation(recording_factory):
    data, _, _ = recording_factory()
    data.values[1] = 0.1
    data.values[1, 0, 0, 0] = 10000
    result = register(
        data,
        moving_winsorize_percentiles=(1, 99),
        max_iterations=2,
        progress=False,
        return_transforms=True,
        device="cpu",
    )
    assert result.transforms is not None
    assert result.diagnostics.status[1] == "invalid_metric"
    assert result.diagnostics.n_iterations[1] == 0
    assert_array_equal(result.transforms[1], np.eye(4))
    assert result.diagnostics.n_iterations[2] == 2


@pytest.mark.parametrize("side", ["moving", "fixed"])
def test_percentile_overflow_is_descriptive(recording_factory, side):
    data, _, _ = recording_factory()
    data = data.isel(j=slice(0, 2), i=slice(0, 2)).astype(np.float64)
    data.values[:] = np.array([[-1e308, -1e308], [1e308, 1e308]])
    options: RegistrationOptions = {f"{side}_winsorize_percentiles": (40, 60)}
    with pytest.raises(ValueError, match=f"{side}.*percentile bounds overflowed"):
        register(data, data.isel(time=0, drop=True), progress=False, **options)


def test_endpoint_percentiles_preserve_float64_fit(recording_factory):
    data, _, _ = recording_factory()
    data = data.astype(np.float64)
    options: RegistrationOptions = {
        "max_iterations": 2,
        "return_transforms": True,
        "progress": False,
    }
    expected = register(data, data.isel(time=0, drop=True), **options)
    actual = register(
        data,
        data.isel(time=0, drop=True),
        moving_winsorize_percentiles=(0, 100),
        fixed_winsorize_percentiles=(0, 100),
        **options,
    )
    assert actual.transforms is not None and expected.transforms is not None
    assert_array_equal(actual.transforms, expected.transforms)
    assert_array_equal(actual.diagnostics.objective, expected.diagnostics.objective)
    assert_array_equal(actual.registered, expected.registered)


def test_disabled_clipping_preserves_default_exactly(recording_factory):
    data, _, _ = recording_factory()
    options: RegistrationOptions = {
        "max_iterations": 2,
        "progress": False,
        "device": "cpu",
    }
    default = register(data, **options)
    disabled = register(
        data,
        moving_winsorize_percentiles=None,
        fixed_winsorize_percentiles=None,
        **options,
    )
    assert_array_equal(disabled.registered, default.registered)
    assert_array_equal(disabled.valid, default.valid)
    assert_array_equal(disabled.diagnostics.objective, default.diagnostics.objective)
