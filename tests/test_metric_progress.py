"""Opt-in metric observation must not alter registration or frame progress."""

import numpy as np
import pytest
from registration_options import RegistrationOptions
import torch
from numpy.testing import assert_array_equal

from voxreg import register

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("transform", ["rigid", "affine"])
def test_metric_callback_preserves_results_and_reports_frame_clocks(
    recording_factory, device, transform
):
    data, _, _ = recording_factory()
    options: RegistrationOptions = {
        "device": device,
        "transform": transform,
        "batch_size": 2,
        "max_iterations": 5,
        "convergence_window_size": 6,
        "progress": False,
    }
    baseline = register(data, return_transforms=True, **options)
    events = []
    reported = register(
        data,
        return_transforms=True,
        **options,
        metric_callback=lambda frames, steps, metric: events.append(
            (frames, steps, metric)
        ),
        metric_interval=2,
    )
    for name in ["registered", "valid", "transforms"]:
        assert_array_equal(
            getattr(reported, name).values, getattr(baseline, name).values
        )
    for name in baseline.diagnostics:
        assert_array_equal(reported.diagnostics[name], baseline.diagnostics[name])
    assert len(events) == 8
    for offset in [0, 4]:
        frame_indices = events[offset][0]
        assert_array_equal(frame_indices, np.arange(0, 2) if offset == 0 else [2])
        for event, expected_clock in zip(
            events[offset : offset + 4], [0, 2, 4, 5], strict=True
        ):
            frames, steps, metric = event
            assert frames.dtype == steps.dtype == np.int64
            assert metric.dtype == np.float64
            assert_array_equal(steps[frames != 0], expected_clock)
            assert np.isfinite(metric).all()
            # Float64 callback storage does not increase float32 fitting precision.
            tolerance = 4 * np.finfo(np.float32).eps
            assert ((metric >= -1 - tolerance) & (metric <= 0)).all()
            assert_array_equal(steps[frames == 0], 0)


@pytest.mark.parametrize("device", DEVICES)
def test_metric_callback_arrays_are_owned_and_invalid_metrics_are_nan(
    recording_factory, device
):
    data, _, _ = recording_factory()
    data = data.copy(deep=True)
    data.values[1] = 0
    options: RegistrationOptions = {
        "device": device,
        "batch_size": 2,
        "max_iterations": 3,
        "progress": False,
    }
    baseline = register(data, return_transforms=True, **options)
    seen = []

    def mutate(frames, steps, metric):
        seen.append((frames.copy(), steps.copy(), metric.copy()))
        frames[:] = 999
        steps[:] = 999
        metric[:] = np.nan

    reported = register(
        data,
        return_transforms=True,
        **options,
        metric_callback=mutate,
        metric_interval=1,
    )
    for name in ["registered", "valid", "transforms"]:
        assert_array_equal(
            getattr(reported, name).values, getattr(baseline, name).values
        )
    for frames, steps, metric in seen:
        assert not (frames == 999).any()
        if 1 in frames:
            assert np.isnan(metric[frames == 1]).all()
            assert (steps[frames == 1] == 0).all()


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_metric_interval_validation(recording_factory, interval):
    data, _, _ = recording_factory()
    with pytest.raises(ValueError, match="metric_interval"):
        register(data, metric_interval=interval)


def test_metric_callback_validation_and_errors_propagate(recording_factory):
    data, _, _ = recording_factory()
    before = data.values.copy()
    with pytest.raises(TypeError, match="metric_callback"):
        register(data, metric_callback=False)  # ty: ignore[invalid-argument-type] -- deliberate invalid-input coverage.

    def fail(*_args):
        raise RuntimeError("metric display failed")

    with pytest.raises(RuntimeError, match="metric display failed"):
        register(data, metric_callback=fail)
    assert_array_equal(data.values, before)
