"""Public device selection, numerical equivalence and failure propagation."""

import numpy as np
import pytest
import torch
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from oracles import max_pull_difference
from registration_options import RegistrationOptions

from voxreg import register, resample


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("transform", ["rigid", "syn", ("rigid", "affine", "syn")])
def test_auto_registration_matches_explicit_device(
    linear_pair_factory, monkeypatch, device, planar, transform
):
    fixed, moving, _ = linear_pair_factory(planar, "rigid")
    calls = []

    def available():
        calls.append(True)
        return device == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", available)
    options: RegistrationOptions = {
        "transform": transform,
        "max_iterations": 2,
        "shrink_factors": (2, 1),
        "smoothing_sigmas": (0.1, 0),
        "return_transforms": True,
        "progress": False,
    }
    explicit = register(moving, fixed, device=device, **options)
    calls.clear()
    # Omitting device exercises the public auto default, including stage dispatch.
    auto = register(moving, fixed, **options)
    assert len(calls) == 1
    xr.testing.assert_allclose(
        auto.registered, explicit.registered, rtol=1e-5, atol=2e-6
    )
    xr.testing.assert_identical(auto.valid, explicit.valid)
    assert (
        auto.forward_transforms is not None and explicit.forward_transforms is not None
    )
    assert (
        auto.inverse_transforms is not None and explicit.inverse_transforms is not None
    )
    assert (
        max_pull_difference(auto.forward_transforms, explicit.forward_transforms, fixed)
        < 0.001
    )
    assert (
        max_pull_difference(
            auto.inverse_transforms, explicit.inverse_transforms, moving
        )
        < 0.001
    )
    assert auto.diagnostics.attrs["device"] == device
    for name in explicit.diagnostics.data_vars:
        if np.issubdtype(explicit.diagnostics[name].dtype, np.floating):
            xr.testing.assert_allclose(
                auto.diagnostics[name], explicit.diagnostics[name], rtol=1e-5, atol=2e-6
            )
        else:
            xr.testing.assert_identical(
                auto.diagnostics[name], explicit.diagnostics[name]
            )
    if auto.stage_diagnostics is not None:
        assert all(stage.attrs["device"] == device for stage in auto.stage_diagnostics)


@pytest.mark.parametrize("interpolation", ["linear", "nearest", "bspline"])
@pytest.mark.parametrize("dense", [False, True])
def test_auto_resampling_matches_explicit_device(
    linear_pair_factory, monkeypatch, device, interpolation, dense
):
    fixed, moving, matrix = linear_pair_factory(False, "affine")
    if dense:
        from voxreg import position_to_world

        indices = np.indices(fixed.shape).reshape(3, -1)
        world = position_to_world(fixed) @ np.vstack((indices, np.ones(fixed.size)))
        displacement = ((matrix @ world - world)[:3]).T.reshape(*fixed.shape, 3)
        transform = fixed.expand_dims(world_component=["z", "y", "x"], axis=-1).copy(
            data=displacement
        )
        transform.attrs = {"spatial_units": "mm", "representation": "displacement"}
    else:
        transform = matrix
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device == "cuda")
    explicit = resample(
        moving, fixed, transform, device=device, interpolation=interpolation
    )
    auto = resample(
        moving, fixed, transform, device="auto", interpolation=interpolation
    )
    default = resample(moving, fixed, transform, interpolation=interpolation)
    for actual in (auto, default):
        xr.testing.assert_identical(actual[0], explicit[0])
        xr.testing.assert_identical(actual[1], explicit[1])


@pytest.mark.parametrize("operation", ["register", "resample"])
def test_explicit_cpu_does_not_probe_cuda(linear_pair_factory, monkeypatch, operation):
    _, moving, _ = linear_pair_factory(True, "rigid")

    def unexpected():
        raise AssertionError("Explicit CPU must not query CUDA availability.")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    if operation == "register":
        recording = moving.expand_dims(time=[0])
        result = register(recording, device="cpu", max_iterations=1, progress=False)
        assert result.diagnostics.attrs["device"] == "cpu"
        assert_array_equal(result.registered, recording)
    else:
        actual, valid = resample(moving, moving, np.eye(4), device="cpu")
        assert_allclose(actual, moving, rtol=0, atol=1e-12)
        assert valid.all()


@pytest.mark.parametrize("operation", ["register", "resample"])
@pytest.mark.parametrize("device", ["cuda", "invalid", None])
def test_device_validation(linear_pair_factory, monkeypatch, operation, device):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(
        ValueError, match="unavailable" if device == "cuda" else "device"
    ):
        if operation == "register":
            register(moving, fixed, device=device, progress=False)
        else:
            resample(moving, fixed, np.eye(4), device=device)


@pytest.mark.parametrize("operation", ["register", "resample"])
@pytest.mark.parametrize("failure_type", [torch.OutOfMemoryError, RuntimeError])
def test_auto_cuda_errors_propagate(
    linear_pair_factory, monkeypatch, operation, failure_type
):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    failure = failure_type("CUDA execution failure must propagate.")

    def fail():
        raise failure

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # Automatic batch sizing queries the selected device before allocating images.
    monkeypatch.setattr(torch.cuda, "mem_get_info", fail)
    with pytest.raises(failure_type) as caught:
        if operation == "register":
            register(moving, fixed, device="auto", progress=False)
        else:
            resample(moving, fixed, np.eye(4), device="auto")
    assert caught.value is failure
