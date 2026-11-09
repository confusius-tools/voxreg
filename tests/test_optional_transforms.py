"""Transform export is opt-in; disabling it must skip allocation, not fitting."""

import numpy as np
import pytest
import xarray as xr
from registration_options import RegistrationOptions

from voxreg import linear, register, syn


@pytest.mark.parametrize("model", ["translation", "rigid", "affine", "syn"])
@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("recording", [True, False])
def test_optional_transforms(
    linear_pair_factory, monkeypatch, device, model, planar, recording
):
    fixed, moving, _ = linear_pair_factory(planar, model="rigid")
    if recording:
        moving = xr.concat(
            [moving, moving.copy(data=np.roll(moving.values, 1, axis=-1)), moving],
            dim=xr.IndexVariable("time", [2.0, 2.1, 2.2]),
        )
    options: RegistrationOptions = {
        "transform": model,
        "device": device,
        "max_iterations": 2,
        "batch_size": 2,
        "intensity_scaling": "none",
        "progress": False,
    }
    target = None if recording else fixed
    exported = register(moving, target, return_transforms=True, **options)
    assert exported.forward_transforms is not None
    assert exported.transforms is exported.forward_transforms
    assert exported.inverse_transforms is not None
    if model != "syn":
        np.testing.assert_allclose(
            exported.inverse_transforms.values @ exported.forward_transforms.values,
            np.broadcast_to(np.eye(4), exported.forward_transforms.shape),
            atol=1e-12,
        )
        assert exported.inverse_transforms.attrs["convention"] == (
            "moving-world-to-fixed-world pull"
        )

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "Transform export must not run when transforms are disabled"
        )

    for name in (
        "_build_world_field",
        "_build_inverse_world_field",
        "_build_labeled_field",
    ):
        monkeypatch.setattr(syn, name, unexpected)
    if model != "syn":
        monkeypatch.setattr(linear, "build_world_matrices", unexpected)
    omitted = register(moving, target, **options)
    assert omitted.forward_transforms is None and omitted.inverse_transforms is None
    xr.testing.assert_identical(omitted.registered, exported.registered)
    xr.testing.assert_identical(omitted.valid, exported.valid)
    # Timing alone changes when export is disabled.
    xr.testing.assert_equal(omitted.diagnostics, exported.diagnostics)
    if recording:
        np.testing.assert_array_equal(omitted.registered[0], moving[0])


@pytest.mark.parametrize("value", [None, 0, 1, "yes", np.bool_(True)])
def test_return_transforms_requires_bool(recording_factory, value):
    moving, _, _ = recording_factory()
    with pytest.raises(TypeError, match="return_transforms"):
        register(moving, return_transforms=value)
