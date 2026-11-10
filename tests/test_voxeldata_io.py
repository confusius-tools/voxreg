"""Complete registration trees save through one shared VoxelData writer."""

import dask
import dask.array as da
import numpy as np
import pytest
import voxeldata as vd
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from registration_options import RegistrationOptions

from voxreg import position_to_world, register, resample


@pytest.mark.parametrize("model", ["rigid", "syn", ("rigid", "syn")])
@pytest.mark.parametrize("lazy", [False, True])
def test_registration_tree_roundtrip(
    recording_factory, tmp_path, monkeypatch, model, lazy
):
    """Save IQ-derived images, maps on distinct grids and stages without refits."""
    from voxreg import registration

    data, _, _ = recording_factory(oblique=True, strided=True)
    data = data.isel(j=slice(0, 16), i=slice(0, 18))
    iq = (data.astype(np.complex64) + np.complex64(1j)).vd.set_affine("lab", np.eye(4))
    fixed = abs(iq.isel(time=0, drop=True).isel(j=slice(1, 15), i=slice(1, 17)))
    options: RegistrationOptions = {
        "device": "cpu",
        "batch_size": 1,
        "max_iterations": 1,
        "return_transforms": True,
        "progress": False,
        "transform": model,
    }
    expected = register(abs(iq), fixed, **options)
    vd.save(iq.chunk(time=1), tmp_path / "iq.zarr")
    calls = []
    execute = registration.execute_registration

    def record(moving, target, specification, settings):
        calls.append(moving.sizes["time"])
        return execute(moving, target, specification, settings)

    monkeypatch.setattr(registration, "execute_registration", record)
    with vd.load_array(tmp_path / "iq.zarr", chunks={}) as source:
        amplitude = abs(source)
        assert isinstance(amplitude.data, da.Array)
        result = register(amplitude if lazy else amplitude.compute(), fixed, **options)
        assert isinstance(result, xr.DataTree)
        assert set(result.children) == {
            "images",
            "diagnostics",
            "forward",
            "inverse",
        } | ({"stages"} if isinstance(model, tuple) else set())
        assert set(result["images"].data_vars) == {"registered", "valid"}
        for node in result.leaves:
            for array in node.to_dataset().data_vars.values():
                if "k" in array.dims:
                    vd.validate_voxeldata(array)
        if lazy:
            assert calls == []
        path = tmp_path / "results.zarr"
        with dask.config.set(scheduler="synchronous"):
            write = vd.save(result, path, compute=False)
            if lazy:
                assert calls == []
            write.compute()
    assert calls == ([1] * data.sizes["time"] if lazy else [data.sizes["time"]])
    with vd.load_tree(path) as stored:
        assert set(stored.groups) == set(result.groups)
        assert isinstance(stored["images"].to_dataset()["registered"].data, da.Array)
        assert stored["images"].to_dataset()["valid"].dtype == np.bool_
        for key in (
            "images/registered",
            "images/valid",
            "forward/transform",
            "inverse/transform",
        ):
            group, variable = key.split("/")
            restored = stored[group].to_dataset()[variable]
            reference = expected[group].to_dataset()[variable]
            assert restored.dims == reference.dims
            actual = restored
            assert_allclose(actual.compute(), reference, rtol=0, atol=1e-6)
            assert actual.dtype == reference.dtype
            assert_array_equal(actual.time, data.time)
            assert actual.time.attrs == data.time.attrs
            if "k" in actual.dims:
                vd.validate_voxeldata(restored)
                assert_allclose(
                    position_to_world(
                        actual.isel(world_component=0, drop=True)
                        if "world_component" in actual.dims
                        else actual
                    ),
                    position_to_world(fixed if key != "inverse/transform" else data),
                )
                assert_array_equal(actual.vd.affines["lab"], np.eye(4))
        sampled, valid = resample(
            abs(iq), fixed, stored["forward"].to_dataset()["transform"], device="cpu"
        )
        sampled, valid = dask.compute(sampled, valid, scheduler="synchronous")
        assert_allclose(
            sampled, expected["images"].to_dataset()["registered"], rtol=2e-6, atol=2e-6
        )
        assert_array_equal(valid, expected["images"].to_dataset()["valid"])
        for node in ("diagnostics", "stages/0", "stages/1"):
            if node not in result:
                continue
            for name in ("objective", "n_iterations", "status"):
                assert_array_equal(stored[node][name].compute(), expected[node][name])
        if model != "rigid":
            assert (
                stored["forward"].to_dataset()["transform"].sizes["j"]
                == fixed.sizes["j"]
            )
            assert (
                stored["inverse"].to_dataset()["transform"].sizes["j"]
                == data.sizes["j"]
            )


def test_tree_compute_and_optional_groups(recording_factory):
    """Computing the native tree preserves the same API and reference values."""
    data, _, _ = recording_factory()
    data = data.isel(j=slice(0, 12), i=slice(0, 14))
    result = register(
        data.chunk(time=1), device="cpu", max_iterations=1, progress=False
    )
    assert set(result.children) == {"images", "diagnostics"}
    computed = result.compute(scheduler="synchronous")
    assert isinstance(computed, xr.DataTree)
    assert isinstance(computed["images"].to_dataset()["registered"].data, np.ndarray)
    assert_array_equal(
        computed["images"].to_dataset()["registered"].isel(time=0), data.isel(time=0)
    )
    assert computed["diagnostics/status"].isel(time=0) == "reference"
