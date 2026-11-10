"""VoxelData I/O shares registration tasks across related lazy outputs."""

import dask
import dask.array as da
import numpy as np
import pytest
import voxeldata as vd
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from registration_options import RegistrationOptions

from voxreg import position_to_world, register


@pytest.mark.parametrize("tree", [False, True])
def test_load_magnitude_register_and_save_share_tasks(
    recording_factory, tmp_path, monkeypatch, tree
):
    """Round-trip IQ-derived images and masks without duplicate registration."""
    from voxreg import registration

    data, _, _ = recording_factory(oblique=True, strided=True)
    data = data.isel(j=slice(0, 16), i=slice(0, 18))
    iq = (data.astype(np.complex64) + np.complex64(1j)).vd.set_affine("lab", np.eye(4))
    options: RegistrationOptions = {
        "device": "cpu",
        "batch_size": 1,
        "max_iterations": 1,
        "progress": False,
    }
    expected = register(abs(iq), **options)
    vd.save(iq.chunk(time=1), tmp_path / "iq.zarr")
    calls = []
    execute = registration.execute_registration

    def record(moving, fixed, specification, settings):
        calls.append(moving.sizes["time"])
        return execute(moving, fixed, specification, settings)

    monkeypatch.setattr(registration, "execute_registration", record)
    with vd.load_array(tmp_path / "iq.zarr", chunks={}) as source:
        amplitude = abs(source)
        assert isinstance(amplitude.data, da.Array)
        result = register(amplitude, **options)
        images = xr.Dataset({"registered": result.registered, "valid": result.valid})
        output = (
            xr.DataTree.from_dict(
                {"/images": images, "/diagnostics": result.diagnostics}
            )
            if tree
            else images
        )
        path = tmp_path / "results.zarr"
        with dask.config.set(scheduler="synchronous"):
            write = vd.save(output, path, compute=False)
            assert calls == []
            write.compute()
    assert calls == [1] * data.sizes["time"]
    with vd.load_dataset(path, group="images" if tree else None) as stored:
        assert isinstance(stored["registered"].data, da.Array)
        assert stored["valid"].dtype == np.bool_
        assert_allclose(stored["registered"].compute(), expected.registered, atol=1e-6)
        assert_array_equal(stored["valid"].compute(), expected.valid)
        for name in ("registered", "valid"):
            vd.validate_voxeldata(stored[name])
            assert_allclose(position_to_world(stored[name]), position_to_world(data))
            assert_array_equal(stored[name].time, data.time)
            assert stored[name].time.attrs == data.time.attrs
            assert_array_equal(stored[name].vd.affines["lab"], np.eye(4))
    if tree:
        with vd.load_tree(path) as stored_tree:
            for name in ("objective", "n_iterations", "status"):
                assert_array_equal(
                    stored_tree["diagnostics"][name].compute(),
                    expected.diagnostics[name],
                )
