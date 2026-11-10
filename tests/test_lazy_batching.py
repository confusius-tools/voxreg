"""Temporal source chunks bound tasks independently of eager fitting batches."""

import dask
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from registration_options import RegistrationOptions

from voxreg import register, resample


@pytest.mark.parametrize("chunks", [(2, 3), (0, 2, 3)])
@pytest.mark.parametrize("override", [None, 1])
def test_registration_uses_source_chunks_and_worker_batch_policy(
    recording_factory, monkeypatch, device, override, chunks
):
    from voxreg import linear, registration

    data, _, _ = recording_factory()
    data = data.isel(
        time=[0, 1, 2, 1, 2], j=slice(0, 16), i=slice(0, 18)
    ).assign_coords(time=np.arange(5))
    expected = register(
        data,
        batch_size=override or 2,
        max_iterations=2,
        progress=False,
        device=device,
        return_transforms=True,
    )
    assert "forward" in expected
    choices, batches, tasks = [], [], []
    optimize = linear._optimize
    execute = registration.execute_registration

    def choose(moving, fixed, model, resolved):
        choices.append((moving.sizes["time"], resolved))
        return 2

    def fit(images, *args, **kwargs):
        batches.append(len(images))
        return optimize(images, *args, **kwargs)

    def run(moving, *args):
        tasks.append(moving.sizes["time"])
        return execute(moving, *args)

    monkeypatch.setattr(registration, "choose_batch_size", choose)
    monkeypatch.setattr(registration, "execute_registration", run)
    monkeypatch.setattr(linear, "_optimize", fit)
    lazy = register(
        data.chunk(time=chunks),
        batch_size=override,
        max_iterations=2,
        progress=False,
        device=device,
        return_transforms=True,
    )
    chunks = lazy["images"].to_dataset()["registered"].chunks
    assert chunks is not None
    assert chunks[0] == (2, 3)
    assert choices == batches == tasks == []
    image, matrices, diagnostics = dask.compute(
        lazy["images"].to_dataset()["registered"],
        lazy["forward"].to_dataset()["transform"],
        lazy["diagnostics"].to_dataset(),
        scheduler="synchronous",
    )
    assert sorted(tasks) == [2, 3]
    assert sorted(batches) == ([1] * 5 if override else [1, 2, 2])
    assert sorted(choices) == ([] if override else [(2, device), (3, device)])
    assert_array_equal(diagnostics.execution_task_size, [2, 2, 3, 3, 3])
    assert_array_equal(
        diagnostics.execution_batch_size, [1] * 5 if override else [2, 2, 2, 2, 1]
    )
    assert_array_equal(diagnostics.execution_batch_start, [0, 0, 2, 2, 2])
    assert_allclose(
        matrices, expected["forward"].to_dataset()["transform"], rtol=0, atol=1e-7
    )
    assert_allclose(
        image, expected["images"].to_dataset()["registered"], rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("override", [None, 1])
@pytest.mark.parametrize("spatial", [False, True])
def test_resampling_uses_source_or_map_chunks_and_worker_policy(
    recording_factory, monkeypatch, device, override, spatial
):
    from voxreg import resampling

    data, truth, _ = recording_factory(oblique=True, strided=True)
    data = data.isel(j=slice(0, 16), i=slice(0, 18))
    reference = data.isel(time=0, drop=True)
    maps = xr.DataArray(
        truth,
        dims=("time", "world_out", "world_in"),
        coords={
            "time": data.time,
            "world_out": ["z", "y", "x", "1"],
            "world_in": ["z", "y", "x", "1"],
        },
    )
    source = reference if spatial else data
    expected = resample(
        source, reference, maps, batch_size=override or 1, device=device
    )
    calls = []

    def choose(moving, fixed, model, resolved):
        calls.append((moving.sizes["time"], resolved))
        return 1

    monkeypatch.setattr(resampling, "choose_batch_size", choose)
    lazy_source = source.chunk(j=8) if spatial else source.chunk(time=(2, 1))
    lazy_maps = maps.chunk(time=(2, 1)) if spatial else maps
    image, valid = resample(
        lazy_source, reference, lazy_maps, batch_size=override, device=device
    )
    assert image.chunks is not None
    assert image.chunks[0] == (2, 1)
    assert calls == []
    actual = dask.compute(image, valid, scheduler="synchronous")
    assert sorted(calls) == ([] if override else [(1, device), (2, device)])
    assert_allclose(actual[0], expected[0], rtol=0, atol=1e-6)
    assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("kind", ["mask", "seed", "fixed"])
def test_registration_infers_auxiliary_chunks_without_moving_time_chunks(
    recording_factory, kind
):
    data, truth, _ = recording_factory()
    options: RegistrationOptions = {
        "max_iterations": 1,
        "progress": False,
        "device": "cpu",
    }
    fixed = None
    expected_chunks = (2, 1)
    if kind == "mask":
        options["moving_mask"] = data.copy(data=np.ones(data.shape, dtype=bool)).chunk(
            time=(2, 1)
        )
    elif kind == "seed":
        options["initial_transform"] = xr.DataArray(
            truth, dims=("time", "world_out", "world_in"), coords={"time": data.time}
        ).chunk(time=(2, 1))
    else:
        fixed = data.isel(time=0, drop=True).chunk(j=8)
        expected_chunks = (1, 1, 1)
    result = register(data, fixed, **options)
    assert result["images"].to_dataset()["registered"].chunks is not None
    chunks = result["images"].to_dataset()["registered"].chunks
    assert chunks is not None
    assert chunks[0] == expected_chunks
    eager_options = {
        name: value.compute() if isinstance(value, xr.DataArray) else value
        for name, value in options.items()
    }
    expected = register(
        data, None if fixed is None else fixed.compute(), **eager_options
    )
    assert_allclose(
        result["images"].to_dataset()["registered"].compute(scheduler="synchronous"),
        expected["images"].to_dataset()["registered"],
        rtol=0,
        atol=1e-6,
    )
