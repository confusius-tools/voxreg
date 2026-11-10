"""Lazy public APIs share bounded eager tasks without loading entire recordings."""

import threading
import time
import weakref

import dask
import dask.array as da
import numpy as np
import pytest
import torch
import xarray as xr
from confusius._utils.geometry import VoxelToWorldIndex
from dask import delayed
from distributed import Client, LocalCluster
from numpy.testing import assert_allclose, assert_array_equal
from oracles import max_pull_difference
from registration_options import RegistrationOptions

from voxreg import Stage, position_to_world, register, resample
from voxreg.geometry import validate_registration_geometry


@pytest.mark.parametrize(
    "model",
    [
        "translation",
        "rigid",
        "affine",
        "syn",
        (
            Stage(
                "rigid",
                shrink_factors=(2, 1),
                smoothing_sigmas=(0.1, 0),
                max_iterations=(1, 2),
            ),
            "syn",
        ),
    ],
)
@pytest.mark.parametrize("explicit", [True, False])
def test_lazy_registration_matches_eager_with_shared_outputs(
    recording_factory, model, explicit, device, monkeypatch
):
    from voxreg import registration

    data, _, _ = recording_factory(oblique=True, strided=True)
    data = data.isel(j=slice(0, 24), i=slice(0, 28)).astype(np.float64)
    fixed = data.isel(time=0, drop=True) if explicit else None
    mask = data.copy(data=np.ones(data.shape, dtype=bool))
    mask.values[..., :2] = False
    options: RegistrationOptions = {
        "transform": model,
        "max_iterations": 2,
        "return_transforms": True,
        "moving_intensity_scaling": "log1p",
        "moving_winsorize_percentiles": (1, 99),
        "moving_mask": mask,
        "device": device,
        "progress": False,
    }
    eager = register(data, fixed, **options)
    calls = []
    execute = registration.execute_registration

    def record(moving, reference, specification, settings):
        calls.append(moving.sizes["time"])
        assert isinstance(moving.data, np.ndarray)
        return execute(moving, reference, specification, settings)

    monkeypatch.setattr(registration, "execute_registration", record)
    options["moving_mask"] = mask.chunk(time=1)
    lazy = register(data.chunk(time=1, j=12, i=14), fixed, **options)
    assert calls == []
    assert lazy.transforms is not None and lazy.inverse_transforms is not None
    assert eager.transforms is not None and eager.inverse_transforms is not None
    for output in (
        lazy.registered,
        lazy.valid,
        lazy.transforms,
        lazy.inverse_transforms,
    ):
        assert isinstance(output.data, da.Array)
        assert max(output.data.chunks[0]) == 1
    outputs = [
        lazy.registered,
        lazy.valid,
        lazy.transforms,
        lazy.inverse_transforms,
        lazy.diagnostics,
    ]
    if lazy.stage_diagnostics:
        outputs.extend(lazy.stage_diagnostics)
    computed = dask.compute(*outputs, scheduler="synchronous")
    image, valid, forward, inverse, diagnostics = computed[:5]
    assert calls == [1, 1, 1]
    assert (
        max_pull_difference(
            forward,
            eager.transforms,
            fixed if explicit else data.isel(time=0, drop=True),
        )
        < 0.01
    )
    assert_allclose(image, eager.registered, rtol=1e-5, atol=1e-5)
    assert_array_equal(valid, eager.valid)
    assert_allclose(inverse, eager.inverse_transforms, rtol=0, atol=1e-5)
    for name in eager.diagnostics.data_vars:
        if eager.diagnostics[name].dtype.kind in "USiu":
            assert_array_equal(diagnostics[name], eager.diagnostics[name])
        else:
            assert_allclose(
                diagnostics[name], eager.diagnostics[name], rtol=1e-5, atol=1e-5
            )
    assert_array_equal(diagnostics.execution_device, [device] * 3)
    assert (diagnostics.batch_prepare_seconds >= 0).all()
    assert_allclose(
        position_to_world(image), position_to_world(eager.registered), atol=0, rtol=0
    )
    validate_registration_geometry(image)
    validate_registration_geometry(valid)
    if lazy.stage_diagnostics:
        assert eager.stage_diagnostics is not None
        for expected, actual in zip(eager.stage_diagnostics, computed[5:], strict=True):
            assert_array_equal(actual.status, expected.status)
            assert_allclose(actual.objective, expected.objective, rtol=1e-5, atol=1e-5)
    if not explicit:
        assert_array_equal(image[0], data[0])
        assert diagnostics.status[0] == "reference"


@pytest.mark.parametrize("batch", [1, 2, 3])
@pytest.mark.parametrize("initialization", [None, "center_moments"])
def test_lazy_chunk_independence_failure_isolation_and_reference(
    recording_factory, batch, initialization
):
    data, _, _ = recording_factory()
    data = data.isel(j=slice(0, 24), i=slice(0, 28)).astype(np.float64)
    data.values[1] = 0
    if initialization == "center_moments":
        data.values[1] = 0.1
    options: RegistrationOptions = {
        "initialization": initialization,
        "max_iterations": 2,
        "return_transforms": True,
        "device": "cpu",
        "progress": False,
    }
    expected = register(data, **options)
    assert expected.transforms is not None
    lazy = register(data.chunk(time=batch), batch_size=batch, **options)
    image, transforms, diagnostic = dask.compute(
        lazy.registered, lazy.transforms, lazy.diagnostics, scheduler="synchronous"
    )
    assert_allclose(transforms, expected.transforms, rtol=0, atol=1e-9)
    assert_allclose(image, expected.registered, rtol=0, atol=1e-6)
    assert_array_equal(diagnostic.status, expected.diagnostics.status)
    assert_array_equal(
        diagnostic.execution_batch_size,
        [min(batch, 3 - index // batch * batch) for index in range(3)],
    )
    assert_array_equal(image[0], data[0])
    assert diagnostic.n_iterations[1] == 0
    assert diagnostic.status[1] == "invalid_metric"


def test_sources_are_deferred_shared_and_only_required_frames_load(
    recording_factory, monkeypatch
):
    from voxreg import registration

    data, _, _ = recording_factory()
    data = data.isel(j=slice(0, 16), i=slice(0, 16))
    loaded, fitted = [], []
    original = registration.execute_registration

    def read(frame):
        loaded.append(frame)
        return data.values[frame : frame + 1]

    def fit(moving, fixed, model, settings):
        fitted.append(moving.time.values.tolist())
        return original(moving, fixed, model, settings)

    monkeypatch.setattr(registration, "execute_registration", fit)
    source = da.concatenate(
        [
            da.from_delayed(
                delayed(read)(frame), shape=(1, *data.shape[1:]), dtype=data.dtype
            )
            for frame in range(3)
        ]
    )
    lazy = register(
        data.copy(data=source),
        max_iterations=1,
        progress=False,
        device="cpu",
        return_transforms=True,
    )
    assert loaded == fitted == []
    lazy.registered.isel(time=2).compute(scheduler="synchronous")
    assert sorted(loaded) == [0, 2]
    assert fitted == [[data.time.values[2]]]


@pytest.mark.parametrize(
    "case,match",
    [
        ("time", "time"),
        ("mask", "time labels"),
        ("bounds", "percentiles"),
        ("stages", "increase"),
        ("seed", "time labels"),
        ("device", "device"),
    ],
)
def test_metadata_errors_are_eager_without_loading_pixels(
    recording_factory, case, match
):
    data, truth, _ = recording_factory()

    def fail():
        raise AssertionError("Graph construction must not load voxel data.")

    lazy = data.copy(
        data=da.from_delayed(
            delayed(fail)(), shape=data.shape, dtype=data.dtype
        ).rechunk((1, *data.shape[1:]))
    )
    options: RegistrationOptions = {"progress": False}
    if case == "time":
        lazy = lazy.assign_coords(time=[0, 0, 1])
    elif case == "mask":
        options["moving_mask"] = data.copy(
            data=np.ones(data.shape, dtype=bool)
        ).assign_coords(time=[1, 2, 3])
    elif case == "bounds":
        options["moving_winsorize_percentiles"] = (99, 1)
    elif case == "stages":
        options["transform"] = ("rigid", "translation")
    elif case == "seed":
        options["initial_transform"] = xr.DataArray(
            truth, dims=("time", "world_out", "world_in"), coords={"time": [1, 2, 3]}
        )
    else:
        options["device"] = "bad"
    with pytest.raises(ValueError, match=match):
        register(lazy, **options)


def test_deferred_data_errors_and_oversized_source_blocks(recording_factory):
    data, _, _ = recording_factory()
    oversized_mask = data.copy(data=np.ones(data.shape, dtype=bool)).chunk(time=3)
    with pytest.raises(ValueError, match="source time chunks"):
        register(data.chunk(time=1), moving_mask=oversized_mask)
    data.values[2, 0, 1, 1] = np.nan
    lazy = register(data.chunk(time=1), max_iterations=1, progress=False, device="cpu")
    with pytest.raises(ValueError, match="finite"):
        lazy.registered.isel(time=2).compute(scheduler="synchronous")
    with pytest.raises(TypeError, match="spatial image pairs"):
        register(data.isel(time=0, drop=True).chunk(j=8), data.isel(time=0, drop=True))


@pytest.mark.parametrize("interpolation", ["linear", "nearest", "bspline"])
@pytest.mark.parametrize(
    "lazy_image,lazy_maps", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("dense", [False, True])
def test_lazy_resampling_matches_eager(
    recording_factory, interpolation, lazy_image, lazy_maps, dense, device
):
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
    if dense:
        affine = position_to_world(reference)
        points = affine @ np.vstack(
            (np.indices(reference.shape).reshape(3, -1), np.ones((1, reference.size)))
        )
        fields = (
            np.einsum("bij,jp->bip", truth - np.eye(4), points)[:, :3]
            .transpose(0, 2, 1)
            .reshape(*data.shape, 3)
        )
        maps = data.expand_dims(world_component=["z", "y", "x"], axis=-1).copy(
            data=fields
        )
        maps.attrs = {"spatial_units": "mm", "representation": "displacement"}
    eager = resample(data, reference, maps, interpolation=interpolation, device=device)
    image, valid = resample(
        data.chunk(time=1) if lazy_image else data,
        reference,
        maps.chunk(time=1) if lazy_maps else maps,
        interpolation=interpolation,
        device=device,
    )
    assert isinstance(image.data, da.Array) and isinstance(valid.data, da.Array)
    actual = dask.compute(image, valid, scheduler="synchronous")
    assert_allclose(actual[0], eager[0], rtol=0, atol=1e-5)
    assert_array_equal(actual[1], eager[1])
    assert_array_equal(actual[0].time, data.time)
    validate_registration_geometry(actual[0])


def test_spatial_and_time_map_broadcasting_and_unused_reference(recording_factory):
    data, truth, _ = recording_factory()
    data = data.isel(j=slice(0, 16), i=slice(0, 18))
    spatial = data.isel(time=0, drop=True)

    def unexpected():
        raise AssertionError("Reference intensities are not used by resampling.")

    reference = spatial.copy(
        data=da.from_delayed(
            delayed(unexpected)(), shape=spatial.shape, dtype=spatial.dtype
        )
    )
    maps = xr.DataArray(
        truth, dims=("time", "world_out", "world_in"), coords={"time": data.time}
    ).chunk(time=1)
    lazy, valid = resample(spatial, reference, maps, device="cpu")
    expected = resample(spatial, spatial, maps.compute(), device="cpu")
    actual = dask.compute(lazy, valid, scheduler="synchronous")
    assert_array_equal(actual[0], expected[0])
    assert_array_equal(actual[1], expected[1])
    shared, support = resample(spatial.chunk(j=8), reference, np.eye(4), device="cpu")
    result = dask.compute(shared, support, scheduler="synchronous")
    assert "time" not in result[0].dims
    assert_array_equal(result[0], spatial)
    assert_array_equal(result[1], True)


def test_chained_resampling_preserves_shared_registration_tasks(
    recording_factory, monkeypatch
):
    from voxreg import registration

    data, _, _ = recording_factory()
    data = data.isel(j=slice(0, 16), i=slice(0, 18))
    execute = registration.execute_registration
    calls = []

    def fit(*args):
        calls.append(1)
        return execute(*args)

    monkeypatch.setattr(registration, "execute_registration", fit)
    result = register(
        data.chunk(time=1),
        max_iterations=1,
        return_transforms=True,
        progress=False,
        device="cpu",
    )
    assert result.inverse_transforms is not None
    image, valid = resample(
        result.registered,
        data.isel(time=0, drop=True),
        result.inverse_transforms,
        device="cpu",
    )
    computed = dask.compute(
        image, valid, result.diagnostics, result.valid, scheduler="synchronous"
    )
    assert len(calls) == 3
    expected = register(
        data, max_iterations=1, return_transforms=True, progress=False, device="cpu"
    )
    assert expected.inverse_transforms is not None
    expected_image, expected_valid = resample(
        expected.registered,
        data.isel(time=0, drop=True),
        expected.inverse_transforms,
        device="cpu",
    )
    assert_allclose(computed[0], expected_image, rtol=0, atol=1e-6)
    assert_array_equal(computed[1], expected_valid)
    assert_array_equal(computed[3][0], True)


def test_callbacks_execute_in_tasks_with_global_indices(recording_factory):
    data, _, _ = recording_factory()
    metrics, progress = [], []

    def report(indices, steps, objective):
        metrics.extend(indices.tolist())

    result = register(
        data.chunk(time=1),
        max_iterations=2,
        metric_interval=1,
        metric_callback=report,
        progress=lambda done, total: progress.append((done, total)),
        device="cpu",
    )
    assert not metrics and not progress
    result.registered.compute(scheduler="synchronous")
    assert set(metrics) == {0, 1, 2}
    assert sorted(progress) == [(1, 3), (2, 3), (3, 3)]


def test_threaded_execution_bounds_fitting_concurrency(recording_factory, monkeypatch):
    from voxreg import registration

    data, _, _ = recording_factory()
    execute = registration.execute_registration
    counts = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def bounded(*args):
        with lock:
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
        try:
            time.sleep(0.01)
            return execute(*args)
        finally:
            with lock:
                counts["active"] -= 1

    monkeypatch.setattr(registration, "execute_registration", bounded)
    result = register(
        data.chunk(time=1), max_iterations=1, progress=False, device="cpu"
    )
    result.registered.compute(scheduler="threads", num_workers=3)
    assert counts == {"active": 0, "peak": 1}


@pytest.mark.parametrize("processes", [False, True])
def test_distributed_worker_execution_and_joint_zarr_storage(
    recording_factory, tmp_path, processes
):
    data, _, _ = recording_factory()
    data = data.isel(j=slice(0, 16), i=slice(0, 18))
    expected = register(
        data, max_iterations=1, progress=False, device="cpu", return_transforms=True
    )

    def callback(indices, steps, objectives):
        from distributed import get_worker

        get_worker().log_event("voxreg-test-callback", {"indices": indices.tolist()})

    lazy = register(
        data.chunk(time=1),
        max_iterations=1,
        progress=False,
        metric_callback=callback,
        device="cpu",
        return_transforms=True,
    )
    assert expected.transforms is not None and lazy.transforms is not None
    with (
        LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=processes,
            dashboard_address=None,
        ) as cluster,
        Client(cluster) as client,
    ):
        outputs = client.compute(
            [lazy.registered, lazy.valid, lazy.transforms, lazy.diagnostics]
        )
        image, valid, transforms, diagnostics = client.gather(outputs)
        assert_allclose(image, expected.registered, rtol=0, atol=1e-6)
        assert_array_equal(valid, expected.valid)
        assert_allclose(transforms, expected.transforms, rtol=0, atol=1e-9)
        assert_array_equal(diagnostics.execution_device, ["cpu"] * 3)
        events = client.get_events("voxreg-test-callback")
        assert {index for _, event in events for index in event["indices"]} == {0, 1, 2}
        index = lazy.registered.xindexes["z"]
        assert isinstance(index, VoxelToWorldIndex)
        attrs = {
            "voxel_to_world": index.voxel_to_world.tolist(),
            "voxel_to_world_units": index.units,
        }
        tree = xr.DataTree.from_dict(
            {
                "images": lazy.registered.drop_vars(["z", "y", "x"])
                .assign_attrs(attrs)
                .to_dataset(name="image"),
                "validity": lazy.valid.drop_vars(["z", "y", "x"])
                .assign_attrs(attrs)
                .to_dataset(name="valid"),
                "diagnostics": lazy.diagnostics,
                "forward": lazy.transforms.to_dataset(name="matrix"),
            }
        )
        tree.to_zarr(tmp_path / "result.zarr", compute=False, zarr_format=2).compute()
        assert len(client.get_events("voxreg-test-callback")) == 2 * len(events)
    from confusius.io import load

    stored = load(tmp_path / "result.zarr", variable="image", group="images")
    stored_valid = load(tmp_path / "result.zarr", variable="valid", group="validity")
    validate_registration_geometry(stored)
    validate_registration_geometry(stored_valid)
    assert_array_equal(stored, image)
    assert_array_equal(stored_valid, valid)
    assert_allclose(position_to_world(stored), position_to_world(image), rtol=0, atol=0)


@pytest.mark.parametrize("frames", [4, 24])
def test_incremental_storage_retains_bounded_source_volumes(
    recording_factory, tmp_path, frames
):
    data, _, _ = recording_factory()
    base = data.isel(time=0, drop=True).isel(j=slice(0, 16), i=slice(0, 18))
    live = {"current": 0, "peak": 0, "read": 0}

    def release():
        live["current"] -= 1

    def read(frame):
        values = base.values[None].copy() + frame * np.float32(0.001)
        live["read"] += 1
        live["current"] += 1
        live["peak"] = max(live["peak"], live["current"])
        weakref.finalize(values, release)
        return values

    source = da.concatenate(
        [
            da.from_delayed(
                delayed(read)(frame), shape=(1, *base.shape), dtype=base.dtype
            )
            for frame in range(frames)
        ]
    )
    recording = base.expand_dims(time=np.arange(frames)).copy(data=source)
    result = register(recording, max_iterations=1, progress=False, device="cpu")
    assert live["read"] == 0
    result.registered.to_dataset(name="image").to_zarr(
        tmp_path / "stream.zarr", compute=False, zarr_format=2
    ).compute(scheduler="synchronous")
    assert live["read"] == frames
    assert live["peak"] <= 3
    stored = xr.open_zarr(tmp_path / "stream.zarr").image
    assert_array_equal(stored.isel(time=0), base)
    assert_allclose(
        stored.isel(time=-1), base + (frames - 1) * np.float32(0.001), atol=1e-6
    )


@pytest.mark.parametrize("scheduler", ["threads", "processes"])
def test_local_parallel_schedulers_preserve_output(recording_factory, scheduler):
    data, _, _ = recording_factory()
    data = data.isel(j=slice(0, 12), i=slice(0, 14))
    expected = register(data, max_iterations=1, progress=False, device="cpu")
    result = register(
        data.chunk(time=1), max_iterations=1, progress=False, device="cpu"
    )
    actual = result.registered.compute(scheduler=scheduler, num_workers=2)
    assert_array_equal(actual, expected.registered)


@pytest.mark.parametrize("model", ["rigid", "syn"])
def test_distributed_device_execution_and_resampling(recording_factory, device, model):
    data, _, _ = recording_factory()
    data = data.isel(j=slice(0, 12), i=slice(0, 14))
    options: RegistrationOptions = {
        "transform": model,
        "max_iterations": 1,
        "return_transforms": True,
        "progress": False,
        "device": device,
    }
    expected = register(data, **options)
    result = register(data.chunk(time=1), **options)
    assert result.transforms is not None
    image, support = resample(
        data.chunk(time=1),
        data.isel(time=0, drop=True),
        result.transforms,
        device=device,
    )
    with (
        LocalCluster(
            n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None
        ) as cluster,
        Client(cluster) as client,
    ):
        actual, valid, diagnostic = client.gather(
            client.compute([image, support, result.diagnostics])
        )
    assert_allclose(actual, expected.registered, rtol=0, atol=1e-5)
    assert_array_equal(valid, expected.valid)
    assert_array_equal(diagnostic.execution_device, [device] * 3)


def test_auto_device_is_resolved_at_execution_time(recording_factory, monkeypatch):
    data, _, _ = recording_factory()
    available = torch.cuda.is_available()
    calls = []

    def probe():
        calls.append(True)
        return available

    monkeypatch.setattr(torch.cuda, "is_available", probe)
    lazy = register(data.chunk(time=1), max_iterations=1, progress=False)
    assert not calls
    diagnostic = lazy.diagnostics.compute(scheduler="synchronous")
    assert calls
    assert_array_equal(
        diagnostic.execution_device, ["cuda" if available else "cpu"] * 3
    )


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("model", ["translation", "rigid", "affine", "syn"])
def test_unequal_grids_spatial_lazy_fixed_and_optional_maps(
    linear_pair_factory, planar, model
):
    fixed, moving, _ = linear_pair_factory(planar, "rigid")
    recording = moving.expand_dims(time=[10, 20]).chunk(time=1)
    fixed_lazy = fixed.chunk({dim: 8 for dim in fixed.dims})
    options: RegistrationOptions = {
        "transform": model,
        "max_iterations": 1,
        "device": "cpu",
        "progress": False,
    }
    expected = register(recording.compute(), fixed, **options)
    actual = register(recording, fixed_lazy, **options)
    assert actual.transforms is None and actual.inverse_transforms is None
    image, support, diagnostics = dask.compute(
        actual.registered, actual.valid, actual.diagnostics, scheduler="synchronous"
    )
    assert_allclose(image, expected.registered, rtol=0, atol=1e-9)
    assert_array_equal(support, expected.valid)
    assert_array_equal(diagnostics.n_iterations, expected.diagnostics.n_iterations)
    assert_allclose(position_to_world(image), position_to_world(fixed), atol=0)


@pytest.mark.parametrize("dtype", [np.float64, np.uint16, np.bool_])
@pytest.mark.parametrize("shared", [True, False])
def test_shared_and_per_frame_lazy_fields_preserve_units_dtype_and_support(
    recording_factory, dtype, shared
):
    data, _, _ = recording_factory(oblique=True, strided=True)
    data = data.isel(j=slice(0, 12), i=slice(0, 14))
    data = data.copy(data=(data.values * 100).astype(dtype))
    reference = data.isel(time=0, drop=True)
    values = np.zeros((*data.shape, 3))
    values[..., 2] = 15
    field = data.expand_dims(world_component=["z", "y", "x"], axis=-1).copy(data=values)
    field.attrs = {"representation": "displacement", "spatial_units": "um"}
    if shared:
        field = field.isel(time=0, drop=True)
    eager = resample(data, reference, field, interpolation="nearest", device="cpu")
    lazy = resample(
        data,
        reference,
        field.chunk({dim: 1 if dim == "time" else -1 for dim in field.dims}),
        interpolation="nearest",
        device="cpu",
    )
    actual = dask.compute(*lazy, scheduler="synchronous")
    assert_array_equal(actual[0], eager[0])
    assert_array_equal(actual[1], eager[1])
    assert actual[0].dtype == dtype


def test_lazy_initial_seeds_keep_global_implicit_reference_rules(recording_factory):
    data, truth, _ = recording_factory()
    seeds = xr.DataArray(
        truth, dims=("time", "world_out", "world_in"), coords={"time": data.time}
    ).chunk(time=1)
    result = register(
        data,
        initial_transform=seeds,
        max_iterations=1,
        progress=False,
        device="cpu",
        return_transforms=True,
    )
    expected = register(
        data,
        initial_transform=seeds.compute(),
        max_iterations=1,
        progress=False,
        device="cpu",
        return_transforms=True,
    )
    assert expected.transforms is not None and result.transforms is not None
    assert_allclose(
        result.transforms.compute(scheduler="synchronous"),
        expected.transforms,
        rtol=0,
        atol=1e-9,
    )
    bad = seeds.compute()
    bad.values[0, 2, 3] = 0.1
    invalid = register(
        data.chunk(time=1),
        initial_transform=bad,
        max_iterations=1,
        progress=False,
        device="cpu",
    )
    with pytest.raises(ValueError, match="implicit reference"):
        invalid.registered.isel(time=0).compute(scheduler="synchronous")


def test_resampling_metadata_failures_before_loading(recording_factory):
    data, truth, _ = recording_factory()
    lazy = data.chunk(time=1)
    maps = xr.DataArray(
        truth, dims=("time", "world_out", "world_in"), coords={"time": [1, 2, 3]}
    ).chunk(time=1)
    with pytest.raises(ValueError, match="time labels"):
        resample(lazy, data.isel(time=0, drop=True), maps)
    maps = maps.assign_coords(time=data.time)
    maps.data = maps.data.map_blocks(lambda x: np.full_like(x, np.nan))
    image, _ = resample(lazy, data.isel(time=0, drop=True), maps, device="cpu")
    with pytest.raises(ValueError, match="finite"):
        image.compute(scheduler="synchronous")
