"""Batch progress is device-independent and leaves numerical execution unchanged."""

from io import StringIO

import dask
import dask.array as da
import numpy as np
import pytest
import torch
import voxeldata as vd
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from registration_options import RegistrationOptions
from rich.console import Console
from voxeldata import create_voxeldata

import voxreg.progress as vr_progress
from voxreg import linear, register


@pytest.mark.parametrize("scheduler", ["synchronous", "threads", "processes"])
@pytest.mark.parametrize("output", ["image", "tree", "save"])
def test_lazy_default_progress_is_one_frame_bar(
    recording_factory, tmp_path, rich_bars, scheduler, output
):
    """Local lazy execution counts source frames once through fused graphs."""
    data, _, _ = recording_factory()
    data = data.isel(j=slice(0, 12), i=slice(0, 14))
    result = register(data.chunk(time=2), max_iterations=1, device="cpu")
    assert not rich_bars
    with dask.config.set(scheduler=scheduler, num_workers=2):
        if output == "save":
            vd.save(result, tmp_path / "result.zarr")
        elif output == "tree":
            result.compute()
        else:
            result["images"].to_dataset()["registered"].compute()
    assert len(rich_bars) == 1
    assert rich_bars[0].tasks[0].completed == rich_bars[0].tasks[0].total == len(data)
    assert not rich_bars[0].live.is_started


@pytest.mark.parametrize("prefix_length", [40, 50, 60])
def test_lazy_progress_counts_selected_chunk_through_long_fused_keys(
    recording_factory, rich_bars, prefix_length
):
    """Truncated fused names must not introduce partial frame identities."""
    data, _, _ = recording_factory()
    base = data.isel(time=0, drop=True).isel(j=slice(0, 12), i=slice(0, 14))
    recording = base.expand_dims(time=np.arange(1203)).copy(
        data=da.broadcast_to(base.values, (1203, *base.shape)).rechunk(
            (300, *base.shape)
        )
    )
    result = register(recording, device="cpu", max_iterations=1)
    selected = result["images"].to_dataset()["registered"].isel(time=slice(1200, 1203))
    output = selected.data.map_blocks(
        np.copy, name="a" * prefix_length, dtype=selected.dtype
    )
    assert_allclose(
        output.compute(scheduler="synchronous"),
        np.broadcast_to(base.values, selected.shape),
        rtol=1e-6,
        atol=0,
    )
    assert len(rich_bars) == 1
    assert rich_bars[0].tasks[0].total == rich_bars[0].tasks[0].completed == 3


@pytest.mark.parametrize("progress", [True, False, None])
def test_lazy_progress_respects_modes_and_recomputation(
    recording_factory, rich_bars, progress
):
    """Explicit lazy modes survive repeated computation without retaining counts."""
    data, _, _ = recording_factory()
    result = register(
        data.chunk(time=2), max_iterations=1, device="cpu", progress=progress
    )
    for _ in range(2):
        result.compute(scheduler="synchronous")
    assert len(rich_bars) == (2 if progress is True else 0)
    assert all(bar.finished for bar in rich_bars)


def test_lazy_progress_closes_on_error(recording_factory, monkeypatch, rich_bars):
    """A failed lazy fit leaves an incomplete bar and propagates its error."""

    def fail(*args, **kwargs):
        raise RuntimeError("lazy optimizer failed")

    data, _, _ = recording_factory()
    result = register(data.chunk(time=2), device="cpu")
    monkeypatch.setattr(linear, "_optimize", fail)
    with pytest.raises(RuntimeError, match="lazy optimizer failed"):
        result.compute(scheduler="synchronous")
    assert len(rich_bars) == 1
    assert not rich_bars[0].finished
    assert not rich_bars[0].live.is_started


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_batch_progress_is_silent_and_preserves_results(
    recording_factory, device, batch_size, monkeypatch, capsys, rich_bars
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    data, _, _ = recording_factory()
    synchronizations = []
    original = torch.cuda.synchronize

    def synchronize():
        synchronizations.append(None)
        original()

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    options: RegistrationOptions = {
        "device": device,
        "batch_size": batch_size,
        "max_iterations": 3,
    }
    baseline = register(data, return_transforms=True, **options, progress=False)
    baseline_syncs = len(synchronizations)
    synchronizations.clear()
    updates = []
    reported = register(
        data,
        return_transforms=True,
        **options,
        progress=lambda completed, total: updates.append((completed, total)),
    )
    total = len(data)
    assert updates == [
        (min(first + batch_size, total), total) for first in range(0, total, batch_size)
    ]
    assert len(synchronizations) == baseline_syncs
    for field in [
        "images/registered",
        "images/valid",
        "forward/transform",
        "diagnostics",
    ]:
        xr.testing.assert_equal(reported[field], baseline[field])
    assert capsys.readouterr().out == ""
    assert not rich_bars


def test_progress_counts_reference_and_failed_frames(recording_factory, rich_bars):
    data, _, _ = recording_factory()
    data = data.copy(deep=True)
    data.values[1] = 0
    updates = []
    result = register(
        data,
        batch_size=1,
        max_iterations=3,
        progress=lambda completed, total: updates.append((completed, total)),
    )
    assert updates == [(1, 3), (2, 3), (3, 3)]
    assert result["diagnostics"].to_dataset().status[0] == "reference"
    assert result["diagnostics"].to_dataset().status[1] == "invalid_metric"
    assert result["diagnostics"].to_dataset().n_iterations[1] == 0
    register(data, batch_size=2, max_iterations=3)
    assert rich_bars[0].tasks[0].completed == len(data)


@pytest.mark.parametrize("progress", [0, 1, 42, "progress", object()])
def test_progress_must_be_supported_type(recording_factory, progress):
    data, _, _ = recording_factory()
    with pytest.raises(TypeError, match="progress must be True, False, None"):
        register(data, progress=progress)


def test_progress_callback_errors_propagate_without_mutating_input(recording_factory):
    data, _, _ = recording_factory()
    untouched = data.values.copy()

    def failed(completed, total):
        assert (completed, total) == (2, len(data))
        raise RuntimeError("progress display failed")

    with pytest.raises(RuntimeError, match="progress display failed"):
        register(data, batch_size=2, max_iterations=3, progress=failed)
    assert_array_equal(data.values, untouched)


@pytest.fixture
def rich_bars(monkeypatch):
    """Exercise real Rich rendering without a timer thread or terminal dependency."""
    bars = []
    original = vr_progress.Progress

    def create(*args, **kwargs):
        bar = original(
            *args,
            **kwargs,
            console=Console(file=StringIO(), force_terminal=True, width=120),
            auto_refresh=False,
        )
        bars.append(bar)
        return bar

    monkeypatch.setattr(vr_progress, "Progress", create)
    return bars


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_default_rich_progress_preserves_results_and_syncs(
    recording_factory, device, batch_size, monkeypatch, rich_bars
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    data, _, _ = recording_factory()
    options: RegistrationOptions = {
        "device": device,
        "batch_size": batch_size,
        "max_iterations": 3,
    }
    synchronizations = []
    original = torch.cuda.synchronize

    def synchronize():
        synchronizations.append(None)
        original()

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    quiet = register(data, return_transforms=True, **options, progress=False)
    baseline_syncs = len(synchronizations)
    synchronizations.clear()
    reported = register(data, return_transforms=True, **options)
    assert len(synchronizations) == baseline_syncs
    for field in [
        "images/registered",
        "images/valid",
        "forward/transform",
        "diagnostics",
    ]:
        xr.testing.assert_equal(reported[field], quiet[field])
    assert len(rich_bars) == 1
    bar = rich_bars[0]
    assert bar.tasks[0].total == bar.tasks[0].completed == len(data)
    assert bar.finished
    assert not bar.live.is_started
    output = bar.console.file.getvalue()
    assert "Registering frames" in output and "3/3" in output


@pytest.mark.parametrize("progress", [False, None])
def test_quiet_modes_create_no_rich_display(
    recording_factory, progress, rich_bars, capsys
):
    data, _, _ = recording_factory()
    register(data, max_iterations=3, progress=progress)
    assert not rich_bars
    assert capsys.readouterr().out == ""


def test_rich_display_is_closed_on_registration_error(
    recording_factory, monkeypatch, rich_bars
):
    data, _, _ = recording_factory()

    def fail(*args, **kwargs):
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(linear, "_optimize", fail)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        register(data)
    assert len(rich_bars) == 1
    assert not rich_bars[0].live.is_started
    assert not rich_bars[0].finished


@pytest.mark.parametrize("shape", [(1, 8, 9), (4, 5, 6)])
@pytest.mark.parametrize("mode", ["default", "auto", True, False, None])
def test_spatial_pair_progress_is_auto_quiet(shape, mode, rich_bars):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
    data = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    options: RegistrationOptions = {"max_iterations": 2}
    quiet = register(data, data, return_transforms=True, progress=False, **options)
    kwargs = {} if mode == "default" else {"progress": mode}
    result = register(data, data, return_transforms=True, **kwargs, **options)
    for name in [
        "images/registered",
        "images/valid",
        "forward/transform",
        "diagnostics",
    ]:
        xr.testing.assert_equal(result[name], quiet[name])
    assert len(rich_bars) == (1 if mode is True else 0)
    if mode is True:
        assert rich_bars[0].tasks[0].completed == 1


@pytest.mark.parametrize("mode", ["default", "auto"])
def test_single_frame_recording_is_auto_quiet(recording_factory, mode, rich_bars):
    data, _, _ = recording_factory()
    kwargs = {} if mode == "default" else {"progress": mode}
    register(data.isel(time=slice(0, 1)), max_iterations=2, **kwargs)
    assert not rich_bars


def test_spatial_pair_callback_is_honored(recording_factory, rich_bars):
    data, _, _ = recording_factory()
    fixed = data.isel(time=0, drop=True)
    updates = []
    register(
        fixed,
        fixed,
        max_iterations=2,
        progress=lambda done, total: updates.append((done, total)),
    )
    assert updates == [(1, 1)]
    assert not rich_bars
