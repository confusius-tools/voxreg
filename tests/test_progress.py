"""Batch progress is device-independent and leaves numerical execution unchanged."""

from io import StringIO

import confusius._utils.progress as cf_progress
import numpy as np
import pytest
import torch
import xarray as xr
from confusius import create_voxeldata
from numpy.testing import assert_array_equal
from rich.console import Console

from voxreg import register_rigid, rigid


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
    options = {"device": device, "batch_size": batch_size, "max_iterations": 3}
    baseline = register_rigid(data, **options, progress=False)
    baseline_syncs = len(synchronizations)
    synchronizations.clear()
    updates = []
    reported = register_rigid(
        data,
        **options,
        progress=lambda completed, total: updates.append((completed, total)),
    )
    total = len(data)
    assert updates == [
        (min(first + batch_size, total), total) for first in range(0, total, batch_size)
    ]
    assert len(synchronizations) == baseline_syncs
    for field in ["registered", "valid", "transforms", "diagnostics"]:
        xr.testing.assert_equal(getattr(reported, field), getattr(baseline, field))
    assert capsys.readouterr().out == ""
    assert not rich_bars


def test_progress_counts_reference_and_failed_frames(recording_factory, rich_bars):
    data, _, _ = recording_factory()
    data = data.copy(deep=True)
    data.values[1] = 0
    updates = []
    result = register_rigid(
        data,
        batch_size=1,
        max_iterations=3,
        progress=lambda completed, total: updates.append((completed, total)),
    )
    assert updates == [(1, 3), (2, 3), (3, 3)]
    assert result.diagnostics.status[0] == "reference"
    assert result.diagnostics.status[1] == "invalid_metric"
    assert result.diagnostics.n_iterations[1] == 0
    register_rigid(data, batch_size=2, max_iterations=3)
    assert rich_bars[0].tasks[0].completed == len(data)


@pytest.mark.parametrize("progress", [0, 1, 42, "progress", object()])
def test_progress_must_be_supported_type(recording_factory, progress):
    data, _, _ = recording_factory()
    with pytest.raises(TypeError, match="progress must be True, False, None"):
        register_rigid(data, progress=progress)


def test_progress_callback_errors_propagate_without_mutating_input(recording_factory):
    data, _, _ = recording_factory()
    untouched = data.values.copy()

    def failed(completed, total):
        assert (completed, total) == (2, len(data))
        raise RuntimeError("progress display failed")

    with pytest.raises(RuntimeError, match="progress display failed"):
        register_rigid(data, batch_size=2, max_iterations=3, progress=failed)
    assert_array_equal(data.values, untouched)


@pytest.fixture
def rich_bars(monkeypatch):
    """Exercise real Rich rendering without a timer thread or terminal dependency."""
    bars = []
    original = cf_progress.Progress

    def create(*args, **kwargs):
        bar = original(
            *args,
            **kwargs,
            console=Console(file=StringIO(), force_terminal=True, width=120),
            auto_refresh=False,
        )
        bars.append(bar)
        return bar

    monkeypatch.setattr(cf_progress, "Progress", create)
    return bars


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_default_rich_progress_preserves_results_and_syncs(
    recording_factory, device, batch_size, monkeypatch, rich_bars
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    data, _, _ = recording_factory()
    options = {"device": device, "batch_size": batch_size, "max_iterations": 3}
    synchronizations = []
    original = torch.cuda.synchronize

    def synchronize():
        synchronizations.append(None)
        original()

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    quiet = register_rigid(data, **options, progress=False)
    baseline_syncs = len(synchronizations)
    synchronizations.clear()
    reported = register_rigid(data, **options)
    assert len(synchronizations) == baseline_syncs
    for field in ["registered", "valid", "transforms", "diagnostics"]:
        xr.testing.assert_equal(getattr(reported, field), getattr(quiet, field))
    assert len(rich_bars) == 1
    bar = rich_bars[0]
    assert bar.tasks[0].total == bar.tasks[0].completed == len(data)
    assert bar.finished
    assert not bar.live.is_started
    output = bar.console.file.getvalue()
    assert "Registering frames" in output and "100%" in output


@pytest.mark.parametrize("progress", [False, None])
def test_quiet_modes_create_no_rich_display(
    recording_factory, progress, rich_bars, capsys
):
    data, _, _ = recording_factory()
    register_rigid(data, max_iterations=3, progress=progress)
    assert not rich_bars
    assert capsys.readouterr().out == ""


def test_rich_display_is_closed_on_registration_error(
    recording_factory, monkeypatch, rich_bars
):
    data, _, _ = recording_factory()

    def fail(*args, **kwargs):
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(rigid, "_optimize", fail)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        register_rigid(data)
    assert len(rich_bars) == 1
    assert not rich_bars[0].live.is_started
    assert not rich_bars[0].finished


@pytest.mark.parametrize("shape", [(1, 8, 9), (4, 5, 6)])
@pytest.mark.parametrize("mode", ["default", "auto", True, False, None])
def test_spatial_pair_progress_is_auto_quiet(shape, mode, rich_bars):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
    data = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    options = {"intensity_scaling": "none", "max_iterations": 2}
    quiet = register_rigid(data, data, progress=False, **options)
    kwargs = {} if mode == "default" else {"progress": mode}
    result = register_rigid(data, data, **kwargs, **options)
    for name in ["registered", "valid", "transforms", "diagnostics"]:
        xr.testing.assert_equal(getattr(result, name), getattr(quiet, name))
    assert len(rich_bars) == (1 if mode is True else 0)
    if mode is True:
        assert rich_bars[0].tasks[0].completed == 1


@pytest.mark.parametrize("mode", ["default", "auto"])
def test_single_frame_recording_is_auto_quiet(recording_factory, mode, rich_bars):
    data, _, _ = recording_factory()
    kwargs = {} if mode == "default" else {"progress": mode}
    register_rigid(data.isel(time=slice(0, 1)), max_iterations=2, **kwargs)
    assert not rich_bars


def test_spatial_pair_callback_is_honored(recording_factory, rich_bars):
    data, _, _ = recording_factory()
    fixed = data.isel(time=0, drop=True)
    updates = []
    register_rigid(
        fixed,
        fixed,
        max_iterations=2,
        progress=lambda done, total: updates.append((done, total)),
    )
    assert updates == [(1, 1)]
    assert not rich_bars
