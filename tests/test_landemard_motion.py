"""Whole-recording QC retains one reference and exports native corrected data."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import confusius as cf
import numpy as np
import pytest
from PIL import Image

from voxreg import position_to_world


@pytest.mark.parametrize("batch_size", [2, 128])
def test_small_staged_motion_qc(recording_factory, monkeypatch, tmp_path, batch_size):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "research/confusius-examples")
    )
    script = importlib.import_module("compare_landemard_motion")
    data, _, _ = recording_factory(oblique=True)
    original = data.values.copy()
    calls = []
    expected = {model: [] for model in ("rigid", "affine", "syn")}
    register = script.register

    def observe(moving, **kwargs):
        assert moving is data  # Exactly one whole-recording call per method.
        assert kwargs.get("fixed") is None
        assert kwargs["batch_size"] == batch_size
        assert kwargs["return_transforms"] == (kwargs["transform"] != "syn")
        np.testing.assert_array_equal(moving.values, original)
        calls.append((kwargs["transform"], kwargs["initial_transform"]))
        result = register(moving, **kwargs)
        expected[kwargs["transform"]].append(result.registered.values.copy())
        return result

    monkeypatch.setattr(script, "register", observe)
    output = tmp_path / "qc"
    scores = script.compare(data, output, iterations=1, batch_size=batch_size)
    assert [name for name, _ in calls] == ["rigid", "affine", "syn"]
    assert calls[0][1] is None
    assert calls[1][1].dims == calls[2][1].dims == ("time", "world_out", "world_in")
    np.testing.assert_array_equal(data.values, original)
    assert scores.shape == (3, 4)
    np.testing.assert_allclose(scores[0], 1, atol=1e-7)
    csv = np.loadtxt(output / "qc.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(csv[:, 0], data.time.values)
    np.testing.assert_allclose(csv[:, 1:], scores)
    assert json.loads((output / "settings.json").read_text())["middle_i"] == 40
    with Image.open(output / "qc.png") as image:
        assert image.width > image.height
    for model, frames in expected.items():
        saved = cf.load(output / f"{model}.nii").compute()
        assert saved.dtype == data.dtype
        np.testing.assert_array_equal(saved.values, np.concatenate(frames))
        np.testing.assert_allclose(saved.time, data.time)
        np.testing.assert_allclose(
            position_to_world(saved), position_to_world(data), atol=1e-5
        )
        np.testing.assert_array_equal(saved.values[0], original[0])
    assert not list(output.glob("*.npy"))
    assert not list(output.glob("*.gif"))
    with pytest.raises(FileExistsError):
        script.compare(data, output, iterations=1)


def test_four_recordings_share_confusius_napari_viewer(
    recording_factory, monkeypatch, tmp_path
):
    import napari

    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "research/confusius-examples")
    )
    script = importlib.import_module("compare_landemard_motion")
    data, _, _ = recording_factory(oblique=True)
    for model in ("rigid", "affine", "syn"):
        cf.io.save_nifti(data, tmp_path / f"{model}.nii")
    viewer = SimpleNamespace(grid=SimpleNamespace())
    calls = []
    loops = []

    def plot(recording, **options):
        assert options["viewer"] is (None if not calls else viewer)
        np.testing.assert_allclose(recording.time, data.time)
        np.testing.assert_array_equal(recording.values, data.values)
        calls.append(options)
        return viewer, object()

    def run():
        loops.append(True)
        assert len(calls) == 4
        assert viewer.grid.enabled and viewer.grid.shape == (2, 2)
        assert viewer.grid.stride == 1

    monkeypatch.setattr(cf.plotting, "plot_napari", plot)
    monkeypatch.setattr(napari, "run", run)
    assert script.show_recordings(data, tmp_path) is viewer
    assert loops == [True]
    assert [call["name"] for call in calls] == ["uncorrected", "rigid", "affine", "syn"]
    assert all(
        call["contrast_limits"] == (0, float(data.isel(time=0).max())) for call in calls
    )
    assert all(call["colormap"] == "gray" for call in calls)


def test_time_summary_preserves_dips_last_frame_and_missing_bins(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "research/confusius-examples")
    )
    script = importlib.import_module("compare_landemard_motion")
    values = np.ones((2401, 2))
    values[1100, 0] = -0.5
    values[-1, 0] = 0.25
    values[:3, 1] = np.nan
    edges = np.linspace(0, len(values), 1201, dtype=int)
    mean, low, high = script.time_summary(values, edges)
    assert mean.shape == (1200, 2)
    assert np.nanmin(low[:, 0]) == -0.5
    assert low[-1, 0] == 0.25
    assert high[-1, 0] == 1
    assert np.isnan(mean[0, 1]) and np.isnan(low[0, 1]) and np.isnan(high[0, 1])
