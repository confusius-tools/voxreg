"""Research-only live plotting must preserve fitting and restore scoped observers."""

import importlib
from pathlib import Path

import confusius as cf
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal

from voxreg import register, rigid


@pytest.fixture
def plotting(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "research/confusius-examples")
    )
    module = importlib.import_module("plot_registration")
    yield module
    plt.close("all")


@pytest.fixture
def pair():
    k, j, i = np.indices((5, 7, 9))
    values = -15 + 5 * (np.sin(i * 0.6) + np.cos(j * 0.4) + 0.1 * k * k)
    affine = np.diag([0.2, 0.3, 0.4, 1.0])
    affine[:3, 3] = [-0.4, -0.9, -1.6]
    volume = cf.create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=affine)
    return volume, volume.isel(k=slice(2, 3))


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
@pytest.mark.parametrize("model", ["translation", "rigid", "affine"])
@pytest.mark.parametrize("moving_slice", [False, True])
def test_plot_preserves_outputs_and_displays_returned_pose(
    plotting, pair, model, device, moving_slice
):
    volume, plane = pair
    moving, fixed = (plane, volume) if moving_slice else (volume, plane)
    seed = np.eye(4)
    seed[:3, 3] = [0, 0.02, -0.01]
    options = {
        "transform": model,
        "initialization": seed,
        "intensity_scaling": "none",
        "max_iterations": 3,
        "device": device,
        "progress": False,
    }
    expected = register(moving, fixed, **options)
    seen = []
    result, fig = plotting.register_with_plot(
        moving,
        fixed,
        show=False,
        metric_interval=1,
        metric_callback=lambda *values: seen.append(tuple(v.copy() for v in values)),
        **options,
    )
    for name in ["registered", "transforms", "valid"]:
        assert_array_equal(getattr(result, name), getattr(expected, name))
    for name in result.diagnostics:
        assert_array_equal(result.diagnostics[name], expected.diagnostics[name])
    assert [int(x[1][0]) for x in seen] == [0, 1, 2, 3]
    assert len(fig.axes) == 2 and "support" not in fig.axes[0].get_title().lower()
    assert fig.axes[0].get_aspect() == pytest.approx(0.3 / 0.4)
    public = np.asarray(result.transforms)
    matrix = np.linalg.inv(public) if moving_slice else public
    values, _, _ = plotting.display_sample(matrix, plane, volume)
    assert_allclose(
        fig.axes[0].images[0].get_array(),
        plotting.composite_rgb(plane.values[0], values[0]),
        rtol=0,
        atol=0,
    )
    assert_array_equal(fig.axes[1].lines[0].get_xdata(), [0, 1, 2, 3])
    assert_allclose(
        fig.axes[1].lines[1].get_ydata(),
        result.diagnostics.objective.item(),
        rtol=0,
        atol=0,
    )


def test_live_preview_uses_matching_initial_pose_and_throttles(
    plotting, pair, monkeypatch
):
    volume, plane = pair
    seed = np.eye(4)
    seed[1, 3] = 0.08
    expected, _, _ = plotting.display_sample(seed, plane, volume)
    calls = []
    sampler = plotting.display_sample

    def observe(*args):
        calls.append(np.array(args[0]))
        return sampler(*args)

    monkeypatch.setattr(plotting, "display_sample", observe)
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)

    def check(indices, steps, metric):
        if steps[0] == 0:
            image = np.asarray(plt.gcf().axes[0].images[0].get_array())
            assert_allclose(
                image,
                plotting.composite_rgb(plane.values[0], expected[0]),
                rtol=0,
                atol=0,
            )
        indices[:] = -1
        steps[:] = -1
        metric[:] = 123

    result, fig = plotting.register_with_plot(
        volume,
        plane,
        initialization=seed,
        intensity_scaling="none",
        max_iterations=3,
        progress=False,
        metric_interval=1,
        metric_callback=check,
        redraw_interval=1e6,
    )
    assert len(calls) == 2
    assert_allclose(calls[0], seed, atol=1e-14, rtol=0)
    assert_allclose(calls[-1], result.transforms, atol=1e-14, rtol=0)
    assert_array_equal(fig.axes[1].lines[0].get_xdata(), [0, 1, 2, 3])
    assert max(fig.axes[1].lines[0].get_ydata()) <= 0


def test_callback_failure_restores_engine_and_closes_figure(plotting, pair):
    volume, plane = pair
    optimizer, sampler = rigid._optimize, rigid.sample_rigid
    figures = plt.get_fignums()

    def fail(*args):
        raise RuntimeError("viewer callback failed")

    with pytest.raises(RuntimeError, match="viewer callback failed"):
        plotting.register_with_plot(
            volume,
            plane,
            show=False,
            metric_callback=fail,
            intensity_scaling="none",
            progress=False,
        )
    assert rigid._optimize is optimizer and rigid.sample_rigid is sampler
    assert plt.get_fignums() == figures


def test_pose_observer_failure_restores_engine(plotting, pair):
    volume, plane = pair
    optimizer, sampler = rigid._optimize, rigid.sample_rigid

    def fail(*args):
        raise RuntimeError("observer failed")

    with (
        pytest.raises(RuntimeError, match="observer failed"),
        plotting.observe_optimizer(fail),
    ):
        register(volume, plane, intensity_scaling="none", progress=False)
    assert rigid._optimize is optimizer and rigid.sample_rigid is sampler


def test_volumes_show_central_fixed_slice_and_export(plotting, pair, tmp_path):
    volume, _ = pair
    _, fig = plotting.register_with_plot(
        volume,
        volume,
        show=False,
        max_iterations=1,
        intensity_scaling="none",
        progress=False,
    )
    assert "central k slice" in fig._suptitle.get_text()
    fig.savefig(tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 1000


def test_singleton_time_axis_and_unusable_metric(plotting, pair):
    volume, plane = pair
    moving = volume.copy(data=np.full(volume.shape, -20.0)).expand_dims(time=[7])
    result, fig = plotting.register_with_plot(
        moving, plane, show=False, intensity_scaling="none", progress=False
    )
    assert result.diagnostics.status.item() == "invalid_metric"
    assert "unusable metric" in fig.axes[0].get_title()
    assert len(fig.axes[1].lines) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"redraw_interval": -1},
        {"redraw_interval": np.nan},
        {"display_range": (1, 1)},
        {"display_range": (0, np.inf)},
        {"metric_callback": 1},
    ],
)
def test_plot_options_fail_without_leaking_figures(plotting, pair, kwargs):
    volume, plane = pair
    figures = plt.get_fignums()
    with pytest.raises((ValueError, TypeError)):
        plotting.register_with_plot(volume, plane, show=False, **kwargs)
    assert plt.get_fignums() == figures


def test_movie_requires_explicit_frame_selection(plotting, pair):
    volume, plane = pair
    movie = volume.expand_dims(time=[1, 2])
    with pytest.raises(ValueError, match="select a moving/fixed frame"):
        plotting.register_with_plot(movie, plane, show=False)


def test_atlas_cli_retains_unusable_result_and_refuses_overwrite(
    plotting, pair, monkeypatch, tmp_path
):
    import json

    import atlas_inputs
    import explore_atlas

    volume, plane = pair
    constant = volume.copy(data=np.full(volume.shape, -20.0))
    monkeypatch.setattr(
        atlas_inputs, "inputs", lambda: (constant, plane, np.eye(4), [], [])
    )
    output = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        [
            "explore_atlas",
            "--headless",
            "--moving",
            "template",
            "--max-iterations",
            "1",
            "--output",
            str(output),
        ],
    )
    explore_atlas.main()
    record = json.loads((output / "results.json").read_text())
    assert record["returned_objective"] is None
    assert (output / "registration.png").is_file() and (output / "arrays.npz").is_file()
    original = (output / "results.json").read_bytes()
    with pytest.raises(FileExistsError):
        explore_atlas.main()
    assert (output / "results.json").read_bytes() == original


def test_shared_helpers_do_not_override_backend(plotting):
    import matplotlib

    before = matplotlib.get_backend()
    importlib.reload(plotting)
    assert matplotlib.get_backend() == before
    assert_array_equal(
        plotting.composite_rgb(np.array([[-30.0]]), np.array([[np.nan]])),
        [[[0.25, 0.25, 0.25]]],
    )
