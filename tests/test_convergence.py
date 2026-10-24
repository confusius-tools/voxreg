"""ITK oracle compatibility and independent device-resident stopping state."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from voxreg import register_rigid
from voxreg.convergence import WindowConvergence

REFERENCE = json.loads(
    (Path(__file__).parent / "data/convergence_itk.json").read_text()
)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("window", [2, 10, 20, 50])
def test_matches_itk_monitor(device, window):
    losses = torch.tensor(
        list(REFERENCE["losses"].values()), dtype=torch.float64, device=device
    ).T
    expected = torch.tensor(
        list(REFERENCE["expected"][str(window)].values()),
        dtype=torch.float64,
        device=device,
    ).T
    monitor = WindowConvergence(losses.shape[1], window, torch.device(device))
    active = torch.ones(losses.shape[1], dtype=torch.bool, device=device)
    actual = torch.stack([monitor.update(loss, active) for loss in losses])
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=1e-12)
    torch.testing.assert_close(
        actual[: window - 1], expected[: window - 1], rtol=0, atol=0
    )


@pytest.mark.parametrize("device", DEVICES)
def test_inactive_histories_are_frozen_and_frames_are_independent(device):
    monitor = WindowConvergence(3, 10, torch.device(device))
    serial = [WindowConvergence(1, 10, torch.device(device)) for _ in range(3)]
    for step in range(60):
        active = torch.tensor([step < 20, step < 40, True], device=device)
        loss = torch.tensor(
            [-0.5 - step / 1000, -0.7 - step / 2000, -0.8],
            dtype=torch.float64,
            device=device,
        )
        before = [
            state.clone() for state in (monitor.history, monitor.total, monitor.count)
        ]
        # Invalid inactive losses must not contaminate either history or normalization.
        loss = torch.where(active, loss, torch.nan)
        value = monitor.update(loss, active)
        for frame in range(3):
            single = serial[frame].update(
                loss[frame : frame + 1], active[frame : frame + 1]
            )
            torch.testing.assert_close(value[frame], single[0], atol=2e-15, rtol=1e-12)
        for old, new in zip(
            before, (monitor.history, monitor.total, monitor.count), strict=True
        ):
            torch.testing.assert_close(new[~active], old[~active], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_stops_before_next_update_and_disabled_window_reaches_limit(
    recording_factory, monkeypatch, device
):
    from voxreg import rigid

    def flat_metric(moved, fixed, valid):
        return moved.to(torch.float64).sum(dim=(1, 2)) * 0 - 0.9, torch.ones(
            len(moved), dtype=torch.bool, device=moved.device
        )

    monkeypatch.setattr(rigid, "global_ncc", flat_metric)
    data, _, _ = recording_factory()
    options = {"device": device, "max_iterations": 8, "progress": False}
    stopped = register_rigid(
        data, **options, convergence_window_size=4, convergence_minimum_value=1
    )
    limited = register_rigid(data, **options, convergence_window_size=9)
    np.testing.assert_array_equal(stopped.diagnostics.n_iterations[1:], 3)
    np.testing.assert_array_equal(stopped.diagnostics.status[1:], "converged")
    np.testing.assert_array_equal(limited.diagnostics.n_iterations[1:], 8)
    np.testing.assert_array_equal(limited.diagnostics.status[1:], "max_iterations")
    np.testing.assert_array_equal(stopped.transforms, limited.transforms)


@pytest.mark.parametrize(
    "options",
    [
        {"convergence_window_size": 1},
        {"convergence_window_size": 2.5},
        {"convergence_window_size": True},
        {"convergence_minimum_value": -1},
        {"convergence_minimum_value": np.nan},
        {"convergence_minimum_value": np.inf},
        {"convergence_minimum_value": True},
        {"convergence_minimum_value": "1e-8"},
        {"convergence_minimum_value": 1j},
    ],
)
def test_rejects_invalid_convergence_options(recording_factory, options):
    data, _, _ = recording_factory()
    with pytest.raises(ValueError, match="convergence_"):
        register_rigid(data, **options)
