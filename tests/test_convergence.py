"""ITK oracle compatibility and independent device-resident stopping state."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from registration_options import RegistrationOptions
import torch
import xarray as xr

from voxreg import register
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
    options: RegistrationOptions = {
        "device": device,
        "max_iterations": 8,
        "progress": False,
    }
    stopped = register(
        data,
        return_transforms=True,
        **options,
        convergence_window_size=4,
        convergence_minimum_value=1,
    )
    limited = register(
        data, return_transforms=True, **options, convergence_window_size=9
    )
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
        register(data, **options)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("failed_frame", [False, True])
def test_terminal_check_preserves_frozen_full_budget_results(
    recording_factory, monkeypatch, device, failed_frame
):
    from voxreg import rigid

    data, _, _ = recording_factory(oblique=True, strided=True)
    if failed_frame:
        data.values[1] = 0
    sample = rigid.sample_rigid
    calls = []

    def counted_sample(*args, **kwargs):
        calls.append(None)
        return sample(*args, **kwargs)

    monkeypatch.setattr(rigid, "sample_rigid", counted_sample)
    options: RegistrationOptions = {
        "device": device,
        "max_iterations": 400,
        "progress": False,
    }
    stopped = register(data, return_transforms=True, **options)
    stopped_calls = len(calls)
    calls.clear()
    monkeypatch.setattr(rigid, "_all_frames_stopped", lambda status: False)
    full_budget = register(data, return_transforms=True, **options)
    assert stopped_calls < len(calls)
    for field in ["registered", "valid", "transforms", "diagnostics"]:
        xr.testing.assert_equal(getattr(stopped, field), getattr(full_budget, field))


@pytest.mark.parametrize("device", DEVICES)
def test_polling_is_batch_wide_bounded_and_absent_in_fixed_budget_controls(
    recording_factory, monkeypatch, device
):
    from voxreg import rigid

    evaluations, polls = [], []

    def metric(moved, fixed, valid):
        evaluations.append(None)
        return moved.double().sum(dim=(1, 2)) * 0 - 0.9, torch.ones(
            len(moved), dtype=torch.bool, device=moved.device
        )

    def not_finished(status):
        assert status.shape == (3,)
        polls.append(len(evaluations))
        return False

    monkeypatch.setattr(rigid, "global_ncc", metric)
    monkeypatch.setattr(rigid, "_all_frames_stopped", not_finished)
    data, _, _ = recording_factory()
    options: RegistrationOptions = {
        "device": device,
        "max_iterations": 40,
        "progress": False,
    }
    register(data, **options, convergence_window_size=4, convergence_minimum_value=1)
    assert polls == list(range(4, 40, 10 if device == "cuda" else 1))
    polls.clear()
    register(data, **options, convergence_window_size=41)
    assert polls == []


def test_calibration_report_uses_identical_controls_and_common_support(
    recording_factory, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "calibrate", Path(__file__).parents[1] / "research/convergence/calibrate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "POLICIES",
        {
            "fixed100": (3, 4, 1e-6),
            "fixed400": (3, 4, 1e-6),
        },
    )
    data, _, _ = recording_factory(oblique=True, strided=True)
    report = module.compare(data, device="cpu", batch_size=3)
    assert 0.9 < report["common_fraction_median"] <= 1
    for record in report["policies"].values():
        assert len(record["ncc2"]) == 2
        assert np.isfinite(record["ncc2"]).all()
        assert record["vs_control"]["fixed100"]["max_rms_pixels"] == 0
        assert record["vs_control"]["fixed100"]["worst_ncc2_drop"] == 0
    with monkeypatch.context() as patches:
        patches.setattr(module.np, "corrcoef", lambda *args: np.full((2, 2), np.nan))
        with pytest.raises(ValueError, match="Nonfinite.*scores"):
            module.compare(data, device="cpu", batch_size=3)
