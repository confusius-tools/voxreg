"""Staged original-image fitting, option inheritance and frame isolation."""

from dataclasses import FrozenInstanceError
from itertools import pairwise

import numpy as np
import pytest
import torch
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from oracles import max_pull_difference
from registration_options import RegistrationOptions

from voxreg import Stage, position_to_world, register, resample


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("model", ["translation", "rigid", "affine", "syn"])
def test_single_stage_preserves_single_model(
    linear_pair_factory, device, planar, model
):
    fixed, moving, _ = linear_pair_factory(planar, "rigid")
    options: RegistrationOptions = {
        "max_iterations": 3,
        "intensity_scaling": "none",
        "device": device,
        "return_transforms": True,
        "progress": False,
    }
    baseline = register(moving, fixed, transform=model, **options)
    staged = register(moving, fixed, transform=(model,), **options)
    assert baseline.stage_diagnostics is None
    assert staged.stage_diagnostics is not None and len(staged.stage_diagnostics) == 1
    for name in ("registered", "valid", "forward_transforms", "inverse_transforms"):
        xr.testing.assert_identical(getattr(staged, name), getattr(baseline, name))
    xr.testing.assert_equal(staged.diagnostics, baseline.diagnostics)


@pytest.mark.parametrize("planar", [True, False])
def test_stage_overrides_and_manual_handoff(linear_pair_factory, device, planar):
    fixed, moving, _ = linear_pair_factory(planar, "rigid")
    stages = (
        Stage("rigid", max_iterations=2),
        Stage(
            "affine",
            shrink_factors=(2, 1),
            smoothing_sigmas=(0.18, 0),
            max_iterations=(2, 3),
        ),
        Stage("syn", max_iterations=4, syn_update_sigma=0.8),
    )
    options: RegistrationOptions = {
        "intensity_scaling": "none",
        "device": device,
        "return_transforms": True,
        "progress": False,
        "max_iterations": 7,
    }
    actual = register(moving, fixed, transform=stages, **options)
    seed = None
    for stage in stages:
        manual = register(
            moving,
            fixed,
            transform=stage.transform,
            initial_transform=seed,
            **{**options, **stage._overrides()},
        )
        seed = manual.forward_transforms
    assert (
        actual.forward_transforms is not None and manual.forward_transforms is not None
    )
    assert (
        actual.inverse_transforms is not None and manual.inverse_transforms is not None
    )
    assert_allclose(
        actual.forward_transforms, manual.forward_transforms, rtol=0, atol=2e-12
    )
    assert_allclose(
        actual.inverse_transforms, manual.inverse_transforms, rtol=0, atol=2e-12
    )
    assert_allclose(actual.registered, manual.registered, rtol=0, atol=2e-12)
    assert actual.stage_diagnostics is not None
    assert [d.n_iterations.item() for d in actual.stage_diagnostics] == [2, 5, 4]
    assert actual.diagnostics.n_iterations == 11
    assert actual.diagnostics.attrs["stages"] == ["rigid", "affine", "syn"]
    assert [d.attrs["stage"] for d in actual.stage_diagnostics] == [0, 1, 2]


@pytest.mark.parametrize("terminal", ["affine", "syn"])
def test_pipeline_samples_original_only_once(
    linear_pair_factory, monkeypatch, terminal
):
    from voxreg import rigid, syn

    fixed, moving, _ = linear_pair_factory(True, "rigid")
    moving = moving.copy(data=moving.values.astype(np.float64))
    source = moving.copy(deep=True)
    samples = []
    original_sample = rigid.sample_rigid

    def observe(images, *args, **kwargs):
        if images.dtype == torch.float64:
            samples.append(images.detach().cpu().numpy().copy())
        return original_sample(images, *args, **kwargs)

    monkeypatch.setattr(rigid, "sample_rigid", observe)
    monkeypatch.setattr(syn, "sample_rigid", observe)
    models = (
        ("translation", "rigid", "affine")
        if terminal == "affine"
        else ("rigid", "affine", "syn")
    )
    result = register(
        moving,
        fixed,
        transform=models,
        max_iterations=2,
        return_transforms=True,
        progress=False,
    )
    assert len(samples) == 1
    assert_array_equal(samples[0], moving.values)
    assert result.forward_transforms is not None
    output, valid = resample(moving, fixed, result.forward_transforms)
    assert_allclose(result.registered, output, rtol=0, atol=2e-6)
    assert_array_equal(result.valid, valid)
    xr.testing.assert_identical(moving, source)


@pytest.mark.parametrize("terminal", ["affine", "syn"])
def test_motion_stages_isolate_failures_and_preserve_reference(
    linear_pair_factory, terminal
):
    fixed, _, _ = linear_pair_factory(True, "rigid")
    moving = fixed.copy(data=np.roll(fixed.values, 1, axis=-1))
    recording = xr.concat(
        [fixed, moving, fixed.copy(data=np.ones_like(fixed))],
        dim=xr.IndexVariable("time", [1, 3, 5]),
    )
    recording = recording.assign_coords(acquisition=("time", [10, 20, 30]))
    results = []
    for batch_size in (1, 3):
        events = []
        result = register(
            recording,
            transform=("rigid", terminal),
            max_iterations=3,
            shrink_factors=(2, 1),
            smoothing_sigmas=(0.18, 0),
            intensity_scaling="none",
            return_transforms=True,
            batch_size=batch_size,
            progress=lambda done, total, events=events: events.append((done, total)),
        )
        assert events[-1] == (6, 6)
        assert all(a[0] < b[0] for a, b in pairwise(events))
        assert_array_equal(result.registered[0], recording[0])
        assert_array_equal(
            result.diagnostics.status, ["reference", "max_iterations", "invalid_metric"]
        )
        assert_array_equal(result.diagnostics.n_iterations, [0, 12, 0])
        assert result.stage_diagnostics is not None
        assert_array_equal(
            result.stage_diagnostics[1].status,
            ["reference", "max_iterations", "skipped"],
        )
        assert_array_equal(
            result.stage_diagnostics[1].level_status[2], ["skipped", "skipped"]
        )
        assert_array_equal(result.registered.acquisition, recording.acquisition)
        results.append(result)
    first, second = results
    assert (
        first.forward_transforms is not None and second.forward_transforms is not None
    )
    assert_array_equal(
        first.forward_transforms.isel(time=0),
        second.forward_transforms.isel(time=0),
    )
    # Float32 fitting can select different reduction kernels for different batches.
    assert_allclose(
        first.forward_transforms.isel(time=2),
        second.forward_transforms.isel(time=2),
        rtol=0,
        atol=1e-12,
    )
    assert (
        max_pull_difference(first.forward_transforms, second.forward_transforms, fixed)
        < 0.001
    )
    assert_allclose(results[0].registered, results[1].registered, rtol=0, atol=2e-6)


@pytest.mark.parametrize(
    "specification",
    [
        (),
        ("syn", "affine"),
        ("rigid", "rigid"),
        ("affine", "rigid"),
        ("unknown",),
        ("rigid", 42),
        (Stage("affine", max_iterations=0),),
        ("rigid", Stage("syn", smoothing_sigmas=(np.nan,))),
        (Stage("rigid", syn_cc_radius=3),),
    ],
)
def test_invalid_pipeline_rejected_before_fitting(
    linear_pair_factory, monkeypatch, specification
):
    from voxreg import rigid

    fixed, moving, _ = linear_pair_factory(True, "rigid")

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid stage configuration reached the optimizer")

    monkeypatch.setattr(rigid, "_optimize", unexpected)
    with pytest.raises(ValueError):
        register(moving, fixed, transform=specification, progress=False)


def test_stage_owns_sequences_and_is_frozen():
    factors, iterations = [2, 1], [2, 3]
    stage = Stage("rigid", shrink_factors=factors, max_iterations=iterations)
    factors[0], iterations[0] = 8, 10
    assert stage.shrink_factors == (2, 1)
    assert stage.max_iterations == (2, 3)
    with pytest.raises(FrozenInstanceError):
        stage.learning_rate = 1  # ty: ignore[invalid-assignment] -- exercise frozen-instance rejection


def test_shared_syn_settings_only_apply_to_syn(linear_pair_factory):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    result = register(
        moving,
        fixed,
        transform=("rigid", Stage("syn", syn_total_sigma=0)),
        max_iterations=2,
        syn_cc_radius=3,
        syn_update_sigma=0.8,
        return_transforms=False,
        progress=False,
    )
    assert result.stage_diagnostics is not None
    assert result.stage_diagnostics[-1].attrs["cc_radius"] == 3
    assert result.stage_diagnostics[-1].attrs["update_sigma_voxels"] == 0.8
    assert result.stage_diagnostics[-1].attrs["total_sigma_voxels"] == 0
    assert result.forward_transforms is None and result.inverse_transforms is None


@pytest.mark.parametrize("planar", [True, False])
def test_failed_affine_retains_full_seed_in_terminal_syn(linear_pair_factory, planar):
    fixed, moving, seed = linear_pair_factory(planar, "affine")
    moving = moving.copy(data=np.full(moving.shape, 500, dtype=np.uint16))
    result = register(
        moving,
        fixed,
        transform=("affine", "syn"),
        initial_transform=seed,
        max_iterations=2,
        return_transforms=True,
        progress=False,
    )
    assert result.stage_diagnostics is not None
    assert result.stage_diagnostics[0].status == "invalid_metric"
    assert result.stage_diagnostics[1].status == "skipped"
    assert result.diagnostics.status == "invalid_metric"
    assert result.diagnostics.n_iterations == 0
    assert result.registered.dtype == np.uint16
    points = position_to_world(fixed) @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones(fixed.size))
    )
    expected = ((seed @ points) - points)[:3].T.reshape(*fixed.shape, 3)
    assert result.forward_transforms is not None
    assert_allclose(result.forward_transforms, expected, rtol=0, atol=2e-12)
    sampled, valid = resample(moving, fixed, result.forward_transforms)
    assert_array_equal(result.registered, sampled)
    assert_array_equal(result.valid, valid)


@pytest.mark.parametrize("terminal", ["affine", "syn"])
def test_moments_pivot_is_initialized_once(linear_pair_factory, terminal):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    fixed = fixed.copy(data=np.rint(fixed.values * 1000).astype(np.uint16))
    moving = moving.copy(data=np.full(moving.shape, 500, dtype=np.uint16))
    result = register(
        moving,
        fixed,
        transform=("rigid", terminal),
        initialization="center_moments",
        max_iterations=2,
        return_transforms=True,
        progress=False,
    )
    assert result.stage_diagnostics is not None
    first = result.stage_diagnostics[0]
    assert first.attrs["initialization"] == "center_moments"
    assert "initialization" not in result.stage_diagnostics[1].attrs
    if terminal == "affine":
        assert_array_equal(
            first.attrs["rotation_center_zyx_mm"],
            result.stage_diagnostics[1].attrs["center_zyx_mm"],
        )
    # The first failure is retained; later stages do not attempt a fresh initialization.
    assert result.diagnostics.status == "invalid_metric"
    assert result.stage_diagnostics[1].status == "skipped"


@pytest.mark.parametrize("planar", [True, False])
def test_rigid_affine_syn_recovers_physical_truth(linear_pair_factory, device, planar):
    fixed, moving, truth = linear_pair_factory(planar, "rigid")
    result = register(
        moving,
        fixed,
        transform=("rigid", "affine", "syn"),
        max_iterations=100,
        convergence_window_size=1000,
        intensity_scaling="none",
        device=device,
        return_transforms=True,
        progress=False,
    )
    assert result.forward_transforms is not None
    points = position_to_world(fixed) @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones(fixed.size))
    )
    expected = ((truth @ points) - points)[:3].T.reshape(*fixed.shape, 3)
    spacing = np.linalg.norm(
        position_to_world(fixed)[:3, :3][:, np.array(fixed.shape) > 1], axis=0
    ).min()
    residual = result.forward_transforms.values - expected
    assert np.sqrt(np.mean(np.sum(residual**2, axis=-1))) / spacing < 0.30
    assert result.diagnostics.min_jacobian > 0
