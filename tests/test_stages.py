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
        "device": device,
        "return_transforms": True,
        "progress": False,
    }
    baseline = register(moving, fixed, transform=model, **options)
    staged = register(moving, fixed, transform=(model,), **options)
    assert "stages" not in baseline
    assert (
        "stages" in staged
        and len(tuple(node.to_dataset() for node in staged["stages"].children.values()))
        == 1
    )
    for name in (
        "images/registered",
        "images/valid",
        "forward/transform",
        "inverse/transform",
    ):
        xr.testing.assert_identical(staged[name], baseline[name])
    xr.testing.assert_equal(
        staged["diagnostics"].to_dataset(), baseline["diagnostics"].to_dataset()
    )


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
        seed = manual["forward"].to_dataset()["transform"]
    assert "forward" in actual and "forward" in manual
    assert "inverse" in actual and "inverse" in manual
    assert_allclose(
        actual["forward"].to_dataset()["transform"],
        manual["forward"].to_dataset()["transform"],
        rtol=0,
        atol=2e-12,
    )
    assert_allclose(
        actual["inverse"].to_dataset()["transform"],
        manual["inverse"].to_dataset()["transform"],
        rtol=0,
        atol=2e-12,
    )
    assert_allclose(
        actual["images"].to_dataset()["registered"],
        manual["images"].to_dataset()["registered"],
        rtol=0,
        atol=2e-12,
    )
    assert "stages" in actual
    assert [
        d.n_iterations.item()
        for d in (
            tuple(node.to_dataset() for node in actual["stages"].children.values())
        )
    ] == [2, 5, 4]
    assert actual["diagnostics"].to_dataset().n_iterations == 11
    assert actual["diagnostics"].to_dataset().attrs["stages"] == [
        "rigid",
        "affine",
        "syn",
    ]
    assert [
        d.attrs["stage"]
        for d in (
            tuple(node.to_dataset() for node in actual["stages"].children.values())
        )
    ] == [0, 1, 2]


@pytest.mark.parametrize("terminal", ["affine", "syn"])
def test_pipeline_samples_original_only_once(
    linear_pair_factory, monkeypatch, terminal
):
    from voxreg import linear, syn

    fixed, moving, _ = linear_pair_factory(True, "rigid")
    moving = moving.copy(data=moving.values.astype(np.float64))
    source = moving.copy(deep=True)
    samples = []
    original_sample = linear.sample_rigid

    def observe(images, *args, **kwargs):
        if images.dtype == torch.float64:
            samples.append(images.detach().cpu().numpy().copy())
        return original_sample(images, *args, **kwargs)

    monkeypatch.setattr(linear, "sample_rigid", observe)
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
    assert "forward" in result
    output, valid = resample(moving, fixed, result["forward"].to_dataset()["transform"])
    assert_allclose(
        result["images"].to_dataset()["registered"], output, rtol=0, atol=2e-6
    )
    assert_array_equal(result["images"].to_dataset()["valid"], valid)
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
            return_transforms=True,
            batch_size=batch_size,
            progress=lambda done, total, events=events: events.append((done, total)),
        )
        assert events[-1] == (6, 6)
        assert all(a[0] < b[0] for a, b in pairwise(events))
        assert_array_equal(result["images"].to_dataset()["registered"][0], recording[0])
        assert_array_equal(
            result["diagnostics"].to_dataset().status,
            ["reference", "max_iterations", "invalid_metric"],
        )
        assert_array_equal(result["diagnostics"].to_dataset().n_iterations, [0, 12, 0])
        assert "stages" in result
        assert_array_equal(
            result["stages/1"].to_dataset().status,
            ["reference", "max_iterations", "skipped"],
        )
        assert_array_equal(
            result["stages/1"].to_dataset().level_status[2],
            ["skipped", "skipped"],
        )
        assert_array_equal(
            result["images"].to_dataset()["registered"].acquisition,
            recording.acquisition,
        )
        results.append(result)
    first, second = results
    assert "forward" in first and "forward" in second
    assert_array_equal(
        first["forward"].to_dataset()["transform"].isel(time=0),
        second["forward"].to_dataset()["transform"].isel(time=0),
    )
    # Float32 fitting can select different reduction kernels for different batches.
    assert_allclose(
        first["forward"].to_dataset()["transform"].isel(time=2),
        second["forward"].to_dataset()["transform"].isel(time=2),
        rtol=0,
        atol=1e-12,
    )
    assert (
        max_pull_difference(
            first["forward"].to_dataset()["transform"],
            second["forward"].to_dataset()["transform"],
            fixed,
        )
        < 0.01
    )
    intensities = np.asarray(first["images"].to_dataset()["registered"]).reshape(3, -1)
    difference = intensities - np.asarray(
        second["images"].to_dataset()["registered"]
    ).reshape(3, -1)
    assert (
        np.linalg.norm(difference, axis=1) / np.linalg.norm(intensities, axis=1)
    ).max() < 1e-5


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
    from voxreg import linear

    fixed, moving, _ = linear_pair_factory(True, "rigid")

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid stage configuration reached the optimizer")

    monkeypatch.setattr(linear, "_optimize", unexpected)
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
    assert "stages" in result
    assert (
        list(result["stages"].children.values())[-1].to_dataset().attrs["cc_radius"]
        == 3
    )
    assert (
        list(result["stages"].children.values())[-1]
        .to_dataset()
        .attrs["update_sigma_voxels"]
        == 0.8
    )
    assert (
        list(result["stages"].children.values())[-1]
        .to_dataset()
        .attrs["total_sigma_voxels"]
        == 0
    )
    assert "forward" not in result and "inverse" not in result


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
    assert "stages" in result
    assert result["stages/0"].to_dataset().status == "invalid_metric"
    assert result["stages/1"].to_dataset().status == "skipped"
    assert result["diagnostics"].to_dataset().status == "invalid_metric"
    assert result["diagnostics"].to_dataset().n_iterations == 0
    assert result["images"].to_dataset()["registered"].dtype == np.uint16
    points = position_to_world(fixed) @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones(fixed.size))
    )
    expected = ((seed @ points) - points)[:3].T.reshape(*fixed.shape, 3)
    assert "forward" in result
    assert_allclose(
        result["forward"].to_dataset()["transform"].transpose(..., "world_component"),
        expected,
        rtol=0,
        atol=2e-12,
    )
    sampled, valid = resample(
        moving, fixed, result["forward"].to_dataset()["transform"]
    )
    assert_array_equal(result["images"].to_dataset()["registered"], sampled)
    assert_array_equal(result["images"].to_dataset()["valid"], valid)


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
    assert "stages" in result
    first = result["stages/0"].to_dataset()
    assert first.attrs["initialization"] == "center_moments"
    assert "initialization" not in result["stages/1"].to_dataset().attrs
    if terminal == "affine":
        assert_array_equal(
            first.attrs["rotation_center_zyx_mm"],
            result["stages/1"].to_dataset().attrs["center_zyx_mm"],
        )
    # The first failure is retained; later stages do not attempt a fresh initialization.
    assert result["diagnostics"].to_dataset().status == "invalid_metric"
    assert result["stages/1"].to_dataset().status == "skipped"


@pytest.mark.parametrize("planar", [True, False])
def test_rigid_affine_syn_recovers_physical_truth(linear_pair_factory, device, planar):
    fixed, moving, truth = linear_pair_factory(planar, "rigid")
    result = register(
        moving,
        fixed,
        transform=("rigid", "affine", "syn"),
        max_iterations=100,
        convergence_window_size=1000,
        device=device,
        return_transforms=True,
        progress=False,
    )
    assert "forward" in result
    points = position_to_world(fixed) @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones(fixed.size))
    )
    expected = ((truth @ points) - points)[:3].T.reshape(*fixed.shape, 3)
    spacing = np.linalg.norm(
        position_to_world(fixed)[:3, :3][:, np.array(fixed.shape) > 1], axis=0
    ).min()
    residual = (
        result["forward"]
        .to_dataset()["transform"]
        .transpose(..., "world_component")
        .values
        - expected
    )
    assert np.sqrt(np.mean(np.sum(residual**2, axis=-1))) / spacing < 0.30
    assert result["diagnostics"].to_dataset().min_jacobian > 0
