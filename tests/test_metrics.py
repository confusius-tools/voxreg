"""MI density/gradient oracles and integration across fitting domains."""

import numpy as np
import pytest
import torch
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from oracles import max_pull_difference
from scipy.special import softmax

from voxreg import Stage, position_to_world, register, resample
from voxreg.metrics import _mutual_information
from voxreg.preparation import prepare_fitting_images


def _mi_reference(x, y, valid, bins, sigma):
    result = []
    for a, b, mask in zip(x, y, valid, strict=True):
        a, b = a[mask], b[mask]
        centers = np.arange(bins)
        wa = softmax(-0.5 * ((a[:, None] * (bins - 1) - centers) / sigma) ** 2, axis=1)
        wb = softmax(-0.5 * ((b[:, None] * (bins - 1) - centers) / sigma) ** 2, axis=1)
        joint = np.sum(
            [np.outer(u, v) for u, v in zip(wa, wb, strict=True)], axis=0
        ) / len(a)
        independent = np.outer(joint.sum(axis=1), joint.sum(axis=0))
        nonzero = joint > 0
        result.append(
            -np.sum(joint[nonzero] * np.log(joint[nonzero] / independent[nonzero]))
        )
    return np.array(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mi_masked_density_chunking_and_frame_independence(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(751)
    x, y = rng.uniform(0, 1, (2, 3, 5, 7))
    valid = rng.uniform(size=x.shape) > 0.3
    a = torch.tensor(x, device=device, dtype=torch.float64)
    b = torch.tensor(y, device=device, dtype=torch.float64)
    mask = torch.tensor(valid, device=device)
    expected = _mi_reference(x, y, valid, 9, 0.7)
    for chunk in (4, 500):
        loss, good = _mutual_information(a, b, mask, 9, 0.7, chunk_size=chunk)
        assert good.all()
        assert_allclose(loss.cpu(), expected, rtol=0, atol=1e-12)
        for frame in range(len(a)):
            single, _ = _mutual_information(
                a[frame : frame + 1],
                b[frame : frame + 1],
                mask[frame : frame + 1],
                9,
                0.7,
                chunk_size=chunk,
            )
            assert_allclose(single.cpu(), expected[frame], rtol=0, atol=1e-12)
    changed = a.clone()
    changed[~mask] = torch.nan
    actual, good = _mutual_information(changed, b, mask, 9, 0.7)
    assert good.all()
    assert_allclose(actual.cpu(), expected, rtol=0, atol=1e-12)


def test_mi_both_halfway_gradients_and_invalid_support():
    a = (
        torch.linspace(0.17, 0.83, 18, dtype=torch.float64)
        .reshape(2, 9)
        .requires_grad_()
    )
    b = (
        torch.linspace(0.76, 0.11, 18, dtype=torch.float64)
        .reshape(2, 9)
        .requires_grad_()
    )
    valid = torch.ones_like(a, dtype=torch.bool)
    valid[:, 2] = False
    assert torch.autograd.gradcheck(
        lambda x, y: _mutual_information(x, y, valid, 6, 0.65, chunk_size=4)[0], (a, b)
    )
    loss, _ = _mutual_information(a, b, valid, 6, 0.65, chunk_size=4)
    gradients = torch.autograd.grad(loss.sum(), (a, b))
    for gradient in gradients:
        assert_array_equal(gradient[:, 2], 0)
        assert gradient.abs().sum() > 0
    for mask in (torch.zeros_like(valid), torch.arange(9)[None].expand(2, -1) == 0):
        loss, good = _mutual_information(a, b, mask)
        assert_array_equal(good, False)
        assert_array_equal(loss.detach(), 0)
    loss, good = _mutual_information(torch.ones_like(a), b, valid)
    assert_array_equal(good, False)
    assert_array_equal(loss.detach(), 0)


def test_mi_backward_does_not_retain_full_bin_weights():
    x = torch.linspace(0, 1, 10003)[None].requires_grad_()
    y = x.detach().square().requires_grad_()
    shapes = []

    def save(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        loss, good = _mutual_information(
            x, y, torch.ones_like(x, dtype=torch.bool), chunk_size=257
        )
        loss.sum().backward()
    assert good.all()
    assert (1, 10003, 32) not in shapes
    assert x.grad is not None and y.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


def test_mi_native_ranges_are_per_image_and_keep_double_contrast():
    fixed = np.array([[1e12, 1e12 + 1, 1e12 + 4]])
    moving = np.stack((fixed * 3 + 8, fixed * 7 - 2))
    fit, target = prepare_fitting_images(moving, fixed, metric="mi")
    assert_allclose(fit, np.broadcast_to([[0, 0.25, 1]], fit.shape), rtol=0, atol=1e-7)
    assert_allclose(target, [[0, 0.25, 1]], rtol=0, atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mi_native_range_handles_extreme_finite_values(dtype):
    scale = np.finfo(dtype).max
    values = np.array([[[-scale, 0, scale]]], dtype=dtype)
    original = values.copy()
    fit, target = prepare_fitting_images(values, values[0], metric="mi")
    assert_allclose(fit, [[[0, 0.5, 1]]], rtol=0, atol=1e-7)
    assert_allclose(target, [[0, 0.5, 1]], rtol=0, atol=1e-7)
    assert_array_equal(values, original)


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("model", ["translation", "syn"])
def test_mi_nonlinear_contrast_recovers_independent_physical_truth(
    linear_pair_factory, nonlinear_pair_factory, planar, model
):
    if model == "syn":
        fixed, moving, truth, _ = nonlinear_pair_factory(planar)
    else:
        fixed, moving, truth = linear_pair_factory(planar, "translation")
    moving = moving.copy(data=moving.values**2)
    result = register(
        moving,
        fixed,
        transform=model,
        metric="mi",
        mi_bins=16,
        max_iterations=60,
        convergence_window_size=1000,
        return_transforms=True,
        progress=False,
        shrink_factors=(2, 1) if model == "translation" else (1,),
        smoothing_sigmas=(0.18, 0) if model == "translation" else (0,),
    )
    assert result.forward_transforms is not None
    lattice = position_to_world(fixed)
    ell = np.linalg.norm(lattice[:3, :3], axis=0)[np.array(fixed.shape) > 1].min()
    if model == "syn":
        residual = result.forward_transforms.values - truth
    else:
        points = lattice @ np.vstack(
            (np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size)))
        )
        residual = ((result.forward_transforms.values - truth) @ points)[:3].T
    assert np.sqrt(np.mean(np.sum(residual**2, axis=-1))) / ell <= 0.30


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("model", ["translation", "rigid", "affine", "syn"])
def test_mi_masks_pyramids_and_original_intensity_output(
    linear_pair_factory, device, planar, model
):
    fixed, moving, _ = linear_pair_factory(planar, "rigid")
    moving = moving.copy(data=moving.values**2)
    mask = fixed.copy(data=np.ones(fixed.shape, dtype=bool))
    mask.values[..., :3] = False
    reports = []
    result = register(
        moving,
        fixed,
        transform=model,
        device=device,
        metric="mi",
        mi_bins=12,
        fixed_mask=mask,
        max_iterations=2,
        shrink_factors=(2, 1),
        smoothing_sigmas=(0.18, 0),
        return_transforms=True,
        progress=False,
        metric_callback=lambda frames, clocks, loss: reports.append(loss),
    )
    assert result.diagnostics.attrs["metric"] == "negative mutual information"
    assert result.diagnostics.attrs["objective_units"] == "nats"
    assert result.diagnostics.attrs["mi_bins"] == 12
    assert np.isfinite(result.diagnostics.objective).all()
    assert result.diagnostics.level_objective.shape[-1] == 2
    assert reports and all(np.isfinite(loss).all() for loss in reports)
    assert result.forward_transforms is not None
    output, valid = resample(moving, fixed, result.forward_transforms)
    assert_allclose(result.registered, output, rtol=0, atol=2e-6)
    assert_array_equal(result.valid, valid)


@pytest.mark.parametrize("terminal_metric", ["ncc", "local_ncc"])
def test_mi_stages_motion_failure_isolation_and_overrides(
    linear_pair_factory, device, terminal_metric
):
    fixed, _, _ = linear_pair_factory(True, "rigid")
    moving = xr.concat(
        [fixed, fixed.copy(data=np.roll(fixed.values, 1, axis=-1)), fixed],
        dim=xr.IndexVariable("time", [1, 3, 5]),
    )
    mask = moving.copy(data=np.ones(moving.shape, dtype=bool))
    mask.values[2] = False
    results = []
    for batch in (1, 3):
        result = register(
            moving,
            transform=(
                Stage("rigid", metric="mi", mi_bins=8),
                Stage("syn", metric=terminal_metric),
            ),
            moving_mask=mask,
            device=device,
            metric="mi",
            max_iterations=2,
            batch_size=batch,
            return_transforms=True,
            progress=False,
        )
        assert result.stage_diagnostics is not None
        assert result.stage_diagnostics[0].attrs["mi_bins"] == 8
        assert result.stage_diagnostics[1].attrs["metric"] == (
            "negative squared global NCC"
            if terminal_metric == "ncc"
            else "negative squared local NCC"
        )
        assert_array_equal(
            result.diagnostics.status,
            ["reference", "max_iterations", "insufficient_overlap"],
        )
        assert_array_equal(result.registered[0], moving[0])
        results.append(result)
    assert (
        max_pull_difference(
            results[0].forward_transforms, results[1].forward_transforms, fixed
        )
        < 0.01
    )


@pytest.mark.parametrize(
    "options",
    [
        {"metric": "mattes"},
        {"metric": "local_ncc"},
        {"mi_bins": 1},
        {"mi_bins": 257},
        {"mi_bins": True},
        {"mi_sigma": 0},
        {"mi_sigma": np.nan},
        {"mi_sigma": np.inf},
        {"mi_sigma": True},
        {"mi_sigma": 11},
    ],
)
def test_metric_validation_before_fitting(linear_pair_factory, options):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    with pytest.raises(ValueError, match="metric|mi_"):
        register(moving, fixed, **options, progress=False)


def test_mi_later_stage_validation_is_eager(linear_pair_factory, monkeypatch):
    from voxreg import linear

    fixed, moving, _ = linear_pair_factory(True, "rigid")

    def unexpected(*args, **kwargs):
        pytest.fail("invalid metric reached fitting")

    monkeypatch.setattr(linear, "_optimize", unexpected)
    with pytest.raises(ValueError, match="mi_bins"):
        register(
            moving,
            fixed,
            transform=("rigid", Stage("affine", metric="mi", mi_bins=1)),
            progress=False,
        )
