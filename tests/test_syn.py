"""Public SyN contracts: independent nonlinear truth, original samples and batches."""

import numpy as np
import pytest
import torch
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from registration_options import RegistrationOptions
from scipy.ndimage import map_coordinates

from voxreg import position_to_world, register
from voxreg.metrics import compute_local_ncc
from voxreg.sampling import _sample_displaced


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize(
    "options",
    [
        {"syn_cc_radius": 2, "syn_update_sigma": 1.5, "syn_total_sigma": 0.5},
        {"syn_cc_radius": 1},
        {"syn_update_sigma": 0.0},
        {"syn_total_sigma": 0.0},
        {"syn_update_sigma": np.float64(0.8), "syn_total_sigma": np.int64(1)},
    ],
)
def test_public_syn_controls(nonlinear_pair_factory, device, planar, options):
    fixed, moving, _, _ = nonlinear_pair_factory(planar)
    common: RegistrationOptions = {
        "transform": "syn",
        "intensity_scaling": "none",
        "return_transforms": True,
        "max_iterations": 3,
        "device": device,
        "progress": False,
    }
    baseline = register(moving, fixed, **common)
    result = register(moving, fixed, **common, **options)
    for public, diagnostic, default in (
        ("syn_cc_radius", "cc_radius", 2),
        ("syn_update_sigma", "update_sigma_voxels", 1.5),
        ("syn_total_sigma", "total_sigma_voxels", 0.5),
    ):
        assert result.diagnostics.attrs[diagnostic] == options.get(public, default)
    assert result.diagnostics.n_iterations.item() == 3
    assert np.isfinite(result.diagnostics.objective).all()
    assert result.diagnostics.min_jacobian.item() > 0
    assert result.forward_transforms is not None
    assert baseline.forward_transforms is not None
    if options == {
        "syn_cc_radius": 2,
        "syn_update_sigma": 1.5,
        "syn_total_sigma": 0.5,
    }:
        xr.testing.assert_identical(result.registered, baseline.registered)
        xr.testing.assert_identical(result.valid, baseline.valid)
        xr.testing.assert_identical(
            result.forward_transforms, baseline.forward_transforms
        )
        xr.testing.assert_identical(
            result.inverse_transforms, baseline.inverse_transforms
        )
    else:
        assert not np.array_equal(
            result.forward_transforms.values, baseline.forward_transforms.values
        )


@pytest.mark.parametrize(
    "name, values",
    [
        ("syn_cc_radius", [0, -1, 1.5, True, None, "2", np.nan, np.inf]),
        ("syn_update_sigma", [-1, True, None, "1.5", np.nan, np.inf, -np.inf]),
        ("syn_total_sigma", [-1, True, None, "0.5", np.nan, np.inf, -np.inf]),
    ],
)
def test_public_syn_controls_validate(linear_pair_factory, name, values):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    for value in values:
        with pytest.raises(ValueError, match=name):
            register(moving, fixed, transform="syn", **{name: value})


@pytest.mark.parametrize("transform", ["translation", "rigid", "affine"])
@pytest.mark.parametrize(
    "options",
    [
        {"syn_cc_radius": 3},
        {"syn_update_sigma": 0},
        {"syn_total_sigma": 0},
    ],
)
def test_public_syn_controls_reject_linear_models(
    linear_pair_factory, transform, options
):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    with pytest.raises(ValueError, match='require transform="syn"'):
        register(moving, fixed, transform=transform, **options)


@pytest.mark.parametrize("shape", [(23, 31), (7, 9, 11)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_ncc_rejects_roundoff_in_constant_neighborhoods(device, shape, dtype):
    values = torch.full(
        (2, *shape), 0.1, device=device, dtype=dtype, requires_grad=True
    )
    loss, meaningful = compute_local_ncc(
        values, values.detach(), torch.ones_like(values, dtype=torch.bool), 2
    )
    assert not meaningful.any()
    assert torch.equal(loss, torch.zeros_like(loss))
    gradient = torch.autograd.grad(loss.sum(), values)[0]
    assert torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_ncc_follows_fitting_tensor_precision(device, dtype):
    values = torch.arange(40, dtype=dtype, device=device).reshape(2, 4, 5) / 40
    moving = values.sin().requires_grad_()
    fixed = (values * 3).cos()
    valid = torch.ones_like(moving, dtype=torch.bool)
    loss, meaningful = compute_local_ncc(moving, fixed, valid, radius=1)
    assert loss.dtype == dtype and meaningful.all() and torch.isfinite(loss).all()
    gradient = torch.autograd.grad(loss.sum(), moving)[0]
    assert gradient.dtype == dtype and torch.isfinite(gradient).all()


def _ell(data):
    return np.linalg.norm(
        position_to_world(data)[:3, 1 if data.sizes["k"] == 1 else 0 : 3], axis=0
    ).min()


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("fixed_budget", [False, True])
@pytest.mark.parametrize("axis_aligned", [False, True])
def test_syn_known_nonlinear_recovery(
    nonlinear_pair_factory, device, planar, fixed_budget, axis_aligned
):
    fixed, moving, truth, inverse = nonlinear_pair_factory(
        planar, axis_aligned=axis_aligned
    )
    source = moving.copy(deep=True)
    budget: RegistrationOptions = (
        {"max_iterations": 120, "convergence_window_size": 1000} if fixed_budget else {}
    )
    result = register(
        moving,
        fixed,
        return_transforms=True,
        transform="syn",
        intensity_scaling="none",
        device=device,
        **budget,
    )
    ell = _ell(fixed)
    points = np.indices(fixed.shape).reshape(3, -1).T
    design = np.column_stack((points, np.ones(len(points))))
    best_linear = design @ np.linalg.lstsq(design, truth.reshape(-1, 3), rcond=None)[0]
    # Even the best affine correspondence cannot meet the nonlinear truth gate.
    assert (
        np.sqrt(np.mean(np.sum((best_linear - truth.reshape(-1, 3)) ** 2, axis=1)))
        / ell
        > 0.25
    )
    for actual, expected in [
        (result.transforms, truth),
        (result.inverse_transforms, inverse),
    ]:
        assert actual is not None
        assert actual.dims == ("k", "j", "i", "world_component")
        assert actual.dtype == np.float64
        assert list(actual.world_component.values) == ["z", "y", "x"]
        assert actual.attrs["representation"] == "displacement"
        assert (
            np.sqrt(np.mean(np.sum((actual.values - expected) ** 2, axis=-1))) / ell
            < 0.25
        )
    assert result.diagnostics.min_jacobian > 0
    assert np.isfinite(result.diagnostics.inverse_error)
    assert result.diagnostics.ncc_squared > 0.98
    assert result.registered.dims == fixed.dims
    assert result.registered.dtype == moving.dtype
    assert result.registered.attrs == moving.attrs
    xr.testing.assert_identical(moving, source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("fixed_budget", [False, True])
@pytest.mark.parametrize("axis_aligned", [False, True])
def test_syn_cpu_cuda_point_gate(
    nonlinear_pair_factory, planar, fixed_budget, axis_aligned
):
    fixed, moving, _, _ = nonlinear_pair_factory(planar, axis_aligned=axis_aligned)
    budget: RegistrationOptions = (
        {"max_iterations": 120, "convergence_window_size": 1000} if fixed_budget else {}
    )
    results = [
        register(
            moving,
            fixed,
            return_transforms=True,
            transform="syn",
            intensity_scaling="none",
            device=d,
            **budget,
        )
        for d in ("cpu", "cuda")
    ]
    for name in ["transforms", "inverse_transforms"]:
        difference = getattr(results[0], name).values - getattr(results[1], name).values
        assert np.linalg.norm(difference, axis=-1).max() / _ell(fixed) < 0.01


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("dtype", [np.float64, np.uint16])
def test_syn_complete_seed_and_original_output(
    linear_pair_factory, device, planar, dtype
):
    fixed, moving, seed = linear_pair_factory(planar, "affine")
    if dtype == np.uint16:
        moving = moving.copy(data=np.rint(moving.values * 3000).astype(dtype))
    else:
        moving = moving.copy(data=2**30 + moving.values.astype(dtype) * 0.001)
        fixed = fixed.copy(data=2**30 + fixed.values.astype(dtype) * 0.001)
    moving.name = "power"
    moving.attrs.update(units="power", subject="unchanged")
    original = moving.copy(deep=True)
    r = register(
        moving,
        fixed,
        return_transforms=True,
        transform="syn",
        initial_transform=seed,
        intensity_scaling="none",
        device=device,
        max_iterations=1,
        learning_rate=1e-12,
    )
    assert r.inverse_transforms is not None
    assert r.transforms is not None
    points = position_to_world(fixed) @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size)))
    )
    expected = ((seed @ points) - points)[:3].T.reshape(*fixed.shape, 3)
    assert_allclose(r.transforms.values, expected, rtol=0, atol=2e-12)
    mapped = (
        np.linalg.inv(position_to_world(moving))
        @ np.vstack(
            (
                (points[:3].T + r.transforms.values.reshape(-1, 3)).T,
                np.ones((1, fixed.size)),
            )
        )
    )[:3]
    valid = (
        (mapped >= -1e-5) & (mapped <= np.array(moving.shape)[:, None] - 1 + 1e-5)
    ).all(axis=0)
    sampled = map_coordinates(
        moving.values.astype(float), mapped, order=1, mode="nearest", prefilter=False
    )
    sampled = np.where(valid, sampled, moving.values.min()).reshape(fixed.shape)
    if dtype == np.uint16:
        sampled = np.rint(sampled).clip(0, np.iinfo(dtype).max).astype(dtype)
    assert_allclose(
        r.registered.values,
        sampled,
        rtol=0,
        atol=0 if dtype == np.uint16 else 8 * np.spacing(float(moving.values.max())),
    )
    assert_array_equal(r.valid.values, valid.reshape(fixed.shape))
    assert r.registered.xindexes["z"].equals(fixed.xindexes["z"])
    assert r.inverse_transforms.xindexes["z"].equals(moving.xindexes["z"])
    assert r.registered.name == moving.name and r.registered.attrs == moving.attrs
    xr.testing.assert_identical(moving, original)


@pytest.mark.parametrize("planar", [True, False])
def test_syn_reference_failure_isolation_and_chunks(
    nonlinear_pair_factory, device, planar
):
    fixed, moving, truth, _ = nonlinear_pair_factory(planar)
    recording = xr.concat(
        [fixed, moving, moving.copy(data=np.full_like(moving.values, 0.5))],
        dim=xr.IndexVariable("time", [2.0, 4.0, 8.0]),
    )
    recording = recording.assign_coords(acquisition=("time", [10, 11, 12]))
    recording.attrs = moving.attrs.copy()
    recording.name = moving.name
    results = []
    for batch in [1, 3]:
        events = []
        result = register(
            recording,
            return_transforms=True,
            transform="syn",
            intensity_scaling="none",
            device=device,
            max_iterations=60,
            convergence_window_size=1000,
            batch_size=batch,
            progress=lambda n, total, events=events: events.append((n, total)),
        )
        assert result.transforms is not None
        results.append(result)
        assert events[-1] == (3, 3)
        assert_array_equal(result.registered[0], recording[0])
        assert_array_equal(result.transforms[0], 0)
        assert_array_equal(result.transforms[2], 0)
        assert list(result.diagnostics.status.values) == [
            "reference",
            "max_iterations",
            "invalid_metric",
        ]
        assert_array_equal(result.diagnostics.n_iterations, [0, 60, 0])
        assert_array_equal(result.registered.time, recording.time)
        assert_array_equal(result.registered.acquisition, recording.acquisition)
        assert (
            np.sqrt(
                np.mean(np.sum((result.transforms[1].values - truth) ** 2, axis=-1))
            )
            / _ell(fixed)
            < 0.25
        )
    assert results[0].transforms is not None and results[1].transforms is not None
    assert (
        np.linalg.norm(
            results[0].transforms.values - results[1].transforms.values, axis=-1
        ).max()
        / _ell(fixed)
        < 0.001
    )


def test_syn_metric_callback_ownership_and_exceptions(nonlinear_pair_factory):
    fixed, moving, _, _ = nonlinear_pair_factory(True)
    reports = []

    def callback(i, n, o):
        reports.append((i.copy(), n.copy(), o.copy()))
        i[:] = -100
        n[:] = -100
        o[:] = np.nan

    result = register(
        moving,
        fixed,
        transform="syn",
        intensity_scaling="none",
        max_iterations=3,
        metric_interval=2,
        metric_callback=callback,
    )
    assert [int(n[0]) for _, n, _ in reports] == [0, 2, 3]
    assert np.isfinite(result.diagnostics.objective)

    def fail(*args):
        raise RuntimeError("caller error")

    with pytest.raises(RuntimeError, match="caller error"):
        register(
            moving,
            fixed,
            transform="syn",
            intensity_scaling="none",
            max_iterations=1,
            metric_callback=fail,
        )


def test_syn_rejects_mixed_and_reflected_seed(
    slice_volume_pair_factory, linear_pair_factory
):
    volume, plane, _ = slice_volume_pair_factory("rigid")
    with pytest.raises(ValueError, match="mixed-dimensional"):
        register(plane, volume, transform="syn")
    fixed, moving, _ = linear_pair_factory(False)
    reflection = np.eye(4)
    reflection[0, 0] = -1
    with pytest.raises(ValueError, match="orientation"):
        register(moving, fixed, transform="syn", initial_transform=reflection)


@pytest.mark.parametrize("planar", [True, False])
def test_displaced_interpolation_identity_and_gradients(
    nonlinear_pair_factory, device, planar
):
    _, moving, _, _ = nonlinear_pair_factory(planar)
    data = moving.values[0] if planar else moving.values
    image = torch.tensor(data[None], device=device)
    identity = torch.stack(
        torch.meshgrid(
            *(
                torch.linspace(-1, 1, n, device=device, dtype=torch.float64)
                for n in data.shape
            ),
            indexing="ij",
        ),
        dim=-1,
    ).flip(-1)[None]
    assert_array_equal(_sample_displaced(image, identity, "linear").cpu(), data[None])
    # Double output keeps fractional detail; no float32 interpolation hidden inside.
    query = (identity * 0.8 + 0.013).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda p: _sample_displaced(image.double(), p, "linear"),
        (query,),
        fast_mode=True,
        eps=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("planar", [True, False])
def test_syn_float32_working_fields_keep_double_geometry_and_original_data(
    nonlinear_pair_factory, monkeypatch, device, planar
):
    from voxreg import syn

    fixed, moving, *_ = nonlinear_pair_factory(planar)
    fixed = fixed.copy(data=2**30 + fixed.values.astype(np.float64) * 0.001)
    moving = moving.copy(data=2**30 + moving.values.astype(np.float64) * 0.001)
    untouched = moving.copy(deep=True)
    candidate, inverse = syn._build_candidate_fields, syn._compute_final_inverse
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    calls = []

    def candidate_checked(uf, um, gf, gm, grid, *args):
        assert all(t.dtype == torch.float32 for t in (uf, um, gf, gm))
        assert grid.indices.dtype == grid.inverse_lattice.dtype == torch.float32
        assert grid.basis.dtype == grid.lattice.dtype == torch.float64
        calls.append("candidate")
        result = candidate(uf, um, gf, gm, grid, *args)
        assert result[0].dtype == result[1].dtype == torch.float32
        return result

    def inverse_checked(field, grid):
        assert field.dtype == grid.indices.dtype == torch.float64
        calls.append("inverse")
        return inverse(field, grid)

    monkeypatch.setattr(syn, "_build_candidate_fields", candidate_checked)
    monkeypatch.setattr(syn, "_compute_final_inverse", inverse_checked)
    result = register(
        moving,
        fixed,
        transform="syn",
        intensity_scaling="none",
        device=device,
        max_iterations=2,
        return_transforms=True,
    )
    assert calls.count("candidate") == calls.count("inverse") == 2
    assert flags == (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    assert result.transforms is not None and result.inverse_transforms is not None
    assert result.transforms.dtype == result.inverse_transforms.dtype == np.float64
    assert result.registered.dtype == np.float64
    assert float(result.registered.max() - result.registered.min()) > 0
    xr.testing.assert_identical(moving, untouched)


@pytest.mark.parametrize("planar", [True, False])
def test_syn_float32_fitting_is_independent_of_large_world_origin(
    nonlinear_pair_factory, device, planar
):
    from oracles import voxel

    fixed, moving, *_ = nonlinear_pair_factory(planar)
    affine = position_to_world(fixed).copy()
    affine[:3, 3] += [1e9, -2e9, 3e9]
    options: RegistrationOptions = {
        "transform": "syn",
        "intensity_scaling": "none",
        "device": device,
        "max_iterations": 4,
        "return_transforms": True,
    }
    original = register(moving, fixed, **options)
    shifted = register(
        voxel(moving.values, affine), voxel(fixed.values, affine), **options
    )
    assert original.transforms is not None and shifted.transforms is not None
    assert np.any(original.transforms.values != 0)
    assert_allclose(shifted.transforms, original.transforms, rtol=0, atol=1e-9)
    assert_array_equal(shifted.registered, original.registered)
    assert_array_equal(shifted.valid, original.valid)


def test_syn_rejects_unrepresentable_float32_working_lattice(nonlinear_pair_factory):
    from oracles import voxel

    fixed, moving, *_ = nonlinear_pair_factory(False)
    affine = np.diag([1e-40, 1e-40, 1e-40, 1.0])
    with pytest.raises(ValueError, match="working lattice.*float32"):
        register(
            voxel(moving.values, affine),
            voxel(fixed.values, affine),
            transform="syn",
            max_iterations=1,
        )


@pytest.mark.parametrize("planar", [True, False])
def test_syn_exchange_symmetry_is_not_one_way(nonlinear_pair_factory, device, planar):
    fixed, moving, _, _ = nonlinear_pair_factory(planar)
    options: RegistrationOptions = {
        "transform": "syn",
        "intensity_scaling": "none",
        "device": device,
        "max_iterations": 40,
        "convergence_window_size": 1000,
    }
    forward = register(moving, fixed, return_transforms=True, **options)
    assert forward.inverse_transforms is not None
    backward = register(fixed, moving, return_transforms=True, **options)
    assert backward.transforms is not None
    # A same-domain mathematical symmetry check, not anatomical accuracy evidence.
    assert (
        np.linalg.norm(
            forward.inverse_transforms.values - backward.transforms.values, axis=-1
        ).max()
        / _ell(fixed)
        < 0.001
    )


def test_displaced_nearest_uses_even_ties():
    image = torch.arange(5, dtype=torch.float64)[None, None].expand(1, 3, 5)
    query = torch.tensor(
        [[[[-0.75, 0], [-0.25, 0], [0.25, 0], [0.75, 0]]]], dtype=torch.float64
    )
    assert_array_equal(_sample_displaced(image, query, "nearest"), [[[0, 2, 2, 4]]])


@pytest.mark.parametrize("shape", [(6, 7), (4, 5, 6)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_ncc_against_explicit_neighborhoods(device, shape, dtype):
    rng = np.random.default_rng(55)
    x, y = rng.normal(size=(2, *shape)), rng.normal(size=(2, *shape))
    if dtype == torch.float32:
        x, y = x.astype(np.float32), y.astype(np.float32)
    valid = rng.uniform(size=x.shape) > 0.2
    expected = []
    for frame in range(2):
        scores = []
        for point in np.ndindex(shape):
            region = tuple(slice(max(0, p - 1), p + 2) for p in point)
            mask = valid[frame][region]
            if valid[(frame, *point)] and mask.sum() > 1:
                scores.append(
                    np.corrcoef(x[frame][region][mask], y[frame][region][mask])[0, 1]
                    ** 2
                )
        expected.append(-np.mean(scores))
    actual, meaningful = compute_local_ncc(
        torch.tensor(x, device=device),
        torch.tensor(y, device=device),
        torch.tensor(valid, device=device),
        1,
    )
    assert meaningful.all()
    assert_allclose(
        actual.cpu(), expected, rtol=0, atol=2e-6 if dtype == torch.float32 else 1e-13
    )
