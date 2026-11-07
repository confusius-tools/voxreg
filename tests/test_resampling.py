"""Offline pulls reuse physical sampling, original intensities and GPU execution."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import xarray as xr
from confusius import create_voxeldata
from scipy.ndimage import map_coordinates

from voxreg import position_to_world, register, resample


@pytest.mark.parametrize("model", ["translation", "rigid", "affine", "syn"])
@pytest.mark.parametrize("planar", [True, False])
def test_exported_forward_matches_registration(
    linear_pair_factory, device, model, planar
):
    fixed, moving, _ = linear_pair_factory(planar, model="rigid")
    fitted = register(
        moving,
        fixed,
        transform=model,
        return_transforms=True,
        max_iterations=2,
        intensity_scaling="none",
        device=device,
        progress=False,
    )
    assert fitted.forward_transforms is not None
    assert fitted.inverse_transforms is not None
    actual, valid = resample(moving, fixed, fitted.forward_transforms, device=device)
    np.testing.assert_allclose(actual, fitted.registered, rtol=2e-6, atol=2e-6)
    xr.testing.assert_identical(valid, fitted.valid)
    assert actual.dtype == moving.dtype
    assert actual.xindexes["z"].equals(fixed.xindexes["z"])
    # Reverse application is executable for every exported transform type.
    reverse, reverse_valid = resample(
        fixed, moving, fitted.inverse_transforms, device=device
    )
    assert reverse.shape == moving.shape and reverse_valid.dtype == bool


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
def test_physical_pull_matches_scipy(device, dtype, interpolation):
    rng = np.random.default_rng(12)
    matrix = np.array(
        [[0.3, 0.04, 0, 1], [0, 0.2, 0.03, -2], [0.02, 0, 0.1, 3], [0, 0, 0, 1]]
    )
    moving = create_voxeldata(
        (rng.random((7, 9, 11)) * 100).astype(dtype),
        dims=("k", "j", "i"),
        voxel_to_world=matrix,
        k=np.arange(7) * 2 + 4,
        j=np.arange(9)[::-1] + 5,
        i=np.arange(11) * 3 + 2,
    )
    reference = moving.isel(k=slice(1, 6), j=slice(0, 7, 2), i=slice(1, 10, 2))
    pull = np.eye(4)
    pull[:3, :3] = [[1, 0.02, 0], [0, 1, -0.01], [0.01, 0, 1]]
    pull[:3, 3] = [0.11, -0.03, 0.17]
    indices = np.indices(reference.shape).reshape(3, -1)
    mapping = (
        np.linalg.inv(position_to_world(moving)) @ pull @ position_to_world(reference)
    )
    query = mapping[:3, :3] @ indices + mapping[:3, 3, None]
    expected_valid = (
        (query >= 0) & (query <= np.array(moving.shape)[:, None] - 1)
    ).all(axis=0)
    expected = map_coordinates(
        moving.values.astype(np.float64),
        query,
        order=1 if interpolation == "linear" else 0,
        mode="nearest",
    )
    expected[~expected_valid] = moving.values.min()
    if np.issubdtype(dtype, np.integer):
        expected = np.rint(expected)
    actual, valid = resample(
        moving, reference, pull, interpolation=interpolation, device=device
    )
    np.testing.assert_array_equal(valid.values.flatten(), expected_valid)
    np.testing.assert_allclose(
        actual.values.flatten(), expected.astype(dtype), rtol=2e-6, atol=2e-5
    )
    assert actual.dtype == dtype


def test_recordings_broadcast_and_batch(linear_pair_factory, device):
    fixed, moving, _ = linear_pair_factory(True, model="rigid")
    series = xr.concat(
        [moving, moving * 2, moving * 3], dim=xr.IndexVariable("time", [2.0, 3.0, 4.0])
    )
    matrices = xr.DataArray(
        np.repeat(np.eye(4)[None], 3, axis=0),
        dims=("time", "world_out", "world_in"),
        coords={
            "time": series.time,
            "world_out": ["z", "y", "x", "1"],
            "world_in": ["z", "y", "x", "1"],
        },
    )
    once, mask = resample(series, fixed, matrices, batch_size=1, device=device)
    batched, batched_mask = resample(
        series, fixed, np.eye(4), batch_size=3, device=device
    )
    xr.testing.assert_identical(once, batched)
    xr.testing.assert_identical(mask, batched_mask)
    repeated, _ = resample(moving, fixed, matrices, device=device)
    xr.testing.assert_equal(repeated.time, series.time)
    for frame in repeated:
        np.testing.assert_array_equal(frame, repeated[0])
    with pytest.raises(ValueError, match="time labels"):
        resample(series, fixed, matrices.isel(time=slice(None, None, -1)))


def test_dense_field_broadcast_and_validation(linear_pair_factory, device):
    fixed, moving, _ = linear_pair_factory(True, model="rigid")
    series = xr.concat([moving, moving * 2], dim=xr.IndexVariable("time", [3.0, 4.0]))
    fitted = register(
        series,
        fixed,
        transform="syn",
        return_transforms=True,
        max_iterations=1,
        intensity_scaling="none",
        device=device,
        progress=False,
    )
    assert fitted.forward_transforms is not None
    assert fitted.inverse_transforms is not None
    actual, valid = resample(
        series, fixed, fitted.forward_transforms, batch_size=1, device=device
    )
    np.testing.assert_allclose(actual, fitted.registered, rtol=2e-6, atol=2e-6)
    xr.testing.assert_identical(valid, fitted.valid)
    spatial_field = fitted.forward_transforms.isel(time=0, drop=True)
    a, _ = resample(series, fixed, spatial_field, device=device)
    b, _ = resample(
        series,
        fixed,
        xr.concat([spatial_field, spatial_field], dim=series.time),
        device=device,
    )
    xr.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-14)
    reverse, _ = resample(fixed, moving, fitted.inverse_transforms, device=device)
    np.testing.assert_array_equal(reverse.time, series.time)
    with pytest.raises(ValueError, match="reference grid"):
        resample(moving, fixed.isel(i=slice(1, None)), spatial_field)
    bad = spatial_field.assign_attrs(spatial_units="voxels")
    with pytest.raises(ValueError, match="mm"):
        resample(moving, fixed, bad)
    with pytest.raises(ValueError, match="ordered z/y/x"):
        resample(moving, fixed, spatial_field.isel(world_component=[2, 1, 0]))


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
def test_dense_translation_matches_matrix(device, planar, interpolation):
    shape = (1 if planar else 5, 7, 9)
    moving = create_voxeldata(
        np.random.default_rng(3).uniform(1, 10, shape).astype(np.float32),
        dims=("k", "j", "i"),
        voxel_to_world=np.diag([0.2, 0.1, 0.1, 1]),
    )
    offset = [0.0 if planar else 0.04, 0.03, -0.02]
    field = xr.concat(
        [moving.copy(data=np.full(shape, x)) for x in offset],
        dim=xr.IndexVariable("world_component", ["z", "y", "x"]),
    ).transpose("k", "j", "i", "world_component")
    field.attrs = {"representation": "displacement", "spatial_units": "mm"}
    matrix = np.eye(4)
    matrix[:3, 3] = offset
    dense, dense_valid = resample(
        moving, moving, field, device=device, interpolation=interpolation
    )
    linear, linear_valid = resample(
        moving, moving, matrix, device=device, interpolation=interpolation
    )
    xr.testing.assert_allclose(dense, linear, rtol=2e-6, atol=2e-6)
    xr.testing.assert_identical(dense_valid, linear_valid)


def test_singleton_support_and_labels(device):
    moving = create_voxeldata(
        np.arange(20, dtype=np.uint16).reshape(1, 4, 5),
        dims=("k", "j", "i"),
        voxel_to_world=np.eye(4),
    )
    for image in (moving, moving > 10):
        actual, valid = resample(
            image, moving, np.eye(4), interpolation="nearest", device=device
        )
        xr.testing.assert_identical(actual, image)
        assert valid.values.all()
    for depth, supported in [(-0.5, True), (0.49, True), (0.5, False)]:
        pull = np.eye(4)
        pull[0, 3] = depth
        _, valid = resample(
            moving, moving, pull, interpolation="nearest", device=device
        )
        assert valid.values.all() == supported
    reference = create_voxeldata(
        np.zeros((3, 4, 5)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    )
    actual, valid = resample(
        moving.astype(np.float32), reference, np.eye(4), device=device
    )
    np.testing.assert_array_equal(actual[0], moving[0])
    assert valid[0].values.all() and not valid[1:].values.any()


@pytest.mark.parametrize("moving_slice", [False, True])
@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
def test_mixed_support_matches_registration_and_scipy(
    device, moving_slice, interpolation
):
    """Exported mixed pulls retain finite voxel support on every source axis."""
    from oracles import voxel

    shape = (1, 4, 5) if moving_slice else (4, 4, 5)
    moving = voxel(np.arange(np.prod(shape), dtype=float).reshape(shape), np.eye(4))
    affine = np.eye(4)
    affine[:3, 3] = -0.5
    affine[0, 0] = 0.25
    reference = voxel(np.arange(180, dtype=float).reshape(6, 5, 6), affine)
    if not moving_slice:
        reference = reference.isel(k=slice(0, 1))
    fitted = register(
        moving,
        reference,
        transform="translation",
        return_transforms=True,
        intensity_scaling="none",
        interpolation=interpolation,
        learning_rate=1e-20,
        max_iterations=1,
        progress=False,
        device=device,
    )
    assert fitted.forward_transforms is not None
    actual, valid = resample(
        moving,
        reference,
        fitted.forward_transforms,
        interpolation=interpolation,
        device=device,
    )
    query = (
        affine[:3, :3] @ np.indices(reference.shape).reshape(3, -1)
        + affine[:3, 3, None]
    )
    expected_valid = ((query >= -0.5) & (query < np.array(shape)[:, None] - 0.5)).all(
        axis=0
    )
    expected = map_coordinates(
        moving.values,
        query if interpolation == "linear" else np.rint(query),
        order=1 if interpolation == "linear" else 0,
        mode="nearest",
        prefilter=False,
    )
    expected[~expected_valid] = moving.values.min()
    np.testing.assert_array_equal(valid.values.ravel(), expected_valid)
    np.testing.assert_allclose(actual.values.ravel(), expected, atol=1e-12)
    xr.testing.assert_identical(valid, fitted.valid)
    xr.testing.assert_allclose(actual, fitted.registered)


@pytest.mark.parametrize(
    "bad", [np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan), np.empty((0, 4, 4))]
)
def test_invalid_matrices(linear_pair_factory, bad):
    fixed, moving, _ = linear_pair_factory(True)
    with pytest.raises(ValueError):
        resample(moving, fixed, bad)


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"device": "bad"},
        {"interpolation": "cubic"},
    ],
)
def test_invalid_options(linear_pair_factory, options):
    fixed, moving, _ = linear_pair_factory(True)
    with pytest.raises(ValueError):
        resample(moving, fixed, np.eye(4), **options)


def test_invalid_inputs(linear_pair_factory):
    fixed, moving, _ = linear_pair_factory(True)
    with pytest.raises(ValueError, match="nearest"):
        resample(moving > 0, fixed, np.eye(4))
    with pytest.raises(ValueError, match="finite"):
        resample(moving.copy(data=np.full(moving.shape, np.nan)), fixed, np.eye(4))
    with pytest.raises(ValueError, match="2\\*\\*53"):
        resample(
            moving.copy(data=np.full(moving.shape, 2**54, dtype=np.int64)),
            fixed,
            np.eye(4),
        )


@pytest.mark.parametrize("bad_labels", ["missing", "scalar", "duplicate"])
@pytest.mark.parametrize("on_transform", [False, True])
def test_rejects_invalid_time_coordinates(
    linear_pair_factory, bad_labels, on_transform
):
    """Reject malformed labels before broadcasting or allocating sampling grids."""
    fixed, moving, _ = linear_pair_factory(True)
    labeled = (
        xr.DataArray(
            np.repeat(np.eye(4)[None], 2, axis=0),
            dims=("time", "world_out", "world_in"),
            coords={
                "time": [0, 1],
                "world_out": ["z", "y", "x", "1"],
                "world_in": ["z", "y", "x", "1"],
            },
        )
        if on_transform
        else xr.concat([moving, moving], dim=xr.IndexVariable("time", [0, 1]))
    )
    if bad_labels == "missing":
        labeled = labeled.drop_vars("time")
    elif bad_labels == "scalar":
        labeled = labeled.drop_indexes("time").assign_coords(time=0)
    else:
        labeled = labeled.assign_coords(time=[0, 0])
    with pytest.raises(ValueError, match="time labels"):
        resample(
            moving if on_transform else labeled,
            fixed,
            labeled if on_transform else np.eye(4),
        )


@pytest.mark.parametrize("planar", [True, False])
def test_linear_native_double_and_cpu_shared_point_tiles(monkeypatch, planar, device):
    shape = (1 if planar else 3, 5, 7)
    moving = create_voxeldata(
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        dims=("k", "j", "i"),
        voxel_to_world=np.eye(4),
    )
    calls = []
    native = F.grid_sample

    def sample(source, grid, **options):
        calls.append((source.shape, source.stride(0), source.dtype, grid.dtype))
        return native(source, grid, **options)

    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    monkeypatch.setattr(F, "grid_sample", sample)
    try:
        actual, valid = resample(moving, moving, np.eye(4), device=device)
    finally:
        torch.set_num_threads(threads)
    np.testing.assert_array_equal(actual.values, moving.values)
    assert valid.values.all()
    assert len(calls) == 1
    sampled_shape, stride, source_dtype, grid_dtype = calls[0]
    assert source_dtype == grid_dtype == torch.float64
    if planar:
        assert sampled_shape == (1, 1, 5, 7)
    elif device == "cpu":
        # 105 points require padding across four shared-image tasks.
        assert sampled_shape == (4, 1, 3, 5, 7)
        assert stride == 0
    else:
        assert sampled_shape == (1, 1, 3, 5, 7)


@pytest.mark.parametrize("field", [False, True])
def test_shared_queries_preserve_inputs_and_batches(field, device):
    shape = (3, 5, 7)
    moving = create_voxeldata(
        np.random.default_rng(14).uniform(1, 10, shape).astype(np.float32),
        dims=("k", "j", "i"),
        voxel_to_world=np.eye(4),
    )
    series = xr.concat(
        [moving, moving * 2, moving * 3],
        dim=xr.IndexVariable("time", [2.0, 4.0, 8.0]),
    )
    if field:
        mapping = xr.concat(
            [moving.copy(data=np.full(shape, value)) for value in (0.1, -0.2, 0.3)],
            dim=xr.IndexVariable("world_component", ["z", "y", "x"]),
        ).transpose("k", "j", "i", "world_component")
        mapping.attrs = {"representation": "displacement", "spatial_units": "mm"}
    else:
        mapping = np.eye(4)
        mapping[:3, 3] = [0.1, -0.2, 0.3]
    before_image = series.copy(deep=True)
    before_map = (
        mapping.copy(deep=True) if isinstance(mapping, xr.DataArray) else mapping.copy()
    )
    once, once_valid = resample(series, moving, mapping, batch_size=1, device=device)
    batched, valid = resample(series, moving, mapping, batch_size=3, device=device)
    xr.testing.assert_identical(once, batched)
    xr.testing.assert_identical(once_valid, valid)
    xr.testing.assert_identical(series, before_image)
    if field:
        xr.testing.assert_identical(mapping, before_map)
    else:
        np.testing.assert_array_equal(mapping, before_map)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_linear_singleton_faces_remain_half_open(axis, device):
    shape = [3, 5, 7]
    shape[axis] = 1
    moving = create_voxeldata(
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 17,
        dims=("k", "j", "i"),
        voxel_to_world=np.eye(4),
    )
    for offset, supported in [
        (-0.5, True),
        (np.nextafter(0.5, -np.inf), True),
        (0.5, False),
    ]:
        mapping = np.eye(4)
        mapping[axis, 3] = offset
        actual, valid = resample(moving, moving, mapping, device=device)
        assert valid.values.all() == supported
        expected = moving.values if supported else np.full(shape, moving.values.min())
        np.testing.assert_array_equal(actual.values, expected)


def test_composition_overflow_uses_physical_mapping(device):
    moving = create_voxeldata(
        np.full((1, 1, 1), 7, dtype=np.float32),
        dims=("k", "j", "i"),
        voxel_to_world=np.diag([10.0, 10.0, 10.0, 1.0]),
    )
    actual, valid = resample(
        moving, moving, np.diag([4e307, 4e307, 4e307, 1.0]), device=device
    )
    np.testing.assert_array_equal(actual.values, moving.values)
    assert valid.values.all()


def test_nonfinite_physical_queries_still_fail(device):
    moving = create_voxeldata(
        np.ones((2, 2, 2), dtype=np.float32),
        dims=("k", "j", "i"),
        voxel_to_world=np.diag([0.5, 0.5, 0.5, 1.0]),
    )
    mapping = np.eye(4)
    mapping[0, 3] = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="nonfinite sampling coordinates"):
        resample(moving, moving, mapping, device=device)


def test_finite_outside_queries_keep_minimum_fill(device):
    moving = create_voxeldata(
        np.arange(4, dtype=np.float32).reshape(1, 2, 2) + 7,
        dims=("k", "j", "i"),
        voxel_to_world=np.eye(4),
    )
    mapping = np.eye(4)
    mapping[2, 3] = 1e308
    actual, valid = resample(moving, moving, mapping, device=device)
    assert not valid.values.any()
    np.testing.assert_array_equal(actual.values, np.full(moving.shape, 7))


def test_linear_keeps_double_coordinate_cells(device):
    values = np.zeros((1, 2, 3), dtype=np.float32)
    values[..., 2] = 1e8
    moving = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    reference = moving.isel(j=slice(0, 1), i=slice(1, 2))
    mapping = np.eye(4)
    mapping[2, 3] = 2.0**-30
    actual, valid = resample(moving, reference, mapping, device=device)
    assert valid.values.all()
    # A float32 sampler rounds this query onto the knot and incorrectly returns zero.
    np.testing.assert_array_equal(
        actual.values, np.full(reference.shape, np.float32(1e8 * 2.0**-30))
    )


def test_cpu_linear_honors_explicit_device_under_default_context():
    moving = create_voxeldata(
        np.arange(15, dtype=np.float32).reshape(1, 3, 5),
        dims=("k", "j", "i"),
        voxel_to_world=np.eye(4),
    )
    with torch.device("meta"):
        actual, valid = resample(moving, moving, np.eye(4), device="cpu")
    np.testing.assert_array_equal(actual.values, moving.values)
    assert valid.values.all()
