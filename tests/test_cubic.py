"""Public final-image cubic interpolation against independent SciPy/ITK pulls."""

import numpy as np
import pytest
import xarray as xr
from confusius import create_voxeldata
from confusius._utils.geometry import attach_voxel_to_world_index
from numpy.testing import assert_allclose, assert_array_equal
from oracles import max_pull_difference
from registration_options import RegistrationOptions
from scipy.ndimage import map_coordinates

from voxreg import position_to_world, register, resample


def _coordinates(moving, reference, matrix):
    points = position_to_world(reference) @ np.vstack(
        (np.indices(reference.shape).reshape(3, -1), np.ones((1, reference.size)))
    )
    return (np.linalg.inv(position_to_world(moving)) @ matrix @ points)[:3].reshape(
        3, *reference.shape
    )


def _oracle(moving, reference, matrix, valid):
    coordinates = _coordinates(moving, reference, matrix)
    sampled = map_coordinates(
        moving.values.astype(np.float64),
        coordinates,
        order=3,
        mode="mirror",
        prefilter=True,
    )
    sampled = np.where(valid, sampled, moving.values.min())
    if np.issubdtype(moving.dtype, np.integer):
        limits = np.iinfo(moving.dtype)
        sampled = np.clip(np.rint(sampled), limits.min, limits.max)
    return sampled.astype(moving.dtype)


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.uint16])
def test_cubic_unequal_oblique_grids_match_scipy(
    linear_pair_factory, device, planar, dtype
):
    fixed, moving, truth = linear_pair_factory(planar, "affine")
    moving = moving.copy(data=(moving.values * 1000).astype(dtype))
    original = moving.copy(deep=True)
    actual, valid = resample(
        moving, fixed, truth, interpolation="bspline", device=device
    )
    expected = _oracle(moving, fixed, truth, valid.values)
    assert_allclose(
        actual,
        expected,
        rtol=2e-7 if dtype == np.float32 else 0,
        atol=2e-4 if dtype == np.float32 else 1e-9,
    )
    _, linear_valid = resample(moving, fixed, truth, device=device)
    assert_array_equal(valid, linear_valid)
    assert actual.dtype == dtype
    assert actual.xindexes["z"].equals(fixed.xindexes["z"])
    xr.testing.assert_identical(moving, original)


@pytest.mark.parametrize("planar", [True, False])
def test_cubic_matches_native_itk(linear_pair_factory, planar):
    import SimpleITK as sitk

    fixed, moving, truth = linear_pair_factory(planar, "affine")
    actual, valid = resample(moving, fixed, truth, interpolation="bspline")
    source = moving.values[0] if planar else moving.values
    image = sitk.GetImageFromArray(source.astype(np.float64))
    coordinates = _coordinates(moving, fixed, truth).reshape(3, -1)
    selected = np.flatnonzero(valid.values)[:: max(1, int(valid.values.sum()) // 24)]
    expected = [
        image.EvaluateAtContinuousIndex(
            tuple(coordinates[1 if planar else 0 :, index][::-1]), sitk.sitkBSpline
        )
        for index in selected
    ]
    assert_allclose(actual.values.reshape(-1)[selected], expected, rtol=0, atol=1e-8)


@pytest.mark.parametrize("shape", [(1, 2, 2), (2, 2, 2), (1, 3, 4), (3, 4, 5)])
def test_cubic_short_axes_mirror_boundaries_and_identity(shape, device):
    values = np.random.default_rng(852).normal(size=shape)
    moving = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    identity, supported = resample(
        moving, moving, np.eye(4), interpolation="bspline", device=device
    )
    assert_array_equal(identity, moving)
    assert supported.all()
    matrix = np.eye(4)
    matrix[:3, 3] = [0 if shape[0] == 1 else 0.21, -0.17, 0.31]
    actual, valid = resample(
        moving, moving, matrix, interpolation="bspline", device=device
    )
    assert_allclose(
        actual, _oracle(moving, moving, matrix, valid.values), rtol=0, atol=1e-12
    )
    assert not valid.all()
    assert_array_equal(actual.values[~valid.values], values.min())


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_cubic_large_offset_and_integer_knots(dtype, device):
    offset = 2**50 if dtype != np.float32 else 2**20
    values = (offset + np.arange(24).reshape(1, 4, 6)).astype(dtype)
    moving = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    actual, valid = resample(
        moving, moving, np.eye(4), interpolation="bspline", device=device
    )
    assert_array_equal(actual, values)
    assert valid.all()
    matrix = np.eye(4)
    matrix[2, 3] = 0.375
    sampled, valid = resample(
        moving, moving, matrix, interpolation="bspline", device=device
    )
    # Evaluate the small contrast separately to avoid an ill-conditioned oracle.
    centered = moving.copy(data=(values.astype(float) - offset))
    expected = _oracle(centered, moving, matrix, valid.values) + offset
    expected = np.where(valid.values, expected, values.min())
    if dtype == np.int64:
        expected = np.rint(expected)
    assert_allclose(
        sampled, expected.astype(dtype), rtol=0, atol=0.25 if dtype != np.float32 else 0
    )


def test_cubic_float_overshoot_and_integer_clipping():
    values = np.zeros((1, 5, 14), dtype=np.float64)
    values[..., 7:] = 1
    moving = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    matrix = np.eye(4)
    matrix[2, 3] = 0.5
    actual, valid = resample(moving, moving, matrix, interpolation="bspline")
    assert_allclose(
        actual, _oracle(moving, moving, matrix, valid.values), rtol=0, atol=1e-12
    )
    assert actual.values[valid.values].min() < 0
    assert actual.values[valid.values].max() > 1
    integer = moving.copy(data=(values * 255).astype(np.uint8))
    # Avoid an exact half-integer oracle tie, whose last bit depends on the solve.
    matrix[2, 3] = 0.37
    sampled, supported = resample(integer, integer, matrix, interpolation="bspline")
    assert_array_equal(sampled, _oracle(integer, integer, matrix, supported.values))


@pytest.mark.parametrize("upper,expected", [(1, 0), (3, 2)])
def test_cubic_exact_half_integer_rounding(upper, expected):
    values = np.broadcast_to(np.array([0, upper], dtype=np.int16), (1, 2, 2)).copy()
    moving = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    matrix = np.eye(4)
    matrix[2, 3] = 0.5
    sampled, valid = resample(moving, moving, matrix, interpolation="bspline")
    assert_array_equal(sampled.values[valid.values], expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cubic_constant_extreme_finite_input(dtype):
    values = np.full((1, 3, 4), np.finfo(dtype).max, dtype=dtype)
    moving = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    matrix = np.eye(4)
    matrix[2, 3] = 0.3
    sampled, _ = resample(moving, moving, matrix, interpolation="bspline")
    assert_array_equal(sampled, values)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_cubic_unrepresentable_overshoot_and_invalid_query_fill(dtype):
    values = np.zeros((1, 5, 14), dtype=dtype)
    values[..., 7:] = np.finfo(dtype).max
    moving = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    matrix = np.eye(4)
    matrix[2, 3] = 0.5
    with pytest.raises(ValueError, match="nonfinite.*source dtype"):
        resample(moving, moving, matrix, interpolation="bspline")
    matrix[2, 3] = 1e200
    sampled, valid = resample(moving, moving, matrix, interpolation="bspline")
    assert not valid.any()
    assert_array_equal(sampled, 0)


def test_cubic_chunked_queries_match_scipy():
    values = np.random.default_rng(317).normal(size=(1, 9, 11))
    moving = create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=np.eye(4))
    affine = np.diag([1, 8 / 512, 10 / 514, 1])
    fixed = create_voxeldata(
        np.zeros((1, 513, 515)), dims=("k", "j", "i"), voxel_to_world=affine
    )
    sampled, valid = resample(moving, fixed, np.eye(4), interpolation="bspline")
    assert valid.all()
    assert_allclose(
        sampled, _oracle(moving, fixed, np.eye(4), valid.values), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize(
    "model", ["translation", "rigid", "affine", "syn", ("rigid", "affine", "syn")]
)
def test_cubic_registration_preserves_fitting_and_matches_exported_pull(
    linear_pair_factory, device, planar, model
):
    fixed, moving, _ = linear_pair_factory(planar, "rigid")
    options: RegistrationOptions = {
        "transform": model,
        "max_iterations": 2,
        "shrink_factors": (2, 1),
        "smoothing_sigmas": (0.18, 0),
        "return_transforms": True,
        "progress": False,
        "device": device,
    }
    baseline = register(moving, fixed, **options)
    cubic = register(moving, fixed, resample_interpolation="bspline", **options)
    assert (
        baseline.forward_transforms is not None and cubic.forward_transforms is not None
    )
    assert (
        max_pull_difference(
            cubic.forward_transforms, baseline.forward_transforms, fixed
        )
        < 0.001
    )
    assert_array_equal(cubic.diagnostics.status, baseline.diagnostics.status)
    assert_array_equal(
        cubic.diagnostics.n_iterations, baseline.diagnostics.n_iterations
    )
    assert_allclose(
        cubic.diagnostics.objective, baseline.diagnostics.objective, rtol=0, atol=2e-6
    )
    actual, valid = resample(
        moving, fixed, cubic.forward_transforms, interpolation="bspline", device=device
    )
    assert_allclose(cubic.registered, actual, rtol=0, atol=2e-6)
    assert_array_equal(cubic.valid, valid)
    if cubic.stage_diagnostics is not None:
        assert baseline.stage_diagnostics is not None
        for a, b in zip(
            cubic.stage_diagnostics, baseline.stage_diagnostics, strict=True
        ):
            assert_array_equal(a.level_n_iterations, b.level_n_iterations)
            assert_array_equal(a.level_status, b.level_status)


def test_cubic_per_frame_fields_shared_images_and_batches(linear_pair_factory):
    fixed, moving, truth = linear_pair_factory(True, "rigid")
    coordinates = _coordinates(moving, fixed, truth)
    source_world = (
        np.einsum("ij,j...->i...", position_to_world(moving)[:3, :3], coordinates)
        + position_to_world(moving)[:3, 3, None, None, None]
    )
    fixed_world = (
        np.einsum(
            "ij,j...->i...", position_to_world(fixed)[:3, :3], np.indices(fixed.shape)
        )
        + position_to_world(fixed)[:3, 3, None, None, None]
    )
    field = xr.concat(
        [fixed.copy(data=component) for component in source_world - fixed_world],
        dim=xr.IndexVariable("world_component", ["z", "y", "x"]),
    ).transpose("k", "j", "i", "world_component")
    field.attrs = {"spatial_units": "mm", "representation": "displacement"}
    fields = xr.concat([field, field], dim=xr.IndexVariable("time", [3, 7]))
    expected, valid = resample(moving, fixed, truth, interpolation="bspline")
    for batch in (1, 2):
        actual, supported = resample(
            moving, fixed, fields, interpolation="bspline", batch_size=batch
        )
        for time in (3, 7):
            assert_allclose(actual.sel(time=time), expected, rtol=0, atol=1e-10)
            assert_array_equal(supported.sel(time=time), valid)
    recording = xr.concat([moving, moving], dim=fields.time)
    actual, supported = resample(recording, fixed, field, interpolation="bspline")
    assert_allclose(actual[0], expected, rtol=0, atol=1e-10)
    assert_array_equal(supported[0], valid)


@pytest.mark.parametrize("moving_slice", [True, False])
def test_cubic_mixed_slice_volume_cell_support(slice_volume_pair_factory, moving_slice):
    fixed, moving, truth = slice_volume_pair_factory(moving_slice)
    actual, valid = resample(moving, fixed, truth, interpolation="bspline")
    _, expected_valid = resample(moving, fixed, truth)
    assert_array_equal(valid, expected_valid)
    assert_allclose(
        actual, _oracle(moving, fixed, truth, valid.values), rtol=0, atol=1e-10
    )


def test_cubic_strided_reversed_grid_with_native_units(linear_pair_factory):
    fixed, moving, truth = linear_pair_factory(True, "affine")
    moving = moving.isel(j=slice(None, None, -1), i=slice(None, None, 2))
    native = moving.xindexes["z"].voxel_to_world.copy()
    native[:3] *= 1000
    native_moving = attach_voxel_to_world_index(moving, native, units="um")
    actual, valid = resample(native_moving, fixed, truth, interpolation="bspline")
    assert_allclose(
        actual, _oracle(moving, fixed, truth, valid.values), rtol=0, atol=1e-10
    )


def test_final_interpolation_validation(linear_pair_factory):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    options: RegistrationOptions = {
        "max_iterations": 1,
        "progress": False,
    }
    with pytest.raises(ValueError, match="resample_interpolation"):
        register(moving, fixed, resample_interpolation="cubic", **options)  # ty: ignore[invalid-argument-type] -- reject ambiguous kernels.
    with pytest.raises(ValueError, match="nearest"):
        resample(moving > 0, fixed, np.eye(4), interpolation="bspline")
