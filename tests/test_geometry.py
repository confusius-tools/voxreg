"""ConfUSIus interoperability and voxreg's private registration geometry guards."""

import numpy as np
import pytest
import xarray as xr
from confusius import create_voxeldata
from confusius._utils.geometry import VoxelToWorldIndex
from confusius.validation import validate_voxeldata
from numpy.testing import assert_allclose, assert_array_equal

from voxreg import position_to_world, register
from voxreg.geometry import _check_affine, validate_registration_geometry


def test_native_coordinates_and_world_index(recording_factory):
    data, _, expected = recording_factory(oblique=True, strided=True)
    assert_allclose(position_to_world(data), expected, atol=1e-12)
    p = np.array([0, 12, 21, 1])
    world = expected @ p
    for axis, name in enumerate(("z", "y", "x")):
        assert_allclose(data[name].isel(k=0, j=12, i=21).item(), world[axis])
        assert data[name].attrs["units"] == "mm"
    assert not isinstance(data.z.variable._data, np.ndarray)
    assert data.xindexes["z"] is data.xindexes["x"]


def test_slice_reverse_and_time_reduction(recording_factory):
    data, _, expected = recording_factory(oblique=True)
    selected = data.isel(j=slice(5, 45, 2), i=slice(60, 9, -3)).mean("time")
    validate_registration_geometry(selected)
    conversion = np.eye(4)
    conversion[1, 1], conversion[2, 2] = 2, -3
    conversion[:3, 3] = [0, 5, 60]
    assert_allclose(position_to_world(selected), expected @ conversion, atol=1e-12)
    assert_array_equal(selected.i, data.i[60:9:-3])
    validate_registration_geometry(data.isel(time=0, drop=True))
    vector = data.isel(i=[2, 4, 6])
    validate_registration_geometry(vector)
    assert_array_equal(vector.x.values, data.x.isel(i=[2, 4, 6]).values)


def test_confusius_input_and_output_share_voxeldata(recording_factory):
    data, _, _ = recording_factory(oblique=True, strided=True)
    external = create_voxeldata(
        data.data,
        dims=data.dims,
        **{dim: data.coords[dim] for dim in data.dims},
        voxel_to_world=data.xindexes["z"].voxel_to_world,
        attrs=data.attrs,
    )
    validate_registration_geometry(external)
    assert type(data.xindexes["z"]) is type(external.xindexes["z"]) is VoxelToWorldIndex
    assert np.shares_memory(data.data, external.data)
    assert_allclose(position_to_world(external), position_to_world(data), atol=1e-12)
    result = register(external, batch_size=2, max_iterations=2)
    for output in [result.registered, result.valid]:
        validate_voxeldata(output)
        assert isinstance(output.xindexes["z"], VoxelToWorldIndex)
        assert output.time.attrs == external.time.attrs
        assert_array_equal(
            output.fusi.affine.voxel_to_world, external.fusi.affine.voxel_to_world
        )
        for dim in external.dims:
            assert_array_equal(output[dim], external[dim])
    assert_array_equal(result.registered[0], external[0])


def test_missing_index_and_stale_coordinates_raise(recording_factory):
    data, _, _ = recording_factory()
    with pytest.raises(ValueError, match="VoxelToWorldIndex"):
        validate_registration_geometry(xr.DataArray(data.values, dims=data.dims))
    with pytest.raises(ValueError, match="disagree"):
        validate_registration_geometry(data.assign_coords(i=data.i + 1))
    with pytest.raises(ValueError, match="dimensions"):
        validate_registration_geometry(data.isel(k=0))
    labels = {
        world: data[world].isel(k=0, j=12, i=21).item() for world in ["z", "y", "x"]
    }
    assert_array_equal(data.sel(**labels, method="nearest"), data.isel(k=0, j=12, i=21))


@pytest.mark.parametrize(
    "affine",
    [np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(3), np.diag([1, 1, 1, 2])],
)
def test_invalid_affines(affine):
    with pytest.raises(ValueError, match="affine|voxel_to_world"):
        _check_affine(affine)


@pytest.mark.parametrize("coordinates", [[0, 1, 3], [0, 0, 0], [0, np.nan, 2]])
def test_invalid_native_coordinates(coordinates):
    data = create_voxeldata(
        np.ones((1, 3, 4)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    )
    with pytest.raises((TypeError, ValueError), match="coordinate"):
        validate_registration_geometry(data.assign_coords(j=coordinates))


def test_native_indices_must_be_integers(recording_factory):
    data, _, _ = recording_factory()
    with pytest.raises(TypeError, match="integer"):
        validate_registration_geometry(data.assign_coords(j=data.j.astype(float)))


def test_unsigned_descending_native_indices():
    data = create_voxeldata(
        np.ones((1, 3, 4)),
        dims=("k", "j", "i"),
        j=np.array([5, 3, 1], dtype=np.uint64),
        voxel_to_world=np.eye(4),
    )
    expected = np.eye(4)
    expected[1, 1], expected[1, 3] = -2, 5
    assert_array_equal(position_to_world(data), expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_large_native_indices_preserve_exact_uniform_steps(reverse):
    coordinates = 2**53 + np.arange(4, dtype=np.int64)
    if reverse:
        coordinates = coordinates[::-1]
    data = create_voxeldata(
        np.ones((1, 4, 4)),
        dims=("k", "j", "i"),
        j=coordinates,
        voxel_to_world=np.eye(4),
    )
    affine = position_to_world(data)
    assert affine[1, 1] == (-1 if reverse else 1)
    assert affine[1, 3] == coordinates[0]


def test_large_nonuniform_native_steps_are_rejected():
    data = create_voxeldata(
        np.ones((1, 3, 4)),
        dims=("k", "j", "i"),
        j=[0, 100_000_000, 200_000_001],
        voxel_to_world=np.eye(4),
    )
    with pytest.raises(ValueError, match="uniformly spaced"):
        position_to_world(data)


def test_native_indices_cannot_wrap_in_confusius():
    data = create_voxeldata(
        np.ones((1, 3, 4)),
        dims=("k", "j", "i"),
        k=np.array([2**63], dtype=np.uint64),
        voxel_to_world=np.eye(4),
    )
    with pytest.raises(ValueError, match="signed int64"):
        validate_registration_geometry(data)


def test_units_and_unsupported_dimensions():
    data = create_voxeldata(
        np.ones((1, 4, 5)), dims=("k", "j", "i"), voxel_to_world=np.eye(4), units="m"
    )
    validate_registration_geometry(data)
    poses = create_voxeldata(
        np.ones((2, 1, 4, 5)),
        dims=("pose", "k", "j", "i"),
        voxel_to_world=np.repeat(np.eye(4)[None], 2, axis=0),
    )
    with pytest.raises(ValueError, match="dimensions"):
        validate_registration_geometry(poses)


def test_valid_confusius_dask_data_is_not_supported(recording_factory):
    data, _, _ = recording_factory()
    lazy = data.chunk({"time": 1})
    validate_voxeldata(lazy)
    with pytest.raises(TypeError, match="NumPy-backed"):
        register(lazy)
