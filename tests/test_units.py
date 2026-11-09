"""Length-unit invariance on native image/mask grids and physical transforms."""

import numpy as np
import pytest
import xarray as xr
from confusius._utils.geometry import attach_voxel_to_world_index
from numpy.testing import assert_allclose, assert_array_equal
from oracles import max_pull_difference, sample_mixed
from registration_options import RegistrationOptions

from voxreg import position_to_world, register, resample

SCALES = {"mm": 1, "m": 1000, "cm": 10, "um": 1e-3, "µm": 1e-3, "nm": 1e-6}


def _in_units(data, units):
    affine = data.xindexes["z"].voxel_to_world.copy()
    affine[:3] /= SCALES[units]
    return attach_voxel_to_world_index(data, affine, units=units)


@pytest.mark.parametrize("unit", SCALES)
def test_geometry_units_with_crop_stride_and_obliquity(recording_factory, unit):
    original, _, _ = recording_factory(oblique=True, strided=True)
    saved = original.copy(deep=True)
    data = _in_units(original, unit).isel(j=slice(2, 30, 2), i=slice(40, 3, -3))
    expected = position_to_world(original.isel(j=slice(2, 30, 2), i=slice(40, 3, -3)))
    assert_allclose(position_to_world(data), expected, rtol=0, atol=1e-12)
    native = expected.copy()
    native[:3] /= SCALES[unit]
    assert_allclose(position_to_world(data, units=unit), native, rtol=1e-14, atol=1e-12)
    assert_array_equal(position_to_world(data)[3], [0, 0, 0, 1])
    assert all(data.coords[axis].attrs["units"] == unit for axis in ("z", "y", "x"))
    xr.testing.assert_identical(original, saved)


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize(
    "model", ["translation", "rigid", "affine", "syn", ("rigid", "affine", "syn")]
)
def test_registration_mixed_units_masks_and_pyramids(
    linear_pair_factory, planar, model
):
    fixed, moving, _ = linear_pair_factory(planar, "rigid")
    fixed_mask = fixed.copy(data=np.ones(fixed.shape, dtype=bool))
    fixed_mask.values[..., :3] = False
    moving_mask = moving.copy(data=np.ones(moving.shape, dtype=bool))
    options: RegistrationOptions = {
        "transform": model,
        "max_iterations": 2,
        "shrink_factors": (2, 1),
        "smoothing_sigmas": (0.18, 0),
        "return_transforms": True,
        "progress": False,
    }
    baseline = register(
        moving, fixed, fixed_mask=fixed_mask, moving_mask=moving_mask, **options
    )
    native_fixed, native_moving = _in_units(fixed, "m"), _in_units(moving, "um")
    result = register(
        native_moving,
        native_fixed,
        fixed_mask=_in_units(fixed_mask, "cm"),
        moving_mask=_in_units(moving_mask, "nm"),
        **options,
    )
    assert (
        baseline.forward_transforms is not None
        and result.forward_transforms is not None
    )
    assert (
        max_pull_difference(
            result.forward_transforms, baseline.forward_transforms, fixed
        )
        < 0.001
    )
    difference = np.linalg.norm(result.registered.values - baseline.registered.values)
    assert difference / np.linalg.norm(baseline.registered.values) < 1e-5
    assert_array_equal(result.diagnostics.status, baseline.diagnostics.status)
    assert_array_equal(
        result.diagnostics.n_iterations, baseline.diagnostics.n_iterations
    )
    assert_array_equal(result.diagnostics.smoothing_sigma, [0.18, 0])
    assert result.diagnostics.smoothing_sigma.attrs["units"] == "mm"
    for output in (result.registered, result.valid):
        assert output.coords["z"].attrs["units"] == "m"
        assert output.xindexes["z"].equals(native_fixed.xindexes["z"])
    assert result.forward_transforms.attrs["spatial_units"] == "mm"
    sampled, valid = resample(native_moving, native_fixed, result.forward_transforms)
    assert_allclose(result.registered, sampled, rtol=0, atol=2e-6)
    assert_array_equal(result.valid, valid)


@pytest.mark.parametrize("unit", SCALES)
@pytest.mark.parametrize("planar", [True, False])
def test_labelled_matrix_units_seed_and_resampling(linear_pair_factory, planar, unit):
    fixed, moving, truth = linear_pair_factory(planar, "affine")
    values = truth.copy()
    values[:3, 3] /= SCALES[unit]
    matrix = xr.DataArray(
        values,
        dims=("world_out", "world_in"),
        coords={name: ["z", "y", "x", "1"] for name in ("world_out", "world_in")},
        attrs={"spatial_units": unit},
    )
    saved = matrix.copy(deep=True)
    output, valid = resample(_in_units(moving, "cm"), _in_units(fixed, "nm"), matrix)
    expected, supported = sample_mixed(moving, fixed, truth)
    expected = np.where(supported, expected, moving.values.min())
    assert_allclose(output, expected, rtol=0, atol=2e-6)
    assert_array_equal(valid, supported)
    options: RegistrationOptions = {
        "transform": "affine",
        "max_iterations": 1,
        "progress": False,
        "return_transforms": True,
    }
    baseline = register(moving, fixed, initial_transform=truth, **options)
    for seed in ("initial_transform", "initialization"):
        seed_options: RegistrationOptions = {seed: matrix}
        result = register(
            _in_units(moving, "m"), _in_units(fixed, "µm"), **seed_options, **options
        )
        assert (
            baseline.forward_transforms is not None
            and result.forward_transforms is not None
        )
        assert (
            max_pull_difference(
                result.forward_transforms, baseline.forward_transforms, fixed
            )
            < 0.001
        )
    xr.testing.assert_identical(matrix, saved)


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("component_unit", ["m", "µm", "nm"])
def test_fields_separate_grid_units_and_displacement_units(
    linear_pair_factory, planar, component_unit
):
    fixed, moving, truth = linear_pair_factory(planar, "translation")
    field = xr.concat(
        [fixed.copy(data=np.full(fixed.shape, value)) for value in truth[:3, 3]],
        dim=xr.IndexVariable("world_component", ["z", "y", "x"]),
    ).transpose("k", "j", "i", "world_component")
    field = _in_units(field, "um")
    field = field.copy(data=field.values / SCALES[component_unit])
    field.attrs = {
        "representation": "displacement",
        "spatial_units": component_unit,
        "units": component_unit,
    }
    saved = field.copy(deep=True)
    expected, supported = resample(moving, fixed, truth)
    actual, valid = resample(_in_units(moving, "m"), _in_units(fixed, "cm"), field)
    assert_allclose(actual, expected, rtol=0, atol=2e-6)
    assert_array_equal(valid, supported)
    xr.testing.assert_identical(field, saved)
    shifted = field.assign_coords(i=field.i + 1)
    affine = shifted.xindexes["z"].voxel_to_world.copy()
    shifted = attach_voxel_to_world_index(shifted, affine, units="um")
    with pytest.raises(ValueError, match="reference grid"):
        resample(moving, fixed, shifted)


@pytest.mark.parametrize("units", ["s", "voxels", "unknown"])
def test_unknown_or_non_length_units_rejected(linear_pair_factory, units):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    invalid = attach_voxel_to_world_index(
        moving, moving.xindexes["z"].voxel_to_world, units=units
    )
    with pytest.raises(ValueError, match="length unit"):
        register(invalid, fixed, progress=False)
    with pytest.raises(ValueError, match="length unit"):
        position_to_world(fixed, units=units)


def test_same_numeric_affine_with_different_units_is_not_same_grid(linear_pair_factory):
    fixed, moving, _ = linear_pair_factory(True, "rigid")
    # Relabel without rescaling: this is a physically different slice placement.
    invalid = attach_voxel_to_world_index(
        moving, moving.xindexes["z"].voxel_to_world, units="m"
    )
    with pytest.raises(ValueError, match="Noncoplanar"):
        register(invalid, fixed, progress=False)


@pytest.mark.parametrize("planar", [True, False])
def test_unit_conversion_preserves_large_origin_fit(linear_pair_factory, planar):
    fixed, moving, _ = linear_pair_factory(planar, "rigid")
    relocated = []
    for data in (fixed, moving):
        affine = data.xindexes["z"].voxel_to_world.copy()
        affine[:3, 3] += [1e8, -2e8, 3e8]
        relocated.append(attach_voxel_to_world_index(data, affine, units="mm"))
    fixed, moving = relocated
    options: RegistrationOptions = {
        "max_iterations": 3,
        "return_transforms": True,
        "progress": False,
    }
    baseline = register(moving, fixed, **options)
    native = register(_in_units(moving, "µm"), _in_units(fixed, "m"), **options)
    assert (
        baseline.forward_transforms is not None
        and native.forward_transforms is not None
    )
    assert (
        max_pull_difference(
            native.forward_transforms, baseline.forward_transforms, fixed
        )
        < 0.001
    )


def test_world_coordinate_unit_metadata_must_agree(linear_pair_factory):
    fixed, _, _ = linear_pair_factory(True, "rigid")
    fixed = fixed.copy(deep=True)
    fixed.coords["x"].attrs["units"] = "m"
    with pytest.raises(ValueError, match="match VoxelToWorldIndex"):
        position_to_world(fixed)
