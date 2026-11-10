"""Recover known motions through the public registration API."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from voxeldata import create_voxeldata, validate_voxeldata

from voxreg import position_to_world, register


def assert_mapping_close(data, actual, expected, tolerance_pixels):
    """Compare physical grid displacements, not origin-dependent matrix offsets."""
    affine = position_to_world(data)
    j, i = np.indices(data.shape[-2:])
    points = affine @ np.stack((np.zeros_like(j), j, i, np.ones_like(j))).reshape(4, -1)
    displacement = np.einsum(
        "bij,jp->bip", np.asarray(actual) - np.asarray(expected), points
    )[:, :3]
    ell = np.linalg.norm(affine[:3, [1, 2]], axis=0).min()
    assert np.linalg.norm(displacement, axis=1).max() < tolerance_pixels * ell


@pytest.mark.parametrize(
    "oblique,strided", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("scaling", [None, "log1p", "sqrt", 2.0])
def test_known_rigid_recovery(recording_factory, oblique, strided, scaling):
    data, truth, affine = recording_factory(oblique=oblique, strided=strided)
    result = register(data, return_transforms=True, moving_intensity_scaling=scaling)
    assert "forward" in result
    validate_voxeldata(result["images"].to_dataset()["registered"])
    validate_voxeldata(result["images"].to_dataset()["valid"])
    j, i = np.mgrid[6:58:3, 6:74:3]
    positions = np.stack((np.zeros_like(j), j, i, np.ones_like(j))).reshape(4, -1)
    world = affine @ positions
    error = np.einsum(
        "bij,jp->bip", result["forward"].to_dataset()["transform"].values - truth, world
    )[:, :3]
    rms = np.sqrt(np.mean(np.sum(error**2, axis=1), axis=1))
    ell = np.linalg.norm(affine[:3, [2, 1]], axis=0).min()
    assert np.max(rms) < 0.25 * ell, (rms / ell, result["diagnostics"].to_dataset())
    assert_array_equal(result["images"].to_dataset()["registered"][0], data[0])
    assert_array_equal(result["images"].to_dataset()["registered"].time, data.time)
    assert_array_equal(result["diagnostics"].to_dataset().n_iterations[0], 0)
    assert result["diagnostics"].to_dataset().status[0] == "reference"
    assert result["images"].to_dataset()["registered"].attrs["units"] == "power"
    assert result["images"].to_dataset()["valid"].dtype == bool
    for frame in range(1, 3):
        valid = result["images"].to_dataset()["valid"].values[frame, 0]
        before = np.corrcoef(data.values[0, 0][valid], data.values[frame, 0][valid])[
            0, 1
        ]
        after = np.corrcoef(
            data.values[0, 0][valid],
            result["images"].to_dataset()["registered"].values[frame, 0][valid],
        )[0, 1]
        assert after > before and after > 0.998
        assert_array_equal(
            result["images"].to_dataset()["registered"].values[frame, 0][~valid],
            data.values[frame].min(),
        )


def test_batch_composition_and_failed_frame_independence(recording_factory):
    data, _, _ = recording_factory()
    combined = register(
        data,
        return_transforms=True,
        batch_size=3,
        max_iterations=140,
    )
    serial = register(
        data,
        return_transforms=True,
        batch_size=1,
        max_iterations=140,
    )
    assert_mapping_close(
        data,
        combined["forward"].to_dataset()["transform"],
        serial["forward"].to_dataset()["transform"],
        1e-5,
    )
    assert_allclose(
        combined["images"].to_dataset()["registered"],
        serial["images"].to_dataset()["registered"],
        atol=2e-4,
    )
    failed = data.copy(data=data.values.copy())
    failed.values[1] = 0
    with_bad = register(failed, return_transforms=True, max_iterations=140)
    assert "forward" in with_bad
    assert with_bad["diagnostics"].to_dataset().status[1] == "invalid_metric"
    assert with_bad["diagnostics"].to_dataset().n_iterations[1] == 0
    assert np.isnan(with_bad["diagnostics"].to_dataset().objective[1])
    assert_array_equal(with_bad["forward"].to_dataset()["transform"][1], np.eye(4))
    assert_allclose(
        with_bad["diagnostics"].to_dataset().angle[2],
        combined["diagnostics"].to_dataset().angle[2],
        atol=1e-4,
    )
    assert_allclose(
        with_bad["diagnostics"].to_dataset().translation_i[2],
        combined["diagnostics"].to_dataset().translation_i[2],
        atol=3e-4,
    )
    assert_allclose(
        with_bad["diagnostics"].to_dataset().translation_j[2],
        combined["diagnostics"].to_dataset().translation_j[2],
        atol=3e-4,
    )
    # Exercise freezing within this budget; ITK's default 1e-8 is more conservative.
    long = register(
        data,
        return_transforms=True,
        max_iterations=400,
        convergence_minimum_value=1e-6,
    )
    assert np.all(long["diagnostics"].to_dataset().status[1:].values == "converged")
    # Continuing the chunk cannot move frames whose own optimizer has stopped.
    longer = register(
        data,
        return_transforms=True,
        max_iterations=450,
        convergence_minimum_value=1e-6,
    )
    assert_array_equal(
        long["forward"].to_dataset()["transform"],
        longer["forward"].to_dataset()["transform"],
    )
    assert_array_equal(
        long["diagnostics"].to_dataset().n_iterations,
        longer["diagnostics"].to_dataset().n_iterations,
    )


def test_explicit_fixed_and_low_overlap(recording_factory):
    data, _, _ = recording_factory()
    fixed = data.isel(time=0, drop=True)
    explicit = register(data, return_transforms=True, fixed=fixed)
    assert "forward" in explicit
    assert explicit["diagnostics"].to_dataset().status[0] != "reference"
    assert_allclose(
        explicit["forward"].to_dataset()["transform"][0], np.eye(4), atol=1e-3
    )
    failed = register(
        data,
        return_transforms=True,
        learning_rate=200,
        max_iterations=3,
    )
    assert "forward" in failed
    assert np.all(
        failed["diagnostics"].to_dataset().status[1:].values == "insufficient_overlap"
    )
    assert np.all(failed["diagnostics"].to_dataset().valid_fraction.values >= 0.9)
    assert_allclose(
        failed["forward"].to_dataset()["transform"],
        np.broadcast_to(np.eye(4), (3, 4, 4)),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "option",
    [
        {"batch_size": 0},
        {"max_iterations": 0},
        {"learning_rate": -1},
        {"learning_rate": np.nan},
        {"learning_rate": True},
        {"learning_rate": "0.1"},
        {"learning_rate": 1j},
        {"learning_rate": [0.1, 0.2]},
        {"device": "invalid"},
        {"moving_intensity_scaling": "invalid"},
        {"resample_interpolation": "cubic"},
    ],
)
def test_invalid_options(recording_factory, option):
    data, _, _ = recording_factory()
    with pytest.raises(ValueError):
        register(data, **option)


@pytest.mark.parametrize("bad,match", [(np.nan, "finite"), (np.inf, "finite")])
def test_invalid_intensities(recording_factory, bad, match):
    data, _, _ = recording_factory()
    data.values[2, 0, 2, 3] = bad
    with pytest.raises(ValueError, match=match):
        register(data)


def test_invalid_geometry_constants_and_dtypes(recording_factory):
    data, _, _ = recording_factory()
    fixed = data.isel(time=0, j=slice(1, None), drop=True)
    cropped = register(data, fixed=fixed, max_iterations=2, progress=False)
    assert cropped["images"].to_dataset()["registered"].shape[1:] == fixed.shape
    assert (
        cropped["images"]
        .to_dataset()["registered"]
        .xindexes["z"]
        .equals(fixed.xindexes["z"])
    )
    with pytest.raises(ValueError, match="constant"):
        register(data.copy(data=np.ones_like(data.values)))
    with pytest.raises(ValueError, match="all-zero"):
        register(data.copy(data=np.zeros_like(data.values)))
    with pytest.raises(ValueError, match="all-zero"):
        register(
            data.copy(data=np.zeros_like(data.values)),
            fixed=data.isel(time=0, drop=True),
        )
    with pytest.raises(TypeError, match="real integer or floating"):
        register(data.astype(bool))
    with pytest.raises(ValueError, match="time"):
        register(data.isel(time=0, drop=True))
    volume = create_voxeldata(
        np.ones((2, 2, 1, 5), dtype=np.float32),
        dims=("time", "k", "j", "i"),
        dt=1,
        voxel_to_world=np.eye(4),
    )
    with pytest.raises(ValueError, match="k=1"):
        register(volume)


def test_nonzero_constant_frame_is_not_optimized(recording_factory):
    data, _, _ = recording_factory()
    data.values[1] = 1
    result = register(data, return_transforms=True)
    assert "forward" in result
    assert result["diagnostics"].to_dataset().status[1] == "invalid_metric"
    assert result["diagnostics"].to_dataset().n_iterations[1] == 0
    assert np.isnan(result["diagnostics"].to_dataset().objective[1])
    assert_array_equal(result["forward"].to_dataset()["transform"][1], np.eye(4))


def test_empty_and_invalid_time_labels(recording_factory):
    data, _, _ = recording_factory()
    with pytest.raises(ValueError, match="zero-length"):
        register(data.isel(time=slice(0, 0)))
    for invalid in (data.drop_vars("time"), data.assign_coords(time=[1, 1, 2])):
        with pytest.raises(ValueError, match="time"):
            register(invalid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("batch_size", [1, 3])
def test_cuda_matches_cpu(recording_factory, batch_size):
    data, _, _ = recording_factory(oblique=True, strided=True)
    cpu = register(data, return_transforms=True)
    cuda = register(
        data,
        return_transforms=True,
        device="cuda",
        batch_size=batch_size,
    )
    assert_mapping_close(
        data,
        cuda["forward"].to_dataset()["transform"],
        cpu["forward"].to_dataset()["transform"],
        0.01,
    )
    # Same-coordinate interpolation is tested separately; accepted transforms differ.
    assert (
        np.mean(
            cuda["images"].to_dataset()["valid"].values
            != cpu["images"].to_dataset()["valid"].values
        )
        < 1e-3
    )
