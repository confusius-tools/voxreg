"""Recover known motions through the public registration API."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal

from voxreg import (
    create_voxeldata,
    position_to_world,
    register_rigid,
    validate_voxeldata,
)


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
@pytest.mark.parametrize("scaling", ["none", "db"])
def test_known_rigid_recovery(recording_factory, oblique, strided, scaling):
    data, truth, affine = recording_factory(oblique=oblique, strided=strided)
    result = register_rigid(data, intensity_scaling=scaling)
    validate_voxeldata(result.registered)
    validate_voxeldata(result.valid)
    j, i = np.mgrid[6:58:3, 6:74:3]
    positions = np.stack((np.zeros_like(j), j, i, np.ones_like(j))).reshape(4, -1)
    world = affine @ positions
    error = np.einsum("bij,jp->bip", result.transforms.values - truth, world)[:, :3]
    rms = np.sqrt(np.mean(np.sum(error**2, axis=1), axis=1))
    ell = np.linalg.norm(affine[:3, [2, 1]], axis=0).min()
    assert np.max(rms) < 0.25 * ell, (rms / ell, result.diagnostics)
    assert_array_equal(result.registered[0], data[0])
    assert_array_equal(result.registered.time, data.time)
    assert_array_equal(result.diagnostics.n_iterations[0], 0)
    assert result.diagnostics.status[0] == "reference"
    assert result.registered.attrs["units"] == "power"
    assert result.valid.dtype == bool
    for frame in range(1, 3):
        valid = result.valid.values[frame, 0]
        before = np.corrcoef(data.values[0, 0][valid], data.values[frame, 0][valid])[
            0, 1
        ]
        after = np.corrcoef(
            data.values[0, 0][valid], result.registered.values[frame, 0][valid]
        )[0, 1]
        assert after > before and after > 0.998
        assert_array_equal(
            result.registered.values[frame, 0][~valid], data.values[frame].min()
        )


def test_batch_composition_and_failed_frame_independence(recording_factory):
    data, _, _ = recording_factory()
    combined = register_rigid(
        data, intensity_scaling="none", batch_size=3, max_iterations=140
    )
    serial = register_rigid(
        data, intensity_scaling="none", batch_size=1, max_iterations=140
    )
    assert_mapping_close(data, combined.transforms, serial.transforms, 1e-5)
    assert_allclose(combined.registered, serial.registered, atol=2e-4)
    failed = data.copy(data=data.values.copy())
    failed.values[1] = 0
    with_bad = register_rigid(failed, intensity_scaling="none", max_iterations=140)
    assert with_bad.diagnostics.status[1] == "invalid_metric"
    assert with_bad.diagnostics.n_iterations[1] == 0
    assert np.isnan(with_bad.diagnostics.objective[1])
    assert_array_equal(with_bad.transforms[1], np.eye(4))
    assert_allclose(
        with_bad.diagnostics.angle[2], combined.diagnostics.angle[2], atol=1e-4
    )
    assert_allclose(
        with_bad.diagnostics.translation_i[2],
        combined.diagnostics.translation_i[2],
        atol=3e-4,
    )
    assert_allclose(
        with_bad.diagnostics.translation_j[2],
        combined.diagnostics.translation_j[2],
        atol=3e-4,
    )
    long = register_rigid(data, intensity_scaling="none", max_iterations=400)
    assert np.all(long.diagnostics.status[1:].values == "converged")
    # Continuing the chunk cannot move frames whose own optimizer has stopped.
    longer = register_rigid(data, intensity_scaling="none", max_iterations=450)
    assert_array_equal(long.transforms, longer.transforms)
    assert_array_equal(long.diagnostics.n_iterations, longer.diagnostics.n_iterations)


def test_explicit_fixed_and_low_overlap(recording_factory):
    data, _, _ = recording_factory()
    fixed = data.isel(time=0, drop=True)
    explicit = register_rigid(data, fixed=fixed, intensity_scaling="none")
    assert explicit.diagnostics.status[0] != "reference"
    assert_allclose(explicit.transforms[0], np.eye(4), atol=1e-3)
    failed = register_rigid(
        data, intensity_scaling="none", learning_rate=200, max_iterations=3
    )
    assert np.all(failed.diagnostics.status[1:].values == "insufficient_overlap")
    assert np.all(failed.diagnostics.valid_fraction.values >= 0.9)
    assert_allclose(
        failed.transforms, np.broadcast_to(np.eye(4), (3, 4, 4)), atol=1e-12
    )


@pytest.mark.parametrize(
    "option",
    [
        {"batch_size": 0},
        {"max_iterations": 0},
        {"learning_rate": -1},
        {"learning_rate": np.nan},
        {"device": "invalid"},
        {"intensity_scaling": "invalid"},
    ],
)
def test_invalid_options(recording_factory, option):
    data, _, _ = recording_factory()
    with pytest.raises(ValueError):
        register_rigid(data, **option)


@pytest.mark.parametrize(
    "bad,match", [(np.nan, "finite"), (np.inf, "finite"), (-1, "nonnegative")]
)
def test_invalid_intensities(recording_factory, bad, match):
    data, _, _ = recording_factory()
    data.values[2, 0, 2, 3] = bad
    with pytest.raises(ValueError, match=match):
        register_rigid(data)


def test_invalid_geometry_constants_and_dtypes(recording_factory):
    data, _, _ = recording_factory()
    with pytest.raises(ValueError, match="identical"):
        register_rigid(data, fixed=data.isel(time=0, j=slice(1, None), drop=True))
    with pytest.raises(ValueError, match="constant"):
        register_rigid(data.copy(data=np.ones_like(data.values)))
    with pytest.raises(ValueError, match="all-zero"):
        register_rigid(data.copy(data=np.zeros_like(data.values)))
    with pytest.raises(ValueError, match="all-zero"):
        register_rigid(
            data.copy(data=np.zeros_like(data.values)),
            fixed=data.isel(time=0, drop=True),
        )
    with pytest.raises(TypeError, match="floating"):
        register_rigid(data.astype(np.int32))
    with pytest.raises(ValueError, match="time"):
        register_rigid(data.isel(time=0, drop=True))
    volume = create_voxeldata(
        np.ones((2, 2, 4, 5), dtype=np.float32),
        dims=("time", "k", "j", "i"),
        voxel_to_world=np.eye(4),
    )
    with pytest.raises(ValueError, match="k=1"):
        register_rigid(volume)


def test_nonzero_constant_frame_is_not_optimized(recording_factory):
    data, _, _ = recording_factory()
    data.values[1] = 1
    result = register_rigid(data, intensity_scaling="none")
    assert result.diagnostics.status[1] == "invalid_metric"
    assert result.diagnostics.n_iterations[1] == 0
    assert np.isnan(result.diagnostics.objective[1])
    assert_array_equal(result.transforms[1], np.eye(4))


def test_empty_and_invalid_time_labels(recording_factory):
    with pytest.raises(ValueError, match="nonempty"):
        create_voxeldata(
            np.empty((0, 1, 4, 5)),
            dims=("time", "k", "j", "i"),
            voxel_to_world=np.eye(4),
        )
    data, _, _ = recording_factory()
    for invalid in (data.drop_vars("time"), data.assign_coords(time=[1, 1, 2])):
        with pytest.raises(ValueError, match="time labels"):
            register_rigid(invalid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("batch_size", [1, 3])
def test_cuda_matches_cpu(recording_factory, batch_size):
    data, _, _ = recording_factory(oblique=True, strided=True)
    cpu = register_rigid(data, intensity_scaling="none")
    cuda = register_rigid(
        data, intensity_scaling="none", device="cuda", batch_size=batch_size
    )
    assert_mapping_close(data, cuda.transforms, cpu.transforms, 1e-3)
    common = cuda.valid.values & cpu.valid.values
    assert_allclose(
        cuda.registered.values[common], cpu.registered.values[common], atol=5e-4
    )
    assert np.mean(cuda.valid.values != cpu.valid.values) < 1e-3
