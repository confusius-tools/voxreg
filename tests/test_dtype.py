"""Storage precision and final interpolation are independent of fitting tensors."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from scipy.ndimage import map_coordinates

from voxreg import position_to_world, register, rigid


@pytest.fixture
def prescribed_translation(monkeypatch):
    """Exercise output sampling with a known non-integer physical pull."""

    def optimize(
        images, fixed, plane, reference, max_iterations, learning_rate, **convergence
    ):
        lattice = torch.linalg.inv(plane.inverse_lattice)
        pixel_shift = lattice.new_tensor([0.271828, -0.314159])
        parameters = lattice.new_zeros((len(images), 3))
        parameters[:, 1:] = lattice @ pixel_shift
        parameters[reference] = 0
        return parameters, torch.where(reference, 0, 1), torch.where(reference, 4, 5)

    monkeypatch.setattr(rigid, "_optimize", optimize)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ],
)
def test_original_dtype_and_fractional_sampling(
    recording_factory, prescribed_translation, dtype, interpolation, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    data, _, _ = recording_factory(oblique=True, strided=True)
    original = data.values.astype(np.float64)
    if np.issubdtype(dtype, np.integer):
        original = np.rint(original * 20 + 10)
    else:
        original += np.linspace(0, 1e-9, original.size).reshape(original.shape)
    data = data.copy(data=original.astype(dtype))
    untouched = data.values.copy()
    result = register(data, interpolation=interpolation, device=device)
    assert result.registered.dtype == dtype
    assert_array_equal(data.values, untouched)
    assert_array_equal(result.registered[0], data[0])
    assert_array_equal(result.valid[0], True)
    assert_array_equal(result.registered.time, data.time)
    j, i = np.indices(data.shape[-2:], dtype=float)
    mapped = np.stack((j - 0.314159, i + 0.271828))
    for frame in [1, 2]:
        expected = map_coordinates(
            data.values[frame, 0].astype(np.float64),
            mapped,
            order=1 if interpolation == "linear" else 0,
            mode="nearest",
        )
        expected = np.where(
            result.valid.values[frame, 0], expected, data.values[frame].min()
        )
        if np.issubdtype(dtype, np.integer):
            expected = np.rint(expected)
        expected = expected.astype(dtype)
        tolerance = 2e-12 if dtype == np.float64 else 5e-6
        if dtype == np.float16:
            tolerance = np.finfo(dtype).eps * float(np.abs(expected).max())
        assert_allclose(
            result.registered.values[frame, 0], expected, rtol=0, atol=tolerance
        )


@pytest.mark.parametrize("scaling", ["none", "db"])
def test_float64_low_contrast_preprocessing_and_original_precision(
    recording_factory, scaling
):
    data, _, _ = recording_factory()
    data = data.copy(data=1 + data.values.astype(np.float64) * 1e-9)
    assert np.ptp(data.values.astype(np.float32)) == 0
    result = register(data, intensity_scaling=scaling, max_iterations=2)
    assert result.registered.dtype == np.float64
    assert_array_equal(result.registered[0], data[0])
    assert np.all(result.diagnostics.n_iterations[1:] == 2)
    assert np.isfinite(result.registered).all()


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.uint16])
def test_dtype_preserving_fit_recovers_known_motion(recording_factory, dtype):
    data, truth, _ = recording_factory(oblique=True, strided=True)
    values = data.values.astype(np.float64)
    if np.issubdtype(dtype, np.integer):
        values = np.rint(values * 1000)
    data = data.copy(data=values.astype(dtype))
    result = register(data, intensity_scaling="none")
    affine = position_to_world(data)
    j, i = np.indices(data.shape[-2:])
    points = affine @ np.stack((np.zeros_like(j), j, i, np.ones_like(j))).reshape(4, -1)
    delta = np.einsum("bij,jp->bip", result.transforms.values - truth, points)[:, :3]
    spacing = np.linalg.norm(affine[:3, [1, 2]], axis=0).min()
    assert np.linalg.norm(delta, axis=1).max() / spacing < 0.25
    assert result.registered.dtype == dtype
    assert_array_equal(result.registered[0], data[0])


def test_float64_outside_float32_range(recording_factory):
    data, _, _ = recording_factory()
    data = data.copy(data=data.values.astype(np.float64) * 1e100)
    result = register(data, max_iterations=1)
    assert_array_equal(result.registered[0], data[0])
    assert np.isfinite(result.registered).all()


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_unsafe_large_integers_are_rejected(recording_factory, dtype):
    data, _, _ = recording_factory()
    data = data.copy(data=np.full(data.shape, 2**53 + 1, dtype=dtype))
    with pytest.raises(ValueError, match="exact floating-point representation"):
        register(data)


def test_nearest_changes_only_output_not_fitting(recording_factory):
    data, _, _ = recording_factory()
    linear = register(data, max_iterations=25)
    nearest = register(data, interpolation="nearest", max_iterations=25)
    assert_array_equal(linear.transforms, nearest.transforms)
    assert_array_equal(linear.diagnostics.objective, nearest.diagnostics.objective)
    assert_array_equal(linear.valid, nearest.valid)
    assert not np.array_equal(linear.registered, nearest.registered)


def test_integer_rounding_ties_and_saturation(recording_factory, monkeypatch):
    data, _, _ = recording_factory()
    data = data.copy(data=np.rint(data.values * 20).astype(np.int8))
    sample = rigid.sample_rigid

    def half_values(images, parameters, plane, **kwargs):
        if kwargs:
            values = images.new_full(images.shape, 126.5)
            values[:, 0, 0] = -129.5
            return values, torch.ones_like(images, dtype=torch.bool)
        return sample(images, parameters, plane)

    monkeypatch.setattr(rigid, "sample_rigid", half_values)
    result = register(data, max_iterations=1)
    assert result.registered.values[1, 0, 0, 0] == -128
    assert result.registered.values[1, 0, 1, 1] == 126
    assert_array_equal(result.registered[0], data[0])
