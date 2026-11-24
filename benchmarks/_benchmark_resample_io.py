"""NumPy-only input exchange and independent SciPy oracle for both environments."""

import hashlib
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates

LIMIT = 1e-5

SAMPLE_CENTER_TOLERANCE = 1e-5
"""Native-voxel boundary tolerance used by the independent benchmark oracle."""


def sample_center_support(
    coordinates: npt.NDArray[np.float64], shape: tuple[int, ...]
) -> npt.NDArray[np.bool_]:
    """Check native sample centers on the matching grids used by the benchmarks.

    Parameters
    ----------
    coordinates : (3, N) numpy.ndarray
        Source voxel positions in k/j/i order.
    shape : tuple of int
        Spatial shape of the source image.

    Returns
    -------
    (N,) numpy.ndarray
        Boolean mask of positions supported by the source image.
    """
    support = np.ones(coordinates.shape[1], dtype=bool)
    for positions, size in zip(coordinates, shape, strict=True):
        if size == 1:
            support &= (positions >= -0.5) & (positions < 0.5)
        else:
            support &= (positions >= -SAMPLE_CENTER_TOLERANCE) & (
                positions <= size - 1 + SAMPLE_CENTER_TOLERANCE
            )
    return support


def _digest(arrays, metadata):
    digest = hashlib.sha256()
    for name in (
        "moving",
        "fixed",
        "transforms" if "transforms" in arrays else "amplitudes",
        "time",
    ):
        digest.update(str((arrays[name].shape, arrays[name].dtype.str)).encode())
        digest.update(memoryview(arrays[name]))
    digest.update(
        json.dumps(
            {k: v for k, v in metadata.items() if k != "input_sha256"}, sort_keys=True
        ).encode()
    )
    return digest.hexdigest()


def _load(directory, config):
    directory = Path(directory)
    metadata = json.loads((directory / "inputs.json").read_text())
    arrays = {
        name: np.load(directory / f"{name}.npy", allow_pickle=False)
        for name in (
            "moving",
            "fixed",
            "amplitudes" if metadata["operation"] == "registration" else "transforms",
            "time",
        )
    }
    if _digest(arrays, metadata) != metadata["input_sha256"]:
        raise AssertionError("Prepared image/map/geometry hashes differ")
    count = arrays["moving"].shape[0] if metadata["recording"] else 1
    expected = config["frames"] if config["workload"].endswith("+t") else 1
    if (
        (metadata["operation"] != config["operation"])
        or metadata["model"] != config["model"]
        or metadata["workload"] != config["workload"]
        or count != expected
        or len(arrays["time"]) != expected
        or ("amplitudes" in arrays and len(arrays["amplitudes"]) != expected)
    ):
        raise AssertionError("Prepared inputs differ from the requested full workload")
    return arrays, metadata


def _errors(output, arrays, metadata, validity=None):
    shape = arrays["fixed"].shape
    affine = np.array(metadata["affine"])
    indices = np.indices(shape).reshape(3, -1)
    points = affine[:3, :3] @ indices + affine[:3, 3, None]
    inverse = np.linalg.inv(affine[:3, :3])
    recording = metadata["recording"]
    raw = arrays["moving"] if recording else arrays["moving"][None]
    maps = arrays["transforms"] if recording else arrays["transforms"][None]
    result = output if recording else output[None]
    masks = validity if recording or validity is None else validity[None]
    if (
        output.shape != arrays["moving"].shape
        or output.dtype != arrays["moving"].dtype
        or not np.isfinite(output).all()
    ):
        raise AssertionError("Invalid resampled shape, dtype or intensities")
    errors = []
    for index, (actual, original, transform) in enumerate(
        zip(result, raw, maps, strict=True)
    ):
        mapped = (
            points + transform.reshape(-1, 3).T
            if metadata["model"] == "syn"
            else transform[:3, :3] @ points + transform[:3, 3, None]
        )
        query = inverse @ (mapped - affine[:3, 3, None])
        valid = sample_center_support(query, shape)
        if not valid.any():
            raise AssertionError("No common interpolation support")
        if masks is not None and not np.array_equal(masks[index].ravel(), valid):
            raise AssertionError(f"Physical support mask mismatch at frame {index}")
        expected = map_coordinates(
            original.astype(np.float64), query, order=1, mode="nearest", prefilter=False
        )
        delta = actual.ravel()[valid].astype(np.float64) - expected[valid]
        errors.append(float(np.sqrt(np.mean(delta**2) / np.mean(expected[valid] ** 2))))
    return errors
