"""ConfUSIus VoxelData geometry with voxreg's registration restrictions."""

from typing import cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from confusius._utils.geometry import VoxelToWorldIndex, convert_length_units

VOXEL_DIMS = ("k", "j", "i")
"""Native voxel dimensions, always last in a supported registration array."""
WORLD_DIMS = ("z", "y", "x")
"""World coordinate and affine component order."""


def _length_scale_to_mm(units):
    """Use ConfUSIus's supported physical length units, without guessing a scale."""
    return float(convert_length_units(1, units, "mm"))


def _check_affine(affine: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Check the finite, nonsingular 3D affine required by the rigid sampler."""
    matrix = np.array(affine, dtype=np.float64, copy=True)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("voxel_to_world must be a finite (4, 4) affine.")
    if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError("voxel_to_world must have homogeneous last row [0, 0, 0, 1].")
    if np.linalg.matrix_rank(matrix[:3, :3]) != 3:
        raise ValueError("voxel_to_world must be nonsingular.")
    matrix.setflags(write=False)
    return matrix


def _validate_registration_geometry(data: xr.DataArray) -> None:
    """Check the ConfUSIus geometry subset supported by registration.

    Parameters
    ----------
    data : xarray.DataArray
        NumPy-backed spatial or time-dependent VoxelData. Boolean masks are also
        accepted here; registration validates intensity types separately.

    Raises
    ------
    TypeError
        If the input is not a NumPy-backed DataArray or native indices are not integers.
    ValueError
        If geometry, dimensions, units or coordinates are unsupported.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("Expected an xarray.DataArray.")
    if data.dims not in (VOXEL_DIMS, ("time", *VOXEL_DIMS)):
        raise ValueError(
            "Expected dimensions (k, j, i) or (time, k, j, i); pose is not supported."
        )
    if any(size == 0 for size in data.shape):
        raise ValueError("VoxelData dimensions must be nonempty.")
    if not isinstance(data.data, np.ndarray):
        raise TypeError("This prototype requires NumPy-backed VoxelData.")
    index = data.xindexes.get("z")
    if not isinstance(index, VoxelToWorldIndex) or any(
        data.xindexes.get(world) is not index for world in WORLD_DIMS
    ):
        raise ValueError("VoxelData requires ConfUSIus's VoxelToWorldIndex on z/y/x.")
    matrix = _check_affine(index.voxel_to_world)
    _length_scale_to_mm(index.units)
    for world in WORLD_DIMS:
        if data.coords[world].attrs.get("units") != index.units:
            raise ValueError(
                "World coordinate units must match VoxelToWorldIndex units."
            )
    coordinates = {}
    for dim in VOXEL_DIMS:
        if dim not in data.coords or data.coords[dim].dims != (dim,):
            raise ValueError(f"Missing one-dimensional {dim} coordinate.")
        values = data.coords[dim].values
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"{dim} coordinates must be integer native voxel indices.")
        if int(values.min()) < -(2**63) or int(values.max()) > 2**63 - 1:
            raise ValueError(
                f"{dim} coordinates must fit ConfUSIus's signed int64 indices."
            )
        if len(values) > 1:
            # Integer differences preserve unit steps above float64's exact range.
            steps = np.diff(values.astype(np.int64))
            increasing = values[1] > values[0]
            if steps[0] == 0 or not (
                np.all(steps == steps[0])
                and np.all((values[1:] > values[:-1]) == increasing)
            ):
                raise ValueError(
                    f"{dim} coordinates must be uniformly spaced and strictly monotonic."
                )
        coordinates[dim] = values
    # Detect coordinate reassignment without inspecting ConfUSIus's transform internals.
    expected = VoxelToWorldIndex.from_affine(coordinates, matrix, units=index.units)
    if not index.equals(expected):
        raise ValueError(
            "Native coordinates disagree with the world index; rebuild VoxelData "
            "after assigning coordinates."
        )


def _compute_center_positions(
    data: xr.DataArray, mode: str, name: str
) -> npt.NDArray[np.float64]:
    """Return geometric or original-intensity centroids in array-position k/j/i."""
    midpoint = (np.array([data.sizes[d] for d in VOXEL_DIMS]) - 1) / 2
    images = data.values if "time" in data.dims else data.values[None]
    if mode == "center_geometry":
        return np.broadcast_to(midpoint, (len(images), 3))
    centers = []
    for frame, image in enumerate(images):
        values = np.asarray(image, dtype=np.float64)
        peak = values.max()
        if values.min() < 0 or peak <= 0:
            raise ValueError(
                f"{name} frame {frame}: center_moments requires original nonnegative "
                "intensities with positive mass. Pass unscaled data and use "
                "intensity_scaling for fitting, or choose center_geometry."
            )
        # Positive rescaling preserves moments and bounds the float64 mass sum.
        weights = values / peak
        mass = weights.sum()
        offsets = []
        for axis, size in enumerate(image.shape):
            marginal = weights.sum(axis=tuple(d for d in range(3) if d != axis))
            offsets.append(np.dot(marginal, np.arange(size) - midpoint[axis]) / mass)
        centers.append(midpoint + offsets)
    return np.asarray(centers)


def _compute_plane_alignment(
    fixed: xr.DataArray, moving: xr.DataArray, mapped_normal=None
):
    """Align normals (unoriented unless seeded), with normal-only center shift."""
    affines = [position_to_world(data) for data in (fixed, moving)]
    normals, centers = [], []
    for data, affine in zip((fixed, moving), affines, strict=True):
        normal = np.cross(affine[:3, 2], affine[:3, 1])
        normals.append(normal / np.linalg.norm(normal))
        half = (np.array([data.sizes[d] for d in VOXEL_DIMS]) - 1) / 2
        centers.append(affine[:3, 3] + affine[:3, :3] @ half)
    first, second = normals
    if (first if mapped_normal is None else mapped_normal) @ second < 0:
        second = -second
    axis = np.cross(first, second)
    sine, cosine = np.linalg.norm(axis), np.clip(first @ second, -1, 1)
    if sine > 1e-12:
        # Unit-axis Rodrigues avoids division by 1+cos near opposite normals.
        z, y, x = axis / sine
        skew = np.array([[0, -x, y], [x, 0, -z], [-y, z, 0]])
        rotation = np.eye(3) + sine * skew + (1 - cosine) * (skew @ skew)
    elif cosine < 0:
        axis = affines[0][:3, 2] / np.linalg.norm(affines[0][:3, 2])
        rotation = 2 * np.outer(axis, axis) - np.eye(3)
    else:
        rotation = np.eye(3)
    return rotation, second * (second @ (centers[1] - centers[0]))


def position_to_world(
    data: xr.DataArray, *, units: str = "mm"
) -> npt.NDArray[np.float64]:
    """Return the array-position to world affine in explicit physical units.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData with uniform integer native coordinates.
    units : str, default: "mm"
        Output length unit supported by ConfUSIus: m, cm, mm, um/µm or nm.
        The default normalizes geometry for voxreg's mm transforms regardless
        of the input index unit. Native coordinates and metadata are not modified.

    Returns
    -------
    (4, 4) numpy.ndarray
        Array-position k/j/i to world z/y/x map in units, accounting for crop
        and stride. Both lattice coefficients and world origins are converted.
    """
    _validate_registration_geometry(data)
    conversion = np.eye(4)
    for axis, dim in enumerate(VOXEL_DIMS):
        values = data.coords[dim].values
        conversion[axis, 3] = values[0]
        conversion[axis, axis] = (
            int(values[1]) - int(values[0]) if len(values) > 1 else 1.0
        )
    index = cast(VoxelToWorldIndex, data.xindexes["z"])
    scale = float(convert_length_units(1, index.units, units))
    matrix = index.voxel_to_world @ conversion
    if scale != 1:
        with np.errstate(over="ignore", invalid="ignore"):
            matrix[:3] *= scale
        _check_affine(matrix)
    return matrix


def _same_physical_grid(first, second, *, affines=None):
    """Compare array-position lattices in mm, allowing only unit-conversion roundoff."""
    if any(first.sizes[d] != second.sizes[d] for d in VOXEL_DIMS):
        return False
    a, b = (
        (position_to_world(first), position_to_world(second))
        if affines is None
        else affines
    )
    ell = min(np.linalg.norm(matrix[:3, :3], axis=0).min() for matrix in (a, b))
    tolerance = np.minimum(
        32 * np.finfo(float).eps * np.maximum(np.abs(a[:3]), np.abs(b[:3]))
        + 32 * np.finfo(float).eps * ell,
        1e-7 * ell,
    )
    return bool((np.abs(a[:3] - b[:3]) <= tolerance).all())
