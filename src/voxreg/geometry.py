"""ConfUSIus VoxelData geometry with voxreg's registration restrictions."""

from typing import cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from confusius._utils.geometry import VoxelToWorldIndex

VOXEL_DIMS = ("k", "j", "i")
"""Native voxel dimensions, always last in a supported registration array."""
WORLD_DIMS = ("z", "y", "x")
"""World coordinate and affine component order."""


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
    for world in WORLD_DIMS:
        if data.coords[world].attrs.get("units") != "mm" or index.units != "mm":
            raise ValueError("World coordinate units must be mm.")
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
            steps = np.diff(values.astype(float))
            if steps[0] == 0 or not np.allclose(steps, steps[0], rtol=1e-7, atol=1e-10):
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


def position_to_world(data: xr.DataArray) -> npt.NDArray[np.float64]:
    """Return the physical affine mapping zero-based array positions to world.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData with uniform integer native coordinates.

    Returns
    -------
    affine : (4, 4) numpy.ndarray
        Array-position k/j/i to world z/y/x map, accounting for crop and stride.
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
    return index.voxel_to_world @ conversion
