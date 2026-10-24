"""Independent, lazy xarray geometry for the VoxelData model."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from xarray.indexes import CoordinateTransform, CoordinateTransformIndex

VOXEL_DIMS = ("k", "j", "i")
"""Native voxel dimensions, always last in a VoxelData array."""
WORLD_DIMS = ("z", "y", "x")
"""World coordinate and affine component order."""


def _check_affine(affine: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Validate and copy a homogeneous native-to-world affine.

    Parameters
    ----------
    affine : array-like
        Homogeneous k/j/i to z/y/x affine.

    Returns
    -------
    matrix : (4, 4) numpy.ndarray
        Read-only float64 affine.

    Raises
    ------
    ValueError
        If the affine is nonfinite, singular or not homogeneous.
    """
    matrix = np.array(affine, dtype=np.float64, copy=True)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("voxel_to_world must be a finite (4, 4) affine.")
    if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError("voxel_to_world must have homogeneous last row [0, 0, 0, 1].")
    if np.linalg.matrix_rank(matrix[:3, :3]) != 3:
        raise ValueError("voxel_to_world must be nonsingular.")
    matrix.setflags(write=False)
    return matrix


class VoxelToWorldTransform(CoordinateTransform):
    """Lazily map array positions through native coordinates into world space.

    Parameters
    ----------
    coordinates : mapping
        One-dimensional native k/j/i coordinate arrays.
    voxel_to_world : array-like
        Homogeneous k/j/i to z/y/x affine.
    units : str, default: "mm"
        Shared spatial unit.
    """

    def __init__(
        self,
        coordinates: Mapping[str, npt.ArrayLike],
        voxel_to_world: npt.ArrayLike,
        units: str = "mm",
    ) -> None:
        self.coordinates = {
            dim: np.array(coordinates[dim], dtype=float, copy=True)
            for dim in VOXEL_DIMS
        }
        for coordinate in self.coordinates.values():
            coordinate.setflags(write=False)
        self.voxel_to_world = _check_affine(voxel_to_world)
        self.units = units
        super().__init__(
            WORLD_DIMS, {dim: len(self.coordinates[dim]) for dim in VOXEL_DIMS}
        )

    def forward(self, dim_positions: dict[str, Any]) -> dict[str, Any]:
        """Derive world coordinates at integer array positions.

        Parameters
        ----------
        dim_positions : dict
            Broadcastable positional k/j/i indexers.

        Returns
        -------
        coordinates : dict
            World z/y/x coordinate values.
        """
        native = np.broadcast_arrays(
            *[self.coordinates[dim][dim_positions[dim]] for dim in VOXEL_DIMS]
        )
        return {
            world: self.voxel_to_world[row, 3]
            + sum(
                self.voxel_to_world[row, column] * value
                for column, value in enumerate(native)
            )
            for row, world in enumerate(WORLD_DIMS)
        }

    def equals(self, other: CoordinateTransform, **kwargs: Any) -> bool:
        """Compare geometry without generating world-coordinate meshes.

        Parameters
        ----------
        other : xarray.indexes.CoordinateTransform
            Transform to compare.
        **kwargs : object
            Xarray alignment options; all spatial geometry is compared.

        Returns
        -------
        equal : bool
            Whether the native coordinates, affine and units agree exactly.
        """
        return (
            isinstance(other, VoxelToWorldTransform)
            and self.units == other.units
            and np.array_equal(self.voxel_to_world, other.voxel_to_world)
            and all(
                np.array_equal(self.coordinates[dim], other.coordinates[dim])
                for dim in VOXEL_DIMS
            )
        )


class VoxelToWorldIndex(CoordinateTransformIndex):
    """World-coordinate index retaining native geometry through spatial slicing.

    Parameters
    ----------
    transform : VoxelToWorldTransform
        Lazy world transform.
    """

    transform: VoxelToWorldTransform

    def create_variables(self, variables: Mapping | None = None) -> dict:
        """Create lazy world variables with spatial units.

        Parameters
        ----------
        variables : mapping, optional
            Existing coordinate variables whose attrs should be retained.

        Returns
        -------
        variables : dict
            Lazy world-coordinate variables.
        """
        result = super().create_variables(variables)
        for variable in result.values():
            variable.attrs["units"] = self.transform.units
        return result

    def isel(self, indexers: Mapping) -> VoxelToWorldIndex | None:
        """Rebuild geometry after positional selection without reindexing voxels.

        Parameters
        ----------
        indexers : mapping
            Xarray spatial indexers.

        Returns
        -------
        index : VoxelToWorldIndex, optional
            Updated index. Removing a spatial dimension leaves the VoxelData model.

        Raises
        ------
        ValueError
            If a vectorized indexer would introduce new spatial dimensions.
        """
        coordinates = {}
        for dim in VOXEL_DIMS:
            indexer = indexers.get(dim, slice(None))
            if isinstance(indexer, xr.Variable):
                if indexer.dims != (dim,):
                    raise ValueError("Vectorized spatial indexing is not supported.")
                indexer = indexer.values
            values = self.transform.coordinates[dim][indexer]
            if np.ndim(values) == 0:
                return None
            coordinates[dim] = values
        return type(self)(
            VoxelToWorldTransform(
                coordinates, self.transform.voxel_to_world, self.transform.units
            )
        )

    def sel(self, labels: dict, **kwargs: Any) -> Any:
        """Reject world-label lookup in this narrow registration prototype.

        Parameters
        ----------
        labels : dict
            Requested world labels.
        **kwargs : object
            Selection options.

        Raises
        ------
        NotImplementedError
            World-label lookup is outside the current API.
        """
        raise NotImplementedError(
            "Select native k/j/i coordinates instead of world labels."
        )


def create_voxeldata(
    values: npt.ArrayLike,
    *,
    dims: Sequence[str],
    voxel_to_world: npt.ArrayLike,
    coords: Mapping[str, npt.ArrayLike] | None = None,
    units: str = "mm",
    attrs: Mapping[str, Any] | None = None,
) -> xr.DataArray:
    """Construct a VoxelData array with lazy physical coordinates.

    Parameters
    ----------
    values : array-like
        Spatial-only or time-dependent voxel values.
    dims : sequence of str
        Exactly k/j/i, optionally preceded by time.
    voxel_to_world : (4, 4) array-like
        Native k/j/i to world z/y/x affine; not an array-position affine.
    coords : mapping, optional
        Native and time coordinates. Unspecified coordinates start at zero.
    units : str, default: "mm"
        Spatial units; this prototype accepts mm only.
    attrs : mapping, optional
        Intensity and recording metadata, not another geometry definition.

    Returns
    -------
    data : xarray.DataArray
        Data with an independent VoxelToWorldIndex on z/y/x.

    Raises
    ------
    ValueError
        If dimensions, units, geometry or coordinates are invalid.
    """
    array = np.asarray(values)
    dims = tuple(dims)
    if dims not in (VOXEL_DIMS, ("time", *VOXEL_DIMS)) or array.ndim != len(dims):
        raise ValueError("Expected dimensions (k, j, i) or (time, k, j, i).")
    if units != "mm":
        raise ValueError("Spatial units must be mm in this prototype.")
    supplied = dict(coords or {})
    if set(supplied) - set(dims):
        raise ValueError("coords must contain only data dimension names.")
    dimension_coords = {
        dim: supplied.get(dim, np.arange(size))
        for dim, size in zip(dims, array.shape, strict=True)
    }
    data = xr.DataArray(
        array, dims=dims, coords=dimension_coords, attrs=dict(attrs or {})
    )
    transform = VoxelToWorldTransform(
        {dim: data.coords[dim].values for dim in VOXEL_DIMS}, voxel_to_world, units
    )
    index = VoxelToWorldIndex(transform)
    data = data.assign_coords(xr.Coordinates.from_xindex(index))
    validate_voxeldata(data)
    return data


def validate_voxeldata(data: xr.DataArray) -> None:
    """Check the independent VoxelData structure and uniform native coordinates.

    Parameters
    ----------
    data : xarray.DataArray
        Array to validate; both intensity images and boolean masks are accepted.

    Raises
    ------
    TypeError
        If the input is not a DataArray or is not NumPy-backed.
    ValueError
        If geometry, dimensions, units or coordinates violate the model.
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
        raise ValueError("VoxelData requires one VoxelToWorldIndex on z/y/x.")
    transform = index.transform
    _check_affine(transform.voxel_to_world)
    for world in WORLD_DIMS:
        if data.coords[world].attrs.get("units") != "mm" or transform.units != "mm":
            raise ValueError("World coordinate units must be mm.")
    for dim in VOXEL_DIMS:
        if dim not in data.coords or data.coords[dim].dims != (dim,):
            raise ValueError(f"Missing one-dimensional {dim} coordinate.")
        values = data.coords[dim].values
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"{dim} coordinates must be finite numeric values.")
        if not np.array_equal(values, transform.coordinates[dim]):
            raise ValueError(
                f"{dim} coordinates disagree with the world index; rebuild VoxelData after assigning coordinates."
            )
        if len(values) > 1:
            steps = np.diff(values.astype(float))
            if steps[0] == 0 or not np.allclose(steps, steps[0], rtol=1e-7, atol=1e-10):
                raise ValueError(
                    f"{dim} coordinates must be uniformly spaced and strictly monotonic."
                )


def position_to_world(data: xr.DataArray) -> npt.NDArray[np.float64]:
    """Return the physical affine mapping zero-based array positions to world.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData with uniform native coordinates.

    Returns
    -------
    affine : (4, 4) numpy.ndarray
        Array-position k/j/i to world z/y/x map, accounting for crop and stride.
    """
    validate_voxeldata(data)
    conversion = np.eye(4)
    for axis, dim in enumerate(VOXEL_DIMS):
        values = data.coords[dim].values
        conversion[axis, 3] = values[0]
        conversion[axis, axis] = values[1] - values[0] if len(values) > 1 else 1.0
    index = cast(VoxelToWorldIndex, data.xindexes["z"])
    return index.transform.voxel_to_world @ conversion
