"""Registration trees, shared status codes and native-grid result assembly."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import xarray as xr
from voxeldata import ensure_voxeldata

STATUS_LABELS: npt.NDArray[np.str_] = np.array(
    [
        "running",
        "converged",
        "insufficient_overlap",
        "invalid_metric",
        "reference",
        "max_iterations",
        "invalid_transform",
        "skipped",
    ]
)
"""Per-frame optimizer status labels shared by linear, nonlinear and staged fits."""


def assemble_result_tree(
    registered: xr.DataArray,
    valid: xr.DataArray,
    forward: xr.DataArray | None,
    diagnostics: xr.Dataset,
    inverse: xr.DataArray | None = None,
    stages: tuple[xr.Dataset, ...] | None = None,
) -> xr.DataTree:
    """Assemble node-local registration outputs into one saveable tree.

    Parameters
    ----------
    registered : xarray.DataArray
        Original intensities resampled on the fixed grid, retaining moving dtype.
    valid : xarray.DataArray
        Boolean VoxelData support mask on the registered image grid.
    forward : xarray.DataArray, optional
        Fixed-world-to-moving-world pull matrix or displacement in mm.
    diagnostics : xarray.Dataset
        Summary fitting and execution diagnostics.
    inverse : xarray.DataArray, optional
        Moving-world-to-fixed-world pull on the original moving grid.
    stages : tuple of xarray.Dataset, optional
        Independent stage diagnostics in execution order.

    Returns
    -------
    xarray.DataTree
        Images at `images/registered` and `images/valid`, summary at
        `diagnostics`, optional maps at `forward/transform` and
        `inverse/transform`, and optional stage diagnostics at `stages/0`, etc.
        All spatial coordinates are local to leaf nodes. Lazy outputs retain
        shared task identities for a single `voxeldata.save` write.
    """
    nodes = {
        "/images": xr.Dataset({"registered": registered, "valid": valid}),
        "/diagnostics": diagnostics,
    }
    for name, transform in (("forward", forward), ("inverse", inverse)):
        if transform is not None:
            if "world_component" in transform.dims:
                transform = ensure_voxeldata(transform)
            nodes[f"/{name}"] = transform.to_dataset(name="transform")
    if stages is not None:
        nodes.update({f"/stages/{index}": data for index, data in enumerate(stages)})
    return xr.DataTree.from_dict(nodes)


@dataclass(frozen=True)
class LinearFit:
    """Internal complete linear state without resampled original intensities."""

    matrices: npt.NDArray[np.float64]
    diagnostics: xr.Dataset
    center: npt.NDArray[np.float64]


def build_output_template(
    moving: xr.DataArray, fixed: xr.DataArray, same_grid: bool
) -> xr.DataArray:
    """Preserve moving metadata and time coordinates on the fixed spatial grid.

    Parameters
    ----------
    moving : xarray.DataArray
        Original recording, including its time metadata.
    fixed : xarray.DataArray
        Native fixed grid defining output spatial coordinates.
    same_grid : bool
        Whether both inputs have the same native grid and coordinate units.

    Returns
    -------
    xarray.DataArray
        Output template with moving intensity metadata and fixed spatial geometry.
    """
    if same_grid:
        return moving
    template = fixed.drop_vars("time", errors="ignore").expand_dims(time=moving.time)
    template = template.assign_coords(
        {
            name: coord
            for name, coord in moving.coords.items()
            if coord.dims == ("time",)
        }
    )
    template.attrs = moving.attrs.copy()
    template.name = moving.name
    return template


def build_result(
    template: xr.DataArray,
    values: npt.NDArray,
    validity: npt.NDArray[np.bool_],
    forward: xr.DataArray | None,
    diagnostics: xr.Dataset,
    inverse: xr.DataArray | None,
    spatial_input: bool,
) -> xr.DataTree:
    """Package native outputs, removing synthetic time for spatial pairs.

    Parameters
    ----------
    template : xarray.DataArray
        VoxelData output template including time and native spatial coordinates.
    values : numpy.ndarray
        Original intensities resampled on the fixed grid.
    validity : numpy.ndarray
        Boolean support of each output sample.
    forward : xarray.DataArray, optional
        Complete forward world pulls.
    diagnostics : xarray.Dataset
        Per-frame fitting and output diagnostics.
    inverse : xarray.DataArray, optional
        Complete inverse world pulls.
    spatial_input : bool
        Whether the original moving input had no time dimension.

    Returns
    -------
    xarray.DataTree
        Labelled image, support, transform and diagnostic outputs.
    """
    if values.ndim == 3:
        values, validity = values[:, None], validity[:, None]
    registered, valid = template.copy(data=values), template.copy(data=validity)
    valid.attrs = {"description": "Valid moving-image sample centers"}
    if spatial_input:
        registered = registered.isel(time=0, drop=True)
        valid = valid.isel(time=0, drop=True)
        diagnostics = diagnostics.isel(time=0, drop=True)
        forward = None if forward is None else forward.isel(time=0, drop=True)
        inverse = None if inverse is None else inverse.isel(time=0, drop=True)
    return assemble_result_tree(registered, valid, forward, diagnostics, inverse)
