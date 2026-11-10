"""Registration outputs, shared status codes and native-grid result assembly."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import xarray as xr

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


@dataclass(frozen=True)
class RegistrationResult:
    """Labeled registration outputs, not a replacement spatial container.

    Attributes
    ----------
    registered : xarray.DataArray
        Original intensities resampled on the fixed grid, retaining moving dtype.
        Dask inputs retain lazy time blocks shared with maps and diagnostics.
        Integer output is rounded to nearest-even and clipped to its type range.
    valid : xarray.DataArray
        Boolean VoxelData mask of valid moving-image sample centers.
    forward_transforms : xarray.DataArray, optional
        None unless register(return_transforms=True). Linear models: homogeneous world z/y/x pull matrices. SyN: complete
        fixed-world-to-moving-world displacement in mm on the fixed grid, with
        a final world_component axis labeled z/y/x. Recordings retain time.
    diagnostics : xarray.Dataset
        Objective, update count and status, plus linear motion parameters or
        SyN regularity/inverse-consistency diagnostics. Pyramid fitting adds
        level_objective, level_status and level_n_iterations; n_iterations sums
        the actual updates across levels, while status describes the final level
        including native-grid SyN finalization checks.
        For staged calls, n_iterations sums all stages, while objective and
        model-specific variables describe the final model. stage_diagnostics
        retains the original counts and statuses of each stage separately.
        Eager attrs["device"] records the resolved CPU/CUDA choice. Lazy results
        expose execution_device, execution_task_size, execution_batch_size and
        execution_batch_start variables. Tasks follow source chunks; fitting batch
        sizes can be smaller. Timings in batch_*_seconds repeat per task frame. Compute or
        store related outputs jointly to avoid repeated fitting; one DataTree
        write preserves sharing across image, map and diagnostic groups.
    inverse_transforms : xarray.DataArray, optional
        None unless register(return_transforms=True). Linear models: inverse
        world z/y/x pull matrices. SyN: complete moving-world-to-fixed-world
        displacement in mm on the original moving grid. Sampled field inverses
        are approximate.
    stage_diagnostics : tuple of xarray.Dataset, optional
        One dataset per stage for an explicit pipeline, including each stage's
        pyramid diagnostics. None for the original single-model string API.
    """

    registered: xr.DataArray
    valid: xr.DataArray
    forward_transforms: xr.DataArray | None
    diagnostics: xr.Dataset
    inverse_transforms: xr.DataArray | None = None
    stage_diagnostics: tuple[xr.Dataset, ...] | None = None

    @property
    def transforms(self) -> xr.DataArray | None:
        """Access forward_transforms through the original attribute name.

        Returns
        -------
        xarray.DataArray or None
            The same forward pull, without copying or allocating transforms.
        """
        return self.forward_transforms


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
) -> RegistrationResult:
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
    RegistrationResult
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
    return RegistrationResult(registered, valid, forward, diagnostics, inverse)
