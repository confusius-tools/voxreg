"""Apply exported world-space pulls without fitting or intensity conditioning."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr

from voxreg.geometry import (
    VOXEL_DIMS,
    length_scale_to_mm,
    position_to_world,
    same_physical_grid,
    validate_registration_geometry,
)
from voxreg.interpolation import (
    Interpolation,
    prepare_cubic_spline,
    restore_image_dtype,
    sample_cubic_spline,
    select_resampling_dtype,
)
from voxreg.preparation import choose_batch_size, resolve_device
from voxreg.results import build_output_template
from voxreg.sampling import (
    SAMPLE_CENTER_TOLERANCE,
    sample_linear_at_indices,
    sample_nearest,
    sample_normalized_grid,
)
from voxreg.transforms import validate_matrices

_LINEAR_BLOCK_POINTS = 2**17
"""Target number of CPU samples sharing temporary linear interpolation buffers."""


def _prepare_sampling_axes(
    sizes: npt.NDArray[np.intp], voxel_extent: bool, device: str
) -> torch.Tensor:
    """Prepare support bounds and grid normalization in query-axis order.

    Parameters
    ----------
    sizes : numpy.ndarray
        Source axis lengths in the order of the query components.
    voxel_extent : bool
        Whether to include the outer half voxel on every axis.
    device : str
        Device on which to prepare the constants.

    Returns
    -------
    (4, 3) torch.Tensor
        Rows give the inclusive lower and upper bounds, followed by the scale
        and shift mapping voxel positions to PyTorch's normalized grid.
    """
    singleton = sizes == 1
    extent = singleton | voxel_extent
    lower = np.where(extent, -0.5, -SAMPLE_CENTER_TOLERANCE)
    # Use the preceding float64 value to keep the upper half-voxel face open.
    upper = np.where(
        extent, np.nextafter(sizes - 0.5, -np.inf), sizes - 1 + SAMPLE_CENTER_TOLERANCE
    )
    scale = np.where(singleton, 0.0, 2 / np.maximum(sizes - 1, 1))
    shift = np.where(singleton, 0.0, -1.0)
    return torch.tensor(np.stack((lower, upper, scale, shift)), device=device)


def _sample_linear_blocks(images: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Sample CPU float64 images in bounded blocks of voxel positions.

    Parameters
    ----------
    images : torch.Tensor
        Source frames, including integer images promoted to float64.
    query : torch.Tensor
        Batched source positions in x, y, z order.

    Returns
    -------
    torch.Tensor
        Interpolated values on the query grid.
    """
    flat = query.flatten(1, -2)
    sampled = images.new_empty(flat.shape[:-1])
    block_size = max(1, _LINEAR_BLOCK_POINTS // len(images))
    for start in range(0, flat.shape[1], block_size):
        block = slice(start, start + block_size)
        positions = flat[:, block].unbind(-1)[::-1]
        bounded = tuple(
            c.clamp(0, size - 1)
            for c, size in zip(positions, images.shape[1:], strict=True)
        )
        sampled[:, block] = sample_linear_at_indices(
            images, bounded, skip_aligned_axes=True
        )
    return sampled.view(query.shape[:-1])


def _sample_linear(
    images: torch.Tensor, query: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """Sample at source voxel positions, preserving exact centers.

    Parameters
    ----------
    images : torch.Tensor
        Source images with a leading batch dimension.
    query : torch.Tensor
        Owned x/y/z positions. Frames may share a grid.
    scale, shift : torch.Tensor
        Per-axis factors for conversion to normalized coordinates.

    Returns
    -------
    torch.Tensor
        Interpolated intensities in the source tensor dtype.
    """
    if images.dtype == torch.float64 and images.device.type == "cpu":
        return _sample_linear_blocks(images, query)

    coordinates = query.unbind(-1)[::-1]
    # A shared grid needs only one copy for center checks and normalization.
    base = query[:1] if query.stride(0) == 0 else query
    centers = (base == base.trunc()).all(dim=-1)
    if centers.all():
        return sample_nearest(images, coordinates)
    if images.dtype == torch.float64:
        bounded = tuple(
            c.clamp(0, size - 1)
            for c, size in zip(coordinates, images.shape[1:], strict=True)
        )
        return sample_linear_at_indices(images, bounded)

    # Normalization can shift exact centers. Save their values before modifying base.
    exact = sample_nearest(images, coordinates) if centers.any() else None
    planar = images.shape[1] == query.shape[1] == 1
    source = images
    if planar:
        source = images[:, 0]
        base, scale, shift = base[:, 0, ..., :2], scale[:2], shift[:2]
    grid = base.mul_(scale).add_(shift)
    grid = grid.expand(len(images), *grid.shape[1:])
    sampled = sample_normalized_grid(
        source.double(),
        grid,
        "linear",
        tile_cpu_volume=images.device.type == "cpu" and not planar,
    )
    sampled = sampled.reshape(query.shape[:-1]).to(images.dtype)
    if exact is not None:
        sampled = torch.where(centers, exact, sampled)
    return sampled


def validate_resampling_metadata(
    moving: xr.DataArray,
    reference: xr.DataArray,
    transforms: npt.ArrayLike | xr.DataArray,
    interpolation: Interpolation,
    batch_size: int | None,
) -> None:
    """Validate resampling geometry, labels and options without reading voxel data.

    Parameters
    ----------
    moving, reference : xarray.DataArray
        Spatial data or recording and spatial reference geometry.
    transforms : array-like or xarray.DataArray
        Shared or per-frame matrices or labelled displacement fields.
    interpolation : {"linear", "nearest", "bspline"}
        Final image interpolation.
    batch_size : int, optional
        Positive frame batch limit.

    Returns
    -------
    None
        Valid metadata requires no conversion.

    Raises
    ------
    TypeError
        If an image or transform backend/dtype is unsupported.
    ValueError
        If geometry, labels, transform metadata or options are invalid.
    """
    validate_registration_geometry(moving)
    validate_registration_geometry(reference)
    if reference.dims != VOXEL_DIMS:
        raise ValueError("reference must be spatial VoxelData without time.")
    if interpolation not in ("linear", "nearest", "bspline"):
        raise ValueError("interpolation must be linear, nearest or bspline.")
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("batch_size must be None or a positive integer.")
    if moving.dtype.kind not in "iufb" or (
        moving.dtype.kind == "f" and moving.dtype.itemsize > 8
    ):
        raise TypeError("moving must have real integer or floating-point intensities.")
    if moving.dtype == np.bool_ and interpolation != "nearest":
        raise ValueError("Boolean images require nearest interpolation.")
    if (
        isinstance(transforms, xr.DataArray)
        and "time" in transforms.dims
        and (
            "time" not in transforms.coords
            or transforms.time.dims != ("time",)
            or not transforms.get_index("time").is_unique
        )
    ):
        raise ValueError("transforms must have unique time labels.")
    if isinstance(transforms, xr.DataArray):
        from voxreg.lazy import has_lazy_data

        if not isinstance(transforms.data, np.ndarray) and not has_lazy_data(
            transforms
        ):
            raise TypeError("transforms must be NumPy- or Dask-backed.")
        if transforms.dtype.kind not in "iuf":
            raise ValueError("transforms must contain finite real numeric values.")
        length_scale_to_mm(transforms.attrs.get("spatial_units", "mm"))
        if (
            "time" in moving.dims
            and "time" in transforms.dims
            and not moving.get_index("time").equals(transforms.get_index("time"))
        ):
            raise ValueError("moving and transforms time labels must match exactly.")
        if "world_component" in transforms.dims:
            expected = (("time",) if "time" in transforms.dims else ()) + (
                *VOXEL_DIMS,
                "world_component",
            )
            if set(transforms.dims) != set(expected) or list(
                transforms.world_component.values
            ) != ["z", "y", "x"]:
                raise ValueError(
                    "Displacements must have dimensions [time,] k/j/i/world_component with components ordered z/y/x."
                )
            scalar = transforms.isel(world_component=0, drop=True).transpose(
                *(("time",) if "time" in transforms.dims else ()), *VOXEL_DIMS
            )
            validate_registration_geometry(scalar)
            if not same_physical_grid(scalar, reference):
                raise ValueError(
                    "Displacement field must match the reference grid exactly."
                )
            if (
                transforms.attrs.get("representation") != "displacement"
                or "spatial_units" not in transforms.attrs
            ):
                raise ValueError(
                    "Fields must declare representation='displacement' and physical spatial_units (for example mm)."
                )
            if "units" in transforms.attrs and length_scale_to_mm(
                transforms.attrs["units"]
            ) != length_scale_to_mm(transforms.attrs["spatial_units"]):
                raise ValueError(
                    "Field units and spatial_units must describe the same length unit."
                )
            return
        if transforms.dims not in (
            ("world_out", "world_in"),
            ("time", "world_out", "world_in"),
        ):
            raise ValueError("Matrix dimensions must be [time,] world_out/world_in.")
        for axis in ("world_out", "world_in"):
            if axis in transforms.coords and list(transforms[axis].values) != [
                "z",
                "y",
                "x",
                "1",
            ]:
                raise ValueError("Matrix axes must be ordered z/y/x/1.")
        shape = transforms.shape
    else:
        shape = np.shape(transforms)
    if (
        len(shape) not in (2, 3)
        or shape[-2:] != (4, 4)
        or (len(shape) == 3 and shape[0] < 1)
    ):
        raise ValueError(
            "Expected nonempty transform matrices shaped (4, 4) or (N, 4, 4)."
        )
    if len(shape) == 3 and "time" in moving.dims and shape[0] != moving.sizes["time"]:
        raise ValueError("Expected one transform per moving frame.")


@torch.no_grad()
def resample(
    moving: xr.DataArray,
    reference: xr.DataArray,
    transforms: npt.ArrayLike | xr.DataArray,
    *,
    interpolation: Interpolation = "linear",
    device: Literal["auto", "cpu", "cuda"] = "auto",
    batch_size: int | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Resample original intensities using a reference-to-moving world pull.

    Parameters
    ----------
    moving : xarray.DataArray
        Finite real NumPy- or Dask-backed VoxelData, spatial or time-dependent. Boolean
        images are supported with nearest interpolation. Integers must be in
        [-2**53, 2**53], as for registration.
    reference : xarray.DataArray
        Spatial VoxelData defining the output lattice; its intensities are unused.
        Input world units may differ; geometry is converted to mm internally.
        Outputs retain this reference grid's native coordinates and length unit.
    transforms : array-like or xarray.DataArray
        World z/y/x homogeneous pull matrix (4, 4), per-frame matrices (N, 4, 4),
        or a world displacement DataArray. Bare matrices use mm. Labelled
        matrices use spatial_units (default mm); field components must declare
        spatial_units. VoxelData length units are converted to mm. Field lattice
        units are independent of component units. Fields must match the physical
        reference grid; they are not implicitly regridded. Use forward
        transforms with moving images and the fixed reference, or inverse
        transforms with fixed images and the moving reference.
        Spatial images/maps broadcast over time-dependent maps/images. When both
        have time labels they must match exactly. Anonymous per-frame matrices
        applied to a spatial image produce integer time labels.
    interpolation : {"linear", "nearest", "bspline"}, default="linear"
        Use nearest for categorical labels. Exact half-voxel positions select
        the even source array index along that axis. Outside samples use each original
        moving frame's minimum and are marked invalid. bspline uses prefiltered
        cubic B-splines with whole-sample mirror boundaries in 2D and 3D. Its
        coefficients and queries use float64 on the selected device. Floating
        overshoot is retained; integer outputs are rounded and dtype-clipped.
        No extra support erosion is applied for the cubic kernel radius.
    device : {"auto", "cpu", "cuda"}, default: "auto"
        Auto selects CUDA when available, otherwise CPU, once per call. Explicit
        CUDA requires availability. CUDA execution errors, including out-of-memory,
        propagate without a CPU retry. Dask source images/maps return lazy outputs;
        auto resolution occurs at worker execution, not graph construction.
    batch_size : int, optional
        Positive eager frame batch limit; not provided uses the registration memory
        policy, inside workers for lazy calls. Lazy task sizes follow source time
        chunks independently of this limit. Spatial blocks are assembled within
        each task; temporal chunks must fit worker RAM.

    Returns
    -------
    resampled : xarray.DataArray
        Original intensities on the reference grid, retaining moving dtype and
        time metadata. Integer interpolation is rounded to nearest-even and clipped.
    valid : xarray.DataArray
        Boolean mask of supported moving-image sample centers. Singleton axes
        have half-open voxel-extent support. Mixed slice/volume pairs use this
        support on every axis, matching registration. Other axes include samples
        up to 1e-5 voxels beyond the outermost sample centers.

    Raises
    ------
    TypeError
        If images, transforms or their numeric types are unsupported.
    ValueError
        If geometry, labels, units, transforms or sampling options are invalid.
        Also raised when cubic overshoot cannot be represented finitely in the
        moving image's floating-point dtype.
    """
    validate_resampling_metadata(
        moving, reference, transforms, interpolation, batch_size
    )
    from voxreg.lazy import has_lazy_data, resample_lazy

    if isinstance(transforms, xr.DataArray) and "world_component" in transforms.dims:
        # VoxelData stores components before spatial axes; sampling uses them last.
        transforms = transforms.transpose(
            *(("time",) if "time" in transforms.dims else ()),
            *VOXEL_DIMS,
            "world_component",
        )
    if has_lazy_data(moving, transforms):
        return resample_lazy(
            moving, reference, transforms, interpolation, device, batch_size
        )
    device = resolve_device(device)
    integer = np.issubdtype(moving.dtype, np.integer)
    if not np.isfinite(moving.values).all():
        raise ValueError("moving intensities must be finite.")
    if integer and (
        int(moving.values.min()) < -(2**53) or int(moving.values.max()) > 2**53
    ):
        raise ValueError("moving integers must lie within [-2**53, 2**53].")

    labeled = isinstance(transforms, xr.DataArray)
    field = labeled and "world_component" in transforms.dims
    values = np.asarray(transforms)
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("transforms must contain finite real numeric values.")
    if field:
        assert isinstance(transforms, xr.DataArray)
        scale = length_scale_to_mm(transforms.attrs["spatial_units"])
        if scale != 1:
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.asarray(values, dtype=np.float64) * scale
            if not np.isfinite(values).all():
                raise ValueError(
                    "Displacements must remain finite when converted to mm."
                )
        count = transforms.sizes.get("time", 1)
        shared = "time" not in transforms.dims
    else:
        shared = values.ndim == 2
        count = 1 if shared else len(values) if values.ndim == 3 else 0
        if count < 1:
            raise ValueError(
                "Expected nonempty transform matrices shaped (4, 4) or (N, 4, 4)."
            )
        values = validate_matrices(transforms, count)
        if np.any(np.linalg.matrix_rank(values[:, :3, :3]) != 3):
            raise ValueError("Transform matrices must be nonsingular.")

    spatial = "time" not in moving.dims and shared
    if "time" not in moving.dims:
        time = (
            transforms.time
            if labeled and "time" in transforms.dims
            else np.arange(count)
        )
        moving = moving.expand_dims(time=time)
        validate_registration_geometry(moving)
    frames = moving.sizes["time"]
    if not shared and count != frames:
        raise ValueError("Expected one transform per moving frame.")
    if batch_size is None:
        batch_size = choose_batch_size(
            moving, reference, "syn" if field else "affine", device
        )

    native_linear = interpolation == "linear"
    mixed = (moving.sizes["k"] == 1) != (reference.sizes["k"] == 1)
    source_affine, target_affine = (
        position_to_world(moving),
        position_to_world(reference),
    )
    same_lattice = native_linear and np.array_equal(
        source_affine[:3, :3], target_affine[:3, :3]
    )
    shape = tuple(reference.sizes[d] for d in VOXEL_DIMS)
    axes = tuple(torch.arange(n, device=device, dtype=torch.float64) for n in shape)
    local = None
    if not native_linear or (field and not same_lattice):
        local = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        local = local @ torch.tensor(target_affine[:3, :3].T, device=device)
    inverse = np.linalg.inv(source_affine[:3, :3]).T
    inverse_lattice = torch.tensor(
        inverse[:, ::-1].copy() if native_linear else inverse, device=device
    )
    sizes = np.array(moving.shape[1:])
    lower, upper, scale, shift = _prepare_sampling_axes(
        sizes[::-1] if native_linear else sizes, mixed, device
    )
    dtype = select_resampling_dtype(moving.dtype, interpolation)
    outputs, masks = [], []
    for first in range(0, frames, batch_size):
        last = min(first + batch_size, frames)
        if shared:
            selected = values if field else values[:1]
        else:
            selected = values[first:last]
        if field:
            # PyTorch cannot read a NumPy view with a reversed axis.
            if min(selected.strides) < 0:
                selected = selected.copy()
            displacement = torch.tensor(selected, device=device, dtype=torch.float64)
            offset = torch.tensor(
                target_affine[:3, 3] - source_affine[:3, 3], device=device
            )
            if same_lattice:
                # Add exact voxel indices without building another volume-sized grid.
                query = (displacement + offset) @ inverse_lattice
                k, j, i = axes
                query[..., 0].add_(i)
                query[..., 1].add_(j[:, None])
                query[..., 2].add_(k[:, None, None])
            else:
                assert local is not None
                query = (local + displacement + offset) @ inverse_lattice
            if shared:
                query = query.unsqueeze(0).expand(last - first, *shape, 3)
        else:
            matrices = torch.tensor(selected, device=device)
            # Cancel the largest opposing terms first so a small shift or origin survives.
            low, middle, high = np.sort(
                np.broadcast_arrays(
                    selected[:, :3, :3] @ target_affine[:3, 3],
                    selected[:, :3, 3],
                    -source_affine[:3, 3],
                ),
                axis=0,
            )
            offset = (low + high) + middle
            linear = None
            if native_linear:
                linear = inverse_lattice.T @ (
                    matrices[:, :3, :3]
                    @ torch.tensor(target_affine[:3, :3], device=device)
                )
                if same_lattice:
                    translations = (selected[:, :3, :3] == np.eye(3)).all(axis=(1, 2))
                    if translations.any():
                        linear[translations] = torch.eye(
                            3, dtype=linear.dtype, device=device
                        ).flip(0)
            if linear is not None and torch.isfinite(linear).all():
                mapped_offset = torch.tensor(offset, device=device) @ inverse_lattice
                # Broadcast one-dimensional axes instead of materializing an input grid.
                k, j, i = axes
                query = (
                    (
                        linear[:, :, 0, None, None, None] * k[:, None, None]
                        + linear[:, :, 1, None, None, None] * j[:, None]
                        + linear[:, :, 2, None, None, None] * i
                    )
                    .add_(mapped_offset[:, :, None, None, None])
                    .movedim(1, -1)
                )
            else:
                # Extreme finite coefficients may overflow only when composed.
                if local is None:
                    local = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
                    local = local @ torch.tensor(target_affine[:3, :3].T, device=device)
                query = (
                    torch.einsum("bij,...j->b...i", matrices[:, :3, :3], local)
                    + torch.tensor(offset, device=device)[:, None, None, None]
                ) @ inverse_lattice
            if shared:
                query = query.expand(last - first, *shape, 3)
        query_finite = (
            torch.isfinite(torch.stack(torch.aminmax(query))).all()
            if native_linear
            else torch.isfinite(query).all()
        )
        if not query_finite:
            raise ValueError("Transform produced nonfinite sampling coordinates.")
        valid = ((query >= lower) & (query <= upper)).all(dim=-1)
        original = torch.tensor(
            np.ascontiguousarray(moving.values[first:last], dtype=dtype), device=device
        )
        if native_linear:
            sampled = _sample_linear(original, query, scale, shift)
        elif interpolation == "bspline":
            sampled = sample_cubic_spline(prepare_cubic_spline(original), query, valid)
        else:
            # A normalized-coordinate round trip can move an exact tie to either side.
            sampled = sample_nearest(original, query.unbind(-1))
        sampled = torch.where(
            valid, sampled, original.amin(dim=(1, 2, 3))[:, None, None, None]
        )
        output = sampled.cpu().numpy()
        outputs.append(restore_image_dtype(output, moving.dtype, interpolation))
        masks.append(valid.cpu().numpy())
    template = build_output_template(moving, reference, False)
    resampled = template.copy(data=np.concatenate(outputs))
    valid_data = template.copy(data=np.concatenate(masks))
    valid_data.attrs = {"description": "Valid moving-image sample centers"}
    if spatial:
        return resampled.isel(time=0, drop=True), valid_data.isel(time=0, drop=True)
    return resampled, valid_data
