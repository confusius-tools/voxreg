"""Apply exported world-space pulls without fitting or intensity conditioning."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import xarray as xr

from voxreg.geometry import (
    VOXEL_DIMS,
    length_scale_to_mm,
    position_to_world,
    same_physical_grid,
    validate_registration_geometry,
)
from voxreg.preparation import choose_batch_size
from voxreg.results import build_output_template
from voxreg.sampling import sample_displaced
from voxreg.transforms import validate_matrices


def _sample_linear(
    images: torch.Tensor, query: torch.Tensor, *, voxel_extent: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume owned xyz index queries without fitting-only coordinate round trips."""
    sizes = np.array(images.shape[1:][::-1])
    extent = (sizes == 1) | voxel_extent
    # One transfer; the adjacent double preserves half-open singleton faces.
    lower, upper, scale, shift = torch.tensor(
        np.stack(
            (
                np.where(extent, -0.5, -1e-10),
                np.where(extent, np.nextafter(sizes - 0.5, -np.inf), sizes - 1 + 1e-10),
                np.where(sizes == 1, 0.0, 2 / np.maximum(sizes - 1, 1)),
                np.where(sizes == 1, 0.0, -1.0),
            )
        ),
        device=query.device,
    ).unbind()
    valid = ((query >= lower) & (query <= upper)).all(dim=-1)
    shape = query.shape[:-1]
    planar = images.shape[1] == shape[1] == 1
    # Normalize owned storage once, even when several frames share the query.
    base = query[:1] if query.stride(0) == 0 else query
    if planar:
        grid = base[:, 0, ..., :2].mul_(scale[:2]).add_(shift[:2])
        source = images[:, 0].double()
    else:
        grid = base.mul_(scale).add_(shift)
        source = images.double()
    grid = grid.expand(len(images), *grid.shape[1:])
    tiles = (
        min(torch.get_num_threads(), grid.numel() // 3)
        if images.device.type == "cpu" and not planar and len(images) == 1
        else 1
    )
    if tiles > 1:
        # As in registration, partition points while sharing one promoted image.
        points = grid.numel() // 3
        length = (points + tiles - 1) // tiles
        grid = grid.reshape(points, 3)
        if tiles * length != points:
            grid = F.pad(grid, (0, 0, 0, tiles * length - points))
        grid = grid.reshape(tiles, 1, 1, length, 3)
        source = source.expand(tiles, *source.shape[1:])
    sampled = F.grid_sample(
        source[:, None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[:, 0]
    if tiles > 1:
        sampled = sampled.reshape(-1)[:points].reshape(shape)
    elif planar:
        sampled = sampled[:, None]
    return sampled.to(images.dtype), valid


@torch.no_grad()
def resample(
    moving: xr.DataArray,
    reference: xr.DataArray,
    transforms: npt.ArrayLike | xr.DataArray,
    *,
    interpolation: Literal["linear", "nearest"] = "linear",
    device: Literal["cpu", "cuda"] = "cpu",
    batch_size: int | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Resample original intensities using a reference-to-moving world pull.

    Parameters
    ----------
    moving : xarray.DataArray
        Finite real NumPy-backed VoxelData, spatial or time-dependent. Boolean
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
        spatial_units. ConfUSIus length units are converted to mm. Field lattice
        units are independent of component units. Fields must match the physical
        reference grid; they are not implicitly regridded. Use forward
        transforms with moving images and the fixed reference, or inverse
        transforms with fixed images and the moving reference.
        Spatial images/maps broadcast over time-dependent maps/images. When both
        have time labels they must match exactly. Anonymous per-frame matrices
        applied to a spatial image produce integer time labels.
    interpolation : {"linear", "nearest"}, default="linear"
        Use nearest for categorical labels. Outside samples use each original
        moving frame's minimum and are marked invalid.
    device : {"cpu", "cuda"}, default="cpu"
        Sampling device. Results are NumPy-backed on CPU in either case.
    batch_size : int, optional
        Positive frame batch limit; omitted uses the registration memory policy.

    Returns
    -------
    resampled : xarray.DataArray
        Original intensities on the reference grid, retaining moving dtype and
        time metadata. Integer interpolation is rounded to nearest-even and clipped.
    valid : xarray.DataArray
        Boolean mask of supported moving-image sample centers. Singleton axes
        have half-open voxel-extent support. Mixed slice/volume pairs use this
        support on every axis, matching registration.

    Raises
    ------
    TypeError
        If images, transforms or their numeric types are unsupported.
    ValueError
        If geometry, labels, units, transforms or sampling options are invalid.
    """
    validate_registration_geometry(moving)
    validate_registration_geometry(reference)
    if reference.dims != VOXEL_DIMS:
        raise ValueError("reference must be spatial VoxelData without time.")
    if interpolation not in ("linear", "nearest"):
        raise ValueError("interpolation must be linear or nearest.")
    if device not in ("cpu", "cuda") or (
        device == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("device must be cpu or an available cuda device.")
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("batch_size must be None or a positive integer.")
    integer = np.issubdtype(moving.dtype, np.integer)
    boolean = moving.dtype == np.bool_
    if not (
        integer or boolean or moving.dtype.kind == "f" and moving.dtype.itemsize <= 8
    ):
        raise TypeError("moving must have real integer or floating-point intensities.")
    if boolean and interpolation != "nearest":
        raise ValueError("Boolean images require nearest interpolation.")
    if not np.isfinite(moving.values).all():
        raise ValueError("moving intensities must be finite.")
    if integer and (
        int(moving.values.min()) < -(2**53) or int(moving.values.max()) > 2**53
    ):
        raise ValueError("moving integers must lie within [-2**53, 2**53].")

    labeled = isinstance(transforms, xr.DataArray)
    field = labeled and "world_component" in transforms.dims
    if labeled:
        if not isinstance(transforms.data, np.ndarray):
            raise TypeError("transforms must be NumPy-backed.")
        length_scale_to_mm(transforms.attrs.get("spatial_units", "mm"))
        if "time" in transforms.dims:
            if (
                "time" not in transforms.coords
                or transforms.time.dims != ("time",)
                or not transforms.get_index("time").is_unique
            ):
                raise ValueError("transforms must have unique time labels.")
            if "time" in moving.dims and not moving.get_index("time").equals(
                transforms.get_index("time")
            ):
                raise ValueError(
                    "moving and transforms time labels must match exactly."
                )
        for axis in ("world_in", "world_out"):
            if axis in transforms.dims and list(transforms[axis].values) != [
                "z",
                "y",
                "x",
                "1",
            ]:
                raise ValueError("Matrix axes must be ordered z/y/x/1.")
    values = np.asarray(transforms)
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("transforms must contain finite real numeric values.")
    if field:
        assert isinstance(transforms, xr.DataArray)
        expected = (("time",) if "time" in transforms.dims else ()) + (
            *VOXEL_DIMS,
            "world_component",
        )
        if transforms.dims != expected or list(transforms.world_component.values) != [
            "z",
            "y",
            "x",
        ]:
            raise ValueError(
                "Displacements must have dimensions [time,] k/j/i/world_component ordered z/y/x."
            )
        scalar = transforms.isel(world_component=0, drop=True)
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
        scale = length_scale_to_mm(transforms.attrs["spatial_units"])
        if (
            "units" in transforms.attrs
            and length_scale_to_mm(transforms.attrs["units"]) != scale
        ):
            raise ValueError(
                "Field units and spatial_units must describe the same length unit."
            )
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
        if labeled and transforms.dims not in (
            ("world_out", "world_in"),
            ("time", "world_out", "world_in"),
        ):
            raise ValueError("Matrix dimensions must be [time,] world_out/world_in.")
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
    if (
        "time" not in moving.coords
        or moving.time.dims != ("time",)
        or not moving.get_index("time").is_unique
    ):
        raise ValueError("moving must have unique time labels.")
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
    shape = tuple(reference.sizes[d] for d in VOXEL_DIMS)
    indices = torch.stack(
        torch.meshgrid(
            *(torch.arange(n, device=device, dtype=torch.float64) for n in shape),
            indexing="ij",
        ),
        dim=-1,
    )
    local = (
        indices @ torch.tensor(target_affine[:3, :3].T, device=device)
        if field or not native_linear
        else indices
    )
    inverse = np.linalg.inv(source_affine[:3, :3]).T
    inverse_lattice = torch.tensor(
        inverse[:, ::-1].copy() if native_linear else inverse, device=device
    )
    dtype = np.float64 if integer or moving.dtype.itemsize > 4 else np.float32
    outputs, masks = [], []
    for first in range(0, frames, batch_size):
        last = min(first + batch_size, frames)
        if shared:
            selected = values if field else values[:1]
        else:
            selected = values[first:last]
        if field:
            displacement = torch.tensor(selected, device=device, dtype=torch.float64)
            offset = torch.tensor(
                target_affine[:3, 3] - source_affine[:3, 3], device=device
            )
            query = (local + displacement + offset) @ inverse_lattice
            if shared:
                query = query.unsqueeze(0).expand(last - first, *shape, 3)
        else:
            matrices = torch.tensor(selected, device=device)
            offset = (
                selected[:, :3, :3] @ target_affine[:3, 3]
                + selected[:, :3, 3]
                - source_affine[:3, 3]
            )
            linear = None
            if native_linear:
                linear = inverse_lattice.T @ (
                    matrices[:, :3, :3]
                    @ torch.tensor(target_affine[:3, :3], device=device)
                )
            if linear is not None and torch.isfinite(linear).all():
                mapped_offset = torch.tensor(offset, device=device) @ inverse_lattice
                if device == "cuda":
                    # Separable coordinates avoid a dense, skinny float64 GEMM.
                    k = indices[:, 0, 0, 0][None, None, :, None, None]
                    j = indices[0, :, 0, 1][None, None, None, :, None]
                    i = indices[0, 0, :, 2][None, None, None, None, :]
                    query = (
                        linear[:, :, 0, None, None, None] * k
                        + linear[:, :, 1, None, None, None] * j
                        + linear[:, :, 2, None, None, None] * i
                        + mapped_offset[:, :, None, None, None]
                    ).movedim(1, -1)
                else:
                    query = (
                        torch.einsum("bij,...j->b...i", linear, indices)
                        + mapped_offset[:, None, None, None]
                    )
            else:
                # Extreme finite coefficients may overflow only when composed.
                if native_linear:
                    local = indices @ torch.tensor(
                        target_affine[:3, :3].T, device=device
                    )
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
        if not native_linear:
            valid = torch.ones(query.shape[:-1], device=device, dtype=torch.bool)
            normalized = []
            for axis, size in enumerate(moving.shape[1:]):
                coordinate = query[..., axis]
                if mixed or size == 1:
                    valid &= (coordinate >= -0.5) & (coordinate < size - 0.5)
                else:
                    # Allow only double-coordinate roundoff at ordinary lattice faces.
                    valid &= (coordinate >= -1e-10) & (coordinate <= size - 1 + 1e-10)
                normalized.append(
                    torch.zeros_like(coordinate)
                    if size == 1
                    else coordinate * (2 / (size - 1)) - 1
                )
        original = torch.tensor(
            np.ascontiguousarray(moving.values[first:last], dtype=dtype), device=device
        )
        if native_linear:
            sampled, valid = _sample_linear(original, query, voxel_extent=mixed)
        else:
            sampled = sample_displaced(
                original, torch.stack(normalized[::-1], dim=-1), interpolation
            )
        sampled = torch.where(
            valid, sampled, original.amin(dim=(1, 2, 3))[:, None, None, None]
        )
        output = sampled.cpu().numpy()
        if integer:
            limits = np.iinfo(moving.dtype)
            output = np.clip(np.rint(output), limits.min, limits.max)
        outputs.append(output.astype(moving.dtype, copy=False))
        masks.append(valid.cpu().numpy())
    template = build_output_template(moving, reference, False)
    resampled = template.copy(data=np.concatenate(outputs))
    valid_data = template.copy(data=np.concatenate(masks))
    valid_data.attrs = {"description": "Valid moving-image sample centers"}
    if spatial:
        return resampled.isel(time=0, drop=True), valid_data.isel(time=0, drop=True)
    return resampled, valid_data
