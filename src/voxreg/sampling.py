"""Differentiable physical-space sampling on independent fixed and moving grids."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import xarray as xr

from voxreg.geometry import _plane_alignment, position_to_world
from voxreg.transforms import (
    initial_linear_parameters,
    intrinsic_linear,
    linear_world_matrices,
    validate_matrices,
)


@dataclass(frozen=True)
class Plane:
    """Resident intrinsic-plane geometry and host-side world embedding.

    Parameters
    ----------
    inverse_lattice : torch.Tensor
        Plane distance to pixel-offset map (2, 2).
    plane_to_normalized : torch.Tensor
        Physical plane distances to normalized grid offsets (2, 2).
    normalized_to_plane : torch.Tensor
        Normalized grid offsets to physical plane distances (2, 2).
    normalized_i : torch.Tensor
        Normalized i positions (1, 1, W).
    normalized_j : torch.Tensor
        Normalized j positions (1, H, 1).
    support_tolerance : torch.Tensor
        Normalized equivalent of the native 1e-5-pixel support tolerance (2,).
    identity : torch.Tensor
        Cached identity rotation (2, 2).
    basis : numpy.ndarray
        Orthonormal world z/y/x plane basis (3, 2).
    center : numpy.ndarray
        Physical world z/y/x rotation center.
    parameter_scale : torch.Tensor
        Radian/mm conversion for scaled optimizer parameters.
    rotation_offset : torch.Tensor, optional
        Intrinsic displacement from the rotation pivot to the geometric grid center.
    transform : str, default: "rigid"
        Selected translation, rigid or affine parameterization.
    """

    inverse_lattice: torch.Tensor
    plane_to_normalized: torch.Tensor
    normalized_to_plane: torch.Tensor
    normalized_i: torch.Tensor
    normalized_j: torch.Tensor
    support_tolerance: torch.Tensor
    identity: torch.Tensor
    basis: npt.NDArray[np.float64]
    center: npt.NDArray[np.float64]
    parameter_scale: torch.Tensor
    rotation_offset: torch.Tensor | None = None
    transform: str = "rigid"


def make_plane(
    data: xr.DataArray,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    center: npt.NDArray[np.float64] | None = None,
) -> Plane:
    """Build intrinsic-plane geometry without large-world-origin cancellation.

    Parameters
    ----------
    data : xarray.DataArray
        Single-k VoxelData grid with at least two j and i samples.
    device : str, default: "cpu"
        Torch device.
    dtype : torch.dtype, default: torch.float32
        Numerical sampling precision.
    center : numpy.ndarray, optional
        Physical world z/y/x in-plane pivot. Defaults to the geometric grid center.

    Returns
    -------
    plane : Plane
        Geometry for physical rotation/translation and world matrix export.

    Raises
    ------
    ValueError
        If the spatial shape is not a supported single plane.
    """
    affine = position_to_world(data)
    if data.sizes["k"] != 1 or min(data.sizes["j"], data.sizes["i"]) < 2:
        raise ValueError(
            "Rigid registration requires k=1 and j/i sizes of at least two."
        )
    pixels = affine[:3, [2, 1]]
    e_i = pixels[:, 0] / np.linalg.norm(pixels[:, 0])
    e_j = pixels[:, 1] - e_i * np.dot(e_i, pixels[:, 1])
    e_j /= np.linalg.norm(e_j)
    basis = np.column_stack((e_i, e_j))
    lattice = basis.T @ pixels
    h, w = data.sizes["j"], data.sizes["i"]
    center_pixel = np.array([(w - 1) / 2, (h - 1) / 2])
    j, i = np.indices((h, w), dtype=float)
    points = (np.stack((i, j), axis=-1) - center_pixel) @ lattice.T
    geometric_center = affine[:3, 3] + pixels @ center_pixel
    rotation_offset = None
    if center is not None:
        rotation_offset = basis.T @ (geometric_center - center)
        points = points + rotation_offset
    radius = np.sqrt(np.mean(np.sum(points**2, axis=-1)))
    ell = np.linalg.norm(pixels, axis=0).min()
    options = {"device": device, "dtype": dtype}
    inverse_lattice = torch.tensor(np.linalg.inv(lattice), **options)
    half = torch.tensor(center_pixel, **options)
    bounds = 2 * half
    return Plane(
        inverse_lattice=inverse_lattice,
        plane_to_normalized=inverse_lattice / half[:, None],
        normalized_to_plane=torch.tensor(lattice, **options) * half[None, :],
        normalized_i=torch.arange(w, **options)[None, None, :] / bounds[0] * 2 - 1,
        normalized_j=torch.arange(h, **options)[None, :, None] / bounds[1] * 2 - 1,
        support_tolerance=2e-5 / bounds,
        identity=torch.eye(2, **options),
        basis=basis,
        center=geometric_center if center is None else center,
        parameter_scale=torch.tensor([ell / radius, ell, ell], **options),
        rotation_offset=(
            torch.tensor(rotation_offset, **options)
            if rotation_offset is not None
            else None
        ),
    )


@dataclass(frozen=True)
class SpatialGrid:
    """Independent grids, centered physical transforms and fixed-domain sampling."""

    world_to_normalized: torch.Tensor
    normalized_to_world: torch.Tensor
    base: torch.Tensor
    offset: torch.Tensor
    coordinates: tuple[torch.Tensor, ...]
    lower: torch.Tensor
    upper: torch.Tensor
    resident_basis: torch.Tensor
    basis: npt.NDArray[np.float64]
    center: npt.NDArray[np.float64]
    parameter_scale: torch.Tensor
    planar: bool
    inverse_slice_affine: bool = False
    slice_basis: npt.NDArray[np.float64] | None = None
    resident_slice_basis: torch.Tensor | None = None
    plane_rotation: npt.NDArray[np.float64] | None = None
    plane_shift: npt.NDArray[np.float64] | None = None
    resident_rotation: torch.Tensor | None = None
    resident_shift: torch.Tensor | None = None
    rotation_offset: torch.Tensor | None = None
    transform: str = "rigid"


def make_grid(
    fixed: xr.DataArray,
    moving: xr.DataArray,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    center: npt.NDArray[np.float64] | None = None,
    plane_seed: npt.ArrayLike | None = None,
) -> Plane | SpatialGrid:
    """Build fixed-domain sampling geometry without conflating voxel spacing with motion.

    Slice pairs require a plane-compatible seed when not initially coplanar.
    A fixed slice against a moving volume uses its full 3D pose and samples only
    the slice domain. Optional center controls the pivot and physical scales.
    """
    fs = np.array([fixed.sizes[d] for d in ("k", "j", "i")])
    ms = np.array([moving.sizes[d] for d in ("k", "j", "i")])
    planar = fs[0] == ms[0] == 1
    slice_volume = fs[0] == 1 and min(*ms) >= 2
    volume_slice = ms[0] == 1 and min(*fs) >= 2
    mixed = slice_volume or volume_slice
    if min(*fs[1:], *ms[1:]) < 2 or (not planar and not mixed and min(*fs, *ms) < 2):
        raise ValueError("Use k=1 slices or 3D volumes with all axes >= 2.")
    af, am = position_to_world(fixed), position_to_world(moving)
    lf, lm = af[:3, :3], am[:3, :3]
    seeds, reversed_normal = None, False
    if planar and plane_seed is not None:
        seeds = validate_matrices(plane_seed, moving.sizes.get("time", 1))
        normal = np.cross(lf[:, 2], lf[:, 1])
        reversed_normal = (seeds[0, :3, :3] @ normal) @ normal < 0
    if (
        planar
        and not reversed_normal
        and fixed.xindexes["z"].equals(moving.xindexes["z"])
    ):
        # Retain the qualified same-grid 2D arithmetic and kernel exactly.
        return make_plane(fixed, device=device, dtype=dtype, center=center)
    hf, hm = (fs - 1) / 2, (ms - 1) / 2
    geometric_center = af[:3, 3] + lf @ hf
    origin_difference = af[:3, 3] - am[:3, 3]
    plane_rotation = plane_shift = None
    if planar:
        native = np.linalg.solve(lm, lf)
        offset = np.linalg.solve(lm, origin_difference)
        if (
            reversed_normal
            or abs(offset[0]) + np.abs(native[0] * (fs - 1)).sum() > 1e-5
        ):
            if plane_seed is None:
                raise ValueError(
                    "Noncoplanar slices need a plane-compatible initial_transform or initialization='align_planes'; centering alone cannot correct tilt."
                )
            plane_rotation, plane_shift = _plane_alignment(
                fixed, moving, seeds[0, :3, :3] @ normal
            )
        plane = make_plane(fixed, device=device, dtype=dtype, center=center)
        basis, scales = plane.basis, plane.parameter_scale
    else:
        basis = np.eye(3)
        axis_variance = lf**2 @ ((fs.astype(float) ** 2 - 1) / 12)
        if center is not None:
            axis_variance += (geometric_center - center) ** 2
        # Sum off-axis variances directly, avoiding cancellation for elongated grids.
        radii = np.sqrt(axis_variance[[1, 2, 0]] + axis_variance[[2, 0, 1]])
        ell = np.linalg.norm(lf, axis=0).min()
        scales = torch.tensor(np.r_[ell / radii, [ell] * 3], device=device, dtype=dtype)
    half = np.where(hm > 0, hm, 1)
    world_to_normalized = np.linalg.inv(lm) / half[:, None]
    normalized_to_world = lf if mixed else lf * hf
    center_offset = origin_difference + lf @ hf
    normalized_center = np.where(hm > 0, 1, 0)
    dimensions = 2 if planar else 3
    coordinates = []
    for axis, size in enumerate(fs):
        shape = [1] * (dimensions + 1)
        if not planar or axis > 0:
            shape[axis + (0 if planar else 1)] = int(size)
        coordinate = torch.arange(int(size), device=device, dtype=dtype)
        if mixed:
            # Integer-centered indices avoid a normalize/denormalize roundoff at voxel faces.
            coordinate = coordinate - hf[axis]
        else:
            coordinate = coordinate / (hf[axis] if hf[axis] > 0 else 1) - (
                1 if size > 1 else 0
            )
        coordinates.append(coordinate.reshape(shape))
    options = {"device": device, "dtype": dtype}
    tolerance = 1e-5 / half
    slice_basis = (
        make_plane(fixed if slice_volume else moving, device=device, dtype=dtype).basis
        if mixed
        else None
    )
    return SpatialGrid(
        world_to_normalized=torch.tensor(world_to_normalized, **options),
        normalized_to_world=torch.tensor(normalized_to_world, **options),
        base=torch.tensor(world_to_normalized @ normalized_to_world, **options),
        offset=torch.tensor(
            world_to_normalized @ center_offset - normalized_center, **options
        ),
        coordinates=tuple(coordinates),
        lower=torch.tensor(
            -normalized_center - (0.5 / half if mixed else tolerance), **options
        ),
        upper=torch.tensor(
            normalized_center + (0.5 / half if mixed else tolerance), **options
        ),
        resident_basis=torch.tensor(basis, **options),
        basis=basis,
        center=geometric_center if center is None else center,
        parameter_scale=scales,
        planar=bool(planar),
        inverse_slice_affine=bool(volume_slice),
        slice_basis=slice_basis,
        resident_slice_basis=torch.tensor(slice_basis, **options)
        if slice_basis is not None
        else None,
        plane_rotation=plane_rotation,
        plane_shift=plane_shift,
        resident_rotation=torch.tensor(plane_rotation, **options)
        if plane_rotation is not None
        else None,
        resident_shift=torch.tensor(plane_shift, **options)
        if plane_shift is not None
        else None,
        rotation_offset=(
            torch.tensor(geometric_center - center, **options)
            if center is not None
            else None
        ),
    )


def rotation_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """Right-handed XYZ axis-angle rotations, with vector/matrix axes stored z/y/x."""
    x, y, z = vectors.flip(-1).unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    angle = torch.linalg.vector_norm(vectors, dim=-1)
    a = torch.sinc(angle / torch.pi)[:, None, None]
    b = (0.5 * torch.sinc(angle / (2 * torch.pi)).square())[:, None, None]
    identity = torch.eye(3, dtype=vectors.dtype, device=vectors.device)
    return (identity + a * skew + b * (skew @ skew)).flip((-2, -1))


def _model_physical_transform(
    parameters: torch.Tensor, grid: SpatialGrid
) -> tuple[torch.Tensor, torch.Tensor]:
    if grid.transform != "rigid":
        linear, translation = intrinsic_linear(parameters, grid)
        if grid.planar:
            basis = grid.resident_basis
            identity = torch.eye(2, device=parameters.device, dtype=parameters.dtype)
            linear = (
                torch.eye(3, device=parameters.device, dtype=parameters.dtype)
                + basis @ (linear - identity) @ basis.T
            )
            translation = translation @ basis.T
        return linear, translation
    if grid.planar:
        basis = grid.resident_basis
        identity = torch.eye(2, device=parameters.device, dtype=parameters.dtype)
        rotation = torch.eye(3, device=parameters.device, dtype=parameters.dtype) + (
            basis @ (rotation_matrices(parameters[:, 0]) - identity) @ basis.T
        )
        translation = parameters[:, 1:] @ basis.T
    else:
        rotation, translation = rotation_vectors(parameters[:, :3]), parameters[:, 3:]
    return rotation, translation


def _physical_transform(parameters, grid):
    linear, translation = _model_physical_transform(parameters, grid)
    if grid.resident_rotation is not None:
        linear = grid.resident_rotation @ linear
        translation = translation @ grid.resident_rotation.T + grid.resident_shift
    return linear, translation


def _grid_coordinates(parameters, grid, dimensions):
    rotation, translation = _physical_transform(parameters, grid)
    identity = torch.eye(3, device=parameters.device, dtype=parameters.dtype)
    linear = (
        grid.base
        + grid.world_to_normalized @ (rotation - identity) @ grid.normalized_to_world
    )
    offset = grid.offset + translation @ grid.world_to_normalized.T
    if grid.rotation_offset is not None:
        offset = (
            offset
            + ((rotation - identity) @ grid.rotation_offset)
            @ grid.world_to_normalized.T
        )
    expand = (slice(None),) + (None,) * dimensions
    components = []
    for axis in range(3):
        value = offset[:, axis][expand]
        for column, coordinates in enumerate(grid.coordinates):
            value = value + linear[:, axis, column][expand] * coordinates
        components.append(value)
    return torch.stack(components, dim=-1), rotation


def _sample_grid(
    images: torch.Tensor,
    parameters: torch.Tensor,
    grid: SpatialGrid,
    interpolation: str,
    displacement: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mapped, rotation = _grid_coordinates(parameters, grid, images.ndim - 1)
    if displacement is not None:
        mapping = grid.world_to_normalized @ rotation @ grid.resident_basis
        return _sample_prepared(
            images, displacement, (mapped, mapping, grid), interpolation
        )
    upper_valid = (
        mapped < grid.upper if grid.slice_basis is not None else mapped <= grid.upper
    )
    valid = ((mapped >= grid.lower) & upper_valid).all(dim=-1)
    sampler_grid = mapped[..., [2, 1] if grid.planar else [2, 1, 0]]
    sampled = F.grid_sample(
        images[:, None],
        sampler_grid.to(images.dtype),
        mode={"linear": "bilinear", "nearest": "nearest"}[interpolation],
        padding_mode="border",
        align_corners=True,
    )[:, 0]
    return sampled, valid


def _plane_coordinates(parameters, plane):
    if plane.transform == "rigid":
        rotation = rotation_matrices(parameters[:, 0])
        physical_translation = parameters[:, 1:]
    else:
        rotation, physical_translation = intrinsic_linear(parameters, plane)
    # Compose tiny matrices first; dense, skinny GEMMs are slow on CPU and CUDA.
    delta = (
        plane.plane_to_normalized
        @ (rotation - plane.identity)
        @ plane.normalized_to_plane
    )
    if plane.rotation_offset is not None:
        physical_translation = (
            physical_translation + (rotation - plane.identity) @ plane.rotation_offset
        )
    translation = physical_translation @ plane.plane_to_normalized.T
    nx, ny = plane.normalized_i, plane.normalized_j
    # Displacement from identity avoids cancellation at zero motion.
    gx = nx + (
        delta[:, 0, 0, None, None] * nx
        + delta[:, 0, 1, None, None] * ny
        + translation[:, 0, None, None]
    )
    gy = ny + (
        delta[:, 1, 0, None, None] * nx
        + delta[:, 1, 1, None, None] * ny
        + translation[:, 1, None, None]
    )
    return torch.stack((gx, gy), dim=-1), rotation


def _prepare_displaced_sampling(images, parameters, geometry):
    """Cache complete seed queries, never a resampled or prewarped image."""
    if isinstance(geometry, SpatialGrid):
        base, rotation = _grid_coordinates(parameters, geometry, images.ndim - 1)
        mapping = geometry.world_to_normalized @ rotation @ geometry.resident_basis
    else:
        base, rotation = _plane_coordinates(parameters, geometry)
        mapping = geometry.plane_to_normalized @ rotation
    return base, mapping, geometry


def _sample_prepared(images, displacement, prepared, interpolation="linear"):
    """Sample with a resident seed grid/matrix; only the deformation varies."""
    base, mapping, geometry = prepared
    mapped = base + _map_displacement(mapping, displacement)
    if isinstance(geometry, SpatialGrid):
        upper = (
            mapped < geometry.upper
            if geometry.slice_basis is not None
            else mapped <= geometry.upper
        )
        valid = ((mapped >= geometry.lower) & upper).all(dim=-1)
        query = mapped[..., [2, 1] if geometry.planar else [2, 1, 0]]
    else:
        tolerance = geometry.support_tolerance
        valid = ((mapped >= -1 - tolerance) & (mapped <= 1 + tolerance)).all(dim=-1)
        query = mapped
    return _sample_displaced(images, query, interpolation), valid


def _map_displacement(matrix, field, axes=None):
    """Apply tiny geometry matrices without strided, per-voxel batched GEMMs."""
    if (
        field.device.type == "cpu"
        and axes is None
        and not torch.compiler.is_compiling()
    ):
        batch = max(matrix.shape[0], field.shape[0])
        values = field.reshape(*field.shape[:2], -1).expand(batch, -1, -1)
        mapped = torch.bmm(matrix.expand(batch, -1, -1), values)
        # Downstream samplers and norms need the original contiguous component-last layout.
        return (
            mapped.reshape(batch, matrix.shape[1], *field.shape[2:])
            .movedim(1, -1)
            .contiguous()
        )
    expand = (-1,) + (1,) * (field.ndim - 2)
    components = []
    for row in range(matrix.shape[-2]):
        if axes is not None:
            column = axes[row]
            value = matrix[:, row, column].reshape(expand) * field[:, column]
        else:
            value = matrix[:, row, 0].reshape(expand) * field[:, 0]
            for column in range(1, field.shape[1]):
                value = (
                    value + matrix[:, row, column].reshape(expand) * field[:, column]
                )
        components.append(value)
    return torch.stack(components, dim=-1)


def _sample_displaced(images, normalized, interpolation):
    """Fixed-order linear interpolation with double geometry and image-dtype math.

    Unlike float32 normalized grid_sample, voxel coordinates are not quantized
    before selecting interpolation cells. At float32-fitting voxel knots, remove
    only double roundoff and choose a consistent one-sided gradient on both devices.
    Final float64 interpolation retains unsnapped float64 coordinates and weights.
    """
    dimensions = images.ndim - 1
    native_cpu = (
        images.device.type == "cpu"
        and images.dtype == torch.float32
        and interpolation == "linear"
    )
    coordinates = []
    for axis, size in enumerate(images.shape[1:]):
        coordinate = (normalized[..., dimensions - axis - 1] + 1) * ((size - 1) / 2)
        if images.dtype == torch.float32:
            nearest = coordinate.round()
            snapped_knot = nearest
            if native_cpu:
                # Preserve the fitting derivative side through normalization roundoff.
                snapped_knot = torch.where(
                    (nearest > 0) & (nearest < size - 1), nearest + 1e-10, nearest
                )
            snapped = torch.where(
                (coordinate - nearest).abs() < 1e-10, snapped_knot, coordinate
            )
            coordinate = coordinate + (snapped - coordinate).detach()
        coordinate = torch.where(
            (coordinate > 0) & (coordinate < size - 1),
            coordinate,
            coordinate.detach().clamp(0, size - 1),
        )
        coordinates.append(coordinate)
    if native_cpu:
        normalized_cpu = [
            c / ((size - 1) / 2) - 1
            for c, size in zip(coordinates, images.shape[1:], strict=True)
        ]
        query = torch.stack(normalized_cpu[::-1], dim=-1)
        return F.grid_sample(
            images[:, None].double(),
            query,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[:, 0].float()
    if interpolation == "nearest":
        index = torch.zeros_like(coordinates[0], dtype=torch.int64)
        for size, coordinate in zip(images.shape[1:], coordinates, strict=True):
            index = index * size + coordinate.round().long()
        return images.flatten(1).gather(1, index.flatten(1)).reshape(index.shape)
    lower = [c.floor().long() for c in coordinates]
    fractions = [
        (c - i).to(images.dtype) for c, i in zip(coordinates, lower, strict=True)
    ]
    result = images.new_zeros(coordinates[0].shape)
    for corner in product((0, 1), repeat=dimensions):
        index = torch.zeros_like(lower[0])
        weight = torch.ones_like(fractions[0])
        for size, i, f, side in zip(
            images.shape[1:], lower, fractions, corner, strict=True
        ):
            index = index * size + (i + side).clamp(0, size - 1)
            weight = weight * (f if side else 1 - f)
        values = images.flatten(1).gather(1, index.flatten(1)).reshape(index.shape)
        result = result + weight * values
    return result


def rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    """Build differentiable 2D rotations.

    Parameters
    ----------
    angles : (B,) torch.Tensor
        Rotation angles in radians.

    Returns
    -------
    rotations : (B, 2, 2) torch.Tensor
        Physical plane rotations.
    """
    c, s = angles.cos(), angles.sin()
    return torch.stack((c, -s, s, c), dim=-1).reshape(-1, 2, 2)


def sample_rigid(
    images: torch.Tensor,
    parameters: torch.Tensor,
    plane: Plane | SpatialGrid,
    *,
    interpolation: str = "linear",
    displacement: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull-sample the fixed domain under the geometry's centered physical model.

    Parameters
    ----------
    images : (B, H, W) or (B, D, H, W) torch.Tensor
        Moving image intensities.
    parameters : (B, P) torch.Tensor
        Rigid slices: angle and intrinsic i/j translations. Rigid volumes: world
        z/y/x rotation vector (right-handed XYZ), then z/y/x translation.
        Translation: d center shifts. Affine: d*d row-major identity deviations
        then d center shifts, where d=2 in the plane or 3 for volumes.
    plane : Plane or SpatialGrid
        Fixed/moving geometry, on the image device and parameter dtype.
        The final normalized interpolation grid uses the image dtype.
    interpolation : {"linear", "nearest"}, default: "linear"
        Sampling rule; registration fitting always uses linear.
    displacement : torch.Tensor, optional
        SyN's physical midpoint-to-fixed-domain displacement, component-first.
        Uses intrinsic i/j components for planes and world z/y/x for volumes;
        applied before the parameterized seed. The None path is unchanged.

    Returns
    -------
    sampled : torch.Tensor
        Fixed-domain images with border extension pending explicit fill.
    valid : torch.Tensor
        Pixel-center support mask, excluding out-of-field samples.
    """
    if isinstance(plane, SpatialGrid):
        return _sample_grid(images, parameters, plane, interpolation, displacement)
    grid, rotation = _plane_coordinates(parameters, plane)
    if displacement is not None:
        mapping = plane.plane_to_normalized @ rotation
        return _sample_prepared(
            images, displacement, (grid, mapping, plane), interpolation
        )
    tolerance = plane.support_tolerance
    valid = ((grid >= -1 - tolerance) & (grid <= 1 + tolerance)).all(dim=-1)
    sampled = F.grid_sample(
        images[:, None],
        grid.to(images.dtype),
        mode={"linear": "bilinear", "nearest": "nearest"}[interpolation],
        padding_mode="border",
        align_corners=True,
    )[:, 0]
    return sampled, valid


def global_ncc(
    moved: torch.Tensor, fixed: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute independent, valid-sample negative squared global correlations.

    Parameters
    ----------
    moved : (B, ...) torch.Tensor
        Sampled moving intensities.
    fixed : (...) torch.Tensor
        Fixed image intensities.
    valid : (B, ...) torch.Tensor
        Valid moving sample centers.

    Returns
    -------
    loss : (B,) torch.Tensor
        Negative squared correlations, with guards for invalid metric arithmetic.
    meaningful : (B,) torch.Tensor
        Whether support and both centered variances are nonzero.
    """
    axes = tuple(range(1, moved.ndim))
    # Near unit NCC, float32 loss quantization can select different best iterates.
    moved, fixed = moved.to(torch.float64), fixed.to(torch.float64)
    weights = valid.to(moved.dtype)
    count = weights.sum(dim=axes, keepdim=True).clamp_min(1)
    x = (moved - (moved * weights).sum(dim=axes, keepdim=True) / count) * weights
    y = (fixed - (fixed * weights).sum(dim=axes, keepdim=True) / count) * weights
    vx = x.square().sum(dim=axes)
    vy = y.square().sum(dim=axes)
    denominator = vx * vy
    meaningful = (
        (count.reshape(len(moved)) > 1)
        & (vx > 0)
        & (vy > 0)
        & torch.isfinite(denominator)
        & (denominator > torch.finfo(moved.dtype).tiny)
    )
    safe_denominator = torch.where(
        meaningful, denominator, torch.ones_like(denominator)
    )
    loss = -(x * y).sum(dim=axes).square() / safe_denominator
    return torch.where(meaningful, loss, torch.zeros_like(loss)), meaningful


def _initial_parameters(
    transforms: npt.ArrayLike, plane: Plane | SpatialGrid, count: int
) -> npt.NDArray[np.float64]:
    """Validate world pull seeds and recover the optimized centered parameters."""
    values = validate_matrices(transforms, count)
    if isinstance(plane, SpatialGrid) and plane.plane_rotation is not None:
        values = values.copy()
        displacement = (
            np.einsum("bij,j->bi", values[:, :3, :3] - np.eye(3), plane.center)
            + values[:, :3, 3]
        )
        values[:, :3, :3] = plane.plane_rotation.T @ values[:, :3, :3]
        displacement = (displacement - plane.plane_shift) @ plane.plane_rotation
        values[:, :3, 3] = displacement - np.einsum(
            "bij,j->bi", values[:, :3, :3] - np.eye(3), plane.center
        )
    if plane.transform != "rigid":
        return initial_linear_parameters(values, plane)
    rotations = values[:, :3, :3]
    if not np.allclose(
        rotations.transpose(0, 2, 1) @ rotations, np.eye(3), rtol=0, atol=1e-6
    ) or not np.allclose(np.linalg.det(rotations), 1, rtol=0, atol=1e-6):
        raise ValueError(
            "initial_transform must be proper rigid: no scale, shear or reflection."
        )
    translations = (
        np.einsum("bij,j->bi", rotations - np.eye(3), plane.center) + values[:, :3, 3]
    )
    # Remove only accepted float32 roundoff, preserving the mapped fixed center.
    left, _, right = np.linalg.svd(rotations)
    rotations = left @ right
    planar = isinstance(plane, Plane) or plane.planar
    if planar:
        basis = plane.basis
        intrinsic = basis.T @ rotations @ basis
        embedded = np.eye(3) + basis @ (intrinsic - np.eye(2)) @ basis.T
        in_plane = translations @ basis
        ell = float(plane.parameter_scale[1].cpu())
        tolerance = 1e-5 * ell + 32 * np.finfo(float).eps * max(
            1, np.abs(plane.center).max()
        )
        if not np.allclose(rotations, embedded, rtol=0, atol=1e-6) or not np.allclose(
            translations, in_plane @ basis.T, rtol=0, atol=tolerance
        ):
            raise ValueError(
                "Slice initial_transform must preserve the fixed plane after accounting for seeded plane placement; use align_planes when centering cannot correct tilt."
            )
        angles = np.arctan2(
            intrinsic[:, 1, 0] - intrinsic[:, 0, 1],
            intrinsic[:, 0, 0] + intrinsic[:, 1, 1],
        )
        return np.column_stack((angles, in_plane))
    vectors = []
    for rotation in rotations[:, ::-1, ::-1]:
        # Largest quaternion component avoids the axis-angle singularity at pi.
        q = np.empty(4)
        trace = np.trace(rotation)
        if trace > 0:
            scale = 2 * np.sqrt(1 + trace)
            q[:3] = (
                np.array(
                    [
                        rotation[2, 1] - rotation[1, 2],
                        rotation[0, 2] - rotation[2, 0],
                        rotation[1, 0] - rotation[0, 1],
                    ]
                )
                / scale
            )
            q[3] = scale / 4
        else:
            i = int(np.argmax(np.diag(rotation)))
            j, k = (i + 1) % 3, (i + 2) % 3
            scale = 2 * np.sqrt(1 + rotation[i, i] - rotation[j, j] - rotation[k, k])
            q[i] = scale / 4
            q[j] = (rotation[j, i] + rotation[i, j]) / scale
            q[k] = (rotation[k, i] + rotation[i, k]) / scale
            q[3] = (rotation[k, j] - rotation[j, k]) / scale
        if q[3] < 0:
            q = -q
        length = np.linalg.norm(q[:3])
        vectors.append(
            q[:3][::-1] * (2 * np.arctan2(length, q[3]) / length if length else 2)
        )
    return np.column_stack((vectors, translations))


def _model_world_matrices(
    parameters: npt.ArrayLike, plane: Plane | SpatialGrid
) -> npt.NDArray[np.float64]:
    """Embed centered model transforms in homogeneous world z/y/x coordinates.

    Parameters
    ----------
    parameters : (B, P) array-like
        Rigid slice angle/i/j shifts or volume rotation-vector/shifts; translation
        d center shifts; affine d*d row-major identity deviations then d shifts.
        Rotation uses radians; affine entries are dimensionless; shifts use mm.
    plane : Plane or SpatialGrid
        World basis and fixed-domain rotation center.

    Returns
    -------
    transforms : (B, 4, 4) numpy.ndarray
        Fixed-world-to-moving-world pull matrices.
    """
    if plane.transform != "rigid":
        return linear_world_matrices(parameters, plane)
    values = np.asarray(parameters, dtype=np.float64)
    if isinstance(plane, SpatialGrid) and not plane.planar:
        rotation = rotation_vectors(torch.from_numpy(values[:, :3])).numpy()
        matrices = np.broadcast_to(np.eye(4), (len(values), 4, 4)).copy()
        matrices[:, :3, :3] = rotation
        matrices[:, :3, 3] = values[:, 3:] - np.einsum(
            "bij,j->bi", rotation - np.eye(3), plane.center
        )
        return matrices
    angles = values[:, 0]
    c, s = np.cos(angles), np.sin(angles)
    rotations = np.stack((c, -s, s, c), axis=-1).reshape(-1, 2, 2)
    matrices = np.broadcast_to(np.eye(4), (len(values), 4, 4)).copy()
    projector = plane.basis @ plane.basis.T
    linear = plane.basis @ rotations @ plane.basis.T + np.eye(3) - projector
    matrices[:, :3, :3] = linear
    matrices[:, :3, 3] = (
        plane.center + values[:, 1:] @ plane.basis.T - linear @ plane.center
    )
    matrices[~values.any(axis=1)] = np.eye(4)
    return matrices


def world_matrices(parameters, plane):
    """Export the complete fixed-world-to-moving-world pull, including plane placement."""
    matrices = _model_world_matrices(parameters, plane)
    if isinstance(plane, SpatialGrid) and plane.plane_rotation is not None:
        displacement = (
            np.einsum("bij,j->bi", matrices[:, :3, :3] - np.eye(3), plane.center)
            + matrices[:, :3, 3]
        )
        matrices[:, :3, :3] = plane.plane_rotation @ matrices[:, :3, :3]
        displacement = displacement @ plane.plane_rotation.T + plane.plane_shift
        matrices[:, :3, 3] = displacement - np.einsum(
            "bij,j->bi", matrices[:, :3, :3] - np.eye(3), plane.center
        )
    return matrices
