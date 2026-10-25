"""Differentiable physical-space sampling on independent fixed and moving grids."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import xarray as xr

from voxreg.geometry import position_to_world


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


def make_plane(
    data: xr.DataArray, *, device: str = "cpu", dtype: torch.dtype = torch.float32
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
        center=affine[:3, 3] + pixels @ center_pixel,
        parameter_scale=torch.tensor([ell / radius, ell, ell], **options),
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


def make_grid(
    fixed: xr.DataArray,
    moving: xr.DataArray,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Plane | SpatialGrid:
    """Build fixed-domain sampling geometry without conflating voxel spacing with motion.

    Slice pairs must be coplanar. Volume pairs require at least two samples on
    each axis. Initialization/slice-to-volume registration are not inferred.
    """
    fs = np.array([fixed.sizes[d] for d in ("k", "j", "i")])
    ms = np.array([moving.sizes[d] for d in ("k", "j", "i")])
    planar = fs[0] == ms[0] == 1
    if (planar and min(*fs[1:], *ms[1:]) < 2) or (not planar and min(*fs, *ms) < 2):
        raise ValueError("Use coplanar k=1 slices or 3D volumes with all axes >= 2.")
    if planar and fixed.xindexes["z"].equals(moving.xindexes["z"]):
        # Retain the qualified same-grid 2D arithmetic and kernel exactly.
        return make_plane(fixed, device=device, dtype=dtype)
    af, am = position_to_world(fixed), position_to_world(moving)
    lf, lm = af[:3, :3], am[:3, :3]
    hf, hm = (fs - 1) / 2, (ms - 1) / 2
    origin_difference = af[:3, 3] - am[:3, 3]
    if planar:
        native = np.linalg.solve(lm, lf)
        offset = np.linalg.solve(lm, origin_difference)
        if abs(offset[0]) + np.abs(native[0] * (fs - 1)).sum() > 1e-5:
            raise ValueError(
                "Slice planes must be coplanar; plane initialization is not supported yet."
            )
        plane = make_plane(fixed, device=device, dtype=dtype)
        basis, scales = plane.basis, plane.parameter_scale
    else:
        basis = np.eye(3)
        axis_variance = lf**2 @ ((fs.astype(float) ** 2 - 1) / 12)
        # Sum off-axis variances directly, avoiding cancellation for elongated grids.
        radii = np.sqrt(axis_variance[[1, 2, 0]] + axis_variance[[2, 0, 1]])
        ell = np.linalg.norm(lf, axis=0).min()
        scales = torch.tensor(np.r_[ell / radii, [ell] * 3], device=device, dtype=dtype)
    half = np.where(hm > 0, hm, 1)
    world_to_normalized = np.linalg.inv(lm) / half[:, None]
    normalized_to_world = lf * hf
    center_offset = origin_difference + lf @ hf
    normalized_center = np.where(hm > 0, 1, 0)
    dimensions = 2 if planar else 3
    coordinates = []
    for axis, size in enumerate(fs):
        shape = [1] * (dimensions + 1)
        if not planar or axis > 0:
            shape[axis + (0 if planar else 1)] = int(size)
        coordinate = torch.arange(int(size), device=device, dtype=dtype)
        coordinate = coordinate / (hf[axis] if hf[axis] > 0 else 1) - (
            1 if size > 1 else 0
        )
        coordinates.append(coordinate.reshape(shape))
    options = {"device": device, "dtype": dtype}
    tolerance = 1e-5 / half
    return SpatialGrid(
        world_to_normalized=torch.tensor(world_to_normalized, **options),
        normalized_to_world=torch.tensor(normalized_to_world, **options),
        base=torch.tensor(world_to_normalized @ normalized_to_world, **options),
        offset=torch.tensor(
            world_to_normalized @ center_offset - normalized_center, **options
        ),
        coordinates=tuple(coordinates),
        lower=torch.tensor(-normalized_center - tolerance, **options),
        upper=torch.tensor(normalized_center + tolerance, **options),
        resident_basis=torch.tensor(basis, **options),
        basis=basis,
        center=af[:3, 3] + lf @ hf,
        parameter_scale=scales,
        planar=bool(planar),
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


def _physical_transform(
    parameters: torch.Tensor, grid: SpatialGrid
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _sample_grid(
    images: torch.Tensor,
    parameters: torch.Tensor,
    grid: SpatialGrid,
    interpolation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation, translation = _physical_transform(parameters, grid)
    identity = torch.eye(3, device=parameters.device, dtype=parameters.dtype)
    linear = (
        grid.base
        + grid.world_to_normalized @ (rotation - identity) @ grid.normalized_to_world
    )
    offset = grid.offset + translation @ grid.world_to_normalized.T
    dimensions = images.ndim - 1
    expand = (slice(None),) + (None,) * dimensions
    components = []
    for axis in range(3):
        value = offset[:, axis][expand]
        for column, coordinates in enumerate(grid.coordinates):
            value = value + linear[:, axis, column][expand] * coordinates
        components.append(value)
    mapped = torch.stack(components, dim=-1)
    valid = ((mapped >= grid.lower) & (mapped <= grid.upper)).all(dim=-1)
    sampler_grid = mapped[..., [2, 1] if grid.planar else [2, 1, 0]]
    sampled = F.grid_sample(
        images[:, None],
        sampler_grid.to(images.dtype),
        mode={"linear": "bilinear", "nearest": "nearest"}[interpolation],
        padding_mode="border",
        align_corners=True,
    )[:, 0]
    return sampled, valid


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull-sample on the fixed domain under centered physical rigid transforms.

    Parameters
    ----------
    images : (B, H, W) or (B, D, H, W) torch.Tensor
        Moving image intensities.
    parameters : (B, 3) or (B, 6) torch.Tensor
        Slices: angle and intrinsic i/j translations. Volumes: world z/y/x
        rotation vector (right-handed XYZ convention), then z/y/x translation.
    plane : Plane or SpatialGrid
        Fixed/moving geometry, on the image device and parameter dtype.
        The final normalized interpolation grid uses the image dtype.
    interpolation : {"linear", "nearest"}, default: "linear"
        Sampling rule; registration fitting always uses linear.

    Returns
    -------
    sampled : torch.Tensor
        Fixed-domain images with border extension pending explicit fill.
    valid : torch.Tensor
        Pixel-center support mask, excluding out-of-field samples.
    """
    if isinstance(plane, SpatialGrid):
        return _sample_grid(images, parameters, plane, interpolation)
    rotation = rotation_matrices(parameters[:, 0])
    # Compose tiny matrices first; dense, skinny GEMMs are slow on CPU and CUDA.
    delta = (
        plane.plane_to_normalized
        @ (rotation - plane.identity)
        @ plane.normalized_to_plane
    )
    translation = parameters[:, 1:] @ plane.plane_to_normalized.T
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
    grid = torch.stack((gx, gy), dim=-1)
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
    values = np.asarray(transforms)
    if values.dtype.kind not in "iuf":
        raise TypeError("initial_transform must contain real numeric matrices.")
    values = np.array(values, dtype=np.float64, copy=True)
    if values.shape == (4, 4):
        values = np.broadcast_to(values, (count, 4, 4))
    if values.shape != (count, 4, 4) or not np.isfinite(values).all():
        raise ValueError(
            "initial_transform must be finite with shape (4, 4) or (N, 4, 4), "
            "where N is the number of moving frames."
        )
    if not np.allclose(values[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError(
            "initial_transform must have homogeneous last row [0, 0, 0, 1]."
        )
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
                "Slice initial_transform must preserve the fixed plane; out-of-plane motion is unsupported."
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


def world_matrices(
    parameters: npt.ArrayLike, plane: Plane | SpatialGrid
) -> npt.NDArray[np.float64]:
    """Embed centered rigid transforms in homogeneous world z/y/x coordinates.

    Parameters
    ----------
    parameters : (B, 3) or (B, 6) array-like
        Slice angle/i/j translations or volume rotation-vector/translations.
        Rotation uses radians; center translations use physical mm.
    plane : Plane or SpatialGrid
        World basis and fixed-domain rotation center.

    Returns
    -------
    transforms : (B, 4, 4) numpy.ndarray
        Fixed-world-to-moving-world pull matrices.
    """
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
