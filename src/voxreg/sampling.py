"""Differentiable physical-space sampling in a single voxel plane."""

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
    points : torch.Tensor
        Centered physical plane coordinates (H, W, 2), ordered i, j.
    inverse_lattice : torch.Tensor
        Plane distance to pixel-offset map (2, 2).
    image_center : torch.Tensor
        Image center in i, j pixel positions.
    basis : numpy.ndarray
        Orthonormal world z/y/x plane basis (3, 2).
    center : numpy.ndarray
        Physical world z/y/x rotation center.
    parameter_scale : torch.Tensor
        Radian/mm conversion for scaled optimizer parameters.
    """

    points: torch.Tensor
    inverse_lattice: torch.Tensor
    image_center: torch.Tensor
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
    return Plane(
        torch.tensor(points, **options),
        torch.tensor(np.linalg.inv(lattice), **options),
        torch.tensor(center_pixel, **options),
        basis,
        affine[:3, 3] + pixels @ center_pixel,
        torch.tensor([ell / radius, ell, ell], **options),
    )


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
    images: torch.Tensor, parameters: torch.Tensor, plane: Plane
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull-sample images under centered physical in-plane rigid transforms.

    Parameters
    ----------
    images : (B, H, W) torch.Tensor
        Moving image intensities on the plane grid.
    parameters : (B, 3) torch.Tensor
        Physical angle, i-direction translation, j-direction translation.
    plane : Plane
        Shared plane geometry, on the same device as images and dtype as parameters.
        The final normalized interpolation grid uses the image dtype.

    Returns
    -------
    sampled : (B, H, W) torch.Tensor
        Bilinearly sampled images; outside values are zero pending explicit fill.
    valid : (B, H, W) torch.Tensor
        Pixel-center support mask, excluding out-of-field samples.
    """
    rotation = rotation_matrices(parameters[:, 0])
    mapped = (
        torch.einsum("bij,hwj->bhwi", rotation, plane.points)
        + parameters[:, None, None, 1:]
    )
    positions = mapped @ plane.inverse_lattice.T + plane.image_center
    bounds = images.new_tensor([images.shape[2] - 1, images.shape[1] - 1])
    # A tiny tolerance keeps roundoff at exact identity boundaries from losing pixels.
    valid = ((positions >= -1e-5) & (positions <= bounds + 1e-5)).all(dim=-1)
    grid = positions.clamp(min=0) / bounds * 2 - 1
    sampled = F.grid_sample(
        images[:, None],
        grid.to(images.dtype),
        mode="bilinear",
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
    moved : (B, H, W) torch.Tensor
        Sampled moving intensities.
    fixed : (H, W) torch.Tensor
        Fixed image intensities.
    valid : (B, H, W) torch.Tensor
        Valid moving sample centers.

    Returns
    -------
    loss : (B,) torch.Tensor
        Negative squared correlations, with guards for invalid metric arithmetic.
    meaningful : (B,) torch.Tensor
        Whether support and both centered variances are nonzero.
    """
    axes = (1, 2)
    # Near unit NCC, float32 loss quantization can select different best iterates.
    moved, fixed = moved.to(torch.float64), fixed.to(torch.float64)
    weights = valid.to(moved.dtype)
    count = weights.sum(dim=axes, keepdim=True).clamp_min(1)
    x = moved - (moved * weights).sum(dim=axes, keepdim=True) / count
    y = fixed - (fixed * weights).sum(dim=axes, keepdim=True) / count
    vx = (x.square() * weights).sum(dim=axes)
    vy = (y.square() * weights).sum(dim=axes)
    denominator = vx * vy
    meaningful = (
        (count[:, 0, 0] > 1)
        & (vx > 0)
        & (vy > 0)
        & torch.isfinite(denominator)
        & (denominator > torch.finfo(moved.dtype).tiny)
    )
    safe_denominator = torch.where(
        meaningful, denominator, torch.ones_like(denominator)
    )
    loss = -(x * y * weights).sum(dim=axes).square() / safe_denominator
    return torch.where(meaningful, loss, torch.zeros_like(loss)), meaningful


def world_matrices(parameters: npt.ArrayLike, plane: Plane) -> npt.NDArray[np.float64]:
    """Embed physical in-plane transforms in homogeneous world z/y/x coordinates.

    Parameters
    ----------
    parameters : (B, 3) array-like
        Angles in radians and translations in physical mm.
    plane : Plane
        World plane basis and rotation center.

    Returns
    -------
    transforms : (B, 4, 4) numpy.ndarray
        Fixed-world-to-moving-world pull matrices.
    """
    values = np.asarray(parameters, dtype=np.float64)
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
