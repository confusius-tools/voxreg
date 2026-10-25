"""Translation/affine state on the same centered physical sampling geometry."""

from dataclasses import replace

import numpy as np
import torch

from voxreg.geometry import position_to_world


def configure_transform(grid, fixed, transform):
    """Keep rigid geometry verbatim; scale other models by physical RMS shifts."""
    if transform == "rigid":
        return grid
    dimensions = grid.basis.shape[1]
    affine = position_to_world(fixed)
    lattice = grid.basis.T @ affine[:3, :3]
    sizes = np.array([fixed.sizes[d] for d in ("k", "j", "i")])
    ell = np.linalg.norm(affine[:3, :3][:, sizes > 1], axis=0).min()
    scales = np.full(dimensions, ell)
    if transform == "affine":
        center_offset = grid.basis.T @ (
            affine[:3, 3] + affine[:3, :3] @ ((sizes - 1) / 2) - grid.center
        )
        variance = lattice**2 @ ((sizes.astype(float) ** 2 - 1) / 12) + center_offset**2
        scales = np.r_[np.tile(ell / np.sqrt(variance), dimensions), scales]
    return replace(
        grid,
        transform=transform,
        parameter_scale=grid.parameter_scale.new_tensor(scales),
    )


def intrinsic_linear(parameters, grid):
    """Return centered intrinsic linear map and center displacement (non-rigid models)."""
    dimensions = grid.basis.shape[1]
    identity = torch.eye(dimensions, dtype=parameters.dtype, device=parameters.device)
    linear = identity.expand(len(parameters), -1, -1)
    if grid.transform == "affine":
        linear = linear + parameters[:, : dimensions**2].reshape(
            -1, dimensions, dimensions
        )
    return linear, parameters[:, -dimensions:]


def affine_properties(parameters, grid):
    """Finite nonsingular map and orientation, without device scalar reads.

    Normalize before the small determinant to avoid overflow/underflow. A
    normalized determinant <= 1e-8 is treated as numerically singular.
    """
    linear, _ = intrinsic_linear(parameters.detach(), grid)
    largest = linear.abs().amax(dim=(-2, -1), keepdim=True)
    matrix = linear / largest.clamp_min(torch.finfo(linear.dtype).tiny)
    if matrix.shape[-1] == 2:
        determinant = (
            matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]
        )
    else:
        determinant = (
            matrix[:, 0] * torch.linalg.cross(matrix[:, 1], matrix[:, 2])
        ).sum(dim=-1)
    valid = (
        torch.isfinite(parameters).all(dim=-1)
        & torch.isfinite(determinant)
        & (determinant.abs() > 1e-8)
    )
    return valid, determinant.sign()


def validate_matrices(transforms, count):
    """Copy and validate common homogeneous world-pull seed structure."""
    values = np.asarray(transforms)
    if values.dtype.kind not in "iuf":
        raise TypeError("initial_transform must contain real numeric matrices.")
    values = np.array(values, dtype=np.float64, copy=True)
    if values.shape == (4, 4):
        values = np.broadcast_to(values, (count, 4, 4))
    if values.shape != (count, 4, 4) or not np.isfinite(values).all():
        raise ValueError(
            "initial_transform must be finite with shape (4, 4) or (N, 4, 4), where N is the number of moving frames."
        )
    if not np.allclose(values[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError(
            "initial_transform must have homogeneous last row [0, 0, 0, 1]."
        )
    return values


def initial_linear_parameters(values, grid):
    """Recover the complete optimized translation/affine seed, never a frozen chain."""
    linear = values[:, :3, :3]
    translation = (
        np.einsum("bij,j->bi", linear - np.eye(3), grid.center) + values[:, :3, 3]
    )
    dimensions = grid.basis.shape[1]
    intrinsic = grid.basis.T @ linear @ grid.basis
    local_translation = translation @ grid.basis
    if grid.transform == "translation" and not np.allclose(
        linear, np.eye(3), atol=1e-12, rtol=0
    ):
        raise ValueError(
            "Translation initial_transform must have identity linear part; rotation, scale and shear are unsupported."
        )
    if dimensions == 2:
        embedded = np.eye(3) + grid.basis @ (intrinsic - np.eye(2)) @ grid.basis.T
        ell = float(grid.parameter_scale[-1].cpu())
        tolerance = 1e-5 * ell + 32 * np.finfo(float).eps * max(
            1, np.abs(grid.center).max()
        )
        if not np.allclose(linear, embedded, rtol=0, atol=1e-6) or not np.allclose(
            translation, local_translation @ grid.basis.T, rtol=0, atol=tolerance
        ):
            raise ValueError(
                "Slice initial_transform must preserve the fixed plane and its normal; out-of-plane motion is unsupported."
            )
    if grid.transform == "translation":
        return local_translation
    parameters = np.column_stack(
        ((intrinsic - np.eye(dimensions)).reshape(len(values), -1), local_translation)
    )
    valid, _ = affine_properties(torch.from_numpy(parameters), grid)
    if not valid.all():
        raise ValueError(
            "Affine initial_transform must be nonsingular and numerically well-conditioned (normalized |determinant| > 1e-8)."
        )
    return parameters


def linear_world_matrices(parameters, grid):
    """Export full centered world pulls, including affine scale/shear/reflection."""
    values = np.asarray(parameters, dtype=np.float64)
    dimensions = grid.basis.shape[1]
    linear = np.broadcast_to(np.eye(3), (len(values), 3, 3)).copy()
    if grid.transform == "affine":
        delta = values[:, : dimensions**2].reshape(-1, dimensions, dimensions)
        linear += grid.basis @ delta @ grid.basis.T
    matrices = np.broadcast_to(np.eye(4), (len(values), 4, 4)).copy()
    matrices[:, :3, :3] = linear
    matrices[:, :3, 3] = values[:, -dimensions:] @ grid.basis.T - np.einsum(
        "bij,j->bi", linear - np.eye(3), grid.center
    )
    return matrices


def linear_diagnostics(parameters, grid):
    """Actual linear coefficients (not identity deltas) and physical center shifts."""
    axes = ["i", "j"] if grid.basis.shape[1] == 2 else ["z", "y", "x"]
    dimensions = len(axes)
    motion = {}
    if grid.transform == "affine":
        matrix = np.eye(dimensions) + parameters[:, : dimensions**2].reshape(
            -1, dimensions, dimensions
        )
        for row, out in enumerate(axes):
            for col, inp in enumerate(axes):
                motion[f"linear_{out}{inp}"] = (
                    "time",
                    matrix[:, row, col],
                    {
                        "units": "1",
                        "basis": "intrinsic i/j" if dimensions == 2 else "world z/y/x",
                    },
                )
    for col, axis in enumerate(axes):
        motion[f"translation_{axis}"] = (
            "time",
            parameters[:, -dimensions + col],
            {
                "units": "mm",
                "meaning": "center displacement",
                "basis": "intrinsic i/j" if dimensions == 2 else "world z/y/x",
            },
        )
    return motion
