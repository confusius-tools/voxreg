"""Translation/affine state on the same centered physical sampling geometry."""

from dataclasses import replace

import numpy as np
import torch

from voxreg.geometry import _length_scale_to_mm, position_to_world


def configure_transform(grid, fixed, transform):
    """Keep rigid geometry verbatim; scale other models by physical RMS shifts.

    Parameters
    ----------
    grid : Plane or SpatialGrid
        Centered sampling geometry to configure.
    fixed : xarray.DataArray
        Fixed VoxelData defining physical parameter scales.
    transform : {"translation", "rigid", "affine"}
        Linear model to optimize.

    Returns
    -------
    Plane or SpatialGrid
        Geometry carrying the selected model and its physical parameter scales.
    """
    if transform == "rigid":
        return grid
    dimensions = grid.basis.shape[1]
    affine = position_to_world(fixed)
    sizes = np.array([fixed.sizes[d] for d in ("k", "j", "i")])
    ell = np.linalg.norm(affine[:3, :3][:, sizes > 1], axis=0).min()
    scales = np.full(dimensions, ell)
    if transform == "affine":
        basis = getattr(grid, "slice_basis", None)
        basis = grid.basis if basis is None else basis
        lattice = basis.T @ affine[:3, :3]
        center_offset = basis.T @ (
            affine[:3, 3] + affine[:3, :3] @ ((sizes - 1) / 2) - grid.center
        )
        variance = lattice**2 @ ((sizes.astype(float) ** 2 - 1) / 12) + center_offset**2
        scales = np.r_[np.tile(ell / np.sqrt(variance), dimensions), scales]
    return replace(
        grid,
        transform=transform,
        parameter_scale=grid.parameter_scale.new_tensor(scales),
    )


def build_intrinsic_linear(parameters, grid):
    """Return centered intrinsic linear maps and shifts for non-rigid models.

    Parameters
    ----------
    parameters : torch.Tensor
        Physical translation or affine parameters, with frames on the first axis.
    grid : Plane or SpatialGrid
        Geometry and selected model, including mixed-dimensional completion.

    Returns
    -------
    linear : torch.Tensor
        Batched linear maps in the model's intrinsic coordinates.
    displacement : torch.Tensor
        Batched physical center displacements in millimetres.
    """
    dimensions = grid.basis.shape[1]
    if grid.transform == "affine" and getattr(grid, "slice_basis", None) is not None:
        basis = grid.resident_slice_basis.to(parameters)
        tangents = basis + parameters[:, :6].reshape(-1, 3, 2)
        normal = torch.linalg.cross(tangents[:, :, 0], tangents[:, :, 1])
        normal = normal / torch.linalg.vector_norm(
            normal, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(parameters.dtype).tiny)
        frame = torch.cat(
            (basis, torch.linalg.cross(basis[:, 0], basis[:, 1])[:, None]), dim=1
        )
        linear = torch.cat((tangents, normal[:, :, None]), dim=-1) @ frame.T
        if grid.inverse_slice_affine:
            # Constrain the slice-to-volume map, even when the requested pull is its inverse.
            linear = torch.linalg.inv_ex(linear).inverse
        return linear, parameters[:, -3:]
    identity = torch.eye(dimensions, dtype=parameters.dtype, device=parameters.device)
    linear = identity.expand(len(parameters), -1, -1)
    if grid.transform == "affine":
        linear = linear + parameters[:, : dimensions**2].reshape(
            -1, dimensions, dimensions
        )
    return linear, parameters[:, -dimensions:]


def check_affine_parameters(parameters, grid):
    """Check map finiteness, nonsingularity and orientation without scalar reads.

    Normalize before the small determinant to avoid overflow/underflow. A
    normalized determinant <= 1e-8 is treated as numerically singular.

    Parameters
    ----------
    parameters : torch.Tensor
        Batched physical model parameters.
    grid : Plane or SpatialGrid
        Resident geometry and selected non-rigid linear model.

    Returns
    -------
    valid : torch.Tensor
        Per-frame finite, nonsingular-map mask.
    orientation : torch.Tensor
        Per-frame sign of the normalized linear determinant.
    """
    linear, _ = build_intrinsic_linear(parameters.detach(), grid)
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
    """Copy and validate common homogeneous world-pull seed structure.

    Parameters
    ----------
    transforms : array-like
        One (4, 4) pull or a (count, 4, 4) array of world z/y/x pulls.
        Bare matrices use mm. Labelled spatial_units convert translations to mm;
        the dimensionless linear part and homogeneous row remain unchanged.
    count : int
        Number of moving frames after broadcasting a shared matrix.

    Returns
    -------
    numpy.ndarray
        Finite float64 homogeneous matrices, shaped (count, 4, 4).

    Raises
    ------
    TypeError
        If matrix values are not real numeric data.
    ValueError
        If shape, finiteness or homogeneous last rows are invalid.
    """
    values = np.asarray(transforms)
    if values.dtype.kind not in "iuf":
        raise TypeError("initial_transform must contain real numeric matrices.")
    values = np.array(values, dtype=np.float64, copy=True)
    try:
        scale = _length_scale_to_mm(
            getattr(transforms, "attrs", {}).get("spatial_units", "mm")
        )
    except ValueError as error:
        raise ValueError(f"initial_transform spatial_units: {error}") from error
    if values.shape[-2:] == (4, 4) and scale != 1:
        with np.errstate(over="ignore", invalid="ignore"):
            values[..., :3, 3] *= scale
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


def recover_linear_parameters(values, grid):
    """Recover the complete optimized translation/affine seed, never a frozen chain.

    Parameters
    ----------
    values : numpy.ndarray
        Validated homogeneous world pull matrices, shaped (batch, 4, 4).
    grid : Plane or SpatialGrid
        Resident geometry and selected translation or affine model.

    Returns
    -------
    numpy.ndarray
        Complete physical seed parameters in the model's intrinsic coordinates.

    Raises
    ------
    ValueError
        If the seed is singular or cannot be represented by the selected model.
    """
    linear = values[:, :3, :3]
    translation = (
        np.einsum("bij,j->bi", linear - np.eye(3), grid.center) + values[:, :3, 3]
    )
    dimensions = grid.basis.shape[1]
    intrinsic = grid.basis.T @ linear @ grid.basis
    local_translation = translation @ grid.basis
    if grid.transform == "affine" and getattr(grid, "slice_basis", None) is not None:
        basis = grid.slice_basis
        if grid.inverse_slice_affine:
            try:
                tangent_linear = np.linalg.inv(linear)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "Slice-to-volume affine seed must be nonsingular."
                ) from error
        else:
            tangent_linear = linear
        tangents = tangent_linear @ basis
        parameters = np.column_stack(
            ((tangents - basis).reshape(len(values), -1), translation)
        )
        reconstructed, _ = build_intrinsic_linear(torch.from_numpy(parameters), grid)
        valid, _ = check_affine_parameters(torch.from_numpy(parameters), grid)
        if not valid.all():
            raise ValueError(
                "Slice-to-volume affine seed must have independent tangent columns."
            )
        if not np.allclose(linear, reconstructed.numpy(), rtol=0, atol=1e-6):
            raise ValueError(
                "Slice-to-volume affine seed has unidentifiable normal scale/shear/reflection; use a unit orthogonal normal completion."
            )
        return parameters
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
                "Slice initial_transform must preserve the fixed plane and its normal after accounting for seeded plane placement; use align_planes when centering cannot correct tilt."
            )
    if grid.transform == "translation":
        return local_translation
    parameters = np.column_stack(
        ((intrinsic - np.eye(dimensions)).reshape(len(values), -1), local_translation)
    )
    valid, _ = check_affine_parameters(torch.from_numpy(parameters), grid)
    if not valid.all():
        raise ValueError(
            "Affine initial_transform must be nonsingular and numerically well-conditioned (normalized |determinant| > 1e-8)."
        )
    return parameters


def build_linear_world_matrices(parameters, grid):
    """Export full centered world pulls, including affine scale/shear/reflection.

    Parameters
    ----------
    parameters : array-like
        Batched physical translation or affine parameters.
    grid : Plane or SpatialGrid
        Centered model geometry and world embedding.

    Returns
    -------
    numpy.ndarray
        Float64 homogeneous world z/y/x pulls, shaped (batch, 4, 4).
    """
    values = np.asarray(parameters, dtype=np.float64)
    dimensions = grid.basis.shape[1]
    if grid.transform == "affine" and getattr(grid, "slice_basis", None) is not None:
        linear, _ = build_intrinsic_linear(torch.from_numpy(values), grid)
        matrices = np.broadcast_to(np.eye(4), (len(values), 4, 4)).copy()
        matrices[:, :3, :3] = linear.numpy()
        matrices[:, :3, 3] = values[:, -3:] - np.einsum(
            "bij,j->bi", matrices[:, :3, :3] - np.eye(3), grid.center
        )
        return matrices
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


def build_linear_diagnostics(parameters, grid):
    """Report actual linear coefficients and physical center shifts.

    Parameters
    ----------
    parameters : numpy.ndarray
        Batched physical translation or affine parameters.
    grid : Plane or SpatialGrid
        Geometry defining intrinsic axes and model parameterization.

    Returns
    -------
    dict
        Labeled per-frame coefficients and millimetre center displacements,
        suitable for constructing an xarray diagnostic dataset.
    """
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
