"""Independent numerical oracles shared by the package's registration tests."""

import numpy as np
from confusius import create_voxeldata
from scipy.ndimage import map_coordinates

from voxreg import position_to_world


def voxel(values, affine):
    return create_voxeldata(values, dims=("k", "j", "i"), voxel_to_world=affine)


def sample_mixed(data, target, matrix):
    """Float64 SciPy pull with the mixed-grid half-open finite-voxel support."""
    points = position_to_world(target) @ np.vstack(
        (np.indices(target.shape).reshape(3, -1), np.ones((1, target.size)))
    )
    native = (np.linalg.inv(position_to_world(data)) @ matrix @ points)[:3]
    valid = ((native >= -0.5) & (native < np.array(data.shape)[:, None] - 0.5)).all(
        axis=0
    )
    values = map_coordinates(
        np.asarray(data.values, dtype=np.float64),
        native,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return values.reshape(target.shape), valid.reshape(target.shape)


def compose_displacements_reference(outer, inner, lattice):
    """Independent SciPy composition of physical fields, with finite-cell support."""
    shape = inner.shape[2:]
    offsets = np.einsum("ij,bj...->bi...", np.linalg.inv(lattice), inner)
    mapped = np.indices(shape)[None] + offsets
    valid = (
        (mapped >= -0.5)
        & (mapped < np.array(shape).reshape(1, -1, *([1] * len(shape))) - 0.5)
    ).all(axis=1)
    sampled = np.array(
        [
            [
                map_coordinates(
                    component, coordinates, order=1, mode="nearest", prefilter=False
                )
                for component in frame
            ]
            for frame, coordinates in zip(outer, mapped, strict=True)
        ]
    )
    return inner + np.where(valid[:, None], sampled, 0)


def itk_displacement_reference(field, lattice):
    """Native ITK physical transform, origin-free like the field operations.

    Planes use intrinsic x/y components; volumes convert world z/y/x to ITK x/y/z.
    The lattice maps array-axis offsets to physical components, including shear.
    """
    import SimpleITK as sitk

    dimensions = field.shape[0]
    rows = np.arange(dimensions)[::-1] if dimensions == 3 else np.arange(dimensions)
    physical_lattice = lattice[rows, ::-1]
    spacing = np.linalg.norm(physical_lattice, axis=0)
    image = sitk.GetImageFromArray(np.moveaxis(field[rows], 0, -1), isVector=True)
    image.SetSpacing(spacing.tolist())
    image.SetDirection((physical_lattice / spacing).ravel().tolist())
    return sitk.DisplacementFieldTransform(image)
