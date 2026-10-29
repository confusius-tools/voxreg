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
