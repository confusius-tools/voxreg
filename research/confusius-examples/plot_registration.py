"""Shared research plotting; never selects a backend or imports an experiment runner."""

import numpy as np
from scipy.ndimage import map_coordinates

from voxreg import position_to_world


def sample(matrix, fixed, moving):
    af, am = position_to_world(fixed), position_to_world(moving)
    points = af @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size)))
    )
    coordinates = (np.linalg.inv(am) @ matrix @ points)[:3]
    valid = (
        (coordinates >= -1e-5)
        & (coordinates <= np.array(moving.shape)[:, None] - 1 + 1e-5)
    ).all(axis=0)
    values = map_coordinates(
        moving.values, coordinates, order=1, mode="nearest", prefilter=False
    )
    return values.reshape(fixed.shape), valid.reshape(fixed.shape), coordinates


def display_sample(matrix, fixed, moving):
    """Never present nearest-edge extrapolation as an observed anatomical match."""
    values, valid, coordinates = sample(matrix, fixed, moving)
    return np.where(valid, values, np.nan), valid, coordinates


def supported_composites(reference, images, title, *, reference_is_red=False):
    import matplotlib.pyplot as plt

    assert not reference_is_red
    fig, axes = plt.subplots(
        1, len(images), figsize=(5 * len(images), 5), constrained_layout=True
    )
    cyan = np.clip((np.asarray(reference)[0].astype(float) + 30) / 30, 0, 1)
    for ax, (name, values) in zip(np.atleast_1d(axes), images.items(), strict=True):
        red = np.clip((np.asarray(values)[0].astype(float) + 30) / 30, 0, 1)
        valid = np.isfinite(red) & np.isfinite(cyan)
        rgb = np.stack((red, cyan, cyan), axis=-1)
        rgb[~valid] = 0.25
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(name)
        ax.set_axis_off()
    fig.suptitle(
        title
        + " — template red / recording cyan; unsupported pixels gray; shared −30…0dB"
    )
