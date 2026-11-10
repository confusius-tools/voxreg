"""Physical truth scoring shared by resident registration backends."""

import numpy as np
from _benchmark_data import SPACING, _matrix, _truth


def _point_rms(mapped, points, amplitude, model, shape):
    truth = _truth(points, amplitude, model, shape)
    spacing = SPACING[1:] if shape[0] == 1 else SPACING
    return float(
        np.sqrt(np.mean(np.sum((mapped - truth) ** 2, axis=0))) / spacing.min()
    )


def _linear_rms(matrices, amplitudes, model, shape):
    variance = (np.array(shape, dtype=float) ** 2 - 1) * SPACING**2 / 12
    errors = []
    for matrix, amplitude in zip(matrices, amplitudes, strict=True):
        difference = matrix - _matrix(model, amplitude, shape)
        errors.append(
            float(
                np.sqrt(
                    (difference[:3, :3] ** 2 * variance).sum()
                    + (difference[:3, 3] ** 2).sum()
                )
                / SPACING.min()
            )
        )
    return errors
