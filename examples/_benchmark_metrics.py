"""Physical truth scoring shared by resident and native benchmark configurations."""

import numpy as np
from _benchmark_data import SPACING, _grid, _matrix, _truth


def _point_rms(mapped, points, amplitude, model, shape):
    truth = _truth(points, amplitude, model, shape)
    spacing = SPACING[1:] if shape[0] == 1 else SPACING
    return float(
        np.sqrt(np.mean(np.sum((mapped - truth) ** 2, axis=0))) / spacing.min()
    )


def _matrix_rms(matrices, amplitudes, shape):
    """Exact all-voxel RMS on the centered benchmark lattice, without dense exports."""
    return _linear_rms(matrices, amplitudes, "rigid", shape)


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


def _native_motion_rms(run, shape, amplitudes):
    import nibabel as nib
    import SimpleITK as sitk

    image = nib.load(run / "reference-mean.nii.gz")
    values = np.asarray(image.dataobj, dtype=np.float64)
    dimension = values.ndim
    if dimension not in (2, 3) or not np.isfinite(values).all() or values.sum() <= 0:
        raise ValueError("Invalid native mean-reference moment center")
    center_voxel = np.array(
        [
            np.sum(
                values.sum(axis=tuple(j for j in range(dimension) if j != i))
                * np.arange(values.shape[i])
            )
            / values.sum()
            for i in range(dimension)
        ]
    )
    affine = image.affine.copy()
    affine[:2] *= -1  # RAS header -> LPS physical coordinates.
    center = affine[:dimension, :dimension] @ center_voxel + affine[:dimension, 3]
    parameters = np.loadtxt(
        run / "ants_MOCOparams.csv", delimiter=",", skiprows=1, ndmin=2
    )
    if (
        parameters.shape != (len(amplitudes), 5 if dimension == 2 else 8)
        or not np.isfinite(parameters).all()
    ):
        raise ValueError("Unsupported or incomplete native Euler motion parameters")
    matrices = []
    for row in parameters:
        transform = (
            sitk.Euler2DTransform() if dimension == 2 else sitk.Euler3DTransform()
        )
        if dimension == 3:
            transform.SetComputeZYX(False)
        transform.SetCenter(tuple(center))
        transform.SetParameters(tuple(row[2:]))
        rotation = np.array(transform.GetMatrix()).reshape(dimension, dimension)
        matrix = np.eye(4)
        matrix[:dimension, :dimension] = rotation
        matrix[:dimension, 3] = (
            center + np.array(transform.GetTranslation()) - rotation @ center
        )
        matrices.append(matrix)
    inverse_reference = np.linalg.inv(matrices[0])
    permutation = np.eye(4)[[2, 1, 0, 3]]
    relative = [
        permutation @ matrix @ inverse_reference @ permutation for matrix in matrices
    ]
    return _matrix_rms(relative, amplitudes, shape)


def _native_syn_rms(directory, run, shape, amplitudes):
    """Score fixed-to-moving LPS-mm pulls exported by antsMotionCorr -w 1."""
    import nibabel as nib

    dimension = 2 if shape[0] == 1 else 3
    fixed = nib.load(directory / "fixed.nii")
    image = nib.load(run / "ants_Warp.nii.gz")
    spatial = tuple(shape[:0:-1]) if dimension == 2 else tuple(shape[::-1])
    expected = (
        (*spatial, len(amplitudes), 1, 3)
        if dimension == 2
        else (*spatial, len(amplitudes), 4)
    )
    if (
        image.shape != expected
        or image.get_data_dtype() != np.dtype("float64")
        or image.header.get_intent()[0] != "vector"
        or not np.allclose(
            image.affine[:3, :dimension],
            fixed.affine[:3, :dimension],
            rtol=0,
            atol=1e-6,
        )
        or not np.allclose(image.affine[:3, 3], fixed.affine[:3, 3], rtol=0, atol=1e-6)
        or (
            len(amplitudes) > 1
            and not np.isclose(
                image.header.get_zooms()[dimension], 0.1, rtol=0, atol=1e-6
            )
        )
    ):
        raise ValueError("Invalid native SyN displacement shape/geometry/time/intent")
    # Decode gzip once, rather than re-inflating preceding frames for every query.
    values = np.asarray(image.dataobj)
    points = _grid(shape)
    errors = []
    for index, amplitude in enumerate(amplitudes):
        field = values[:, :, index, 0] if dimension == 2 else values[:, :, :, index]
        if not np.isfinite(field).all() or np.any(field[..., dimension] != 0):
            raise ValueError("Nonfinite displacement or nonzero time displacement")
        # Vector-intent components remain LPS x/y[/z]; only the header is RAS.
        displacement = (
            field[..., :dimension]
            .transpose(*range(dimension - 1, -1, -1), dimension)
            .reshape(-1, dimension)
            .T[::-1]
        )
        mapped = points.copy()
        mapped[3 - dimension :] += displacement
        errors.append(_point_rms(mapped, points, amplitude, "syn", shape))
    return errors


def _native_pair_rms(directory, run, shape):
    """ANTs image pull: affine after the fixed-domain displacement, in LPS mm."""
    import json

    import SimpleITK as sitk

    truth = json.loads((directory / "truth.json").read_text())
    linear = np.array(truth["linear_matrix"])
    transform = sitk.ReadTransform(str(run / "ants_0GenericAffine.mat"))
    rotation = np.array(transform.GetMatrix()).reshape(3, 3)
    center = np.array(transform.GetCenter())
    translation = np.array(transform.GetTranslation()) + center - rotation @ center
    field_image = sitk.ReadImage(str(run / "ants_1Warp.nii.gz"))
    fixed_image = sitk.ReadImage(str(directory / "fixed.nii"))
    for getter in ("GetOrigin", "GetSpacing", "GetDirection"):
        if not np.allclose(
            getattr(field_image, getter)(),
            getattr(fixed_image, getter)(),
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError("Native displacement is not on the fixed physical lattice")
    field = sitk.GetArrayFromImage(field_image)
    if field.shape != (*shape, 3) or not np.isfinite(field).all():
        raise ValueError("Invalid native displacement field")
    points = _grid(shape)
    mapped_xyz = (
        rotation @ (points[::-1] + field.reshape(-1, 3).T) + translation[:, None]
    )
    nonlinear = _truth(points, truth["nonlinear_amplitude"], "syn", shape)
    expected = linear[:3, :3] @ nonlinear + linear[:3, 3, None]
    return [
        float(
            np.sqrt(np.mean(np.sum((mapped_xyz[::-1] - expected) ** 2, axis=0)))
            / SPACING.min()
        )
    ]
