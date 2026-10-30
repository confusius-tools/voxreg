"""Bounded analytic linear-model qualification; not a timing benchmark.

Run sequentially with one CPU thread. ITK uses native double/GD physical-shift
scales; voxreg keeps mixed precision/Adam RMS scales. Same inputs and fixed160
budgets do not imply equal optimizer trajectories. JSON retains every gate result.
"""

import hashlib
import json
from pathlib import Path

import itk
import numpy as np
import torch
from confusius import create_voxeldata
from scipy.ndimage import map_coordinates

from voxreg import position_to_world, register


def fixture(planar, model):
    """Orthogonal independent grids sample the same continuous Gaussian anatomy."""
    rng = np.random.default_rng(871)
    dimensions = 2 if planar else 3
    basis = np.eye(3)[:, [2, 1]] if planar else np.eye(3)
    center = np.array([10.0, -20.0, 30.0])
    linear = np.eye(dimensions)
    if model == "affine":
        linear += (
            np.array([[0.04, 0.035], [-0.02, -0.03]])
            if planar
            else np.array(
                [[0.04, 0.025, -0.015], [-0.01, -0.03, 0.02], [0.015, 0, 0.025]]
            )
        )
    elif model == "rigid":
        if planar:
            c, s = np.cos(0.025), np.sin(0.025)
            linear = np.array([[c, -s], [s, c]])
        else:
            from scipy.spatial.transform import Rotation

            linear = Rotation.from_rotvec([0.02, -0.015, 0.01]).as_matrix()[::-1, ::-1]
    shift = np.array([0.1, -0.08]) if planar else np.array([0.1, -0.08, 0.06])
    truth = np.eye(4)
    truth[:3, :3] += basis @ (linear - np.eye(dimensions)) @ basis.T
    truth[:3, 3] = basis @ shift - (truth[:3, :3] - np.eye(3)) @ center
    locations = rng.uniform(-1.4, 1.4, (16, dimensions))
    widths = rng.uniform(0.3, 0.65, (16, dimensions))
    amplitudes = rng.uniform(0.5, 2, 16)
    images = []
    shapes = [(1, 25, 29), (1, 33, 37)] if planar else [(17, 19, 21), (25, 27, 29)]
    for index, shape in enumerate(shapes):
        affine = np.eye(4)
        affine[:3, :3] = np.diag([0.24, 0.21, 0.18])
        affine[:3, 3] = center - affine[:3, :3] @ ((np.array(shape) - 1) / 2)
        points = affine @ np.vstack(
            (np.indices(shape).reshape(3, -1), np.ones((1, np.prod(shape))))
        )
        if index:
            points = np.linalg.inv(truth) @ points
        local = (points[:3] - center[:, None]).T @ basis
        values = 0.05 + np.sum(
            amplitudes
            * np.exp(
                -0.5 * np.sum(((local[:, None] - locations) / widths) ** 2, axis=-1)
            ),
            axis=1,
        )
        images.append(
            create_voxeldata(
                values.reshape(shape), dims=("k", "j", "i"), voxel_to_world=affine
            )
        )
    return *images, truth


def fit_itk(fixed, moving, planar, model):
    axes = [2, 1] if planar else [2, 1, 0]
    dimension = len(axes)

    def image(data):
        affine = position_to_world(data)
        out = itk.image_from_array(
            np.ascontiguousarray(data.values[0] if planar else data.values)
        )
        out.SetSpacing(tuple(np.linalg.norm(affine[:3, :3], axis=0)[axes]))
        out.SetOrigin(tuple(affine[axes, 3]))
        return out

    f, m = image(fixed), image(moving)
    metric = itk.CorrelationImageToImageMetricv4[type(f), type(m)].New()
    metric.SetUseFixedImageGradientFilter(False)
    metric.SetUseMovingImageGradientFilter(False)
    cls = {
        "translation": itk.TranslationTransform,
        "rigid": itk.Euler2DTransform if planar else itk.Euler3DTransform,
        "affine": itk.AffineTransform,
    }[model]
    transform = cls[itk.D].New() if model == "rigid" else cls[itk.D, dimension].New()
    if model != "translation":
        transform.SetCenter(tuple(np.array([10.0, -20.0, 30.0])[axes]))
    optimizer = itk.GradientDescentOptimizerv4Template[itk.D].New(
        LearningRate=0.1,
        NumberOfIterations=160,
        MinimumConvergenceValue=1e-6,
        ConvergenceWindowSize=161,
        DoEstimateLearningRateOnce=False,
        DoEstimateLearningRateAtEachIteration=False,
        ReturnBestParametersAndValue=True,
    )
    optimizer.SetScalesEstimator(
        itk.RegistrationParameterScalesFromPhysicalShift[type(metric)].New(
            Metric=metric, TransformForward=True
        )
    )
    registration = itk.ImageRegistrationMethodv4[type(f), type(m)].New(
        FixedImage=f,
        MovingImage=m,
        Metric=metric,
        Optimizer=optimizer,
        InitialTransform=transform,
        InPlace=True,
    )
    registration.SetNumberOfLevels(1)
    registration.SetShrinkFactorsPerLevel([1])
    registration.SetSmoothingSigmasPerLevel([0.0])
    registration.Update()
    matrix = np.eye(4)
    if model != "translation":
        matrix[np.ix_(axes, axes)] = np.array(transform.GetMatrix())
    matrix[axes, 3] = np.array(transform.GetOffset())
    return matrix, int(optimizer.GetCurrentIteration())


def run():
    torch.set_num_threads(1)
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(1)
    results = {
        "protocol": "analytic-independent-orthogonal-grids/fixed160/no-fitting-scaling",
        "precision": {
            "voxreg": "float32 fitting/interpolation; float64 geometry/state/NCC",
            "itk": "double images/native GD",
        },
        "versions": {"torch": torch.__version__, "itk": itk.Version.GetITKVersion()},
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *Path("src/voxreg").glob("*.py")]
        },
        "cases": [],
    }
    for planar in [True, False]:
        for model in ["translation", "rigid", "affine"]:
            fixed, moving, truth = fixture(planar, model)
            af, am = position_to_world(fixed), position_to_world(moving)
            points = af @ np.vstack(
                (np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size)))
            )
            ell = np.linalg.norm(af[:3, :3][:, np.array(fixed.shape) > 1], axis=0).min()
            record = {
                "dimension": 2 if planar else 3,
                "transform": model,
                "input_sha256": hashlib.sha256(
                    fixed.values.tobytes()
                    + moving.values.tobytes()
                    + af.tobytes()
                    + am.tobytes()
                    + truth.tobytes()
                ).hexdigest(),
                "truth": truth.tolist(),
                "runs": {},
            }
            matrices = {}
            for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
                r = register(
                    moving,
                    fixed,
                    return_transforms=True, transform=model,
                    initialization="center_geometry",
                    intensity_scaling="none",
                    max_iterations=160,
                    convergence_window_size=161,
                    device=device,
                    progress=False,
                )
                matrices[device] = r.transforms.values
                record["runs"][device] = {
                    "updates": int(r.diagnostics.n_iterations),
                    "status": str(r.diagnostics.status.values),
                }
            matrices["itk"], updates = fit_itk(fixed, moving, planar, model)
            record["runs"]["itk"] = {
                "updates": updates,
                "status": "native-fixed-budget",
            }
            coordinates = {
                k: (np.linalg.inv(am) @ matrix @ points)[:3]
                for k, matrix in matrices.items()
            }
            common = np.ones(fixed.size, dtype=bool)
            for c in coordinates.values():
                common &= ((c >= 0) & (c <= np.array(moving.shape)[:, None] - 1)).all(
                    axis=0
                )
            y = fixed.values.reshape(-1)[common]
            y = y - y.mean()
            for backend, matrix in matrices.items():
                delta = np.linalg.norm(((matrix - truth) @ points)[:3], axis=0) / ell
                x = map_coordinates(
                    moving.values,
                    coordinates[backend],
                    order=1,
                    mode="nearest",
                    prefilter=False,
                )[common]
                x = x - x.mean()
                rms = float(np.sqrt(np.mean(delta**2)))
                record["runs"][backend].update(
                    matrix=matrix.tolist(),
                    truth_rms_pixels=rms,
                    truth_gate_pass=rms < 0.25,
                    ncc2=float((x @ y) ** 2 / ((x @ x) * (y @ y))),
                )
            if "cuda" in matrices:
                maximum = float(
                    np.linalg.norm(
                        ((matrices["cpu"] - matrices["cuda"]) @ points)[:3], axis=0
                    ).max()
                    / ell
                )
                record.update(
                    cpu_cuda_max_pointwise_pixels=maximum,
                    precision_gate_pass=maximum < 0.001,
                )
            record["joint_common_fraction"] = float(common.mean())
            results["cases"].append(record)
            print(
                model,
                record["dimension"],
                {k: v["truth_rms_pixels"] for k, v in record["runs"].items()},
                record.get("cpu_cuda_max_pointwise_pixels"),
                flush=True,
            )
    Path("research/linear/results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    assert all(
        r["truth_gate_pass"] for case in results["cases"] for r in case["runs"].values()
    )
    assert all(case.get("precision_gate_pass", True) for case in results["cases"])


if __name__ == "__main__":
    run()
