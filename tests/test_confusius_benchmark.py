"""The example benchmark must retain native settings and recover public poses."""

import importlib.util
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

PATH = (
    Path(__file__).resolve().parents[1]
    / "research/confusius-examples/benchmark_examples.py"
)
spec = importlib.util.spec_from_file_location("example_benchmark", PATH)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_public_motion_table_matrix_roundtrip():
    import confusius as cf
    from confusius.registration.motion import create_motion_dataframe

    rng = np.random.default_rng(19)
    matrices = np.broadcast_to(np.eye(4), (5, 4, 4)).copy()
    matrices[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, 0.3, (5, 3))).as_matrix()
    matrices[:, :3, 3] = rng.normal(size=(5, 3))
    fixed = cf.create_voxeldata(
        np.ones((3, 4, 5)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    )
    frame = create_motion_dataframe([np.asarray(matrix) for matrix in matrices], fixed)
    assert_allclose(benchmark.matrices_from_motion(frame), matrices, atol=1e-14, rtol=0)


def test_example_options_and_explicit_cpu_budget():
    example = benchmark.options("series", "example", "rigid", "confusius")
    assert example["learning_rate"] == 1.0
    assert example["n_jobs"] == 8
    assert example["moving_intensity_scaling"] == "db"
    assert "convergence_window_size" not in example
    fixed = benchmark.options("pair", "fixed400", "affine", "confusius")
    assert fixed["number_of_iterations"] == 400
    assert fixed["convergence_window_size"] == 401
    assert fixed["sitk_threads"] == 8
    assert (
        benchmark.options("series", "example", "rigid", "voxreg-cuda120")["batch_size"]
        == 120
    )
    assert (
        benchmark.options("series", "fixed100", "rigid", "voxreg-cpu")[
            "convergence_window_size"
        ]
        == 101
    )


def test_unusable_scores_are_counted_not_silently_dropped():
    summary = benchmark.distribution([None, 0.5, 0.9])
    assert summary["n_invalid"] == 1
    assert summary["n_valid"] == 2
    assert summary["median"] == 0.7
    assert benchmark.distribution([None])["median"] is None
