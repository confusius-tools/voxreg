"""Check direct ITK comparison geometry and the process-tree memory monitor."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("itk")
spec = importlib.util.spec_from_file_location(
    "benchmark_example",
    Path(__file__).parents[1] / "examples/benchmark_registration.py",
)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_direct_itk_recovers_known_motion(recording_factory, tmp_path):
    data, truth, affine = recording_factory()
    path = tmp_path / "input.npz"
    np.savez(
        path,
        values=data.values,
        voxel_to_world=data.xindexes["z"].voxel_to_world,
        **{dim: data[dim].values for dim in data.dims},
    )
    benchmark.initialize_itk(str(path))
    registered, matrices, steps = benchmark.run_itk(data.values)
    j, i = np.mgrid[6:58:3, 6:74:3]
    points = affine @ np.stack((np.zeros_like(j), j, i, np.ones_like(j))).reshape(4, -1)
    error = np.einsum("bij,jp->bip", matrices - truth, points)[:, :3]
    rms = np.sqrt(np.mean(np.sum(error**2, axis=1), axis=1))
    assert rms.max() < 0.25 * min(affine[1, 1], affine[2, 2])
    assert_allclose(registered[0], data.values[0], atol=0)
    assert np.all((steps[1:] > 0) & (steps[1:] <= 100))
    benchmark.initialize_itk(str(path), work_units=8)
    _, threaded_matrix, _ = benchmark.register_itk_frame(1)
    assert_allclose(threaded_matrix, matrices[1], atol=1e-9)


@pytest.mark.parametrize("device,batch_size", [("cpu", 1), ("cuda", 2)])
def test_fireants_recovers_known_motion(recording_factory, device, batch_size):
    import torch
    from scipy.ndimage import map_coordinates

    pytest.importorskip("fireants")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    data, truth, affine = recording_factory()
    registered, matrices, steps = benchmark.run_fireants(
        data.values, affine, device, batch_size
    )
    j, i = np.indices(data.shape[-2:])
    grid = np.stack((np.zeros_like(j), j, i, np.ones_like(j))).reshape(4, -1)
    points = affine @ grid
    error = np.einsum("bij,jp->bip", matrices - truth, points)[:, :3]
    rms = np.sqrt(np.mean(np.sum(error**2, axis=1), axis=1))
    assert rms.max() < 0.25 * min(affine[1, 1], affine[2, 2])
    assert_allclose(registered[0], data.values[0], atol=0)
    assert_allclose(steps, [0, 100, 100], atol=0)
    for frame in (1, 2):
        mapped = np.linalg.inv(affine) @ matrices[frame] @ affine @ grid
        expected = map_coordinates(
            data.values[frame, 0], mapped[1:3], order=1, mode="nearest"
        ).reshape(j.shape)
        assert_allclose(
            registered[frame, 0, 6:-6, 6:-6], expected[6:-6, 6:-6], atol=2e-4
        )


def test_input_rejects_complex_power(tmp_path):
    path = tmp_path / "complex.npz"
    np.savez(
        path,
        values=np.arange(8).reshape(2, 1, 2, 2) + 1j,
        voxel_to_world=np.eye(4),
        time=[0, 1],
        k=[0],
        j=[0, 1],
        i=[0, 1],
    )
    with pytest.raises(ValueError, match="must be real"):
        benchmark.load_input(path)


@pytest.mark.skipif(
    not Path("/proc/self/smaps_rollup").exists(), reason="Linux memory monitor"
)
def test_memory_monitor_includes_child_process():
    script = "import sys; data=bytearray(32*1024**2); print('ready', flush=True); sys.stdin.read()"
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        child_rss, child_pss, _ = benchmark.tree_memory(process.pid)
        root_rss = benchmark.rss_kib(os.getpid())
        rss, pss, pids = benchmark.tree_memory(os.getpid())
        assert process.pid in pids
        assert child_rss > 32 * 1024 and child_pss > 32 * 1024
        assert rss >= root_rss + child_rss - 2048
        assert pss >= child_pss
    finally:
        process.communicate(timeout=10)
