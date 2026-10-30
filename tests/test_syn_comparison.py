"""Zero-update ownership checks for the research comparison's physical adapters."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from voxreg import position_to_world


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("backend", ["itk", "fireants"])
def test_reference_zero_updates_preserve_complete_seed(
    nonlinear_pair_factory, planar, backend
):
    pytest.importorskip(backend)
    path = Path(__file__).parents[1] / "research/syn/compare.py"
    spec = importlib.util.spec_from_file_location("syn_comparison", path)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    shape = (40, 48) if planar else (32, 32, 32)
    fixed, moving, _, _ = nonlinear_pair_factory(planar, shape=shape, sheared=False)
    af = position_to_world(fixed)
    if planar:
        axis = np.cross(af[:3, 2], af[:3, 1])
        axis = axis / np.linalg.norm(axis) * 0.03
    else:
        axis = np.array([0.02, -0.03, 0.01])
    linear = Rotation.from_rotvec(axis).as_matrix()
    center = (af @ np.r_[(np.array(fixed.shape) - 1) / 2, 1])[:3]
    seed = np.eye(4)
    seed[:3, :3] = linear
    seed[:3, 3] = (np.eye(3) - linear) @ center + 0.1 * af[
        :3, 1 if planar else 0
    ] / np.linalg.norm(af[:3, 1 if planar else 0])
    function = comparison.fit_itk if backend == "itk" else comparison.fit_fireants
    options = {"threads": 1} if backend == "itk" else {"threads": 1, "device": "cpu"}
    fields, *_ = function(moving, fixed, seed, 0, **options)
    points = af @ np.vstack(
        (np.indices(fixed.shape).reshape(3, -1), np.ones((1, fixed.size)))
    )
    expected = ((seed @ points) - points)[:3].T.reshape(*fixed.shape, 3)
    ell = np.linalg.norm(af[:3, 1 if planar else 0 : 3], axis=0).min()
    assert np.linalg.norm(fields[0] - expected, axis=-1).max() / ell < 0.001
