"""Exercise the optional standalone example's I/O and scoring boundaries."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from voxreg import position_to_world

pytest.importorskip("nibabel")
pytest.importorskip("matplotlib")
spec = importlib.util.spec_from_file_location(
    "motion_example",
    Path(__file__).parents[1] / "examples/notebook_motion_correction.py",
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int16, np.int64, np.uint64]
)
def test_nifti_round_trip_and_window(recording_factory, tmp_path, dtype):
    data, _, affine = recording_factory(oblique=True, strided=True)
    if np.issubdtype(dtype, np.integer):
        data = data.copy(data=np.rint(data.values * 100).astype(dtype))
    else:
        data = data.astype(dtype)
        if dtype == np.float64:
            data.values[:] += 1e-10
    path = tmp_path / "recording.nii.gz"
    example.save_nifti(path, data)
    restored = example.load_recording(path)
    assert_array_equal(restored.values, data.values)
    assert restored.dtype == data.dtype
    assert_array_equal(restored.time, data.time)
    assert_allclose(position_to_world(restored), affine, atol=5e-6)
    assert_array_equal(
        example.load_recording(path, slice(1, 3)).values, data.values[1:]
    )
    sidecar = tmp_path / "recording.json"
    sidecar.write_text(json.dumps({"VolumeTiming": [1, 2]}))
    with pytest.raises(ValueError, match="VolumeTiming"):
        example.load_recording(path)
    sidecar.write_text(json.dumps({"VolumeTiming": [1, 1, 2]}))
    with pytest.raises(ValueError, match="increasing"):
        example.load_recording(path)
    sidecar.unlink()
    restored_header_time = example.load_recording(path)
    assert_allclose(restored_header_time.time, data.time, atol=1e-6)


def test_independent_scoring_sampler(recording_factory):
    data, truth, affine = recording_factory(oblique=True, strided=True)
    values = data.values[:, 0]
    sampled, valid = example.sample_matrices(values, truth, affine)
    for frame in (1, 2):
        mask = valid[frame]
        assert np.corrcoef(sampled[frame][mask], values[0][mask])[0, 1] > 0.999
    assert_allclose(sampled[0], values[0], atol=1e-12)
    assert valid[0].all()
    outside = truth.copy()
    outside[:, :3, 3] += 1e4
    _, valid = example.sample_matrices(values, outside, affine)
    assert not valid.any()
