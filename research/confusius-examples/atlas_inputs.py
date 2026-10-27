"""Load the cached atlas exploration pair without selecting a plotting backend."""

from pathlib import Path

import confusius as cf
import numpy as np


def inputs():
    root = Path.home() / ".cache/confusius"
    paths = [
        root
        / "pepe-mariani-2026-template/1.0.0/pepe-mariani-2026-fusi-template.nii.gz",
        root
        / "nunez-elizalde-2022-bids/1.0.0/sub-CR022/ses-20201007/fusi/sub-CR022_ses-20201007_task-spontaneous_acq-slice02_pwd.nii.gz",
    ]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(
            "Cached template and CR022 recording required; no implicit downloads."
        )
    template = cf.load(paths[0], coordinate_affine="qform").compute()
    recording = cf.load(paths[1])
    moving = recording.mean("time").fusi.scale.db().compute()
    placement = np.eye(4)
    placement[:3, 3] = [5.594638656430411, -2.50293925701927, 5.6650243788545875]
    target_z = placement[0, 3] + float(moving.fusi.origin["z"])
    fixed = template.sel(z=slice(target_z - 1, target_z + 1)).fusi.scale.db().compute()
    return fixed, moving, placement, paths, list(recording.shape)
