"""Small, deterministic physical registration fixtures."""

import numpy as np
import pytest
import torch
from confusius import create_voxeldata
from scipy.spatial.transform import Rotation


@pytest.fixture(autouse=True, scope="session")
def single_thread():
    """Keep tiny CPU numerical tests independent of machine thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def recording_factory():
    """Provide analytic asymmetric images and independently known world mappings."""

    def make(*, oblique=False, strided=False):
        h, w = 64, 80
        orientation = (
            Rotation.from_euler("xyz", [24, -13, 37], degrees=True).as_matrix()
            if oblique
            else np.eye(3)
        )
        affine = np.eye(4)
        affine[:3, :3] = orientation @ np.diag([0.5, 0.2, 0.12])
        affine[:3, 3] = [104.0, -73.0, 29.0]
        coords = {
            "k": np.array([7]),
            "j": np.arange(h),
            "i": np.arange(w),
        }
        if strided:
            coords["j"] = 5 - 2 * coords["j"]
            coords["i"] = 13 + 2 * coords["i"]
        position_affine = affine.copy()
        for axis, dim in enumerate(("k", "j", "i")):
            position_affine[:3, 3] += affine[:3, axis] * coords[dim][0]
            if len(coords[dim]) > 1:
                position_affine[:3, axis] *= coords[dim][1] - coords[dim][0]
        pixels = position_affine[:3, [2, 1]]
        e_i = pixels[:, 0] / np.linalg.norm(pixels[:, 0])
        e_j = pixels[:, 1] - e_i * np.dot(e_i, pixels[:, 1])
        e_j /= np.linalg.norm(e_j)
        basis = np.column_stack((e_i, e_j))
        lattice = basis.T @ pixels
        j, i = np.indices((h, w))
        pixel_center = np.array([(w - 1) / 2, (h - 1) / 2])
        points = (np.stack((i, j), axis=-1) - pixel_center) @ lattice.T
        center = position_affine[:3, 3] + pixels @ pixel_center
        ell = np.linalg.norm(pixels, axis=0).min()
        rng = np.random.default_rng(19)
        locations = rng.uniform(-0.7, 0.7, (18, 2)) * np.abs(points).max(axis=(0, 1))
        widths = rng.uniform(2, 5, (18, 2)) * np.linalg.norm(pixels, axis=0)
        amplitudes = rng.uniform(0.5, 2, 18)

        def phantom(p):
            return 0.05 + np.sum(
                amplitudes
                * np.exp(
                    -0.5
                    * np.sum(((p[..., None, :] - locations) / widths) ** 2, axis=-1)
                ),
                axis=-1,
            )

        params = np.array(
            [
                [0, 0, 0],
                [np.deg2rad(2), 1.3 * ell, -0.75 * ell],
                [-np.deg2rad(1.5), -2 * ell, ell],
            ]
        )
        values, matrices = [], []
        for angle, ti, tj in params:
            c, s = np.cos(angle), np.sin(angle)
            r = np.array([[c, -s], [s, c]])
            values.append(phantom((points - [ti, tj]) @ r))
            matrix = np.eye(4)
            matrix[:3, :3] = basis @ r @ basis.T + np.eye(3) - basis @ basis.T
            matrix[:3, 3] = center + basis @ [ti, tj] - matrix[:3, :3] @ center
            matrices.append(matrix)
        coords["time"] = np.array([22.0, 22.1, 22.2])
        data = create_voxeldata(
            np.array(values, dtype=np.float32)[:, None],
            dims=("time", "k", "j", "i"),
            **coords,
            voxel_to_world=affine,
            attrs={"units": "power"},
        )
        return data, np.array(matrices), position_affine

    return make
