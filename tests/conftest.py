"""Small, deterministic physical registration fixtures."""

import numpy as np
import pytest
import torch
from confusius import create_voxeldata
from oracles import sample_mixed, voxel
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


@pytest.fixture
def linear_pair_factory():
    """Analytic 2D/3D pairs with oblique unequal grids and independent physical truth."""

    def make(planar, model="affine"):
        q = Rotation.from_rotvec([0.15, -0.2, 0.1]).as_matrix()
        lattice = q @ np.diag([0.24, 0.21, 0.18])
        basis = q[:, [2, 1]] if planar else np.eye(3)
        center = np.array([10.0, -20.0, 30.0])
        dimensions = 2 if planar else 3
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
                linear = Rotation.from_rotvec([0.04, -0.03, 0.02]).as_matrix()
        translation = (
            np.array([0.10, -0.08]) if planar else np.array([0.10, -0.08, 0.06])
        )
        truth = np.eye(4)
        truth[:3, :3] += basis @ (linear - np.eye(dimensions)) @ basis.T
        truth[:3, 3] = basis @ translation - (truth[:3, :3] - np.eye(3)) @ center
        rng = np.random.default_rng(871)
        locations = rng.uniform(-1.4, 1.4, (16, dimensions))
        widths = rng.uniform(0.3, 0.65, (16, dimensions))
        amplitudes = rng.uniform(0.5, 2, 16)
        fixed_shape = (1, 25, 29) if planar else (17, 19, 21)
        moving_shape = (1, 33, 37) if planar else (25, 27, 29)
        images = []
        for shape, pose in [(fixed_shape, np.eye(4)), (moving_shape, truth)]:
            af = np.eye(4)
            af[:3, :3] = lattice
            af[:3, 3] = center - lattice @ ((np.array(shape) - 1) / 2)
            points = (
                lattice
                @ (
                    np.indices(shape).reshape(3, -1)
                    - (np.array(shape) - 1)[:, None] / 2
                )
            ).T @ basis
            values = 0.05 + np.sum(
                amplitudes
                * np.exp(
                    -0.5
                    * np.sum(((points[:, None] - locations) / widths) ** 2, axis=-1)
                ),
                axis=1,
            )
            images.append(
                create_voxeldata(
                    values.reshape(shape),
                    dims=("k", "j", "i"),
                    voxel_to_world=pose @ af,
                )
            )
        return *images, truth

    return make


@pytest.fixture
def slice_volume_pair_factory():
    """Finite native slice and volume; truth does not use voxreg parameter decoding."""

    def make(model):
        shape = (25, 27, 29)
        affine = np.eye(4)
        affine[:3, :3] *= 0.4
        affine[:3, 3] = -(np.array(shape) - 1) * 0.2
        coordinates = (
            affine
            @ np.vstack(
                (np.indices(shape).reshape(3, -1), np.ones((1, np.prod(shape))))
            )
        )[:3]
        rng = np.random.default_rng(42)
        values = np.ones(coordinates.shape[1]) * 0.02
        for center, widths, weight in zip(
            rng.uniform(-3, 3, (18, 3)),
            rng.uniform(0.45, 1.2, (18, 3)),
            rng.uniform(0.3, 2, 18),
            strict=True,
        ):
            values += weight * np.exp(
                -0.5
                * (((coordinates - center[:, None]) / widths[:, None]) ** 2).sum(axis=0)
            )
        volume = voxel(values.reshape(shape), affine)
        af = np.eye(4)
        af[:3, :3] *= 0.35
        af[:3, 3] = [0, -3.5, -3.85]
        target = voxel(np.ones((1, 21, 23)), af)
        truth = np.eye(4)
        if model == "rigid":
            truth[:3, :3] = Rotation.from_rotvec([0.025, -0.02, 0.03]).as_matrix()
        elif model == "affine":
            stretch = np.eye(3)
            stretch[1:, 1:] = [[1.015, 0.008], [-0.012, 0.99]]
            truth[:3, :3] = (
                Rotation.from_rotvec([0.015, -0.01, 0.008]).as_matrix() @ stretch
            )
        truth[:3, 3] = [0.12, -0.08, 0.09]
        sampled, valid = sample_mixed(volume, target, truth)
        assert valid.all()
        return voxel(sampled, af), volume, truth

    return make
