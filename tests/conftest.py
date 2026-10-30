"""Small, deterministic physical registration fixtures."""

import numpy as np
import pytest
import torch
from confusius import create_voxeldata
from oracles import sample_mixed, voxel
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation

from voxreg import position_to_world


@pytest.fixture(autouse=True, scope="session")
def single_thread():
    """Keep tiny CPU numerical tests independent of machine thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ]
)
def device(request):
    """Shared CPU/CUDA selection; unavailable CUDA remains an explicit skip."""
    return request.param


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
            k=coords["k"],
            j=coords["j"],
            i=coords["i"],
            time=coords["time"],
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
            center, widths = np.asarray(center), np.asarray(widths)
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


@pytest.fixture
def nonlinear_pair_factory(displacement_factory):
    """Textured native pairs with an independently inverted analytic nonlinear pull."""

    def make(
        planar,
        *,
        shape=None,
        amplitude=None,
        dtype=np.float32,
        sheared=True,
        axis_aligned=False,
    ):
        data, _, _, _ = displacement_factory(planar)
        af = position_to_world(data)
        if not sheared:
            af[:3, 1] -= 0.2 * af[:3, 2]
        if axis_aligned:
            af[:3, :3] = np.diag(np.linalg.norm(af[:3, :3], axis=0))
        base = data.values[0] if planar else data.values
        if shape is not None:
            shape = tuple(shape)
            factors = (np.array(base.shape) - 1) / (np.array(shape) - 1)
            coordinates = np.indices(shape) * factors.reshape(-1, *([1] * len(shape)))
            base = map_coordinates(base, coordinates, order=3, mode="nearest")
            af[:3, 1 if planar else 0 : 3] *= factors
        shape = base.shape
        rng = np.random.default_rng(472)
        texture = gaussian_filter(rng.normal(size=shape), 0.9)
        signal = (
            0.3 * (base - base.min()) / np.ptp(base) + 0.15 * texture / texture.std()
        )
        signal = signal - signal.min() + 0.05
        coordinates = np.indices(shape, dtype=float)
        amplitude = (1.5 if planar else 2.0) if amplitude is None else amplitude
        rates = amplitude * np.array([1.0, -0.6, 0.35])[: len(shape)]
        assert (np.abs(rates) * np.pi / (np.array(shape) - 1) < 1).all()

        def window_at(points, axis):
            others = [i for i in range(len(shape)) if i != axis]
            scale = (np.array(shape)[others] - 1).reshape(-1, *([1] * len(shape)))
            return np.sin(np.pi * points[others] / scale).prod(axis=0)

        forward = coordinates.copy()
        for axis, rate in enumerate(rates):
            forward[axis] += (
                rate
                * np.sin(np.pi * forward[axis] / (shape[axis] - 1))
                * window_at(forward, axis)
            )
        inverse = coordinates.copy()
        for axis in reversed(range(len(shape))):
            target = inverse[axis].copy()
            window = window_at(inverse, axis)
            rate = rates[axis]
            for _ in range(20):
                angle = np.pi * inverse[axis] / (shape[axis] - 1)
                residual = inverse[axis] + rate * np.sin(angle) * window - target
                derivative = (
                    1 + rate * np.pi / (shape[axis] - 1) * np.cos(angle) * window
                )
                inverse[axis] -= residual / derivative
            assert (
                np.abs(
                    inverse[axis]
                    + rate * np.sin(np.pi * inverse[axis] / (shape[axis] - 1)) * window
                    - target
                ).max()
                < 1e-12
            )
        moved = map_coordinates(
            signal, inverse, order=3, mode="nearest", prefilter=True
        )
        if np.issubdtype(dtype, np.integer):
            scale = 30000 / max(signal.max(), moved.max())
            signal, moved = np.rint(signal * scale), np.rint(moved * scale)
        lattice = af[:3, 1 if planar else 0 : 3]
        truth = np.moveaxis(forward - coordinates, 0, -1) @ lattice.T
        reverse = np.moveaxis(inverse - coordinates, 0, -1) @ lattice.T
        if planar:
            signal, moved, truth, reverse = (
                signal[None],
                moved[None],
                truth[None],
                reverse[None],
            )
        fixed, moving = voxel(signal.astype(dtype), af), voxel(moved.astype(dtype), af)
        moving.attrs.update(units="power", subject="nonlinear-fixture")
        moving.name = "power"
        return fixed, moving, truth, reverse

    return make


@pytest.fixture
def displacement_factory(linear_pair_factory):
    """Zero-boundary physical fields on oblique, sheared/reversed 2D/3D grids."""

    def make(planar, *, strided=False):
        data = linear_pair_factory(planar, "translation")[0]
        affine = position_to_world(data)
        affine[:3, 1] += 0.2 * affine[:3, 2]
        data = voxel(data.values.copy(), affine)
        if strided:
            data = data.isel(j=slice(None, None, -2), i=slice(None, None, -2))
            if not planar:
                data = data.isel(k=slice(None, None, 2))
        affine = position_to_world(data)
        if planar:
            first = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
            second = affine[:3, 1] - first * np.dot(first, affine[:3, 1])
            basis = np.column_stack((first, second / np.linalg.norm(second)))
        else:
            basis = np.eye(3)
        lattice = basis.T @ affine[:3, 1 if planar else 0 : 3]
        shape = data.shape[1:] if planar else data.shape
        fractions = np.indices(shape) / np.array([n - 1 for n in shape]).reshape(
            -1, *([1] * len(shape))
        )
        window = np.prod(np.sin(np.pi * fractions), axis=0)
        for axis in range(len(shape)):
            window[(fractions[axis] == 0) | (fractions[axis] == 1)] = 0
        ell = np.linalg.norm(lattice, axis=0).min()
        field = np.stack(
            [
                (0.5 + 0.2 * np.cos(np.pi * fractions[(axis + 1) % len(shape)]))
                * (-1) ** axis
                * window
                * 0.35
                * ell
                for axis in range(len(shape))
            ]
        )
        return data, basis, lattice, np.stack((field, -0.6 * field))

    return make
