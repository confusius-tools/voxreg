"""Native-grid ROI masks sampled through complete physical registration pulls."""

from dataclasses import dataclass

import numpy as np
import torch

from voxreg.geometry import (
    VOXEL_DIMS,
    _validate_registration_geometry,
    position_to_world,
)
from voxreg.sampling import Plane, _build_physical_transform, build_plane_rotations
from voxreg.transforms import build_intrinsic_linear


def _validate_masks(fixed_mask, moving_mask, moving):
    """Require boolean VoxelData and exact moving time-label correspondence."""
    for name, mask in (("fixed_mask", fixed_mask), ("moving_mask", moving_mask)):
        if mask is None:
            continue
        _validate_registration_geometry(mask)
        if mask.dtype != np.bool_:
            raise TypeError(f"{name} must be boolean VoxelData.")
        if name == "fixed_mask" and mask.dims != VOXEL_DIMS:
            raise ValueError("fixed_mask must be spatial-only.")
        if "time" in mask.dims and (
            "time" not in moving.dims
            or "time" not in mask.coords
            or not mask.get_index("time").equals(moving.get_index("time"))
        ):
            raise ValueError("moving_mask time labels must match moving exactly.")


@dataclass(frozen=True)
class _MaskSource:
    """Resident binary values and their own physical voxel lattice."""

    values: torch.Tensor
    affine: np.ndarray


def _source(mask, device, first=0, last=None):
    """Upload only the current frame batch, or one shared spatial mask."""
    if mask is None:
        return None
    if "time" in mask.dims:
        mask = mask.isel(time=slice(first, last))
        values = mask.values
    else:
        values = mask.values[None]
    return _MaskSource(
        torch.tensor(np.ascontiguousarray(values), device=device, dtype=torch.bool)[
            :, None
        ],
        position_to_world(mask),
    )


class _MaskSampler:
    """Cache origin-free domain coordinates for nearest physical mask sampling."""

    def __init__(self, source, domain, geometry=None):
        self.source, self.geometry = source, geometry
        self.planar = domain.sizes["k"] == 1 and (
            geometry is None or isinstance(geometry, Plane) or geometry.planar
        )
        affine = position_to_world(domain)
        dims = ("j", "i") if self.planar else VOXEL_DIMS
        axes = torch.meshgrid(
            *(
                torch.arange(
                    domain.sizes[d], device=source.values.device, dtype=torch.float64
                )
                for d in dims
            ),
            indexing="ij",
        )
        indices = torch.stack(axes, dim=-1)
        self.local = (
            indices
            @ torch.tensor(
                affine[:3, 1 if self.planar else 0 : 3], device=source.values.device
            ).T
        )
        self.origin = torch.tensor(
            affine[:3, 3] - source.affine[:3, 3], device=source.values.device
        )
        self.pivot = (
            None
            if geometry is None
            else torch.tensor(
                affine[:3, 3] - geometry.center, device=source.values.device
            )
        )
        self.inverse = torch.tensor(
            np.linalg.inv(source.affine[:3, :3]), device=source.values.device
        )
        self.sizes = torch.tensor(
            source.values.shape[2:], device=source.values.device, dtype=torch.float64
        )

    @torch.no_grad()
    def sample(self, count, parameters=None, displacement=None):
        """Evaluate binary cells; no ROI gradient or out-of-domain border extension."""
        geometry = self.geometry
        if parameters is None:
            query = (self.local + self.origin)[None].expand(count, *self.local.shape)
            linear = None
        else:
            parameters = parameters.detach()
            if isinstance(geometry, Plane):
                basis = torch.tensor(geometry.basis, device=parameters.device)
                if geometry.transform == "rigid":
                    rotation, shift = (
                        build_plane_rotations(parameters[:, 0]),
                        parameters[:, 1:],
                    )
                else:
                    rotation, shift = build_intrinsic_linear(parameters, geometry)
                linear = (
                    torch.eye(3, device=parameters.device)
                    + basis @ (rotation - geometry.identity) @ basis.T
                )
                translation = shift @ basis.T
            else:
                linear, translation = _build_physical_transform(parameters, geometry)
            offset = (
                self.origin
                + torch.einsum(
                    "bij,j->bi",
                    linear - torch.eye(3, device=parameters.device),
                    self.pivot,
                )
                + translation
            )
            query = torch.einsum(
                "bij,...j->b...i", linear, self.local
            ) + offset.reshape(count, *([1] * (self.local.ndim - 1)), 3)
        if displacement is not None:
            assert geometry is not None
            basis = torch.tensor(geometry.basis, device=self.source.values.device)
            mapping = basis if linear is None else linear @ basis
            query = query + torch.einsum(
                "ic,bc...->b...i" if linear is None else "bic,bc...->b...i",
                mapping,
                displacement.detach().double(),
            )
        positions = query @ self.inverse.T
        valid = ((positions >= -0.5) & (positions < self.sizes - 0.5)).all(-1)
        # ITK physical mask membership rounds half-integer indices upwards.
        indices = torch.floor(positions + 0.5).long()
        sizes = self.source.values.shape[2:]
        k, j, i = (
            indices[..., axis].clamp(0, size - 1) for axis, size in enumerate(sizes)
        )
        flat = ((k * sizes[1] + j) * sizes[2] + i).reshape(count, -1)
        values = self.source.values.reshape(len(self.source.values), -1).expand(
            count, -1
        )
        sampled = values.gather(1, flat).reshape(valid.shape)
        return valid & sampled


class _MetricMasks:
    """Intersect ROI masks with image support in the current metric domain."""

    def __init__(self, fixed, moving, domain, geometry):
        self.sources = (fixed, moving)
        self.fixed = None if fixed is None else _MaskSampler(fixed, domain, geometry)
        self.moving = None if moving is None else _MaskSampler(moving, domain, geometry)

    def apply(
        self, valid, parameters, moving_displacement=None, fixed_displacement=None
    ):
        """Warp each ROI through its own complete pull into the metric domain."""
        if self.fixed is not None:
            valid = valid & self.fixed.sample(
                len(valid), displacement=fixed_displacement
            )
        if self.moving is not None:
            valid = valid & self.moving.sample(
                len(valid), parameters, moving_displacement
            )
        return valid
