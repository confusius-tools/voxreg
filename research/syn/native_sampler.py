"""Experimental native-double interpolation of float32 fitting samples.

Keep original float64 final interpolation and nearest ties unchanged. A tiny
right-sided displacement at interior fitting knots fixes the derivative side
without depending on CPU/CUDA normalized-coordinate roundoff.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from voxreg import sampling

_original = sampling._sample_displaced


def native_sample(images, normalized, interpolation):
    if images.dtype != torch.float32 or interpolation != "linear":
        return _original(images, normalized, interpolation)
    dimensions = images.ndim - 1
    coordinates = []
    for axis, size in enumerate(images.shape[1:]):
        coordinate = (normalized[..., dimensions - axis - 1] + 1) * ((size - 1) / 2)
        nearest = coordinate.round()
        interior = (nearest > 0) & (nearest < size - 1)
        snapped = torch.where(interior, nearest + 1e-10, nearest)
        snapped = torch.where((coordinate - nearest).abs() < 1e-10, snapped, coordinate)
        coordinate = coordinate + (snapped - coordinate).detach()
        coordinate = torch.where(
            (coordinate > 0) & (coordinate < size - 1),
            coordinate,
            coordinate.detach().clamp(0, size - 1),
        )
        coordinates.append(coordinate / ((size - 1) / 2) - 1)
    grid = torch.stack(coordinates[::-1], dim=-1)
    return F.grid_sample(
        images[:, None].double(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[:, 0].float()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    with patch.object(sampling, "_sample_displaced", native_sample):
        if args.tests:
            import pytest

            raise SystemExit(
                pytest.main(
                    ["-q", "tests/test_syn.py", "tests/test_optional_transforms.py"]
                )
            )
        else:
            if args.output is None:
                parser.error("--output required unless --tests")
            if args.benchmark:
                from landemard import run

                args.batch_size = None
                args.fireants_batch_size = 256
            else:
                from profile_syn import run

                args.compile = []
            run(args)
