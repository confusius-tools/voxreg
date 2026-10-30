"""Measure channel-as-batch Gaussian filtering against grouped FP64 filtering."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import confusius as cf
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from voxreg.fields import make_field_grid, smooth_field


def batch_smooth(field, grid, *, sigma):
    """Same separable filter and boundaries, with independent channels in N."""
    if sigma == 0:
        return field
    dimensions = len(grid.shape)
    radius = int(3 * sigma + 0.5)
    positions = torch.arange(
        -radius, radius + 1, dtype=field.dtype, device=field.device
    )
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel = kernel / kernel.sum()
    result = field.flatten(0, 1)[:, None]
    convolution = F.conv2d if dimensions == 2 else F.conv3d
    for axis in range(dimensions):
        shape = [1] * dimensions
        shape[axis] = len(kernel)
        weights = kernel.reshape(1, 1, *shape)
        padding = [0] * (2 * dimensions)
        padding[2 * (dimensions - axis - 1) : 2 * (dimensions - axis)] = [
            radius,
            radius,
        ]
        result = convolution(F.pad(result, padding, mode="replicate"), weights)
    return result.reshape_as(field).masked_fill(grid.boundary, 0)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    rows = []
    for device in args.devices:
        for batch, shape in [(32, (145, 80)), (1, (32, 40, 48)), (8, (32, 40, 48))]:
            dimensions = len(shape)
            native = (1, *shape) if dimensions == 2 else shape
            data = cf.create_voxeldata(
                np.zeros(native, np.float32),
                dims=("k", "j", "i"),
                voxel_to_world=np.diag([0.24, 0.21, 0.18, 1]),
                units="mm",
            )
            grid = make_field_grid(data, device=device)
            values = (
                np.random.default_rng(74).normal(size=(batch, dimensions, *shape))
                * 0.02
            )
            field = torch.tensor(values, device=device)
            for sigma in (1.5, 0.5):
                expected = values.copy()
                for axis in range(dimensions):
                    expected = gaussian_filter1d(
                        expected,
                        sigma,
                        axis=axis + 2,
                        mode="nearest",
                        radius=int(3 * sigma + 0.5),
                    )
                for axis in range(dimensions):
                    at = [slice(None)] * expected.ndim
                    at[axis + 2] = [0, -1]
                    expected[tuple(at)] = 0
                for name, function in [
                    ("grouped", smooth_field),
                    ("batch", batch_smooth),
                ]:
                    with torch.no_grad():
                        actual = function(field, grid, sigma=sigma)
                        np.testing.assert_allclose(
                            actual.cpu(), expected, rtol=2e-14, atol=2e-16
                        )
                        for _ in range(3):
                            function(field, grid, sigma=sigma)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        start = perf_counter()
                        for _ in range(args.repeats):
                            function(field, grid, sigma=sigma)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        seconds = (perf_counter() - start) / args.repeats
                    row = {
                        "device": device,
                        "batch": batch,
                        "shape": shape,
                        "sigma": sigma,
                        "variant": name,
                        "seconds": seconds,
                    }
                    print(row, flush=True)
                    rows.append(row)
    (args.output / "results.json").write_text(
        json.dumps(
            {"threads": args.threads, "repeats": args.repeats, "rows": rows}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"]
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    run(parser.parse_args())
