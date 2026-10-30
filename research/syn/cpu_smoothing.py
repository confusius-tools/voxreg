"""Isolate CPU filter backend/precision without changing registration or boundaries."""

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import confusius as cf
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from voxreg import fields, syn

_original = fields.smooth_field
MODES = ["baseline", "torch64", "gradient32", "all32"]


class CPUSmoother:
    """Experiment-owned kernels; no process-global cache or precision setting."""

    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.kernels = {}
        self.calls = {"float32": 0, "float64": 0, "baseline": 0}

    def __call__(self, field, grid, *, sigma):
        if (
            self.mode == "baseline"
            or field.device.type != "cpu"
            or sigma == 0
            or not isfinite(sigma)
            or sigma < 0
            or (torch.is_grad_enabled() and field.requires_grad)
            or (self.mode == "gradient32" and sigma != 1.5)
        ):
            self.calls["baseline"] += 1
            return _original(field, grid, sigma=sigma)
        fields._check_field(field, grid)
        low_precision = self.mode in ("gradient32", "all32")
        dtype = torch.float32 if low_precision else torch.float64
        dimensions = len(grid.shape)
        key = (sigma, dtype, dimensions)
        if key not in self.kernels:
            radius = int(3 * sigma + 0.5)
            positions = torch.arange(-radius, radius + 1, dtype=dtype)
            kernel = torch.exp(-0.5 * (positions / sigma).square())
            kernel = kernel / kernel.sum()
            weights = []
            for axis in range(dimensions):
                shape = [1] * dimensions
                shape[axis] = len(kernel)
                weights.append(
                    kernel.reshape(1, 1, *shape)
                    .expand(dimensions, 1, *shape)
                    .contiguous()
                )
            self.kernels[key] = (radius, weights)
        radius, weights = self.kernels[key]
        scale = None
        result = field
        if low_precision:
            scale = field.abs().flatten(1).amax(1).reshape(-1, 1, *([1] * dimensions))
            scale = scale.clamp_min(torch.finfo(field.dtype).tiny)
            result = (field / scale).float()
        convolution = F.conv2d if dimensions == 2 else F.conv3d
        for axis, weight in enumerate(weights):
            padding = [0] * (2 * dimensions)
            padding[2 * (dimensions - axis - 1) : 2 * (dimensions - axis)] = [
                radius,
                radius,
            ]
            result = convolution(
                F.pad(result, padding, mode="replicate"), weight, groups=dimensions
            )
        result = result.masked_fill(grid.boundary, 0)
        self.calls[str(dtype).removeprefix("torch.")] += 1
        if low_precision:
            result = result.clamp(-1, 1).double() * scale
        return result


def micro(args):
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    torch.set_num_threads(args.threads)
    for batch, shape in [(32, (1, 145, 80)), (1, (32, 40, 48)), (8, (32, 32, 32))]:
        data = cf.create_voxeldata(
            np.zeros(shape, np.float32),
            dims=("k", "j", "i"),
            voxel_to_world=np.diag([0.24, 0.21, 0.18, 1]),
            units="mm",
        )
        grid = fields.make_field_grid(data)
        values = (
            np.random.default_rng(37).normal(size=(batch, len(grid.shape), *grid.shape))
            * 0.02
        )
        field = torch.tensor(values)
        for mode in MODES:
            smoother = CPUSmoother(mode)
            for sigma in [1.5, 0.5]:
                expected = gaussian_filter(
                    values,
                    sigma=sigma,
                    radius=int(3 * sigma + 0.5),
                    axes=tuple(range(2, field.ndim)),
                    mode="nearest",
                )
                expected = np.where(grid.boundary.numpy(), 0, expected)
                actual = smoother(field, grid, sigma=sigma)
                error = float(np.abs(actual.numpy() - expected).max())
                bound = (
                    float(np.abs(values).max()) * 2e-6
                    if mode in ("gradient32", "all32")
                    else 1e-14
                )
                assert error <= bound, (mode, sigma, error, bound)
                np.testing.assert_array_equal(field, values)
                assert actual.dtype == torch.float64
                for _ in range(3):
                    smoother(field, grid, sigma=sigma)
                start = perf_counter()
                for _ in range(args.repeats):
                    smoother(field, grid, sigma=sigma)
                row = {
                    "mode": mode,
                    "batch": batch,
                    "shape": shape,
                    "sigma": sigma,
                    "seconds": (perf_counter() - start) / args.repeats,
                    "max_error": error,
                }
                rows.append(row)
                print(row, flush=True)
    (args.output / "results.json").write_text(
        json.dumps(
            {"threads": args.threads, "repeats": args.repeats, "rows": rows}, indent=2
        )
        + "\n"
    )


def run(args):
    torch.set_num_threads(args.threads)
    if args.micro:
        micro(args)
        return
    smoother = CPUSmoother(args.mode)
    with patch.object(syn, "smooth_field", smoother):
        if args.tests:
            import pytest

            raise SystemExit(
                pytest.main(
                    ["-q", "tests/test_syn.py", "tests/test_optional_transforms.py"]
                )
            )
        from compare import run as compare

        compare(args)
    path = args.output / "results.json"
    report = json.loads(path.read_text())
    report["cpu_smoothing"] = {
        "mode": args.mode,
        "calls": smoother.calls,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_storage": "float64",
        "coordinates": "float64",
        "statistics": "float64",
        "radius": "round(3*sigma)",
        "padding": "replicate",
        "production": False,
    }
    path.write_text(json.dumps(report, indent=2) + "\n")
    assert all(
        arm.get("truth_gate_pass")
        for case in report["cases"].values()
        for arm in case["backends"].values()
    )
    for case in report["cases"].values():
        if "precision_gate_pass" in case:
            assert case["precision_gate_pass"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=MODES, default="baseline")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--cases", nargs="+", default=["2d-medium", "3d-medium", "3d-batch8"]
    )
    parser.add_argument(
        "--backends", nargs="+", default=["voxreg-cpu", "voxreg-cuda", "fireants-cpu"]
    )
    args = parser.parse_args()
    if args.output is None and not args.tests:
        parser.error("--output required unless --tests")
    if min(args.threads, args.iterations, args.repeats) < 1:
        parser.error("threads, iterations and repeats must be positive")
    run(args)
