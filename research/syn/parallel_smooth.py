"""Measure independent CPU frame filters with a bounded, scoped thread pool."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import confusius as cf
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from voxreg.fields import make_field_grid, smooth_field


def parallel_smooth(field, grid, *, sigma, pool):
    values = list(
        pool.map(
            partial(
                gaussian_filter,
                sigma=sigma,
                radius=int(3 * sigma + 0.5),
                axes=tuple(range(1, field.ndim - 1)),
                mode="nearest",
            ),
            field.detach().numpy(),
        )
    )
    return torch.from_numpy(np.stack(values)).masked_fill(grid.boundary, 0)


def threaded_smooth(field, grid, *, sigma):
    if (
        field.device.type != "cpu"
        or len(grid.shape) != 3
        or len(field) < 2
        or torch.get_num_threads() < 2
        or sigma <= 0
        or not np.isfinite(sigma)
        or (torch.is_grad_enabled() and field.requires_grad)
    ):
        return smooth_field(field, grid, sigma=sigma)
    from voxreg.fields import _check_field

    _check_field(field, grid)
    with ThreadPoolExecutor(
        max_workers=min(len(field), torch.get_num_threads())
    ) as pool:
        return parallel_smooth(field, grid, sigma=sigma, pool=pool)


def benchmark(args):
    from compare import run as compare

    from voxreg import fields, syn

    args.backends = ["voxreg-cpu", "fireants-cpu"]
    args.iterations = 120
    with (
        patch.object(fields, "smooth_field", threaded_smooth),
        patch.object(syn, "smooth_field", threaded_smooth),
    ):
        compare(args)
    path = args.output / "results.json"
    report = json.loads(path.read_text())
    report["parallel_smoothing"] = {
        "pool_lifetime": "one filter call",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path.write_text(json.dumps(report, indent=2) + "\n")
    assert all(
        arm.get("truth_gate_pass")
        for case in report["cases"].values()
        for arm in case["backends"].values()
    )


def run(args):
    if args.benchmark:
        benchmark(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    rows = []
    for batch, shape in [(32, (1, 145, 80)), (8, (32, 40, 48))]:
        data = cf.create_voxeldata(
            np.zeros(shape, np.float32),
            dims=("k", "j", "i"),
            voxel_to_world=np.diag([0.24, 0.21, 0.18, 1]),
            units="mm",
        )
        grid = make_field_grid(data)
        field = torch.tensor(
            np.random.default_rng(62).normal(size=(batch, len(grid.shape), *grid.shape))
            * 0.02
        )
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            for sigma in [1.5, 0.5]:
                expected = smooth_field(field, grid, sigma=sigma)
                for name in ["serial", "pooled", "new-pool"]:

                    def apply(
                        name=name, field=field, grid=grid, sigma=sigma, pool=pool
                    ):
                        if name == "serial":
                            return smooth_field(field, grid, sigma=sigma)
                        if name == "pooled":
                            return parallel_smooth(field, grid, sigma=sigma, pool=pool)
                        with ThreadPoolExecutor(max_workers=args.threads) as temporary:
                            return parallel_smooth(
                                field, grid, sigma=sigma, pool=temporary
                            )

                    torch.testing.assert_close(apply(), expected, rtol=0, atol=0)
                    for _ in range(3):
                        apply()
                    start = perf_counter()
                    for _ in range(args.repeats):
                        apply()
                    row = {
                        "batch": batch,
                        "shape": shape,
                        "sigma": sigma,
                        "variant": name,
                        "seconds": (perf_counter() - start) / args.repeats,
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
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["2d-medium", "3d-medium", "3d-batch8"],
        default=["3d-medium", "3d-batch8"],
    )
    args = parser.parse_args()
    if args.repeats is None:
        args.repeats = 2 if args.benchmark else 20
    run(args)
