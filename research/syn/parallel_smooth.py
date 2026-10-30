"""Measure independent CPU frame filters with a bounded, scoped thread pool."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

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
            field.numpy(),
        )
    )
    return torch.from_numpy(np.stack(values)).masked_fill(grid.boundary, 0)


def run(args):
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
    parser.add_argument("--repeats", type=int, default=20)
    run(parser.parse_args())
