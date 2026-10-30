"""Synchronized no-transform CUDA batch sweep on real ref20 and a shared 3D fixture."""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import confusius as cf
import numpy as np
import torch
import xarray as xr

from voxreg import register

ROOT = Path(__file__).resolve().parents[2]


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    root = cf.datasets.fetch_landemard_2026(
        datasets="rawdata", subjects="ALD001", acqs="ref20", datatypes="fusi"
    )
    source = next(root.glob("sub-ALD001/fusi/*_acq-ref20*_pwd.nii.gz"))
    real = cf.load(source).isel(time=slice(0, args.frames)).compute()
    sys.path.insert(0, str(ROOT / "research/syn"))
    from compare import inputs

    fixed, moving, *_ = inputs("3d-medium")
    volume = xr.concat(
        [moving] * args.volumes, dim=xr.IndexVariable("time", np.arange(args.volumes))
    )
    report = {
        "gpu": torch.cuda.get_device_name(),
        "iterations": args.iterations,
        "repeats": args.repeats,
        "source": str(source),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/voxreg").glob("*.py"))
        },
        "notes": "Fixed budget, no transforms, one whole-recording call per trial. Imports/I/O excluded; preparation, upload, fitting and output included. Each shape/model/batch warmed with two updates. Timed repeats use the cached allocator; allocated peaks exclude reusable reserved cache. 3D frames repeat a shared nonlinear fixture, not a real 3D recording.",
        "rows": [],
    }
    for case, data, target in [("ref20", real, None), ("3d", volume, fixed)]:
        if case not in args.cases:
            continue
        for model in args.methods:
            baseline = None
            for batch in args.batches:
                options = {
                    "transform": model,
                    "device": "cuda",
                    "batch_size": batch,
                    "intensity_scaling": "db" if case == "ref20" else "none",
                    "convergence_window_size": args.iterations + 1,
                    "progress": False,
                }
                try:
                    del_warm = register(data, target, max_iterations=2, **options)
                    del del_warm
                except torch.cuda.OutOfMemoryError as error:
                    report["rows"].append(
                        {
                            "case": case,
                            "model": model,
                            "batch_size": batch,
                            "error": str(error),
                        }
                    )
                    (args.output / "results.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                    print(f"OOM: {case} {model} batch={batch}", flush=True)
                    error.__traceback__ = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                for repeat in range(args.repeats):
                    gc.collect()
                    torch.cuda.synchronize()
                    free, total = torch.cuda.mem_get_info()
                    torch.cuda.reset_peak_memory_stats()
                    start = perf_counter()
                    result = register(
                        data, target, max_iterations=args.iterations, **options
                    )
                    torch.cuda.synchronize()
                    seconds = perf_counter() - start
                    assert (
                        result.transforms is None and result.inverse_transforms is None
                    )
                    assert result.registered.shape == data.shape
                    if target is None:
                        np.testing.assert_array_equal(result.registered[0], data[0])
                    row = {
                        "case": case,
                        "shape": list(data.shape),
                        "model": model,
                        "batch_size": batch,
                        "repeat": repeat,
                        "seconds": seconds,
                        "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                        "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
                        "free_bytes_before": free,
                        "total_bytes": total,
                        "status_counts": {
                            str(k): int(v)
                            for k, v in zip(
                                *np.unique(
                                    result.diagnostics.status, return_counts=True
                                ),
                                strict=True,
                            )
                        },
                        "updates": int(result.diagnostics.n_iterations.sum()),
                        "stage_seconds": result.diagnostics.attrs["timings_seconds"],
                    }
                    if baseline is None:
                        baseline = result.registered.values.copy()
                    row["image_rms_vs_first_trial"] = float(
                        np.sqrt(
                            np.mean(
                                (result.registered.values.astype(float) - baseline) ** 2
                            )
                        )
                    )
                    report["rows"].append(row)
                    (args.output / "results.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                    print(json.dumps(row), flush=True)
                    del result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases", nargs="+", choices=["ref20", "3d"], default=["ref20", "3d"]
    )
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--volumes", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 128, 256])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["rigid", "affine", "syn"],
        default=["rigid", "affine", "syn"],
    )
    args = parser.parse_args()
    if min(args.frames, args.volumes, args.iterations, args.repeats, *args.batches) < 1:
        parser.error("counts and batches must be positive")
    run(args)
