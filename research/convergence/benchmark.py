"""Time one stopping policy in a fresh process, without profiling or memory sampling."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from confusius import create_voxeldata

from voxreg import register, rigid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--minimum", type=float, default=1e-6)
    parser.add_argument("--mode", choices=["full", "terminal"], default="terminal")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    outputs = [args.output.resolve(), args.output.with_suffix(".npz").resolve()]
    if outputs[0] == outputs[1] or args.input.resolve() in outputs:
        parser.error("output paths must be distinct and must not overwrite the input")
    torch.set_num_threads(8)
    with np.load(args.input, allow_pickle=False) as archive:
        data = create_voxeldata(
            archive["values"],
            dims=("time", "k", "j", "i"),
            voxel_to_world=archive["voxel_to_world"],
            **{dim: archive[dim] for dim in ["time", "k", "j", "i"]},
        )
    check = (
        rigid._all_frames_stopped if args.mode == "terminal" else lambda status: False
    )
    seconds = []
    with patch.object(rigid, "_all_frames_stopped", check):
        for _ in range(4):
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            result = register(
                data,
                return_transforms=True, device=args.device,
                batch_size=args.batch_size,
                convergence_minimum_value=args.minimum,
                progress=False,
            )
            if args.device == "cuda":
                torch.cuda.synchronize()
            seconds.append(perf_counter() - start)
    np.testing.assert_array_equal(result.transforms.values[0], np.eye(4))
    assert np.all(result.diagnostics.n_iterations.values <= 100)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        transforms=result.transforms.values,
        registered=result.registered.values,
        iterations=result.diagnostics.n_iterations.values,
        status=result.diagnostics.status.values,
    )
    record = {
        "input": str(args.input),
        "device": args.device,
        "batch_size": args.batch_size,
        "minimum": args.minimum,
        "mode": args.mode,
        "threads": 8,
        "cold_seconds": seconds[0],
        "warm_seconds": seconds[1:],
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(args.output, seconds[1:])


if __name__ == "__main__":
    main()
