"""Run the test suite with an isolated sampler or compiled-NCC substitution."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import torch
from affine_probe import make_sampler
from cpu_ablation import clone_function, fit32_optimizer, masked_ncc

from voxreg import rigid, sampling


def main():
    """Scope the substitution to this process and preserve pytest's exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["production", "affine64", "affine32", "elementwise64"]
    )
    ncc_modes = parser.add_mutually_exclusive_group()
    ncc_modes.add_argument("--compiled-ncc", action="store_true")
    ncc_modes.add_argument("--masked-ncc", action="store_true")
    ncc_modes.add_argument("--fit32", action="store_true")
    ncc_modes.add_argument("--ncc32", action="store_true")
    args = parser.parse_args()
    sampler = (
        sampling.sample_rigid
        if args.mode == "production"
        else make_sampler(
            args.mode == "affine32", elementwise=args.mode == "elementwise64"
        )
    )
    tests = Path(__file__).resolve().parents[2] / "tests"
    with ExitStack() as patches:
        patches.enter_context(patch.object(rigid, "sample_rigid", sampler))
        patches.enter_context(patch.object(sampling, "sample_rigid", sampler))
        if args.compiled_ncc:
            ncc = torch.compile(sampling.global_ncc, fullgraph=True, dynamic=True)
            patches.enter_context(patch.object(rigid, "global_ncc", ncc))
        if args.fit32:
            directory = patches.enter_context(TemporaryDirectory())
            optimizer = fit32_optimizer(Path(directory) / "ncc.py")
            patches.enter_context(patch.object(rigid, "_optimize", optimizer))
        if args.ncc32:
            directory = patches.enter_context(TemporaryDirectory())
            ncc = clone_function(
                sampling.global_ncc,
                [
                    (
                        "moved.to(torch.float64), fixed.to(torch.float64)",
                        "moved.to(torch.float32), fixed.to(torch.float32)",
                    )
                ],
                vars(sampling),
                Path(directory) / "ncc.py",
            )
            patches.enter_context(patch.object(rigid, "global_ncc", ncc))
        if args.masked_ncc:
            directory = patches.enter_context(TemporaryDirectory())
            ncc = masked_ncc(Path(directory) / "ncc.py")
            patches.enter_context(patch.object(rigid, "global_ncc", ncc))
            patches.enter_context(patch.object(sampling, "global_ncc", ncc))
        return pytest.main([str(tests), "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
