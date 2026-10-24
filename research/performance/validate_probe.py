"""Run the public-API test suite with only the diagnostic sampler replaced."""

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
from affine_probe import make_sampler

from voxreg import rigid, sampling


def main():
    """Scope the substitution to this process and preserve pytest's exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["affine64", "affine32", "elementwise64"])
    args = parser.parse_args()
    sampler = make_sampler(
        args.mode == "affine32", elementwise=args.mode == "elementwise64"
    )
    tests = Path(__file__).resolve().parents[2] / "tests"
    with (
        patch.object(rigid, "sample_rigid", sampler),
        patch.object(sampling, "sample_rigid", sampler),
    ):
        return pytest.main([str(tests), "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
