"""Run unchanged registration truth/precision tests through experimental CUDA phases."""

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import profile_syn
import pytest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compile",
        nargs="*",
        choices=["loss", "update", "inverse"],
        default=["loss", "update", "inverse"],
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["tests/test_syn.py", "tests/test_optional_transforms.py"],
    )
    parser.add_argument(
        "--backend",
        choices=["inductor", "inductor-native-grid", "cudagraphs"],
        default="inductor",
    )
    parser.add_argument("--gradient32", action="store_true")
    parser.add_argument("--inverse-rsqrt", action="store_true")
    parser.add_argument("--contract-inverse", action="store_true")
    parser.add_argument("--cpu-fast", action="store_true")
    parser.add_argument("--smooth-batch", action="store_true")
    parser.add_argument("--inductor-graphs", action="store_true")
    parser.add_argument("--jacobian-direct", action="store_true")
    parser.add_argument("--cpu-map", action="store_true")
    parser.add_argument("--maxnorm-squared", action="store_true")
    parser.add_argument("--box-filter", action="store_true")
    parser.add_argument("--newton-inverse", action="store_true")
    parser.add_argument("--newton3d", action="store_true")
    parser.add_argument("--fused-arithmetic", action="store_true")
    args = parser.parse_args()

    def check(_):
        raise SystemExit(pytest.main(["-q", *args.tests]))

    # Registration-only aliases are patched; standalone primitive gradchecks stay eager.
    with patch.object(profile_syn, "_run", check):
        profile_syn.run(
            SimpleNamespace(
                compile=args.compile,
                benchmark=False,
                backend=args.backend,
                gradient32=args.gradient32,
                inverse_rsqrt=args.inverse_rsqrt,
                contract_inverse=args.contract_inverse,
                cpu_fast=args.cpu_fast,
                smooth_batch=args.smooth_batch,
                inductor_graphs=args.inductor_graphs,
                jacobian_direct=args.jacobian_direct,
                cpu_map=args.cpu_map,
                maxnorm_squared=args.maxnorm_squared,
                box_filter=args.box_filter,
                newton_inverse=args.newton_inverse,
                newton3d=args.newton3d,
                fused_arithmetic=args.fused_arithmetic,
            )
        )
