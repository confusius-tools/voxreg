"""Research CPU native-double fitting samples and separable SciPy field filtering."""

import argparse
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from native_sampler import native_sample
from scipy.ndimage import gaussian_filter

from voxreg import sampling, syn
from voxreg.fields import smooth_field

_original_sample = sampling._sample_displaced


def cpu_sample(images, normalized, interpolation):
    if images.device.type == "cpu":
        return native_sample(images, normalized, interpolation)
    return _original_sample(images, normalized, interpolation)


def scipy_smooth(field, grid, *, sigma):
    if field.device.type != "cpu" or (torch.is_grad_enabled() and field.requires_grad):
        return smooth_field(field, grid, sigma=sigma)
    if sigma == 0:
        return field
    values = gaussian_filter(
        field.detach().numpy(),
        sigma=sigma,
        radius=int(3 * sigma + 0.5),
        axes=tuple(range(2, field.ndim)),
        mode="nearest",
    )
    return torch.from_numpy(values).masked_fill(grid.boundary, 0)


@contextmanager
def patches():
    with (
        patch.object(sampling, "_sample_displaced", cpu_sample),
        patch.object(syn, "smooth_field", scipy_smooth),
    ):
        yield


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--cases", nargs="+", default=["2d-medium", "3d-medium", "3d-batch8"]
    )
    parser.add_argument("--backends", nargs="+", default=["voxreg-cpu", "fireants-cpu"])
    parser.add_argument("--jacobian-direct", action="store_true")
    parser.add_argument("--contract-inverse", action="store_true")
    parser.add_argument("--cpu-map", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        stack.enter_context(patches())
        if args.cpu_map:
            from cpu_map import cpu_map

            from voxreg import fields

            stack.enter_context(patch.object(sampling, "_map_displacement", cpu_map))
            stack.enter_context(patch.object(fields, "_map_displacement", cpu_map))
        if args.jacobian_direct:
            from jacobian_direct import jacobian_direct

            stack.enter_context(
                patch.object(syn, "jacobian_determinant", jacobian_direct)
            )
        if args.contract_inverse:
            from contract_inverse import contract_inverse

            stack.enter_context(patch.object(syn, "invert_field", contract_inverse))
        if args.tests:
            import pytest

            raise SystemExit(
                pytest.main(
                    [
                        "-q",
                        "tests/test_syn.py",
                        "tests/test_optional_transforms.py",
                        "tests/test_sampling.py",
                    ]
                )
            )
        if args.output is None:
            parser.error("--output required unless --tests")
        from compare import run

        run(args)
        report_path = args.output / "results.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if args.contract_inverse:
                from contract_inverse import COUNTS

                report["inverse_solver_counts"] = COUNTS
            report["cpu_experiment"] = {
                "fitting_sampler": "native-double",
                "state_filter": "scipy-float64",
                "production": False,
                "jacobian_direct": args.jacobian_direct,
                "contract_inverse": args.contract_inverse,
                "cpu_map": args.cpu_map,
            }
            report_path.write_text(json.dumps(report, indent=2) + "\n")
