"""Run shared native comparison cases through the research inverse backend."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import landemard
from compare import run as compare
from profile_syn import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--cases", nargs="+", default=["2d-medium", "3d-medium", "3d-batch8"]
    )
    parser.add_argument(
        "--backends", nargs="+", default=["voxreg-cuda", "fireants-cuda"]
    )
    args = parser.parse_args()
    args.compile = ["loss", "update", "inverse"] if args.compile else []
    args.backend = "inductor-native-grid"
    args.benchmark = True
    args.newton3d = True
    args.frames = None
    with patch.object(landemard, "run", compare):
        run(args)
    report = json.loads((args.output / "results.json").read_text())
    for case in report["cases"].values():
        for arm in case["backends"].values():
            assert arm.get("truth_gate_pass"), arm
