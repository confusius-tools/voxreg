"""Explore one native atlas fit with a composite and metric curve; cached inputs only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--moving", choices=["recording", "template"], default="recording"
    )
    parser.add_argument(
        "--transform", choices=["translation", "rigid", "affine"], default="affine"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.headless:
        import matplotlib

        matplotlib.use("Agg")
    from atlas_inputs import inputs
    from plot_registration import register_with_plot

    template, recording, placement, paths, _ = inputs()
    moving, fixed, seed = (
        (recording, template, np.linalg.inv(placement))
        if args.moving == "recording"
        else (template, recording, placement)
    )
    result, figure = register_with_plot(
        moving,
        fixed,
        show=not args.headless,
        transform=args.transform,
        initialization=seed,
        intensity_scaling="none",
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        device=args.device,
        progress=False,
    )
    figure.savefig(args.output / "registration.png", dpi=150)
    np.savez_compressed(
        args.output / "arrays.npz",
        registered=result.registered.values,
        valid=result.valid.values,
        transforms=result.transforms.values,
        updates=figure.axes[1].lines[0].get_xdata(),
        objective=figure.axes[1].lines[0].get_ydata(),
    )
    objective = float(result.diagnostics.objective.item())
    record = {
        "moving": args.moving,
        "transform": args.transform,
        "device": args.device,
        "seed": seed.tolist(),
        "moving_shape": list(moving.shape),
        "fixed_shape": list(fixed.shape),
        "learning_rate": args.learning_rate,
        "max_iterations": args.max_iterations,
        "steps": int(result.diagnostics.n_iterations.item()),
        "status": result.diagnostics.status.item(),
        "returned_objective": objective if np.isfinite(objective) else None,
        "metric_interval": 10,
        "expanded_data": False,
        "plot_enabled_not_a_benchmark": True,
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("plot_registration.py"),
                Path(__file__).with_name("atlas_inputs.py"),
                *sorted(
                    (Path(__file__).resolve().parents[2] / "src/voxreg").glob("*.py")
                ),
            ]
        },
    }
    (args.output / "results.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print(args.output / "registration.png")
    if not args.headless:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
