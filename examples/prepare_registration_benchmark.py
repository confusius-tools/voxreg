"""Freeze analytic 3D pairs or a known-pose ConfUSIus template control (not session alignment)."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def motion():
    """One declared small physical pull, identical for every resolution/backend."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(
        np.array([0.015, -0.02, 0.01])[::-1]
    ).as_matrix()[::-1, ::-1]
    matrix[:3, 3] = [0.1, -0.08, 0.12]
    return matrix


def synthetic(size, independent):
    """Paint both grids analytically; never generate motion with a fitting sampler."""
    if size < 32:
        raise ValueError("size must be at least 32 (FireANTs full-resolution minimum)")
    spacing = np.array([8.0, 10.0, 7.0]) / (size - 1)

    def affine(shape, angles, step):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler(
            "xyz", angles, degrees=True
        ).as_matrix() @ np.diag(step)
        result[:3, 3] = -result[:3, :3] @ ((np.array(shape) - 1) / 2)
        return result

    shape = (size,) * 3
    moving_shape = (size + 16,) * 3 if independent else shape
    af = affine(shape, [3, -2, 4], spacing)
    am = (
        affine(moving_shape, [-1, 2, 1], spacing * [1.04, 0.96, 1.02])
        if independent
        else af.copy()
    )
    rng = np.random.default_rng(871)
    locations = rng.uniform([-2.6, -3.2, -2.2], [2.6, 3.2, 2.2], (20, 3))
    widths = rng.uniform(0.35, 0.9, (20, 3))
    amplitudes = rng.uniform(0.5, 2, 20)

    def paint(shape, affine):
        lattice = np.ogrid[tuple(slice(0, n) for n in shape)]
        points = [
            affine[row, 3] + sum(affine[row, col] * lattice[col] for col in range(3))
            for row in range(3)
        ]
        result = np.full(shape, 0.02)
        for location, width, amplitude in zip(
            locations, widths, amplitudes, strict=True
        ):
            distance = sum(
                ((p - c) / s) ** 2
                for p, c, s in zip(points, location, width, strict=True)
            )
            result += amplitude * np.exp(-0.5 * distance)
        return result.astype(np.float32)

    truth = motion()
    return {
        "fixed": paint(shape, af),
        "moving": paint(moving_shape, np.linalg.inv(truth) @ am),
        "fixed_affine": af,
        "moving_affine": am,
        "truth": truth,
    }, {
        "kind": "analytic-known-motion",
        "seed": 871,
        "size": size,
        "independent_grids": independent,
        "physical_extent_mm": [8, 10, 7],
        "motion_generation": "independent continuous Gaussian mixture; no interpolation",
    }


def template_pair(path):
    """Real 3D intensity structure with synthetic rigid pose, not a natural pair."""
    import confusius as cf

    from voxreg import position_to_world

    data = cf.load(path, coordinate_affine="qform").compute()
    if data.dims != ("k", "j", "i") or min(data.shape) < 32:
        raise ValueError("Template must be a genuine spatial-only 3D volume")
    raw = data.values.astype(np.float32)
    af = position_to_world(data)
    center = (af @ np.r_[(np.array(raw.shape) - 1) / 2, 1])[:3]
    truth = motion()
    truth[:3, 3] -= (truth[:3, :3] - np.eye(3)) @ center
    with Path(path).open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "fixed": raw,
        "moving": raw.copy(),
        "fixed_affine": af,
        "moving_affine": truth @ af,
        "truth": truth,
    }, {
        "kind": "template-known-pose",
        "source": str(Path(path).resolve()),
        "source_sha256": source_hash,
        "source_dtype": str(data.dtype),
        "geometry": "NIfTI qform via ConfUSIus; known physical pose changes moving affine, not values",
        "warning": "Real intensity structure, synthetic motion; not independent sessions or biological truth",
        "example_data_reference": "Suggested source: Pepe, Mariani et al. 2026, doi:10.64898/2026.02.05.704055, CC BY 4.0; ConfUSIus registration example 03. Citation/license apply to that source, not arbitrary user templates.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--size",
        type=int,
        help="Synthetic cube size; default 64. Use 64 and 128 for the initial suite.",
    )
    parser.add_argument("--independent-grids", action="store_true")
    parser.add_argument(
        "--template",
        type=Path,
        help="Existing 3D NIfTI, e.g. the template fetched by the ConfUSIus example; no implicit download.",
    )
    args = parser.parse_args()
    if args.output.suffix != ".npz" or args.output.exists():
        parser.error("output must be a new .npz file")
    if args.template and (args.size is not None or args.independent_grids):
        parser.error(
            "template control uses its native grid; size/independent-grids apply only to synthetic inputs"
        )
    if args.template and args.template.resolve() == args.output.resolve():
        parser.error("output must not overwrite input")
    try:
        arrays, provenance = (
            template_pair(args.template)
            if args.template
            else synthetic(
                args.size if args.size is not None else 64, args.independent_grids
            )
        )
    except ValueError as error:
        parser.error(str(error))
    if any(not np.isfinite(value).all() for value in arrays.values()):
        parser.error("generated inputs must be finite")
    provenance.update(
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_policy="Common float32 benchmark inputs; not a dtype-preservation benchmark",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays, provenance=json.dumps(provenance))
    print(
        f"Wrote {args.output}: fixed {arrays['fixed'].shape}, moving {arrays['moving'].shape}"
    )


if __name__ == "__main__":
    main()
