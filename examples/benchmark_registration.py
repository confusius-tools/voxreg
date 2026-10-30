"""Compare time, host RAM and CUDA memory in isolated processes (Linux only).

Inputs are a frozen NPZ containing values, time, k/j/i and voxel_to_world. ITK and
FireANTs are used directly, not through ConfUSIus registration. Without --workload,
the historical axis-aligned slice protocol is retained. --workload pair or series
selects the shared standardized protocol in benchmark_workloads.py, including 3D
and independent geometry. Timing and sampled memory use separate sequential processes.
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep

_INPUT = None
_FIXED = None
_SCALE = None
_ITK_THREADS = 1


def load_input(path: Path):
    """Load original float32 samples and derive array-position world geometry."""
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        values = archive["values"]
        if np.iscomplexobj(values):
            raise ValueError("Power intensities must be real, not complex IQ samples.")
        raw = values.astype(np.float32)
        affine = archive["voxel_to_world"].copy()
        coords = {dim: archive[dim] for dim in ("time", "k", "j", "i")}
    if (
        raw.ndim != 4
        or raw.shape[1] != 1
        or raw.shape[0] < 2
        or min(raw.shape[-2:]) < 2
        or not np.isfinite(raw).all()
        or raw.min() < 0
    ):
        raise ValueError("Expected finite nonnegative power, shape (time, 1, j, i).")
    conversion = np.eye(4)
    for axis, dim in enumerate(("k", "j", "i")):
        values = coords[dim]
        conversion[axis, 3] = values[0]
        if len(values) > 1:
            differences = np.diff(values)
            if not np.allclose(differences, differences[0]):
                raise ValueError("Native coordinates must be uniform.")
            conversion[axis, axis] = differences[0]
    position_affine = affine @ conversion
    if not np.allclose(
        position_affine[:3, :3], np.diag(np.diag(position_affine[:3, :3]))
    ) or np.any(np.diag(position_affine) <= 0):
        raise ValueError(
            "This comparison requires positive, axis-aligned spatial geometry."
        )
    if raw.max() <= 0 or np.ptp(raw[0]) == 0:
        raise ValueError(
            "The reference must have meaningful spatial intensity variance."
        )
    return raw, affine, position_affine, coords


def itk_image(values, affine):
    """Construct a double-precision ITK image in physical x/y coordinates."""
    import itk
    import numpy as np

    image = itk.image_from_array(np.asarray(values, dtype=np.float64))
    image.SetSpacing((float(affine[2, 2]), float(affine[1, 1])))
    image.SetOrigin((float(affine[2, 3]), float(affine[1, 3])))
    return image


def initialize_itk(path: str, work_units: int = 1) -> None:
    """Initialize immutable resident inputs in one ITK process."""
    global _INPUT, _FIXED, _SCALE, _ITK_THREADS
    import itk
    import numpy as np

    _ITK_THREADS = work_units
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(work_units)
    _INPUT = load_input(Path(path))
    raw, _, affine, _ = _INPUT
    _SCALE = float(raw.max())
    _FIXED = itk_image(
        10 * np.log10(np.maximum(raw[0, 0].astype(float) / _SCALE, 1e-12)), affine
    )


def register_itk_frame(index: int):
    """Fit one physical Euler2D pull transform and resample original power."""
    import itk
    import numpy as np

    raw, _, affine, _ = _INPUT
    frame = raw[index, 0]
    fit = 10 * np.log10(np.maximum(frame.astype(float) / _SCALE, 1e-12))
    moving = itk_image(fit, affine)
    image_type = type(moving)
    metric = itk.CorrelationImageToImageMetricv4[image_type, image_type].New()
    metric.SetUseFixedImageGradientFilter(False)
    metric.SetUseMovingImageGradientFilter(False)
    metric.SetMaximumNumberOfWorkUnits(_ITK_THREADS)
    transform = itk.Euler2DTransform[itk.D].New()
    center = (affine @ [0, (frame.shape[0] - 1) / 2, (frame.shape[1] - 1) / 2, 1])[
        [2, 1]
    ]
    transform.SetCenter(tuple(center))
    optimizer = itk.GradientDescentOptimizerv4Template[itk.D].New(
        LearningRate=0.1,
        NumberOfIterations=100,
        MinimumConvergenceValue=1e-6,
        ConvergenceWindowSize=10,
        DoEstimateLearningRateOnce=False,
        DoEstimateLearningRateAtEachIteration=False,
    )
    optimizer.SetScalesEstimator(
        itk.RegistrationParameterScalesFromPhysicalShift[type(metric)].New(
            Metric=metric, TransformForward=True
        )
    )
    registration = itk.ImageRegistrationMethodv4[image_type, image_type].New(
        FixedImage=_FIXED,
        MovingImage=moving,
        Metric=metric,
        Optimizer=optimizer,
        InitialTransform=transform,
        InPlace=True,
    )
    registration.SetNumberOfWorkUnits(_ITK_THREADS)
    registration.SetNumberOfLevels(1)
    registration.SetShrinkFactorsPerLevel([1])
    registration.SetSmoothingSigmasPerLevel([0.0])
    registration.Update()
    original = itk_image(frame, affine)
    resampler = itk.ResampleImageFilter[image_type, image_type].New(
        Input=original,
        ReferenceImage=_FIXED,
        UseReferenceImage=True,
        Transform=transform,
        DefaultPixelValue=float(frame.min()),
        Interpolator=itk.LinearInterpolateImageFunction[image_type, itk.D].New(),
    )
    resampler.SetNumberOfWorkUnits(_ITK_THREADS)
    resampler.Update()
    matrix = np.eye(4)
    rotation = np.asarray(transform.GetMatrix())
    offset = np.asarray(transform.TransformPoint((0.0, 0.0)))
    matrix[np.ix_([2, 1], [2, 1])] = rotation
    matrix[[2, 1], 3] = offset
    return (
        itk.array_from_image(resampler.GetOutput()).astype(np.float32),
        matrix,
        int(optimizer.GetCurrentIteration()),
    )


def run_itk(raw, pool=None):
    """Run serial or process-parallel ITK; include result transport/resampling."""
    import numpy as np

    results = (
        list(map(register_itk_frame, range(1, len(raw))))
        if pool is None
        else list(pool.map(register_itk_frame, range(1, len(raw)), chunksize=1))
    )
    output = np.concatenate((raw[:1], np.array([item[0] for item in results])[:, None]))
    matrices = np.concatenate(
        (np.eye(4)[None], np.array([item[1] for item in results]))
    )
    steps = np.array([0, *[item[2] for item in results]])
    return output, matrices, steps


def run_fireants(raw, affine, device: str, batch_size: int):
    """Fit native float32 FireANTs rigid transforms and resample original power.

    A second constant image channel supplies warped validity for global squared
    NCC. It excludes out-of-grid samples without modifying FireANTs internals.
    Preprocessing, image construction, transfers and host outputs are included.
    """
    import numpy as np
    import SimpleITK as sitk
    import torch
    import torch.nn.functional as F
    from fireants.io.image import BatchedImages, Image
    from fireants.registration.rigid import RigidRegistration

    db = 10 * np.log10(np.maximum(raw[:, 0] / raw.max(), 1e-12))

    def image(values):
        channels = np.stack((values, np.ones_like(values)), axis=-1)
        result = sitk.GetImageFromArray(channels, isVector=True)
        result.SetSpacing((float(affine[2, 2]), float(affine[1, 1])))
        result.SetOrigin((float(affine[2, 3]), float(affine[1, 3])))
        return Image(result, device=device)

    reference = image(db[0])
    outputs, transforms, steps = [raw[:1]], [np.eye(4)[None]], [0]
    for start in range(1, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        count = stop - start
        fixed = BatchedImages([reference] * count, optimize_memory=False)
        moving = BatchedImages([image(values) for values in db[start:stop]])
        calls = 0

        def loss(moved, target):
            nonlocal calls
            calls += 1
            weights = (moved[:, 1] >= 1 - 1e-5).to(moved.dtype)
            n = weights.sum(dim=(1, 2), keepdim=True).clamp_min(2)
            x, y = moved[:, 0], target[:, 0]
            x = (x - (x * weights).sum(dim=(1, 2), keepdim=True) / n) * weights
            y = (y - (y * weights).sum(dim=(1, 2), keepdim=True) / n) * weights
            covariance = (x * y).sum(dim=(1, 2))
            variance = x.square().sum(dim=(1, 2)) * y.square().sum(dim=(1, 2))
            return -(covariance.square() / variance.clamp_min(1e-12)).sum()

        registration = RigidRegistration(
            scales=[1],
            iterations=[100],
            fixed_images=fixed,
            moving_images=moving,
            loss_type="custom",
            custom_loss=loss,
            optimizer="Adam",
            optimizer_lr=0.01,
            scaling=False,
            blur=False,
            max_tolerance_iters=101,
            progress_bar=False,
        )
        registration.optimize()
        if calls != 100:
            raise RuntimeError(f"Expected 100 FireANTs updates, got {calls}.")
        with torch.no_grad():
            xy = registration.get_rigid_matrix().cpu().numpy()
            world = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            world[np.ix_(np.arange(count), [2, 1], [2, 1])] = xy[:, :2, :2]
            world[:, [2, 1], 3] = xy[:, :2, 2]
            grid = registration.get_warped_coordinates(fixed, moving)
            original = torch.as_tensor(raw[start:stop], device=device)
            sampled = F.grid_sample(
                original, grid, align_corners=True, padding_mode="border"
            )
            valid = ((grid >= -1) & (grid <= 1)).all(dim=-1)
            fill = original.amin(dim=(1, 2, 3))[:, None, None, None]
            sampled = torch.where(valid[:, None], sampled, fill)
            outputs.append(sampled.cpu().numpy())
        transforms.append(world)
        steps.extend([calls] * count)
    return np.concatenate(outputs), np.concatenate(transforms), np.array(steps)


def rss_kib(pid: int) -> int:
    """Read current Linux resident size without importing a monitoring package."""
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        return next(int(line.split()[1]) for line in lines if line.startswith("VmRSS:"))
    except OSError, StopIteration:
        return 0


def tree_memory(pid: int) -> tuple[int, int, set[int]]:
    """Sample process-tree RSS and PSS; PSS apportions shared physical pages."""
    pending, seen, rss, pss = [pid], set(), 0, 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        rss += rss_kib(current)
        try:
            for line in Path(f"/proc/{current}/smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    pss += int(line.split()[1])
            for children in Path(f"/proc/{current}/task").glob("*/children"):
                pending.extend(map(int, children.read_text().split()))
        except OSError, ProcessLookupError:
            continue
    return rss, pss, seen


def gpu_memory(pids: set[int]) -> int:
    """Sample nvidia-smi process VRAM, including CUDA context/driver allocations."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return sum(
            int(memory.strip())
            for line in result.stdout.splitlines()
            for pid, memory in [line.split(",")]
            if int(pid.strip()) in pids
        )
    except OSError, ValueError, subprocess.SubprocessError:
        return 0


def worker(args) -> None:
    """Execute one backend in a clean interpreter and save labeled outputs."""
    if getattr(args, "workload", None):
        from benchmark_workloads import standard_worker

        return standard_worker(args, measure)
    setup = perf_counter()
    raw, affine, position_affine, coords = load_input(args.input)
    fireants = args.case.startswith("fireants-")
    cuda = "cuda" in args.case.split("-")
    pool = None
    metadata = {"case": args.case, "input_shape": list(raw.shape), "pid": os.getpid()}
    if args.case.startswith("itk"):
        import itk

        metadata["itk"] = itk.Version.GetITKVersion()
        workers = int(args.case.split("-")[1])
        if workers == 1:
            initialize_itk(str(args.input), 8 if args.case.endswith("-t8") else 1)
        else:
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_itk,
                initargs=(str(args.input),),
            )

        def run():
            return run_itk(raw, pool)
    else:
        import torch

        batch = int(args.case.split("-")[2 if fireants else 1])
        threads = 8 if args.case.endswith("-t8") else 1
        torch.set_num_threads(threads)
        metadata.update(torch=torch.__version__, cpu_threads=threads, batch_size=batch)
        if fireants:
            from importlib.metadata import version

            metadata.update(
                fireants=version("fireants"),
                simpleitk=version("SimpleITK"),
                precision="native float32",
                optimizer="Adam",
                learning_rate=0.01,
                iterations=100,
                metric="valid-support global squared NCC via custom loss",
            )

            def run():
                return run_fireants(
                    raw, position_affine, "cuda" if cuda else "cpu", batch
                )
        else:
            from confusius import create_voxeldata

            from voxreg import register

            data = create_voxeldata(
                raw, dims=("time", "k", "j", "i"), **coords, voxel_to_world=affine
            )

            def run():
                result = register(
                    data,
                    return_transforms=True,
                    device="cuda" if cuda else "cpu",
                    batch_size=batch,
                    progress=False,
                    convergence_window_size=101,
                )
                return (
                    result.registered.values,
                    result.transforms.values,
                    result.diagnostics.n_iterations.values,
                )

    metadata["setup_seconds"] = perf_counter() - setup
    measure(args, run, metadata, pool)


def power_state():
    """Read optional Linux power/profile endpoints; never change machine settings."""
    root = Path("/sys/class/power_supply")
    paths = [Path("/sys/firmware/acpi/platform_profile")]
    for field in ("type", "online", "status"):
        paths.extend(sorted(root.glob(f"*/{field}")))
    snapshot = {}
    for path in paths:
        try:
            snapshot[str(path)] = path.read_text().strip()
        except OSError:
            snapshot[str(path)] = None
    return snapshot


def measure(args, run, metadata, pool=None):
    """Shared timing/memory boundary for legacy and standardized workloads."""
    import numpy as np

    cuda = "cuda" in args.case.split("-")
    fireants = args.case.startswith("fireants-")
    if cuda:
        import torch
    if getattr(args, "workload", None):
        metadata["power_before_measurements"] = power_state()
    metadata["loaded_baseline_rss_mib"] = rss_kib(os.getpid()) / 1024
    start = perf_counter()
    result = run()
    if cuda:
        torch.cuda.synchronize()
    metadata["cold_call_seconds"] = perf_counter() - start
    del result
    if fireants:
        from fireants.interpolator import fireants_interpolator

        metadata["fireants_fused_ops"] = fireants_interpolator.use_ffo
    metadata["warm_root_baseline_rss_mib"] = rss_kib(os.getpid()) / 1024
    rss, pss, _ = tree_memory(os.getpid())
    metadata["warm_tree_baseline_rss_mib"] = rss / 1024
    metadata["warm_tree_baseline_pss_mib"] = pss / 1024
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        metadata["gpu"] = torch.cuda.get_device_name()
    elapsed = []
    result = None
    for _ in range(args.repeats if args.phase == "time" else 1):
        result = None
        if cuda:
            torch.cuda.synchronize()
        start = perf_counter()
        result = run()
        if cuda:
            torch.cuda.synchronize()
        elapsed.append(perf_counter() - start)
    arrays = (
        result
        if isinstance(result, dict)
        else dict(
            zip(("registered", "transforms", "n_iterations"), result, strict=True)
        )
    )
    metadata["warm_seconds"] = elapsed
    if getattr(args, "workload", None):
        metadata["n_iterations"] = arrays["n_iterations"].tolist()
        metadata["status"] = arrays["status"].tolist()
        if (
            not np.isfinite(arrays["transforms"]).all()
            or not np.isfinite(arrays["registered"]).all()
        ):
            raise ValueError("Backend returned nonfinite transforms or intensities.")
    if cuda:
        metadata["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        metadata["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    if getattr(args, "workload", None):
        metadata["power_after_measurements"] = power_state()
    args.output.with_suffix(".measure_done").touch()
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        **arrays,
    )
    if pool is not None:
        pool.shutdown()
    args.output.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def score(input_path: Path, files: dict[str, Path]) -> dict:
    """Score each method on one common support, with independent interpolation."""
    import numpy as np
    from scipy.ndimage import map_coordinates

    raw, _, affine, _ = load_input(input_path)
    db = 10 * np.log10(np.maximum(raw[:, 0].astype(float) / raw.max(), 1e-12))
    j, i = np.indices(db.shape[1:])
    grid = np.stack((np.zeros_like(i), j, i, np.ones_like(i)))
    mappings = {}
    common = np.ones(db.shape, dtype=bool)
    for case, path in files.items():
        with np.load(path, allow_pickle=False) as archive:
            mapped = np.einsum(
                "tab,bhw->tahw",
                np.linalg.inv(affine) @ archive["transforms"] @ affine,
                grid,
            )
        common &= (
            (np.abs(mapped[:, 0]) < 1e-5)
            & (mapped[:, 1] >= -1e-5)
            & (mapped[:, 1] <= db.shape[1] - 1 + 1e-5)
            & (mapped[:, 2] >= -1e-5)
            & (mapped[:, 2] <= db.shape[2] - 1 + 1e-5)
        )
        mappings[case] = mapped
    quality = {}
    for case, mapped in mappings.items():
        scores = []
        for frame, mask in enumerate(common):
            if mask.sum() < 2:
                raise ValueError("Insufficient common support for scoring.")
            sampled = map_coordinates(
                db[frame], mapped[frame, 1:3], order=1, mode="nearest", prefilter=False
            )
            scores.append(np.corrcoef(db[0][mask], sampled[mask])[0, 1] ** 2)
        quality[case] = {
            "median_ncc2": float(np.median(scores[1:])),
            "min_ncc2": float(min(scores[1:])),
            "common_fraction_median": float(np.median(common.mean(axis=(1, 2)))),
        }
    return quality


def main() -> None:
    """Run time and memory passes sequentially, then compare common quality."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/notebook/input.npz")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--workload",
        choices=["pair", "series"],
        help="Use the standardized pair/series protocol; omission preserves the historical slice protocol.",
    )
    parser.add_argument("--policy", choices=["fixed", "convergence"], default="fixed")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--intensity-scaling", choices=["none", "db"], default="none")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
    )
    legacy_cases = [
        "itk-1",
        "itk-1-t8",
        "itk-8",
        "cpu-1",
        "cpu-32",
        "cpu-32-t8",
        "cpu-120",
        "cuda-1",
        "cuda-32",
        "cuda-120",
        "fireants-cpu-32",
        "fireants-cpu-32-t8",
        "fireants-cuda-32",
        "fireants-cuda-120",
    ]
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument(
        "--phase", choices=["time", "memory"], default="time", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.max_iterations < 1:
        parser.error("repeats and max-iterations must be positive")
    default_batch = 32 if args.workload == "series" else 1
    args.cases = args.cases or (
        [
            "itk-1-t8",
            f"cpu-{default_batch}-t8",
            f"cuda-{default_batch}",
            f"fireants-cpu-{default_batch}-t8",
            f"fireants-cuda-{default_batch}",
        ]
        if args.workload
        else legacy_cases
    )
    if args.workload:
        import re

        # shortcut: standardized ITK is single-process; add pooling before claiming best series throughput.
        pattern = r"(?:itk-1(?:-t8)?|(?:fireants-)?(?:cpu|cuda)-[1-9][0-9]*(?:-t8)?)"
        if any(
            re.fullmatch(pattern, case) is None for case in [args.case] if case
        ) or any(re.fullmatch(pattern, case) is None for case in args.cases):
            parser.error(
                "Unsupported standardized case; ITK uses one process with 1 or 8 internal threads."
            )
        if len(set(args.cases)) != len(args.cases):
            parser.error("Cases must be unique.")
    if args.case:
        worker(args)
        return
    if not Path("/proc/self/smaps_rollup").exists():
        parser.error("Memory measurements currently require Linux /proc.")
    if args.workload and args.output.exists() and any(args.output.iterdir()):
        parser.error(
            "Standardized output directory must be empty; historical results will not be overwritten."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "python": sys.version,
        "input": str(args.input.resolve()),
        "notes": [
            "Warm end-to-end calls on resident host inputs, including fitting preprocessing and original-power resampling/return to host; no data loading or output-file compression in the timed region.",
            "ITK caches its immutable fixed fitting image; process workers cache inputs; scheduling and result IPC are timed. ITK uses 1 work unit per worker (8 for the explicit -t8 single-process control), global NCC, Euler2D, GD rate .1 (validated on synthetic motions; rate 1 overshot), 100 maximum iterations, 1e-6 convergence window 10. voxreg uses Adam rate .1 and a convergence window of 101 to retain the fixed 100-update protocol (as for FireANTs); this is not the default early-freeze policy.",
            "Separate fresh processes for time and memory passes. RSS/PSS sampled during setup, cold and warm calls at ~20ms intervals (longer for CUDA nvidia-smi queries); sampled peaks may miss brief transients. RSS sums double-count shared pages; PSS apportions them. CUDA allocator counters are exact peaks since warmup, not full context memory; nvidia-smi is sampled process VRAM.",
            "GPU cold-call time includes CUDA setup; ITK parallel cold-call time includes worker creation/template loading. Warm timings exclude those startup costs. Check for competing workloads before running; this script does not enforce machine idleness. Normal desktop activity may affect timings.",
            "FireANTs uses native float32 2D rigid/Adam rate .01 for 100 updates, no multiresolution or scaling; a custom per-frame global squared NCC excludes invalid samples via a warped ones channel. Original-power resampling/host return and per-call image construction/upload are included. Selected optional fused-ops backend is recorded per worker.",
            "Different optimizers/stopping rules: report quality alongside runtime, not an equal-quality speedup claim. Scores exclude reference frame, and use common sample-center support across all cases.",
        ],
        "results": {},
    }
    if args.workload:
        import platform

        from benchmark_workloads import fingerprint, fit_images, load_dataset

        data = load_dataset(args.input, args.workload)
        fit_images(data, args.intensity_scaling)
        summary["power_before_suite"] = power_state()
        summary["system"] = {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cpu_model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            ),
            "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "load_average_at_start": os.getloadavg(),
            "clocks": "unfixed",
        }
        summary.update(
            schema_version=2,
            protocol="standardized-v1",
            workload=args.workload,
            policy=args.policy,
            max_iterations=args.max_iterations,
            intensity_scaling=args.intensity_scaling,
            input_sha256=fingerprint(args.input),
            provenance=data["provenance"],
            fixed_affine_zyx=data["af"].tolist(),
            moving_affine_zyx=data["am"].tolist(),
            truth_pull_matrices_zyx=data["truth"].tolist()
            if data["truth"] is not None
            else None,
            scored_phase="time",
            source_sha256={
                str(path.relative_to(Path(__file__).resolve().parents[1])): fingerprint(
                    path
                )
                for path in [
                    Path(__file__).resolve(),
                    Path(__file__).with_name("benchmark_workloads.py").resolve(),
                    *sorted(
                        (Path(__file__).resolve().parents[1] / "src/voxreg").glob(
                            "*.py"
                        )
                    ),
                ]
            },
        )
        summary["notes"] = [
            "Pair and series share resident-host warm calls: preprocessing, fitting, original float32 resampling and host return; no loading, scoring, compression or progress in the timed region. No fixed fitting image is cached across standardized calls. Series reference is copied exactly and excluded from aggregate quality.",
            "Both policies use identity world initialization, global squared NCC and one resolution. Fixed means the same maximum update budget, not equal optimizers/evaluations/quality; guards can stop voxreg early. Convergence uses explicitly recorded backend rules; FireANTs native batch-loss slope is not ITK's monitor.",
            "Common float32 source/fitting arrays; ITK double images/metric/interpolation, voxreg mixed geometry/state/NCC64 and interpolation32, FireANTs native32. These are not precision-matched backends. ITK uses physical-shift scales/GD .1, voxreg RMS lever arms/Adam .1, FireANTs quaternion/Adam .01. Returned transforms are fixed-world-to-moving-world z/y/x pulls.",
            "Separate sequential fresh interpreters for timing and sampled memory. Memory process-tree peaks include setup/cold/warm; CUDA allocator peaks reset after cold warmup. GPU process VRAM includes context; sampling may miss transients. Interpreter/import/startup and file compression are outside warm timings; setup/cold times recorded separately.",
            "Independent SciPy float64 interpolation/scoring on joint pixel-center support, including identity. Truth errors cover the entire fixed grid in minimum-spacing pixels. Backend internal scores are not compared as common quality. Timing-phase outputs supply quality; memory-phase outputs and diagnostics are retained separately.",
            "No enforced machine idleness or fixed clocks. No initialization search, pyramids, masks, or process-parallel ITK in the standardized protocol. Historical mode retains its former cached fixed image and stopping settings.",
        ]
        del data
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    outputs = {}
    for case in args.cases:
        record = {}
        for phase in ("time", "memory"):
            path = args.output / f"{case}-{phase}.json"
            print(f"{case}: {phase}", flush=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--input",
                str(args.input.resolve()),
                "--output",
                str(path.resolve()),
                "--case",
                case,
                "--phase",
                phase,
                "--repeats",
                str(args.repeats),
            ]
            if args.workload:
                command.extend(
                    [
                        "--workload",
                        args.workload,
                        "--policy",
                        args.policy,
                        "--max-iterations",
                        str(args.max_iterations),
                        "--intensity-scaling",
                        args.intensity_scaling,
                    ]
                )
            path.with_suffix(".measure_done").unlink(missing_ok=True)
            with path.with_suffix(".log").open("w") as log:
                process = subprocess.Popen(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
                peaks = [0, 0, 0]
                if phase == "memory":
                    next_gpu = 0
                    while (
                        process.poll() is None
                        and not path.with_suffix(".measure_done").exists()
                    ):
                        rss, pss, pids = tree_memory(process.pid)
                        peaks[0], peaks[1] = max(peaks[0], rss), max(peaks[1], pss)
                        if "cuda" in case.split("-") and perf_counter() >= next_gpu:
                            peaks[2] = max(peaks[2], gpu_memory(pids))
                            next_gpu = perf_counter() + 0.25
                        sleep(0.02)
                code = process.wait()
            if code:
                raise RuntimeError(f"{case} failed; see {path.with_suffix('.log')}")
            record[phase] = json.loads(path.read_text())
            if phase == "memory":
                record[phase].update(
                    tree_peak_rss_mib=peaks[0] / 1024,
                    tree_peak_pss_mib=peaks[1] / 1024,
                    sampled_gpu_process_peak_mib=peaks[2],
                )
                path.write_text(
                    json.dumps(record[phase], indent=2, allow_nan=False) + "\n"
                )
            if not args.workload or phase == "time":
                outputs[case] = path.with_suffix(".npz")
        summary["results"][case] = record
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
    if args.workload:
        from benchmark_workloads import score_standard

        summary["quality"] = score_standard(
            args.input, outputs, args.workload, args.intensity_scaling
        )
    else:
        summary["quality"] = score(args.input, outputs)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    for case, record in summary["results"].items():
        import statistics

        t, m = record["time"], record["memory"]
        print(
            f"{case:14} {statistics.median(t['warm_seconds']):8.3f} s | RSS {m['tree_peak_rss_mib']:7.0f} MiB | PSS {m['tree_peak_pss_mib']:7.0f} MiB | CUDA alloc {m.get('cuda_peak_allocated_mib', 0):6.1f} MiB | GPU process {m['sampled_gpu_process_peak_mib']:5} MiB | NCC² {summary['quality'][case]['median_ncc2']}"
        )


if __name__ == "__main__":
    main()
