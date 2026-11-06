"""Shared serial supervision/reporting and the existing registration workload."""

import argparse
import fnmatch
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import _benchmark_ants as ants
import _benchmark_data as data
import numpy as np
import torch
from _benchmark_metrics import _native_motion_rms, _native_pair_rms, _native_syn_rms
from _benchmark_resample_io import LIMIT as RESAMPLE_LIMIT
from _benchmark_resident import _check, _run
from _benchmark_syn_reference import REFERENCE as SYN_REFERENCE
from rich.console import Console
from rich.table import Table

_ACTIVE = None
FRAMES = 100
ITERATIONS = 100
TRUTH_LIMIT = 0.30
PHASES = ("timing", "memory", "accuracy")
PACKAGES = (
    "torch",
    "numpy",
    "scipy",
    "xarray",
    "confusius",
    "fireants",
    "SimpleITK",
    "nibabel",
    "rich",
    "antspyx",
)


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _configs():
    cases = [
        {
            "id": f"resident-{workload}-{model}-{backend}",
            "suite": "resident",
            "workload": workload,
            "model": model,
            "backend": backend,
        }
        for workload, model, backend in itertools.product(
            data.SHAPES,
            ("rigid", "affine", "syn"),
            ("cpu", "cuda", "fireants-cpu", "fireants-cuda"),
        )
    ]
    cases += [
        {
            "id": f"native-motion-{workload}-{sampling}",
            "suite": "native",
            "workflow": "motion",
            "workload": workload,
            "sampling": sampling,
        }
        for workload, sampling in itertools.product(
            ("2d+t", "3d+t"), ("dense", "random5")
        )
    ]
    cases.append(
        {
            "id": "native-registration-3d",
            "suite": "native",
            "workflow": "registration",
            "workload": "3d",
            "sampling": "wrapper-defaults",
        }
    )
    cases += [
        {
            "id": f"native-syn-{workload}",
            "suite": "native",
            "workflow": "syn",
            "workload": workload,
            "sampling": "dense",
        }
        for workload in data.SHAPES
    ]
    return cases


def _worker(config_path, mode):
    config = json.loads(config_path.read_text())
    directory = config_path.parent
    record = {
        "mode": mode,
        "status": "running",
        "versions": _versions(),
        "power_start": data._power(),
    }
    path = directory / f"{mode}.json"
    _write(path, record)
    print(f"{config['id']} / {mode}; config: {config_path}; result: {path}", flush=True)
    try:
        # Protect the supervisor if a deliberately large configuration exhausts RAM.
        score = Path("/proc/self/oom_score_adj")
        try:
            score.write_text("500")
        except OSError:
            pass
        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)
        cuda = config.get("backend", "").endswith("cuda")
        if cuda and not torch.cuda.is_available():
            record.update(status="unavailable", reason="CUDA unavailable")
            return 0
        accuracy = mode == "accuracy"
        calls = config["runs"] if mode == "timing" else 1
        if config["suite"] == "resident":
            moving, fixed, amplitudes = data._dataset(
                data.SHAPES[config["workload"]],
                config["frames"],
                config["model"],
                config["workload"].endswith("+t"),
            )
            record["input_sha256"] = hashlib.sha256(
                memoryview(moving.values)
            ).hexdigest()
            from voxreg.rigid import _automatic_batch_size

            series = moving if "time" in moving.dims else moving.expand_dims(time=[0])
            config["batch_size"] = min(
                series.sizes["time"],
                config["batch_size"]
                or _automatic_batch_size(
                    series, fixed, config["model"], "cuda" if cuda else "cpu"
                ),
            )
            if config["backend"].startswith("fireants-"):
                import importlib

                for module in ("rigid", "affine", "syn"):
                    importlib.import_module(f"fireants.registration.{module}")
        else:
            record["inputs"] = ants._prepare(config, directory, config["frames"])
            record["input_sha256"] = record["inputs"]["moving_array_sha256"]
            recording = config["workload"].endswith("+t")
            count = config["frames"] if recording else 1
            amplitudes = 0.8 + 0.2 * np.sin(2 * np.pi * np.arange(count) / 16)
            amplitudes[0] = 0 if recording else 1
            if config["workflow"] == "syn":
                record["native_settings"] = {
                    "threads": config["threads"],
                    "reference": "supplied fixed image (frame 0 for recordings)",
                    "metric": "dense CC radius 2",
                    "transform": "SyN[0.1,3,0]",
                    "iteration_budget": [100],
                    "shrink_factors": [1],
                    "smoothing_sigmas_vox": [0],
                    "stopping": "native SyN convergence remains enabled",
                    "observed_updates": None,
                    "observed_updates_reason": "antsMotionCorr does not emit per-iteration SyN diagnostics",
                    "field_export": accuracy,
                    "fitted_frames": count,
                }
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.synchronize()
            record["cuda_device"] = {
                "name": torch.cuda.get_device_name(),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
            }
            if mode == "memory":
                torch.cuda.reset_peak_memory_stats()
        times = []
        for index in range(calls):
            run = directory / f"{mode}-{index + 1}"
            if config["suite"] == "resident":
                if cuda:
                    torch.cuda.synchronize()
                start = perf_counter()
                output, errors, details = _run(
                    moving, fixed, amplitudes, config, accuracy
                )
                if cuda:
                    torch.cuda.synchronize()
                seconds = perf_counter() - start
                _check(output, moving, config, details)
                if (
                    record.get("backend_settings", details)["batch_size"]
                    != details["batch_size"]
                ):
                    raise RuntimeError(
                        "Automatic batch size changed between timed calls"
                    )
                record["backend_settings"] = details
                del output
            else:
                run.mkdir()
                commands = ants._commands(
                    config,
                    directory,
                    run,
                    config["tools"],
                    config["frames"],
                    config["threads"],
                    1,
                    config["sampling"],
                    export=accuracy,
                )
                _write(run / "commands.json", commands)
                seconds = 0.0
                for stage, argv in commands:
                    print(f"Starting {stage}: {run.name}", flush=True)
                    with (run / f"{stage}.log").open("w") as log:
                        start = perf_counter()
                        process = subprocess.run(
                            argv, stdout=log, stderr=subprocess.STDOUT, check=False
                        )
                        seconds += perf_counter() - start
                    if process.returncode:
                        raise RuntimeError(
                            f"Native {stage} exited {process.returncode}; see {run}"
                        )
                ants._output_qc(
                    config, directory, run, config["frames"], export=accuracy
                )
                errors = None
                if accuracy:
                    shape = data.SHAPES[config["workload"]]
                    if config["workflow"] == "syn":
                        errors = _native_syn_rms(directory, run, shape, amplitudes)
                    elif config["workflow"] == "motion":
                        errors = _native_motion_rms(run, shape, amplitudes)
                    else:
                        errors = _native_pair_rms(directory, run, shape)
            if mode == "timing":
                times.append(seconds)
                record["wall_seconds"] = times
                print(
                    f"Completed timed registration {index + 1}/{calls}: {seconds:.6f} s",
                    flush=True,
                )
                _write(path, record)
            if accuracy:
                expected_count = (
                    config["frames"] if config["workload"].endswith("+t") else 1
                )
                if (
                    errors is None
                    or len(errors) != expected_count
                    or not np.isfinite(errors).all()
                ):
                    raise AssertionError("Missing/nonfinite full-workload truth errors")
                fitted_errors = (
                    errors[1:]
                    if config["suite"] == "resident"
                    and config["workload"].endswith("+t")
                    else errors
                )
                worst = max(fitted_errors)
                limit = TRUTH_LIMIT
                comparison = "<="
                record.update(
                    rms_per_frame_voxels=errors,
                    worst_frame_rms_voxels=worst,
                    truth_rms_limit_voxels=limit,
                    truth_rms_comparison=comparison,
                    accuracy_pass=worst <= limit,
                )
                print(
                    f"Worst-frame truth RMS: {worst:.6f} vox; required {comparison} {limit}",
                    flush=True,
                )
            # Native output files are retained only for the accuracy run; logs always stay.
            if config["suite"] == "native" and not accuracy:
                for output_path in run.iterdir():
                    if (
                        output_path.name != "commands.json"
                        and not output_path.name.endswith(".log")
                    ):
                        output_path.unlink()
        if mode == "memory" and cuda:
            record["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        record["status"] = (
            "quality_failed" if accuracy and not record["accuracy_pass"] else "passed"
        )
    except Exception:
        record.update(status="error", error=traceback.format_exc())
        raise
    finally:
        record["power_end"] = data._power()
        _write(path, record)
        print(f"{mode}: {record['status']}", flush=True)
    return int(record["status"] == "error")


def _group_rss(group):
    """Linux process-group RSS sum; memory sampling is never done during timing."""
    total = 0
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != group:
                continue
            for line in (directory / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total


def _stop(signum, frame):
    if _ACTIVE is not None and _ACTIVE.poll() is None:
        os.killpg(_ACTIVE.pid, signal.SIGTERM)
        try:
            _ACTIVE.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(_ACTIVE.pid, signal.SIGKILL)
            _ACTIVE.wait()
    raise KeyboardInterrupt


def _phase(script, config_path, mode, environment):
    global _ACTIVE
    directory = config_path.parent
    peak = 0
    config = json.loads(config_path.read_text())
    if config.get("worker_script"):
        script = script.parent / config["worker_script"]
    python = config.get("worker_python") or sys.executable
    with (directory / f"{mode}.log").open("w") as log:
        _ACTIVE = subprocess.Popen(
            [python, str(script), "--worker", str(config_path), "--mode", mode],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process = _ACTIVE
        while process.poll() is None:
            if mode == "memory":
                peak = max(peak, _group_rss(process.pid))
            sleep(0.02 if mode == "memory" else 0.2)
        code = process.returncode
        _ACTIVE = None
    path = directory / f"{mode}.json"
    record = (
        json.loads(path.read_text())
        if path.exists()
        else {"status": "error", "error": "Worker ended without a result"}
    )
    record["exit_code"] = code
    if code:
        record["status"] = "error"
    if mode == "memory":
        record["peak_ram_bytes"] = peak
        if "baseline_ram_bytes" in record:
            record["peak_ram_above_baseline_bytes"] = max(
                0, peak - record["baseline_ram_bytes"]
            )
    _write(path, record)
    return record


def _table(console, records, suite, runs, resampling=False):
    table = Table(
        title=(
            "Resident-array resampling"
            if resampling
            else "Resident-array registrations"
        )
        if suite == "resident"
        else "Native ANTs workflows — CLI / I/O included"
    )
    if runs != 5:
        table.title += f" — DIAGNOSTIC ({runs} timed calls)"
    table.add_column("Configuration", overflow="fold")
    for title in (
        "Batch",
        "Seconds: median [min–max]",
        "Relative RMS" if resampling else "Truth RMS (vox)",
        "RAM GiB",
        "VRAM GiB",
    ):
        table.add_column(title, justify="right")
    table.add_column("Status")
    for row in records:
        if row["suite"] != suite:
            continue
        times = row.get("timing", {}).get("wall_seconds", [])
        memory = row.get("memory", {})
        error = row.get("accuracy", {}).get(
            "worst_frame_relative_rmse" if resampling else "worst_frame_rms_voxels"
        )
        batch = row.get("timing", {}).get("backend_settings", {}).get("batch_size")
        table.add_row(
            row["id"],
            str(batch) if batch is not None else "—",
            f"{np.median(times):.3f} [{min(times):.3f}–{max(times):.3f}]"
            if times
            else "—",
            (f"{error:.2e}" if resampling else f"{error:.6f}")
            if error is not None
            else "—",
            f"{memory['peak_ram_bytes'] / 2**30:.3f}"
            if "peak_ram_bytes" in memory
            else "—",
            f"{memory['peak_vram_bytes'] / 2**30:.3f}"
            if "peak_vram_bytes" in memory
            else "—",
            row["status"],
            style="green"
            if row["status"] == "passed"
            else "yellow"
            if row["status"] == "unavailable"
            else "red",
        )
    if table.row_count:
        console.print(table)


def _main(script, configs=_configs, worker=_worker, *, resampling=False, prepare=None):
    script = Path(script)
    operation = "resampling" if resampling else "registration"
    parser = argparse.ArgumentParser(
        description=f"Serial {operation} benchmark; see BENCHMARK.md."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Independent calls per config, including the first call",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Shared effective voxreg/FireANTs batch; defaults to voxreg's automatic policy",
    )
    parser.add_argument("--threads", type=int, default=8)
    if resampling:
        parser.add_argument(
            "--antspy-python",
            type=Path,
            help="Isolated Python with antspyx installed; required for its resident cases",
        )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["resident-*"] if resampling else ["*"],
        help="Configuration ID globs; resampling defaults to resident APIs (native CLI is opt-in)",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("timing", "memory", "accuracy"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.worker:
        if not args.mode:
            parser.error("Worker mode required")
        return worker(args.worker, args.mode)
    if (
        args.runs < 1
        or args.threads < 1
        or (args.batch_size is not None and args.batch_size < 1)
    ):
        parser.error("runs, threads and batch-size must be positive")
    cases = [
        c
        for c in configs()
        if any(fnmatch.fnmatchcase(c["id"], pattern) for pattern in args.configs)
    ]
    if not cases:
        parser.error("No configurations match --configs")
    if args.list:
        print("\n".join(c["id"] for c in cases))
        return 0
    if args.output is None:
        parser.error("--output is required")
    if sys.platform != "linux":
        parser.error("This benchmark requires Linux process-group RSS accounting")
    output = args.output.resolve()
    native = any(c["suite"] == "native" for c in cases)
    if native and not re.fullmatch(r"[\w./+\-]+", str(output)):
        parser.error(
            "Native ANTs output paths cannot contain whitespace or shell/bracket metacharacters"
        )
    environment = dict(os.environ)
    binary_directory = (
        Path(environment["ANTSPATH"])
        if native and environment.get("ANTSPATH")
        else None
    )
    if binary_directory is not None:
        if not binary_directory.is_dir():
            parser.error("ANTSPATH must name an existing directory")
        environment["PATH"] = (
            str(binary_directory.resolve()) + os.pathsep + environment.get("PATH", "")
        )
        environment["ANTSPATH"] = str(binary_directory.resolve()) + os.sep
    environment.update(
        ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(args.threads),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        ANTS_RANDOM_SEED="1",
    )
    required = (
        (
            ("antsApplyTransforms", "antsRegistration")
            if resampling
            else (
                "antsMotionCorr",
                "antsRegistration",
                "antsRegistrationSyN.sh",
                "PrintHeader",
            )
        )
        if native
        else ()
    )
    tools = {
        name: shutil.which(name, path=environment.get("PATH")) for name in required
    }
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        parser.error(
            f"Missing native ANTs tools: {', '.join(missing)}; set PATH or ANTSPATH"
        )
    antspy_python = None
    antspy_versions = None
    if resampling and any(c.get("backend") == "antspyx" for c in cases):
        if args.antspy_python is None:
            parser.error(
                "ANTsPy cases require --antspy-python; see BENCHMARK.md for the isolated environment"
            )
        antspy_python = args.antspy_python.expanduser().absolute()
        try:
            probe = subprocess.run(
                [
                    str(antspy_python),
                    str(script.parent / "_benchmark_antspy.py"),
                    "--versions",
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            antspy_versions = json.loads(probe.stdout)
            if antspy_versions.get("antspyx") != "0.6.3":
                parser.error("Qualified resident ANTsPy version is antspyx==0.6.3")
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            parser.error(f"ANTsPy interpreter preflight failed: {exc}")
    ants_version = None
    if native:
        version = subprocess.run(
            [tools["antsRegistration"], "--version"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        ants_version = version.stdout + version.stderr
        if version.returncode or re.search(r"\b2\.6\.5\b", ants_version) is None:
            parser.error(
                "Native benchmarks require qualified ANTs 2.6.5; set PATH or ANTSPATH"
            )
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output already exists: {output}; choose a fresh directory")
    source = output / "source"
    shutil.copytree(
        ROOT / "src/voxreg",
        source / "src/voxreg",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (source / "examples").mkdir()
    scripts = [script, *sorted(script.parent.glob("_benchmark_*.py"))]
    for source_script in scripts:
        shutil.copy2(source_script, source / "examples" / source_script.name)
    for name in ("pyproject.toml", "uv.lock", "BENCHMARK.md"):
        shutil.copy2(ROOT / name, source / name)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "protocol": "voxreg-resample-benchmark-v2"
        if resampling
        else "voxreg-benchmark-v2",
        "kind": "benchmark" if args.runs == 5 else "diagnostic",
        "status": "running",
        "runs": args.runs,
        "frames": FRAMES,
        "threads": args.threads,
        "resident_iterations": None if resampling else ITERATIONS,
        "resident_batch_size": args.batch_size,
        "command": sys.argv,
        "git_commit": revision.stdout.strip(),
        "git_status": dirty.stdout,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "ram_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "versions": _versions(),
        "power_start": data._power(),
        "source_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*"))
            if p.is_file()
        },
        "memory_definition": (
            "separate worker: sampled peak RSS including imports, loaded resident inputs and call allocations; dataset/map generation staged outside workers; baseline RSS and sampled excess also recorded (20 ms; short peaks may be missed); CUDA peak Torch allocated, not driver/context or reserved memory"
            if resampling
            else "separate worker: sampled peak process-group RSS including resident inputs/imports/preparation (20 ms; shared pages may count twice); CUDA peak Torch allocated, excluding driver/context and non-Torch allocations"
        ),
        "accuracy_definition": (
            "worst-frame intensity RMSE / reference intensity RMS on common center support; independent float64 SciPy linear interpolation; separate full-workload call, same batching"
            if resampling
            else "worst-frame full-grid physical pull RMS / minimum active spacing; separate full-workload call, same batching"
        ),
        "policy": {
            "objective": "minimize median runtime among accuracy-passing configurations",
            **(
                {"relative_rmse_limit": RESAMPLE_LIMIT}
                if resampling
                else {
                    "truth_rms_limit_voxels": TRUTH_LIMIT,
                    "truth_rms_comparison": "<=",
                    "syn_reference": SYN_REFERENCE,
                }
            ),
            "memory": "report peak RAM/VRAM; allocation or execution failure fails the case",
            "cross_device_and_old_new_agreement": "diagnostic only; not acceptance gates",
        },
        "cases": [],
    }
    report["batch_policy"] = (
        "One effective voxreg/FireANTs batch per workload/model/device, resolved by the first timing worker using voxreg's automatic policy (or --batch-size) and fixed for all remaining phases/backends. FireANTs-only selections also use voxreg's policy. ANTsPy processes one frame at a time."
    )
    if resampling:
        report["input_preparation"] = (
            "Images, fixed lattice, maps and time labels staged once per workload/model outside measured workers; each worker verifies the same serialized array/geometry hash. Conversion/application/mask/CPU return are timed; file loading is not."
        )
        report["antspy_environment"] = {
            "python": str(antspy_python) if antspy_python else None,
            "versions": antspy_versions,
        }
    if ants_version is not None:
        report["ants_version"] = ants_version
        report["tool_sha256"] = {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in tools.items()
        }
    console = Console(record=True)
    console.rule(f"{operation.capitalize()} benchmark")
    console.print(
        (
            f"Optimize runtime subject to relative interpolation RMS < {RESAMPLE_LIMIT} and fitting in memory."
            if resampling
            else f"Optimize runtime subject to truth RMS <= {TRUTH_LIMIT} vox and fitting in memory."
        ),
        style="bold",
    )
    console.print(
        f"{len(cases)} configurations · {args.runs} timed calls + memory + accuracy each"
    )
    console.print(f"Output: {output}", markup=False)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    shared_batches = {}
    try:
        for number, case in enumerate(cases, 1):
            directory = output / case["id"]
            directory.mkdir()
            config = {
                **case,
                "runs": args.runs,
                "frames": FRAMES,
                "iterations": None if resampling else ITERATIONS,
                "threads": args.threads,
                "batch_size": args.batch_size if case["suite"] == "resident" else None,
                "tools": tools,
            }
            shared_key = None
            if case.get("backend") in (
                "cpu",
                "cuda",
                "fireants-cpu",
                "fireants-cuda",
            ):
                shared_key = (
                    case["workload"],
                    case["model"],
                    case["backend"].removeprefix("fireants-"),
                )
                config["batch_size"] = args.batch_size or shared_batches.get(shared_key)
            if case.get("backend") == "antspyx":
                config.update(
                    worker_python=str(antspy_python),
                    worker_script="_benchmark_antspy.py",
                    worker_versions=antspy_versions,
                )
            config_path = directory / "config.json"
            _write(config_path, config)
            row = {**case, "status": "running"}
            report["cases"].append(row)
            _write(output / "results.json", report)
            if prepare is not None:
                prepare(config_path)
                config = json.loads(config_path.read_text())
            for mode in PHASES:
                console.log(
                    f"[{number}/{len(cases)}] {case['id']} / {mode}; {directory / (mode + '.log')}",
                    markup=False,
                )
                with console.status(f"{case['id']} / {mode}"):
                    row[mode] = _phase(
                        source / "examples" / script.name,
                        config_path,
                        mode,
                        environment,
                    )
                _write(output / "results.json", report)
                if row[mode]["status"] != "error" and row[mode].get(
                    "versions"
                ) != config.get("worker_versions", report["versions"]):
                    row[mode].update(
                        status="error", error="Dependencies changed during the run"
                    )
                if row[mode]["status"] in ("error", "unavailable"):
                    row["status"] = row[mode]["status"]
                    break
                if mode == "timing" and shared_key is not None:
                    batch = row[mode]["backend_settings"]["batch_size"]
                    if batch != shared_batches.get(shared_key, batch):
                        row.update(
                            status="error", error="voxreg/FireANTs batch sizes differ"
                        )
                        break
                    shared_batches[shared_key] = batch
                    config["batch_size"] = batch
                    _write(config_path, config)
            else:
                digests = {row[mode]["input_sha256"] for mode in PHASES}
                if len(digests) != 1:
                    row.update(status="error", error="Phase inputs differ")
                elif len(row["timing"]["wall_seconds"]) != args.runs:
                    row.update(status="error", error="Incomplete timing repetitions")
                elif (
                    case["suite"] == "resident"
                    and len(
                        {row[mode]["backend_settings"]["batch_size"] for mode in PHASES}
                    )
                    != 1
                ):
                    row.update(
                        status="error",
                        error="Batch sizes differ between timing, memory and accuracy",
                    )
                else:
                    row["status"] = row["accuracy"]["status"]
            _write(output / "results.json", report)
            _table(console, [row], case["suite"], args.runs, resampling)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception:  # noqa: BLE001 -- CLI boundary: preserve partial results and the traceback.
        report.update(status="error", error=traceback.format_exc())
        console.print_exception()
    else:
        report["status"] = (
            "failed"
            if any(c["status"] in ("error", "quality_failed") for c in report["cases"])
            else "completed"
        )
    finally:
        report["power_end"] = data._power()
        _write(output / "results.json", report)
        for suite in ("resident", "native"):
            _table(console, report["cases"], suite, args.runs, resampling)
        console.print(f"{report['status']}: {output / 'results.json'}", markup=False)
        console.save_text(str(output / "benchmark.log"), clear=False)
        try:
            from _benchmark_report import _render

            _render([output / "results.json"], output / "report.html")
        except Exception:  # noqa: BLE001 -- Preserve benchmark results if graph rendering fails.
            console.print_exception()
            console.save_html(str(output / "report.html"), clear=False)
    return (
        130
        if report["status"] == "interrupted"
        else int(report["status"] != "completed")
    )
