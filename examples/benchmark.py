"""One serial benchmark: wall time, peak memory and full-workload truth RMS.

See BENCHMARK.md. Use --plan-only to inspect the configuration matrix safely.
"""

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
from _benchmark_metrics import _native_motion_rms, _native_pair_rms
from _benchmark_resident import _check, _run
from rich.console import Console
from rich.table import Table

_ACTIVE = None
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
            if config["backend"].startswith("fireants-"):
                import importlib

                for module in ("rigid", "affine", "syn"):
                    importlib.import_module(f"fireants.registration.{module}")
        else:
            record["inputs"] = ants._prepare(config, directory, config["frames"])
            record["input_sha256"] = record["inputs"]["moving_array_sha256"]
            amplitudes = 0.8 + 0.2 * np.sin(
                2 * np.pi * np.arange(config["frames"]) / 16
            )
            amplitudes[0] = 0
        if cuda:
            torch.cuda.synchronize()
            record["cuda_device"] = {
                "name": torch.cuda.get_device_name(),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
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
                ants._output_qc(config, directory, run, config["frames"])
                errors = (
                    (
                        _native_motion_rms(
                            run, data.SHAPES[config["workload"]], amplitudes
                        )
                        if config["workflow"] == "motion"
                        else _native_pair_rms(
                            directory, run, data.SHAPES[config["workload"]]
                        )
                    )
                    if accuracy
                    else None
                )
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
                record.update(
                    rms_per_frame_voxels=errors,
                    worst_frame_rms_voxels=worst,
                    accuracy_pass=worst < 0.25,
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
    with (directory / f"{mode}.log").open("w") as log:
        _ACTIVE = subprocess.Popen(
            [sys.executable, str(script), "--worker", str(config_path), "--mode", mode],
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
    _write(path, record)
    return record


def _table(console, records, suite):
    table = Table(
        title="Resident-array registrations"
        if suite == "resident"
        else "Native ANTs workflows — CLI / I/O included"
    )
    for title in (
        "Configuration",
        "Seconds: median [min–max]",
        "RAM GiB (RSS)",
        "Torch VRAM GiB",
        "Worst RMS (voxels)",
        "Status",
    ):
        table.add_column(title)
    for row in records:
        if row["suite"] != suite:
            continue
        times = row.get("timing", {}).get("wall_seconds", [])
        memory = row.get("memory", {})
        error = row.get("accuracy", {}).get("worst_frame_rms_voxels")
        table.add_row(
            row["id"],
            f"{np.median(times):.3f} [{min(times):.3f}–{max(times):.3f}]"
            if times
            else "—",
            f"{memory['peak_ram_bytes'] / 2**30:.3f}"
            if "peak_ram_bytes" in memory
            else "—",
            f"{memory['peak_vram_bytes'] / 2**30:.3f}"
            if "peak_vram_bytes" in memory
            else "—",
            f"{error:.6f}" if error is not None else "—",
            row["status"],
            style="green"
            if row["status"] == "passed"
            else "yellow"
            if row["status"] == "unavailable"
            else "red",
        )
    if table.row_count:
        console.print(table)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Independent registrations per config, including the first call",
    )
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Resident-model update budget; native presets are unchanged",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["*"],
        help="Configuration ID glob patterns; quote wildcards",
    )
    parser.add_argument("--ants-bin", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("timing", "memory", "accuracy"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.worker:
        if not args.mode:
            parser.error("Worker mode required")
        return _worker(args.worker, args.mode)
    if min(args.runs, args.iterations, args.threads) < 1 or args.frames < 2:
        parser.error("runs, iterations and threads must be positive; frames >=2")
    cases = [
        c
        for c in _configs()
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
    if args.ants_bin is not None and not args.ants_bin.is_dir():
        parser.error("--ants-bin must name an existing directory")
    environment = dict(os.environ)
    binary_directory = args.ants_bin or (
        Path(environment["ANTSPATH"]) if environment.get("ANTSPATH") else None
    )
    if binary_directory is not None:
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
        ("antsMotionCorr", "antsRegistration", "antsRegistrationSyN.sh", "PrintHeader")
        if native
        else ()
    )
    tools = {
        name: shutil.which(name, path=environment.get("PATH")) for name in required
    }
    missing = [name for name, path in tools.items() if path is None]
    if missing and not args.plan_only:
        parser.error(
            f"Missing native ANTs tools: {', '.join(missing)}; supply --ants-bin"
        )
    ants_version = None
    if native and not args.plan_only:
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
                "Native truth reconstruction requires qualified ANTs 2.6.5; supply --ants-bin"
            )
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    shutil.copytree(
        ROOT / "src/voxreg",
        source / "src/voxreg",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (source / "examples").mkdir()
    scripts = [Path(__file__), *sorted(Path(__file__).parent.glob("_benchmark_*.py"))]
    for script in scripts:
        shutil.copy2(script, source / "examples" / script.name)
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
        "protocol": "voxreg-benchmark-v1",
        "status": "planned" if args.plan_only else "running",
        "runs": args.runs,
        "frames": args.frames,
        "threads": args.threads,
        "resident_iterations": args.iterations,
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
        "memory_definition": "separate worker: sampled peak process-group RSS including resident inputs/imports/preparation (20 ms; shared pages may count twice); CUDA peak Torch allocated, excluding driver/context and non-Torch allocations",
        "accuracy_definition": "worst-frame full-grid physical pull RMS / minimum active spacing; separate full-workload call, same batching, <0.25",
        "cases": [],
    }
    if ants_version is not None:
        report["ants_version"] = ants_version
        report["tool_sha256"] = {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in tools.items()
        }
    console = Console()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        for case in cases:
            directory = output / case["id"]
            directory.mkdir()
            config = {
                **case,
                "runs": args.runs,
                "frames": args.frames,
                "iterations": args.iterations,
                "threads": args.threads,
                "tools": tools,
            }
            config_path = directory / "config.json"
            _write(config_path, config)
            row = {**case, "status": "planned" if args.plan_only else "running"}
            report["cases"].append(row)
            _write(output / "results.json", report)
            if args.plan_only:
                continue
            for mode in ("timing", "memory", "accuracy"):
                console.log(f"{case['id']} / {mode}; see {directory / (mode + '.log')}")
                with console.status(f"{case['id']} / {mode}"):
                    row[mode] = _phase(
                        source / "examples/benchmark.py", config_path, mode, environment
                    )
                _write(output / "results.json", report)
                if (
                    row[mode]["status"] != "error"
                    and row[mode].get("versions") != report["versions"]
                ):
                    row[mode].update(
                        status="error", error="Dependencies changed during the run"
                    )
                if row[mode]["status"] in ("error", "unavailable"):
                    row["status"] = row[mode]["status"]
                    break
            else:
                digests = {
                    row[mode]["input_sha256"]
                    for mode in ("timing", "memory", "accuracy")
                }
                if len(digests) != 1:
                    row.update(status="error", error="Phase inputs differ")
                elif len(row["timing"]["wall_seconds"]) != args.runs:
                    row.update(status="error", error="Incomplete timing repetitions")
                elif (
                    case["suite"] == "resident"
                    and len(
                        {
                            row[mode]["backend_settings"]["batch_size"]
                            for mode in ("timing", "memory", "accuracy")
                        }
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
            _table(console, [row], case["suite"])
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        _write(output / "results.json", report)
        return 130
    report["status"] = (
        "planned"
        if args.plan_only
        else "failed"
        if any(c["status"] in ("error", "quality_failed") for c in report["cases"])
        else "completed"
    )
    report["power_end"] = data._power()
    _write(output / "results.json", report)
    if not args.plan_only:
        for suite in ("resident", "native"):
            _table(console, report["cases"], suite)
    console.print(
        f"{len(cases)} configurations; results: {output / 'results.json'}", markup=False
    )
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(_main())
