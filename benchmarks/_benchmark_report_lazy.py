"""Offline lazy scaling graphs, separate from resident backend comparisons."""

import base64
import io
import json
import math
import statistics
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any


def _measurement(value: Any) -> float | int | None:
    """Validate a recorded scalar without converting missing data to zero.

    Parameters
    ----------
    value : object
        Optional numeric measurement.

    Returns
    -------
    float or int or None
        Original measurement.

    Raises
    ------
    ValueError
        If the measurement is negative, nonfinite or not numeric.
    """
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid lazy measurement: {value!r}")
    return value


def _times(case: dict[str, Any]) -> list[float]:
    """Read strictly positive timing samples.

    Parameters
    ----------
    case : dict
        One recorded configuration.

    Returns
    -------
    list of float
        Recorded wall times, or an empty list when missing.

    Raises
    ------
    ValueError
        If a timing sample is missing, invalid or nonpositive.
    """
    values = [_measurement(v) for v in case.get("timing", {}).get("wall_seconds", [])]
    if any(v is None or v == 0 for v in values):
        raise ValueError(f"Nonpositive lazy timing in {case['id']}")
    return values


def _qualified(case: dict[str, Any]) -> bool:
    """Require completed phases, requested samples and explicit accuracy success.

    Parameters
    ----------
    case : dict
        Recorded configuration and phase results.

    Returns
    -------
    bool
        Whether the case can contribute to a scaling line.
    """
    return (
        case.get("status") == "passed"
        and len(_times(case)) == case["config"]["runs"]
        and all(
            case.get(p, {}).get("status") == "passed"
            for p in ("timing", "memory", "accuracy")
        )
        and case["accuracy"].get("accuracy_pass") is True
    )


def _policy(case: dict[str, Any]) -> tuple[Any, ...]:
    """Identify an execution policy without pooling different settings.

    Parameters
    ----------
    case : dict
        Recorded configuration.

    Returns
    -------
    tuple
        Device, chunk, batch, worker, memory, thread and update settings.
    """
    config = case["config"]
    return tuple(
        config.get(k)
        for k in (
            "device",
            "chunk_frames",
            "effective_batch_size",
            "workers",
            "worker_memory_bytes",
            "threads",
            "iterations",
        )
    )


def _memory(case: dict[str, Any], name: str) -> float | None:
    """Read process-tree memory in GiB, retaining missing values.

    Parameters
    ----------
    case : dict
        Recorded configuration.
    name : str
        Memory-phase byte measurement key.

    Returns
    -------
    float or None
        Validated GiB measurement, or no recorded value.

    Raises
    ------
    ValueError
        If the recorded measurement is invalid.
    """
    value = _measurement(case.get("memory", {}).get(name))
    return value / 2**30 if value is not None else None


def _chart(cases: list[dict[str, Any]], metric: str) -> str:
    """Plot recorded samples without pooling different execution policies.

    Parameters
    ----------
    cases : list of dict
        Configurations from a single experiment.
    metric : str
        Time, RAM, GPU memory, intensity accuracy or physical truth selection.

    Returns
    -------
    str
        Embedded SVG figure, or an empty string when no measurements exist.

    Raises
    ------
    ValueError
        If a recorded measurement is invalid.
    """
    from matplotlib import rc_context
    from matplotlib.figure import Figure

    titles = {
        "time": "Whole-call runtime — median, range and first call",
        "ram": "Process-tree RAM — baseline, peak and increment",
        "gpu": "Largest per-worker GPU peak — not a device total",
        "accuracy": "Relative intensity RMSE — independent SciPy reference",
        "truth": "Physical truth RMS — minimum-spacing voxels",
    }
    groups = defaultdict(list)
    for case in cases:
        size = _measurement(case["config"].get("input_image_bytes"))
        if size is not None and size > 0:
            groups[_policy(case)].append((size / 2**30, case))
    figure = Figure(figsize=(8, 4.4), layout="constrained")
    axis = figure.subplots()
    plotted = False
    magnitudes = []
    for index, (policy, rows) in enumerate(groups.items()):
        rows.sort(key=lambda row: row[0])
        device, chunk, batch, workers, _, threads, iterations = policy
        label = f"{device} · chunk {chunk} · batch {batch} · workers {workers}\nthreads {threads} · configured updates {iterations}"
        color = f"C{index % 10}"
        series = defaultdict(list)
        for size, case in rows:
            qualified = _qualified(case)
            if metric == "time":
                times = _times(case)
                values = {"median": statistics.median(times)} if times else {}
                if times:
                    median = values["median"]
                    axis.errorbar(
                        size,
                        median,
                        yerr=[[median - min(times)], [max(times) - median]],
                        color=color,
                        capsize=4,
                    )
                    axis.plot(size, times[0], marker="_", color="black", markersize=10)
            elif metric == "ram":
                values = {
                    name: _memory(case, key)
                    for name, key in (
                        ("peak", "peak_process_tree_rss_bytes"),
                        ("baseline", "baseline_process_tree_rss_bytes"),
                        ("increment", "peak_rss_increment_bytes"),
                    )
                }
            elif metric == "gpu":
                peaks = case.get("memory", {}).get("worker_gpu_peaks", {}).values()
                values = {}
                for name in ("allocated", "reserved"):
                    measurements = [_measurement(p.get(name + "_bytes")) for p in peaks]
                    measurements = [v for v in measurements if v is not None]
                    values[name] = max(measurements) / 2**30 if measurements else None
            else:
                accuracy = case.get("accuracy", {})
                key = (
                    "worst_truth_rms_voxels"
                    if metric == "truth"
                    else "worst_relative_intensity_rmse"
                )
                values = {"error": _measurement(accuracy.get(key))}
                limit = _measurement(
                    accuracy.get(
                        "truth_limit_voxels" if metric == "truth" else "intensity_limit"
                    )
                )
                if limit is not None:
                    magnitudes.append(limit)
                    axis.plot(size, limit, marker="_", color="#bd3030", markersize=15)
            for name, value in values.items():
                if value is None:
                    continue
                plotted = True
                magnitudes.append(value)
                # Only complete accuracy-passing points can form scaling lines.
                if qualified:
                    series[name].append((size, value))
                axis.plot(
                    size,
                    value,
                    marker="o" if qualified else "X",
                    color=color,
                    markersize=6,
                )
        for offset, (name, points) in enumerate(series.items()):
            axis.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                color=color,
                linestyle=("-", "--", ":")[offset % 3],
                label=f"{label} · {name}",
            )
    if not plotted:
        return ""
    axis.set_xlabel("Input image size (GiB)")
    axis.set_ylabel(
        "Seconds"
        if metric == "time"
        else "GiB"
        if metric in ("ram", "gpu")
        else "Error (red ticks: acceptance limit)"
    )
    positive = [v for v in magnitudes if v > 0]
    if (
        metric in ("accuracy", "truth")
        and positive
        and max(positive) / min(positive) > 100
    ):
        axis.set_yscale("symlog", linthresh=min(positive), linscale=0.5)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.2)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=7, loc="best")
    with io.BytesIO() as buffer, rc_context({"svg.fonttype": "none"}):
        figure.savefig(buffer, format="svg")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    title = escape(titles[metric])
    return f'<figure><figcaption>{title}</figcaption><img alt="{title}" src="data:image/svg+xml;base64,{encoded}"></figure>'


def render_lazy_section(path: Path, report: dict[str, Any], index: int) -> str:
    """Render one lazy experiment without comparing it with resident timings.

    Parameters
    ----------
    path : pathlib.Path
        Source result file.
    report : dict
        Parsed lazy benchmark result.
    index : int
        Unique navigation anchor number in the combined report.

    Returns
    -------
    str
        Graph-first HTML with collapsed measurements.

    Raises
    ------
    ValueError
        If the operation or measurements are invalid.
    """
    cases = report["cases"]
    operation = report["operation"]
    if operation not in ("register", "resample"):
        raise ValueError(f"Invalid lazy operation in {path}: {operation!r}")
    parts = [
        f'<section id="run-{index}"><h2>Lazy {escape(operation)} — {escape(path.parent.name)}</h2>',
        f"<p><strong>{escape(report['status'])}</strong> · {escape(report['protocol'])} · voxreg only; no backend ranking.</p>",
        f"<p>{escape(report.get('timing_boundary', 'Timing boundary not recorded.'))}</p>",
        "<p>Lines connect only complete accuracy-passing cases with identical execution policies. Crosses are failed/incomplete cases. RAM is process-tree RSS, not the worker limit. GPU lines show the largest worker peak, never summed device totals. Two sizes do not prove a universal memory bound.</p>",
    ]
    if report.get("diagnostic"):
        parts.append(
            '<p class="warning">Diagnostic run, not a formal size-sweep benchmark.</p>'
        )
    if report.get("status") not in ("passed", "failed"):
        parts.append(
            '<p class="warning">Partial run: missing phases are not zero measurements.</p>'
        )
    evictions = [
        e
        for c in cases
        for p in ("timing", "memory", "accuracy")
        for e in c.get(p, {}).get("cache_evictions", [])
    ]
    verified = sum(
        e.get("verified") is True
        and e.get("resident_pages_after") == 0
        and bool(e.get("worker_cached_tasks"))
        and all(n == 0 for n in e["worker_cached_tasks"].values())
        for e in evictions
    )
    expected = sum(c["config"]["runs"] + 2 for c in cases)
    parts.append(
        f"<p>Recorded cold-input checks: <strong>{verified}/{len(evictions)}</strong> verified zero-page/zero-cached-task checks; {expected} expected passes across these configurations. Missing checks are not cold-input qualification. {escape(report.get('cache_policy', 'No cold-input policy recorded.'))}</p>"
    )
    parts.append('<div class="plots">')
    for metric in ("time", "ram", "gpu", "accuracy", "truth"):
        if metric != "truth" or operation == "register":
            parts.append(_chart(cases, metric))
    parts.append(
        '</div><details><summary>Detailed lazy measurements and policies</summary><div class="table"><table><tr><th>Configuration</th><th>Outcome</th><th>Input GiB</th><th>Median seconds</th><th>Peak RAM GiB</th><th>Intensity RMSE</th></tr>'
    )
    for case in cases:
        times = _times(case)
        values = [
            case["id"],
            case["status"],
            _measurement(case["config"].get("input_image_bytes")) / 2**30
            if case["config"].get("input_image_bytes") is not None
            else None,
            statistics.median(times) if times else None,
            _memory(case, "peak_process_tree_rss_bytes"),
            _measurement(case.get("accuracy", {}).get("worst_relative_intensity_rmse")),
        ]
        parts.append(
            "<tr>"
            + "".join(
                f"<td>{escape(f'{v:.4g}' if isinstance(v, (int, float)) else str(v) if v is not None else '—')}</td>"
                for v in values
            )
            + "</tr>"
        )
    parts.append(
        "</table></div><pre>"
        + escape(json.dumps([c["config"] for c in cases], indent=2))
        + "</pre></details></section>"
    )
    return "\n".join(parts)
