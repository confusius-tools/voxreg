"""Offline graphs from recorded runs; never pool hosts, revisions or operations."""

import argparse
import base64
import hashlib
import io
import json
import math
import statistics
from collections import Counter
from html import escape
from itertools import groupby
from pathlib import Path

_PROTOCOLS = {
    "voxreg-benchmark-v2": "Registration",
    "voxreg-resample-benchmark-v2": "Resampling",
}
_COLORS = {
    "voxreg": "#16867a",
    "FireANTs": "#476eb1",
    "ANTsPy": "#bd761c",
    "Native ANTs": "#9259a0",
}


def _engine(case):
    if case["suite"] == "native":
        return "Native ANTs"
    backend = case["backend"]
    return (
        "FireANTs"
        if backend.startswith("fireants-")
        else "ANTsPy"
        if backend == "antspyx"
        else "voxreg"
    )


def _number(value):
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid measurement: {value!r}")
    return value


def _times(case):
    values = [_number(v) for v in case.get("timing", {}).get("wall_seconds", [])]
    if any(v is None or v == 0 for v in values):
        raise ValueError(f"Nonpositive timing in {case['id']}")
    return values


def _batch(case):
    values = {
        case[p]["backend_settings"]["batch_size"]
        for p in ("timing", "memory", "accuracy")
        if "backend_settings" in case.get(p, {})
    }
    return next(iter(values)) if len(values) == 1 else "mismatch" if values else None


def _value(case, metric, resampling):
    if metric == "time":
        times = _times(case)
        return statistics.median(times) if times else None
    if metric == "accuracy":
        return _number(
            case.get("accuracy", {}).get(
                "worst_frame_relative_rmse" if resampling else "worst_frame_rms_voxels"
            )
        )
    value = _number(case.get("memory", {}).get(f"peak_{metric}_bytes"))
    return value / 2**30 if value is not None else None


def _qualified(case, runs):
    return (
        case["status"] == "passed"
        and len(_times(case)) == runs
        and all(
            case.get(p, {}).get("status") == "passed"
            for p in ("timing", "memory", "accuracy")
        )
        and case["accuracy"].get("accuracy_pass") is True
    )


def _comparison(case, cases, runs):
    if case.get("backend") not in ("cpu", "cuda"):
        return "—"
    peer = next(
        (
            c
            for c in cases
            if c.get("backend") == "fireants-" + case["backend"]
            and c.get("workload") == case.get("workload")
            and c.get("model") == case.get("model")
        ),
        None,
    )
    if peer is None or not all(_qualified(c, runs) for c in (case, peer)):
        return "not accuracy-qualified" if peer else "—"
    hashes = {
        c[p].get("input_sha256")
        for c in (case, peer)
        for p in ("timing", "memory", "accuracy")
    }
    if len(hashes) != 1 or None in hashes:
        return "input hashes differ / missing"
    ratio = statistics.median(_times(peer)) / statistics.median(_times(case))
    a, b = _batch(case), _batch(peer)
    note = (
        "matched batch" if isinstance(a, int) and a == b else f"UNMATCHED batch {a}/{b}"
    )
    return f"{ratio:.2f}× ({note})"


def _row_group(case, resampling):
    if case["suite"] == "native":
        return f"Native {case['workflow'].replace('_', ' ').title()} · CPU"
    model = case["model"]
    name = (
        "Displacement"
        if resampling and model == "syn"
        else {"syn": "SyN"}.get(model, model.title())
    )
    device = "CUDA" if case["backend"].endswith("cuda") else "CPU"
    return f"{name} · {device}"


def _chart(cases, metric, resampling, runs, limit=None):
    # Matplotlib is already an examples dependency; no browser scripts or CDN.
    from matplotlib import rc_context
    from matplotlib.figure import Figure

    rows = [(c, _value(c, metric, resampling)) for c in cases]
    rows = [(c, v) for c, v in rows if v is not None]
    if not rows:
        return ""
    title = {
        "time": "Runtime (seconds; median and min–max)",
        "ram": "Peak process-group RAM (GiB)",
        "vram": "Peak Torch-allocated VRAM (GiB)",
        "accuracy": "Worst-frame relative intensity RMSE"
        if resampling
        else "Worst-frame truth RMS (minimum-spacing voxels)",
    }[metric]
    positions, headings = [], []
    cursor = 0
    for label, members in groupby(rows, key=lambda row: _row_group(row[0], resampling)):
        headings.append((cursor, label))
        cursor += 0.9
        for _ in members:
            positions.append(cursor)
            cursor += 1
        cursor += 0.5
    figure = Figure(figsize=(12, max(2.4, cursor * 0.28 + 1.2)), layout="constrained")
    axis = figure.subplots()
    for y, label in headings:
        axis.text(
            0,
            y,
            label,
            transform=axis.get_yaxis_transform(),
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        axis.axhline(y - 0.45, color="#dce3e8", linewidth=0.8)
    for y, (case, value) in zip(positions, rows, strict=True):
        color = _COLORS[_engine(case)]
        marker = "o" if _qualified(case, runs) else "X"
        if metric == "time":
            times = _times(case)
            axis.errorbar(
                value,
                y,
                xerr=[[value - min(times)], [max(times) - value]],
                fmt=marker,
                color=color,
                capsize=3,
                markersize=5,
            )
            axis.plot(times[0], y, marker="|", color="#343434", markersize=11)
        else:
            axis.plot(value, y, marker=marker, color=color, markersize=5)
    if all(v > 0 for _, v in rows):
        axis.set_xscale("log")
    if limit is not None:
        if isinstance(limit, dict):
            labelled = set()
            for y, (case, _) in zip(positions, rows, strict=True):
                threshold, comparison = limit[case["id"]]
                label = f"acceptance {comparison} {threshold:g}"
                axis.vlines(
                    threshold,
                    y - 0.4,
                    y + 0.4,
                    color="#be3030",
                    linestyle="--",
                    label=label if label not in labelled else None,
                )
                labelled.add(label)
        else:
            axis.axvline(
                limit,
                color="#be3030",
                linestyle="--",
                label=f"strict acceptance < {limit:g}",
            )
        axis.legend(loc="best", fontsize=9)
    axis.set_yticks(
        positions,
        [c["id"].removeprefix("resident-") for c, _ in rows],
        fontsize=9,
    )
    axis.set_ylim(cursor - 1, -0.6)
    axis.grid(axis="x", alpha=0.25, which="both")
    axis.set_xlabel(title + " · lower is better")
    with io.BytesIO() as buffer, rc_context({"svg.fonttype": "none"}):
        figure.savefig(buffer, format="svg")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<figure><figcaption>{escape(title)}</figcaption><img alt="{escape(title)}" src="data:image/svg+xml;base64,{encoded}"></figure>'


def _section(path, report, index):
    operation = _PROTOCOLS[report["protocol"]]
    resampling = operation == "Resampling"
    cases = sorted(
        report["cases"],
        key=lambda c: (
            c["suite"] == "native",
            c.get("backend", "").endswith("cuda"),
            {"rigid": 0, "affine": 1, "syn": 2}.get(c.get("model"), 3),
            c.get("workflow", ""),
            c["workload"],
            tuple(_COLORS).index(_engine(c)),
            c["id"],
        ),
    )
    counts = ", ".join(
        f"{n} {status}" for status, n in Counter(c["status"] for c in cases).items()
    )
    gpus = sorted(
        {
            c[p]["cuda_device"]["name"]
            for c in cases
            for p in ("timing", "memory", "accuracy")
            if "cuda_device" in c.get(p, {})
        }
    )
    host = report.get("host", {})
    parts = [
        f'<section id="run-{index}"><h2>{escape(operation)} — {escape(path.parent.name)}</h2>',
        f"<p><strong>{escape(str(report['status']))}</strong> · {escape(counts)}</p>",
        f"<p>{escape(host.get('cpu', 'CPU not recorded'))} · {escape(', '.join(gpus) or 'GPU not recorded')}<br>Revision <code>{escape(report.get('git_commit', 'unknown'))}</code>{' (dirty checkout)' if report.get('git_status') else ''}<br>{report['runs']} requested timed calls/case; {report.get('frames', '?')} recording frames; {report.get('threads', '?')} threads. First call included.</p>",
    ]
    if report.get("kind") != "benchmark" or report["runs"] != 5:
        parts.append(
            '<p class="warning">Diagnostic run, not a formal five-call benchmark.</p>'
        )
    if report["status"] not in ("completed", "failed"):
        parts.append(
            '<p class="warning">Partial run: timings can exist without completed memory or accuracy qualification.</p>'
        )
    if not report.get("batch_policy"):
        parts.append(
            '<p class="warning">Historical backend-specific batch policies: do not interpret unmatched batches as controlled comparisons.</p>'
        )
    parts.append(
        f"<p>{escape(report.get('memory_definition', 'Memory boundary not recorded.'))}</p>"
    )
    metadata = {k: v for k, v in report.items() if k != "cases"}
    parts.append(
        f"<details><summary>Protocol, versions, source hashes and provenance</summary><p>Input: <code>{escape(str(path))}</code><br>results.json SHA256: <code>{hashlib.sha256(path.read_bytes()).hexdigest()}</code></p><pre>{escape(json.dumps(metadata, indent=2))}</pre></details>"
    )
    limit = report["policy"][
        "relative_rmse_limit" if resampling else "truth_rms_limit_voxels"
    ]
    _number(limit)
    syn_limit = report["policy"].get("syn_truth_limit_voxels")
    if not resampling and report["policy"].get("truth_rms_comparison") == "<=":
        limit = {c["id"]: (limit, "≤") for c in cases}
    elif not resampling and syn_limit is not None:
        _number(syn_limit)
        limit = {
            c["id"]: (syn_limit, "≤")
            if c.get("model", c.get("workflow")) == "syn"
            else (limit, "<")
            for c in cases
        }
    for group in (
        "CPU resident APIs",
        "CUDA resident APIs",
        "Native CLI workflows (includes CLI / I/O)",
    ):
        subset = [
            c
            for c in cases
            if (
                c["suite"] == "native"
                if group.startswith("Native")
                else c["suite"] == "resident"
                and c["backend"].endswith("cuda") == group.startswith("CUDA")
            )
        ]
        if not subset:
            continue
        parts.append(f'<h3>{escape(group)}</h3><div class="plots">')
        for metric in ("time", "ram", "vram", "accuracy"):
            parts.append(
                _chart(
                    subset,
                    metric,
                    resampling,
                    report["runs"],
                    limit if metric == "accuracy" else None,
                )
            )
        parts.append("</div>")
    parts.append(
        '<h3>Recorded measurements</h3><p>Ratios are FireANTs time / voxreg time within this run only; greater than 1 means voxreg was faster. They require both complete accuracy-passing cases and identical phase input hashes. Unmatched historical batches are explicitly flagged, not claimed as controlled wins. Hover over runtime for every timed call.</p><div class="table"><table><thead><tr><th>Configuration</th><th>Outcome</th><th>Batch</th><th>Seconds: median [min–max]</th><th>First call (s)</th><th>RAM GiB</th><th>VRAM GiB</th><th>Worst-frame error</th><th>FireANTs / voxreg</th></tr></thead><tbody>'
    )
    previous_group = None
    for case in cases:
        group = _row_group(case, resampling)
        if group != previous_group:
            if previous_group is not None:
                parts.append("</tbody><tbody>")
            parts.append(
                f'<tr><th colspan="9" scope="rowgroup">{escape(group)}</th></tr>'
            )
            previous_group = group
        times = _times(case)
        timing = (
            f"{statistics.median(times):.4g} [{min(times):.4g}–{max(times):.4g}]"
            if times
            else "—"
        )
        measurements = [
            _value(case, metric, resampling) for metric in ("ram", "vram", "accuracy")
        ]
        cells = [
            case["id"],
            case["status"],
            str(_batch(case) or "—"),
            timing,
            f"{times[0]:.4g}" if times else "—",
            *(f"{v:.4g}" if v is not None else "—" for v in measurements),
            _comparison(case, cases, report["runs"]),
        ]
        parts.append(
            f'<tr class="{"pass" if _qualified(case, report["runs"]) else "fail"}" title="{escape(str(times))}">'
            + "".join(f"<td>{escape(v)}</td>" for v in cells)
            + "</tr>"
        )
    parts.append("</tbody></table></div></section>")
    return "\n".join(parts)


def _render(paths, output):
    reports = []
    for path in paths:
        path = Path(path)
        path = (path / "results.json" if path.is_dir() else path).resolve()
        report = json.loads(path.read_text())
        if report.get("protocol") not in _PROTOCOLS:
            raise ValueError(
                f"Unsupported protocol in {path}: {report.get('protocol')!r}; keep historical protocols separate"
            )
        if (
            not isinstance(report.get("cases"), list)
            or not isinstance(report.get("runs"), int)
            or report["runs"] < 1
        ):
            raise ValueError(f"Invalid cases/runs in {path}")
        reports.append((path, report))
    output = Path(output).resolve()
    if output.suffix != ".html" or any(output == path for path, _ in reports):
        raise ValueError("Output must be an HTML file, not an input result")
    legend = " · ".join(
        f'<span style="color:{color}">● {name}</span>'
        for name, color in _COLORS.items()
    )
    navigation = " · ".join(
        f'<a href="#run-{i}">{escape(_PROTOCOLS[r["protocol"]])}: {escape(p.parent.name)}</a>'
        for i, (p, r) in enumerate(reports, 1)
    )
    sections = [_section(p, r, i) for i, (p, r) in enumerate(reports, 1)]
    html = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>voxreg benchmark report</title><style>
body{font:15px system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#202b36;background:#fafbfc;line-height:1.5}h1,h2,h3{line-height:1.2}h2{border-top:3px solid #16867a;padding-top:24px;margin-top:48px}a{color:#146b75}code,pre{font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef1f4;padding:16px}figure{background:white;margin:14px 0;padding:12px;border:1px solid #ddd;border-radius:6px}figcaption{font-weight:600}img{width:100%;height:auto}table{border-collapse:collapse;font-size:12px;white-space:nowrap;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd}th{background:#e8edf0}.table{overflow-x:auto}.fail td:nth-child(2),.warning{color:#962626;font-weight:600}.pass td:nth-child(2){color:#126355}summary{cursor:pointer}section{scroll-margin:20px}
</style><body><h1>voxreg performance and accuracy</h1><p>Separate experiments, not a combined registration + resampling pipeline measurement. Hosts, revisions, input maps and measurement boundaries remain separate: no cross-host speedups or pooled ranking. Recorded accuracy is not recomputed by this report.</p><p>"""
        + legend
        + """</p><p>● Accuracy-passing case · × Failed / unqualified case · dark vertical tick: first timed call · whisker: min–max · all recorded calls contribute to the median. Graph axes are logarithmic when all values are positive. Missing measurements are omitted, never treated as zero. Native controls stay separate from resident APIs.</p><nav>"""
        + navigation
        + "</nav>"
        + "\n".join(sections)
        + "</body></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(html)
    temporary.replace(output)


def _main():
    parser = argparse.ArgumentParser(
        description="Build an offline graphical report; each results.json remains a separate experiment."
    )
    parser.add_argument(
        "results", nargs="+", type=Path, help="Run directories or results.json files"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Self-contained HTML destination"
    )
    args = parser.parse_args()
    try:
        _render(args.results, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(args.output)
    return 0
