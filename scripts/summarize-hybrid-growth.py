#!/usr/bin/env python3
"""Summarize service and hardware telemetry from a hybrid-growth run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from datetime import datetime
from pathlib import Path


def finite(values):
    return [value for value in values if math.isfinite(value)]


def mean(values):
    values = finite(values)
    return statistics.fmean(values) if values else None


def percentile(values, fraction):
    values = sorted(finite(values))
    if not values:
        return None
    index = fraction * (len(values) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - index) + values[hi] * (index - lo)


def value_summary(values):
    values = finite(values)
    return {
        "count": len(values),
        "min": min(values, default=None),
        "mean": mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values, default=None),
    }


def csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def local_timestamp(value: str) -> float:
    value = value.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            pass
    return datetime.fromisoformat(value).timestamp()


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def prometheus_values(text: str, name: str):
    pattern = re.compile(rf"^llamacpp:{re.escape(name)}\s+([-+0-9.eE]+)$", re.MULTILINE)
    return [float(value) for value in pattern.findall(text)]


def prometheus_series(text: str, name: str):
    """Return timestamped samples from the benchmark's Prometheus snapshots."""
    metric = f"llamacpp:{name}"
    timestamp = None
    samples = []
    for line in text.splitlines():
        if line.startswith("# timestamp "):
            try:
                timestamp = local_timestamp(line.removeprefix("# timestamp "))
            except ValueError:
                timestamp = None
            continue
        if timestamp is None or not line.startswith(metric):
            continue
        fields = line.split()
        if len(fields) == 2 and fields[0] == metric:
            samples.append({"time_unix": timestamp, "value": number(fields[1])})
    return samples


def counter_window_delta(series, started: float, finished: float):
    """Bracket an interval with monitor samples and report a monotonic delta."""
    if not series:
        return None
    before = next((sample for sample in reversed(series) if sample["time_unix"] <= started), None)
    after = next((sample for sample in series if sample["time_unix"] >= finished), None)
    before = before or series[0]
    after = after or series[-1]
    if after["time_unix"] < before["time_unix"]:
        return None
    return {
        "delta": max(0, int(after["value"] - before["value"])),
        "sample_started_unix": before["time_unix"],
        "sample_finished_unix": after["time_unix"],
        "leading_overhang_s": max(0.0, started - before["time_unix"]),
        "trailing_overhang_s": max(0.0, after["time_unix"] - finished),
    }


def fa_page_window_summary(checked_series, skipped_series, started: float, finished: float):
    checked = counter_window_delta(checked_series, started, finished)
    skipped = counter_window_delta(skipped_series, started, finished)
    if checked is None or skipped is None:
        return {}
    checked_delta = checked["delta"]
    skipped_delta = skipped["delta"]
    return {
        "checked": checked_delta,
        "skipped": skipped_delta,
        "skip_ratio": skipped_delta / checked_delta if checked_delta else None,
        "sample_started_unix": min(
            checked["sample_started_unix"], skipped["sample_started_unix"]
        ),
        "sample_finished_unix": max(
            checked["sample_finished_unix"], skipped["sample_finished_unix"]
        ),
        "leading_overhang_s": max(
            checked["leading_overhang_s"], skipped["leading_overhang_s"]
        ),
        "trailing_overhang_s": max(
            checked["trailing_overhang_s"], skipped["trailing_overhang_s"]
        ),
    }


def load_events(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pcie_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 5 or not fields[0].isdigit():
            continue
        try:
            timestamp = datetime.strptime(fields[0] + fields[1], "%Y%m%d%H:%M:%S").timestamp()
            rows.append({"time_unix": timestamp, "rx_mib_s": float(fields[3]), "tx_mib_s": float(fields[4])})
        except ValueError:
            continue
    return rows


def numastat_summary(path: Path):
    samples = []
    if not path.exists():
        return {"samples": 0}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 6 or fields[0] != "Total":
            continue
        try:
            nodes = [float(value) for value in fields[1:5]]
            total = float(fields[5])
        except ValueError:
            continue
        samples.append((nodes, total))
    if not samples:
        return {"samples": 0}
    nodes, total = samples[-1]
    node_mean = statistics.fmean(nodes)
    return {
        "samples": len(samples),
        "latest_node_mib": nodes,
        "latest_total_mib": total,
        "latest_max_min_skew_mib": max(nodes) - min(nodes),
        "latest_skew_pct_of_mean": (
            100 * (max(nodes) - min(nodes)) / node_mean if node_mean else None
        ),
        "peak_total_mib": max(item[1] for item in samples),
    }


def span_summary(gpu, pcie, started, finished):
    gpu_span = [row for row in gpu if started <= row["time_unix"] <= finished]
    pcie_span = [row for row in pcie if started <= row["time_unix"] <= finished]
    return {
        "gpu_samples": len(gpu_span),
        "gpu_util_avg_pct": mean([row["gpu_util_pct"] for row in gpu_span]),
        "gpu_util_p50_pct": percentile([row["gpu_util_pct"] for row in gpu_span], 0.50),
        "gpu_util_p95_pct": percentile([row["gpu_util_pct"] for row in gpu_span], 0.95),
        "gpu_memory_util_avg_pct": mean([row["memory_util_pct"] for row in gpu_span]),
        "gpu_util_zero_pct": (
            100 * sum(row["gpu_util_pct"] == 0 for row in gpu_span) / len(gpu_span)
            if gpu_span
            else None
        ),
        "gpu_memory_peak_mib": max((row["memory_used_mib"] for row in gpu_span), default=None),
        "gpu_power_avg_w": mean([row["power_w"] for row in gpu_span]),
        "pcie_samples": len(pcie_span),
        "pcie_rx_avg_mib_s": mean([row["rx_mib_s"] for row in pcie_span]),
        "pcie_tx_avg_mib_s": mean([row["tx_mib_s"] for row in pcie_span]),
        "pcie_rx_p50_mib_s": percentile([row["rx_mib_s"] for row in pcie_span], 0.50),
        "pcie_tx_p50_mib_s": percentile([row["tx_mib_s"] for row in pcie_span], 0.50),
        "pcie_rx_p95_mib_s": percentile([row["rx_mib_s"] for row in pcie_span], 0.95),
        "pcie_tx_p95_mib_s": percentile([row["tx_mib_s"] for row in pcie_span], 0.95),
        "pcie_rx_peak_mib_s": max((row["rx_mib_s"] for row in pcie_span), default=None),
        "pcie_tx_peak_mib_s": max((row["tx_mib_s"] for row in pcie_span), default=None),
    }


def process_span_summary(process, started, finished):
    span = [row for row in process if started <= row["time_unix"] <= finished]
    if not span:
        return {}
    elapsed = span[-1]["time_unix"] - span[0]["time_unix"]
    ticks = span[-1]["cpu_ticks"] - span[0]["cpu_ticks"]
    return {
        "peak_rss_gib": max(row["rss_kib"] for row in span) / 1024 / 1024,
        "read_gib": (span[-1]["read_bytes"] - span[0]["read_bytes"]) / 1024**3,
        "write_gib": (span[-1]["write_bytes"] - span[0]["write_bytes"]) / 1024**3,
        "average_cpu_cores": ticks / os.sysconf("SC_CLK_TCK") / elapsed if elapsed > 0 else None,
    }


def merge_intervals(intervals):
    merged = []
    for started, finished in sorted(intervals):
        if finished <= started:
            continue
        if not merged or started > merged[-1][1]:
            merged.append([started, finished])
        else:
            merged[-1][1] = max(merged[-1][1], finished)
    return merged


def interval_components(intervals):
    components = []
    for index, (started, finished) in sorted(enumerate(intervals), key=lambda item: item[1]):
        if finished <= started:
            continue
        if not components or started > components[-1][1]:
            components.append([started, finished, [index]])
        else:
            components[-1][1] = max(components[-1][1], finished)
            components[-1][2].append(index)
    return components


def interval_hardware_summary(gpu, pcie, process, intervals):
    intervals = merge_intervals(intervals)

    def contained(timestamp):
        return any(started <= timestamp <= finished for started, finished in intervals)

    gpu_rows = [row for row in gpu if contained(row["time_unix"])]
    pcie_samples = [row for row in pcie if contained(row["time_unix"])]
    result = {
        "interval_count": len(intervals),
        "interval_union_s": sum(finished - started for started, finished in intervals),
        "gpu_samples": len(gpu_rows),
        "gpu_util_avg_pct": mean([row["gpu_util_pct"] for row in gpu_rows]),
        "gpu_util_p50_pct": percentile([row["gpu_util_pct"] for row in gpu_rows], 0.50),
        "gpu_util_p95_pct": percentile([row["gpu_util_pct"] for row in gpu_rows], 0.95),
        "gpu_memory_util_avg_pct": mean([row["memory_util_pct"] for row in gpu_rows]),
        "gpu_util_zero_pct": (
            100 * sum(row["gpu_util_pct"] == 0 for row in gpu_rows) / len(gpu_rows)
            if gpu_rows
            else None
        ),
        "gpu_memory_peak_mib": max((row["memory_used_mib"] for row in gpu_rows), default=None),
        "gpu_power_avg_w": mean([row["power_w"] for row in gpu_rows]),
        "pcie_samples": len(pcie_samples),
        "pcie_rx_avg_mib_s": mean([row["rx_mib_s"] for row in pcie_samples]),
        "pcie_tx_avg_mib_s": mean([row["tx_mib_s"] for row in pcie_samples]),
        "pcie_rx_p50_mib_s": percentile([row["rx_mib_s"] for row in pcie_samples], 0.50),
        "pcie_tx_p50_mib_s": percentile([row["tx_mib_s"] for row in pcie_samples], 0.50),
        "pcie_rx_p95_mib_s": percentile([row["rx_mib_s"] for row in pcie_samples], 0.95),
        "pcie_tx_p95_mib_s": percentile([row["tx_mib_s"] for row in pcie_samples], 0.95),
        "pcie_rx_peak_mib_s": max((row["rx_mib_s"] for row in pcie_samples), default=None),
        "pcie_tx_peak_mib_s": max((row["tx_mib_s"] for row in pcie_samples), default=None),
    }

    process_totals = []
    for started, finished in intervals:
        summary = process_span_summary(process, started, finished)
        if summary:
            process_totals.append((finished - started, summary))
    if process_totals:
        cpu_totals = [
            item for item in process_totals if item[1]["average_cpu_cores"] is not None
        ]
        cpu_elapsed = sum(item[0] for item in cpu_totals)
        result.update(
            {
                "peak_rss_gib": max(item[1]["peak_rss_gib"] for item in process_totals),
                "read_gib": sum(item[1]["read_gib"] for item in process_totals),
                "write_gib": sum(item[1]["write_gib"] for item in process_totals),
                "average_cpu_cores": (
                    sum(item[0] * item[1]["average_cpu_cores"] for item in cpu_totals)
                    / cpu_elapsed
                    if cpu_elapsed > 0
                    else None
                ),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.result_dir

    events = load_events(root / "events.jsonl")
    stages = [event for event in events if event.get("event") == "stage_complete"]
    sweeps = [event for event in events if event.get("event") == "sweep_complete"]
    metrics_text = (
        (root / "metrics.prom").read_text(encoding="utf-8", errors="replace")
        if (root / "metrics.prom").exists()
        else ""
    )
    fa_pages_checked_series = prometheus_series(
        metrics_text, "hybrid_kv_fa_pages_checked_total"
    )
    fa_pages_skipped_series = prometheus_series(
        metrics_text, "hybrid_kv_fa_pages_skipped_total"
    )

    gpu = []
    for row in csv_rows(root / "gpu.csv"):
        try:
            gpu.append(
                {
                    "time_unix": local_timestamp(row["timestamp"]),
                    "memory_used_mib": number(row["memory_used_mib"]),
                    "memory_free_mib": number(row["memory_free_mib"]),
                    "gpu_util_pct": number(row["gpu_util_pct"]),
                    "memory_util_pct": number(row["memory_util_pct"]),
                    "power_w": number(row["power_w"]),
                }
            )
        except (KeyError, ValueError):
            continue
    pcie = pcie_rows(root / "pcie-dmon.txt")

    process = []
    for row in csv_rows(root / "process-io.csv"):
        try:
            process.append(
                {
                    "time_unix": local_timestamp(row["timestamp"]),
                    "rss_kib": number(row["rss_kib"]),
                    "read_bytes": number(row["read_bytes"]),
                    "write_bytes": number(row["write_bytes"]),
                    "cpu_ticks": number(row["utime_ticks"]) + number(row["stime_ticks"]),
                }
            )
        except (KeyError, ValueError):
            continue

    stage_rows = []
    for stage in stages:
        started = number(stage.get("started_unix"))
        finished = number(stage.get("time_unix"), started + number(stage.get("wall_s")))
        requests = stage.get("requests") or []
        prompt_end = [
            number(row.get("finished_unix")) - number((row.get("timings") or {}).get("predicted_ms")) / 1000
            for row in requests
        ]
        finish = [number(row.get("finished_unix")) for row in requests]
        decode_intervals = list(zip(prompt_end, finish))
        prompt_intervals = [
            (
                number(request.get("finished_unix"))
                - (
                    number((request.get("timings") or {}).get("prompt_ms"))
                    + number((request.get("timings") or {}).get("predicted_ms"))
                )
                / 1000,
                number(request.get("finished_unix"))
                - number((request.get("timings") or {}).get("predicted_ms")) / 1000,
            )
            for request in requests
        ]
        prompt_union_s = sum(end - start for start, end in merge_intervals(prompt_intervals))
        decode_union_s = sum(end - start for start, end in merge_intervals(decode_intervals))
        prompt_n = sum(number((request.get("timings") or {}).get("prompt_n")) for request in requests)
        predicted_n = sum(number((request.get("timings") or {}).get("predicted_n")) for request in requests)
        prompt_window_end = max(prompt_end, default=started)
        decode_window_start = min(prompt_end, default=started)
        decode_window_end = max(finish, default=finished)
        decode_after_all_prompts = max(prompt_end, default=started)
        decode_components = []
        for component_start, component_end, indices in interval_components(decode_intervals):
            component_tokens = sum(
                number((requests[index].get("timings") or {}).get("predicted_n"))
                for index in indices
            )
            component_s = component_end - component_start
            decode_components.append(
                {
                    "request_count": len(indices),
                    "sessions": [requests[index].get("session") for index in indices],
                    "tokens": component_tokens,
                    "union_s": component_s,
                    "aggregate_tps": component_tokens / component_s if component_s > 0 else None,
                }
            )
        row = {
            "logical_target": stage.get("logical_target"),
            "n_predict": stage.get("n_predict"),
            "wall_s": stage.get("wall_s"),
            "service_prompt_tps": stage.get("service_prompt_tps"),
            "service_output_tps": stage.get("service_output_tps"),
            "service_prompt_window_s": stage.get("service_prompt_window_s"),
            "service_prompt_window_tps": stage.get("service_prompt_window_tps"),
            "service_prompt_union_s": stage.get("service_prompt_union_s", prompt_union_s),
            "service_prompt_union_tps": stage.get(
                "service_prompt_union_tps", prompt_n / prompt_union_s if prompt_union_s else None
            ),
            "service_decode_window_s": stage.get("service_decode_window_s"),
            "service_decode_window_tps": stage.get("service_decode_window_tps"),
            "service_decode_union_s": stage.get("service_decode_union_s", decode_union_s),
            "service_decode_union_tps": stage.get(
                "service_decode_union_tps", predicted_n / decode_union_s if decode_union_s else None
            ),
            "prompt_finish_skew_s": (
                stage.get("prompt_finish_skew_s")
                if stage.get("prompt_finish_skew_s") is not None
                else (max(prompt_end) - min(prompt_end) if prompt_end else None)
            ),
            "request_finish_skew_s": (
                stage.get("request_finish_skew_s")
                if stage.get("request_finish_skew_s") is not None
                else (max(finish) - min(finish) if finish else None)
            ),
            "request_decode_tps": [
                number((request.get("timings") or {}).get("predicted_per_second"))
                for request in requests
            ],
            "draft_accepted": [
                int(number((request.get("timings") or {}).get("draft_n_accepted")))
                for request in requests
            ],
            "draft_total": [
                int(number((request.get("timings") or {}).get("draft_n"))) for request in requests
            ],
            "decode_concurrency_components": decode_components,
            # These activity windows intentionally overlap when one request
            # starts decoding while another is still in prefill.  Keeping
            # both views makes staggered production scheduling observable
            # without incorrectly attributing their shared interval to only
            # one phase.
            "prefill_window_hardware": {
                **span_summary(gpu, pcie, started, prompt_window_end),
                **process_span_summary(process, started, prompt_window_end),
            },
            "decode_window_hardware": {
                **span_summary(gpu, pcie, decode_window_start, decode_window_end),
                **process_span_summary(process, decode_window_start, decode_window_end),
            },
            "decode_union_hardware": interval_hardware_summary(
                gpu, pcie, process, decode_intervals
            ),
            "decode_after_all_prompts_hardware": {
                **span_summary(gpu, pcie, decode_after_all_prompts, decode_window_end),
                **process_span_summary(process, decode_after_all_prompts, decode_window_end),
            },
            "decode_fa_pages": fa_page_window_summary(
                fa_pages_checked_series,
                fa_pages_skipped_series,
                decode_window_start,
                decode_window_end,
            ),
            **span_summary(gpu, pcie, started, finished),
            **process_span_summary(process, started, finished),
        }
        stage_rows.append(row)

    sweep_rows = []
    for sweep in sweeps:
        requests = sweep.get("requests") or []
        started = min((number(request.get("started_unix")) for request in requests), default=0.0)
        finished = max(
            (number(request.get("finished_unix")) for request in requests),
            default=number(sweep.get("time_unix")),
        )
        decode_intervals = [
            (
                number(request.get("finished_unix"))
                - number(request.get("predicted_ms")) / 1000,
                number(request.get("finished_unix")),
            )
            for request in requests
        ]
        sweep_rows.append(
            {
                "sweep_index": sweep.get("sweep_index"),
                "mode": sweep.get("mode"),
                "concurrency_level": sweep.get("concurrency_level", len(requests)),
                "speculative_n_max": sweep.get("speculative_n_max"),
                "wall_s": sweep.get("wall_s"),
                "service_span_s": sweep.get("service_span_s"),
                "prompt_n": sweep.get("prompt_n"),
                "predicted_n": sweep.get("predicted_n"),
                "service_output_tps": sweep.get("service_output_tps"),
                "decode_union_s": sweep.get("decode_union_s"),
                "decode_union_tps": sweep.get("decode_union_tps"),
                "decode_span_tps": sweep.get("decode_span_tps"),
                "request_decode_tps": sweep.get("request_decode_tps"),
                "draft_n": sweep.get("draft_n"),
                "draft_n_accepted": sweep.get("draft_n_accepted"),
                "hardware": {
                    **span_summary(gpu, pcie, started, finished),
                    **process_span_summary(process, started, finished),
                },
                "decode_union_hardware": interval_hardware_summary(
                    gpu, pcie, process, decode_intervals
                ),
            }
        )

    process_summary = {}
    if process:
        elapsed = process[-1]["time_unix"] - process[0]["time_unix"]
        ticks = process[-1]["cpu_ticks"] - process[0]["cpu_ticks"]
        process_summary = {
            "peak_rss_gib": max(row["rss_kib"] for row in process) / 1024 / 1024,
            "read_gib": (process[-1]["read_bytes"] - process[0]["read_bytes"]) / 1024**3,
            "write_gib": (process[-1]["write_bytes"] - process[0]["write_bytes"]) / 1024**3,
            "average_cpu_cores": ticks / os.sysconf("SC_CLK_TCK") / elapsed if elapsed > 0 else None,
        }

    cache_disk = []
    for row in csv_rows(root / "cache-disk.csv"):
        try:
            cache_disk.append(
                {
                    "time_unix": local_timestamp(row["timestamp"]),
                    "bytes": number(row["bytes"]),
                    "filesystem_available_bytes": number(
                        row.get("filesystem_available_bytes"), math.nan
                    ),
                }
            )
        except (KeyError, ValueError):
            continue
    cache_filesystem_available = finite(
        [row["filesystem_available_bytes"] for row in cache_disk]
    )

    server_log = (root / "server.log").read_text(encoding="utf-8", errors="replace") if (root / "server.log").exists() else ""
    if not server_log:
        adjacent = root.with_name(root.name + "-server.log")
        if adjacent.exists():
            server_log = adjacent.read_text(encoding="utf-8", errors="replace")

    started = min((event.get("time_unix", math.inf) for event in events), default=math.inf)
    finished = max((event.get("time_unix", -math.inf) for event in events), default=-math.inf)
    overall_hardware = span_summary(gpu, pcie, started, finished) if started <= finished else {}
    host_kv_mib = [float(value) for value in re.findall(r"CUDA_Host KV buffer size =\s*([0-9.]+) MiB", server_log)]
    cuda_kv_mib = [float(value) for value in re.findall(r"CUDA0 KV buffer size =\s*([0-9.]+) MiB", server_log)]
    cuda_compute_mib = [
        float(value) for value in re.findall(r"CUDA0 compute buffer size =\s*([0-9.]+) MiB", server_log)
    ]
    per_step_mib = [
        float(value) for value in re.findall(r"CUDA0 per-step buffer =\s*([0-9.]+) MiB", server_log)
    ]
    adaptive_depth = prometheus_values(metrics_text, "mtp_adaptive_depth")
    adaptive_chunk = prometheus_values(metrics_text, "mtp_adaptive_prompt_chunk")
    adaptive_width = prometheus_values(metrics_text, "mtp_adaptive_decode_width")
    adaptive_resident = prometheus_values(metrics_text, "mtp_adaptive_resident_decode")
    adaptive_draft_row_budget = prometheus_values(
        metrics_text, "mtp_adaptive_checkpoint_draft_row_budget"
    )
    adaptive_draft_rows = prometheus_values(
        metrics_text, "mtp_adaptive_checkpoint_draft_rows"
    )
    adaptive_feasible_depth = prometheus_values(
        metrics_text, "mtp_adaptive_max_feasible_depth"
    )
    adaptive_reward = prometheus_values(metrics_text, "mtp_adaptive_reward")
    adaptive_prefill_ref = prometheus_values(metrics_text, "mtp_adaptive_prefill_reference_tps")
    adaptive_decode_ref = prometheus_values(metrics_text, "mtp_adaptive_decode_reference_tps")
    adaptive_decode_arm_rate = prometheus_values(
        metrics_text, "mtp_adaptive_decode_arm_rate_tps"
    )
    adaptive_width_arm_rate = prometheus_values(
        metrics_text, "mtp_adaptive_decode_width_arm_rate_tps"
    )
    adaptive_decode_updates = prometheus_values(metrics_text, "mtp_adaptive_decode_updates_total")
    adaptive_width_updates = prometheus_values(
        metrics_text, "mtp_adaptive_decode_width_updates_total"
    )
    adaptive_width_deferred = prometheus_values(
        metrics_text, "mtp_adaptive_decode_width_deferred_total"
    )
    adaptive_prompt_updates = prometheus_values(metrics_text, "mtp_adaptive_prompt_updates_total")
    adaptive_prompt_censored = prometheus_values(
        metrics_text, "mtp_adaptive_prompt_censored_total"
    )
    adaptive_prior_transfers = prometheus_values(
        metrics_text, "mtp_adaptive_decode_prior_transfers_total"
    )
    adaptive_width_prior_transfers = prometheus_values(
        metrics_text, "mtp_adaptive_decode_width_prior_transfers_total"
    )
    kv_free_cells = prometheus_values(metrics_text, "kv_cache_free_cells")
    kv_max_contiguous = prometheus_values(metrics_text, "kv_cache_max_contiguous_cells")
    kv_fragmentation_ratio = prometheus_values(metrics_text, "kv_cache_fragmentation_ratio")
    kv_batch_cap = prometheus_values(metrics_text, "kv_batch_effective_cap")
    kv_capacity_caps = prometheus_values(metrics_text, "kv_batch_capacity_caps_total")
    kv_batch_retries = prometheus_values(metrics_text, "kv_batch_retries_total")
    fa_pages_checked = prometheus_values(metrics_text, "hybrid_kv_fa_pages_checked_total")
    fa_pages_skipped = prometheus_values(metrics_text, "hybrid_kv_fa_pages_skipped_total")
    cache_load_ms = [
        float(value) for value in re.findall(r"prompt cache load took\s+([0-9.]+) ms", server_log)
    ]
    cache_save_ms = [
        float(value) for value in re.findall(r"prompt cache save took\s+([0-9.]+) ms", server_log)
    ]
    cache_stage_ms = [
        float(value)
        for value in re.findall(r"staged self-consistent hybrid checkpoint.*?in\s+([0-9.]+) ms", server_log)
    ]
    output = {
        "result_dir": str(root.resolve()),
        "completed": any(event.get("event") == "driver_complete" for event in events),
        "stage_count": len(stages),
        "sweep_count": len(sweeps),
        "hardware": overall_hardware,
        "process": process_summary,
        "numa_residency": numastat_summary(root / "numastat.txt"),
        "server_signals": {
            "hybrid_overlap_enabled": "hybrid CPU cold / GPU hot attention overlap enabled" in server_log,
            "cuda_oom_count": server_log.lower().count("out of memory"),
            "irregular_mixed_batch_warnings": server_log.count("irregular mixed-sequence batch"),
            "full_prompt_reprocess_count": server_log.count("forcing full prompt re-processing"),
            "prompt_cache_full_replace_decision_count": server_log.count("full_replace: true"),
            "checkpoint_created_count": server_log.count("created context checkpoint"),
            "checkpoint_restore_count": server_log.count("restored context checkpoint"),
            "superseded_staged_snapshot_release_count": server_log.count(
                "superseded staged snapshot"
            ),
            "disk_cache_eviction_count": server_log.count(
                "making disk prompt-cache room, removing oldest spilled entry"
            ),
            "disk_cache_atomic_headroom_eviction_count": server_log.count(
                "reason: atomic-write headroom"
            )
            + server_log.count(" + atomic-write headroom"),
            "disk_cache_enospc_count": server_log.count("No space left on device"),
            "adaptive_decode_selection_count": server_log.count("adaptive MTP decode arm selected"),
            "adaptive_decode_observation_count": server_log.count("adaptive MTP decode arm observed"),
            "adaptive_width_selection_count": server_log.count(
                "adaptive decode-width arm selected"
            ),
            "adaptive_width_observation_count": server_log.count(
                "adaptive decode-width arm observed"
            ),
            "adaptive_width_deferred_count": server_log.count(
                "adaptive decode-width observation deferred"
            ),
            "adaptive_prompt_selection_count": server_log.count("adaptive MTP prompt arm selected"),
            "adaptive_prompt_observation_count": server_log.count("adaptive MTP prompt arm observed"),
            "adaptive_prompt_censored_count": server_log.count("adaptive MTP prompt arm censored"),
            "adaptive_decode_prior_transfer_count": server_log.count(
                "adaptive MTP decode prior transferred"
            ),
            "adaptive_width_prior_transfer_count": server_log.count(
                "adaptive decode-width prior transferred"
            ),
            "fragmented_sequence_restore_count": server_log.count(
                "fragmented sequence restore"
            ),
            "fragmented_sequence_restore_failure_count": server_log.count(
                "failed to place restored sequence"
            ),
            "kv_batch_capacity_cap_log_count": server_log.count(
                "KV-fragmentation-aware prompt batch cap"
            ),
            "kv_batch_retry_warning_count": server_log.count(
                "failed to find free space in the KV cache"
            ),
        },
        "server_memory_layout": {
            "host_kv_mib": host_kv_mib,
            "cuda_kv_mib": cuda_kv_mib,
            "cuda_compute_mib": cuda_compute_mib,
            "cuda_per_step_mib": per_step_mib,
        },
        "disk_cache": {
            "samples": len(cache_disk),
            "peak_gib": max((row["bytes"] for row in cache_disk), default=0) / 1024**3,
            "filesystem_available_min_gib": (
                min(cache_filesystem_available) / 1024**3
                if cache_filesystem_available
                else None
            ),
            "load_ms": value_summary(cache_load_ms),
            "save_ms": value_summary(cache_save_ms),
            "stage_ms": value_summary(cache_stage_ms),
        },
        "hbm_counters": {
            "available": False,
            "reason": "uncore PMU access is unavailable to the benchmark user; use the documented LIKWID root recipe",
        },
        "hybrid_kv_fa_pages": {
            "sample_count": len(fa_pages_checked),
            "checked_total": int(fa_pages_checked[-1]) if fa_pages_checked else 0,
            "skipped_total": int(fa_pages_skipped[-1]) if fa_pages_skipped else 0,
            "skip_ratio": (
                fa_pages_skipped[-1] / fa_pages_checked[-1]
                if fa_pages_checked and fa_pages_skipped and fa_pages_checked[-1] > 0
                else None
            ),
        },
        "adaptive_scheduler": {
            "enabled": any(value > 0 for value in prometheus_values(metrics_text, "mtp_adaptive_enabled")),
            "sample_count": len(adaptive_depth),
            "depth_arms_observed": sorted({int(value) for value in adaptive_depth if value >= 0}),
            "prompt_chunk_arms_observed": sorted({int(value) for value in adaptive_chunk if value > 0}),
            "decode_width_arms_observed": sorted({int(value) for value in adaptive_width if value > 0}),
            "resident_decode_counts_observed": sorted(
                {int(value) for value in adaptive_resident if value > 0}
            ),
            "checkpoint_draft_row_budgets_observed": sorted(
                {int(value) for value in adaptive_draft_row_budget if value > 0}
            ),
            "checkpoint_draft_rows_observed": sorted(
                {int(value) for value in adaptive_draft_rows if value >= 0}
            ),
            "max_feasible_depths_observed": sorted(
                {int(value) for value in adaptive_feasible_depth if value >= 0}
            ),
            "last_reward": adaptive_reward[-1] if adaptive_reward else None,
            "last_prefill_reference_tps": adaptive_prefill_ref[-1] if adaptive_prefill_ref else None,
            "last_decode_reference_tps": adaptive_decode_ref[-1] if adaptive_decode_ref else None,
            "last_decode_arm_rate_tps": (
                adaptive_decode_arm_rate[-1] if adaptive_decode_arm_rate else None
            ),
            "last_decode_width_arm_rate_tps": (
                adaptive_width_arm_rate[-1] if adaptive_width_arm_rate else None
            ),
            "decode_updates": int(adaptive_decode_updates[-1]) if adaptive_decode_updates else 0,
            "decode_width_updates": (
                int(adaptive_width_updates[-1]) if adaptive_width_updates else 0
            ),
            "decode_width_deferred": (
                int(adaptive_width_deferred[-1]) if adaptive_width_deferred else 0
            ),
            "prompt_updates": int(adaptive_prompt_updates[-1]) if adaptive_prompt_updates else 0,
            "prompt_censored": (
                int(adaptive_prompt_censored[-1]) if adaptive_prompt_censored else 0
            ),
            "decode_prior_transfers": (
                int(adaptive_prior_transfers[-1]) if adaptive_prior_transfers else 0
            ),
            "decode_width_prior_transfers": (
                int(adaptive_width_prior_transfers[-1])
                if adaptive_width_prior_transfers
                else 0
            ),
        },
        "kv_fragmentation": {
            "sample_count": len(kv_fragmentation_ratio),
            "free_cells": value_summary(kv_free_cells),
            "max_contiguous_cells": value_summary(kv_max_contiguous),
            "fragmentation_ratio": value_summary(kv_fragmentation_ratio),
            "effective_batch_cap": value_summary(kv_batch_cap),
            "capacity_caps_total": int(kv_capacity_caps[-1]) if kv_capacity_caps else 0,
            "batch_retries_total": int(kv_batch_retries[-1]) if kv_batch_retries else 0,
        },
        "stages": stage_rows,
        "sweeps": sweep_rows,
    }

    output_path = args.output or root / "observability-summary.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
