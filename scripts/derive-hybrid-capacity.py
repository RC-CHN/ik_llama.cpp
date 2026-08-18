#!/usr/bin/env python3
"""Select a hybrid-KV hot-ring size from measured capacity and throughput.

The capacity calculation uses the measured peak GPU working set and the
model-specific hot-ring bytes/token reported while restoring the benchmark's
slot state.  The selector then walks the benchmark's capacity-profile chain
and exploits the best compatible, correct observation.  When an explicit
throughput target has not been reached, it proposes a bounded exploration
point.  There is no model, context-length, or machine lookup table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


COMPATIBILITY_KEYS = (
    "server_binary_sha256",
    "model_sha256",
    "workload",
    "parallel",
    "sessions",
    "ctx_size",
    "draft_ctx",
    "batch_size",
    "ubatch_size",
    "cache_type_k",
    "cache_type_v",
    "cache_type_k_draft",
    "cache_type_v_draft",
    "mtp_max",
    "mtp_adaptive",
    "milestones",
    "decode_tokens",
    "final_decode_tokens",
    "round_robin_context",
    "round_robin_decode_tokens",
)


def manifest_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def gpu_memory_mib(path: Path) -> tuple[float, float]:
    peak_used = 0.0
    total_samples: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                used = float(row["memory_used_mib"])
                free = float(row["memory_free_mib"])
            except (KeyError, TypeError, ValueError):
                continue
            peak_used = max(peak_used, used)
            total_samples.append(used + free)
    if not total_samples or peak_used <= 0:
        raise ValueError(f"no usable GPU memory samples in {path}")
    return peak_used, min(total_samples)


def hot_bytes_per_token(path: Path) -> float:
    pattern = re.compile(
        r"rebuilt\s+(\d+)\s+hybrid GPU hot-ring rows\s+"
        r"\(([0-9.]+)\s+MiB,"
    )
    samples: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        rows = int(match.group(1))
        size_mib = float(match.group(2))
        if rows > 0 and size_mib > 0:
            samples.append(size_mib * 1024 * 1024 / rows)
    if not samples:
        raise ValueError(
            f"{path} has no hot-ring rebuild measurement; "
            "use a completed restored-state benchmark as the capacity profile"
        )
    return max(samples)


def align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_parent(profile: Path, value: str) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (profile.parent / candidate).resolve()


def profile_chain(start: Path) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    current: Path | None = start.resolve()
    while current is not None and current not in seen:
        seen.add(current)
        result.append(current)
        manifest_path = current / "manifest.txt"
        if not manifest_path.exists():
            break
        manifest = manifest_values(manifest_path)
        current = resolve_parent(current, manifest.get("capacity_profile", ""))
    return result


def compatible(candidate: dict[str, str], reference: dict[str, str]) -> tuple[bool, list[str]]:
    mismatches = [
        key
        for key in COMPATIBILITY_KEYS
        if candidate.get(key) != reference.get(key)
    ]
    return not mismatches, mismatches


def valid_observation(profile: Path, reference: dict[str, str]) -> tuple[dict | None, str]:
    manifest_path = profile / "manifest.txt"
    if not manifest_path.exists():
        return None, "missing_manifest"
    manifest = manifest_values(manifest_path)
    is_compatible, mismatches = compatible(manifest, reference)
    if not is_compatible:
        return None, "incompatible:" + ",".join(mismatches)

    summary = load_json(profile / "summary.json")
    observability = load_json(profile / "observability-summary.json")
    if not summary.get("completed") or manifest.get("exit_status") != "0":
        return None, "incomplete"
    gates = summary.get("gates")
    if isinstance(gates, dict) and not all(gates.values()):
        return None, "failed_correctness_gate"
    signals = observability.get("server_signals", {})
    unsafe_signal_keys = (
        "cuda_oom_count",
        "full_prompt_reprocess_count",
        "fragmented_sequence_restore_failure_count",
        "kv_batch_retry_warning_count",
        "disk_cache_enospc_count",
    )
    if any(float(signals.get(key, 0) or 0) > 0 for key in unsafe_signal_keys):
        return None, "unsafe_server_signal"
    try:
        hot_tokens = int(manifest["hot_tokens"])
        throughput = float(summary["service_decode_union_tps"])
    except (KeyError, TypeError, ValueError):
        return None, "missing_hot_or_throughput"
    if hot_tokens <= 0 or not math.isfinite(throughput) or throughput <= 0:
        return None, "invalid_hot_or_throughput"
    return {
        "profile_result_dir": str(profile),
        "hot_tokens": hot_tokens,
        "service_decode_union_tps": throughput,
        "service_output_tps": summary.get("service_output_tps"),
        "manifest_sha256": file_sha256(manifest_path),
        "summary_sha256": file_sha256(profile / "summary.json"),
        "observability_summary_sha256": file_sha256(
            profile / "observability-summary.json"
        ),
    }, "valid"


def next_exploration(
    observations: list[dict], minimum: int, capacity: int, alignment: int
) -> int:
    """Choose a deterministic, block-aligned point near the current optimum.

    First sample the unobserved capacity endpoint.  Once both sides of the
    best observation are known, bisect the largest adjacent unobserved span.
    This is a bounded one-dimensional search, not a hardware-specific table.
    """

    observed = {int(row["hot_tokens"]) for row in observations}
    best = max(observations, key=lambda row: row["service_decode_union_tps"])
    best_hot = int(best["hot_tokens"])
    if capacity > best_hot and capacity not in observed:
        return capacity
    if minimum < best_hot and minimum not in observed:
        return minimum

    points = sorted({minimum, capacity, *observed})
    intervals = []
    for left, right in zip(points, points[1:]):
        if right - left <= alignment:
            continue
        distance = min(abs(best_hot - left), abs(best_hot - right))
        intervals.append((right - left, -distance, left, right))
    if not intervals:
        return best_hot
    _, _, left, right = max(intervals)
    midpoint = align_down((left + right) // 2, alignment)
    if midpoint <= left:
        midpoint = left + alignment
    return min(midpoint, right - alignment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--ubatch", type=int, required=True)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--headroom-fraction", type=float, default=0.0625)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--throughput-target", type=float)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.ubatch <= 0 or args.block <= 0:
        raise ValueError("--ubatch and --block must be positive")
    if not 0 < args.headroom_fraction < 1:
        raise ValueError("--headroom-fraction must be between zero and one")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.throughput_target is not None and args.throughput_target <= 0:
        raise ValueError("--throughput-target must be positive")

    profile = args.result_dir.resolve()
    manifest = manifest_values(profile / "manifest.txt")
    current_hot = int(manifest["hot_tokens"])
    peak_used_mib, total_mib = gpu_memory_mib(profile / "gpu.csv")
    bytes_per_token = hot_bytes_per_token(profile / "server.log")

    current_hot_mib = current_hot * bytes_per_token / (1024 * 1024)
    baseline_peak_mib = max(0.0, peak_used_mib - current_hot_mib)
    reserve_mib = total_mib * args.headroom_fraction
    hot_budget_mib = max(0.0, total_mib - baseline_peak_mib - reserve_mib)
    capacity = align_down(
        math.floor(hot_budget_mib * 1024 * 1024 / bytes_per_token),
        args.block,
    )
    minimum = align_up(args.ubatch, args.block)
    capacity = max(minimum, capacity)
    if args.max_tokens is not None:
        capacity = max(minimum, min(capacity, align_down(args.max_tokens, args.block)))

    observations = []
    rejected_profiles = []
    for candidate in profile_chain(profile):
        observation, reason = valid_observation(candidate, manifest)
        if observation is None:
            rejected_profiles.append(
                {"profile_result_dir": str(candidate), "reason": reason}
            )
            continue
        observation["capacity_eligible"] = observation["hot_tokens"] <= capacity
        observations.append(observation)

    # Keep the highest-throughput result for duplicate hot-window settings.
    observations_by_hot = {}
    for observation in observations:
        hot = observation["hot_tokens"]
        previous = observations_by_hot.get(hot)
        if previous is None or observation["service_decode_union_tps"] > previous[
            "service_decode_union_tps"
        ]:
            observations_by_hot[hot] = observation
    observations = sorted(observations_by_hot.values(), key=lambda row: row["hot_tokens"])

    eligible_observations = [row for row in observations if row["capacity_eligible"]]
    if eligible_observations:
        best = max(
            eligible_observations, key=lambda row: row["service_decode_union_tps"]
        )
        target_met = (
            args.throughput_target is None
            or best["service_decode_union_tps"] >= args.throughput_target
        )
        if target_met:
            selected = int(best["hot_tokens"])
            selection_reason = (
                "best_observed_target_met"
                if args.throughput_target is not None
                else "best_observed"
            )
        else:
            selected = next_exploration(
                eligible_observations, minimum, capacity, args.block
            )
            selection_reason = "bounded_throughput_exploration"
    else:
        best = None
        target_met = False
        selected = minimum
        selection_reason = "cold_start_minimum"

    result = {
        "hot_tokens": selected,
        "selection_reason": selection_reason,
        "throughput_target": args.throughput_target,
        "throughput_target_met": target_met,
        "best_observation": best,
        "observations": observations,
        "rejected_profiles": rejected_profiles,
        "capacity_hot_tokens": capacity,
        "minimum_hot_tokens": minimum,
        "block_tokens": args.block,
        "ubatch_tokens": args.ubatch,
        "profile_current_hot_tokens": current_hot,
        "hot_bytes_per_token": bytes_per_token,
        "gpu_peak_used_mib": peak_used_mib,
        "gpu_total_conservative_mib": total_mib,
        "baseline_peak_without_hot_mib": baseline_peak_mib,
        "reserved_headroom_mib": reserve_mib,
        "headroom_fraction": args.headroom_fraction,
        "hot_budget_mib": hot_budget_mib,
        "profile_result_dir": str(profile),
        "profile_manifest_sha256": file_sha256(profile / "manifest.txt"),
        "profile_gpu_csv_sha256": file_sha256(profile / "gpu.csv"),
        "profile_server_log_sha256": file_sha256(profile / "server.log"),
    }
    print(json.dumps(result, ensure_ascii=False) if args.json else selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
