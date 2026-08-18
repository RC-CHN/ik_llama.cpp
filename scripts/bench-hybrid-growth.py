#!/usr/bin/env python3
"""Exercise queued logical sessions while their contexts grow to a target size."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import struct
import threading
import time
import urllib.request
from pathlib import Path


def request_json(url: str, path: str, body: dict | None = None, timeout: float = 7200.0):
    payload = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    headers = {} if payload is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url.rstrip("/") + path, data=payload, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tokenize(url: str, text: str) -> list[int]:
    response = request_json(url, "/tokenize", {"content": text, "add_special": False}, 600)
    return [int(token) for token in response["tokens"]]


def tokenize_count(url: str, text: str) -> int:
    return len(tokenize(url, text))


def load_checkpoint_positions(path: Path) -> list[dict]:
    """Read only checkpoint metadata while skipping the large recurrent payloads."""
    header = struct.Struct("@IIQ")
    record = struct.Struct("@iiiiQ")
    with path.open("rb") as stream:
        raw = stream.read(header.size)
        if len(raw) != header.size:
            raise ValueError(f"truncated checkpoint header: {path}")
        magic, version, count = header.unpack(raw)
        if count > 1024:
            raise ValueError(f"implausible checkpoint count {count}: {path}")
        checkpoints = []
        for _ in range(count):
            raw = stream.read(record.size)
            if len(raw) != record.size:
                raise ValueError(f"truncated checkpoint record: {path}")
            pos_min, pos_max, pos_min_prompt, pos_max_prompt, data_len = record.unpack(raw)
            checkpoints.append(
                {
                    "pos_min": pos_min,
                    "pos_max": pos_max,
                    "pos_min_prompt": pos_min_prompt,
                    "pos_max_prompt": pos_max_prompt,
                    "data_bytes": data_len,
                }
            )
            stream.seek(data_len, 1)
    return checkpoints


def load_slot_states(path: Path) -> dict[int, dict]:
    result = {}
    for tokens_path in sorted(path.glob("slot-*.bin.tokens.json")):
        stem = tokens_path.name.removeprefix("slot-").removesuffix(".bin.tokens.json")
        try:
            id_slot = int(stem)
        except ValueError:
            continue
        checkpoint_path = path / f"slot-{id_slot}.bin.checkpoints"
        payload = json.loads(tokens_path.read_text(encoding="utf-8"))
        result[id_slot] = {
            "tokens": [int(token) for token in payload.get("tokens", [])],
            "checkpoints": (
                load_checkpoint_positions(checkpoint_path) if checkpoint_path.exists() else []
            ),
        }
    return result


def restored_prefix_status(prompt_tokens: list[int], state: dict) -> dict:
    saved_tokens = state["tokens"]
    common_prefix = 0
    for current, saved in zip(prompt_tokens, saved_tokens):
        if current != saved:
            break
        common_prefix += 1

    # apply_checkpoint() requires a checkpoint strictly before pos_next - 1.
    # Deriving this from the saved metadata avoids a context-size or machine
    # specific tolerance in the benchmark.
    usable = [
        checkpoint
        for checkpoint in state["checkpoints"]
        if checkpoint["pos_max"] < max(0, common_prefix - 1)
    ]
    best = max(usable, key=lambda checkpoint: checkpoint["pos_max"], default=None)
    expected_prompt_n_max = (
        max(0, len(prompt_tokens) - (best["pos_max"] + 1))
        if best is not None
        else len(prompt_tokens)
    )
    return {
        "saved_tokens": len(saved_tokens),
        "common_prefix_tokens": common_prefix,
        "checkpoint_count": len(state["checkpoints"]),
        "usable_checkpoint_count": len(usable),
        "best_checkpoint_pos_max": best["pos_max"] if best is not None else None,
        "expected_prompt_n_max": expected_prompt_n_max,
        "reuse_available": best is not None,
    }


def calibrate_cuts(url: str, corpus: str, milestones: list[int]) -> dict[int, tuple[int, int]]:
    full_tokens = tokenize_count(url, corpus)
    if full_tokens < milestones[-1]:
        raise ValueError(f"corpus has only {full_tokens} tokens, need {milestones[-1]}")

    result: dict[int, tuple[int, int]] = {}
    lower_pos = 0
    lower_count = 0
    for target in milestones:
        lo_pos, lo_count = lower_pos, lower_count
        hi_pos, hi_count = len(corpus), full_tokens
        best_pos, best_count = lo_pos, lo_count
        for _ in range(16):
            if hi_pos - lo_pos <= 1:
                break
            fraction = (target - lo_count) / max(1, hi_count - lo_count)
            guess = lo_pos + round((hi_pos - lo_pos) * fraction)
            guess = min(max(guess, lo_pos + 1), hi_pos - 1)
            count = tokenize_count(url, corpus[:guess])
            if abs(count - target) < abs(best_count - target):
                best_pos, best_count = guess, count
            if abs(count - target) <= 2:
                break
            if count < target:
                lo_pos, lo_count = guess, count
            else:
                hi_pos, hi_count = guess, count
        result[target] = (best_pos, best_count)
        lower_pos, lower_count = best_pos, best_count
    return result


def number(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def interval_union_seconds(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for started, finished in sorted(intervals):
        if finished <= started:
            continue
        if not merged or started > merged[-1][1]:
            merged.append([started, finished])
        else:
            merged[-1][1] = max(merged[-1][1], finished)
    return sum(finished - started for started, finished in merged)


def completion_body(
    prompt: str,
    n_predict: int,
    seed: int,
    id_slot: int | None = None,
    preserve_displaced: bool = True,
    speculative_n_max: int | None = None,
) -> dict:
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": True,
        "cache_prompt_preserve": preserve_displaced,
        "stream": False,
        "temperature": 0,
        "seed": seed,
        "ignore_eos": True,
        "saturate_predict": True,
    }
    if id_slot is not None:
        body["id_slot"] = id_slot
    if speculative_n_max is not None:
        body["speculative.n_max"] = speculative_n_max
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--milestones",
        default="20000,40000,60000,80000,100000,120000,140000,160000,180000,200000",
    )
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--final-decode-tokens", type=int, default=2048)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--speculative-n-max", type=int)
    parser.add_argument("--slot-state-dir", type=Path)
    parser.add_argument("--required-initial-restores", type=int, default=0)
    parser.add_argument(
        "--sticky-slots",
        action="store_true",
        help="pin session A/B/... to physical slot 0/1/... from the first request",
    )
    args = parser.parse_args()

    milestones = [int(value) for value in args.milestones.split(",") if value.strip()]
    if not milestones or milestones != sorted(set(milestones)) or milestones[0] <= 0:
        raise ValueError("--milestones must be a strictly increasing positive list")
    if args.sessions <= 0 or args.sessions > 26:
        raise ValueError("--sessions must be between 1 and 26")
    if args.speculative_n_max is not None and args.speculative_n_max < 0:
        raise ValueError("--speculative-n-max must be non-negative")
    if args.required_initial_restores < 0 or args.required_initial_restores > args.sessions:
        raise ValueError("--required-initial-restores must be between zero and --sessions")
    if args.required_initial_restores and args.slot_state_dir is None:
        raise ValueError("--required-initial-restores needs --slot-state-dir")

    initial_slot_states = load_slot_states(args.slot_state_dir) if args.slot_state_dir else {}
    if args.required_initial_restores > len(initial_slot_states):
        raise ValueError(
            f"need {args.required_initial_restores} initial slot states, found "
            f"{len(initial_slot_states)} in {args.slot_state_dir}"
        )

    args.result_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.result_dir / "events.jsonl"
    phase_path = args.result_dir / "phase.json"
    event_lock = threading.Lock()

    def emit(event: str, **fields) -> dict:
        row = {"event": event, "time_unix": time.time(), "time_monotonic": time.monotonic(), **fields}
        with event_lock:
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
        return row

    def phase(name: str, **fields) -> None:
        phase_path.write_text(
            json.dumps({"phase": name, "time_unix": time.time(), **fields}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    corpus = args.corpus.read_text(encoding="utf-8", errors="replace")
    phase("calibrating_token_cuts")
    cuts = calibrate_cuts(args.url, corpus, milestones)
    emit(
        "calibration_complete",
        corpus_sha256=hashlib.sha256(corpus.encode()).hexdigest(),
        cuts={str(target): {"chars": pos, "tokens": count} for target, (pos, count) in cuts.items()},
    )

    session_names = [chr(ord("A") + index) for index in range(args.sessions)]
    sessions = {
        name: {
            "prompt": f"// LOGICAL_PRODUCTION_SESSION_{name}\n",
            "cursor": 0,
            "seed": 101 + index,
            "id_slot": index if args.sticky_slots else None,
            "previous_prompt_tokens_client": 0,
            "previous_predicted_n": 0,
        }
        for index, name in enumerate(session_names)
    }
    stage_summaries: list[dict] = []

    def complete(name: str, stage_index: int, target: int, n_predict: int, barrier: threading.Barrier) -> dict:
        state = sessions[name]
        prompt = str(state["prompt"])
        prompt_token_ids = tokenize(args.url, prompt)
        prompt_tokens_client = len(prompt_token_ids)
        previous_prompt_tokens_client = int(state["previous_prompt_tokens_client"])
        previous_predicted_n = int(state["previous_predicted_n"])
        barrier.wait()
        started_unix = time.time()
        started = time.perf_counter()
        emit(
            "request_start",
            session=name,
            stage_index=stage_index,
            logical_target=target,
            prompt_tokens_client=prompt_tokens_client,
            n_predict=n_predict,
        )
        response = request_json(
            args.url,
            "/completion",
            completion_body(
                prompt,
                n_predict,
                int(state["seed"]),
                int(state["id_slot"])
                if args.sticky_slots and state["id_slot"] is not None
                else None,
                not args.sticky_slots,
                args.speculative_n_max,
            ),
        )
        finished_unix = time.time()
        content = response.get("content", "")
        timings = response.get("timings") or {}
        incremental_prompt_tokens_client = max(
            0, prompt_tokens_client - previous_prompt_tokens_client
        )
        # A reusable snapshot can lag by at most the previous response: some
        # live slots retain generated rows while a disk checkpoint represents
        # the preceding prompt tail.  Derive the allowance from the workload
        # itself rather than a fixed context-size or token tolerance.
        incremental_prompt_expected_max = (
            prompt_tokens_client
            if previous_prompt_tokens_client == 0
            else incremental_prompt_tokens_client + previous_predicted_n
        )
        prompt_n_server = int(number(timings.get("prompt_n")))
        prompt_reuse_valid = prompt_n_server <= incremental_prompt_expected_max
        initial_restore = None
        response_slot = response.get("id_slot")
        if stage_index == 0 and response_slot is not None and initial_slot_states:
            slot_state = initial_slot_states.get(int(response_slot))
            if slot_state is not None:
                initial_restore = restored_prefix_status(prompt_token_ids, slot_state)
                initial_restore["id_slot"] = int(response_slot)
                initial_restore["actual_prompt_n"] = prompt_n_server
                initial_restore["valid"] = bool(
                    initial_restore["reuse_available"]
                    and prompt_n_server <= initial_restore["expected_prompt_n_max"]
                )
                incremental_prompt_expected_max = initial_restore["expected_prompt_n_max"]
                prompt_reuse_valid = initial_restore["valid"]
        model_window_s = (
            number(timings.get("prompt_ms")) + number(timings.get("predicted_ms"))
        ) / 1000
        model_started_unix = finished_unix - model_window_s
        row = {
            "session": name,
            "stage_index": stage_index,
            "logical_target": target,
            "prompt_tokens_client": prompt_tokens_client,
            "previous_prompt_tokens_client": previous_prompt_tokens_client,
            "incremental_prompt_tokens_client": incremental_prompt_tokens_client,
            "incremental_prompt_expected_max": incremental_prompt_expected_max,
            "prompt_reuse_valid": prompt_reuse_valid,
            "prompt_reprocess_excess": max(
                0, prompt_n_server - incremental_prompt_expected_max
            ),
            "n_predict_requested": n_predict,
            "started_unix": started_unix,
            "finished_unix": finished_unix,
            "wall_s": time.perf_counter() - started,
            "pre_model_wait_s": max(0.0, model_started_unix - started_unix),
            "id_slot": response.get("id_slot"),
            "tokens_cached": response.get("tokens_cached"),
            "truncated": response.get("truncated"),
            "content_chars": len(content),
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "initial_restore": initial_restore,
            "timings": timings,
            "content": content,
        }
        emit("request_complete", **{key: value for key, value in row.items() if key != "content"})
        if args.sticky_slots:
            response_slot = response.get("id_slot")
            if response_slot is None:
                raise RuntimeError(f"server did not return id_slot for sticky session {name}")
            previous_slot = state["id_slot"]
            if previous_slot is not None and int(response_slot) != int(previous_slot):
                raise RuntimeError(
                    f"sticky session {name} moved from slot {previous_slot} to {response_slot}"
                )
            state["id_slot"] = int(response_slot)
        return row

    benchmark_started = time.perf_counter()
    for stage_index, target in enumerate(milestones):
        cut_pos, calibrated_tokens = cuts[target]
        for name, state in sessions.items():
            old_cursor = int(state["cursor"])
            state["prompt"] = (
                str(state["prompt"])
                + corpus[old_cursor:cut_pos]
                + f"\n// session {name}: context milestone {target}; continue the code.\n"
            )
            state["cursor"] = cut_pos

        n_predict = args.final_decode_tokens if stage_index == len(milestones) - 1 else args.decode_tokens
        phase("growth_and_decode", stage_index=stage_index, logical_target=target, n_predict=n_predict)
        emit(
            "stage_start",
            stage_index=stage_index,
            logical_target=target,
            calibrated_corpus_tokens=calibrated_tokens,
            n_predict=n_predict,
        )

        barrier = threading.Barrier(len(sessions) + 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sessions)) as pool:
            futures = {
                name: pool.submit(complete, name, stage_index, target, n_predict, barrier)
                for name in sessions
            }
            barrier.wait()
            stage_started_unix = time.time()
            stage_started = time.perf_counter()
            rows = [futures[name].result() for name in sessions]
            stage_wall_s = time.perf_counter() - stage_started

        initial_restore_rows = [
            row for row in rows if row.get("initial_restore") is not None
        ]
        initial_restore_valid = sum(
            bool(row["initial_restore"].get("valid")) for row in initial_restore_rows
        )
        if stage_index == 0 and args.required_initial_restores:
            emit(
                "initial_restore_validation",
                required=args.required_initial_restores,
                valid=initial_restore_valid,
                requests=[
                    {
                        "session": row["session"],
                        **row["initial_restore"],
                    }
                    for row in initial_restore_rows
                ],
            )
            if initial_restore_valid < args.required_initial_restores:
                raise RuntimeError(
                    f"only {initial_restore_valid}/{args.required_initial_restores} "
                    "required initial slot restores reused a checkpoint"
                )

        reuse_violations = [
            row
            for row in rows
            if not row["prompt_reuse_valid"] and row.get("initial_restore") is None
        ]
        if reuse_violations:
            phase(
                "invalid_cache_reuse",
                stage_index=stage_index,
                logical_target=target,
            )
            emit(
                "stage_invalid",
                stage_index=stage_index,
                logical_target=target,
                reason="prompt_reprocess_exceeded_workload_derived_increment",
                violations=[
                    {
                        "session": row["session"],
                        "prompt_n": int(number(row["timings"].get("prompt_n"))),
                        "incremental_prompt_expected_max": row[
                            "incremental_prompt_expected_max"
                        ],
                        "prompt_reprocess_excess": row["prompt_reprocess_excess"],
                    }
                    for row in reuse_violations
                ],
            )
            raise RuntimeError(
                "cache reuse regression at logical target "
                f"{target}: "
                + ", ".join(
                    f"{row['session']} processed {int(number(row['timings'].get('prompt_n')))} "
                    f"> {row['incremental_prompt_expected_max']} expected"
                    for row in reuse_violations
                )
            )

        for row in rows:
            sessions[row["session"]]["prompt"] = str(sessions[row["session"]]["prompt"]) + row.pop("content")
            sessions[row["session"]]["previous_prompt_tokens_client"] = row[
                "prompt_tokens_client"
            ]
            sessions[row["session"]]["previous_predicted_n"] = int(
                number(row["timings"].get("predicted_n"))
            )

        prompt_n = sum(number(row["timings"].get("prompt_n")) for row in rows)
        prompt_ms = sum(number(row["timings"].get("prompt_ms")) for row in rows)
        predicted_n = sum(number(row["timings"].get("predicted_n")) for row in rows)
        predicted_ms = sum(number(row["timings"].get("predicted_ms")) for row in rows)
        request_model_s = [
            (number(row["timings"].get("prompt_ms")) + number(row["timings"].get("predicted_ms")))
            / 1000
            for row in rows
        ]
        summed_request_model_s = sum(request_model_s)
        critical_request_model_s = max(request_model_s, default=0.0)

        # Request timings overlap when the scheduler admits both slots.  Keep the
        # sum for per-request model throughput, but use interval unions/critical
        # paths for system throughput and scheduler overhead.
        decode_intervals = [
            (
                number(row.get("finished_unix"))
                - number(row["timings"].get("predicted_ms")) / 1000,
                number(row.get("finished_unix")),
            )
            for row in rows
            if number(row["timings"].get("predicted_ms")) > 0
        ]
        decode_start_unix = min((started for started, _ in decode_intervals), default=stage_started_unix)
        decode_end_unix = max(
            (number(row.get("finished_unix")) for row in rows),
            default=stage_started_unix,
        )
        service_decode_window_s = max(0.0, decode_end_unix - decode_start_unix)
        service_decode_union_s = interval_union_seconds(decode_intervals)
        prompt_intervals = [
            (
                number(row.get("finished_unix"))
                - (
                    number(row["timings"].get("prompt_ms"))
                    + number(row["timings"].get("predicted_ms"))
                )
                / 1000,
                number(row.get("finished_unix"))
                - number(row["timings"].get("predicted_ms")) / 1000,
            )
            for row in rows
            if number(row["timings"].get("prompt_ms")) > 0
        ]
        service_prompt_union_s = interval_union_seconds(prompt_intervals)
        prompt_end_times = [
            number(row.get("finished_unix"))
            - number(row["timings"].get("predicted_ms")) / 1000
            for row in rows
        ]
        prompt_end_unix = max(prompt_end_times, default=stage_started_unix)
        service_prompt_window_s = max(0.0, prompt_end_unix - stage_started_unix)
        request_finish_times = [number(row.get("finished_unix")) for row in rows]
        request_decode_times = [number(row["timings"].get("predicted_ms")) / 1000 for row in rows]
        request_decode_tps = [number(row["timings"].get("predicted_per_second")) for row in rows]
        pre_model_wait_times = [number(row.get("pre_model_wait_s")) for row in rows]
        stage_summary = {
            "stage_index": stage_index,
            "logical_target": target,
            "calibrated_corpus_tokens": calibrated_tokens,
            "n_predict": n_predict,
            "started_unix": stage_started_unix,
            "wall_s": stage_wall_s,
            "prompt_n": prompt_n,
            "predicted_n": predicted_n,
            "service_prompt_tps": prompt_n / stage_wall_s if stage_wall_s else None,
            "service_output_tps": predicted_n / stage_wall_s if stage_wall_s else None,
            "service_useful_tps": (prompt_n + predicted_n) / stage_wall_s if stage_wall_s else None,
            "model_prompt_tps": prompt_n / (prompt_ms / 1000) if prompt_ms else None,
            "model_decode_tps": predicted_n / (predicted_ms / 1000) if predicted_ms else None,
            "model_prompt_s": prompt_ms / 1000,
            "model_decode_s": predicted_ms / 1000,
            "summed_request_model_s": summed_request_model_s,
            "critical_request_model_s": critical_request_model_s,
            "request_overlap_potential_s": max(0.0, summed_request_model_s - critical_request_model_s),
            "critical_path_overhead_s": stage_wall_s - critical_request_model_s,
            "service_prompt_window_s": service_prompt_window_s,
            "service_prompt_window_tps": prompt_n / service_prompt_window_s if service_prompt_window_s else None,
            "service_prompt_union_s": service_prompt_union_s,
            "service_prompt_union_tps": prompt_n / service_prompt_union_s if service_prompt_union_s else None,
            "service_decode_window_s": service_decode_window_s,
            "service_decode_window_tps": predicted_n / service_decode_window_s if service_decode_window_s else None,
            "service_decode_union_s": service_decode_union_s,
            "service_decode_union_tps": predicted_n / service_decode_union_s if service_decode_union_s else None,
            "prompt_finish_skew_s": (
                max(prompt_end_times) - min(prompt_end_times) if prompt_end_times else 0.0
            ),
            "request_finish_skew_s": (
                max(request_finish_times) - min(request_finish_times) if request_finish_times else 0.0
            ),
            "request_decode_s_min": min(request_decode_times, default=0.0),
            "request_decode_s_max": max(request_decode_times, default=0.0),
            "request_decode_tps_min": min(request_decode_tps, default=0.0),
            "request_decode_tps_max": max(request_decode_tps, default=0.0),
            "pre_model_wait_s_min": min(pre_model_wait_times, default=0.0),
            "pre_model_wait_s_max": max(pre_model_wait_times, default=0.0),
            "pre_model_wait_s_mean": (
                sum(pre_model_wait_times) / len(pre_model_wait_times)
                if pre_model_wait_times
                else 0.0
            ),
            "requests": rows,
        }
        stage_summaries.append(stage_summary)
        emit("stage_complete", **stage_summary)

    total_wall_s = time.perf_counter() - benchmark_started
    total_prompt_n = sum(stage["prompt_n"] for stage in stage_summaries)
    total_predicted_n = sum(stage["predicted_n"] for stage in stage_summaries)
    total_prompt_s = sum(stage["model_prompt_s"] for stage in stage_summaries)
    total_decode_s = sum(stage["model_decode_s"] for stage in stage_summaries)
    total_summed_request_model_s = sum(stage["summed_request_model_s"] for stage in stage_summaries)
    total_critical_request_model_s = sum(stage["critical_request_model_s"] for stage in stage_summaries)
    total_service_prompt_window_s = sum(stage["service_prompt_window_s"] for stage in stage_summaries)
    total_service_decode_window_s = sum(stage["service_decode_window_s"] for stage in stage_summaries)
    total_service_prompt_union_s = sum(stage["service_prompt_union_s"] for stage in stage_summaries)
    total_service_decode_union_s = sum(stage["service_decode_union_s"] for stage in stage_summaries)
    summary = {
        "completed": len(stage_summaries) == len(milestones),
        "sessions": list(sessions),
        "sticky_slots": args.sticky_slots,
        "speculative_n_max": args.speculative_n_max,
        "required_initial_restores": args.required_initial_restores,
        "session_slot_map": {name: state["id_slot"] for name, state in sessions.items()},
        "milestones": milestones,
        "decode_tokens": args.decode_tokens,
        "final_decode_tokens": args.final_decode_tokens,
        "wall_s": total_wall_s,
        "total_prompt_n": total_prompt_n,
        "total_predicted_n": total_predicted_n,
        "service_prompt_tps": total_prompt_n / total_wall_s if total_wall_s else None,
        "service_output_tps": total_predicted_n / total_wall_s if total_wall_s else None,
        "service_useful_tps": (total_prompt_n + total_predicted_n) / total_wall_s if total_wall_s else None,
        "model_prompt_tps": total_prompt_n / total_prompt_s if total_prompt_s else None,
        "model_decode_tps": total_predicted_n / total_decode_s if total_decode_s else None,
        "model_prompt_s": total_prompt_s,
        "model_decode_s": total_decode_s,
        "summed_request_model_s": total_summed_request_model_s,
        "critical_request_model_s": total_critical_request_model_s,
        "request_overlap_potential_s": max(
            0.0, total_summed_request_model_s - total_critical_request_model_s
        ),
        "critical_path_overhead_s": total_wall_s - total_critical_request_model_s,
        "service_prompt_window_s": total_service_prompt_window_s,
        "service_prompt_window_tps": (
            total_prompt_n / total_service_prompt_window_s if total_service_prompt_window_s else None
        ),
        "service_prompt_union_s": total_service_prompt_union_s,
        "service_prompt_union_tps": (
            total_prompt_n / total_service_prompt_union_s if total_service_prompt_union_s else None
        ),
        "service_decode_window_s": total_service_decode_window_s,
        "service_decode_window_tps": (
            total_predicted_n / total_service_decode_window_s if total_service_decode_window_s else None
        ),
        "service_decode_union_s": total_service_decode_union_s,
        "service_decode_union_tps": (
            total_predicted_n / total_service_decode_union_s if total_service_decode_union_s else None
        ),
        "final_prompt_tokens_client": {
            name: tokenize_count(args.url, str(state["prompt"])) for name, state in sessions.items()
        },
        "stage_summaries": stage_summaries,
    }
    (args.result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    phase("complete")
    emit("driver_complete", summary_path=str((args.result_dir / "summary.json").resolve()))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
