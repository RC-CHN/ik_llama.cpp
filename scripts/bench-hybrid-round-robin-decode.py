#!/usr/bin/env python3
"""Measure three logical long-context sessions on two physical slots.

The schedule is the three-edge Euler cycle AB -> BC -> CA.  Every logical
session receives two equal decode quanta while one member of each adjacent
pair remains resident.  This keeps both physical slots useful.  The scheduler
derives displaced-session liveness from the remaining queue, so the tiered
cache writes only a session that will actually return.
"""

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


def common_prefix_size(left: list[int], right: list[int]) -> int:
    result = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        result += 1
    return result


def load_checkpoint_positions(path: Path) -> list[dict]:
    header = struct.Struct("@IIQ")
    record = struct.Struct("@iiiiQ")
    with path.open("rb") as stream:
        raw = stream.read(header.size)
        if len(raw) != header.size:
            raise ValueError(f"truncated checkpoint header: {path}")
        _magic, _version, count = header.unpack(raw)
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
    common_prefix = common_prefix_size(prompt_tokens, state["tokens"])
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
        "saved_tokens": len(state["tokens"]),
        "common_prefix_tokens": common_prefix,
        "checkpoint_count": len(state["checkpoints"]),
        "usable_checkpoint_count": len(usable),
        "best_checkpoint_pos_max": best["pos_max"] if best is not None else None,
        "expected_prompt_n_max": expected_prompt_n_max,
        "checkpoint_lag": expected_prompt_n_max,
        "reuse_available": best is not None,
    }


def calibrate_cut(url: str, corpus: str, target: int) -> tuple[int, int]:
    full_tokens = len(tokenize(url, corpus))
    if full_tokens < target:
        raise ValueError(f"corpus has only {full_tokens} tokens, need {target}")
    lo_pos, lo_count = 0, 0
    hi_pos, hi_count = len(corpus), full_tokens
    best_pos, best_count = lo_pos, lo_count
    for _ in range(20):
        if hi_pos - lo_pos <= 1:
            break
        fraction = (target - lo_count) / max(1, hi_count - lo_count)
        guess = lo_pos + round((hi_pos - lo_pos) * fraction)
        guess = min(max(guess, lo_pos + 1), hi_pos - 1)
        count = len(tokenize(url, corpus[:guess]))
        if abs(count - target) < abs(best_count - target):
            best_pos, best_count = guess, count
        if abs(count - target) <= 2:
            break
        if count < target:
            lo_pos, lo_count = guess, count
        else:
            hi_pos, hi_count = guess, count
    return best_pos, best_count


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
    id_slot: int,
    preserve_displaced: bool,
    speculative_n_max: int | None,
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
        "id_slot": id_slot,
    }
    if speculative_n_max is not None:
        body["speculative.n_max"] = speculative_n_max
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--slot-state-dir", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=200000)
    parser.add_argument("--decode-tokens", type=int, default=2048)
    parser.add_argument("--physical-slots", type=int, default=2)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--speculative-n-max", type=int)
    parser.add_argument("--throughput-target", type=float, default=20.0)
    args = parser.parse_args()

    if args.physical_slots != 2 or args.sessions != 3:
        raise ValueError("round-robin decode currently requires exactly 2 slots and 3 sessions")
    if args.decode_tokens <= 0 or args.decode_tokens % 2 != 0:
        raise ValueError("--decode-tokens must be a positive even number")
    if args.speculative_n_max is not None and args.speculative_n_max < 0:
        raise ValueError("--speculative-n-max must be non-negative")

    args.result_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.result_dir / "events.jsonl"
    phase_path = args.result_dir / "phase.json"
    event_lock = threading.Lock()

    def emit(event: str, **fields) -> dict:
        row = {
            "event": event,
            "time_unix": time.time(),
            "time_monotonic": time.monotonic(),
            **fields,
        }
        with event_lock:
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
        return row

    def phase(name: str, **fields) -> None:
        phase_path.write_text(
            json.dumps({"phase": name, "time_unix": time.time(), **fields}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    corpus = args.corpus.read_text(encoding="utf-8", errors="replace")
    phase("calibrating_token_cut", context_tokens=args.context_tokens)
    cut_pos, calibrated_tokens = calibrate_cut(args.url, corpus, args.context_tokens)

    def base_prompt(name: str) -> str:
        return (
            f"// LOGICAL_PRODUCTION_SESSION_{name}\n"
            + corpus[:cut_pos]
            + f"\n// session {name}: context milestone {args.context_tokens}; continue the code.\n"
        )

    prompts = {
        "A": base_prompt("A"),
        "B": base_prompt("B"),
        # C is a production-style fork of A.  It owns an independent logical
        # history while deliberately exercising long-prefix reuse instead of
        # manufacturing a third 13 GiB copy of identical canonical KV rows.
        "C": base_prompt("A")
        + "\n// forked logical session C: preserve the shared context and continue independently.\n",
    }
    logical = {
        name: {
            "prompt": prompt,
            "seed": 401 + index,
            "last_prompt_tokens": None,
            "last_predicted_n": 0,
            "decoded": 0,
            "checkpoint_lag": 0,
        }
        for index, (name, prompt) in enumerate(prompts.items())
    }

    initial_states = load_slot_states(args.slot_state_dir)
    if len(initial_states) < args.physical_slots:
        raise ValueError(
            f"need {args.physical_slots} initial slot states, found {len(initial_states)}"
        )
    initial_status = {}
    for id_slot, name in enumerate(("A", "B")):
        prompt_tokens = tokenize(args.url, str(logical[name]["prompt"]))
        status = restored_prefix_status(prompt_tokens, initial_states[id_slot])
        status["id_slot"] = id_slot
        status["session"] = name
        initial_status[name] = status
        logical[name]["checkpoint_lag"] = status["checkpoint_lag"]
    logical["C"]["checkpoint_lag"] = logical["A"]["checkpoint_lag"]
    emit(
        "calibration_complete",
        context_tokens=args.context_tokens,
        calibrated_corpus_tokens=calibrated_tokens,
        corpus_sha256=hashlib.sha256(corpus.encode()).hexdigest(),
        prompt_tokens={name: len(tokenize(args.url, str(state["prompt"]))) for name, state in logical.items()},
        initial_states=initial_status,
    )

    # All three edges of K3.  The slot assignment keeps the shared vertex in
    # place across each transition: AB -> CB -> CA.
    schedule = [(("A", 0), ("B", 1)), (("C", 0), ("B", 1)), (("C", 0), ("A", 1))]
    chunk_tokens = args.decode_tokens // 2
    slot_runtime: dict[int, dict] = {
        0: {"session": "A", "prompt_tokens": tokenize(args.url, prompts["A"]), "predicted_n": 0},
        1: {"session": "B", "prompt_tokens": tokenize(args.url, prompts["B"]), "predicted_n": 0},
    }
    all_rows: list[dict] = []
    stage_summaries: list[dict] = []
    all_reuse_valid = True

    def complete(
        name: str,
        id_slot: int,
        wave_index: int,
        barrier: threading.Barrier,
    ) -> dict:
        state = logical[name]
        prompt = str(state["prompt"])
        prompt_tokens = tokenize(args.url, prompt)
        previous_slot = slot_runtime[id_slot]
        future_sessions = {
            future_name
            for future_wave in schedule[wave_index + 1 :]
            for future_name, _ in future_wave
        }
        preserve_displaced = (
            previous_slot["session"] != name
            and previous_slot["session"] in future_sessions
        )
        physical_lcp = common_prefix_size(prompt_tokens, previous_slot["prompt_tokens"])
        physical_expected = (
            max(len(prompt_tokens) - physical_lcp, int(previous_slot["predicted_n"]))
            + int(state["checkpoint_lag"])
        )
        last_prompt_tokens = state["last_prompt_tokens"]
        logical_expected = None
        if last_prompt_tokens is not None:
            logical_lcp = common_prefix_size(prompt_tokens, last_prompt_tokens)
            logical_expected = (
                max(len(prompt_tokens) - logical_lcp, int(state["last_predicted_n"]))
                + int(state["checkpoint_lag"])
            )
        expected_prompt_n_max = (
            min(physical_expected, logical_expected)
            if logical_expected is not None
            else physical_expected
        )
        if wave_index == 0:
            expected_prompt_n_max = int(initial_status[name]["expected_prompt_n_max"])

        barrier.wait()
        started_unix = time.time()
        started = time.perf_counter()
        emit(
            "request_start",
            wave_index=wave_index,
            session=name,
            id_slot=id_slot,
            prompt_tokens_client=len(prompt_tokens),
            n_predict=chunk_tokens,
            previous_resident=previous_slot["session"],
            preserve_displaced=preserve_displaced,
        )
        response = request_json(
            args.url,
            "/completion",
            completion_body(
                prompt,
                chunk_tokens,
                int(state["seed"]),
                id_slot,
                preserve_displaced,
                args.speculative_n_max,
            ),
        )
        finished_unix = time.time()
        timings = response.get("timings") or {}
        prompt_n = int(number(timings.get("prompt_n")))
        predicted_n = int(number(timings.get("predicted_n")))
        content = response.get("content", "")
        row = {
            "session": name,
            "wave_index": wave_index,
            "id_slot": response.get("id_slot"),
            "previous_resident": previous_slot["session"],
            "preserve_displaced": preserve_displaced,
            "started_unix": started_unix,
            "finished_unix": finished_unix,
            "wall_s": time.perf_counter() - started,
            "prompt_tokens_client": len(prompt_tokens),
            "physical_common_prefix_tokens": physical_lcp,
            "logical_expected_prompt_n_max": logical_expected,
            "expected_prompt_n_max": expected_prompt_n_max,
            "prompt_reuse_valid": prompt_n <= expected_prompt_n_max,
            "prompt_reprocess_excess": max(0, prompt_n - expected_prompt_n_max),
            "n_predict_requested": chunk_tokens,
            "tokens_cached": response.get("tokens_cached"),
            "truncated": response.get("truncated"),
            "content_chars": len(content),
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "timings": timings,
            "content": content,
            "prompt_token_ids": prompt_tokens,
            "predicted_n": predicted_n,
        }
        emit(
            "request_complete",
            **{key: value for key, value in row.items() if key not in {"content", "prompt_token_ids"}},
        )
        return row

    benchmark_started_unix = time.time()
    benchmark_started = time.perf_counter()
    previous_wave_finished = benchmark_started_unix
    for wave_index, placements in enumerate(schedule):
        phase(
            "round_robin_decode",
            wave_index=wave_index,
            placements=[{"session": name, "id_slot": id_slot} for name, id_slot in placements],
        )
        emit(
            "stage_start",
            stage_index=wave_index,
            wave_index=wave_index,
            logical_target=args.context_tokens,
            n_predict=chunk_tokens,
            placements=[{"session": name, "id_slot": id_slot} for name, id_slot in placements],
        )
        barrier = threading.Barrier(len(placements) + 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(placements)) as pool:
            futures = {
                name: pool.submit(complete, name, id_slot, wave_index, barrier)
                for name, id_slot in placements
            }
            barrier.wait()
            wave_started_unix = time.time()
            wave_started = time.perf_counter()
            rows = [futures[name].result() for name, _ in placements]
            wave_wall_s = time.perf_counter() - wave_started

        for row in rows:
            if int(row["id_slot"]) != next(
                id_slot for session, id_slot in placements if session == row["session"]
            ):
                raise RuntimeError(
                    f"session {row['session']} moved to unexpected slot {row['id_slot']}"
                )
            if row["predicted_n"] != chunk_tokens:
                raise RuntimeError(
                    f"session {row['session']} emitted {row['predicted_n']} != {chunk_tokens}"
                )
            all_reuse_valid = all_reuse_valid and bool(row["prompt_reuse_valid"])
            state = logical[row["session"]]
            state["prompt"] = str(state["prompt"]) + row["content"]
            state["last_prompt_tokens"] = row["prompt_token_ids"]
            state["last_predicted_n"] = row["predicted_n"]
            state["decoded"] = int(state["decoded"]) + row["predicted_n"]
            slot_runtime[int(row["id_slot"])] = {
                "session": row["session"],
                "prompt_tokens": row["prompt_token_ids"],
                "predicted_n": row["predicted_n"],
            }

        public_rows = [
            {key: value for key, value in row.items() if key not in {"content", "prompt_token_ids"}}
            for row in rows
        ]
        all_rows.extend(public_rows)
        prompt_n = sum(number(row["timings"].get("prompt_n")) for row in rows)
        prompt_ms = sum(number(row["timings"].get("prompt_ms")) for row in rows)
        predicted_n = sum(number(row["timings"].get("predicted_n")) for row in rows)
        decode_intervals = [
            (
                number(row["finished_unix"])
                - number(row["timings"].get("predicted_ms")) / 1000,
                number(row["finished_unix"]),
            )
            for row in rows
        ]
        decode_union_s = interval_union_seconds(decode_intervals)
        stage_summary = {
            "stage_index": wave_index,
            "wave_index": wave_index,
            "logical_target": args.context_tokens,
            "calibrated_corpus_tokens": calibrated_tokens,
            "n_predict": chunk_tokens,
            "started_unix": wave_started_unix,
            "wall_s": wave_wall_s,
            "transition_since_previous_wave_s": max(
                0.0, wave_started_unix - previous_wave_finished
            ),
            "prompt_n": prompt_n,
            "predicted_n": predicted_n,
            "service_prompt_tps": prompt_n / wave_wall_s if wave_wall_s else None,
            "service_output_tps": predicted_n / wave_wall_s if wave_wall_s else None,
            "service_useful_tps": (prompt_n + predicted_n) / wave_wall_s if wave_wall_s else None,
            "model_prompt_tps": prompt_n / (prompt_ms / 1000) if prompt_ms else None,
            "service_decode_union_s": decode_union_s,
            "service_decode_union_tps": predicted_n / decode_union_s if decode_union_s else None,
            "service_decode_window_s": max(end for _, end in decode_intervals)
            - min(start for start, _ in decode_intervals),
            "service_decode_window_tps": predicted_n
            / (max(end for _, end in decode_intervals) - min(start for start, _ in decode_intervals)),
            "requests": public_rows,
        }
        stage_summaries.append(stage_summary)
        emit("stage_complete", **stage_summary)
        previous_wave_finished = max(row["finished_unix"] for row in rows)

        if not all(row["prompt_reuse_valid"] for row in rows):
            raise RuntimeError(
                "prompt-cache reuse validation failed in wave "
                f"{wave_index}: "
                + ", ".join(
                    f"{row['session']} prompt_n={row['timings'].get('prompt_n')} "
                    f"> expected={row['expected_prompt_n_max']}"
                    for row in rows
                    if not row["prompt_reuse_valid"]
                )
            )

    total_wall_s = time.perf_counter() - benchmark_started
    total_prompt_n = sum(
        int(number(row["timings"].get("prompt_n"))) for row in all_rows
    )
    total_predicted_n = sum(int(state["decoded"]) for state in logical.values())
    decode_intervals = [
        (
            number(row["finished_unix"])
            - number(row["timings"].get("predicted_ms")) / 1000,
            number(row["finished_unix"]),
        )
        for row in all_rows
    ]
    decode_union_s = interval_union_seconds(decode_intervals)
    decode_window_s = max(end for _, end in decode_intervals) - min(
        start for start, _ in decode_intervals
    )
    decoded_by_session = {name: int(state["decoded"]) for name, state in logical.items()}
    final_prompt_tokens = {
        name: len(tokenize(args.url, str(state["prompt"]))) for name, state in logical.items()
    }
    service_output_tps = total_predicted_n / total_wall_s if total_wall_s else None
    decode_union_tps = total_predicted_n / decode_union_s if decode_union_s else None
    decode_window_tps = total_predicted_n / decode_window_s if decode_window_s else None
    gates = {
        "each_session_output_complete": all(
            decoded == args.decode_tokens for decoded in decoded_by_session.values()
        ),
        "prompt_reuse_valid": all_reuse_valid,
        "service_output_target_met": bool(
            service_output_tps is not None and service_output_tps >= args.throughput_target
        ),
        "decode_union_target_met": bool(
            decode_union_tps is not None and decode_union_tps >= args.throughput_target
        ),
        "decode_window_target_met": bool(
            decode_window_tps is not None and decode_window_tps >= args.throughput_target
        ),
    }
    summary = {
        "completed": True,
        "workload": "round-robin-decode",
        "context_tokens": args.context_tokens,
        "calibrated_corpus_tokens": calibrated_tokens,
        "physical_slots": args.physical_slots,
        "sessions": list(logical),
        "schedule": [
            [{"session": name, "id_slot": id_slot} for name, id_slot in placements]
            for placements in schedule
        ],
        "chunk_tokens": chunk_tokens,
        "decode_tokens_per_session": args.decode_tokens,
        "speculative_n_max": args.speculative_n_max,
        "throughput_target": args.throughput_target,
        "wall_s": total_wall_s,
        "started_unix": benchmark_started_unix,
        "total_prompt_n": total_prompt_n,
        "total_predicted_n": total_predicted_n,
        "decoded_by_session": decoded_by_session,
        "service_prompt_tps": total_prompt_n / total_wall_s if total_wall_s else None,
        "service_output_tps": service_output_tps,
        "service_useful_tps": (
            (total_prompt_n + total_predicted_n) / total_wall_s if total_wall_s else None
        ),
        "service_decode_union_s": decode_union_s,
        "service_decode_union_tps": decode_union_tps,
        "service_decode_window_s": decode_window_s,
        "service_decode_window_tps": decode_window_tps,
        "final_prompt_tokens_client": final_prompt_tokens,
        "initial_restore_validation": initial_status,
        "gates": gates,
        "stage_summaries": stage_summaries,
    }
    (args.result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    phase("complete", gates=gates)
    emit("driver_complete", summary_path=str((args.result_dir / "summary.json").resolve()), gates=gates)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
