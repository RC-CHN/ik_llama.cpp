#!/usr/bin/env python3
"""Sweep speculative depth and request concurrency on resident long contexts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
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


def tokenize_count(url: str, text: str) -> int:
    response = request_json(url, "/tokenize", {"content": text, "add_special": False}, 600)
    return len(response["tokens"])


def calibrate_cut(url: str, corpus: str, target: int) -> tuple[int, int]:
    full_tokens = tokenize_count(url, corpus)
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
        count = tokenize_count(url, corpus[:guess])
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
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def completion_body(prompt: str, n_predict: int, seed: int, slot: int, depth: int) -> dict:
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": True,
        "stream": False,
        "temperature": 0,
        "seed": seed,
        "ignore_eos": True,
        "saturate_predict": True,
        "id_slot": slot,
        "speculative.n_max": depth,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=40000)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--depths", default="0,2,3,4,5,6,8")
    parser.add_argument("--modes", default="concurrent,serial")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument(
        "--concurrency-levels",
        default="",
        help="comma-separated active-session counts; default uses all resident sessions",
    )
    args = parser.parse_args()

    depths = [int(value) for value in args.depths.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if not depths or min(depths) < 0 or len(depths) != len(set(depths)):
        raise ValueError("--depths must contain unique non-negative integers")
    if not modes or any(mode not in {"concurrent", "serial"} for mode in modes):
        raise ValueError("--modes must contain concurrent and/or serial")
    if args.sessions <= 0 or args.sessions > 26:
        raise ValueError("--sessions must be between 1 and 26")
    concurrency_levels = (
        [int(value) for value in args.concurrency_levels.split(",") if value.strip()]
        if args.concurrency_levels
        else [args.sessions]
    )
    if (
        not concurrency_levels
        or min(concurrency_levels) <= 0
        or max(concurrency_levels) > args.sessions
        or len(concurrency_levels) != len(set(concurrency_levels))
    ):
        raise ValueError("--concurrency-levels must contain unique counts from 1 through --sessions")
    if args.context_tokens <= 0 or args.decode_tokens <= 0 or args.warmup_tokens < 0:
        raise ValueError("context/decode tokens must be positive and warmup tokens non-negative")

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

    phase("calibrating_token_cut", context_tokens=args.context_tokens)
    corpus = args.corpus.read_text(encoding="utf-8", errors="replace")
    cut_pos, calibrated_tokens = calibrate_cut(args.url, corpus, args.context_tokens)
    session_names = [chr(ord("A") + index) for index in range(args.sessions)]
    prompts = {
        name: (
            f"// RESIDENT_DECODE_SWEEP_SESSION_{name}\n"
            + corpus[:cut_pos]
            + f"\n// session {name}: resident decode sweep at {args.context_tokens}; continue the code.\n"
        )
        for name in session_names
    }
    prompt_tokens = {name: tokenize_count(args.url, prompt) for name, prompt in prompts.items()}
    emit(
        "calibration_complete",
        context_tokens=args.context_tokens,
        calibrated_corpus_tokens=calibrated_tokens,
        cut_chars=cut_pos,
        prompt_tokens=prompt_tokens,
        corpus_sha256=hashlib.sha256(corpus.encode()).hexdigest(),
    )

    def invoke(name: str, depth: int, n_predict: int, sweep_index: int, kind: str) -> dict:
        slot = session_names.index(name)
        started_unix = time.time()
        started = time.perf_counter()
        emit(
            "request_start",
            kind=kind,
            sweep_index=sweep_index,
            session=name,
            id_slot=slot,
            speculative_n_max=depth,
            n_predict=n_predict,
        )
        response = request_json(
            args.url,
            "/completion",
            completion_body(prompts[name], n_predict, 101 + slot, slot, depth),
        )
        finished_unix = time.time()
        timings = response.get("timings") or {}
        row = {
            "session": name,
            "id_slot": response.get("id_slot"),
            "started_unix": started_unix,
            "finished_unix": finished_unix,
            "wall_s": time.perf_counter() - started,
            "prompt_n": number(timings.get("prompt_n")),
            "prompt_ms": number(timings.get("prompt_ms")),
            "predicted_n": number(timings.get("predicted_n")),
            "predicted_ms": number(timings.get("predicted_ms")),
            "predicted_per_second": number(timings.get("predicted_per_second")),
            "draft_n": number(timings.get("draft_n")),
            "draft_n_accepted": number(timings.get("draft_n_accepted")),
            "draft_by_depth": timings.get("draft_by_depth") or [],
            "content_sha256": hashlib.sha256(response.get("content", "").encode()).hexdigest(),
            "tokens_cached": response.get("tokens_cached"),
        }
        if int(row["id_slot"]) != slot:
            raise RuntimeError(f"session {name} moved from slot {slot} to {row['id_slot']}")
        emit("request_complete", kind=kind, sweep_index=sweep_index, **row)
        return row

    def run_group(
        mode: str,
        depth: int,
        n_predict: int,
        sweep_index: int,
        kind: str,
        active_names: list[str] | None = None,
    ) -> tuple[list[dict], float]:
        names = session_names if active_names is None else active_names
        group_started = time.perf_counter()
        if mode == "serial":
            rows = [invoke(name, depth, n_predict, sweep_index, kind) for name in names]
        else:
            gate = threading.Barrier(len(names) + 1)

            def gated_invoke(name: str) -> dict:
                gate.wait()
                return invoke(name, depth, n_predict, sweep_index, kind)

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
                futures = {name: pool.submit(gated_invoke, name) for name in names}
                gate.wait()
                rows = [futures[name].result() for name in names]
        return rows, time.perf_counter() - group_started

    phase("resident_context_load", context_tokens=args.context_tokens)
    emit("resident_load_start", context_tokens=args.context_tokens)
    load_rows, load_wall_s = run_group("concurrent", 0, 1, -1, "resident_load")
    emit("resident_load_complete", wall_s=load_wall_s, requests=load_rows)

    sweep_summaries: list[dict] = []
    sweep_index = 0
    for depth in depths:
        for concurrency_level in concurrency_levels:
            active_names = session_names[:concurrency_level]
            for mode in modes:
                if args.warmup_tokens:
                    phase(
                        "decode_warmup",
                        sweep_index=sweep_index,
                        mode=mode,
                        concurrency_level=concurrency_level,
                        speculative_n_max=depth,
                    )
                    run_group(
                        mode,
                        depth,
                        args.warmup_tokens,
                        sweep_index,
                        "warmup",
                        active_names,
                    )

                phase(
                    "decode_measure",
                    sweep_index=sweep_index,
                    mode=mode,
                    concurrency_level=concurrency_level,
                    speculative_n_max=depth,
                )
                emit(
                    "sweep_start",
                    sweep_index=sweep_index,
                    mode=mode,
                    concurrency_level=concurrency_level,
                    speculative_n_max=depth,
                    n_predict=args.decode_tokens,
                )
                rows, wall_s = run_group(
                    mode,
                    depth,
                    args.decode_tokens,
                    sweep_index,
                    "measure",
                    active_names,
                )
                decode_intervals = [
                    (row["finished_unix"] - row["predicted_ms"] / 1000, row["finished_unix"])
                    for row in rows
                    if row["predicted_ms"] > 0
                ]
                predicted_n = sum(row["predicted_n"] for row in rows)
                decode_union_s = interval_union_seconds(decode_intervals)
                decode_span_s = (
                    max(end for _, end in decode_intervals) - min(start for start, _ in decode_intervals)
                    if decode_intervals
                    else 0.0
                )
                service_span_s = (
                    max(row["finished_unix"] for row in rows) - min(row["started_unix"] for row in rows)
                    if rows
                    else wall_s
                )
                summary = {
                    "sweep_index": sweep_index,
                    "mode": mode,
                    "concurrency_level": concurrency_level,
                    "speculative_n_max": depth,
                    "wall_s": wall_s,
                    "service_span_s": service_span_s,
                    "prompt_n": sum(row["prompt_n"] for row in rows),
                    "predicted_n": predicted_n,
                    "decode_union_s": decode_union_s,
                    "decode_span_s": decode_span_s,
                    "service_output_tps": predicted_n / service_span_s if service_span_s else None,
                    "decode_union_tps": predicted_n / decode_union_s if decode_union_s else None,
                    "decode_span_tps": predicted_n / decode_span_s if decode_span_s else None,
                    "request_decode_tps": [row["predicted_per_second"] for row in rows],
                    "draft_n": sum(row["draft_n"] for row in rows),
                    "draft_n_accepted": sum(row["draft_n_accepted"] for row in rows),
                    "requests": rows,
                }
                sweep_summaries.append(summary)
                emit("sweep_complete", **summary)
                sweep_index += 1

    summary = {
        "completed": True,
        "context_tokens": args.context_tokens,
        "calibrated_corpus_tokens": calibrated_tokens,
        "sessions": session_names,
        "concurrency_levels": concurrency_levels,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "warmup_tokens": args.warmup_tokens,
        "depths": depths,
        "modes": modes,
        "resident_load_wall_s": load_wall_s,
        "resident_load_requests": load_rows,
        "sweep_summaries": sweep_summaries,
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
