#!/usr/bin/env python3
"""Regression tests for the measured hybrid hot-window selector."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "scripts" / "derive-hybrid-capacity.py"

COMPATIBLE_MANIFEST = {
    "server_binary_sha256": "server-build",
    "model_sha256": "model",
    "workload": "growth",
    "parallel": "2",
    "sessions": "2",
    "ctx_size": "262144",
    "draft_ctx": "131072",
    "batch_size": "512",
    "ubatch_size": "256",
    "cache_type_k": "bf16",
    "cache_type_v": "bf16",
    "cache_type_k_draft": "q8_0",
    "cache_type_v_draft": "q8_0",
    "mtp_max": "10",
    "mtp_adaptive": "1",
    "milestones": "128000",
    "decode_tokens": "2048",
    "final_decode_tokens": "2048",
    "round_robin_context": "200000",
    "round_robin_decode_tokens": "2048",
}


def write_profile(
    path: Path,
    *,
    hot_tokens: int,
    throughput: float,
    peak_used_mib: int,
    total_mib: int = 24125,
    parent: Path | None = None,
) -> None:
    path.mkdir()
    manifest = {
        **COMPATIBLE_MANIFEST,
        "hot_tokens": str(hot_tokens),
        "capacity_profile": str(parent.resolve()) if parent else "",
        "exit_status": "0",
    }
    (path / "manifest.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in manifest.items()), encoding="utf-8"
    )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "completed": True,
                "service_decode_union_tps": throughput,
                "service_output_tps": throughput - 0.25,
            }
        ),
        encoding="utf-8",
    )
    (path / "observability-summary.json").write_text(
        json.dumps({"server_signals": {"cuda_oom_count": 0}}), encoding="utf-8"
    )
    (path / "gpu.csv").write_text(
        "memory_used_mib,memory_free_mib\n"
        f"{peak_used_mib},{total_mib - peak_used_mib}\n",
        encoding="utf-8",
    )
    hot_mib = hot_tokens * 65536 / 1024 / 1024
    (path / "server.log").write_text(
        f"rebuilt {hot_tokens} hybrid GPU hot-ring rows ({hot_mib:.3f} MiB, test)\n",
        encoding="utf-8",
    )


def select(profile: Path, target: float | None = None, max_tokens: int = 262144) -> dict:
    command = [
        "python3",
        str(SELECTOR),
        str(profile),
        "--ubatch",
        "256",
        "--block",
        "256",
        "--headroom-fraction",
        "0.0625",
        "--max-tokens",
        str(max_tokens),
        "--json",
    ]
    if target is not None:
        command.extend(("--throughput-target", str(target)))
    return json.loads(subprocess.check_output(command, text=True))


class HybridCapacitySelectorTest(unittest.TestCase):
    def test_feedback_rejects_larger_but_slower_hot_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fast = root / "fast"
            slow = root / "slow"
            write_profile(
                fast,
                hot_tokens=22528,
                throughput=48.89,
                peak_used_mib=22036,
            )
            write_profile(
                slow,
                hot_tokens=31744,
                throughput=39.14,
                peak_used_mib=22666,
                parent=fast,
            )

            plan = select(slow, target=45)

            self.assertEqual(plan["hot_tokens"], 22528)
            self.assertEqual(plan["selection_reason"], "best_observed_target_met")
            self.assertTrue(plan["throughput_target_met"])
            self.assertEqual(len(plan["observations"]), 2)
            self.assertLess(plan["capacity_hot_tokens"], 31744)

    def test_unmet_target_proposes_bounded_aligned_exploration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "initial"
            write_profile(
                profile,
                hot_tokens=4096,
                throughput=18.0,
                peak_used_mib=18000,
                total_mib=24125,
            )

            plan = select(profile, target=45, max_tokens=32768)

            self.assertEqual(plan["selection_reason"], "bounded_throughput_exploration")
            self.assertFalse(plan["throughput_target_met"])
            self.assertEqual(plan["hot_tokens"], plan["capacity_hot_tokens"])
            self.assertEqual(plan["hot_tokens"] % 256, 0)
            self.assertLessEqual(plan["hot_tokens"], 32768)


if __name__ == "__main__":
    unittest.main()
