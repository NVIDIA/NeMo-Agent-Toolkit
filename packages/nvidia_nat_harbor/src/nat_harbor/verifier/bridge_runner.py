# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLI runner for ATIF evaluator bridge in Harbor verifier scripts."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from nat_harbor.verifier.evaluator_adapter import evaluate_artifact_sync


def _resolve_artifact_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path("/logs/agent") / path


def _write_reward(output_dir: Path, reward: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_json = output_dir / "reward.json"
    reward_txt = output_dir / "reward.txt"
    payload = {"reward": float(reward)}
    reward_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    reward_txt.write_text(f"{float(reward)}\n", encoding="utf-8")


def _write_details(output_dir: Path, details: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")


def _raw_output_details() -> dict[str, Any]:
    raw_path = Path("/logs/agent/nemo-agent-output.txt")
    raw_text = raw_path.read_text(encoding="utf-8").strip() if raw_path.exists() else ""
    return {
        "raw_output_path": str(raw_path),
        "raw_output_exists": raw_path.exists(),
        "raw_output_prefix": raw_text[:120] if raw_text else "",
    }


def run_bridge(
    *,
    artifact_path: str,
    evaluator_kind: str,
    evaluator_ref: str | None,
    output_dir: str,
    fallback_mode: str,
    config_file: str | None,
    evaluator_name: str | None,
) -> int:
    """Run bridge evaluation and emit Harbor verifier artifacts."""
    artifact = _resolve_artifact_path(artifact_path)
    out_dir = Path(output_dir)
    details: dict[str, Any] = {
        "artifact_path": str(artifact),
        "artifact_exists": artifact.exists(),
        "evaluator_kind": evaluator_kind,
        "evaluator_ref": evaluator_ref,
        "fallback_mode": fallback_mode,
    }
    if not artifact.exists():
        if fallback_mode == "raw_output":
            details["result"] = "raw_fallback_missing_artifact"
            details.update(_raw_output_details())
            _write_reward(out_dir, 0.0)
            _write_details(out_dir, details)
            return 0
        details["result"] = "missing_artifact_fail"
        _write_details(out_dir, details)
        return 1

    try:
        reward, evaluator_details = evaluate_artifact_sync(
            artifact_path=artifact,
            evaluator_kind=evaluator_kind,
            evaluator_ref=evaluator_ref,
            config_file=config_file,
            evaluator_name=evaluator_name,
        )
    except Exception as exc:
        details["error"] = str(exc)
        details["error_type"] = type(exc).__name__
        details["traceback"] = traceback.format_exc()
        if fallback_mode == "raw_output":
            details["result"] = "raw_fallback_evaluator_error"
            details.update(_raw_output_details())
            _write_reward(out_dir, 0.0)
            _write_details(out_dir, details)
            return 0
        details["result"] = "evaluator_error_fail"
        _write_details(out_dir, details)
        return 1

    details["result"] = "evaluated"
    details["reward"] = float(reward)
    details["evaluator_details"] = evaluator_details
    _write_reward(out_dir, reward)
    _write_details(out_dir, details)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build bridge runner command line parser."""
    parser = argparse.ArgumentParser(description="Run NAT ATIF evaluator bridge from Harbor verifier script.")
    parser.add_argument("--artifact-path", default="trajectory.json", help="ATIF artifact JSON path.")
    parser.add_argument(
        "--evaluator-kind",
        required=True,
        help=("Evaluator dispatch kind. Use `custom` with --evaluator-ref, "
              "or provide a builtin evaluator name (for example `trajectory`, `tunable_rag`, `ragas`)."),
    )
    parser.add_argument("--evaluator-ref", default=None, help="Custom evaluator ref in module:function format.")
    parser.add_argument("--output-dir", default="/logs/verifier", help="Verifier output directory.")
    parser.add_argument(
        "--fallback-mode",
        default="fail",
        choices=["fail", "raw_output"],
        help="Fallback behavior when artifacts/evaluation fail.",
    )
    parser.add_argument("--config-file", default=None, help="NAT config path for builtin evaluators.")
    parser.add_argument("--evaluator-name", default=None, help="Evaluator name inside --config-file.")
    return parser


def main() -> int:
    """CLI entrypoint for bridge runner."""
    args = build_parser().parse_args()
    return run_bridge(
        artifact_path=args.artifact_path,
        evaluator_kind=args.evaluator_kind,
        evaluator_ref=args.evaluator_ref,
        output_dir=args.output_dir,
        fallback_mode=args.fallback_mode,
        config_file=args.config_file,
        evaluator_name=args.evaluator_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
