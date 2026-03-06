#!/usr/bin/env python3
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
"""Compare two eval run output directories.

This script compares evaluator outputs from two run directories.
By default it prioritizes common files (RAGAS, trajectory, and tunable RAG),
and it also auto-discovers any additional ``*_output.json`` evaluator files.

It prints:
- average score delta per evaluator
- per-item score change count
- optional per-item score diffs (with --show-item-diffs)

Example:
    python3 packages/nvidia_nat_eval/scripts/compare_eval_runs.py \
      .tmp/nat/examples/evaluation_and_profiling/simple_web_query_eval/atif/llama-33-70b \
      .tmp/nat/examples/evaluation_and_profiling/simple_web_query_eval/llama-33-70b \
      --show-item-diffs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EVALUATOR_FILES = (
    "accuracy_output.json",
    "groundedness_output.json",
    "relevance_output.json",
    "trajectory_accuracy_output.json",
    "tuneable_eval_output.json",
    "tunable_eval_output.json",
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _score_delta(a: object, b: object) -> float | None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) - float(b)
    return None


def _fmt_score(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _discover_evaluator_files(run_a: Path, run_b: Path) -> list[str]:
    """Discover evaluator output files from both run directories.

    Includes all ``*_output.json`` files except workflow outputs.
    Preferred known evaluator files are listed first for stable output.
    """
    excluded = {"workflow_output.json", "workflow_output_atif.json"}
    discovered = set()
    for run_dir in (run_a, run_b):
        if not run_dir.exists():
            continue
        for path in run_dir.glob("*_output.json"):
            if path.name not in excluded:
                discovered.add(path.name)

    ordered: list[str] = []
    for name in EVALUATOR_FILES:
        if name in discovered:
            ordered.append(name)

    for name in sorted(discovered):
        if name not in ordered:
            ordered.append(name)

    return ordered


def compare_evaluator(run_a: Path, run_b: Path, file_name: str, show_item_diffs: bool) -> None:
    """Compare a single evaluator output file across two runs.

    Args:
        run_a: Path to the first run output directory.
        run_b: Path to the second run output directory.
        file_name: Evaluator output JSON file name to compare.
        show_item_diffs: Whether to print per-item score differences.

    Returns:
        None.
    """
    path_a = run_a / file_name
    path_b = run_b / file_name

    if not path_a.exists() or not path_b.exists():
        print(f"- {file_name}: missing in one/both runs")
        return

    try:
        data_a = _read_json(path_a)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"- {file_name}: unreadable in run_a ({path_a}): {e}")
        return

    try:
        data_b = _read_json(path_b)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"- {file_name}: unreadable in run_b ({path_b}): {e}")
        return

    avg_a = data_a.get("average_score")
    avg_b = data_b.get("average_score")
    delta = _score_delta(avg_a, avg_b)

    items_a = {}
    skipped_a = 0
    for item in data_a.get("eval_output_items", []):
        if not isinstance(item, dict):
            skipped_a += 1
            continue
        item_id = item.get("id")
        if item_id is None:
            skipped_a += 1
            continue
        items_a[str(item_id)] = item

    items_b = {}
    skipped_b = 0
    for item in data_b.get("eval_output_items", []):
        if not isinstance(item, dict):
            skipped_b += 1
            continue
        item_id = item.get("id")
        if item_id is None:
            skipped_b += 1
            continue
        items_b[str(item_id)] = item

    all_ids = sorted(set(items_a) | set(items_b), key=lambda x: (len(x), x))

    changed_ids: list[str] = []
    for item_id in all_ids:
        score_a = items_a.get(item_id, {}).get("score")
        score_b = items_b.get(item_id, {}).get("score")
        if score_a != score_b:
            changed_ids.append(item_id)

    print(f"\n{file_name}")
    print(f"  avg_score run_a={_fmt_score(avg_a)} run_b={_fmt_score(avg_b)}", end="")
    if delta is not None:
        print(f" delta={delta:+.6f}")
    else:
        print(" delta=N/A")
    print(f"  item_count run_a={len(items_a)} run_b={len(items_b)} changed_items={len(changed_ids)}")
    if skipped_a or skipped_b:
        print(f"  skipped_items run_a={skipped_a} run_b={skipped_b}")

    if show_item_diffs and changed_ids:
        for item_id in changed_ids:
            score_a = items_a.get(item_id, {}).get("score")
            score_b = items_b.get(item_id, {}).get("score")
            print(f"    id={item_id} run_a={_fmt_score(score_a)} run_b={_fmt_score(score_b)}")


def main() -> int:
    """Run the CLI to compare evaluator outputs from two run directories.

    Parses positional run directory arguments and an optional per-item diff flag,
    then compares all discovered evaluator output files.

    Returns:
        Process exit code. Returns 0 for normal CLI completion.
    """
    parser = argparse.ArgumentParser(description="Compare evaluator outputs between two eval runs.")
    parser.add_argument("run_a", type=Path, help="Path to first run output directory")
    parser.add_argument("run_b", type=Path, help="Path to second run output directory")
    parser.add_argument("--show-item-diffs", action="store_true", help="Print per-item score deltas for changed items")
    args = parser.parse_args()

    print(f"Run A: {args.run_a}")
    print(f"Run B: {args.run_b}")

    evaluator_files = _discover_evaluator_files(args.run_a, args.run_b)
    if not evaluator_files:
        print("\nNo evaluator output files found in either run directory.")
        return 0

    for file_name in evaluator_files:
        compare_evaluator(args.run_a, args.run_b, file_name, args.show_item_diffs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
