# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Sensitivity report printer for prediction trie JSON files.

Usage:
    python -m latency_sensitivity_demo.sensitivity_report <prediction_trie.json>

Walks the trie recursively and prints a human-readable table showing each
node's inferred latency sensitivity along with the underlying metrics.
"""

import sys
from pathlib import Path

from nat.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.profiler.prediction_trie.serialization import load_prediction_trie

# ANSI color codes
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"

_SENSITIVITY_LABELS = {
    1: ("LOW", _GREEN),
    2: ("LOW-MED", _GREEN),
    3: ("MEDIUM", _YELLOW),
    4: ("MED-HIGH", _RED),
    5: ("HIGH", _RED),
}


def _sensitivity_str(score: int | None) -> str:
    """Return a colored sensitivity string."""
    if score is None:
        return "N/A"
    label, color = _SENSITIVITY_LABELS.get(score, ("?", _RESET))
    return f"{color}{score}/5 ({label}){_RESET}"


def _print_node(node: PredictionTrieNode, path: str, rows: list[dict]) -> None:
    """Recursively collect rows from the trie."""
    for call_idx, pred in sorted(node.predictions_by_call_index.items()):
        rows.append({
            "path": path,
            "call_index": call_idx,
            "remaining_calls_mean": pred.remaining_calls.mean,
            "interarrival_ms_mean": pred.interarrival_ms.mean,
            "output_tokens_mean": pred.output_tokens.mean,
            "sensitivity": pred.latency_sensitivity,
        })

    if node.predictions_any_index and not node.predictions_by_call_index:
        pred = node.predictions_any_index
        rows.append({
            "path": path,
            "call_index": "any",
            "remaining_calls_mean": pred.remaining_calls.mean,
            "interarrival_ms_mean": pred.interarrival_ms.mean,
            "output_tokens_mean": pred.output_tokens.mean,
            "sensitivity": pred.latency_sensitivity,
        })

    for child_name, child_node in sorted(node.children.items()):
        _print_node(child_node, f"{path}/{child_name}", rows)


def print_report(trie_root: PredictionTrieNode) -> None:
    """Print the sensitivity report to stdout."""
    rows: list[dict] = []
    _print_node(trie_root, trie_root.name, rows)

    if not rows:
        print("No prediction data found in the trie.")
        return

    # Header
    print()
    print("=" * 100)
    print("LATENCY SENSITIVITY REPORT")
    print("=" * 100)
    print()

    # Column headers
    header = (f"{'Path':<45} {'Call#':<6} {'Remaining':<10} {'IAT (ms)':<10} "
              f"{'Tokens':<8} {'Sensitivity':<20}")
    print(header)
    print("-" * 100)

    # Data rows
    for row in rows:
        call_idx_str = str(row["call_index"])
        sens_str = _sensitivity_str(row["sensitivity"])
        print(f"{row['path']:<45} {call_idx_str:<6} {row['remaining_calls_mean']:<10.1f} "
              f"{row['interarrival_ms_mean']:<10.1f} {row['output_tokens_mean']:<8.1f} {sens_str}")

    print()

    # Summary
    print("=" * 100)
    print("ROUTING RECOMMENDATIONS")
    print("=" * 100)
    print()
    print(f"  {_RED}HIGH (4-5){_RESET}   : Route to dedicated/priority workers for lowest latency")
    print(f"  {_YELLOW}MEDIUM (3){_RESET}  : Standard routing — balance between latency and throughput")
    print(f"  {_GREEN}LOW (1-2){_RESET}    : Route to shared/batch workers — throughput over latency")
    print()


def main() -> None:
    """Entry point for the sensitivity report CLI."""
    if len(sys.argv) != 2:
        print("Usage: python -m latency_sensitivity_demo.sensitivity_report <prediction_trie.json>")
        sys.exit(1)

    trie_path = Path(sys.argv[1])
    if not trie_path.exists():
        print(f"Error: File not found: {trie_path}")
        sys.exit(1)

    trie_root = load_prediction_trie(trie_path)
    print_report(trie_root)


if __name__ == "__main__":
    main()
