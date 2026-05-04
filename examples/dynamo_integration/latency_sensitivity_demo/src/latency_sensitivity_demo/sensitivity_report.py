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
    python -m latency_sensitivity_demo.sensitivity_report <prediction_trie.json> [--csv <profiler.csv>]

Walks the trie recursively and prints a human-readable table showing each
node's inferred latency sensitivity along with the underlying metrics.

When a profiler CSV is provided (``standardized_data_all.csv``), the report
also shows measured p50/p90/mean latency and tokens-per-second for each
function node.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

from nat.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.profiler.prediction_trie.serialization import load_prediction_trie

# ANSI color codes
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_SENSITIVITY_LABELS = {
    1: ("LOW", _GREEN),
    2: ("LOW-MED", _GREEN),
    3: ("MEDIUM", _YELLOW),
    4: ("MED-HIGH", _RED),
    5: ("HIGH", _RED),
}


def _sensitivity_str(score: int | None, width: int = 16) -> str:
    """Return a colored sensitivity string padded to *width* visible chars."""
    if score is None:
        return "N/A".ljust(width)
    label, color = _SENSITIVITY_LABELS.get(score, ("?", _RESET))
    visible = f"{score}/5 ({label})"
    # Pad to `width` visible characters, then wrap with ANSI codes so
    # the terminal alignment is correct despite invisible escape bytes.
    return f"{color}{visible.ljust(width)}{_RESET}"


def _percentile(data: list[float], pct: int) -> float:
    """Compute a percentile value from a sorted-on-the-fly list."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = (pct / 100) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms"


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_latency_from_csv(csv_path: Path) -> dict[str, list[dict]]:
    """Parse a profiler CSV and return per-function latency records.

    Returns:
        ``{function_name: [{"duration_s": ..., "completion_tokens": ..., "tps": ...}, ...]}``
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    starts: dict[str, dict] = {}
    ends: dict[str, dict] = {}
    for row in rows:
        et = row.get("event_type", "")
        uid = row.get("UUID", "")
        if not uid:
            continue
        if et == "LLM_START":
            starts[uid] = row
        elif et == "LLM_END":
            ends[uid] = row

    by_fn: dict[str, list[dict]] = {}
    for uid, s in starts.items():
        e = ends.get(uid)
        if not e:
            continue
        dur = float(e["event_timestamp"]) - float(s["event_timestamp"])
        comp = int(e.get("completion_tokens") or 0)
        fn = s.get("function_name", "")
        by_fn.setdefault(fn, []).append({
            "duration_s": dur,
            "completion_tokens": comp,
            "tps": comp / dur if dur > 0 else 0.0,
        })

    return by_fn


# ---------------------------------------------------------------------------
# Trie collection
# ---------------------------------------------------------------------------


def _collect_rows(node: PredictionTrieNode, path: str, rows: list[dict]) -> None:
    """Recursively collect rows from the trie."""
    # Extract the leaf function name (last path segment) for CSV joining
    segments = path.split("/")
    leaf_name = segments[-1] if segments else ""

    for call_idx, pred in sorted(node.predictions_by_call_index.items()):
        rows.append({
            "path": path,
            "leaf_name": leaf_name,
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
            "leaf_name": leaf_name,
            "call_index": "any",
            "remaining_calls_mean": pred.remaining_calls.mean,
            "interarrival_ms_mean": pred.interarrival_ms.mean,
            "output_tokens_mean": pred.output_tokens.mean,
            "sensitivity": pred.latency_sensitivity,
        })

    for child_name, child_node in sorted(node.children.items()):
        _collect_rows(child_node, f"{path}/{child_name}", rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(
    trie_root: PredictionTrieNode,
    latency_by_fn: dict[str, list[dict]] | None = None,
) -> None:
    """Print the sensitivity report to stdout.

    Args:
        trie_root: The root of the prediction trie.
        latency_by_fn: Optional per-function latency records from
            :func:`parse_latency_from_csv`.  When provided the table includes
            measured p50/p90/mean latency and tokens-per-second columns.
    """
    rows: list[dict] = []
    _collect_rows(trie_root, trie_root.name, rows)

    if not rows:
        print("No prediction data found in the trie.")
        return

    show_latency = latency_by_fn is not None and len(latency_by_fn) > 0

    # Compute the path column width from the data (minimum 20, +2 padding)
    path_w = max(20, max(len(row["path"]) for row in rows) + 2)
    sens_w = 16  # visible width of sensitivity column (e.g. "5/5 (MED-HIGH)")

    # Build the table width
    #   path  call#(5)  remaining(10)  iat(10)  tokens(8)  sensitivity(sens_w)
    base_w = path_w + 5 + 10 + 10 + 8 + sens_w + 5 * 2  # 5 inter-column gaps of 2
    if show_latency:
        #   p50(9) p90(9) mean(9) tps(7)
        base_w += 9 + 9 + 9 + 7
    table_w = base_w

    # Header
    print()
    print("=" * table_w)
    print("LATENCY SENSITIVITY REPORT")
    print("=" * table_w)
    print()

    # Column headers
    hdr = (f"{'Path':<{path_w}}  {'Call#':>5}  {'Remaining':>10}  {'IAT (ms)':>10}  {'Tokens':>8}"
           f"  {'Sensitivity':<{sens_w}}")
    if show_latency:
        hdr += f"  {'p50':>7}  {'p90':>7}  {'Mean':>7}  {'TPS':>5}"
    print(hdr)
    print("-" * table_w)

    # Data rows
    for row in rows:
        call_idx_str = str(row["call_index"])
        sens_str = _sensitivity_str(row["sensitivity"], width=sens_w)
        line = (f"{row['path']:<{path_w}}  {call_idx_str:>5}  {row['remaining_calls_mean']:>10.1f}"
                f"  {row['interarrival_ms_mean']:>10.1f}  {row['output_tokens_mean']:>8.1f}  {sens_str}")

        if show_latency:
            fn_records = latency_by_fn.get(row["leaf_name"], [])
            if fn_records:
                durations = [r["duration_s"] for r in fn_records]
                tps_values = [r["tps"] for r in fn_records]
                line += (f"  {_fmt_ms(statistics.median(durations)):>7}"
                         f"  {_fmt_ms(_percentile(durations, 90)):>7}"
                         f"  {_fmt_ms(statistics.mean(durations)):>7}"
                         f"  {statistics.mean(tps_values):>5.1f}")
            else:
                line += f"  {'—':>7}  {'—':>7}  {'—':>7}  {'—':>5}"

        print(line)

    print()

    # Summary
    print("=" * table_w)
    print("ROUTING RECOMMENDATIONS")
    print("=" * table_w)
    print()
    print(f"  {_RED}HIGH (4-5){_RESET}   : Route to dedicated/priority workers for lowest latency")
    print(f"  {_YELLOW}MEDIUM (3){_RESET}  : Standard routing — balance between latency and throughput")
    print(f"  {_GREEN}LOW (1-2){_RESET}    : Route to shared/batch workers — throughput over latency")
    print()


def main() -> None:
    """Entry point for the sensitivity report CLI."""
    parser = argparse.ArgumentParser(
        description="Print a latency sensitivity report from a prediction trie.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trie", type=Path, help="Path to prediction_trie.json")
    parser.add_argument("--csv",
                        type=Path,
                        default=None,
                        help="Path to standardized_data_all.csv for measured latency columns")

    args = parser.parse_args()

    if not args.trie.exists():
        print(f"Error: File not found: {args.trie}", file=sys.stderr)
        sys.exit(1)

    latency_by_fn = None
    if args.csv is not None:
        if not args.csv.exists():
            print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
            sys.exit(1)
        latency_by_fn = parse_latency_from_csv(args.csv)

    trie_root = load_prediction_trie(args.trie)
    print_report(trie_root, latency_by_fn)


if __name__ == "__main__":
    main()
