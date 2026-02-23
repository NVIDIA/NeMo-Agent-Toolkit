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
Compare LLM call performance grouped by latency sensitivity.

Usage:
    python -m latency_sensitivity_demo.compare_sensitivity_perf \\
        --trie <prediction_trie.json> \\
        --csv <standardized_data_all.csv> [--csv <another.csv> ...]

Reads per-LLM-call timing data from one or more profiler CSVs, joins each call
with its sensitivity score from the prediction trie, and prints a comparison
showing whether HIGH-priority calls achieved lower latency than LOW-priority
calls.

When multiple CSVs are provided (e.g. a baseline NIM run and a Dynamo run with
sensitivity hints), the report prints side-by-side columns so you can see the
improvement.
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
_DIM = "\033[2m"
_RESET = "\033[0m"

_SENSITIVITY_LABELS = {
    1: ("LOW", _GREEN),
    2: ("LOW-MED", _GREEN),
    3: ("MEDIUM", _YELLOW),
    4: ("MED-HIGH", _RED),
    5: ("HIGH", _RED),
}

_PRIORITY_GROUPS = {
    "HIGH (4-5)": lambda s: s >= 4,
    "MEDIUM (3)": lambda s: s == 3,
    "LOW (1-2)": lambda s: s <= 2,
}

# ---------------------------------------------------------------------------
# Trie helpers
# ---------------------------------------------------------------------------


def _collect_sensitivity_map(node: PredictionTrieNode, path: str = "") -> dict[str, int]:
    """Walk the trie and return {function_name: sensitivity} for leaf nodes."""
    result: dict[str, int] = {}

    for call_idx, pred in node.predictions_by_call_index.items():
        if pred.latency_sensitivity is not None and node.name not in ("root", "<workflow>"):
            result[node.name] = pred.latency_sensitivity

    for child_name, child_node in node.children.items():
        result.update(_collect_sensitivity_map(child_node, f"{path}/{child_name}"))

    return result


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def _parse_csv(csv_path: Path) -> list[dict]:
    """Parse a profiler CSV and return per-LLM-call records with duration.

    Each record contains:
        function_name, example_number, duration_s, completion_tokens,
        prompt_tokens, total_tokens, ttft_s (if derivable)
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Index START and END events by UUID
    starts: dict[str, dict] = {}
    ends: dict[str, dict] = {}
    for row in rows:
        event_type = row.get("event_type", "")
        uuid = row.get("UUID", "")
        if not uuid:
            continue
        if event_type == "LLM_START":
            starts[uuid] = row
        elif event_type == "LLM_END":
            ends[uuid] = row

    calls: list[dict] = []
    for uuid, start_row in starts.items():
        end_row = ends.get(uuid)
        if not end_row:
            continue

        start_ts = float(start_row["event_timestamp"])
        end_ts = float(end_row["event_timestamp"])
        duration_s = end_ts - start_ts

        completion_tokens = int(end_row.get("completion_tokens") or 0)

        calls.append({
            "function_name": start_row.get("function_name", ""),
            "example_number": start_row.get("example_number", ""),
            "duration_s": duration_s,
            "completion_tokens": completion_tokens,
            "prompt_tokens": int(end_row.get("prompt_tokens") or 0),
            "tokens_per_second": completion_tokens / duration_s if duration_s > 0 else 0.0,
        })

    return calls


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_ms(value: float) -> str:
    """Format a duration value in seconds as milliseconds."""
    return f"{value * 1000:.0f}ms"


def _fmt_tps(value: float) -> str:
    """Format tokens per second."""
    return f"{value:.1f}"


def _pct_change(baseline: float, current: float) -> str:
    """Format percentage change with color."""
    if baseline == 0:
        return ""
    pct = ((current - baseline) / baseline) * 100
    if pct < -1:
        return f"  {_GREEN}{pct:+.1f}%{_RESET}"
    if pct > 1:
        return f"  {_RED}{pct:+.1f}%{_RESET}"
    return f"  {_DIM}{pct:+.1f}%{_RESET}"


def print_report(
    sensitivity_map: dict[str, int],
    csv_datasets: list[tuple[str, list[dict]]],
) -> None:
    """Print the sensitivity performance comparison report."""

    # Attach sensitivity to each call
    enriched_datasets: list[tuple[str, list[dict]]] = []
    for label, calls in csv_datasets:
        enriched: list[dict] = []
        for call in calls:
            fn = call["function_name"]
            sensitivity = sensitivity_map.get(fn)
            if sensitivity is not None:
                enriched.append({**call, "sensitivity": sensitivity})
        enriched_datasets.append((label, enriched))

    if not enriched_datasets or not enriched_datasets[0][1]:
        print("No LLM calls matched the prediction trie. Check that function names match.")
        return

    # --- Header ---
    print()
    print(f"{_BOLD}{'=' * 110}{_RESET}")
    print(f"{_BOLD}LATENCY SENSITIVITY PERFORMANCE COMPARISON{_RESET}")
    print(f"{_BOLD}{'=' * 110}{_RESET}")
    print()

    # --- Per-function detail table ---
    print(f"{_BOLD}Per-Function Breakdown{_RESET}")
    print()

    # Collect all function names, sorted by sensitivity (descending)
    all_fns = sorted(sensitivity_map.keys(), key=lambda fn: -sensitivity_map.get(fn, 0))

    if len(enriched_datasets) == 1:
        _print_single_run_table(all_fns, sensitivity_map, enriched_datasets[0])
    else:
        _print_multi_run_table(all_fns, sensitivity_map, enriched_datasets)

    # --- Priority group summary ---
    print()
    print(f"{_BOLD}Priority Group Summary{_RESET}")
    print()

    for label, calls in enriched_datasets:
        if len(enriched_datasets) > 1:
            print(f"  {_BOLD}{label}{_RESET}")

        for group_name, group_filter in _PRIORITY_GROUPS.items():
            group_calls = [c for c in calls if group_filter(c["sensitivity"])]
            if not group_calls:
                continue

            durations = [c["duration_s"] for c in group_calls]
            tps_values = [c["tokens_per_second"] for c in group_calls]
            fn_names = sorted(set(c["function_name"] for c in group_calls))

            color = _RED if "HIGH" in group_name else (_YELLOW if "MEDIUM" in group_name else _GREEN)
            print(f"  {color}{group_name}{_RESET}  "
                  f"p50={_fmt_ms(statistics.median(durations)):>8}  "
                  f"p90={_fmt_ms(_percentile(durations, 90)):>8}  "
                  f"mean={_fmt_ms(statistics.mean(durations)):>8}  "
                  f"tps={_fmt_tps(statistics.mean(tps_values)):>6}  "
                  f"n={len(group_calls):<3}  "
                  f"fns=[{', '.join(fn_names)}]")

        print()

    # --- Key insight ---
    first_label, first_calls = enriched_datasets[0]
    high_calls = [c for c in first_calls if c["sensitivity"] >= 4]
    low_calls = [c for c in first_calls if c["sensitivity"] <= 2]
    if high_calls and low_calls:
        high_p50 = statistics.median([c["duration_s"] for c in high_calls])
        low_p50 = statistics.median([c["duration_s"] for c in low_calls])
        print(f"{_BOLD}Key Comparison ({first_label}):{_RESET}")
        print(f"  HIGH-priority p50: {_fmt_ms(high_p50):>8}")
        print(f"  LOW-priority  p50: {_fmt_ms(low_p50):>8}")
        if low_p50 > 0:
            ratio = low_p50 / high_p50
            if ratio > 1:
                print(f"  LOW calls are {_RED}{ratio:.1f}x slower{_RESET} than HIGH calls")
            else:
                print(f"  LOW calls are {_GREEN}{1/ratio:.1f}x faster{_RESET} than HIGH calls")
        print()


def _print_single_run_table(
    all_fns: list[str],
    sensitivity_map: dict[str, int],
    dataset: tuple[str, list[dict]],
) -> None:
    """Print a single-run per-function table."""
    label, calls = dataset
    calls_by_fn = _group_by_fn(calls)

    header = (f"  {'Function':<22} {'Sens':>5}  {'p50':>8}  {'p90':>8}  {'Mean':>8}  "
              f"{'TPS':>6}  {'Tokens':>6}  {'N':>3}")
    print(f"  {_DIM}{label}{_RESET}")
    print(header)
    print(f"  {'-' * 80}")

    for fn in all_fns:
        fn_calls = calls_by_fn.get(fn, [])
        if not fn_calls:
            continue
        _print_fn_row(fn, sensitivity_map.get(fn, 0), fn_calls)

    print()


def _print_multi_run_table(
    all_fns: list[str],
    sensitivity_map: dict[str, int],
    datasets: list[tuple[str, list[dict]]],
) -> None:
    """Print a multi-run comparison table."""
    baseline_label, baseline_calls = datasets[0]
    baseline_by_fn = _group_by_fn(baseline_calls)

    for idx, (label, calls) in enumerate(datasets):
        calls_by_fn = _group_by_fn(calls)
        is_baseline = (idx == 0)

        suffix = " (baseline)" if is_baseline else ""
        print(f"  {_BOLD}{label}{suffix}{_RESET}")
        header = (f"  {'Function':<22} {'Sens':>5}  {'p50':>8}  {'p90':>8}  {'Mean':>8}  "
                  f"{'TPS':>6}  {'Tokens':>6}  {'N':>3}")
        print(header)
        print(f"  {'-' * 80}")

        for fn in all_fns:
            fn_calls = calls_by_fn.get(fn, [])
            if not fn_calls:
                continue

            delta = ""
            if not is_baseline:
                bl_calls = baseline_by_fn.get(fn, [])
                if bl_calls:
                    bl_mean = statistics.mean([c["duration_s"] for c in bl_calls])
                    cur_mean = statistics.mean([c["duration_s"] for c in fn_calls])
                    delta = _pct_change(bl_mean, cur_mean)

            _print_fn_row(fn, sensitivity_map.get(fn, 0), fn_calls, delta)

        print()


def _print_fn_row(fn: str, sensitivity: int, fn_calls: list[dict], delta: str = "") -> None:
    """Print a single function row."""
    durations = [c["duration_s"] for c in fn_calls]
    tps_values = [c["tokens_per_second"] for c in fn_calls]
    tokens = [c["completion_tokens"] for c in fn_calls]

    sens_label, sens_color = _SENSITIVITY_LABELS.get(sensitivity, ("?", _RESET))
    sens_str = f"{sens_color}{sensitivity}/5{_RESET}"

    print(f"  {fn:<22} {sens_str:>14}  "
          f"{_fmt_ms(statistics.median(durations)):>8}  "
          f"{_fmt_ms(_percentile(durations, 90)):>8}  "
          f"{_fmt_ms(statistics.mean(durations)):>8}  "
          f"{_fmt_tps(statistics.mean(tps_values)):>6}  "
          f"{statistics.mean(tokens):>6.0f}  "
          f"{len(fn_calls):>3}"
          f"{delta}")


def _group_by_fn(calls: list[dict]) -> dict[str, list[dict]]:
    """Group calls by function_name."""
    by_fn: dict[str, list[dict]] = {}
    for c in calls:
        by_fn.setdefault(c["function_name"], []).append(c)
    return by_fn


def _percentile(data: list[float], pct: int) -> float:
    """Compute a percentile value."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (pct / 100) * (len(sorted_data) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_data) - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the sensitivity performance comparison CLI."""
    parser = argparse.ArgumentParser(
        description="Compare LLM call performance grouped by latency sensitivity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run analysis
  python -m latency_sensitivity_demo.compare_sensitivity_perf \\
      --trie outputs/profile/jobs/<job_id>/prediction_trie.json \\
      --csv  outputs/profile/jobs/<job_id>/standardized_data_all.csv

  # Compare baseline (NIM) vs Dynamo with sensitivity hints
  python -m latency_sensitivity_demo.compare_sensitivity_perf \\
      --trie outputs/profile/jobs/<job_id>/prediction_trie.json \\
      --csv  outputs/profile/jobs/<nim_job>/standardized_data_all.csv \\
      --csv  outputs/with_trie/jobs/<dynamo_job>/standardized_data_all.csv \\
      --labels "NIM (baseline)" "Dynamo + sensitivity"
""",
    )
    parser.add_argument("--trie", required=True, type=Path, help="Path to prediction_trie.json")
    parser.add_argument("--csv",
                        required=True,
                        type=Path,
                        action="append",
                        dest="csvs",
                        help="Path to standardized_data_all.csv (can specify multiple)")
    parser.add_argument("--labels", nargs="*", help="Labels for each CSV (default: filenames)")

    args = parser.parse_args()

    if not args.trie.exists():
        print(f"Error: Trie file not found: {args.trie}", file=sys.stderr)
        sys.exit(1)

    for csv_path in args.csvs:
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
            sys.exit(1)

    # Load trie and build sensitivity map
    trie_root = load_prediction_trie(args.trie)
    sensitivity_map = _collect_sensitivity_map(trie_root)

    if not sensitivity_map:
        print("Error: No sensitivity scores found in the prediction trie.", file=sys.stderr)
        sys.exit(1)

    # Parse CSVs
    labels = args.labels or [p.parent.name for p in args.csvs]
    if len(labels) < len(args.csvs):
        labels.extend(p.parent.name for p in args.csvs[len(labels):])

    csv_datasets = []
    for label, csv_path in zip(labels, args.csvs):
        calls = _parse_csv(csv_path)
        csv_datasets.append((label, calls))

    print_report(sensitivity_map, csv_datasets)


if __name__ == "__main__":
    main()
