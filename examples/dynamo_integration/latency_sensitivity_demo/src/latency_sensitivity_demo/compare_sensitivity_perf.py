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

When multiple CSVs are provided (e.g. a baseline Dynamo run and a Dynamo run
with sensitivity hints), the report prints side-by-side columns so you can see
the improvement.  The ``--skip-warmup N`` flag drops the first *N* examples to
remove cold-cache effects.
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
        prompt_tokens, tokens_per_second, ms_per_token
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
        tps = completion_tokens / duration_s if duration_s > 0 else 0.0
        ms_per_tok = (duration_s * 1000 / completion_tokens) if completion_tokens > 0 else 0.0

        calls.append({
            "function_name": start_row.get("function_name", ""),
            "example_number": start_row.get("example_number", ""),
            "duration_s": duration_s,
            "completion_tokens": completion_tokens,
            "prompt_tokens": int(end_row.get("prompt_tokens") or 0),
            "tokens_per_second": tps,
            "ms_per_token": ms_per_tok,
        })

    return calls


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _fmt_ms(value: float) -> str:
    """Format a duration value in seconds as milliseconds."""
    return f"{value * 1000:.0f}ms"


def _fmt_tps(value: float) -> str:
    """Format tokens per second."""
    return f"{value:.1f}"


def _fmt_mspt(value: float) -> str:
    """Format milliseconds per token."""
    return f"{value:.1f}"


def _pct_change(baseline: float, current: float) -> str:
    """Format percentage change with color (lower is better for latency)."""
    if baseline == 0:
        return ""
    pct = ((current - baseline) / baseline) * 100
    if pct < -1:
        return f"  {_GREEN}{pct:+.1f}%{_RESET}"
    if pct > 1:
        return f"  {_RED}{pct:+.1f}%{_RESET}"
    return f"  {_DIM}{pct:+.1f}%{_RESET}"


def _pct_change_higher_better(baseline: float, current: float) -> str:
    """Format percentage change with color (higher is better, e.g. TPS)."""
    if baseline == 0:
        return ""
    pct = ((current - baseline) / baseline) * 100
    if pct > 1:
        return f"  {_GREEN}{pct:+.1f}%{_RESET}"
    if pct < -1:
        return f"  {_RED}{pct:+.1f}%{_RESET}"
    return f"  {_DIM}{pct:+.1f}%{_RESET}"


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


def _sensitivity_str(score: int, width: int = 14) -> str:
    """Return a colored sensitivity string padded to *width* visible chars."""
    label, color = _SENSITIVITY_LABELS.get(score, ("?", _RESET))
    visible = f"{score}/5 ({label})"
    return f"{color}{visible.ljust(width)}{_RESET}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


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

    table_w = 110

    # --- Header ---
    print()
    print(f"{_BOLD}{'=' * table_w}{_RESET}")
    print(f"{_BOLD}LATENCY SENSITIVITY PERFORMANCE COMPARISON{_RESET}")
    print(f"{_BOLD}{'=' * table_w}{_RESET}")
    print()

    # Collect all function names, sorted by sensitivity (descending)
    all_fns = sorted(sensitivity_map.keys(), key=lambda fn: -sensitivity_map.get(fn, 0))

    # --- Per-function detail table ---
    print(f"{_BOLD}Per-Function Breakdown{_RESET}")
    print()

    if len(enriched_datasets) == 1:
        _print_single_run_table(all_fns, sensitivity_map, enriched_datasets[0])
    else:
        _print_multi_run_table(all_fns, sensitivity_map, enriched_datasets)

    # --- Priority group summary ---
    _print_priority_summary(enriched_datasets)

    # --- Cross-run priority ratio comparison ---
    if len(enriched_datasets) > 1:
        _print_priority_ratio_comparison(enriched_datasets)


def _print_fn_header() -> str:
    """Return the column header line for function tables."""
    return (f"  {'Function':<22}  {'Sensitivity':<14}  {'p50':>7}  {'p90':>7}  {'Mean':>7}"
            f"  {'ms/tok':>6}  {'TPS':>5}  {'Tokens':>6}  {'N':>3}")


def _print_single_run_table(
    all_fns: list[str],
    sensitivity_map: dict[str, int],
    dataset: tuple[str, list[dict]],
) -> None:
    """Print a single-run per-function table."""
    label, calls = dataset
    calls_by_fn = _group_by_fn(calls)

    print(f"  {_DIM}{label}{_RESET}")
    print(_print_fn_header())
    print(f"  {'-' * 100}")

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
    """Print a multi-run comparison table with ms/tok delta."""
    baseline_label, baseline_calls = datasets[0]
    baseline_by_fn = _group_by_fn(baseline_calls)

    for idx, (label, calls) in enumerate(datasets):
        calls_by_fn = _group_by_fn(calls)
        is_baseline = (idx == 0)

        suffix = " (baseline)" if is_baseline else ""
        print(f"  {_BOLD}{label}{suffix}{_RESET}")
        print(_print_fn_header())
        print(f"  {'-' * 100}")

        for fn in all_fns:
            fn_calls = calls_by_fn.get(fn, [])
            if not fn_calls:
                continue

            delta = ""
            if not is_baseline:
                bl_calls = baseline_by_fn.get(fn, [])
                if bl_calls:
                    bl_mspt = statistics.mean([c["ms_per_token"] for c in bl_calls])
                    cur_mspt = statistics.mean([c["ms_per_token"] for c in fn_calls])
                    delta = _pct_change(bl_mspt, cur_mspt)

            _print_fn_row(fn, sensitivity_map.get(fn, 0), fn_calls, delta)

        print()


def _print_fn_row(fn: str, sensitivity: int, fn_calls: list[dict], delta: str = "") -> None:
    """Print a single function row."""
    durations = [c["duration_s"] for c in fn_calls]
    tps_values = [c["tokens_per_second"] for c in fn_calls]
    mspt_values = [c["ms_per_token"] for c in fn_calls]
    tokens = [c["completion_tokens"] for c in fn_calls]

    sens_str = _sensitivity_str(sensitivity)

    print(f"  {fn:<22}  {sens_str}  "
          f"{_fmt_ms(statistics.median(durations)):>7}  "
          f"{_fmt_ms(_percentile(durations, 90)):>7}  "
          f"{_fmt_ms(statistics.mean(durations)):>7}  "
          f"{_fmt_mspt(statistics.mean(mspt_values)):>6}  "
          f"{_fmt_tps(statistics.mean(tps_values)):>5}  "
          f"{statistics.mean(tokens):>6.0f}  "
          f"{len(fn_calls):>3}"
          f"{delta}")


def _print_priority_summary(enriched_datasets: list[tuple[str, list[dict]]]) -> None:
    """Print per-priority-group summary with ms/tok."""
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
            mspt_values = [c["ms_per_token"] for c in group_calls]
            fn_names = sorted(set(c["function_name"] for c in group_calls))

            color = _RED if "HIGH" in group_name else (_YELLOW if "MEDIUM" in group_name else _GREEN)
            print(f"  {color}{group_name}{_RESET}  "
                  f"p50={_fmt_ms(statistics.median(durations)):>8}  "
                  f"mean={_fmt_ms(statistics.mean(durations)):>8}  "
                  f"ms/tok={_fmt_mspt(statistics.mean(mspt_values)):>5}  "
                  f"tps={_fmt_tps(statistics.mean(tps_values)):>5}  "
                  f"n={len(group_calls):<3}  "
                  f"fns=[{', '.join(fn_names)}]")

        print()


def _print_priority_ratio_comparison(enriched_datasets: list[tuple[str, list[dict]]]) -> None:
    """Print cross-run comparison of the HIGH/LOW priority ratio.

    The key metric: within each run, how much faster (per token) are HIGH
    calls vs LOW calls?  If sensitivity routing works, the ratio should
    improve (HIGH gets relatively faster).
    """
    print(f"{_BOLD}Priority Routing Effectiveness{_RESET}")
    print()

    ratios: list[tuple[str, float]] = []

    for label, calls in enriched_datasets:
        high_calls = [c for c in calls if c["sensitivity"] >= 4]
        low_calls = [c for c in calls if c["sensitivity"] <= 2]

        if not high_calls or not low_calls:
            continue

        high_mspt = statistics.mean([c["ms_per_token"] for c in high_calls])
        low_mspt = statistics.mean([c["ms_per_token"] for c in low_calls])
        high_tps = statistics.mean([c["tokens_per_second"] for c in high_calls])
        low_tps = statistics.mean([c["tokens_per_second"] for c in low_calls])

        ratio = low_mspt / high_mspt if high_mspt > 0 else 0.0
        ratios.append((label, ratio))

        print(f"  {_BOLD}{label}{_RESET}")
        print(f"    HIGH-priority  ms/tok: {_fmt_mspt(high_mspt):>6}   tps: {_fmt_tps(high_tps):>5}")
        print(f"    LOW-priority   ms/tok: {_fmt_mspt(low_mspt):>6}   tps: {_fmt_tps(low_tps):>5}")
        if ratio > 1:
            print(f"    HIGH calls are {_GREEN}{ratio:.2f}x faster per token{_RESET} than LOW")
        elif ratio < 1:
            inv = 1.0 / ratio if ratio > 0 else 0.0
            print(f"    HIGH calls are {_RED}{inv:.2f}x slower per token{_RESET} than LOW")
        else:
            print("    HIGH and LOW are equal per token")
        print()

    # Cross-run comparison
    if len(ratios) >= 2:
        baseline_label, baseline_ratio = ratios[0]
        print(f"  {_BOLD}Routing Impact{_RESET}")
        for run_label, run_ratio in ratios[1:]:
            improvement = run_ratio - baseline_ratio
            if improvement > 0.01:
                print(f"    {run_label}: HIGH/LOW ratio improved "
                      f"{_GREEN}{baseline_ratio:.2f}x → {run_ratio:.2f}x{_RESET} "
                      f"({_GREEN}+{improvement:.2f}{_RESET})")
            elif improvement < -0.01:
                print(f"    {run_label}: HIGH/LOW ratio regressed "
                      f"{_RED}{baseline_ratio:.2f}x → {run_ratio:.2f}x{_RESET} "
                      f"({_RED}{improvement:.2f}{_RESET})")
            else:
                print(f"    {run_label}: HIGH/LOW ratio unchanged "
                      f"{_DIM}{baseline_ratio:.2f}x → {run_ratio:.2f}x{_RESET}")
        print()


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
      --trie outputs/profile/prediction_trie.json \\
      --csv  outputs/profile/standardized_data_all.csv

  # Compare baseline vs Dynamo with sensitivity hints
  python -m latency_sensitivity_demo.compare_sensitivity_perf \\
      --trie outputs/profile/prediction_trie.json \\
      --csv  outputs/profile/standardized_data_all.csv \\
      --csv  outputs/with_trie/standardized_data_all.csv \\
      --labels "Dynamo (baseline)" "Dynamo + sensitivity"

  # Skip first 2 examples to remove warmup effects
  python -m latency_sensitivity_demo.compare_sensitivity_perf \\
      --trie outputs/profile/prediction_trie.json \\
      --csv  outputs/profile/standardized_data_all.csv \\
      --skip-warmup 2
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
    parser.add_argument("--skip-warmup",
                        type=int,
                        default=0,
                        metavar="N",
                        help="Drop the first N examples from each CSV (removes cold-cache effects)")

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

        # Apply warmup filter
        if args.skip_warmup > 0:
            skip_examples = set()
            all_examples = sorted(set(c["example_number"] for c in calls))
            skip_examples = set(all_examples[:args.skip_warmup])
            before = len(calls)
            calls = [c for c in calls if c["example_number"] not in skip_examples]
            skipped = before - len(calls)
            if skipped > 0:
                print(f"{_DIM}  [{label}] Skipped {skipped} calls from first "
                      f"{args.skip_warmup} examples (warmup){_RESET}")

        csv_datasets.append((label, calls))

    if args.skip_warmup > 0:
        print()

    print_report(sensitivity_map, csv_datasets)


if __name__ == "__main__":
    main()
