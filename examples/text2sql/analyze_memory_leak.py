#!/usr/bin/env python3
"""
Analyze memory leak patterns from load test results.

This script analyzes memory CSV files to identify:
- Memory growth rate and patterns
- Leak severity
- Correlation with request volumes
- Potential leak sources
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def analyze_memory_file(csv_file: Path) -> dict:
    """
    Analyze a memory CSV file and return statistics.

    Args:
        csv_file: Path to memory CSV file

    Returns:
        Dictionary with analysis results
    """
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"error": "No data in CSV file"}

    # Extract memory values
    timestamps = [datetime.fromisoformat(row['timestamp']) for row in rows]
    rss_mb = [float(row['rss_mb']) for row in rows]
    rss_total_mb = [float(row['rss_total_mb']) for row in rows]
    vms_mb = [float(row['vms_mb']) for row in rows]
    num_children = [int(row['num_children']) for row in rows]

    # Calculate statistics
    duration = (timestamps[-1] - timestamps[0]).total_seconds()
    initial_memory = rss_total_mb[0]
    final_memory = rss_total_mb[-1]
    peak_memory = max(rss_total_mb)
    min_memory = min(rss_total_mb)
    avg_memory = sum(rss_total_mb) / len(rss_total_mb)

    memory_growth = final_memory - initial_memory
    growth_rate = memory_growth / duration if duration > 0 else 0
    growth_percentage = (memory_growth / initial_memory) * 100 if initial_memory > 0 else 0

    # Detect leak pattern
    if growth_percentage > 50:
        leak_severity = "SEVERE"
        leak_detected = True
    elif growth_percentage > 20:
        leak_severity = "MODERATE"
        leak_detected = True
    elif growth_percentage > 10:
        leak_severity = "MINOR"
        leak_detected = True
    else:
        leak_severity = "NONE"
        leak_detected = False

    # Calculate growth phases (divide test into quarters)
    quarter_size = len(rss_total_mb) // 4
    if quarter_size > 0:
        q1_growth = rss_total_mb[quarter_size] - rss_total_mb[0]
        q2_growth = rss_total_mb[quarter_size * 2] - rss_total_mb[quarter_size]
        q3_growth = rss_total_mb[quarter_size * 3] - rss_total_mb[quarter_size * 2]
        q4_growth = rss_total_mb[-1] - rss_total_mb[quarter_size * 3]

        # Check if growth is linear (consistent leak) or stabilizing
        growth_pattern = [q1_growth, q2_growth, q3_growth, q4_growth]
        if all(g > 0 for g in growth_pattern[-2:]):
            pattern_type = "CONTINUOUS_LEAK"
        elif growth_pattern[-1] < growth_pattern[0]:
            pattern_type = "STABILIZING"
        else:
            pattern_type = "IRREGULAR"
    else:
        growth_pattern = []
        pattern_type = "INSUFFICIENT_DATA"

    return {
        "file": csv_file.name,
        "duration_seconds": duration,
        "sample_count": len(rows),
        "initial_memory_mb": initial_memory,
        "final_memory_mb": final_memory,
        "peak_memory_mb": peak_memory,
        "min_memory_mb": min_memory,
        "avg_memory_mb": avg_memory,
        "memory_growth_mb": memory_growth,
        "growth_rate_mb_per_sec": growth_rate,
        "growth_percentage": growth_percentage,
        "leak_detected": leak_detected,
        "leak_severity": leak_severity,
        "pattern_type": pattern_type,
        "quarterly_growth": growth_pattern,
        "avg_children": sum(num_children) / len(num_children),
        "max_children": max(num_children),
    }


def print_analysis(analysis: dict):
    """Print formatted analysis results."""
    print("\n" + "=" * 70)
    print("MEMORY LEAK ANALYSIS")
    print("=" * 70)
    print(f"\nFile: {analysis['file']}")
    print(f"Duration: {analysis['duration_seconds']:.2f} seconds")
    print(f"Samples: {analysis['sample_count']}")

    print("\n--- Memory Statistics ---")
    print(f"Initial memory:    {analysis['initial_memory_mb']:.2f} MB")
    print(f"Final memory:      {analysis['final_memory_mb']:.2f} MB")
    print(f"Peak memory:       {analysis['peak_memory_mb']:.2f} MB")
    print(f"Min memory:        {analysis['min_memory_mb']:.2f} MB")
    print(f"Average memory:    {analysis['avg_memory_mb']:.2f} MB")

    print("\n--- Memory Growth ---")
    print(f"Total growth:      {analysis['memory_growth_mb']:.2f} MB")
    print(f"Growth rate:       {analysis['growth_rate_mb_per_sec']:.4f} MB/sec")
    print(f"Growth percentage: {analysis['growth_percentage']:.2f}%")

    print("\n--- Leak Detection ---")
    if analysis['leak_detected']:
        severity_color = {"MINOR": "⚠️", "MODERATE": "⚠️⚠️", "SEVERE": "🔴"}
        symbol = severity_color.get(analysis['leak_severity'], "⚠️")
        print(f"Leak detected:     {symbol} YES")
        print(f"Severity:          {analysis['leak_severity']}")
    else:
        print(f"Leak detected:     ✓ NO (memory growth within normal range)")

    print(f"Pattern:           {analysis['pattern_type']}")

    if analysis['quarterly_growth']:
        print("\n--- Growth by Quarter ---")
        for i, growth in enumerate(analysis['quarterly_growth'], 1):
            print(f"Quarter {i}:         {growth:+.2f} MB")

    print("\n--- Process Statistics ---")
    print(f"Avg child procs:   {analysis['avg_children']:.1f}")
    print(f"Max child procs:   {analysis['max_children']}")

    print("\n--- Interpretation ---")
    if analysis['leak_detected']:
        if analysis['pattern_type'] == "CONTINUOUS_LEAK":
            print("⚠️  Memory continues to grow throughout the test.")
            print("   This indicates an active memory leak.")
            print("\n   Likely causes:")
            print("   - Objects not being garbage collected")
            print("   - Session/connection state accumulating")
            print("   - Caches growing without bounds")
            print("   - Event listeners not being removed")
        elif analysis['pattern_type'] == "STABILIZING":
            print("⚠️  Memory growth is slowing down.")
            print("   This may be normal caching behavior.")
            print("\n   Consider:")
            print("   - Is this a reasonable cache size?")
            print("   - Does it eventually stabilize?")
            print("   - Run longer test to confirm")
        else:
            print("⚠️  Memory growth pattern is irregular.")
            print("   This could indicate intermittent issues.")
    else:
        print("✓  Memory growth is within acceptable limits.")
        print("   The application appears to be managing memory properly.")

    print("=" * 70 + "\n")


def compare_multiple_files(csv_files: list[Path]):
    """Compare multiple memory test runs."""
    print("\n" + "=" * 70)
    print("COMPARING MULTIPLE TEST RUNS")
    print("=" * 70)

    analyses = []
    for csv_file in csv_files:
        if csv_file.exists():
            analysis = analyze_memory_file(csv_file)
            analyses.append(analysis)
        else:
            print(f"Warning: {csv_file} not found")

    if not analyses:
        print("No valid files to compare")
        return

    print(f"\n{'File':<40} {'Growth MB':<12} {'Growth %':<12} {'Leak'}")
    print("-" * 70)
    for analysis in analyses:
        leak_indicator = "YES" if analysis['leak_detected'] else "NO"
        severity = analysis.get('leak_severity', 'N/A')
        if analysis['leak_detected']:
            leak_str = f"{leak_indicator} ({severity})"
        else:
            leak_str = leak_indicator
        print(
            f"{analysis['file']:<40} {analysis['memory_growth_mb']:>10.2f}  {analysis['growth_percentage']:>10.2f}%  {leak_str}"
        )

    # Calculate averages
    avg_growth = sum(a['memory_growth_mb'] for a in analyses) / len(analyses)
    avg_growth_pct = sum(a['growth_percentage'] for a in analyses) / len(analyses)
    leak_count = sum(1 for a in analyses if a['leak_detected'])

    print("-" * 70)
    print(f"Average growth: {avg_growth:.2f} MB ({avg_growth_pct:.2f}%)")
    print(f"Leak detected in {leak_count}/{len(analyses)} runs")
    print("=" * 70 + "\n")


def generate_recommendations(analysis: dict):
    """Generate specific recommendations based on analysis."""
    print("\n" + "=" * 70)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 70)

    if not analysis['leak_detected']:
        print("\n✓ No significant memory leak detected.")
        print("\nOptional: Run longer tests to confirm stability.")
        return

    print(f"\n⚠️  {analysis['leak_severity']} memory leak detected")
    print(f"   Growth: {analysis['memory_growth_mb']:.2f} MB ({analysis['growth_percentage']:.2f}%)")

    print("\n1. Profile the Application")
    print("   - Use memory_profiler to identify growing objects")
    print("   - Use tracemalloc to find allocation sources")
    print("   - Use objgraph to visualize object references")

    print("\n2. Check Common Leak Sources")
    print("   - Review session management (sessions not being closed)")
    print("   - Check connection pools (connections not released)")
    print("   - Examine caches (unbounded growth)")
    print("   - Look for circular references")
    print("   - Check event listeners (not being unregistered)")

    print("\n3. Run Targeted Tests")
    print("   - Isolate specific endpoints/functions")
    print("   - Test with different load patterns")
    print("   - Compare different configurations")

    print("\n4. Use Profiling Tools")
    print("   Run with memory profiling:")
    print("   python -m memory_profiler run_text2sql_memory_leak_test.py")
    print("\n   Or use the leak identification script:")
    print("   python identify_leak_source.py")

    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze memory leak patterns from load test results")
    parser.add_argument("files", nargs="+", help="Memory CSV file(s) to analyze")
    parser.add_argument("--compare", action="store_true", help="Compare multiple files")
    parser.add_argument("--recommendations", action="store_true", help="Generate recommendations")

    args = parser.parse_args()

    csv_files = [Path(f) for f in args.files]

    if args.compare and len(csv_files) > 1:
        compare_multiple_files(csv_files)

    # Analyze each file
    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"Error: {csv_file} not found")
            continue

        analysis = analyze_memory_file(csv_file)
        print_analysis(analysis)

        if args.recommendations:
            generate_recommendations(analysis)

    return 0


if __name__ == "__main__":
    sys.exit(main())
