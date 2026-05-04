#!/bin/bash
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

#===============================================================================
# Dynamo Concurrency Benchmark Script
#===============================================================================
# This script runs the banking evaluation with different concurrency levels
# and collects throughput statistics for performance analysis.
#===============================================================================

# Example output:
# Files created:
#   - benchmark_results.csv    (machine-readable data)
#   - benchmark_report.md      (human-readable report)
#   - analysis_*.txt           (detailed analysis for each run)

# Quick summary:

# Concurrency | Per-Req (mean) | Aggregate | Speedup | TTFT (mean) | ITL (mean)
# ------------|----------------|-----------|---------|-------------|------------
# 16          |    57.88 tok/s | 862.47 tok/s |  14.90x |    84.77 ms | 16.77 ms
# 32          |    44.62 tok/s | 1181.21 tok/s |  26.47x |   107.19 ms | 22.06 ms

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go up to dynamo_integration directory
DYNAMO_INTEGRATION_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# Go up to NeMo-Agent-Toolkit root
PROJECT_ROOT="$( cd "$DYNAMO_INTEGRATION_DIR/../.." && pwd )"
# Config and output paths are in react_benchmark_agent subdirectory
CONFIG_FILE="${DYNAMO_INTEGRATION_DIR}/react_benchmark_agent/configs/eval_config_rethinking_full_test.yml"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/throughput_analysis.py"
OUTPUT_BASE="${DYNAMO_INTEGRATION_DIR}/react_benchmark_agent/outputs"

echo "================================================================================"
echo "Dynamo Concurrency Benchmark"
echo "================================================================================"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Check if analysis script exists
if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo -e "${RED}ERROR: Analysis script not found: $ANALYSIS_SCRIPT${NC}"
    exit 1
fi

# Prompt for output filename
echo -e "${BLUE}Enter a unique name for the benchmark results (no extension):${NC}"
read -p "> " BENCHMARK_NAME

if [ -z "$BENCHMARK_NAME" ]; then
    echo -e "${RED}ERROR: Benchmark name cannot be empty${NC}"
    exit 1
fi

# Create output directory for this benchmark
BENCHMARK_DIR="${OUTPUT_BASE}/benchmarks/${BENCHMARK_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCHMARK_DIR"

echo -e "${GREEN}✓ Benchmark results will be saved to: $BENCHMARK_DIR${NC}"
echo ""

# Create temporary config file
TEMP_CONFIG="${BENCHMARK_DIR}/temp_config.yml"

# Concurrency levels to test
CONCURRENCY_LEVELS=(16 32)

# Array to store job information
declare -a JOB_IDS
declare -a JOB_DIRS
declare -a CONCURRENCY_VALUES

echo "================================================================================"
echo "Starting benchmark runs..."
echo "================================================================================"
echo ""

# Run evals for each concurrency level
for CONCURRENCY in "${CONCURRENCY_LEVELS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo -e "${BLUE}[Run $((${#JOB_IDS[@]} + 1))/8]${NC} Running eval with max_concurrency = ${YELLOW}$CONCURRENCY${NC}"
    echo "--------------------------------------------------------------------------------"
    
    # Create modified config with current concurrency
    cp "$CONFIG_FILE" "$TEMP_CONFIG"
    
    # Update max_concurrency in the temp config using sed
    sed -i "s/max_concurrency:.*$/max_concurrency: $CONCURRENCY/" "$TEMP_CONFIG"
    
    # Fix relative paths to absolute paths (for file_path in dataset section)
    # Replace ./examples/... with absolute path
    sed -i "s|file_path: \./examples/|file_path: ${PROJECT_ROOT}/examples/|g" "$TEMP_CONFIG"
    sed -i "s|tools_json_path: \./examples/|tools_json_path: ${PROJECT_ROOT}/examples/|g" "$TEMP_CONFIG"
    sed -i "s|dir: \./examples/|dir: ${PROJECT_ROOT}/examples/|g" "$TEMP_CONFIG"
    
    echo "Config: $TEMP_CONFIG"
    echo "Concurrency: $CONCURRENCY"
    echo ""
    
    # Run the eval and capture output
    echo "Running: nat eval --config_file $TEMP_CONFIG"
    
    # Run eval and capture job directory from output
    if [ -t 1 ]; then
        # Interactive terminal - show output with tee
        EVAL_OUTPUT=$(cd "$PROJECT_ROOT" && nat eval --config_file "$TEMP_CONFIG" 2>&1 | tee /dev/tty)
    else
        # Non-interactive - just capture and display
        EVAL_OUTPUT=$(cd "$PROJECT_ROOT" && nat eval --config_file "$TEMP_CONFIG" 2>&1)
        echo "$EVAL_OUTPUT"
    fi
    
    # Extract job directory from output
    # Looking for pattern like: "outputs/dynamo_evals/<experiment_name>/jobs/job_<uuid>"
    # The experiment name comes from the eval config's output.dir setting
    JOB_DIR=$(echo "$EVAL_OUTPUT" | grep -oP "dynamo_evals/[^/]+/jobs/job_[a-f0-9\-]+" | tail -1)
    
    if [ -z "$JOB_DIR" ]; then
        echo -e "${RED}ERROR: Could not extract job directory from eval output${NC}"
        echo "Continuing with next concurrency level..."
        continue
    fi
    
    # Full path to job directory (JOB_DIR already includes experiment_name/jobs/job_<uuid>)
    FULL_JOB_DIR="${OUTPUT_BASE}/${JOB_DIR}"
    
    echo -e "${GREEN}✓ Eval completed${NC}"
    echo "  Job directory: $FULL_JOB_DIR"
    
    # Store job information
    JOB_IDS+=("$(basename $JOB_DIR)")
    JOB_DIRS+=("$FULL_JOB_DIR")
    CONCURRENCY_VALUES+=("$CONCURRENCY")
    
    echo ""
    sleep 2  # Brief pause between runs
done

# Clean up temp config
rm -f "$TEMP_CONFIG"

echo "================================================================================"
echo "All eval runs completed. Analyzing results..."
echo "================================================================================"
echo ""

# Create results CSV header
RESULTS_CSV="${BENCHMARK_DIR}/benchmark_results.csv"
echo "concurrency,total_llm_calls,total_tokens,sum_of_durations_sec,wall_clock_time_sec,ttft_mean_ms,ttft_median_ms,ttft_p90_ms,ttft_p95_ms,ttft_p99_ms,ttft_min_ms,ttft_max_ms,itl_mean_ms,itl_median_ms,itl_p90_ms,itl_p95_ms,itl_p99_ms,itl_min_ms,itl_max_ms,itl_percall_mean_ms,itl_percall_median_ms,itl_percall_p90_ms,itl_percall_p95_ms,per_request_throughput_mean_toks,per_request_throughput_median_toks,per_request_throughput_p90_toks,per_request_throughput_p95_toks,per_request_throughput_p99_toks,per_request_throughput_min_toks,per_request_throughput_max_toks,aggregate_throughput_toks,aggregate_speedup" > "$RESULTS_CSV"

# Create markdown report header
REPORT_MD="${BENCHMARK_DIR}/benchmark_report.md"
cat > "$REPORT_MD" << EOF
# Dynamo Concurrency Benchmark Report

**Benchmark Name:** $BENCHMARK_NAME  
**Date:** $(date '+%Y-%m-%d %H:%M:%S')  
**Config:** eval_config_banking_full_test.yml

## Summary

This benchmark evaluates Dynamo performance across different concurrency levels (1-8).

## Results

EOF

# Process each job
for i in "${!JOB_IDS[@]}"; do
    JOB_ID="${JOB_IDS[$i]}"
    JOB_DIR="${JOB_DIRS[$i]}"
    CONCURRENCY="${CONCURRENCY_VALUES[$i]}"
    
    echo "--------------------------------------------------------------------------------"
    echo -e "${BLUE}Analyzing:${NC} Job $((i + 1))/${#JOB_IDS[@]} (concurrency=$CONCURRENCY)"
    echo "--------------------------------------------------------------------------------"
    
    # Find standardized_data_all.csv
    CSV_FILE="${JOB_DIR}/standardized_data_all.csv"
    
    if [ ! -f "$CSV_FILE" ]; then
        echo -e "${YELLOW}WARNING: CSV file not found: $CSV_FILE${NC}"
        echo "Skipping this job..."
        continue
    fi
    
    echo "CSV: $CSV_FILE"
    
    # Run throughput analysis and capture output
    ANALYSIS_OUTPUT="${BENCHMARK_DIR}/analysis_${CONCURRENCY}.txt"
    python "$ANALYSIS_SCRIPT" "$CSV_FILE" > "$ANALYSIS_OUTPUT" 2>&1
    
    echo -e "${GREEN}✓ Analysis complete${NC}"
    
    # Parse the analysis output to extract statistics
    # This uses grep and awk to extract specific values
    
    # Extract dataset overview
    TOTAL_CALLS=$(grep "Total LLM Calls:" "$ANALYSIS_OUTPUT" | awk '{print $4}')
    TOTAL_TOKENS=$(grep "Total Tokens Generated:" "$ANALYSIS_OUTPUT" | awk '{print $4}' | tr -d ',')
    SUM_OF_DURATIONS=$(grep "Sum of LLM Durations:" "$ANALYSIS_OUTPUT" | awk '{print $5}' | tr -d 's')
    WALL_CLOCK_TIME=$(grep "Wall-Clock Time:" "$ANALYSIS_OUTPUT" | head -1 | awk '{print $3}' | tr -d 's')
    
    # Extract TTFT statistics
    TTFT_MEAN=$(grep "Time To First Token" -A 7 "$ANALYSIS_OUTPUT" | grep "Mean:" | awk '{print $2}')
    TTFT_MEDIAN=$(grep "Time To First Token" -A 7 "$ANALYSIS_OUTPUT" | grep "Median:" | awk '{print $2}')
    TTFT_P90=$(grep "Time To First Token" -A 7 "$ANALYSIS_OUTPUT" | grep "P90:" | awk '{print $2}')
    TTFT_P95=$(grep "Time To First Token" -A 7 "$ANALYSIS_OUTPUT" | grep "P95:" | awk '{print $2}')
    TTFT_P99=$(grep "Time To First Token" -A 7 "$ANALYSIS_OUTPUT" | grep "P99:" | awk '{print $2}')
    TTFT_MIN=$(grep "Time To First Token" -A 7 "$ANALYSIS_OUTPUT" | grep "Min:" | awk '{print $2}')
    TTFT_MAX=$(grep "Time To First Token" -A 7 "$ANALYSIS_OUTPUT" | grep "Max:" | awk '{print $2}')
    
    # Extract ITL global statistics
    ITL_MEAN=$(grep "Global Statistics" -A 7 "$ANALYSIS_OUTPUT" | grep "Mean:" | head -1 | awk '{print $2}')
    ITL_MEDIAN=$(grep "Global Statistics" -A 7 "$ANALYSIS_OUTPUT" | grep "Median:" | head -1 | awk '{print $2}')
    ITL_P90=$(grep "Global Statistics" -A 7 "$ANALYSIS_OUTPUT" | grep "P90:" | head -1 | awk '{print $2}')
    ITL_P95=$(grep "Global Statistics" -A 7 "$ANALYSIS_OUTPUT" | grep "P95:" | head -1 | awk '{print $2}')
    ITL_P99=$(grep "Global Statistics" -A 7 "$ANALYSIS_OUTPUT" | grep "P99:" | head -1 | awk '{print $2}')
    ITL_MIN=$(grep "Global Statistics" -A 7 "$ANALYSIS_OUTPUT" | grep "Min:" | head -1 | awk '{print $2}')
    ITL_MAX=$(grep "Global Statistics" -A 7 "$ANALYSIS_OUTPUT" | grep "Max:" | head -1 | awk '{print $2}')
    
    # Extract ITL per-call statistics
    ITL_PERCALL_MEAN=$(grep "Per-Call Average ITL:" -A 4 "$ANALYSIS_OUTPUT" | grep "Mean:" | awk '{print $2}')
    ITL_PERCALL_MEDIAN=$(grep "Per-Call Average ITL:" -A 4 "$ANALYSIS_OUTPUT" | grep "Median:" | awk '{print $2}')
    ITL_PERCALL_P90=$(grep "Per-Call Average ITL:" -A 4 "$ANALYSIS_OUTPUT" | grep "P90:" | awk '{print $2}')
    ITL_PERCALL_P95=$(grep "Per-Call Average ITL:" -A 4 "$ANALYSIS_OUTPUT" | grep "P95:" | awk '{print $2}')
    
    # Extract per-request throughput statistics
    THROUGHPUT_MEAN=$(grep "Per-Request Throughput" -A 7 "$ANALYSIS_OUTPUT" | grep "Mean:" | awk '{print $2}')
    THROUGHPUT_MEDIAN=$(grep "Per-Request Throughput" -A 7 "$ANALYSIS_OUTPUT" | grep "Median:" | awk '{print $2}')
    THROUGHPUT_P90=$(grep "Per-Request Throughput" -A 7 "$ANALYSIS_OUTPUT" | grep "P90:" | awk '{print $2}')
    THROUGHPUT_P95=$(grep "Per-Request Throughput" -A 7 "$ANALYSIS_OUTPUT" | grep "P95:" | awk '{print $2}')
    THROUGHPUT_P99=$(grep "Per-Request Throughput" -A 7 "$ANALYSIS_OUTPUT" | grep "P99:" | awk '{print $2}')
    THROUGHPUT_MIN=$(grep "Per-Request Throughput" -A 7 "$ANALYSIS_OUTPUT" | grep "Min:" | awk '{print $2}')
    THROUGHPUT_MAX=$(grep "Per-Request Throughput" -A 7 "$ANALYSIS_OUTPUT" | grep "Max:" | awk '{print $2}')
    
    # Extract aggregate throughput statistics (all concurrent requests)
    AGG_THROUGHPUT=$(grep "Aggregate Throughput (All Concurrent" -A 4 "$ANALYSIS_OUTPUT" | grep "Aggregate Throughput:" | awk '{print $3}')
    AGG_SPEEDUP=$(grep "Aggregate Throughput (All Concurrent" -A 4 "$ANALYSIS_OUTPUT" | grep "Speedup" | awk '{print $4}' | tr -d 'x')
    
    # Append to CSV
    echo "$CONCURRENCY,$TOTAL_CALLS,$TOTAL_TOKENS,$SUM_OF_DURATIONS,$WALL_CLOCK_TIME,$TTFT_MEAN,$TTFT_MEDIAN,$TTFT_P90,$TTFT_P95,$TTFT_P99,$TTFT_MIN,$TTFT_MAX,$ITL_MEAN,$ITL_MEDIAN,$ITL_P90,$ITL_P95,$ITL_P99,$ITL_MIN,$ITL_MAX,$ITL_PERCALL_MEAN,$ITL_PERCALL_MEDIAN,$ITL_PERCALL_P90,$ITL_PERCALL_P95,$THROUGHPUT_MEAN,$THROUGHPUT_MEDIAN,$THROUGHPUT_P90,$THROUGHPUT_P95,$THROUGHPUT_P99,$THROUGHPUT_MIN,$THROUGHPUT_MAX,$AGG_THROUGHPUT,$AGG_SPEEDUP" >> "$RESULTS_CSV"
    
    # Append to markdown report
    cat >> "$REPORT_MD" << EOF

### Concurrency = $CONCURRENCY

**Dataset Overview:**
- Total LLM Calls: $TOTAL_CALLS
- Total Tokens Generated: $(printf "%'d" $TOTAL_TOKENS)
- Sum of LLM Durations: ${SUM_OF_DURATIONS}s
- Wall-Clock Time: ${WALL_CLOCK_TIME}s

**Time To First Token (TTFT):**
- Mean: ${TTFT_MEAN} ms
- Median: ${TTFT_MEDIAN} ms
- P90: ${TTFT_P90} ms
- P95: ${TTFT_P95} ms
- P99: ${TTFT_P99} ms

**Inter-Token Latency (ITL):**
- Global Mean: ${ITL_MEAN} ms
- Global Median: ${ITL_MEDIAN} ms
- Per-Call Mean: ${ITL_PERCALL_MEAN} ms
- Per-Call Median: ${ITL_PERCALL_MEDIAN} ms

**Per-Request Throughput** (individual LLM call performance):
- Mean: ${THROUGHPUT_MEAN} tok/s
- Median: ${THROUGHPUT_MEDIAN} tok/s
- P90: ${THROUGHPUT_P90} tok/s
- P95: ${THROUGHPUT_P95} tok/s

**Aggregate Throughput** (total tokens / wall-clock time across all concurrent requests):
- Aggregate: ${AGG_THROUGHPUT} tok/s
- Speedup vs Per-Request Mean: ${AGG_SPEEDUP}x

---

EOF
    
    echo ""
done

# Create summary table
echo "================================================================================"
echo "Benchmark Complete!"
echo "================================================================================"
echo ""
echo "Results saved to: $BENCHMARK_DIR"
echo ""
echo "Files created:"
echo "  - benchmark_results.csv    (machine-readable data)"
echo "  - benchmark_report.md      (human-readable report)"
echo "  - analysis_*.txt           (detailed analysis for each run)"
echo ""
echo "Quick summary:"
echo ""

# Display summary table
echo "Concurrency | Per-Req (mean) | Aggregate | Speedup | TTFT (mean) | ITL (mean)"
echo "------------|----------------|-----------|---------|-------------|------------"

while IFS=, read -r concurrency _ _ _ _ ttft_mean _ _ _ _ _ _ itl_mean _ _ _ _ _ _ _ _ _ _ throughput_mean _ _ _ _ _ _ agg_throughput agg_speedup; do
    if [ "$concurrency" != "concurrency" ]; then  # Skip header
        printf "%-11s | %14s | %9s | %7s | %11s | %s\n" "$concurrency" "${throughput_mean} tok/s" "${agg_throughput} tok/s" "${agg_speedup}x" "${ttft_mean} ms" "${itl_mean} ms"
    fi
done < "$RESULTS_CSV"

echo ""
echo "View full report: cat $BENCHMARK_DIR/benchmark_report.md"
echo "View CSV data: cat $BENCHMARK_DIR/benchmark_results.csv"
echo ""
echo "================================================================================"

