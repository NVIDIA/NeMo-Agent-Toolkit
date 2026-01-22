#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dynamo Metrics Collector
# Saves metrics from Frontend and Worker to timestamped files

OUTPUT_DIR="${1:-./metrics_logs}"
INTERVAL="${2:-30}"  # Collection interval in seconds

mkdir -p "$OUTPUT_DIR"

echo "=== Dynamo Metrics Collector ==="
echo "Output directory: $OUTPUT_DIR"
echo "Collection interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

collect_metrics() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local frontend_file="$OUTPUT_DIR/frontend_${timestamp}.prom"
    local worker_file="$OUTPUT_DIR/worker_${timestamp}.prom"
    local combined_file="$OUTPUT_DIR/combined_${timestamp}.prom"
    
    echo "[$(date)] Collecting metrics..."
    
    # Collect frontend metrics
    curl -s http://localhost:8000/metrics > "$frontend_file" 2>/dev/null
    
    # Collect worker metrics
    curl -s http://localhost:8081/metrics > "$worker_file" 2>/dev/null
    
    # Create combined file with headers
    {
        echo "# Collected at: $(date -Iseconds)"
        echo "# === FRONTEND METRICS ==="
        cat "$frontend_file"
        echo ""
        echo "# === WORKER METRICS ==="
        cat "$worker_file"
    } > "$combined_file"
    
    # Also append to a rolling log (last 24 hours of key metrics)
    {
        echo "# Timestamp: $(date -Iseconds)"
        grep -E '^dynamo_frontend_(requests_total|time_to_first_token|inter_token_latency|inflight)' "$frontend_file" 2>/dev/null
        grep -E '^dynamo_component_(request_duration|inflight|kvstats)' "$worker_file" 2>/dev/null
        echo ""
    } >> "$OUTPUT_DIR/rolling_metrics.log"
    
    echo "  Saved: $combined_file"
}

# Collect once immediately
collect_metrics

# Then collect at intervals
while true; do
    sleep "$INTERVAL"
    collect_metrics
done


