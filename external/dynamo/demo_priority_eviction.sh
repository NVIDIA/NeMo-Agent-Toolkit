#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# MultiLRU Priority Eviction Demo
# ================================
# Demonstrates frequency-based cache eviction protection
#
# Prerequisites:
#   - Start Dynamo with: DYNAMO_NUM_GPU_BLOCKS_OVERRIDE=12
#   - This gives us 12 blocks total (small cache for quick demo)
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  RECOMMENDED: Run the KV Event Observer in a separate terminal         │
# │                                                                         │
# │  This lets you see cache events in real-time as the demo runs:         │
# │    📦 STORED  - Blocks committed to prefix cache                       │
# │    🗑️ REMOVED - Blocks evicted (should be COLD blocks, not HOT!)       │
# │    ✅ CACHE HIT - Tokens served from cache                             │
# │                                                                         │
# │  Run inside the container:                                              │
# │    docker exec -it dynamo-vllm python \                                 │
# │      /workspace/monitoring/scripts/kv_event_observer.py \               │
# │      --port 20080 --verbose --metrics-port 18081                        │
# │                                                                         │
# │  This shows you EXACTLY what the MultiLRU eviction policy is doing:    │
# │  - Watch HOT blocks get stored and stay in cache                       │
# │  - Watch COLD blocks get stored then evicted                           │
# │  - Verify HOT blocks are protected when cache fills up                 │
# └─────────────────────────────────────────────────────────────────────────┘
#
# What this demo shows:
#   1. Access a "HOT" prompt multiple times (promotes to VeryHot pool)
#   2. Fill cache with unique "COLD" prompts (forces eviction)
#   3. Access HOT prompt again - it still gets cache hits!
#   4. Cold blocks were evicted, hot blocks protected

set -euo pipefail

API="http://localhost:8000/v1/completions"
MODEL="llama-3.3-70b"

# Long prompt to fill ~2 blocks (128+ tokens with block_size=64)
HOT_PROMPT="HOT_DEMO: This prompt will be accessed frequently and should be protected from eviction by the MultiLRU frequency-based cache management system. The quick brown fox jumps over the lazy dog multiple times throughout this demonstration. First jump over the lazy dog. Second jump over the lazy dog. Third jump over the lazy dog. Fourth jump over the lazy dog. Fifth jump over the lazy dog. Sixth jump over the lazy dog. Seventh jump over the lazy dog. Eighth jump over the lazy dog. This text ensures we have enough tokens to fill at least two complete KV cache blocks for proper prefix caching behavior."

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     MultiLRU Priority Eviction Demo                         ║"
echo "║     Thresholds: [3, 8, 15] accesses for pool promotion      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Get baseline
get_hits() {
  docker exec dynamo-vllm curl -s http://localhost:18081/metrics 2>/dev/null | \
    grep "prefix_cache_hits_total{" | grep -v external | awk '{print $NF}'
}

BASELINE=$(get_hits)
echo "📊 Baseline cache hits: $BASELINE"
echo ""

# ============================================================
# STEP 1: Make HOT prompt "hot" (20 accesses → VeryHot pool)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔥 STEP 1: Access HOT prompt 20 times (threshold for VeryHot: 15)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for i in {1..20}; do
  curl -s "$API" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"$HOT_PROMPT\",
    \"max_tokens\": 2,
    \"nvext\": {
      \"annotations\": [
        \"prefix_id:hot-demo-prompt\",
        \"backend:frequency_multi_lru\"
      ]
    }
  }" > /dev/null
  echo -n "🔥"
done
echo ""

AFTER_HOT=$(get_hits)
HOT_HITS=$((${AFTER_HOT%.*} - ${BASELINE%.*}))
echo "   Cache hits from HOT prompt: $HOT_HITS tokens"
echo "   → HOT blocks now in VeryHot pool (protected)"
echo ""

# ============================================================
# STEP 2: Fill cache with COLD prompts (forces eviction)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "❄️  STEP 2: Fill cache with 20 unique COLD prompts"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for i in {1..20}; do
  # Each COLD prompt is unique and fills 2+ blocks
  COLD="COLD_$i: This is unique cold prompt number $i designed to fill the KV cache and trigger eviction. The quick brown fox jumps over the lazy dog. First unique jump $i. Second unique jump $i. Third unique jump $i. Fourth unique jump $i. Fifth unique jump $i. Sixth unique jump $i. Adding more padding text to ensure this prompt fills at least two complete cache blocks. Extra content for block filling: $i $i $i $i $i $i $i $i."
  curl -s "$API" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"$COLD\",
    \"max_tokens\": 2,
    \"nvext\": {
      \"annotations\": [
        \"prefix_id:cold-$i\",
        \"backend:frequency_multi_lru\"
      ]
    }
  }" > /dev/null
  echo -n "❄️"
done
echo ""

AFTER_COLD=$(get_hits)
echo "   Cold prompts added (each unique, no cache hits expected)"
echo "   → Eviction should have occurred (cache overflow)"
echo ""

# ============================================================
# STEP 3: Test HOT prompt - should still get cache hits!
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 STEP 3: Access HOT prompt again (was it protected?)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for i in {1..5}; do
  curl -s "$API" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"$HOT_PROMPT\",
    \"max_tokens\": 2,
    \"nvext\": {
      \"annotations\": [
        \"prefix_id:hot-demo-prompt\",
        \"backend:frequency_multi_lru\"
      ]
    }
  }" > /dev/null
  echo -n "🎯"
done
echo ""

FINAL=$(get_hits)
FINAL_HITS=$((${FINAL%.*} - ${AFTER_COLD%.*}))
echo ""

# ============================================================
# RESULTS
# ============================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                        RESULTS                               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  HOT prompt initial cache hits:     %6d tokens           ║\n" "$HOT_HITS"
printf "║  HOT prompt hits AFTER eviction:    %6d tokens           ║\n" "$FINAL_HITS"
echo "╠══════════════════════════════════════════════════════════════╣"

if [ "$FINAL_HITS" -gt 0 ]; then
  echo "║  ✅ SUCCESS: Hot blocks PROTECTED from eviction!            ║"
  echo "║                                                              ║"
  echo "║  MultiLRU frequency-based eviction kept the frequently      ║"
  echo "║  accessed blocks while evicting cold (single-access) ones.  ║"
else
  echo "║  ❌ Hot blocks were evicted (no protection)                  ║"
fi
echo "╚══════════════════════════════════════════════════════════════╝"

