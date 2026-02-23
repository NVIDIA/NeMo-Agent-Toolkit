<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Latency Sensitivity Demo

This example demonstrates **automatic latency sensitivity inference** end-to-end: profiling a multi-step LLM workflow, computing per-node sensitivity scores, and using those scores as Dynamo routing hints at runtime for improved performance.

Agentic workflows are not flat sequences of identical LLM calls. Some calls gate everything downstream (the first classifier), some run in parallel with slack to spare, and some are the last thing before the user sees a response. Treating them all the same leaves performance on the table. This demo shows how the NeMo Agent Toolkit profiler can automatically detect which calls matter most and feed that information to Dynamo so it can route requests accordingly.

## Workflow: Customer Support Triage

The demo implements a customer support pipeline as a LangGraph `StateGraph` with five nodes. Each node is a separately registered NAT function, giving the profiler individual visibility into every LLM call.

<!-- path-check-skip-begin -->
```
                        ┌─── research_context ───┐
  classify_query ──────►│                         ├──► draft_response ──► review_response
    (sequential)        └─── lookup_policy ──────┘       (sequential)       (sequential)
                              (parallel pair)
```
<!-- path-check-skip-end -->

**Why this topology exercises all four sensitivity signals:**

| Node | What It Does | Topology Role |
|------|-------------|---------------|
| `classify_query` | Categorizes the customer query (billing, account, technical, general) | **Entry point.** Every downstream node depends on its output. High fan-out: 4 calls follow it. First position in the sequence. |
| `research_context` | Gathers knowledge-base context for the query | **Parallel sibling (shorter).** Runs concurrently with `lookup_policy`. Typically finishes first, so it has _parallel slack_ — it sat idle waiting for its longer sibling before the workflow could continue. |
| `lookup_policy` | Retrieves company policy for the query category | **Parallel sibling (longer).** The critical path through the parallel pair. Takes longer than `research_context`, so the join point (`draft_response`) was blocked on this node. |
| `draft_response` | Synthesizes context + policy into a customer response | **Join point.** Runs after both parallel siblings complete. Sequential, on the critical path, but positioned in the middle of the workflow. |
| `review_response` | QA review of the draft before sending to the customer | **Exit point.** Last node — its latency directly determines when the user sees a response. Zero fan-out, high critical-path contribution. |

### How Sensitivity Scores Are Computed

The profiler's auto-sensitivity algorithm combines four weighted signals into a composite score per node, then normalizes across the workflow so the full 1–5 scale is used:

| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| **Position** (`w_position`) | 0.50 | U-shaped curve: first and last calls in the sequence score highest. Middle calls score lowest. Reflects that entry and exit nodes have the most impact on end-to-end latency. |
| **Critical path** (`w_critical`) | 0.35 | Fraction of total workflow wall-clock time spent in this call. Long-running calls that dominate execution time score higher. |
| **Fan-out** (`w_fanout`) | 0.15 | How many LLM calls remain after this one. The entry node (4 calls remaining) gets a boost; the exit node (0 remaining) does not. |
| **Parallel slack** (`w_parallel`) | 0.50 | _Penalty_ for parallel siblings that finish early and sit idle. If `research_context` takes 3s but `lookup_policy` takes 5s, `research_context` had 2s of slack — it could have been slower without affecting the workflow. This signal subtracts from the score. |

After computing raw weighted scores for each call in a trace, the algorithm **min-max normalizes** across all calls so the most-sensitive call maps to 5/5 and the least-sensitive maps to 1/5. This ensures clear differentiation regardless of absolute weight values.

**Expected output for this workflow:**

| Node | Score | Rationale |
|------|-------|-----------|
| `classify_query` | **5/5 HIGH** | First position + highest fan-out. Everything depends on it. |
| `review_response` | **5/5 HIGH** | Last position + high critical-path fraction. User is waiting on this. |
| `draft_response` | **3/5 MEDIUM** | Sequential join point, moderate critical path, but mid-position dampens it. |
| `research_context` | **2/5 LOW-MED** | Parallel slack penalty — finishes before its sibling, so it had room to be slower. |
| `lookup_policy` | **1/5 LOW** | Mid-position, moderate critical path, but no fan-out or position boost to compensate. |

### What Dynamo Does With These Scores

When the NeMo Agent Toolkit Dynamo LLM client (`_type: dynamo`) is configured with a prediction trie, it injects `nvext.agent_hints` into the OpenAI-compatible request body for each LLM call. These hints tell Dynamo's router about the call's latency sensitivity, expected output length, interarrival pattern, and request priority. Dynamo can use this to:

- **Priority-route** HIGH-sensitivity calls (classify, review) to dedicated workers for lowest latency
- **Batch-route** LOW-sensitivity calls (research_context, lookup_policy) to shared workers where throughput is maximized
- **Optimize KV cache** allocation based on predicted output sequence length and cache TTL

## Prerequisites

- **Python 3.11+**
- **NeMo Agent Toolkit** installed with LangChain integration
- **NVIDIA API key** for NIM endpoint access (Step 1)
- **Dynamo backend** on a Linux GPU system (Steps 3–4). See the [Dynamo Setup Guide](../../../external/dynamo/README.md) for hardware and software requirements.

## Step 1: Profile the Workflow with NIM

First, run the workflow against an NVIDIA NIM endpoint to collect profiler traces and build the prediction trie. This step does not require Dynamo or GPUs — NIM provides the LLM inference.

```bash
# Install the example package
uv pip install -e ./examples/dynamo_integration/latency_sensitivity_demo

# Set your NVIDIA API key
export NVIDIA_API_KEY=nvapi-...

# Run profiling (8 queries, ~30 seconds)
nat eval --config_file examples/dynamo_integration/latency_sensitivity_demo/src/latency_sensitivity_demo/configs/config_profile.yml
```

The profiler runs the full 5-node workflow for each query in the dataset, records per-node timing spans, and builds a prediction trie with auto-sensitivity scores. Output goes to:

<!-- path-check-skip-begin -->
```
examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/jobs/<job_id>/
├── prediction_trie.json         # The prediction trie with sensitivity scores
├── all_requests_profiler_traces.json  # Raw per-event profiler traces
├── standardized_data_all.csv    # Per-LLM-call timing metrics
├── inference_optimization.json  # Summary statistics
└── config_effective.yml         # Effective config used
```
<!-- path-check-skip-end -->

## Step 2: View the Sensitivity Report

Use the included report tool to print a human-readable summary of the prediction trie:

<!-- path-check-skip-begin -->
```bash
python -m latency_sensitivity_demo.sensitivity_report \
  examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/jobs/<job_id>/prediction_trie.json
```
<!-- path-check-skip-end -->

**Example output:**

<!-- path-check-skip-begin -->
```
====================================================================================================
LATENCY SENSITIVITY REPORT
====================================================================================================

Path                                          Call#  Remaining  IAT (ms)   Tokens   Sensitivity
----------------------------------------------------------------------------------------------------
root                                          1      2.0        315.3      323.2    3/5 (MEDIUM)
root/<workflow>                               1      2.0        315.3      323.2    3/5 (MEDIUM)
root/<workflow>/classify_query                1      4.0        4.1        2.0      5/5 (HIGH)
root/<workflow>/draft_response                1      1.0        2.5        377.4    3/5 (MEDIUM)
root/<workflow>/lookup_policy                 1      2.0        3.0        437.0    1/5 (LOW)
root/<workflow>/research_context              1      3.0        1251.6     330.9    2/5 (LOW-MED)
root/<workflow>/review_response               1      0.0        0.0        469.0    5/5 (HIGH)

====================================================================================================
ROUTING RECOMMENDATIONS
====================================================================================================

  HIGH (4-5)   : Route to dedicated/priority workers for lowest latency
  MEDIUM (3)   : Standard routing — balance between latency and throughput
  LOW (1-2)    : Route to shared/batch workers — throughput over latency
```
<!-- path-check-skip-end -->

**How to read the columns:**

| Column | Meaning |
|--------|---------|
| **Path** | Trie path: `root/<workflow>/<function_name>`. Each registered NAT function gets its own node. |
| **Call#** | The LLM call index within this function (always 1 here since each function makes one call). |
| **Remaining** | Average number of LLM calls that follow this one in the workflow. `classify_query` = 4 (everything after it), `review_response` = 0 (last). |
| **IAT (ms)** | Mean inter-arrival time — milliseconds between this call ending and the next call starting. `research_context` shows ~1250ms because it finishes first and waits for `lookup_policy` to complete before `draft_response` can start. |
| **Tokens** | Mean output token count. `classify_query` outputs ~2 tokens (just a category name), while `review_response` outputs ~469 (a full customer response). |
| **Sensitivity** | The auto-computed score from 1/5 (LOW) to 5/5 (HIGH). |

## Step 3: Start Dynamo Backend

The Dynamo backend runs on a Linux system with NVIDIA GPUs. See the [Dynamo Setup Guide](../../../external/dynamo/README.md) for full prerequisites and instructions.

<!-- path-check-skip-begin -->
```bash
# TODO: Add the startup script command for this demo's Dynamo configuration.
# The script should start Dynamo with the llama-3.3-70b model and the
# Thompson Sampling router that consumes the prediction trie headers.
#
# Example (placeholder):
# cd external/dynamo
# bash start_dynamo_unified.sh
```
<!-- path-check-skip-end -->

Verify the endpoint is responding:

```bash
curl -s http://localhost:8099/v1/models | python3 -m json.tool
```

## Step 4: Run With Latency Sensitivity Hints

Once Dynamo is running, update the prediction trie path in `config_with_trie.yml` and run the workflow. The Dynamo LLM client will inject per-request routing hints based on the profiled sensitivity scores.

<!-- path-check-skip-begin -->
```bash
# Update the trie path to your Step 1 output
sed -i.bak "s|REPLACE_WITH_JOB_ID|<your_job_id>|" \
  examples/dynamo_integration/latency_sensitivity_demo/src/latency_sensitivity_demo/configs/config_with_trie.yml

# Run the workflow against Dynamo with sensitivity-aware routing
nat eval --config_file examples/dynamo_integration/latency_sensitivity_demo/src/latency_sensitivity_demo/configs/config_with_trie.yml
```
<!-- path-check-skip-end -->

The Dynamo LLM client reads the prediction trie and, for each LLM call, injects an `nvext.agent_hints` object into the OpenAI-compatible request body. Dynamo's processor reads these hints directly from the request without any header parsing. The hints include:

| Field | Type | Description |
|-------|------|-------------|
| `prefix_id` | `string` | Unique prefix identifier for KV cache reuse across calls in the same workflow run |
| `total_requests` | `int` | Predicted remaining LLM calls — higher values increase KV cache affinity and worker stickiness |
| `osl` | `int` | Predicted output sequence length (tokens) — informs decode cost estimation |
| `iat` | `int` | Predicted inter-arrival time (ms) — informs request pacing and worker stickiness |
| `latency_sensitivity` | `float` | The auto-computed sensitivity score (1–5 from the prediction trie) |
| `priority` | `int` | Integer complement of sensitivity (`max_sensitivity - latency_sensitivity`). Lower value = higher priority. |

The client also injects `nvext.cache_control` with a TTL computed as `total_requests * iat` (the estimated conversation duration), so KV cache entries auto-expire after the workflow is expected to complete.

**Example request body (abridged):**

```json
{
  "model": "llama-3.3-70b",
  "messages": [...],
  "nvext": {
    "agent_hints": {
      "prefix_id": "eval-q001-abc123-d1",
      "total_requests": 4,
      "osl": 2,
      "iat": 4,
      "latency_sensitivity": 5.0,
      "priority": 995
    },
    "cache_control": {
      "type": "ephemeral",
      "ttl": "1s"
    }
  }
}
```

Dynamo's Thompson Sampling router uses these hints to make better scheduling decisions. HIGH-sensitivity calls (`classify_query`, `review_response`) get priority routing for lowest latency, while LOW-sensitivity calls (`research_context`, `lookup_policy`) can be batched for better throughput.

**To measure the performance improvement**, use the included comparison script. It joins per-LLM-call timing data from the profiler CSV with sensitivity scores from the prediction trie, then groups calls by priority level.

Single-run analysis (shows that HIGH-priority calls are inherently faster or slower based on workflow position):

<!-- path-check-skip-begin -->
```bash
python -m latency_sensitivity_demo.compare_sensitivity_perf \
    --trie  examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/jobs/<job_id>/prediction_trie.json \
    --csv   examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/jobs/<job_id>/standardized_data_all.csv
```
<!-- path-check-skip-end -->

Side-by-side comparison of NIM baseline vs Dynamo with sensitivity hints (shows the routing improvement):

<!-- path-check-skip-begin -->
```bash
python -m latency_sensitivity_demo.compare_sensitivity_perf \
    --trie   examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/jobs/<nim_job_id>/prediction_trie.json \
    --csv    examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/jobs/<nim_job_id>/standardized_data_all.csv \
    --csv    examples/dynamo_integration/latency_sensitivity_demo/outputs/with_trie/jobs/<dynamo_job_id>/standardized_data_all.csv \
    --labels "NIM (baseline)" "Dynamo + sensitivity"
```
<!-- path-check-skip-end -->

**Example output (single run):**

<!-- path-check-skip-begin -->
```
==============================================================================================================
LATENCY SENSITIVITY PERFORMANCE COMPARISON
==============================================================================================================

Per-Function Breakdown

  Function                Sens       p50       p90      Mean     TPS  Tokens    N
  --------------------------------------------------------------------------------
  classify_query           5/5     244ms     259ms     244ms     8.2       2    8
  review_response          5/5    5150ms    7037ms    5444ms    86.4     469    8
  draft_response           3/5    4538ms    5430ms    4529ms    83.9     377    8
  research_context         2/5    3433ms    4567ms    3793ms    87.3     331    8
  lookup_policy            1/5    4833ms    5857ms    5041ms    86.7     437    8

Priority Group Summary

  HIGH (4-5)  p50=  2006ms  p90=  6229ms  mean=  2844ms  tps=  47.3  n=16   fns=[classify_query, review_response]
  MEDIUM (3)  p50=  4538ms  p90=  5430ms  mean=  4529ms  tps=  83.9  n=8    fns=[draft_response]
  LOW (1-2)   p50=  4446ms  p90=  5637ms  mean=  4417ms  tps=  87.0  n=16   fns=[lookup_policy, research_context]

Key Comparison:
  HIGH-priority p50:   2006ms
  LOW-priority  p50:   4446ms
  LOW calls are 2.2x slower than HIGH calls
```
<!-- path-check-skip-end -->

**How to read the output:**

- **Per-Function Breakdown** shows each node sorted by sensitivity (highest first), with p50/p90/mean latency, tokens per second, and sample count. In multi-run mode, a `%` delta column shows improvement vs the baseline.
- **Priority Group Summary** aggregates calls into HIGH/MEDIUM/LOW buckets so you can compare across priority levels regardless of individual function characteristics.
- **Key Comparison** shows the headline number: how much slower LOW-priority calls are compared to HIGH-priority calls. When running against Dynamo with sensitivity hints (Step 4), you should see HIGH-priority calls get meaningfully faster compared to the NIM baseline, while LOW-priority calls may stay the same or get slightly slower (a good tradeoff).

## File Reference

| File | Description |
|------|-------------|
| `workflow.py` | 5 registered NAT functions + LangGraph orchestrator with parallel fan-out |
| `sensitivity_report.py` | CLI tool: `python -m latency_sensitivity_demo.sensitivity_report <trie.json>` |
| `compare_sensitivity_perf.py` | CLI tool: compare LLM call latency grouped by sensitivity level |
| `configs/config_profile.yml` | NIM profiling config — builds prediction trie with auto-sensitivity |
| `configs/config_with_trie.yml` | Dynamo runtime config — uses pre-built trie for hint injection |
| `data/customer_queries.json` | 8 sample customer support queries |

## Running Tests

```bash
pytest examples/dynamo_integration/latency_sensitivity_demo/tests/ -v
```
