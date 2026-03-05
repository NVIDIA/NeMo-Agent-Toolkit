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

The demo implements a customer support pipeline as a LangGraph `StateGraph` with seven nodes. Each node is a separately registered NeMo Agent Toolkit function, giving the profiler individual visibility into every LLM call.

The topology is designed to make priority-based scheduling effective: 4 parallel LOW-priority branches produce long outputs (~500 tokens each) that saturate GPU decode capacity, while 2 HIGH-priority nodes produce short outputs (~5 and ~20 tokens) that benefit from queue-jumping.

<!-- path-check-skip-begin -->
```
                        ┌─── research_context   (LOW,  ~500 tok) ──────┐
                        ├─── lookup_policy       (LOW,  ~500 tok) ──────┤
  classify_query ──────►├─── check_compliance    (LOW,  ~500 tok) ──────├──► draft_response ──► review_response
    (HIGH, ~5 tok)      └─── analyze_sentiment   (LOW,  ~500 tok) ──────┘     (MED, ~500 tok)    (HIGH, ~20 tok)
```

**Why this topology exercises all four sensitivity signals and demonstrates priority scheduling:**

| Node | What It Does | Topology Role | Output |
|------|-------------|---------------|--------|
| `classify_query` | Categorizes the query (billing, account, technical, general) with a single word | **Entry point.** Every downstream node depends on it. Fan-out of 6 calls. First position. | ~5 tokens |
| `research_context` | Comprehensive knowledge-base research | **Parallel sibling.** One of 4 concurrent LOW-priority branches. | ~500 tokens |
| `lookup_policy` | Detailed company policy reference | **Parallel sibling.** Long decode saturates GPU. | ~500 tokens |
| `check_compliance` | Regulatory compliance assessment | **Parallel sibling.** Additional GPU pressure. | ~500 tokens |
| `analyze_sentiment` | Customer sentiment and intent analysis | **Parallel sibling.** Completes the 4:1 LOW:HIGH ratio. | ~500 tokens |
| `draft_response` | Synthesizes all inputs into a customer response | **Join point.** Runs after all 4 parallel siblings. Mid-position. | ~500 tokens |
| `review_response` | QA approval/rejection with one-sentence reason | **Exit point.** Last node. Short output for fast approval. | ~20 tokens |

**Why this creates a measurable priority benefit at high concurrency:** With `max_concurrency: 16`, up to 64 concurrent LOW-priority decode requests saturate the GPU. When a new workflow's `classify_query` (5 tokens, HIGH priority) arrives, it either waits behind all those LOW decode requests (without priority) or jumps the queue (with priority). The 100x difference in output length between HIGH and LOW calls makes the queuing delay dramatic.

### How Sensitivity Scores Are Computed

The profiler's auto-sensitivity algorithm combines four weighted signals into a composite score per node, then normalizes across the workflow so the full 1–5 scale is used:

| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| **Position** (`w_position`) | 0.50 | U-shaped curve: first and last calls in the sequence score highest. Middle calls score lowest. Reflects that entry and exit nodes have the most impact on end-to-end latency. |
| **Critical path** (`w_critical`) | 0.35 | Fraction of total workflow wall-clock time spent in this call. Long-running calls that dominate execution time score higher. |
| **Fan-out** (`w_fanout`) | 0.15 | How many LLM calls remain after this one. The entry node (6 calls remaining) gets a boost; the exit node (0 remaining) does not. |
| **Parallel slack** (`w_parallel`) | 0.50 | _Penalty_ for parallel siblings that finish early and sit idle. If `research_context` takes 3s but `lookup_policy` takes 5s, `research_context` had 2s of slack — it could have been slower without affecting the workflow. This signal subtracts from the score. |

After computing raw weighted scores for each call in a trace, the algorithm **min-max normalizes** across all calls so the most-sensitive call maps to 5/5 and the least-sensitive maps to 1/5. This ensures clear differentiation regardless of absolute weight values.

**Expected output for this workflow:**

| Node | Score | Rationale |
|------|-------|-----------|
| `classify_query` | **5/5 HIGH** | First position + highest fan-out (6 calls follow). Everything depends on it. Short output (~5 tokens). |
| `review_response` | **5/5 HIGH** | Last position + high critical-path fraction. User is waiting. Short output (~20 tokens). |
| `draft_response` | **3/5 MEDIUM** | Sequential join point, moderate critical path, but mid-position dampens it. |
| `research_context` | **1-2/5 LOW** | Parallel slack penalty — one of 4 siblings, likely finishes before the slowest. |
| `lookup_policy` | **1-2/5 LOW** | Parallel slack penalty — mid-position, no fan-out boost. |
| `check_compliance` | **1-2/5 LOW** | Parallel slack penalty — same as other siblings. |
| `analyze_sentiment` | **1-2/5 LOW** | Parallel slack penalty — same as other siblings. |

### What Dynamo Does With These Scores

When the NeMo Agent Toolkit Dynamo LLM client (`_type: dynamo`) is configured with a prediction trie, it injects `nvext.agent_hints` into the OpenAI-compatible request body for each LLM call. These hints tell Dynamo's router about the call's latency sensitivity, expected output length, interarrival pattern, and request priority. Dynamo can use this to:

- **Priority-route** HIGH-sensitivity calls (classify, review) to dedicated workers for lowest latency
- **Batch-route** LOW-sensitivity calls (research, policy, compliance, sentiment) to shared workers where throughput is maximized
- **Optimize KV cache** allocation based on predicted output sequence length and cache TTL

## Prerequisites

- **Python 3.11+**
- **NeMo Agent Toolkit** installed with LangChain integration
- **NVIDIA API key** for NIM endpoint access (Step 1)
- **Dynamo backend** on a Linux GPU system (Steps 3–4). See the [Dynamo Setup Guide](../../../external/dynamo/README.md) for hardware and software requirements. 
- **Dynamo installed from source**: This example requires Dynamo to be installed from source. See the [installation guide](./INSTALL_LIBRARY.md) for instructions.

## Step 1: Profile the Workflow with Baseline Configuration

First, run the workflow against a Dynamo endpoint to collect profiler traces and build the prediction trie. 

### Step 1a: Start a Dynamo Endpoint

In a new terminal and directory, or on another machine, install Dynamo from source by following the [Dynamo installation guide](./INSTALL_LIBRARY.md).

Download the `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` model from Hugging Face by running the command below in the directory where you installed Dynamo.

```bash
export HF_TOKEN=hf_...
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Then deploy the baseline Dynamo deployment by following the steps below. 

#### A. Start infrastructure containers

```bash
cd dynamo/deploy
docker compose -f docker-compose.yml up -d --remove-orphans
```

This starts **etcd** (port 2379) and **NATS** (port 4222/8222).

#### B. (Optional) Start observability stack

```bash
docker compose -f docker-observability.yml up -d --remove-orphans
```

#### C. Run the Dynamo stack

Move `scripts/dynamo_stack.sh` into the directory where you installed Dynamo from source, then run it in the virtual environment. 

```bash
bash dynamo_stack.sh
```

#### D. Verify

From this terminal, verify you can reach the Dynamo endpoint assuming your port for inference is 8099 and available
on localhost:

```bash
curl http://localhost:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL_NAME",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```
### Step 1b: Run the Profiler to Build the Prediction Trie

```bash
# Install the example package
uv pip install -e ./examples/dynamo_integration/latency_sensitivity_demo

# Set your NVIDIA API key
export NVIDIA_API_KEY=nvapi-...

# Run profiling (8*8 queries, ~30 seconds each rep)
nat eval --reps 2 --config_file examples/dynamo_integration/latency_sensitivity_demo/src/latency_sensitivity_demo/configs/config_profile.yml
```

The profiler runs the full 7-node workflow for each query in the dataset, records per-node timing spans, and builds a prediction trie with auto-sensitivity scores. Output goes to:

```
examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/
├── prediction_trie.json         # The prediction trie with sensitivity scores
├── all_requests_profiler_traces.json  # Raw per-event profiler traces
├── standardized_data_all.csv    # Per-LLM-call timing metrics
├── inference_optimization.json  # Summary statistics
└── config_effective.yml         # Effective config used
```

## Step 2: View the Sensitivity Report

Use the included report tool to print a human-readable summary of the prediction trie. Pass `--csv` with the profiler CSV to also see measured latency and throughput for each function:

```bash
python -m latency_sensitivity_demo.sensitivity_report \
  examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/prediction_trie.json \
  --csv examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/standardized_data_all.csv
```

**Example output (with `--csv`):**

```
========================================================================================================
LATENCY SENSITIVITY REPORT
========================================================================================================

Path                                     Call#  Remaining  IAT (ms)   Tokens  Sensitivity            p50      p90     Mean    TPS
--------------------------------------------------------------------------------------------------------
root/<workflow>/classify_query               1        6.0        4.1       5  5/5 (HIGH)           200ms    250ms    210ms    24
root/<workflow>/review_response              1        0.0        0.0      20  5/5 (HIGH)           800ms    950ms    830ms    24
root/<workflow>/draft_response               1        1.0        2.5     500  3/5 (MEDIUM)        9000ms  12000ms   9500ms    53
root/<workflow>/research_context             1        2.0     1250.0     500  1/5 (LOW)           9000ms  11000ms   9200ms    54
root/<workflow>/lookup_policy                1        2.0        3.0     500  1/5 (LOW)           9500ms  12000ms   9800ms    51
root/<workflow>/check_compliance             1        2.0        3.0     500  1/5 (LOW)           9200ms  11500ms   9400ms    53
root/<workflow>/analyze_sentiment            1        2.0        3.0     500  1/5 (LOW)           9100ms  11200ms   9300ms    54

========================================================================================================
ROUTING RECOMMENDATIONS
========================================================================================================

  HIGH (4-5)   : Route to dedicated/priority workers for lowest latency
  MEDIUM (3)   : Standard routing — balance between latency and throughput
  LOW (1-2)    : Route to shared/batch workers — throughput over latency
```

**How to read the columns:**

| Column | Meaning |
|--------|---------|
| **Path** | Trie path: `root/<workflow>/<function_name>`. Each registered NeMo Agent Toolkit function gets its own node. |
| **Call#** | The LLM call index within this function (always 1 here since each function makes one call). |
| **Remaining** | Average number of LLM calls that follow this one in the workflow. `classify_query` = 6 (everything after it), `review_response` = 0 (last). |
| **IAT (ms)** | Mean inter-arrival time — milliseconds between this call ending and the next call starting. `research_context` shows ~1250ms because it finishes first and waits for `lookup_policy` to complete before `draft_response` can start. |
| **Tokens** | Mean output token count. `classify_query` outputs ~2 tokens (just a category name), while `review_response` outputs ~469 (a full customer response). |
| **Sensitivity** | The auto-computed score from 1/5 (LOW) to 5/5 (HIGH). |
| **p50 / p90 / Mean** | Measured latency percentiles and mean (shown when `--csv` is provided). Pairs LLM_START/LLM_END events by UUID to compute duration. |
| **TPS** | Mean tokens per second (completion tokens / duration). Shown when `--csv` is provided. |

## Step 3: Restart Dynamo Backend

Kill your previously running dynamo deployment by pressing `ctrl+c` in the terminal where you ran `dynamo_stack.sh`. Then copy `scripts/dynamo_stack_sensitivity.sh` into the directory where you installed Dynamo from source, and run it.

This ensures you have a fresh deployment ready to receive routing hints in Step 4.

```bash
bash dynamo_stack_sensitivity.sh
```

Verify the endpoint is responding:

```bash
curl -s http://localhost:8099/v1/models | python3 -m json.tool
```

## Step 4: Run With Latency Sensitivity Hints

Once Dynamo is running, update the prediction trie path in `config_with_trie.yml` and run the workflow. The Dynamo LLM client will inject per-request routing hints based on the profiled sensitivity scores.

```bash
# Run the workflow against Dynamo with sensitivity-aware routing
nat eval --reps 2 --config_file examples/dynamo_integration/latency_sensitivity_demo/src/latency_sensitivity_demo/configs/config_with_trie.yml
```

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
      "total_requests": 6,
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

**To measure the performance improvement**, use the included comparison script. It joins per-LLM-call timing data from the profiler CSV with sensitivity scores from the prediction trie, then groups calls by priority level.

Single-run analysis (shows that HIGH-priority calls are inherently faster or slower based on workflow position):

```bash
python -m latency_sensitivity_demo.compare_sensitivity_perf \
    --trie  examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/jobs/<job_id>/prediction_trie.json \
    --csv   examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/jobs/<job_id>/standardized_data_all.csv
```

Side-by-side comparison of NIM baseline vs Dynamo with sensitivity hints (shows the routing improvement):

```bash
python -m latency_sensitivity_demo.compare_sensitivity_perf \
    --trie   examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/prediction_trie.json \
    --csv    examples/dynamo_integration/latency_sensitivity_demo/outputs/profile/standardized_data_all.csv \
    --csv    examples/dynamo_integration/latency_sensitivity_demo/outputs/with_trie/standardized_data_all.csv \
    --labels "Dynamo" "Dynamo + sensitivity"
```

The comparison script normalizes by output tokens (`ms/tok`) so that runs producing different token counts are fairly compared. The `%` delta shows ms/tok change, not raw latency change.

**How to read the output:**

- **Per-Function Breakdown** shows each node sorted by sensitivity (highest first), with p50/p90/mean latency, ms/token, TPS, and sample count. In multi-run mode, a `%` delta on ms/tok shows the normalized improvement vs baseline (green = faster, red = slower).
- **Priority Group Summary** aggregates calls into HIGH/MEDIUM/LOW buckets with ms/tok so you can compare across priority levels regardless of individual function characteristics.
- **Priority Routing Effectiveness** is the key section: it shows within each run how much faster (per token) HIGH calls are vs LOW calls, and whether that ratio improved. When Dynamo's priority scheduling is working, the HIGH/LOW ratio should *increase* — HIGH calls get relatively faster while LOW calls absorb more queuing delay.

Use `--skip-warmup N` to drop the first N examples and remove cold-cache effects from the comparison.

## File Reference

| File | Description |
|------|-------------|
| `workflow.py` | 7 registered NeMo Agent Toolkit functions + LangGraph orchestrator with 4-way parallel fan-out |
| `sensitivity_report.py` | CLI tool: `python -m latency_sensitivity_demo.sensitivity_report <trie.json> [--csv <profiler.csv>]` |
| `compare_sensitivity_perf.py` | CLI tool: compare LLM call latency grouped by sensitivity level |
| `configs/config_profile.yml` | NIM profiling config — builds prediction trie with auto-sensitivity |
| `configs/config_with_trie.yml` | Dynamo runtime config — uses pre-built trie for hint injection |
| `data/customer_queries.json` | 8 sample customer support queries |

## Running Tests

```bash
pytest examples/dynamo_integration/latency_sensitivity_demo/tests/ -v
```

<!-- path-check-skip-end -->
