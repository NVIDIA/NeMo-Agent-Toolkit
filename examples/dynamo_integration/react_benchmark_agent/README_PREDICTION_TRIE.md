<!--
Copyright (c) 2025-2026, NVIDIA CORPORATION

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
<!-- path-check-skip-file -->

# Prediction Trie Optimization for Dynamo

Use profiled execution data to inject accurate per-call prediction headers instead of static guesses.

## Overview

The prediction trie enables **dynamic header injection** for Dynamo's KV-aware routing. Instead of using static values like `prefix_total_requests=10` for every call, the trie provides accurate predictions based on:
- **Function path**: Where in the agent hierarchy the call originates (e.g., `["react_workflow", "react_agent"]`)
- **Call index**: Which LLM call this is within the current function (1st, 2nd, 3rd, etc.)

This allows Dynamo's Thompson Sampling router to make better worker assignment decisions.

## Quick Start

### Phase 1: Build the Prediction Trie

Run profiling to collect execution data and build the trie:

```bash
nat eval --config_file configs/profile_rethinking_full_test.yml
```

**Output location:**
```
outputs/dynamo_evals/rethinking_full_test_for_profiling/<job_id>/prediction_trie.json
```

### Phase 2: Run with Predictions

1. **Update the trie path** in `configs/run_with_prediction_trie.yml`:
   ```yaml
   prediction_trie_path: ./examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/rethinking_full_test_for_profiling/<YOUR_JOB_ID>/prediction_trie.json
   ```

2. **Run with dynamic predictions:**
   ```bash
   nat eval --config_file configs/run_with_prediction_trie.yml
   ```

## How It Works

### During Profiling (Phase 1)

The profiler collects data for each LLM call:
- Function path at time of call
- Call index within the parent function
- Output tokens generated
- Time until the next LLM call
- Remaining LLM calls in the workflow

This data is aggregated into a trie structure with statistical summaries (mean, p50, p90, etc.) at each node.

### During Execution (Phase 2)

For each LLM request:
1. Read the current function path from context
2. Read the call index from the LLM call tracker
3. Look up the prediction in the trie
4. Inject headers into the HTTP request

### Fallback Chain

If an exact match isn't found, the trie lookup falls back:
1. Exact path + exact call index (most specific)
2. Exact path + any call index
3. Partial path + exact call index
4. Root aggregated stats (most general)

This ensures predictions are always available, even for novel execution paths.

## Headers Injected

| Header | Source | Description |
|--------|--------|-------------|
| `x-nat-remaining-llm-calls` | `prediction.remaining_calls.mean` | Expected remaining LLM calls in workflow |
| `x-nat-interarrival-ms` | `prediction.interarrival_ms.mean` | Expected milliseconds until next call |
| `x-nat-expected-output-tokens` | `prediction.output_tokens.p90` | Expected output tokens (90th percentile) |

## Comparing Results

To measure the impact of prediction trie vs static headers:

1. **Run with static headers** (baseline):
   ```bash
   nat eval --config_file configs/eval_config_rethinking_full_test.yml
   ```

2. **Run with prediction trie**:
   ```bash
   nat eval --config_file configs/run_with_prediction_trie.yml
   ```

3. **Compare metrics**:
   - `avg_llm_latency`: Lower is better
   - `avg_workflow_runtime`: Lower is better
   - Look for improvements in KV cache hit rates in Dynamo logs

## Configuration Reference

### Profiler Configuration (Phase 1)

Enable trie building in the profiler section:

```yaml
profiler:
  prediction_trie:
    enable: true
    output_filename: prediction_trie.json  # default
```

### LLM Configuration (Phase 2)

Add the trie path to your Dynamo LLM config:

```yaml
llms:
  dynamo_llm:
    _type: dynamo
    prefix_template: "react-benchmark-{uuid}"

    # Static fallbacks (used if trie lookup fails)
    prefix_total_requests: 10
    prefix_osl: MEDIUM
    prefix_iat: MEDIUM

    # Dynamic predictions from profiled data
    prediction_trie_path: /path/to/prediction_trie.json
```

## Troubleshooting

### "Prediction trie file not found"

The trie file doesn't exist at the configured path. Check:
- Did Phase 1 profiling complete successfully?
- Is the `job_id` in the path correct?
- Is the path relative to where you're running the command?

### "No prediction found for path"

This is normal - it means the trie is using fallback predictions. The trie will fall back to more general predictions when exact matches aren't found.

### Headers not being injected

Ensure:
- `prefix_template` is set (required for Dynamo hooks)
- `prediction_trie_path` points to a valid trie file
- You're using the `dynamo` LLM type

## Files

| File | Purpose |
|------|---------|
| `configs/profile_rethinking_full_test.yml` | Phase 1: Profile and build trie |
| `configs/run_with_prediction_trie.yml` | Phase 2: Run with dynamic predictions |
