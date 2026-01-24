# Prediction Trie Example Config Design

## Overview

Create example configs and documentation demonstrating the two-phase Dynamo optimization workflow using prediction trie for dynamic header injection.

## Two-Phase Workflow

```
Phase 1: Profiling
┌─────────────────────────────────────────────────────────────┐
│  nat eval --config_file profile_rethinking_full_test.yml   │
│                            │                                │
│                            ▼                                │
│            outputs/rethinking_full_test_for_profiling/      │
│                   └── prediction_trie.json                  │
└─────────────────────────────────────────────────────────────┘

Phase 2: Run with Predictions
┌─────────────────────────────────────────────────────────────┐
│  nat eval --config_file run_with_prediction_trie.yml       │
│                            │                                │
│        Loads prediction_trie.json                          │
│                            │                                │
│        Injects dynamic headers per LLM call:               │
│          - x-nat-remaining-llm-calls                       │
│          - x-nat-interarrival-ms                           │
│          - x-nat-expected-output-tokens                    │
└─────────────────────────────────────────────────────────────┘
```

**Key difference from static headers:** Instead of guessing `prefix_total_requests=10`, the trie provides accurate per-call predictions based on function path and call index from profiled data.

## Deliverables

### 1. Update: profile_rethinking_full_test.yml

Add `prediction_trie` section to enable trie building:

```yaml
profiler:
  # ... existing config ...

  # NEW: Build prediction trie from profiled traces
  prediction_trie:
    enable: true
    output_filename: prediction_trie.json
```

### 2. New: run_with_prediction_trie.yml

Config that loads the trie and uses dynamic predictions:

```yaml
llms:
  dynamo_llm:
    _type: dynamo
    model_name: llama-3.3-70b
    base_url: http://localhost:8099/v1
    api_key: dummy
    temperature: 0.0
    max_tokens: 8192
    stop: ["Observation:", "\nThought:"]
    prefix_template: "react-benchmark-{uuid}"

    # Static headers as fallback
    prefix_total_requests: 10
    prefix_osl: MEDIUM
    prefix_iat: MEDIUM

    # NEW: Load prediction trie for dynamic per-call headers
    prediction_trie_path: ./examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/rethinking_full_test_for_profiling/<job_id>/prediction_trie.json

eval:
  general:
    output:
      dir: ./examples/dynamo_integration/react_benchmark_agent/outputs/dynamo_evals/prediction_trie_eval/

    profiler:
      compute_llm_metrics: true
      csv_exclude_io_text: true
```

### 3. New: README_PREDICTION_TRIE.md

Documentation for the two-phase workflow:

```markdown
# Prediction Trie Optimization for Dynamo

## Overview
Use profiled execution data to inject accurate per-call prediction headers
instead of static guesses.

## Quick Start

### Phase 1: Build the Prediction Trie
nat eval --config_file configs/profile_rethinking_full_test.yml

Output: outputs/dynamo_evals/rethinking_full_test_for_profiling/<job_id>/prediction_trie.json

### Phase 2: Run with Predictions
1. Update prediction_trie_path in run_with_prediction_trie.yml
2. Run: nat eval --config_file configs/run_with_prediction_trie.yml

## How It Works
- Phase 1 profiles the agent and builds a trie mapping (function_path, call_index) → predictions
- Phase 2 loads the trie and injects headers dynamically based on current execution context

## Headers Injected
| Header | Source | Description |
|--------|--------|-------------|
| x-nat-remaining-llm-calls | prediction.remaining_calls.mean | Expected remaining calls |
| x-nat-interarrival-ms | prediction.interarrival_ms.mean | Expected time to next call |
| x-nat-expected-output-tokens | prediction.output_tokens.p90 | Expected output tokens |

## Comparing Results
Run both static and prediction-based configs and compare avg_llm_latency metrics.
```

## Files Changed

| File | Type | Description |
|------|------|-------------|
| `examples/dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/configs/profile_rethinking_full_test.yml` | Modify | Add prediction_trie.enable: true |
| `examples/dynamo_integration/react_benchmark_agent/src/react_benchmark_agent/configs/run_with_prediction_trie.yml` | New | Config using prediction_trie_path |
| `examples/dynamo_integration/react_benchmark_agent/README_PREDICTION_TRIE.md` | New | Documentation for two-phase workflow |
