# Prediction Trie for Dynamo Inference Routing

**Date:** 2026-01-23
**Status:** Approved
**Author:** Design session with Claude

## Overview

A prediction system that provides the Dynamo inference server with expected workload characteristics for each LLM call—remaining calls, inter-arrival time, and expected output length—enabling smarter routing decisions.

## Problem

The Dynamo inference server can make better routing decisions if it knows:
- How many more LLM calls are expected in this workflow
- When the next LLM call will arrive
- How long the response will be

Currently, each LLM request arrives without this context. The server treats each call independently, missing optimization opportunities.

## Solution

Build a prediction trie from profiler data that captures LLM call patterns at multiple granularities. At runtime, inject predictions as HTTP headers on inference requests.

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROFILING PHASE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Run profiler on workflow with representative inputs                     │
│  2. Collect IntermediateStep traces with full ancestry                      │
│  3. Build PredictionTrie from LLM_END events                                │
│  4. Serialize to prediction_trie.json                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RUNTIME PHASE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. LLM client loads prediction_trie.json at startup                        │
│  2. On each LLM call:                                                       │
│     a. Get current function path from context                               │
│     b. Increment and get call_index from tracker                            │
│     c. Lookup prediction in trie                                            │
│     d. Inject headers into request                                          │
│  3. Dynamo server uses headers for routing decisions                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Structures

### Prediction Metrics

```python
@dataclass
class PredictionMetrics:
    """Stats for a single metric, pre-computed from profiler data."""
    sample_count: int
    mean: float
    p50: float
    p90: float
    p95: float
```

### LLM Call Prediction

```python
@dataclass
class LLMCallPrediction:
    """What we predict for an LLM call at a given position."""
    remaining_calls: PredictionMetrics      # How many more LLM calls expected
    interarrival_ms: PredictionMetrics      # Time until next LLM call
    output_tokens: PredictionMetrics        # Expected output length
```

### Prediction Trie Node

```python
@dataclass
class PredictionTrieNode:
    """A node in the prediction trie."""
    name: str                                           # Function name at this level
    children: dict[str, PredictionTrieNode]             # Child nodes by function name
    predictions_by_call_index: dict[int, LLMCallPrediction]  # Metrics keyed by call index
    predictions_any_index: LLMCallPrediction | None     # Fallback: aggregated across all indices
```

### Trie Structure Example

```
root
├── workflow (stats: all LLM calls in any workflow)
│   └── react_agent (stats: all LLM calls under react_agent)
│       ├── search_tool (stats: LLM calls under search_tool)
│       │   └── llm:1 (stats: first LLM call in search_tool)
│       │   └── llm:2 (stats: second LLM call)
│       └── calculator_tool
│           └── llm:1 (stats: first LLM call in calculator_tool)
```

## Building the Trie

### LLM Call Context Extraction

For each `LLM_END` event in a profiler trace:

```python
@dataclass
class LLMCallContext:
    path: list[str]           # ["workflow", "react_agent", "search_tool"]
    call_index: int           # Nth LLM call within the immediate parent
    remaining_calls: int      # How many LLM calls left in this workflow run
    time_to_next_ms: float    # Milliseconds until next LLM_START (or None if last)
    output_tokens: int        # Actual completion tokens
```

### Call Index Scoping

Call index is scoped to the immediate parent function:

```
workflow (run_id=1)
  └── react_agent (invocation_id=a1)
        ├── LLM call (call_index=1 within react_agent)
        ├── search_tool
        │     └── LLM call (call_index=1 within search_tool)
        └── LLM call (call_index=2 within react_agent)
```

### Trie Update Algorithm

For each LLM call, walk its ancestry path and update every node:

```python
def update_trie(root: PredictionTrieNode, ctx: LLMCallContext):
    node = root
    # Walk path, updating aggregates at each level
    for func_name in ctx.path:
        node.add_sample(ctx.call_index, ctx.remaining_calls, ctx.time_to_next_ms, ctx.output_tokens)
        node = node.children.setdefault(func_name, PredictionTrieNode(func_name))
    # Update leaf node too
    node.add_sample(ctx.call_index, ctx.remaining_calls, ctx.time_to_next_ms, ctx.output_tokens)
```

This means a single LLM call contributes samples to every ancestor node—giving us aggregated stats at every granularity automatically.

## Runtime Lookup

### Current Context

```python
@dataclass
class CurrentContext:
    path: list[str]    # Current function ancestry
    call_index: int    # Which LLM call this is within the immediate parent
```

### Lookup Algorithm

```python
def lookup(root: PredictionTrieNode, ctx: CurrentContext) -> LLMCallPrediction | None:
    node = root
    deepest_match = None

    # Walk the trie as far as we can match
    for func_name in ctx.path:
        # Capture this node as a potential match before descending
        prediction = node.predictions_by_call_index.get(ctx.call_index)
        if prediction is None:
            prediction = node.predictions_any_index
        if prediction is not None:
            deepest_match = prediction

        # Try to descend
        if func_name not in node.children:
            break
        node = node.children[func_name]

    # Check the final node we reached
    prediction = node.predictions_by_call_index.get(ctx.call_index)
    if prediction is None:
        prediction = node.predictions_any_index
    if prediction is not None:
        deepest_match = prediction

    return deepest_match
```

### Fallback Behavior

1. Try exact path + exact call index (most specific)
2. Try exact path + any call index
3. Try partial path + exact call index
4. Try partial path + any call index (most general)

Novel tool calls automatically get predictions based on agent-level stats.

## Runtime Call Index Tracking

```python
from contextvars import ContextVar

@dataclass
class LLMCallTracker:
    """Tracks LLM call counts per function invocation."""
    counts: dict[str, int] = field(default_factory=dict)

    def increment(self, parent_function_id: str) -> int:
        """Increment and return the call index for this parent."""
        self.counts[parent_function_id] = self.counts.get(parent_function_id, 0) + 1
        return self.counts[parent_function_id]

    def reset(self, parent_function_id: str):
        """Reset when a function invocation completes."""
        self.counts.pop(parent_function_id, None)

_llm_call_tracker: ContextVar[LLMCallTracker] = ContextVar('llm_call_tracker')
```

## Header Injection

### Headers

```
X-NAT-Remaining-LLM-Calls: 3
X-NAT-Interarrival-Ms: 450
X-NAT-Expected-Output-Tokens: 256
X-NAT-Prediction-Confidence: 0.85
```

### Integration Point

```python
class DynamoLangChainLLM(BaseLLM):
    prediction_trie: PredictionTrie | None = None

    def _call(self, prompt: str, **kwargs) -> str:
        headers = self._get_base_headers()

        if self.prediction_trie is not None:
            ctx = self._get_current_context()
            prediction = self.prediction_trie.lookup(ctx)
            if prediction:
                headers["X-NAT-Remaining-LLM-Calls"] = str(prediction.remaining_calls.mean)
                headers["X-NAT-Interarrival-Ms"] = str(prediction.interarrival_ms.mean)
                headers["X-NAT-Expected-Output-Tokens"] = str(prediction.output_tokens.p90)

        return self._make_request(prompt, headers=headers, **kwargs)
```

### Configuration

```yaml
llms:
  my_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    prediction_trie_path: ./profiler_output/prediction_trie.json
```

## Serialization

### JSON Format

```json
{
  "version": "1.0",
  "generated_at": "2026-01-23T10:30:00Z",
  "workflow_name": "my_workflow",
  "sample_count": 150,
  "root": {
    "name": "root",
    "predictions_by_call_index": {
      "1": {
        "remaining_calls": {"sample_count": 150, "mean": 4.2, "p50": 4, "p90": 6, "p95": 7},
        "interarrival_ms": {"sample_count": 150, "mean": 520, "p50": 480, "p90": 890, "p95": 1100},
        "output_tokens": {"sample_count": 150, "mean": 185, "p50": 160, "p90": 320, "p95": 410}
      }
    },
    "predictions_any_index": { ... },
    "children": {
      "react_agent": { ... }
    }
  }
}
```

### Output Files

```
profiler_output/
├── all_requests_profiler_traces.json
├── standardized_data_all.csv
├── inference_optimization.json
├── prediction_trie.json          # NEW
└── prediction_trie_summary.txt   # NEW: human-readable summary
```

## File Organization

```
src/nat/profiler/
├── prediction_trie/
│   ├── __init__.py
│   ├── data_models.py        # PredictionTrieNode, LLMCallPrediction, PredictionMetrics
│   ├── trie_builder.py       # Build trie from profiler traces
│   ├── trie_lookup.py        # Lookup algorithm
│   └── serialization.py      # JSON load/save

src/nat/llm/
├── prediction_context.py     # LLMCallTracker, context variable, path extraction

packages/nvidia_nat_langchain/src/nat/plugins/langchain/
├── llm.py                    # Modify to inject headers
```

## Profiler Configuration

```yaml
profiler:
  base_metrics: true
  prediction_trie: true
  prediction_trie_output: ./prediction_trie.json
```

## Implementation Sequence

1. **Data models** - `PredictionTrieNode`, `LLMCallPrediction`, `PredictionMetrics`
2. **Trie builder** - Parse profiler traces, extract LLM call contexts, build trie
3. **Serialization** - JSON save/load for the trie
4. **Trie lookup** - Walk trie, return deepest match with fallback
5. **Runtime tracking** - `LLMCallTracker` context variable, integrate with existing ancestry tracking
6. **Header injection** - Modify `dynamo_langchain` LLM client to inject headers
7. **Profiler integration** - Add config option, wire trie builder into profiler output
8. **Tests** - Unit tests for trie operations, integration test with sample traces

## Out of Scope

- Concurrency/parallelism tracking
- Input token bucketing for lookup
- Real-time trie updates during runtime
- Multiple trie versions/A-B testing
