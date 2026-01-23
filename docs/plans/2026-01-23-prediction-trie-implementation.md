# Prediction Trie Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a prediction trie that aggregates LLM call patterns from profiler data and injects routing hints as Dynamo headers at runtime.

**Architecture:** The profiler builds a trie from execution traces where each node stores aggregated metrics (remaining calls, interarrival time, output tokens) by call index. At runtime, the Dynamo LLM client walks the trie to find the best match for the current execution context and injects predictions as HTTP headers.

**Tech Stack:** Python 3.11+, Pydantic v2, httpx event hooks, contextvars

---

## Task 1: Data Models

**Files:**
- Create: `src/nat/profiler/prediction_trie/data_models.py`
- Test: `tests/nat/profiler/prediction_trie/test_data_models.py`

### Step 1: Write the failing test for PredictionMetrics

```python
# tests/nat/profiler/prediction_trie/test_data_models.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.profiler.prediction_trie.data_models import PredictionMetrics


def test_prediction_metrics_creation():
    metrics = PredictionMetrics(sample_count=10, mean=5.0, p50=4.5, p90=8.0, p95=9.0)
    assert metrics.sample_count == 10
    assert metrics.mean == 5.0
    assert metrics.p50 == 4.5
    assert metrics.p90 == 8.0
    assert metrics.p95 == 9.0


def test_prediction_metrics_defaults():
    metrics = PredictionMetrics()
    assert metrics.sample_count == 0
    assert metrics.mean == 0.0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/nat/profiler/prediction_trie/test_data_models.py::test_prediction_metrics_creation -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'nat.profiler.prediction_trie'"

### Step 3: Create the prediction_trie package and data models

```python
# src/nat/profiler/prediction_trie/__init__.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode

__all__ = ["PredictionMetrics", "LLMCallPrediction", "PredictionTrieNode"]
```

```python
# src/nat/profiler/prediction_trie/data_models.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class PredictionMetrics(BaseModel):
    """Aggregated statistics for a single metric from profiler data."""

    sample_count: int = Field(default=0, description="Number of samples")
    mean: float = Field(default=0.0, description="Mean value")
    p50: float = Field(default=0.0, description="50th percentile (median)")
    p90: float = Field(default=0.0, description="90th percentile")
    p95: float = Field(default=0.0, description="95th percentile")


class LLMCallPrediction(BaseModel):
    """Predictions for an LLM call at a given position in the call hierarchy."""

    remaining_calls: PredictionMetrics = Field(
        default_factory=PredictionMetrics,
        description="How many more LLM calls are expected after this one",
    )
    interarrival_ms: PredictionMetrics = Field(
        default_factory=PredictionMetrics,
        description="Expected time in milliseconds until the next LLM call",
    )
    output_tokens: PredictionMetrics = Field(
        default_factory=PredictionMetrics,
        description="Expected output token count for this call",
    )


class PredictionTrieNode(BaseModel):
    """A node in the prediction trie representing a function in the call hierarchy."""

    name: str = Field(description="Function name at this level in the hierarchy")
    children: dict[str, PredictionTrieNode] = Field(
        default_factory=dict,
        description="Child nodes keyed by function name",
    )
    predictions_by_call_index: dict[int, LLMCallPrediction] = Field(
        default_factory=dict,
        description="Predictions keyed by call index (1-indexed)",
    )
    predictions_any_index: LLMCallPrediction | None = Field(
        default=None,
        description="Fallback predictions aggregated across all call indices",
    )


# Rebuild model to handle forward references
PredictionTrieNode.model_rebuild()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/nat/profiler/prediction_trie/test_data_models.py -v`
Expected: PASS

### Step 5: Add tests for LLMCallPrediction and PredictionTrieNode

Add to `tests/nat/profiler/prediction_trie/test_data_models.py`:

```python
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionTrieNode


def test_llm_call_prediction_creation():
    prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=5, mean=3.0, p50=3.0, p90=5.0, p95=6.0),
        interarrival_ms=PredictionMetrics(sample_count=5, mean=500.0, p50=450.0, p90=800.0, p95=900.0),
        output_tokens=PredictionMetrics(sample_count=5, mean=150.0, p50=140.0, p90=250.0, p95=300.0),
    )
    assert prediction.remaining_calls.mean == 3.0
    assert prediction.interarrival_ms.mean == 500.0
    assert prediction.output_tokens.mean == 150.0


def test_prediction_trie_node_creation():
    node = PredictionTrieNode(name="root")
    assert node.name == "root"
    assert node.children == {}
    assert node.predictions_by_call_index == {}
    assert node.predictions_any_index is None


def test_prediction_trie_node_with_children():
    child = PredictionTrieNode(name="react_agent")
    root = PredictionTrieNode(name="root", children={"react_agent": child})
    assert "react_agent" in root.children
    assert root.children["react_agent"].name == "react_agent"


def test_prediction_trie_node_with_predictions():
    prediction = LLMCallPrediction()
    node = PredictionTrieNode(
        name="agent",
        predictions_by_call_index={1: prediction, 2: prediction},
        predictions_any_index=prediction,
    )
    assert 1 in node.predictions_by_call_index
    assert 2 in node.predictions_by_call_index
    assert node.predictions_any_index is not None
```

### Step 6: Run all data model tests

Run: `pytest tests/nat/profiler/prediction_trie/test_data_models.py -v`
Expected: PASS (all tests)

### Step 7: Commit

```bash
git add src/nat/profiler/prediction_trie/ tests/nat/profiler/prediction_trie/
git commit --signoff -m "feat(profiler): add prediction trie data models

Add Pydantic models for the prediction trie:
- PredictionMetrics: aggregated stats (mean, p50, p90, p95)
- LLMCallPrediction: predictions for remaining calls, interarrival time, output tokens
- PredictionTrieNode: trie node with children and predictions by call index"
```

---

## Task 2: Metrics Accumulator

**Files:**
- Create: `src/nat/profiler/prediction_trie/metrics_accumulator.py`
- Test: `tests/nat/profiler/prediction_trie/test_metrics_accumulator.py`

### Step 1: Write the failing test for MetricsAccumulator

```python
# tests/nat/profiler/prediction_trie/test_metrics_accumulator.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.profiler.prediction_trie.metrics_accumulator import MetricsAccumulator


def test_accumulator_add_single_sample():
    acc = MetricsAccumulator()
    acc.add_sample(10.0)
    metrics = acc.compute_metrics()
    assert metrics.sample_count == 1
    assert metrics.mean == 10.0
    assert metrics.p50 == 10.0
    assert metrics.p90 == 10.0
    assert metrics.p95 == 10.0


def test_accumulator_add_multiple_samples():
    acc = MetricsAccumulator()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        acc.add_sample(v)
    metrics = acc.compute_metrics()
    assert metrics.sample_count == 10
    assert metrics.mean == 5.5
    assert metrics.p50 == 5.5  # median of 1-10
    assert metrics.p90 == 9.1  # 90th percentile
    assert metrics.p95 == 9.55  # 95th percentile


def test_accumulator_empty():
    acc = MetricsAccumulator()
    metrics = acc.compute_metrics()
    assert metrics.sample_count == 0
    assert metrics.mean == 0.0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/nat/profiler/prediction_trie/test_metrics_accumulator.py::test_accumulator_add_single_sample -v`
Expected: FAIL with "ModuleNotFoundError"

### Step 3: Implement MetricsAccumulator

```python
# src/nat/profiler/prediction_trie/metrics_accumulator.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from nat.profiler.prediction_trie.data_models import PredictionMetrics


class MetricsAccumulator:
    """Accumulates samples and computes aggregated statistics."""

    def __init__(self) -> None:
        self._samples: list[float] = []

    def add_sample(self, value: float) -> None:
        """Add a sample value to the accumulator."""
        self._samples.append(value)

    def compute_metrics(self) -> PredictionMetrics:
        """Compute aggregated metrics from accumulated samples."""
        if not self._samples:
            return PredictionMetrics()

        n = len(self._samples)
        mean_val = sum(self._samples) / n
        sorted_samples = sorted(self._samples)

        return PredictionMetrics(
            sample_count=n,
            mean=mean_val,
            p50=self._percentile(sorted_samples, 50),
            p90=self._percentile(sorted_samples, 90),
            p95=self._percentile(sorted_samples, 95),
        )

    @staticmethod
    def _percentile(sorted_data: list[float], pct: float) -> float:
        """Compute percentile using linear interpolation."""
        if not sorted_data:
            return 0.0
        if len(sorted_data) == 1:
            return sorted_data[0]
        k = (len(sorted_data) - 1) * (pct / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/nat/profiler/prediction_trie/test_metrics_accumulator.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/nat/profiler/prediction_trie/metrics_accumulator.py tests/nat/profiler/prediction_trie/test_metrics_accumulator.py
git commit --signoff -m "feat(profiler): add MetricsAccumulator for prediction trie

Accumulates sample values and computes aggregated statistics
(mean, p50, p90, p95) using linear interpolation for percentiles."
```

---

## Task 3: Trie Builder

**Files:**
- Create: `src/nat/profiler/prediction_trie/trie_builder.py`
- Test: `tests/nat/profiler/prediction_trie/test_trie_builder.py`

### Step 1: Write the failing test for TrieBuilder

```python
# tests/nat/profiler/prediction_trie/test_trie_builder.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.profiler.callbacks.token_usage_base_model import TokenUsageBaseModel
from nat.profiler.prediction_trie.trie_builder import PredictionTrieBuilder


@pytest.fixture(name="simple_trace")
def fixture_simple_trace() -> list[IntermediateStep]:
    """Create a simple trace with two LLM calls."""
    return [
        IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(
                function_id="workflow-1",
                function_name="my_workflow",
                parent_id=None,
                parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                event_timestamp=1000.0,
                UUID="llm-1",
            ),
        ),
        IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(
                function_id="workflow-1",
                function_name="my_workflow",
                parent_id=None,
                parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                event_timestamp=1001.0,
                span_event_timestamp=1000.0,
                UUID="llm-1",
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(completion_tokens=100),
                ),
            ),
        ),
        IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(
                function_id="workflow-1",
                function_name="my_workflow",
                parent_id=None,
                parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                event_timestamp=1002.0,
                UUID="llm-2",
            ),
        ),
        IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(
                function_id="workflow-1",
                function_name="my_workflow",
                parent_id=None,
                parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                event_timestamp=1003.0,
                span_event_timestamp=1002.0,
                UUID="llm-2",
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(completion_tokens=150),
                ),
            ),
        ),
    ]


def test_trie_builder_builds_from_single_trace(simple_trace):
    builder = PredictionTrieBuilder()
    builder.add_trace(simple_trace)
    trie = builder.build()

    assert trie.name == "root"
    assert "my_workflow" in trie.children

    workflow_node = trie.children["my_workflow"]
    # First LLM call: call_index=1, remaining=1
    assert 1 in workflow_node.predictions_by_call_index
    # Second LLM call: call_index=2, remaining=0
    assert 2 in workflow_node.predictions_by_call_index


def test_trie_builder_computes_remaining_calls(simple_trace):
    builder = PredictionTrieBuilder()
    builder.add_trace(simple_trace)
    trie = builder.build()

    workflow_node = trie.children["my_workflow"]
    # First call should predict 1 remaining call
    assert workflow_node.predictions_by_call_index[1].remaining_calls.mean == 1.0
    # Second call should predict 0 remaining calls
    assert workflow_node.predictions_by_call_index[2].remaining_calls.mean == 0.0


def test_trie_builder_computes_output_tokens(simple_trace):
    builder = PredictionTrieBuilder()
    builder.add_trace(simple_trace)
    trie = builder.build()

    workflow_node = trie.children["my_workflow"]
    # First call had 100 completion tokens
    assert workflow_node.predictions_by_call_index[1].output_tokens.mean == 100.0
    # Second call had 150 completion tokens
    assert workflow_node.predictions_by_call_index[2].output_tokens.mean == 150.0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/nat/profiler/prediction_trie/test_trie_builder.py::test_trie_builder_builds_from_single_trace -v`
Expected: FAIL with "ModuleNotFoundError"

### Step 3: Implement PredictionTrieBuilder

```python
# src/nat/profiler/prediction_trie/trie_builder.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.profiler.prediction_trie.metrics_accumulator import MetricsAccumulator


@dataclass
class LLMCallContext:
    """Context for a single LLM call extracted from a trace."""

    path: list[str]
    call_index: int
    remaining_calls: int
    time_to_next_ms: float | None
    output_tokens: int


@dataclass
class _NodeAccumulators:
    """Accumulators for a single trie node."""

    remaining_calls: dict[int, MetricsAccumulator] = field(default_factory=lambda: defaultdict(MetricsAccumulator))
    interarrival_ms: dict[int, MetricsAccumulator] = field(default_factory=lambda: defaultdict(MetricsAccumulator))
    output_tokens: dict[int, MetricsAccumulator] = field(default_factory=lambda: defaultdict(MetricsAccumulator))
    # For aggregated stats across all call indices
    all_remaining_calls: MetricsAccumulator = field(default_factory=MetricsAccumulator)
    all_interarrival_ms: MetricsAccumulator = field(default_factory=MetricsAccumulator)
    all_output_tokens: MetricsAccumulator = field(default_factory=MetricsAccumulator)


class PredictionTrieBuilder:
    """Builds a prediction trie from profiler execution traces."""

    def __init__(self) -> None:
        # Map from path tuple to accumulators
        self._node_accumulators: dict[tuple[str, ...], _NodeAccumulators] = defaultdict(_NodeAccumulators)

    def add_trace(self, steps: list[IntermediateStep]) -> None:
        """Process a single execution trace and update accumulators."""
        contexts = self._extract_llm_contexts(steps)
        for ctx in contexts:
            self._update_accumulators(ctx)

    def _extract_llm_contexts(self, steps: list[IntermediateStep]) -> list[LLMCallContext]:
        """Extract LLM call contexts from a trace."""
        # Sort steps by timestamp
        sorted_steps = sorted(steps, key=lambda s: s.event_timestamp)

        # Find all LLM_END events
        llm_ends: list[IntermediateStep] = []
        for step in sorted_steps:
            if step.event_type == IntermediateStepType.LLM_END:
                llm_ends.append(step)

        # Find all LLM_START events for interarrival time calculation
        llm_starts: list[IntermediateStep] = []
        for step in sorted_steps:
            if step.event_type == IntermediateStepType.LLM_START:
                llm_starts.append(step)

        # Track call index per parent function
        call_counts: dict[str, int] = defaultdict(int)
        contexts: list[LLMCallContext] = []

        for i, end_step in enumerate(llm_ends):
            # Build path from function ancestry
            path = self._build_path(end_step)

            # Determine call index within parent
            parent_key = end_step.function_ancestry.function_id
            call_counts[parent_key] += 1
            call_index = call_counts[parent_key]

            # Remaining calls in this trace
            remaining = len(llm_ends) - i - 1

            # Time to next LLM start (if any)
            time_to_next_ms: float | None = None
            if i + 1 < len(llm_starts):
                next_start_time = llm_starts[i + 1].event_timestamp if i + 1 < len(llm_starts) else None
                if next_start_time is not None:
                    time_to_next_ms = (next_start_time - end_step.event_timestamp) * 1000.0

            # Output tokens
            output_tokens = 0
            if end_step.usage_info and end_step.usage_info.token_usage:
                output_tokens = end_step.usage_info.token_usage.completion_tokens or 0

            contexts.append(
                LLMCallContext(
                    path=path,
                    call_index=call_index,
                    remaining_calls=remaining,
                    time_to_next_ms=time_to_next_ms,
                    output_tokens=output_tokens,
                )
            )

        return contexts

    def _build_path(self, step: IntermediateStep) -> list[str]:
        """Build the function path from ancestry."""
        path: list[str] = []
        ancestry = step.function_ancestry

        # Walk up the ancestry chain
        if ancestry.parent_name:
            path.append(ancestry.parent_name)
        path.append(ancestry.function_name)

        return path

    def _update_accumulators(self, ctx: LLMCallContext) -> None:
        """Update accumulators at every node along the path."""
        # Update root node
        root_key: tuple[str, ...] = ()
        self._add_to_accumulators(root_key, ctx)

        # Update each node along the path
        for i in range(len(ctx.path)):
            path_key = tuple(ctx.path[: i + 1])
            self._add_to_accumulators(path_key, ctx)

    def _add_to_accumulators(self, path_key: tuple[str, ...], ctx: LLMCallContext) -> None:
        """Add context data to accumulators for a specific path."""
        accs = self._node_accumulators[path_key]

        # By call index
        accs.remaining_calls[ctx.call_index].add_sample(float(ctx.remaining_calls))
        accs.output_tokens[ctx.call_index].add_sample(float(ctx.output_tokens))
        if ctx.time_to_next_ms is not None:
            accs.interarrival_ms[ctx.call_index].add_sample(ctx.time_to_next_ms)

        # Aggregated across all indices
        accs.all_remaining_calls.add_sample(float(ctx.remaining_calls))
        accs.all_output_tokens.add_sample(float(ctx.output_tokens))
        if ctx.time_to_next_ms is not None:
            accs.all_interarrival_ms.add_sample(ctx.time_to_next_ms)

    def build(self) -> PredictionTrieNode:
        """Build the final prediction trie from accumulated data."""
        root = PredictionTrieNode(name="root")

        for path_key, accs in self._node_accumulators.items():
            node = self._get_or_create_node(root, path_key)
            self._populate_node_predictions(node, accs)

        return root

    def _get_or_create_node(self, root: PredictionTrieNode, path_key: tuple[str, ...]) -> PredictionTrieNode:
        """Navigate to or create a node at the given path."""
        if not path_key:
            return root

        current = root
        for name in path_key:
            if name not in current.children:
                current.children[name] = PredictionTrieNode(name=name)
            current = current.children[name]
        return current

    def _populate_node_predictions(self, node: PredictionTrieNode, accs: _NodeAccumulators) -> None:
        """Populate a node with computed predictions from accumulators."""
        # Predictions by call index
        all_indices = set(accs.remaining_calls.keys()) | set(accs.interarrival_ms.keys()) | set(accs.output_tokens.keys())

        for idx in all_indices:
            prediction = LLMCallPrediction(
                remaining_calls=accs.remaining_calls[idx].compute_metrics(),
                interarrival_ms=accs.interarrival_ms[idx].compute_metrics(),
                output_tokens=accs.output_tokens[idx].compute_metrics(),
            )
            node.predictions_by_call_index[idx] = prediction

        # Aggregated predictions
        if accs.all_remaining_calls._samples:
            node.predictions_any_index = LLMCallPrediction(
                remaining_calls=accs.all_remaining_calls.compute_metrics(),
                interarrival_ms=accs.all_interarrival_ms.compute_metrics(),
                output_tokens=accs.all_output_tokens.compute_metrics(),
            )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/nat/profiler/prediction_trie/test_trie_builder.py -v`
Expected: PASS

### Step 5: Add test for interarrival time

Add to `tests/nat/profiler/prediction_trie/test_trie_builder.py`:

```python
def test_trie_builder_computes_interarrival_time(simple_trace):
    builder = PredictionTrieBuilder()
    builder.add_trace(simple_trace)
    trie = builder.build()

    workflow_node = trie.children["my_workflow"]
    # First call: next LLM starts at 1002.0, this call ends at 1001.0 -> 1000ms
    assert workflow_node.predictions_by_call_index[1].interarrival_ms.mean == 1000.0
```

### Step 6: Run all builder tests

Run: `pytest tests/nat/profiler/prediction_trie/test_trie_builder.py -v`
Expected: PASS

### Step 7: Commit

```bash
git add src/nat/profiler/prediction_trie/trie_builder.py tests/nat/profiler/prediction_trie/test_trie_builder.py
git commit --signoff -m "feat(profiler): add PredictionTrieBuilder

Builds prediction trie from profiler execution traces:
- Extracts LLM call contexts (path, call index, remaining, interarrival, output tokens)
- Aggregates metrics at every node along the path
- Computes stats by call index and aggregated fallback"
```

---

## Task 4: Trie Lookup

**Files:**
- Create: `src/nat/profiler/prediction_trie/trie_lookup.py`
- Test: `tests/nat/profiler/prediction_trie/test_trie_lookup.py`

### Step 1: Write the failing test for lookup

```python
# tests/nat/profiler/prediction_trie/test_trie_lookup.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup


@pytest.fixture(name="sample_trie")
def fixture_sample_trie() -> PredictionTrieNode:
    """Create a sample trie for testing lookups."""
    prediction_1 = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=3.0, p50=3.0, p90=4.0, p95=5.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )
    prediction_2 = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=2.0, p50=2.0, p90=3.0, p95=4.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=400.0, p50=380.0, p90=600.0, p95=700.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=200.0, p50=190.0, p90=280.0, p95=320.0),
    )
    aggregated = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=20, mean=2.5, p50=2.5, p90=3.5, p95=4.5),
        interarrival_ms=PredictionMetrics(sample_count=20, mean=450.0, p50=415.0, p90=650.0, p95=750.0),
        output_tokens=PredictionMetrics(sample_count=20, mean=175.0, p50=165.0, p90=240.0, p95=285.0),
    )

    agent_node = PredictionTrieNode(
        name="react_agent",
        predictions_by_call_index={1: prediction_1, 2: prediction_2},
        predictions_any_index=aggregated,
    )

    workflow_node = PredictionTrieNode(
        name="my_workflow",
        children={"react_agent": agent_node},
        predictions_any_index=aggregated,
    )

    root = PredictionTrieNode(
        name="root",
        children={"my_workflow": workflow_node},
        predictions_any_index=aggregated,
    )

    return root


def test_lookup_exact_match(sample_trie):
    lookup = PredictionTrieLookup(sample_trie)
    result = lookup.find(path=["my_workflow", "react_agent"], call_index=1)

    assert result is not None
    assert result.remaining_calls.mean == 3.0
    assert result.output_tokens.mean == 150.0


def test_lookup_partial_path_match(sample_trie):
    """When exact path doesn't exist, fall back to closest ancestor."""
    lookup = PredictionTrieLookup(sample_trie)
    # "unknown_tool" doesn't exist, should fall back to react_agent's aggregated
    result = lookup.find(path=["my_workflow", "react_agent", "unknown_tool"], call_index=1)

    assert result is not None
    # Should get react_agent's call_index=1 prediction
    assert result.remaining_calls.mean == 3.0


def test_lookup_unknown_call_index_fallback(sample_trie):
    """When call_index doesn't exist, fall back to aggregated."""
    lookup = PredictionTrieLookup(sample_trie)
    result = lookup.find(path=["my_workflow", "react_agent"], call_index=99)

    assert result is not None
    # Should fall back to predictions_any_index
    assert result.remaining_calls.mean == 2.5


def test_lookup_no_match_returns_root_aggregated(sample_trie):
    """When nothing matches, return root's aggregated."""
    lookup = PredictionTrieLookup(sample_trie)
    result = lookup.find(path=["completely_unknown"], call_index=1)

    assert result is not None
    # Should return root's aggregated prediction
    assert result.remaining_calls.mean == 2.5
```

### Step 2: Run test to verify it fails

Run: `pytest tests/nat/profiler/prediction_trie/test_trie_lookup.py::test_lookup_exact_match -v`
Expected: FAIL with "ModuleNotFoundError"

### Step 3: Implement PredictionTrieLookup

```python
# src/nat/profiler/prediction_trie/trie_lookup.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionTrieNode


class PredictionTrieLookup:
    """Looks up predictions in a prediction trie with graceful fallback."""

    def __init__(self, root: PredictionTrieNode) -> None:
        self._root = root

    def find(self, path: list[str], call_index: int) -> LLMCallPrediction | None:
        """
        Find the best matching prediction for the given path and call index.

        Walks the trie as far as possible along the path, then returns the deepest
        match. Falls back to aggregated predictions when exact call_index isn't found.

        Args:
            path: Function ancestry path (e.g., ["my_workflow", "react_agent"])
            call_index: The Nth LLM call within the current parent function

        Returns:
            Best matching prediction, or None if trie is empty
        """
        node = self._root
        deepest_match: LLMCallPrediction | None = None

        # Check root node first
        deepest_match = self._get_prediction(node, call_index) or deepest_match

        # Walk the trie as far as we can match
        for func_name in path:
            if func_name not in node.children:
                break
            node = node.children[func_name]
            # Update deepest match at each level
            match = self._get_prediction(node, call_index)
            if match is not None:
                deepest_match = match

        return deepest_match

    def _get_prediction(self, node: PredictionTrieNode, call_index: int) -> LLMCallPrediction | None:
        """Get prediction from node, preferring exact call_index, falling back to aggregated."""
        if call_index in node.predictions_by_call_index:
            return node.predictions_by_call_index[call_index]
        return node.predictions_any_index
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/nat/profiler/prediction_trie/test_trie_lookup.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/nat/profiler/prediction_trie/trie_lookup.py tests/nat/profiler/prediction_trie/test_trie_lookup.py
git commit --signoff -m "feat(profiler): add PredictionTrieLookup

Walks the trie to find best matching prediction:
- Exact path + exact call_index (most specific)
- Partial path + exact call_index
- Falls back to aggregated predictions when call_index not found"
```

---

## Task 5: Serialization

**Files:**
- Create: `src/nat/profiler/prediction_trie/serialization.py`
- Test: `tests/nat/profiler/prediction_trie/test_serialization.py`

### Step 1: Write the failing test for serialization

```python
# tests/nat/profiler/prediction_trie/test_serialization.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path

import pytest

from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.profiler.prediction_trie.serialization import load_prediction_trie
from nat.profiler.prediction_trie.serialization import save_prediction_trie


@pytest.fixture(name="sample_trie")
def fixture_sample_trie() -> PredictionTrieNode:
    """Create a sample trie for testing serialization."""
    prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=3.0, p50=3.0, p90=4.0, p95=5.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )

    child = PredictionTrieNode(
        name="react_agent",
        predictions_by_call_index={1: prediction},
        predictions_any_index=prediction,
    )

    root = PredictionTrieNode(
        name="root",
        children={"react_agent": child},
        predictions_any_index=prediction,
    )

    return root


def test_save_and_load_trie(sample_trie):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "prediction_trie.json"

        save_prediction_trie(sample_trie, path, workflow_name="test_workflow")

        loaded = load_prediction_trie(path)

        assert loaded.name == "root"
        assert "react_agent" in loaded.children
        assert loaded.children["react_agent"].predictions_by_call_index[1].remaining_calls.mean == 3.0


def test_saved_file_has_metadata(sample_trie):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "prediction_trie.json"

        save_prediction_trie(sample_trie, path, workflow_name="test_workflow")

        with open(path) as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        assert data["workflow_name"] == "test_workflow"
        assert "generated_at" in data
        assert "root" in data
```

### Step 2: Run test to verify it fails

Run: `pytest tests/nat/profiler/prediction_trie/test_serialization.py::test_save_and_load_trie -v`
Expected: FAIL with "ModuleNotFoundError"

### Step 3: Implement serialization functions

```python
# src/nat/profiler/prediction_trie/serialization.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode

CURRENT_VERSION = "1.0"


def save_prediction_trie(
    trie: PredictionTrieNode,
    path: Path,
    workflow_name: str = "unknown",
) -> None:
    """
    Save a prediction trie to a JSON file.

    Args:
        trie: The prediction trie root node
        path: Path to save the JSON file
        workflow_name: Name of the workflow this trie was built from
    """
    data = {
        "version": CURRENT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow_name": workflow_name,
        "root": _serialize_node(trie),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_prediction_trie(path: Path) -> PredictionTrieNode:
    """
    Load a prediction trie from a JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        The deserialized prediction trie root node
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return _deserialize_node(data["root"])


def _serialize_node(node: PredictionTrieNode) -> dict[str, Any]:
    """Serialize a trie node to a dictionary."""
    result: dict[str, Any] = {
        "name": node.name,
        "predictions_by_call_index": {
            str(k): v.model_dump() for k, v in node.predictions_by_call_index.items()
        },
        "predictions_any_index": node.predictions_any_index.model_dump() if node.predictions_any_index else None,
        "children": {k: _serialize_node(v) for k, v in node.children.items()},
    }
    return result


def _deserialize_node(data: dict[str, Any]) -> PredictionTrieNode:
    """Deserialize a dictionary to a trie node."""
    predictions_by_call_index: dict[int, LLMCallPrediction] = {}
    for k, v in data.get("predictions_by_call_index", {}).items():
        predictions_by_call_index[int(k)] = LLMCallPrediction(
            remaining_calls=PredictionMetrics(**v["remaining_calls"]),
            interarrival_ms=PredictionMetrics(**v["interarrival_ms"]),
            output_tokens=PredictionMetrics(**v["output_tokens"]),
        )

    predictions_any_index = None
    if data.get("predictions_any_index"):
        v = data["predictions_any_index"]
        predictions_any_index = LLMCallPrediction(
            remaining_calls=PredictionMetrics(**v["remaining_calls"]),
            interarrival_ms=PredictionMetrics(**v["interarrival_ms"]),
            output_tokens=PredictionMetrics(**v["output_tokens"]),
        )

    children: dict[str, PredictionTrieNode] = {}
    for k, v in data.get("children", {}).items():
        children[k] = _deserialize_node(v)

    return PredictionTrieNode(
        name=data["name"],
        predictions_by_call_index=predictions_by_call_index,
        predictions_any_index=predictions_any_index,
        children=children,
    )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/nat/profiler/prediction_trie/test_serialization.py -v`
Expected: PASS

### Step 5: Update __init__.py exports

```python
# src/nat/profiler/prediction_trie/__init__.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.profiler.prediction_trie.serialization import load_prediction_trie
from nat.profiler.prediction_trie.serialization import save_prediction_trie
from nat.profiler.prediction_trie.trie_builder import PredictionTrieBuilder
from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup

__all__ = [
    "LLMCallPrediction",
    "PredictionMetrics",
    "PredictionTrieBuilder",
    "PredictionTrieLookup",
    "PredictionTrieNode",
    "load_prediction_trie",
    "save_prediction_trie",
]
```

### Step 6: Commit

```bash
git add src/nat/profiler/prediction_trie/serialization.py src/nat/profiler/prediction_trie/__init__.py tests/nat/profiler/prediction_trie/test_serialization.py
git commit --signoff -m "feat(profiler): add prediction trie serialization

JSON serialization with metadata:
- version, generated_at, workflow_name
- Recursive node serialization/deserialization
- Handles predictions_by_call_index int keys"
```

---

## Task 6: Runtime Call Tracker

**Files:**
- Create: `src/nat/llm/prediction_context.py`
- Test: `tests/nat/llm/test_prediction_context.py`

### Step 1: Write the failing test for LLMCallTracker

```python
# tests/nat/llm/test_prediction_context.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.llm.prediction_context import LLMCallTracker
from nat.llm.prediction_context import get_call_tracker


def test_tracker_increment():
    tracker = LLMCallTracker()
    assert tracker.increment("func-1") == 1
    assert tracker.increment("func-1") == 2
    assert tracker.increment("func-2") == 1
    assert tracker.increment("func-1") == 3


def test_tracker_reset():
    tracker = LLMCallTracker()
    tracker.increment("func-1")
    tracker.increment("func-1")
    tracker.reset("func-1")
    assert tracker.increment("func-1") == 1


def test_tracker_context_variable():
    tracker1 = get_call_tracker()
    tracker1.increment("func-a")

    tracker2 = get_call_tracker()
    # Should be the same tracker in the same context
    assert tracker2.increment("func-a") == 2
```

### Step 2: Run test to verify it fails

Run: `pytest tests/nat/llm/test_prediction_context.py::test_tracker_increment -v`
Expected: FAIL with "ModuleNotFoundError"

### Step 3: Implement LLMCallTracker

```python
# src/nat/llm/prediction_context.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime context management for prediction trie lookups.

Provides tracking of LLM call indices per function invocation,
enabling accurate lookups in the prediction trie at runtime.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field


@dataclass
class LLMCallTracker:
    """Tracks LLM call counts per function invocation."""

    counts: dict[str, int] = field(default_factory=dict)

    def increment(self, parent_function_id: str) -> int:
        """
        Increment and return the call index for this parent.

        Args:
            parent_function_id: Unique ID of the parent function invocation

        Returns:
            The call index (1-indexed) for this LLM call within the parent
        """
        self.counts[parent_function_id] = self.counts.get(parent_function_id, 0) + 1
        return self.counts[parent_function_id]

    def reset(self, parent_function_id: str) -> None:
        """
        Reset call count when a function invocation completes.

        Args:
            parent_function_id: Unique ID of the parent function invocation
        """
        self.counts.pop(parent_function_id, None)


# Thread/async-safe context variable for the call tracker
_llm_call_tracker: ContextVar[LLMCallTracker] = ContextVar("llm_call_tracker")


def get_call_tracker() -> LLMCallTracker:
    """
    Get the LLMCallTracker for the current context.

    Creates a new tracker if one doesn't exist in the current context.

    Returns:
        The LLMCallTracker for this context
    """
    try:
        return _llm_call_tracker.get()
    except LookupError:
        tracker = LLMCallTracker()
        _llm_call_tracker.set(tracker)
        return tracker
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/nat/llm/test_prediction_context.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/nat/llm/prediction_context.py tests/nat/llm/test_prediction_context.py
git commit --signoff -m "feat(llm): add LLMCallTracker for runtime prediction lookups

Context variable-based tracking of LLM call indices per function
invocation. Thread/async-safe using contextvars."
```

---

## Task 7: Profiler Integration

**Files:**
- Modify: `src/nat/data_models/profiler.py`
- Modify: `src/nat/profiler/profile_runner.py`
- Test: `tests/nat/profiler/test_prediction_trie_integration.py`

### Step 1: Add prediction_trie config option

Update `src/nat/data_models/profiler.py`:

```python
# Add to ProfilerConfig class:
class PredictionTrieConfig(BaseModel):
    enable: bool = False
    output_filename: str = "prediction_trie.json"


class ProfilerConfig(BaseModel):

    base_metrics: bool = False
    token_usage_forecast: bool = False
    token_uniqueness_forecast: bool = False
    workflow_runtime_forecast: bool = False
    compute_llm_metrics: bool = False
    csv_exclude_io_text: bool = False
    prompt_caching_prefixes: PromptCachingConfig = PromptCachingConfig()
    bottleneck_analysis: BottleneckConfig = BottleneckConfig()
    concurrency_spike_analysis: ConcurrencySpikeConfig = ConcurrencySpikeConfig()
    prefix_span_analysis: PrefixSpanConfig = PrefixSpanConfig()
    prediction_trie: PredictionTrieConfig = PredictionTrieConfig()  # ADD THIS
```

### Step 2: Write failing test for profiler integration

```python
# tests/nat/profiler/test_prediction_trie_integration.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.profiler import PredictionTrieConfig
from nat.data_models.profiler import ProfilerConfig
from nat.profiler.callbacks.token_usage_base_model import TokenUsageBaseModel
from nat.profiler.prediction_trie import load_prediction_trie
from nat.profiler.profile_runner import ProfilerRunner


@pytest.fixture(name="sample_traces")
def fixture_sample_traces() -> list[list[IntermediateStep]]:
    """Create sample traces for testing profiler integration."""

    def make_trace() -> list[IntermediateStep]:
        return [
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(
                    function_id="workflow-1",
                    function_name="my_workflow",
                    parent_id=None,
                    parent_name=None,
                ),
                payload=IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_START,
                    event_timestamp=1000.0,
                    UUID="llm-1",
                ),
            ),
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(
                    function_id="workflow-1",
                    function_name="my_workflow",
                    parent_id=None,
                    parent_name=None,
                ),
                payload=IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_END,
                    event_timestamp=1001.0,
                    span_event_timestamp=1000.0,
                    UUID="llm-1",
                    usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=100)),
                ),
            ),
        ]

    return [make_trace(), make_trace()]


async def test_profiler_generates_prediction_trie(sample_traces):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        config = ProfilerConfig(
            base_metrics=True,
            prediction_trie=PredictionTrieConfig(enable=True),
        )

        runner = ProfilerRunner(config, output_dir)
        await runner.run(sample_traces)

        trie_path = output_dir / "prediction_trie.json"
        assert trie_path.exists()

        trie = load_prediction_trie(trie_path)
        assert trie.name == "root"
        assert "my_workflow" in trie.children
```

### Step 3: Run test to verify it fails

Run: `pytest tests/nat/profiler/test_prediction_trie_integration.py -v`
Expected: FAIL (prediction_trie.json not generated)

### Step 4: Update ProfilerRunner to generate prediction trie

Add to `src/nat/profiler/profile_runner.py` in the `run` method, after the existing analysis sections (around line 257):

```python
        # After prefix_span_analysis section, add:

        if self.profile_config.prediction_trie.enable:
            # ------------------------------------------------------------
            # Build and save prediction trie
            # ------------------------------------------------------------
            from nat.profiler.prediction_trie import PredictionTrieBuilder
            from nat.profiler.prediction_trie import save_prediction_trie

            logger.info("Building prediction trie from traces...")
            trie_builder = PredictionTrieBuilder()
            for trace in all_steps:
                trie_builder.add_trace(trace)

            prediction_trie = trie_builder.build()

            if self.write_output:
                trie_path = os.path.join(self.output_dir, self.profile_config.prediction_trie.output_filename)
                save_prediction_trie(prediction_trie, Path(trie_path), workflow_name="profiled_workflow")
                logger.info("Wrote prediction trie to: %s", trie_path)
```

### Step 5: Run test to verify it passes

Run: `pytest tests/nat/profiler/test_prediction_trie_integration.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/nat/data_models/profiler.py src/nat/profiler/profile_runner.py tests/nat/profiler/test_prediction_trie_integration.py
git commit --signoff -m "feat(profiler): integrate prediction trie generation

Add PredictionTrieConfig to ProfilerConfig with enable flag.
ProfilerRunner now builds and saves prediction_trie.json when enabled."
```

---

## Task 8: Dynamo Header Injection

**Files:**
- Modify: `src/nat/llm/dynamo_llm.py`
- Modify: `packages/nvidia_nat_langchain/src/nat/plugins/langchain/llm.py`
- Test: `tests/nat/llm/test_dynamo_prediction_headers.py`

### Step 1: Write failing test for header injection

```python
# tests/nat/llm/test_dynamo_prediction_headers.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.llm.dynamo_llm import create_httpx_client_with_prediction_headers
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics


async def test_prediction_headers_injected():
    """Test that prediction headers are injected into requests."""
    prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=3.0, p50=3.0, p90=4.0, p95=5.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )

    # Create a mock request to capture headers
    captured_headers = {}

    async def capture_hook(request):
        captured_headers.update(dict(request.headers))

    client = create_httpx_client_with_prediction_headers(
        prediction=prediction,
        prefix_template="test-{uuid}",
        total_requests=10,
        osl="MEDIUM",
        iat="LOW",
    )

    # Add our capture hook
    client.event_hooks["request"].append(capture_hook)

    # Make a test request (will fail, but headers will be captured)
    try:
        await client.post("http://localhost:1/test", json={})
    except Exception:
        pass

    assert "x-nat-remaining-llm-calls" in captured_headers
    assert captured_headers["x-nat-remaining-llm-calls"] == "3"
    assert "x-nat-interarrival-ms" in captured_headers
    assert captured_headers["x-nat-interarrival-ms"] == "500"
    assert "x-nat-expected-output-tokens" in captured_headers
    assert captured_headers["x-nat-expected-output-tokens"] == "200"  # p90 value

    await client.aclose()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/nat/llm/test_dynamo_prediction_headers.py -v`
Expected: FAIL with "cannot import name 'create_httpx_client_with_prediction_headers'"

### Step 3: Add prediction header injection to dynamo_llm.py

Add to `src/nat/llm/dynamo_llm.py`:

```python
# Add import at top:
from nat.profiler.prediction_trie.data_models import LLMCallPrediction


def _create_prediction_request_hook(
    prediction: LLMCallPrediction,
) -> Callable[["httpx.Request"], Coroutine[Any, Any, None]]:
    """
    Create an httpx event hook that injects prediction headers.

    Args:
        prediction: The prediction data to inject

    Returns:
        An async function suitable for use as an httpx event hook.
    """

    async def on_request(request):
        """Inject prediction headers before each request."""
        request.headers["x-nat-remaining-llm-calls"] = str(int(prediction.remaining_calls.mean))
        request.headers["x-nat-interarrival-ms"] = str(int(prediction.interarrival_ms.mean))
        request.headers["x-nat-expected-output-tokens"] = str(int(prediction.output_tokens.p90))

        logger.debug(
            "Injected prediction headers: remaining=%d, interarrival=%d, output_tokens=%d",
            int(prediction.remaining_calls.mean),
            int(prediction.interarrival_ms.mean),
            int(prediction.output_tokens.p90),
        )

    return on_request


def create_httpx_client_with_prediction_headers(
    prediction: LLMCallPrediction,
    prefix_template: str | None,
    total_requests: int,
    osl: str,
    iat: str,
    timeout: float = 600.0,
) -> "httpx.AsyncClient":
    """
    Create an httpx.AsyncClient with both Dynamo prefix and prediction headers.

    Args:
        prediction: Prediction data for this LLM call
        prefix_template: Template string with {uuid} placeholder
        total_requests: Expected number of requests for this prefix
        osl: Output sequence length hint (LOW/MEDIUM/HIGH)
        iat: Inter-arrival time hint (LOW/MEDIUM/HIGH)
        timeout: HTTP request timeout in seconds

    Returns:
        An httpx.AsyncClient configured with header injection.
    """
    import httpx

    hooks: list[Callable] = []

    # Add Dynamo prefix hook
    prefix_hook = _create_dynamo_request_hook(prefix_template, total_requests, osl, iat)
    hooks.append(prefix_hook)

    # Add prediction hook
    prediction_hook = _create_prediction_request_hook(prediction)
    hooks.append(prediction_hook)

    return httpx.AsyncClient(
        event_hooks={"request": hooks},
        timeout=httpx.Timeout(timeout),
    )
```

### Step 4: Run test to verify it passes

Run: `pytest tests/nat/llm/test_dynamo_prediction_headers.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/nat/llm/dynamo_llm.py tests/nat/llm/test_dynamo_prediction_headers.py
git commit --signoff -m "feat(llm): add prediction header injection to Dynamo client

Injects x-nat-remaining-llm-calls, x-nat-interarrival-ms, and
x-nat-expected-output-tokens headers for server routing optimization."
```

---

## Task 9: LangChain Integration with Trie Loading

**Files:**
- Modify: `src/nat/llm/dynamo_llm.py` (add config field)
- Modify: `packages/nvidia_nat_langchain/src/nat/plugins/langchain/llm.py`
- Test: `tests/nat/plugins/langchain/test_dynamo_prediction_trie.py`

### Step 1: Add prediction_trie_path to DynamoModelConfig

Update `src/nat/llm/dynamo_llm.py`:

```python
# Add to DynamoModelConfig class:
    prediction_trie_path: str | None = Field(
        default=None,
        description="Path to prediction_trie.json file. When set, predictions are "
        "looked up and injected as headers for each LLM call.",
    )

    # Update get_dynamo_field_names():
    @staticmethod
    def get_dynamo_field_names() -> frozenset[str]:
        return frozenset({
            "prefix_template",
            "prefix_total_requests",
            "prefix_osl",
            "prefix_iat",
            "request_timeout",
            "prediction_trie_path",  # ADD THIS
        })
```

### Step 2: Write test for trie-based header injection

```python
# tests/nat/plugins/langchain/test_dynamo_prediction_trie.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from nat.llm.dynamo_llm import DynamoModelConfig
from nat.profiler.prediction_trie import PredictionTrieNode
from nat.profiler.prediction_trie import save_prediction_trie
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics


@pytest.fixture(name="trie_file")
def fixture_trie_file() -> Path:
    """Create a temporary trie file for testing."""
    prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=3.0, p50=3.0, p90=4.0, p95=5.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )

    root = PredictionTrieNode(
        name="root",
        predictions_by_call_index={1: prediction},
        predictions_any_index=prediction,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = Path(f.name)

    save_prediction_trie(root, path)
    yield path
    path.unlink(missing_ok=True)


def test_dynamo_config_with_trie_path(trie_file):
    """Test that DynamoModelConfig accepts prediction_trie_path."""
    config = DynamoModelConfig(
        base_url="http://localhost:8000",
        model_name="test-model",
        api_key="test-key",
        prediction_trie_path=str(trie_file),
    )

    assert config.prediction_trie_path == str(trie_file)
    assert "prediction_trie_path" in DynamoModelConfig.get_dynamo_field_names()
```

### Step 3: Run test to verify config field works

Run: `pytest tests/nat/plugins/langchain/test_dynamo_prediction_trie.py -v`
Expected: PASS

### Step 4: Commit

```bash
git add src/nat/llm/dynamo_llm.py tests/nat/plugins/langchain/test_dynamo_prediction_trie.py
git commit --signoff -m "feat(llm): add prediction_trie_path config to DynamoModelConfig

Allows specifying a prediction_trie.json file path in workflow config.
When set, predictions are looked up and injected as headers."
```

---

## Task 10: End-to-End Integration Test

**Files:**
- Test: `tests/nat/profiler/test_prediction_trie_e2e.py`

### Step 1: Write end-to-end test

```python
# tests/nat/profiler/test_prediction_trie_e2e.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for prediction trie workflow."""

import tempfile
from pathlib import Path

import pytest

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.profiler import PredictionTrieConfig
from nat.data_models.profiler import ProfilerConfig
from nat.profiler.callbacks.token_usage_base_model import TokenUsageBaseModel
from nat.profiler.prediction_trie import PredictionTrieLookup
from nat.profiler.prediction_trie import load_prediction_trie
from nat.profiler.profile_runner import ProfilerRunner


def make_agent_trace(agent_name: str, num_llm_calls: int, base_timestamp: float) -> list[IntermediateStep]:
    """Create a trace with multiple LLM calls in an agent."""
    steps = []
    ts = base_timestamp

    for i in range(num_llm_calls):
        llm_uuid = f"llm-{agent_name}-{i}"

        # LLM_START
        steps.append(
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(
                    function_id=f"{agent_name}-1",
                    function_name=agent_name,
                    parent_id="workflow-1",
                    parent_name="my_workflow",
                ),
                payload=IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_START,
                    event_timestamp=ts,
                    UUID=llm_uuid,
                ),
            )
        )
        ts += 0.5

        # LLM_END
        completion_tokens = 100 + (i * 50)  # Vary tokens by position
        steps.append(
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(
                    function_id=f"{agent_name}-1",
                    function_name=agent_name,
                    parent_id="workflow-1",
                    parent_name="my_workflow",
                ),
                payload=IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_END,
                    event_timestamp=ts,
                    span_event_timestamp=ts - 0.5,
                    UUID=llm_uuid,
                    usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=completion_tokens)),
                ),
            )
        )
        ts += 0.5

    return steps


async def test_e2e_prediction_trie_workflow():
    """Test the complete flow: profiler -> trie -> lookup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create multiple traces with different agents
        traces = [
            make_agent_trace("react_agent", num_llm_calls=3, base_timestamp=1000.0),
            make_agent_trace("react_agent", num_llm_calls=3, base_timestamp=2000.0),
            make_agent_trace("tool_agent", num_llm_calls=2, base_timestamp=3000.0),
        ]

        # Run profiler
        config = ProfilerConfig(
            base_metrics=True,
            prediction_trie=PredictionTrieConfig(enable=True),
        )
        runner = ProfilerRunner(config, output_dir)
        await runner.run(traces)

        # Load trie
        trie_path = output_dir / "prediction_trie.json"
        assert trie_path.exists(), "Trie file should exist"

        trie = load_prediction_trie(trie_path)
        lookup = PredictionTrieLookup(trie)

        # Test lookups
        # react_agent has 3 LLM calls, so at call 1 there are 2 remaining
        result = lookup.find(path=["my_workflow", "react_agent"], call_index=1)
        assert result is not None
        assert result.remaining_calls.mean == 2.0  # 2 remaining after first call

        # At call 3 there are 0 remaining
        result = lookup.find(path=["my_workflow", "react_agent"], call_index=3)
        assert result is not None
        assert result.remaining_calls.mean == 0.0

        # tool_agent should have different stats
        result = lookup.find(path=["my_workflow", "tool_agent"], call_index=1)
        assert result is not None
        assert result.remaining_calls.mean == 1.0  # 1 remaining after first call

        # Unknown agent should fall back to aggregated
        result = lookup.find(path=["my_workflow", "unknown_agent"], call_index=1)
        assert result is not None  # Should still get a result from fallback
```

### Step 2: Run e2e test

Run: `pytest tests/nat/profiler/test_prediction_trie_e2e.py -v`
Expected: PASS

### Step 3: Commit

```bash
git add tests/nat/profiler/test_prediction_trie_e2e.py
git commit --signoff -m "test(profiler): add end-to-end prediction trie test

Validates complete flow: profiler traces -> trie generation -> lookup
with different agents and call indices."
```

---

## Summary

This plan implements the prediction trie feature in 10 tasks:

1. **Data Models** - Pydantic models for trie nodes and predictions
2. **Metrics Accumulator** - Helper for computing statistics
3. **Trie Builder** - Builds trie from profiler traces
4. **Trie Lookup** - Finds best matching prediction with fallback
5. **Serialization** - JSON save/load
6. **Runtime Call Tracker** - Context variable for tracking call indices
7. **Profiler Integration** - Config option and trie generation
8. **Dynamo Header Injection** - httpx hooks for prediction headers
9. **LangChain Integration** - Config field for trie path
10. **End-to-End Test** - Validates complete flow

Each task follows TDD: write failing test, implement, verify, commit.
