# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import RootModel

# -----------------------------------------------------------
# Prompt Caching Data Models
# -----------------------------------------------------------


class PrefixInfo(BaseModel):
    """
    Stores metadata about a particular prefix observed in the LLM text input.
    """
    prefix: str
    prefix_length: int
    calls_count: int
    calls_percentage: float = Field(..., ge=0.0, le=1.0)


class FrameworkLLMPrefixData(BaseModel):
    """
    Metadata for a single (framework, llm_name) group,
    including total calls and all prefix statistics.
    """
    total_calls: int
    prefix_info: list[PrefixInfo]


class CommonPrefixesOutput(RootModel[dict[str, FrameworkLLMPrefixData]]):
    """
    A root model storing a dictionary keyed by '<framework>-<llm>',
    each value is a FrameworkLLMPrefixData instance.
    """

    def to_dict(self) -> dict[str, FrameworkLLMPrefixData]:
        """
        Return the raw dictionary of data, discarding the 'root' wrapper.
        """
        return self.root


# ----------------------------------------------------------------
# Token Uniqueness Models
# ----------------------------------------------------------------


class LLMUniquenessMetrics(BaseModel):
    """
    Stores p90, p95, and p99 for the 'new words' metric.
    """
    p90: float
    p95: float
    p99: float


class LLMUniquenessMetricsByLLM(RootModel[dict[str, LLMUniquenessMetrics]]):
    """
    A RootModel containing a dictionary where each key is an LLM name
    and each value is the LLMUniquenessMetrics for that LLM.
    """

    def to_dict(self) -> dict[str, Any]:
        # Return the raw dictionary for convenience
        return self.root


# ----------------------------------------------------------------
# Workflow Runtime Models
# ----------------------------------------------------------------


class WorkflowRuntimeMetrics(BaseModel):
    """
    Stores p90, p95, and p99 for workflow runtimes across all examples.
    """
    p90: float
    p95: float
    p99: float


# ----------------------------------------------------------------------
# Simple Bottleneck Detection Models
# ----------------------------------------------------------------------


class SimpleOperationStats(BaseModel):
    """
    Statistics for a particular operation name (LLM or tool),
    capturing concurrency, duration, usage, etc.
    """
    operation_type: str  # 'LLM' or 'TOOL'
    operation_name: str  # e.g., "llama-3" or "serpapi"
    usage_count: int  # how many times it appears
    avg_duration: float  # average duration
    p95_duration: float
    p99_duration: float
    max_concurrency: int  # maximum number of concurrent operations
    bottleneck_score: float = Field(..., description="Custom metric to rank bottlenecks.")


class SimpleBottleneckReport(BaseModel):
    """
    A container for all operation stats keyed by 'operation_type:operation_name',
    plus a textual summary that highlights top bottlenecks.
    """
    stats: dict[str, SimpleOperationStats]
    summary: str


# ----------------------------------------------------------------------
# Nested Bottleneck Models
# ----------------------------------------------------------------------


class CallNode(BaseModel):
    """
    A single call (LLM or TOOL) in a nested call tree.

    Attributes
    ----------
    uuid: str
        Unique ID tying together START/END events.
    operation_type: str
        e.g. 'LLM' or 'TOOL'.
    operation_name: str
        e.g. 'llama-3', 'bing-search', ...
    start_time: float
        Time when the call started.
    end_time: float
        Time when the call ended.
    duration: float
        end_time - start_time
    children: list["CallNode"]
        List of nested calls inside this call's time window.
    parent: "CallNode" | None
        Reference to the parent call in the tree (None if top-level).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    uuid: str
    operation_type: str
    operation_name: str
    start_time: float
    end_time: float
    duration: float = Field(..., description="end_time - start_time")
    children: list["CallNode"] = Field(default_factory=list)
    parent: "CallNode | None" = None

    def compute_self_time(self) -> float:
        """
        'Self time' = duration minus the union of child intervals.
        Overlapping child intervals are merged so we don't double-count them.
        """
        if not self.children:
            return self.duration

        intervals = [(c.start_time, c.end_time) for c in self.children]
        # Sort by start time
        intervals.sort(key=lambda x: x[0])

        merged = []
        cur_start, cur_end = intervals[0]
        for i in range(1, len(intervals)):
            s, e = intervals[i]
            if s <= cur_end:
                # Overlap
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))

        # Sum coverage, clamped to [start_time, end_time]
        covered = 0.0
        for (s, e) in merged:
            s_clamped = max(s, self.start_time)
            e_clamped = min(e, self.end_time)
            if e_clamped > s_clamped:
                covered += (e_clamped - s_clamped)

        return max(0.0, self.duration - covered)

    def compute_subtree_time(self) -> float:
        """
        Recursively compute the sum of self_time + children's subtree_time.
        This ensures no overlap double-counting among children.
        """
        total = self.compute_self_time()
        for c in self.children:
            total += c.compute_subtree_time()
        return total

    def __str__(self) -> str:
        return self._repr(0)

    def _repr(self, level: int) -> str:
        indent = "  " * level
        info = (f"{indent}- {self.operation_type} '{self.operation_name}' "
                f"(uuid={self.uuid}, start={self.start_time:.2f}, "
                f"end={self.end_time:.2f}, dur={self.duration:.2f})")
        child_strs = [child._repr(level + 1) for child in self.children]
        return "\n".join([info] + child_strs)


CallNode.model_rebuild()


class NodeMetrics(BaseModel):
    """
    Metrics for a single node:
      - self_time
      - subtree_time
      - concurrency_midpoint (optional)
      - bottleneck_score (example: subtree_time)
    """
    uuid: str
    operation_type: str
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    self_time: float
    subtree_time: float
    concurrency_midpoint: float | None = None
    bottleneck_score: float


class ConcurrencyDistribution(BaseModel):
    """
    Overall concurrency distribution info:
      - timeline_segments: List of (start, end, concurrency)
      - p50, p90, p95, p99 concurrency
    """
    timeline_segments: list[tuple[float, float, int]]
    p50: float
    p90: float
    p95: float
    p99: float


class NestedCallProfilingResult(BaseModel):
    """
    The final Pydantic model returned by 'multi_example_call_profiling'.

    Contains:
      - concurrency: ConcurrencyDistribution
      - node_metrics: dict[uuid, NodeMetrics]
      - top_bottlenecks: The top calls by bottleneck_score
      - textual_report: A multiline string summarizing everything
    """
    concurrency: ConcurrencyDistribution
    node_metrics: dict[str, NodeMetrics]
    top_bottlenecks: list[NodeMetrics]
    textual_report: str


# ----------------------------------------------------------------------
# Concurrency Spike Analysis Models
# ----------------------------------------------------------------------


class ConcurrencyCallNode(CallNode):
    """
    A single call in the nested call tree for one example.
    Each call is matched by a UUID with a `*_START` and `*_END` event.

    Because fields like prompt_tokens, completion_tokens, total_tokens
    may only exist at the END event, we store them only after seeing `*_END`".
    """

    example_number: int

    # Additional fields from END events
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    tool_outputs: str | None = None
    llm_text_output: str | None = None


ConcurrencyCallNode.model_rebuild()


class ConcurrencySpikeInfo(BaseModel):
    """
    Info about one concurrency spike interval:
    - start, end of the spike
    - concurrency level
    - list of calls that overlap
    """
    start_time: float
    end_time: float
    concurrency: int
    active_uuids: list[str] = Field(default_factory=list)


class ConcurrencyCorrelationStats(BaseModel):
    """
    Simple container for correlation / summarized stats of calls overlapping concurrency spikes.
    """
    avg_prompt_tokens: float
    avg_total_tokens: float


class ConcurrencyAnalysisResult(BaseModel):
    """
    The final Pydantic model returned by concurrency_spike_analysis(...).
    Contains:
    - concurrency_distribution: concurrency_level => total_time
    - p50_concurrency, p90_concurrency, p95_concurrency, p99_concurrency
    - spike_threshold, spike_intervals
    - correlation_stats
    - textual_report
    """
    concurrency_distribution: dict[int, float]
    p50_concurrency: float
    p90_concurrency: float
    p95_concurrency: float
    p99_concurrency: float

    spike_threshold: int
    spike_intervals: list[ConcurrencySpikeInfo]
    correlation_stats: ConcurrencyCorrelationStats

    average_latency_by_concurrency: dict[int, float]

    textual_report: str


# ----------------------------------------------------------------------
# PrefixSpan Analysis Models
# ----------------------------------------------------------------------


class PrefixCallNode(BaseModel):
    """
    Represents a single call in an example's workflow.
    - For LLM calls, we also store llm_text_input if available so we can incorporate it into the token.
    """
    uuid: str
    example_number: int
    operation_type: str  # "LLM" or "TOOL"
    operation_name: str  # e.g. "llama-3", "internet-search"
    start_time: float
    end_time: float
    duration: float
    llm_text_input: str | None = None


class FrequentPattern(BaseModel):
    """
    Frequent sub-sequence discovered by PrefixSpan, with coverage and average duration data.
    """
    pattern: list[str]  # e.g. ["LLM:llama-3|Hello world", "TOOL:internet-search"]
    frequency: int  # total occurrences across all examples
    coverage: float  # fraction of distinct examples that contain this pattern
    average_duration: float  # average sum of call durations for calls in that sub-sequence
    examples_containing: list[int]  # which examples have at least one occurrence


class PrefixSpanSubworkflowResult(BaseModel):
    """
    Pydantic model for the final outcome:
    - A list of frequent patterns
    - A textual summary
    """
    patterns: list[FrequentPattern]
    textual_report: str
