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

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.eval.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.plugins.eval.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.plugins.eval.profiler.prediction_trie.metrics_accumulator import MetricsAccumulator


@dataclass
class SensitivityConfig:
    """Configuration for auto-sensitivity scoring."""

    scale: int = 5
    w_critical: float = 0.5
    w_fanout: float = 0.3
    w_position: float = 0.2


@dataclass
class LLMCallContext:
    """Context for a single LLM call extracted from a trace."""

    path: list[str]
    call_index: int
    remaining_calls: int
    time_to_next_ms: float | None
    output_tokens: int
    call_duration_s: float = 0.0
    workflow_duration_s: float = 0.0
    sensitivity_score: float = 0.0


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
    # Sensitivity accumulators
    sensitivity: dict[int, MetricsAccumulator] = field(default_factory=lambda: defaultdict(MetricsAccumulator))
    all_sensitivity: MetricsAccumulator = field(default_factory=MetricsAccumulator)


class PredictionTrieBuilder:
    """Builds a prediction trie from profiler execution traces."""

    def __init__(self, sensitivity_config: SensitivityConfig | None = None) -> None:
        # Map from path tuple to accumulators
        self._node_accumulators: dict[tuple[str, ...], _NodeAccumulators] = defaultdict(_NodeAccumulators)
        self._sensitivity_config = sensitivity_config

    def add_trace(self, steps: list[IntermediateStep]) -> None:
        """Process a single execution trace and update accumulators."""
        contexts = self._extract_llm_contexts(steps)
        if self._sensitivity_config is not None:
            self._compute_sensitivity_scores(contexts)
        for ctx in contexts:
            self._update_accumulators(ctx)

    def _extract_llm_contexts(self, steps: list[IntermediateStep]) -> list[LLMCallContext]:
        """Extract LLM call contexts from a trace."""
        # Sort steps by timestamp
        sorted_steps = sorted(steps, key=lambda s: s.event_timestamp)

        # Workflow duration from first to last event
        workflow_duration_s = (
            sorted_steps[-1].event_timestamp - sorted_steps[0].event_timestamp
            if len(sorted_steps) >= 2 else 0.0
        )

        # Find all LLM_END events
        llm_ends = [s for s in sorted_steps if s.event_type == IntermediateStepType.LLM_END]

        # Find all LLM_START events for interarrival time calculation
        llm_starts = [s for s in sorted_steps if s.event_type == IntermediateStepType.LLM_START]

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
            current_end_time = end_step.event_timestamp
            # Find next LLM_START after this LLM_END
            for start_step in llm_starts:
                if start_step.event_timestamp > current_end_time:
                    time_to_next_ms = (start_step.event_timestamp - current_end_time) * 1000.0
                    break

            # Output tokens
            output_tokens = 0
            if end_step.usage_info and end_step.usage_info.token_usage:
                output_tokens = end_step.usage_info.token_usage.completion_tokens or 0

            # Call duration from span timestamps
            span_start = end_step.span_event_timestamp
            call_duration_s = (end_step.event_timestamp - span_start) if span_start is not None else 0.0

            contexts.append(
                LLMCallContext(
                    path=path,
                    call_index=call_index,
                    remaining_calls=remaining,
                    time_to_next_ms=time_to_next_ms,
                    output_tokens=output_tokens,
                    call_duration_s=call_duration_s,
                    workflow_duration_s=workflow_duration_s,
                ))

        return contexts

    def _compute_sensitivity_scores(self, contexts: list[LLMCallContext]) -> None:
        """Compute composite sensitivity scores for each call in the trace."""
        if not contexts:
            return

        total_calls = len(contexts)
        max_remaining = max(c.remaining_calls for c in contexts)
        cfg = self._sensitivity_config

        for i, ctx in enumerate(contexts):
            # Signal 1: Critical path weight
            if ctx.workflow_duration_s > 0:
                critical_path_weight = min(ctx.call_duration_s / ctx.workflow_duration_s, 1.0)
            else:
                critical_path_weight = 1.0

            # Signal 2: Fan-out score
            if max_remaining > 0:
                fanout_score = ctx.remaining_calls / max_remaining
            else:
                fanout_score = 0.0

            # Signal 3: Position score (U-shaped)
            if total_calls > 1:
                normalized_pos = i / (total_calls - 1)
                position_score = max(1.0 - normalized_pos, normalized_pos)
            else:
                position_score = 1.0

            ctx.sensitivity_score = (
                cfg.w_critical * critical_path_weight
                + cfg.w_fanout * fanout_score
                + cfg.w_position * position_score
            )

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
            path_key = tuple(ctx.path[:i + 1])
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

        # Sensitivity accumulators
        if self._sensitivity_config is not None:
            accs.sensitivity[ctx.call_index].add_sample(ctx.sensitivity_score)
            accs.all_sensitivity.add_sample(ctx.sensitivity_score)

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
        all_indices = set(accs.remaining_calls.keys()) | set(accs.interarrival_ms.keys()) | set(
            accs.output_tokens.keys())

        for idx in all_indices:
            prediction = LLMCallPrediction(
                remaining_calls=accs.remaining_calls[idx].compute_metrics(),
                interarrival_ms=accs.interarrival_ms[idx].compute_metrics(),
                output_tokens=accs.output_tokens[idx].compute_metrics(),
                latency_sensitivity=self._score_to_sensitivity(accs.sensitivity.get(idx)),
            )
            node.predictions_by_call_index[idx] = prediction

        # Aggregated predictions
        if accs.all_remaining_calls.has_samples():
            node.predictions_any_index = LLMCallPrediction(
                remaining_calls=accs.all_remaining_calls.compute_metrics(),
                interarrival_ms=accs.all_interarrival_ms.compute_metrics(),
                output_tokens=accs.all_output_tokens.compute_metrics(),
                latency_sensitivity=self._score_to_sensitivity(accs.all_sensitivity),
            )

    def _score_to_sensitivity(self, acc: MetricsAccumulator | None) -> int | None:
        """Convert accumulated sensitivity scores to a clamped integer."""
        if acc is None or not acc.has_samples() or self._sensitivity_config is None:
            return None
        scale = self._sensitivity_config.scale
        mean_score = acc.compute_metrics().mean
        return max(1, min(scale, round(mean_score * scale)))
