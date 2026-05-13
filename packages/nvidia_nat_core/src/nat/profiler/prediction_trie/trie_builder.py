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
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionTrieNode
from nat.profiler.prediction_trie.metrics_accumulator import MetricsAccumulator


@dataclass
class _SiblingSpan:
    """A paired START/END span used for parallel sibling overlap detection."""

    uuid: str
    parent_id: str
    start_time: float
    end_time: float
    is_llm: bool


@dataclass
class SensitivityConfig:
    """Configuration for auto-sensitivity scoring."""

    sensitivity_scale: int = 5
    w_critical: float = 0.5
    w_fanout: float = 0.3
    w_position: float = 0.2
    w_parallel: float = 0.0


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
    parallel_slack_ratio: float = 0.0
    sensitivity_score: float = 0.0
    span_start_time: float = 0.0
    span_end_time: float = 0.0


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
        workflow_duration_s = (sorted_steps[-1].event_timestamp -
                               sorted_steps[0].event_timestamp if len(sorted_steps) >= 2 else 0.0)

        # Find all LLM_END events
        llm_ends = [s for s in sorted_steps if s.event_type == IntermediateStepType.LLM_END]

        # Find all LLM_START events for interarrival time calculation
        llm_starts = [s for s in sorted_steps if s.event_type == IntermediateStepType.LLM_START]

        # Build sibling map only when w_parallel > 0
        sibling_map: dict[str, list[_SiblingSpan]] = {}
        if self._sensitivity_config is not None and self._sensitivity_config.w_parallel > 0:
            sibling_map = self._build_sibling_map(steps)

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

            # Parallel slack ratio
            # Look up siblings at the function level: use the function ancestry's parent_id
            # (the grandparent of the LLM call) so that sibling *functions* running in parallel
            # under the same orchestrator are compared, not just spans under the same function.
            parallel_slack = 0.0
            if sibling_map and span_start is not None:
                function_parent_id = end_step.function_ancestry.parent_id
                siblings = sibling_map.get(function_parent_id, []) if function_parent_id else []
                if not siblings:
                    siblings = sibling_map.get(end_step.parent_id, [])
                if siblings:
                    parallel_slack = self._compute_parallel_slack(end_step.UUID,
                                                                  span_start,
                                                                  end_step.event_timestamp,
                                                                  siblings)

            contexts.append(
                LLMCallContext(
                    path=path,
                    call_index=call_index,
                    remaining_calls=remaining,
                    time_to_next_ms=time_to_next_ms,
                    output_tokens=output_tokens,
                    call_duration_s=call_duration_s,
                    workflow_duration_s=workflow_duration_s,
                    parallel_slack_ratio=parallel_slack,
                    span_start_time=span_start if span_start is not None else 0.0,
                    span_end_time=end_step.event_timestamp,
                ))

        return contexts

    def _compute_sensitivity_scores(self, contexts: list[LLMCallContext]) -> None:
        """Compute composite sensitivity scores for each call in the trace.

        Parallel siblings are detected via temporal overlap and assigned the
        same logical position so that the U-shaped position signal and fan-out
        signal treat them as a single workflow step rather than spreading them
        across sequential indices.

        After computing raw weighted scores, the values are min-max normalized
        across all calls in the trace so the full 0–1 range is used.  This
        ensures the most-sensitive call in a trace maps to the top of the scale
        and the least-sensitive call maps to the bottom.
        """
        if not contexts:
            return

        cfg = self._sensitivity_config

        # --- Compute logical positions that collapse parallel siblings ---
        #
        # Calls that overlap in time are parallel siblings and should share
        # the same logical position.  We use a greedy sweep: any call whose
        # start time is before the current group's latest end time belongs
        # to the same parallel group.
        logical_positions = self._compute_logical_positions(contexts)
        num_logical_steps = max(logical_positions) + 1 if logical_positions else 1

        # Remaining calls should also reflect logical steps, not raw index.
        # For each call, remaining = (num_logical_steps - 1) - logical_pos.
        max_logical_remaining = num_logical_steps - 1

        # Count how many calls share each logical position.  A group of size > 1
        # is a parallel group.  Members get a parallel-group penalty that
        # reflects "this work is shared / parallelizable."
        from collections import Counter
        group_sizes = Counter(logical_positions)

        raw_scores: list[float] = []
        for i, ctx in enumerate(contexts):
            lpos = logical_positions[i]

            # Signal 1: Critical path weight
            if ctx.workflow_duration_s > 0:
                critical_path_weight = min(ctx.call_duration_s / ctx.workflow_duration_s, 1.0)
            else:
                critical_path_weight = 1.0

            # Signal 2: Fan-out score (based on logical remaining steps)
            logical_remaining = max_logical_remaining - lpos
            if max_logical_remaining > 0:
                fanout_score = logical_remaining / max_logical_remaining
            else:
                fanout_score = 0.0

            # Signal 3: Position score (U-shaped, based on logical position)
            if num_logical_steps > 1:
                normalized_pos = lpos / (num_logical_steps - 1)
                position_score = max(1.0 - normalized_pos, normalized_pos)
            else:
                position_score = 1.0

            # Parallel penalty: combines per-call slack ratio with a group
            # membership penalty.  Any call in a parallel group of size N > 1
            # gets a base penalty of (N-1)/N (e.g. 0.75 for a group of 4).
            # This is averaged with the individual slack ratio so that the
            # longest sibling (slack=0) still gets penalized for being in a
            # parallel group, while shorter siblings get penalized more.
            parallel_penalty = ctx.parallel_slack_ratio
            gs = group_sizes[lpos]
            if gs > 1:
                group_penalty = (gs - 1) / gs
                parallel_penalty = (parallel_penalty + group_penalty) / 2.0

            score = (cfg.w_critical * critical_path_weight + cfg.w_fanout * fanout_score +
                     cfg.w_position * position_score - cfg.w_parallel * parallel_penalty)
            raw_scores.append(score)

        # Min-max normalize across the trace so scores span the full 0–1 range
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max_score - min_score

        for ctx, raw in zip(contexts, raw_scores):
            if score_range > 0:
                ctx.sensitivity_score = (raw - min_score) / score_range
            else:
                ctx.sensitivity_score = 0.5

    @staticmethod
    def _compute_logical_positions(contexts: list[LLMCallContext]) -> list[int]:
        """Assign a logical position to each call, collapsing parallel siblings.

        Uses standard interval-merging: contexts are sorted by span start time,
        and any call whose start is before the current group's *latest* end time
        joins the group (capturing transitive overlaps).  The resulting group
        indices are then mapped back to the original LLM_END ordering.

        All calls in a parallel group share the same logical position index,
        so the U-shaped position signal and fan-out signal treat them as
        occupying a single workflow step.
        """
        if not contexts:
            return []

        n = len(contexts)

        # Sort indices by span start time for interval merging.
        sorted_indices = sorted(range(n), key=lambda i: contexts[i].span_start_time)

        # Merge overlapping intervals using max end time to capture transitive overlaps.
        group_assignments: list[int] = [0] * n
        current_group = 0
        group_max_end = contexts[sorted_indices[0]].span_end_time

        group_assignments[sorted_indices[0]] = current_group
        for k in range(1, n):
            idx = sorted_indices[k]
            if contexts[idx].span_start_time < group_max_end:
                # Overlaps with current group (possibly transitively).
                group_assignments[idx] = current_group
                group_max_end = max(group_max_end, contexts[idx].span_end_time)
            else:
                # No overlap → new sequential step.
                current_group += 1
                group_assignments[idx] = current_group
                group_max_end = contexts[idx].span_end_time

        return group_assignments

    @staticmethod
    def _build_sibling_map(steps: list[IntermediateStep]) -> dict[str, list[_SiblingSpan]]:
        """Pair START/END events by UUID, then group by parent_id.

        Only considers LLM, TOOL, FUNCTION, and SPAN event types.
        Returns a mapping from parent_id to all completed sibling spans under that parent.
        """
        _PAIRED_TYPES = {
            IntermediateStepType.LLM_START,
            IntermediateStepType.LLM_END,
            IntermediateStepType.TOOL_START,
            IntermediateStepType.TOOL_END,
            IntermediateStepType.FUNCTION_START,
            IntermediateStepType.FUNCTION_END,
            IntermediateStepType.SPAN_START,
            IntermediateStepType.SPAN_END,
        }
        _LLM_TYPES = {IntermediateStepType.LLM_START, IntermediateStepType.LLM_END}

        # Collect start/end timestamps keyed by UUID
        starts: dict[str, tuple[float, str, bool]] = {}  # uuid -> (timestamp, parent_id, is_llm)
        ends: dict[str, float] = {}  # uuid -> timestamp

        for step in steps:
            if step.event_type not in _PAIRED_TYPES:
                continue
            uuid = step.UUID
            is_start = step.event_type.value.endswith("_START")
            if is_start:
                starts[uuid] = (step.event_timestamp, step.parent_id, step.event_type in _LLM_TYPES)
            else:
                ends[uuid] = step.event_timestamp

        # Build completed spans grouped by parent_id
        sibling_map: dict[str, list[_SiblingSpan]] = defaultdict(list)
        for uuid, (start_time, parent_id, is_llm) in starts.items():
            if uuid in ends:
                sibling_map[parent_id].append(
                    _SiblingSpan(
                        uuid=uuid,
                        parent_id=parent_id,
                        start_time=start_time,
                        end_time=ends[uuid],
                        is_llm=is_llm,
                    ))

        return dict(sibling_map)

    @staticmethod
    def _compute_parallel_slack(llm_uuid: str, llm_start: float, llm_end: float, siblings: list[_SiblingSpan]) -> float:
        """Compute the parallel slack ratio for an LLM call relative to its siblings.

        slack = max(0, 1 - llm_duration / max_overlapping_sibling_duration)

        Returns 0.0 when the LLM call is the longest overlapping sibling, and
        approaches 1.0 when a much longer sibling runs in parallel.
        """
        llm_duration = llm_end - llm_start
        if llm_duration <= 0:
            return 0.0

        max_sibling_duration = 0.0
        for sib in siblings:
            if sib.uuid == llm_uuid:
                continue
            # Check for temporal overlap
            overlap_start = max(llm_start, sib.start_time)
            overlap_end = min(llm_end, sib.end_time)
            if overlap_start < overlap_end:
                sibling_duration = sib.end_time - sib.start_time
                max_sibling_duration = max(max_sibling_duration, sibling_duration)

        if max_sibling_duration <= 0:
            return 0.0

        return max(0.0, 1.0 - llm_duration / max_sibling_duration)

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
        scale = self._sensitivity_config.sensitivity_scale
        mean_score = acc.compute_metrics().mean
        return max(1, min(scale, round(mean_score * (scale - 1)) + 1))
