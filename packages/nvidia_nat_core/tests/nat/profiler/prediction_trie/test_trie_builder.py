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

import pytest

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.token_usage import TokenUsageBaseModel
from nat.profiler.prediction_trie.trie_builder import LLMCallContext
from nat.profiler.prediction_trie.trie_builder import PredictionTrieBuilder
from nat.profiler.prediction_trie.trie_builder import SensitivityConfig


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
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=100), ),
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
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=150), ),
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


def test_trie_builder_computes_interarrival_time(simple_trace):
    builder = PredictionTrieBuilder()
    builder.add_trace(simple_trace)
    trie = builder.build()

    workflow_node = trie.children["my_workflow"]
    # First call: next LLM starts at 1002.0, this call ends at 1001.0 -> 1000ms
    assert workflow_node.predictions_by_call_index[1].interarrival_ms.mean == 1000.0


def test_extract_contexts_include_call_duration(simple_trace):
    """LLMCallContext should include call_duration_s computed from span timestamps."""
    builder = PredictionTrieBuilder()
    contexts = builder._extract_llm_contexts(simple_trace)

    # First call: LLM_START=1000.0, LLM_END=1001.0 -> duration=1.0s
    assert contexts[0].call_duration_s == pytest.approx(1.0)
    # Second call: LLM_START=1002.0, LLM_END=1003.0 -> duration=1.0s
    assert contexts[1].call_duration_s == pytest.approx(1.0)


def test_extract_contexts_include_workflow_duration(simple_trace):
    """LLMCallContext should include workflow_duration_s (first to last event)."""
    builder = PredictionTrieBuilder()
    contexts = builder._extract_llm_contexts(simple_trace)

    # Workflow: first event=1000.0, last event=1003.0 -> 3.0s
    assert contexts[0].workflow_duration_s == pytest.approx(3.0)
    assert contexts[1].workflow_duration_s == pytest.approx(3.0)


def test_sensitivity_not_computed_without_config(simple_trace):
    """Without SensitivityConfig, latency_sensitivity should be None."""
    builder = PredictionTrieBuilder()
    builder.add_trace(simple_trace)
    trie = builder.build()

    node = trie.children["my_workflow"]
    assert node.predictions_by_call_index[1].latency_sensitivity is None
    assert node.predictions_by_call_index[2].latency_sensitivity is None


def test_sensitivity_computed_with_config(simple_trace):
    """With SensitivityConfig, latency_sensitivity should be an integer in [1, scale]."""
    config = SensitivityConfig(sensitivity_scale=5, w_critical=0.5, w_fanout=0.3, w_position=0.2)
    builder = PredictionTrieBuilder(sensitivity_config=config)
    builder.add_trace(simple_trace)
    trie = builder.build()

    node = trie.children["my_workflow"]
    s1 = node.predictions_by_call_index[1].latency_sensitivity
    s2 = node.predictions_by_call_index[2].latency_sensitivity
    assert s1 is not None
    assert s2 is not None
    assert 1 <= s1 <= 5
    assert 1 <= s2 <= 5


def test_sensitivity_first_call_higher_than_last_call(simple_trace):
    """First call has higher fan-out (remaining=1 vs 0) and is first position,
    so it should get equal or higher sensitivity than the last call."""
    config = SensitivityConfig(sensitivity_scale=5, w_critical=0.5, w_fanout=0.3, w_position=0.2)
    builder = PredictionTrieBuilder(sensitivity_config=config)
    builder.add_trace(simple_trace)
    trie = builder.build()

    node = trie.children["my_workflow"]
    s1 = node.predictions_by_call_index[1].latency_sensitivity
    s2 = node.predictions_by_call_index[2].latency_sensitivity
    assert s1 >= s2


def test_sensitivity_respects_scale():
    """Sensitivity should be clamped to [1, scale] regardless of raw score."""
    trace = [
        IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(
                function_id="wf-1",
                function_name="wf",
                parent_id=None,
                parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                event_timestamp=0.0,
                UUID="a",
            ),
        ),
        IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(
                function_id="wf-1",
                function_name="wf",
                parent_id=None,
                parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                event_timestamp=10.0,
                span_event_timestamp=0.0,
                UUID="a",
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=50)),
            ),
        ),
    ]
    config = SensitivityConfig(sensitivity_scale=3)
    builder = PredictionTrieBuilder(sensitivity_config=config)
    builder.add_trace(trace)
    trie = builder.build()

    node = trie.children["wf"]
    s = node.predictions_by_call_index[1].latency_sensitivity
    assert 1 <= s <= 3


def test_sensitivity_aggregated_across_traces(simple_trace):
    """Multiple traces should be averaged for sensitivity scoring."""
    config = SensitivityConfig(sensitivity_scale=5)
    builder = PredictionTrieBuilder(sensitivity_config=config)
    builder.add_trace(simple_trace)
    builder.add_trace(simple_trace)
    trie = builder.build()

    node = trie.children["my_workflow"]
    s1 = node.predictions_by_call_index[1].latency_sensitivity
    assert s1 is not None
    assert 1 <= s1 <= 5


def test_sensitivity_on_aggregated_any_index(simple_trace):
    """predictions_any_index should also have latency_sensitivity."""
    config = SensitivityConfig(sensitivity_scale=5)
    builder = PredictionTrieBuilder(sensitivity_config=config)
    builder.add_trace(simple_trace)
    trie = builder.build()

    node = trie.children["my_workflow"]
    assert node.predictions_any_index is not None
    assert node.predictions_any_index.latency_sensitivity is not None
    assert 1 <= node.predictions_any_index.latency_sensitivity <= 5


# ---------------------------------------------------------------------------
# Parallel slack tests
# ---------------------------------------------------------------------------


@pytest.fixture(name="parallel_trace")
def fixture_parallel_trace() -> list[IntermediateStep]:
    """Create a trace with a short LLM call running in parallel with a longer TOOL call.

    Parent function: func-1 (t=0.0 - 6.0)
    LLM call: llm-p1 (t=1.0 - 2.0, duration=1s)
    TOOL call: tool-p1 (t=0.5 - 5.5, duration=5s) — the parallel sibling
    Expected slack = 1 - 1/5 = 0.8
    """
    return [
        # Parent FUNCTION_START
        IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(
                function_id="wf-1",
                function_name="my_workflow",
                parent_id=None,
                parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.FUNCTION_START,
                event_timestamp=0.0,
                UUID="func-1",
            ),
        ),
        # TOOL_START (long sibling)
        IntermediateStep(
            parent_id="func-1",
            function_ancestry=InvocationNode(
                function_id="wf-1",
                function_name="my_workflow",
                parent_id=None,
                parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_START,
                event_timestamp=0.5,
                UUID="tool-p1",
            ),
        ),
        # LLM_START (short call)
        IntermediateStep(
            parent_id="func-1",
            function_ancestry=InvocationNode(
                function_id="wf-1",
                function_name="my_workflow",
                parent_id=None,
                parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                event_timestamp=1.0,
                UUID="llm-p1",
            ),
        ),
        # LLM_END
        IntermediateStep(
            parent_id="func-1",
            function_ancestry=InvocationNode(
                function_id="wf-1",
                function_name="my_workflow",
                parent_id=None,
                parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                event_timestamp=2.0,
                span_event_timestamp=1.0,
                UUID="llm-p1",
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=50)),
            ),
        ),
        # TOOL_END (long sibling finishes later)
        IntermediateStep(
            parent_id="func-1",
            function_ancestry=InvocationNode(
                function_id="wf-1",
                function_name="my_workflow",
                parent_id=None,
                parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_END,
                event_timestamp=5.5,
                span_event_timestamp=0.5,
                UUID="tool-p1",
            ),
        ),
        # Parent FUNCTION_END
        IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(
                function_id="wf-1",
                function_name="my_workflow",
                parent_id=None,
                parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.FUNCTION_END,
                event_timestamp=6.0,
                span_event_timestamp=0.0,
                UUID="func-1",
            ),
        ),
    ]


def test_parallel_slack_detected(parallel_trace):
    """LLM call (1s) with a 5s overlapping sibling should have slack ~ 0.8."""
    config = SensitivityConfig(w_parallel=0.3)
    builder = PredictionTrieBuilder(sensitivity_config=config)
    contexts = builder._extract_llm_contexts(parallel_trace)

    assert len(contexts) == 1
    assert contexts[0].parallel_slack_ratio == pytest.approx(0.8)


def test_parallel_slack_zero_when_no_siblings(simple_trace):
    """In simple_trace, LLM calls have no overlapping non-LLM siblings, so slack = 0.0.

    Note: LLM calls under the same parent_id='root' can still be siblings but they are
    sequential (non-overlapping), so no overlap is detected.
    """
    config = SensitivityConfig(w_parallel=0.3)
    builder = PredictionTrieBuilder(sensitivity_config=config)
    contexts = builder._extract_llm_contexts(simple_trace)

    for ctx in contexts:
        assert ctx.parallel_slack_ratio == pytest.approx(0.0)


def test_parallel_slack_zero_when_llm_is_longest():
    """When the LLM call is longer than its sibling, slack should be 0.0."""
    trace = [
        # Parent
        IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(function_id="wf-1", function_name="wf", parent_id=None, parent_name=None),
            payload=IntermediateStepPayload(event_type=IntermediateStepType.FUNCTION_START,
                                            event_timestamp=0.0,
                                            UUID="func-1"),
        ),
        # Short TOOL sibling (1s)
        IntermediateStep(
            parent_id="func-1",
            function_ancestry=InvocationNode(function_id="wf-1", function_name="wf", parent_id=None, parent_name=None),
            payload=IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                            event_timestamp=0.5,
                                            UUID="tool-short"),
        ),
        IntermediateStep(
            parent_id="func-1",
            function_ancestry=InvocationNode(function_id="wf-1", function_name="wf", parent_id=None, parent_name=None),
            payload=IntermediateStepPayload(event_type=IntermediateStepType.TOOL_END,
                                            event_timestamp=1.5,
                                            span_event_timestamp=0.5,
                                            UUID="tool-short"),
        ),
        # Long LLM call (5s)
        IntermediateStep(
            parent_id="func-1",
            function_ancestry=InvocationNode(function_id="wf-1", function_name="wf", parent_id=None, parent_name=None),
            payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_START,
                                            event_timestamp=0.0,
                                            UUID="llm-long"),
        ),
        IntermediateStep(
            parent_id="func-1",
            function_ancestry=InvocationNode(function_id="wf-1", function_name="wf", parent_id=None, parent_name=None),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END,
                event_timestamp=5.0,
                span_event_timestamp=0.0,
                UUID="llm-long",
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=80)),
            ),
        ),
        # Parent end
        IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(function_id="wf-1", function_name="wf", parent_id=None, parent_name=None),
            payload=IntermediateStepPayload(event_type=IntermediateStepType.FUNCTION_END,
                                            event_timestamp=6.0,
                                            span_event_timestamp=0.0,
                                            UUID="func-1"),
        ),
    ]
    config = SensitivityConfig(w_parallel=0.3)
    builder = PredictionTrieBuilder(sensitivity_config=config)
    contexts = builder._extract_llm_contexts(trace)

    assert len(contexts) == 1
    assert contexts[0].parallel_slack_ratio == pytest.approx(0.0)


def test_parallel_slack_not_computed_when_w_parallel_zero(parallel_trace):
    """Default config (w_parallel=0.0) should leave parallel_slack_ratio at 0.0."""
    config = SensitivityConfig()
    builder = PredictionTrieBuilder(sensitivity_config=config)
    contexts = builder._extract_llm_contexts(parallel_trace)

    for ctx in contexts:
        assert ctx.parallel_slack_ratio == pytest.approx(0.0)


def test_sensitivity_reduced_for_parallel_call(parallel_trace):
    """With w_parallel > 0, a call with high slack should get lower sensitivity."""
    config_no_parallel = SensitivityConfig(sensitivity_scale=5,
                                           w_critical=0.5,
                                           w_fanout=0.3,
                                           w_position=0.2,
                                           w_parallel=0.0)
    config_with_parallel = SensitivityConfig(sensitivity_scale=5,
                                             w_critical=0.5,
                                             w_fanout=0.3,
                                             w_position=0.2,
                                             w_parallel=0.3)

    builder_no = PredictionTrieBuilder(sensitivity_config=config_no_parallel)
    builder_no.add_trace(parallel_trace)
    trie_no = builder_no.build()

    builder_with = PredictionTrieBuilder(sensitivity_config=config_with_parallel)
    builder_with.add_trace(parallel_trace)
    trie_with = builder_with.build()

    node_no = trie_no.children["my_workflow"]
    node_with = trie_with.children["my_workflow"]

    s_no = node_no.predictions_by_call_index[1].latency_sensitivity
    s_with = node_with.predictions_by_call_index[1].latency_sensitivity

    assert s_with <= s_no


def test_sensitivity_score_clamped(parallel_trace):
    """Extreme w_parallel should not produce scores outside [0, 1] or sensitivities outside [1, scale]."""
    config = SensitivityConfig(sensitivity_scale=5, w_critical=0.1, w_fanout=0.1, w_position=0.1, w_parallel=5.0)
    builder = PredictionTrieBuilder(sensitivity_config=config)
    builder.add_trace(parallel_trace)
    trie = builder.build()

    node = trie.children["my_workflow"]
    s = node.predictions_by_call_index[1].latency_sensitivity
    assert s is not None
    assert 1 <= s <= 5


def test_build_sibling_map(parallel_trace):
    """Unit test _build_sibling_map directly."""
    sibling_map = PredictionTrieBuilder._build_sibling_map(parallel_trace)

    # Under parent "func-1", expect two siblings: llm-p1 and tool-p1
    assert "func-1" in sibling_map
    siblings = sibling_map["func-1"]
    uuids = {s.uuid for s in siblings}
    assert "llm-p1" in uuids
    assert "tool-p1" in uuids

    # Verify the LLM span is flagged
    llm_span = next(s for s in siblings if s.uuid == "llm-p1")
    assert llm_span.is_llm is True
    assert llm_span.start_time == pytest.approx(1.0)
    assert llm_span.end_time == pytest.approx(2.0)

    tool_span = next(s for s in siblings if s.uuid == "tool-p1")
    assert tool_span.is_llm is False
    assert tool_span.start_time == pytest.approx(0.5)
    assert tool_span.end_time == pytest.approx(5.5)


def test_backward_compat_default_w_parallel(simple_trace):
    """Default SensitivityConfig() should produce the same scores as before (w_parallel=0.0)."""
    config_default = SensitivityConfig()
    config_explicit = SensitivityConfig(w_critical=0.5, w_fanout=0.3, w_position=0.2, w_parallel=0.0)

    builder_default = PredictionTrieBuilder(sensitivity_config=config_default)
    builder_default.add_trace(simple_trace)
    trie_default = builder_default.build()

    builder_explicit = PredictionTrieBuilder(sensitivity_config=config_explicit)
    builder_explicit.add_trace(simple_trace)
    trie_explicit = builder_explicit.build()

    node_default = trie_default.children["my_workflow"]
    node_explicit = trie_explicit.children["my_workflow"]

    for idx in node_default.predictions_by_call_index:
        assert (node_default.predictions_by_call_index[idx].latency_sensitivity ==
                node_explicit.predictions_by_call_index[idx].latency_sensitivity)


def _make_ctx(start: float, end: float) -> LLMCallContext:
    """Helper to create a minimal LLMCallContext with span timestamps."""
    return LLMCallContext(
        path=["root"],
        call_index=1,
        remaining_calls=0,
        time_to_next_ms=None,
        output_tokens=10,
        span_start_time=start,
        span_end_time=end,
    )


def test_logical_positions_transitive_overlap():
    """Transitive overlaps must be collapsed into one group.

    A(0–10) overlaps B(1–3) and C(4–6).  B and C do not overlap each other
    directly, but both overlap with A, so all three should share position 0.
    """
    # LLM_END order: B, C, A
    contexts = [_make_ctx(1, 3), _make_ctx(4, 6), _make_ctx(0, 10)]
    positions = PredictionTrieBuilder._compute_logical_positions(contexts)
    assert positions == [0, 0, 0]


def test_logical_positions_no_overlap():
    """Fully sequential calls get distinct positions."""
    contexts = [_make_ctx(0, 1), _make_ctx(2, 3), _make_ctx(4, 5)]
    positions = PredictionTrieBuilder._compute_logical_positions(contexts)
    assert positions == [0, 1, 2]


def test_logical_positions_two_groups():
    """Two separate parallel groups get two distinct positions."""
    # Group 1: A(0–5), B(1–4)   Group 2: C(10–15), D(11–14)
    # LLM_END order: B, A, D, C
    contexts = [_make_ctx(1, 4), _make_ctx(0, 5), _make_ctx(11, 14), _make_ctx(10, 15)]
    positions = PredictionTrieBuilder._compute_logical_positions(contexts)
    assert positions == [0, 0, 1, 1]


def test_logical_positions_empty():
    """Empty contexts returns empty positions."""
    assert PredictionTrieBuilder._compute_logical_positions([]) == []


def test_logical_positions_single():
    """Single context gets position 0."""
    positions = PredictionTrieBuilder._compute_logical_positions([_make_ctx(0, 1)])
    assert positions == [0]
