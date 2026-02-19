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
from nat.plugins.eval.profiler.prediction_trie.trie_builder import PredictionTrieBuilder
from nat.plugins.eval.profiler.prediction_trie.trie_builder import SensitivityConfig


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
    config = SensitivityConfig(scale=5, w_critical=0.5, w_fanout=0.3, w_position=0.2)
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
    config = SensitivityConfig(scale=5, w_critical=0.5, w_fanout=0.3, w_position=0.2)
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
                function_id="wf-1", function_name="wf", parent_id=None, parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START, event_timestamp=0.0, UUID="a",
            ),
        ),
        IntermediateStep(
            parent_id="root",
            function_ancestry=InvocationNode(
                function_id="wf-1", function_name="wf", parent_id=None, parent_name=None,
            ),
            payload=IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_END, event_timestamp=10.0,
                span_event_timestamp=0.0, UUID="a",
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=50)),
            ),
        ),
    ]
    config = SensitivityConfig(scale=3)
    builder = PredictionTrieBuilder(sensitivity_config=config)
    builder.add_trace(trace)
    trie = builder.build()

    node = trie.children["wf"]
    s = node.predictions_by_call_index[1].latency_sensitivity
    assert 1 <= s <= 3


def test_sensitivity_aggregated_across_traces(simple_trace):
    """Multiple traces should be averaged for sensitivity scoring."""
    config = SensitivityConfig(scale=5)
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
    config = SensitivityConfig(scale=5)
    builder = PredictionTrieBuilder(sensitivity_config=config)
    builder.add_trace(simple_trace)
    trie = builder.build()

    node = trie.children["my_workflow"]
    assert node.predictions_any_index is not None
    assert node.predictions_any_index.latency_sensitivity is not None
    assert 1 <= node.predictions_any_index.latency_sensitivity <= 5
