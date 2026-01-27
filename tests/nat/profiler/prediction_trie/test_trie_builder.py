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
