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

import pytest

from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType as WorkflowEventEnum
from aiq.data_models.intermediate_step import StreamEventData
from aiq.data_models.invocation_node import InvocationNode
from aiq.profiler.inference_optimization.bottleneck_analysis.simple_stack_analysis import profile_workflow_bottlenecks
from aiq.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor

##########################################################
# Fixtures
##########################################################


@pytest.fixture(name="minimal_valid_df")
def minimal_valid_df_fixture():
    """A minimal DataFrame with the columns the code expects."""
    events = [[
        IntermediateStep(parent_id="root",
                         function_ancestry=InvocationNode(function_name="llama-3", function_id="u1"),
                         payload=IntermediateStepPayload(event_type=WorkflowEventEnum.LLM_START,
                                                         event_timestamp=1.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u1",
                                                         name="llama-3",
                                                         data=StreamEventData(input="Hello world"))),
        IntermediateStep(parent_id="root",
                         function_ancestry=InvocationNode(function_name="weather-search", function_id="u2"),
                         payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_START,
                                                         event_timestamp=1.5,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u2",
                                                         name="weather-search")),
        IntermediateStep(parent_id="root",
                         function_ancestry=InvocationNode(function_name="weather-search", function_id="u2"),
                         payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_END,
                                                         event_timestamp=1.6,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u2",
                                                         name="weather-search")),
        IntermediateStep(parent_id="root",
                         function_ancestry=InvocationNode(function_name="llama-3", function_id="u1"),
                         payload=IntermediateStepPayload(event_type=WorkflowEventEnum.LLM_END,
                                                         event_timestamp=2.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u1",
                                                         name="llama-3"))
    ],
              [
                  IntermediateStep(parent_id="root",
                                   function_ancestry=InvocationNode(function_name="google-search", function_id="u3"),
                                   payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_START,
                                                                   event_timestamp=10.0,
                                                                   framework=LLMFrameworkEnum.LANGCHAIN,
                                                                   UUID="u3",
                                                                   name="google-search")),
                  IntermediateStep(parent_id="root",
                                   function_ancestry=InvocationNode(function_name="google-search", function_id="u3"),
                                   payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_END,
                                                                   event_timestamp=11.0,
                                                                   framework=LLMFrameworkEnum.LANGCHAIN,
                                                                   UUID="u3",
                                                                   name="google-search"))
              ]]

    return [[IntermediatePropertyAdaptor.from_intermediate_step(step) for step in steps] for steps in events]


##########################################################
# Tests
##########################################################


def test_profile_workflow_bottlenecks_incomplete_pairs(minimal_valid_df):
    """
    If we have partial data for a particular UUID with no matching END => skip or partial coverage.
    We'll mutate minimal_valid_df so that one operation has only START, no END.
    """
    from aiq.profiler.inference_optimization.data_models import SimpleBottleneckReport

    # We'll remove the LLM_END row => so the LLM calls are partial
    # minimal_valid_df has row with event_type LLM_END => remove it
    df_test = [minimal_valid_df[0][:3], minimal_valid_df[1]]

    result = profile_workflow_bottlenecks(df_test)
    assert isinstance(result, SimpleBottleneckReport)
    assert len(result.stats) >= 1  # Because we still have the tool operation (u2 or u3) that is start/end
    # And the summary should mention "BOTTLENECK REPORT"
    assert "BOTTLENECK REPORT" in result.summary


def test_profile_workflow_bottlenecks_normal(minimal_valid_df):
    """
    Normal usage with a minimal valid df => expect a valid SimpleBottleneckReport
    with stats for LLM and tool operations.
    """
    from aiq.profiler.inference_optimization.data_models import SimpleBottleneckReport
    from aiq.profiler.inference_optimization.data_models import SimpleOperationStats

    result = profile_workflow_bottlenecks(minimal_valid_df)
    assert isinstance(result, SimpleBottleneckReport)
    assert len(result.stats) > 0, "We should have at least some stats for LLM or tool ops."

    # For example, we might see "LLM:llama-3" or "TOOL:weather-search", "TOOL:google-search".
    # Check that the keys reflect operation_type:operation_name
    for _, val in result.stats.items():
        assert isinstance(val, SimpleOperationStats), "Each entry must be a SimpleOperationStats."
        # val usage_count, avg_duration, etc. are floats or ints
        assert val.usage_count >= 1

    # The summary must mention top 3 Bottlenecks
    assert "Top 3 Bottlenecks by bottleneck_score" in result.summary


def test_profile_workflow_bottlenecks_freq_stats(minimal_valid_df):
    """
    Check that the result includes average durations, concurrency, and a bottleneck_score.
    We can do a rough numeric check that it's not NaN or negative.
    """

    result = profile_workflow_bottlenecks(minimal_valid_df)
    for _, stat in result.stats.items():
        # usage_count should be positive
        assert stat.usage_count >= 1
        # durations
        assert stat.avg_duration >= 0.0
        assert stat.p95_duration >= stat.avg_duration, "p95 should be >= average"
        assert stat.p99_duration >= stat.p95_duration, "p99 should be >= p95"
        # concurrency
        assert stat.max_concurrency >= 0
        # bottleneck_score
        assert stat.bottleneck_score >= 0


def test_profile_workflow_bottlenecks_summary(minimal_valid_df):
    """
    Check that the summary is well-formed, mentions the number of distinct operations, etc.
    """

    result = profile_workflow_bottlenecks(minimal_valid_df)
    summary = result.summary
    assert "Total distinct operations found:" in summary
    assert "Top 3 Bottlenecks by bottleneck_score" in summary
    # Also check the concurrency line
    assert "Overall max concurrency" in summary
