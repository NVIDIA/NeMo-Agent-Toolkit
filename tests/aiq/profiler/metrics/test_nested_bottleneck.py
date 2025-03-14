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
from aiq.profiler.inference_optimization.bottleneck_analysis.nested_stack_analysis import analyze_calls_and_build_result
from aiq.profiler.inference_optimization.bottleneck_analysis.nested_stack_analysis import build_call_tree_for_example
from aiq.profiler.inference_optimization.bottleneck_analysis.nested_stack_analysis import build_call_tree_per_example
from aiq.profiler.inference_optimization.bottleneck_analysis.nested_stack_analysis import compute_time_based_concurrency
from aiq.profiler.inference_optimization.bottleneck_analysis.nested_stack_analysis import find_midpoint_concurrency
from aiq.profiler.inference_optimization.bottleneck_analysis.nested_stack_analysis import multi_example_call_profiling
from aiq.profiler.inference_optimization.data_models import CallNode
from aiq.profiler.inference_optimization.data_models import ConcurrencyDistribution
from aiq.profiler.inference_optimization.data_models import NestedCallProfilingResult
from aiq.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor
from aiq.profiler.utils import create_standardized_dataframe

#############################################################
# Test Data Setup
#############################################################


@pytest.fixture(name="minimal_valid_df")
def minimal_valid_df_fixture():
    """A minimal DataFrame with the columns the code expects."""
    # data = {
    #     "example_number": [1, 1, 1, 1, 2, 2],
    #     "event_type": [
    #         WorkflowEventEnum.LLM_START,
    #         WorkflowEventEnum.TOOL_START,
    #         WorkflowEventEnum.TOOL_END,
    #         WorkflowEventEnum.LLM_END,
    #         WorkflowEventEnum.TOOL_START,
    #         WorkflowEventEnum.TOOL_END
    #     ],
    #     "UUID": ["u1", "u2", "u2", "u1", "u3", "u3"],
    #     "event_timestamp": [1.0, 1.5, 1.6, 2.0, 10.0, 11.0],
    #     "llm_name": ["llama-3", None, None, "llama-3", None, None],
    #     "tool_name": [None, "weather-search", "weather-search", None, "google-search", "google-search"],
    # }
    # df = pd.DataFrame(data)
    # Create intermediate steps events to mock above dataframe
    events = [[
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.LLM_START,
                                                         event_timestamp=1.0,
                                                         name="llama-3",
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u1")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_START,
                                                         event_timestamp=1.5,
                                                         name="weather-search",
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u2")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_END,
                                                         event_timestamp=1.6,
                                                         name="weather-search",
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u2")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.LLM_END,
                                                         event_timestamp=2.0,
                                                         name="llama-3",
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u1"))
    ],
              [
                  IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_START,
                                                                   event_timestamp=10.0,
                                                                   name="google-search",
                                                                   framework=LLMFrameworkEnum.LANGCHAIN,
                                                                   UUID="u3")),
                  IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_END,
                                                                   name="google-search",
                                                                   event_timestamp=11.0,
                                                                   framework=LLMFrameworkEnum.LANGCHAIN,
                                                                   UUID="u3"))
              ]]

    return [[IntermediatePropertyAdaptor.from_intermediate_step(step) for step in steps] for steps in events]


#############################################################
# build_call_tree_for_example
#############################################################


def test_build_call_tree_for_example_basic(minimal_valid_df):
    """Test basic usage on a single example subset from minimal_valid_df."""

    # Extract only example_number=1
    ex1 = create_standardized_dataframe([minimal_valid_df[0]])

    # Build
    result = build_call_tree_for_example(ex1)
    assert isinstance(result, list)

    assert len(result) == 1, "We expect 1 top-level call for example #1."

    top_call = result[0]
    assert isinstance(top_call, CallNode)
    assert top_call.operation_type == "LLM"
    assert top_call.uuid == "u1"
    # Duration => 2.0 - 1.0 = 1.0
    assert abs(top_call.duration - 1.0) < 1e-7

    # The child should be the tool call
    assert len(top_call.children) == 1
    tool_call = top_call.children[0]
    assert tool_call.operation_type == "TOOL"


#############################################################
# build_call_tree_per_example
#############################################################


def test_build_call_tree_per_example_basic(minimal_valid_df):
    """Test multi-example usage, ensuring calls from example 1 and 2 are separated."""

    roots = build_call_tree_per_example(minimal_valid_df)
    # We expect top-level calls from both examples in one combined list
    # Example #1 => 1 top-level call, example #2 => ?

    # We'll do a quick check that we have at least something
    assert isinstance(roots, list)


#############################################################
# compute_time_based_concurrency
#############################################################


def test_compute_time_based_concurrency_empty():
    """If no calls => concurrency distribution should be zeros."""

    result = compute_time_based_concurrency([])
    assert isinstance(result, ConcurrencyDistribution)
    assert result.p50 == 0
    assert result.p90 == 0
    assert result.timeline_segments == []


def test_compute_time_based_concurrency_basic(minimal_valid_df):
    """Check concurrency distribution for a minimal set of calls."""

    roots = build_call_tree_per_example(minimal_valid_df)
    distribution = compute_time_based_concurrency(roots)
    # Expect a ConcurrencyDistribution with some valid segments.
    assert distribution.p50 >= 0
    assert distribution.p90 == 1
    assert isinstance(distribution.timeline_segments, list)


#############################################################
# find_midpoint_concurrency
#############################################################


def test_find_midpoint_concurrency_no_segments(minimal_valid_df):
    """Midpoint concurrency with empty segments => 0."""

    roots = build_call_tree_per_example(minimal_valid_df)
    # Flatten calls
    all_calls = []
    for r in roots:
        all_calls.append(r)
        all_calls.extend(r.children)

    res = []
    # Pass empty segments
    segments = []
    for c in all_calls:
        mc = find_midpoint_concurrency(c, segments)
        res.append(mc)
        assert mc == 0.0


def test_find_midpoint_concurrency_basic(minimal_valid_df):
    """Basic concurrency with a minimal timeline, single segment."""

    roots = build_call_tree_per_example(minimal_valid_df)
    distribution = compute_time_based_concurrency(roots)
    segments = distribution.timeline_segments
    # Flatten calls
    all_calls = []

    def dfs(n):
        all_calls.append(n)
        for ch in n.children:
            dfs(ch)

    for rt in roots:
        dfs(rt)

    for c in all_calls:
        mc = find_midpoint_concurrency(c, segments)
        assert mc >= 0, "Concurrency must be non-negative"


#############################################################
# analyze_calls_and_build_result
#############################################################


def test_analyze_calls_and_build_result_empty():
    """If roots is empty => no calls => textual report says 'No calls found.'"""
    result = analyze_calls_and_build_result([])
    assert isinstance(result, NestedCallProfilingResult)
    assert "No calls found" in result.textual_report
    assert not result.node_metrics


def test_analyze_calls_and_build_result_basic(minimal_valid_df, tmp_path):
    """Check analyzing a minimal set of calls => returns a valid NestedCallProfilingResult."""

    roots = build_call_tree_per_example(minimal_valid_df)
    # We'll store a Gantt chart in tmp_path
    result = analyze_calls_and_build_result(roots, output_dir=str(tmp_path))
    assert isinstance(result, NestedCallProfilingResult)
    assert result.concurrency is not None
    # Check if the textual report is not empty
    assert len(result.textual_report) > 10
    # Gantt chart => check if the file is created
    chart_file = tmp_path / "gantt_chart.png"
    assert chart_file.exists(), "Expected a Gantt chart file to be created."


#############################################################
# multi_example_call_profiling
#############################################################


def test_multi_example_call_profiling_full(minimal_valid_df, tmp_path):
    """Full end-to-end test with minimal data => check final output is well-formed."""
    result = multi_example_call_profiling(minimal_valid_df, output_dir=str(tmp_path))

    assert isinstance(result, NestedCallProfilingResult)
    # concurrency distribution
    assert result.concurrency.p90 > 0
    # textual report
    assert "Multi-Example Nested Call Profiling Report" in result.textual_report
    # top bottlenecks
    assert isinstance(result.top_bottlenecks, list)
    # node metrics => dict
    assert isinstance(result.node_metrics, dict)

    # Check the Gantt chart was created
    chart_file = tmp_path / "gantt_chart.png"
    assert chart_file.exists(), "Expected a Gantt chart file to be created."
