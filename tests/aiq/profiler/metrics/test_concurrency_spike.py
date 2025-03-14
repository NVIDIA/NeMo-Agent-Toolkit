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
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.intermediate_step import StreamEventData
from aiq.profiler.inference_optimization.experimental.concurrency_spike_analysis import concurrency_spike_analysis
from aiq.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor

###############################################################################
# Fixtures
###############################################################################


@pytest.fixture(name="minimal_valid_df")
def minimal_valid_df_fixture():
    """
    A minimal DataFrame with the columns the code expects.
    Provided in the prompt, using WorkflowEventEnum for event_type, etc.
    """
    # data = {
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
    # Create list of events that will make the above dataframe
    events = [[
        IntermediateStep(payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_START,
                                                         event_timestamp=1.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         name="llama-3",
                                                         data=StreamEventData(input="Hello world!"),
                                                         UUID="u1")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                                         event_timestamp=1.5,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         name="weather-search",
                                                         data=StreamEventData(input="Hello world!"),
                                                         UUID="u2")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=IntermediateStepType.TOOL_END,
                                                         event_timestamp=1.6,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         name="weather-search",
                                                         data=StreamEventData(output="Hello world!"),
                                                         UUID="u2")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_END,
                                                         event_timestamp=2.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         name="llama-3",
                                                         data=StreamEventData(output="Hello world!"),
                                                         UUID="u1")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                                         event_timestamp=10.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         name="google-search",
                                                         data=StreamEventData(input="Hello world!"),
                                                         UUID="u3")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=IntermediateStepType.TOOL_END,
                                                         event_timestamp=11.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         name="google-search",
                                                         data=StreamEventData(output="Hello world!"),
                                                         UUID="u3"))
    ]]

    return [[IntermediatePropertyAdaptor.from_intermediate_step(step) for step in steps] for steps in events]


###############################################################################
# Test Cases
###############################################################################


def test_concurrency_spike_analysis_minimal(minimal_valid_df):
    """
    Normal run with minimal_valid_df => expect a valid ConcurrencyAnalysisResult with
    concurrency distribution, some spikes or none, correlation stats, average latency, etc.
    """
    from aiq.profiler.inference_optimization.data_models import ConcurrencyAnalysisResult

    result = concurrency_spike_analysis(minimal_valid_df)
    assert isinstance(result, ConcurrencyAnalysisResult), "Must return a ConcurrencyAnalysisResult"

    # Basic checks
    assert result.concurrency_distribution is not None
    assert isinstance(result.concurrency_distribution, dict)
    # p50_concurrency etc. should be >= 0
    assert result.p50_concurrency >= 0
    assert result.p90_concurrency == 1
    # textual_report
    assert "Concurrency Spike Analysis" in result.textual_report
    assert "Total calls in dataset:" in result.textual_report

    # correlation_stats => check it's not None
    corr_stats = result.correlation_stats
    assert corr_stats is not None
    # The average prompt_tokens / total_tokens might be zero or None => check it doesn't blow up
    assert corr_stats.avg_prompt_tokens >= 0
    assert corr_stats.avg_total_tokens >= 0

    # average_latency_by_concurrency => a dict
    assert isinstance(result.average_latency_by_concurrency, dict)


def test_concurrency_spike_analysis_spike_threshold(minimal_valid_df):
    """
    Provide a custom concurrency_spike_threshold => check if that influences the spike intervals.
    For instance, set threshold=1 => we might see intervals for concurrency >=1
    """
    from aiq.profiler.inference_optimization.data_models import ConcurrencyAnalysisResult

    # concurrency_spike_threshold=1 => every call with concurrency >=1 is a spike
    result = concurrency_spike_analysis(minimal_valid_df, concurrency_spike_threshold=1)
    assert isinstance(result, ConcurrencyAnalysisResult)
    # If we have concurrency >=1 at times => we expect spike_intervals not empty
    # minimal_valid_df => definitely concurrency=1 or 2 at some times
    # So we should see some intervals
    if len(result.spike_intervals) == 0:
        pytest.fail("Expected at least one spike interval when threshold=1 for minimal_valid_df")


def test_concurrency_spike_analysis_report_contents(minimal_valid_df):
    """
    Verify textual_report includes concurrency distribution, spike intervals, correlation stats, etc.
    """

    result = concurrency_spike_analysis(minimal_valid_df)
    report = result.textual_report
    assert "Concurrency Spike Analysis" in report
    assert "Detected Spike Intervals" in report
    assert "Correlation Stats for Spiked Calls" in report
    # We also expect "Avg prompt_tokens" etc.
    assert "Avg prompt_tokens in spike calls" in report
    assert "Avg total_tokens in spike calls" in report
    assert "Average Latency by Midpoint Concurrency" in report
