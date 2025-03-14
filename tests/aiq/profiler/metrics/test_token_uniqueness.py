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
from aiq.profiler.inference_optimization.token_uniqueness import compute_inter_query_token_uniqueness_by_llm
from aiq.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor


@pytest.fixture(name="minimal_valid_df")
def minimal_valid_df_fixture():
    """
    A minimal DataFrame with the columns the code expects, including an 'llm_text_input' column
    so we can test 'new words' logic.
    """
    # data = {
    #     "example_number": [1, 1, 1, 1, 2, 2],
    #     "event_type": [
    #         WorkflowEventEnum.LLM_START,  # 1) LLM_START
    #         WorkflowEventEnum.TOOL_START,  # 2) not LLM => ignored
    #         WorkflowEventEnum.TOOL_END,  # 3) not LLM => ignored
    #         WorkflowEventEnum.LLM_END,  # 4) LLM_END => no text needed
    #         WorkflowEventEnum.TOOL_START,  # 5) example #2
    #         WorkflowEventEnum.TOOL_END
    #     ],
    #     "UUID": ["u1", "u2", "u2", "u1", "u3", "u3"],
    #     "event_timestamp": [1.0, 1.5, 1.6, 2.0, 10.0, 11.0],
    #     "llm_name": ["llama-3", None, None, "llama-3", None, None],
    #     "tool_name": [None, "weather-search", "weather-search", None, "google-search", "google-search"],
    #     # Code requires 'llm_text_input'
    # }
    # df = pd.DataFrame(data)
    events = [[
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.LLM_START,
                                                         event_timestamp=1.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u1",
                                                         name="llama-3",
                                                         data=StreamEventData(input="Hello world"))),
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_START,
                                                         event_timestamp=1.5,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u2",
                                                         name="weather-search")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_END,
                                                         event_timestamp=1.6,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u2",
                                                         name="weather-search")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.LLM_END,
                                                         event_timestamp=2.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u1",
                                                         name="llama-3"))
    ],
              [
                  IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_START,
                                                                   event_timestamp=10.0,
                                                                   framework=LLMFrameworkEnum.LANGCHAIN,
                                                                   UUID="u3",
                                                                   name="google-search")),
                  IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_END,
                                                                   event_timestamp=11.0,
                                                                   framework=LLMFrameworkEnum.LANGCHAIN,
                                                                   UUID="u3",
                                                                   name="google-search"))
              ]]

    return [[IntermediatePropertyAdaptor.from_intermediate_step(step) for step in steps] for steps in events]


def test_compute_inter_query_token_uniqueness_by_llm_no_llm_start(minimal_valid_df):
    """
    If we have no LLM_START events => empty root in LLMUniquenessMetricsByLLM.
    We'll remove the LLM_START row from the fixture to simulate that.
    """
    from aiq.profiler.inference_optimization.data_models import LLMUniquenessMetricsByLLM

    df_test = minimal_valid_df.copy()
    # remove the row that has LLM_START
    df_test = [minimal_valid_df[0][1:], minimal_valid_df[1]]

    result = compute_inter_query_token_uniqueness_by_llm(df_test)
    assert isinstance(result, LLMUniquenessMetricsByLLM)
    # Check that it's empty
    assert result.root == {}, "Expected an empty dictionary if no LLM_START events"


def test_compute_inter_query_token_uniqueness_by_llm_minimal(minimal_valid_df):
    """
    Minimal data with 1 LLM_START => no consecutive LLM calls => no new words counts => might be empty
    or zero. Ensure it doesn't crash.
    """
    from aiq.profiler.inference_optimization.data_models import LLMUniquenessMetricsByLLM

    # We'll add text to that single LLM_START row
    df_test = minimal_valid_df.copy()

    result = compute_inter_query_token_uniqueness_by_llm(df_test)
    assert isinstance(result, LLMUniquenessMetricsByLLM)

    # The code checks for consecutive calls in same group => no consecutive => none. Let's see:
    # Possibly root is empty or has 'llama-3' => no new words. Let's see. We'll just confirm no crash.
    if result.root:
        # We can do a small check if there's a key => 'llama-3'
        # But it might not appear if there's no consecutive calls
        pass


def test_compute_inter_query_token_uniqueness_by_llm_two_consecutive_llm_calls():
    """
    We'll build a custom df with 2 consecutive LLM_START calls for the same llm_name => ensure new words are computed.
    """

    from aiq.profiler.inference_optimization.data_models import LLMUniquenessMetrics
    from aiq.profiler.inference_optimization.data_models import LLMUniquenessMetricsByLLM

    events = [[
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.LLM_START,
                                                         event_timestamp=1.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u10",
                                                         name="llama-3",
                                                         data=StreamEventData(input="Hello world"))),
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.LLM_START,
                                                         event_timestamp=2.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u11",
                                                         name="llama-3",
                                                         data=StreamEventData(input="Hello new tokens world")))
    ]]

    events = [[IntermediatePropertyAdaptor.from_intermediate_step(step) for step in steps] for steps in events]

    # Now run
    result = compute_inter_query_token_uniqueness_by_llm(events)
    assert isinstance(result, LLMUniquenessMetricsByLLM)
    # We expect there's a 'llama-3' key with new words counts
    metrics_dict = result.root
    assert "llama-3" in metrics_dict
    # That LLM name => LLMUniquenessMetrics object
    llm_metrics = metrics_dict["llama-3"]
    assert isinstance(llm_metrics, LLMUniquenessMetrics)

    assert llm_metrics.p90 == 2.0
    assert llm_metrics.p95 == 2.0
    assert llm_metrics.p99 == 2.0


def test_compute_inter_query_token_uniqueness_by_llm_multiple_examples(minimal_valid_df):
    """
    If we have multiple examples with multiple LLM calls, ensure we gather all new_words_count in each llm_name group.
    """
    from aiq.profiler.inference_optimization.data_models import LLMUniquenessMetricsByLLM

    new_events = [[
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.LLM_START,
                                                         event_timestamp=10.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="uX",
                                                         name="llama-3",
                                                         data=StreamEventData(input="Testing one"))),
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.LLM_START,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="uY",
                                                         name="llama-3",
                                                         event_timestamp=12.0,
                                                         data=StreamEventData(input="Testing one two")))
    ]]
    df_test = [minimal_valid_df[0], new_events[0]]

    df_test = [[IntermediatePropertyAdaptor.from_intermediate_step(step) for step in steps] for steps in df_test]

    # Now example#1 has 1 LLM_START => no consecutive calls
    # example#2 has 2 consecutive LLM_START => new words => "one two" minus "one" => 1 new word
    result = compute_inter_query_token_uniqueness_by_llm(df_test)
    assert isinstance(result, LLMUniquenessMetricsByLLM)
    metrics_dict = result.root
    # only "llama-3" => check coverage
    assert "llama-3" in metrics_dict
    # p90, p95, p99 => all 1 if "one two" minus "one" => 1 new word
    llm_metrics = metrics_dict["llama-3"]
    assert llm_metrics.p90 == 1.0
    assert llm_metrics.p95 == 1.0
    assert llm_metrics.p99 == 1.0


def test_compute_inter_query_token_uniqueness_by_llm_no_consecutive_calls(minimal_valid_df):
    """
    If there's only single LLM_START in each group => no 'prev' => new_words not computed => empty.
    So we expect either no keys or zero p90/p95/p99.
    """
    # We'll ensure there's exactly one LLM_START in example#1, and none in example#2
    df_test = minimal_valid_df.copy()
    # Remove the LLM_END => no consecutive calls
    df_test = [minimal_valid_df[0][:3], minimal_valid_df[1][:2]]
    # We keep the single LLM_START => no SHIFT => new_words_count won't exist
    result = compute_inter_query_token_uniqueness_by_llm(df_test)
    # check result
    metrics_dict = result.root
    # either empty or the p90=0 if we have an entry
    if metrics_dict:
        pass  # We won't force a check; it's enough that it doesn't crash and is well-formed
