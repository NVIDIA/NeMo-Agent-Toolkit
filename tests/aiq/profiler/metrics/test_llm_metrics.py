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

import pandas as pd
import pytest

from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.intermediate_step import StreamEventData
from aiq.data_models.intermediate_step import UsageInfo
from aiq.data_models.invocation_node import InvocationNode
from aiq.profiler.callbacks.token_usage_base_model import TokenUsageBaseModel
from aiq.profiler.inference_optimization.llm_metrics import LLMMetrics
from aiq.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor


@pytest.fixture(name="sample_dataframe")
def sample_dataframe_fixture():
    """
    Provides a sample DataFrame for testing.
    This fixture can be reused across test cases if needed.
    """
    events = [[
        IntermediateStep(payload=IntermediateStepPayload(
            event_type=IntermediateStepType.LLM_START,
            event_timestamp=1000.0,
            framework=LLMFrameworkEnum.LANGCHAIN,
            name="my_func",
            data=StreamEventData(input="Hello world!"),
            UUID="uuid-abc",
            usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=42))),
                         function_ancestry=InvocationNode(function_name="my_func", function_id="uuid-abc")),
        IntermediateStep(payload=IntermediateStepPayload(
            event_type=IntermediateStepType.LLM_END,
            event_timestamp=1001.0,
            framework=LLMFrameworkEnum.LANGCHAIN,
            name="my_func",
            data=StreamEventData(output="Hello world!"),
            UUID="uuid-abc",
            usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=42))),
                         function_ancestry=InvocationNode(function_name="my_func", function_id="uuid-abc")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_START,
                                                         event_timestamp=1002.5,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         name="my_func",
                                                         data=StreamEventData(input="Hello world!"),
                                                         UUID="uuid-xyz"),
                         function_ancestry=InvocationNode(function_name="my_func", function_id="uuid-xyz")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                                         event_timestamp=1003.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         name="my_func",
                                                         data=StreamEventData(input="Hello world!"),
                                                         UUID="uuid-tool"),
                         function_ancestry=InvocationNode(function_name="my_func", function_id="uuid-tool")),
    ],
              [
                  IntermediateStep(payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_START,
                                                                   event_timestamp=5000.0,
                                                                   framework=LLMFrameworkEnum.LANGCHAIN,
                                                                   name="other_func",
                                                                   data=StreamEventData(input="Hello world!"),
                                                                   UUID="uuid-123"),
                                   function_ancestry=InvocationNode(function_name="other_func",
                                                                    function_id="uuid-123")),
                  IntermediateStep(payload=IntermediateStepPayload(
                      event_type=IntermediateStepType.LLM_END,
                      event_timestamp=5001.0,
                      framework=LLMFrameworkEnum.LANGCHAIN,
                      name="other_func",
                      data=StreamEventData(output="Hello world!"),
                      UUID="uuid-123",
                      usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=100))),
                                   function_ancestry=InvocationNode(function_name="other_func",
                                                                    function_id="uuid-123")),
              ]]

    return [[IntermediatePropertyAdaptor.from_intermediate_step(step) for step in steps] for steps in events]


def test_columns_added(sample_dataframe):
    """
    Test that the compute_profiling_metrics method adds the required
    NOVA- columns to the DataFrame.
    """
    df = LLMMetrics.compute_profiling_metrics(sample_dataframe)

    # Check columns exist
    expected_cols = [
        'NOVA-Event-ID',
        'NOVA-Requests-Remaining-In-Event',
        'NOVA-Time-To-Next-Event',
        'NOVA-Time-To-Event-End',
        'NOVA-Predicted-OSL',
        'NOVA-Time-To-Session-End'
    ]
    for col in expected_cols:
        assert col in df.columns, f"Expected column '{col}' not found in DataFrame."


def test_nova_event_id(sample_dataframe):
    """
    Test that NOVA-Event-ID is correctly populated as the function_name.
    """
    df = LLMMetrics.compute_profiling_metrics(sample_dataframe)
    assert (df['NOVA-Event-ID'] == df['function_name']).all()


def test_requests_remaining_in_event(sample_dataframe):
    """
    Test that 'NOVA-Requests-Remaining-In-Event' is computed correctly.
    We'll focus on rows in 'my_func'.
    """
    df = LLMMetrics.compute_profiling_metrics(sample_dataframe)
    # Filter to example_number == 1 and function_name == 'my_func'
    sub = df[(df['example_number'] == 0) & (df['function_name'] == 'my_func')].copy()
    sub = sub.sort_values('event_timestamp').reset_index(drop=True)

    # We have the following relevant rows in that group:
    # Row 0: event_type = LLM_START, ts=1000.0
    # Row 1: event_type = LLM_END,   ts=1001.0
    # Row 2: event_type = LLM_START, ts=1002.5
    # Row 3: event_type = TOOL_START,ts=1003.0

    # The LLM_START events are at 1000.0 and 1002.5
    # So for row 0 (LLM_START at 1000.0), there's 1 more future LLM_START in that group.
    # For row 1 (LLM_END at 1001.0), still there's 1 future LLM_START (at 1002.5).
    # For row 2 (LLM_START at 1002.5), there's 0 future LLM_STARTs.
    # For row 3 (TOOL_START at 1003.0), there's 0 future LLM_STARTs.

    expected_remaining = [1, 1, 0, 0]
    assert all(sub['NOVA-Requests-Remaining-In-Event'] == expected_remaining), \
        "NOVA-Requests-Remaining-In-Event values are incorrect."


def test_time_to_next_event(sample_dataframe):
    """
    Test that 'NOVA-Time-To-Next-Event' matches expectations.
    Focus on example_number=1, function_name='my_func'.
    """
    df = LLMMetrics.compute_profiling_metrics(sample_dataframe.copy())
    # Filter to example_number == 1 and function_name == 'my_func'
    sub = df[(df['example_number'] == 0) & (df['function_name'] == 'my_func')].copy()
    sub = sub.sort_values('event_timestamp').reset_index(drop=True)

    # LLM_START timestamps: 1000.0, 1002.5
    # - Row 0 (ts=1000.0) => time to next LLM_START (1002.5) = (1002.5 - 1000.0)*1000=2500 ms
    # - Row 1 (ts=1001.0, event=LLM_END) => next LLM_START is still at 1002.5 => (1002.5-1001.0)*1000=1500 ms
    # - Row 2 (ts=1002.5, event=LLM_START) => no future LLM_START => -1
    # - Row 3 (ts=1003.0, event=TOOL_START) => no future LLM_START => -1

    expected_next = [2500, 1500, -1, -1]
    assert all(sub['NOVA-Time-To-Next-Event'].astype(int) == expected_next), \
        "NOVA-Time-To-Next-Event values are incorrect."


def test_time_to_event_end(sample_dataframe):
    """
    Test that 'NOVA-Time-To-Event-End' is the time to the last future LLM_START
    in the same group.
    """
    df = LLMMetrics.compute_profiling_metrics(sample_dataframe.copy())
    # Focus on example_number=1, function_name='my_func'
    sub = df[(df['example_number'] == 0) & (df['function_name'] == 'my_func')]
    sub = sub.sort_values('event_timestamp').reset_index(drop=True)

    # The last LLM_START in the future for each row is at 1002.5 if it's strictly after row's timestamp.
    # - Row 0 (ts=1000.0) => time to last future LLM_START is (1002.5 - 1000.0)*1000 = 2500
    # - Row 1 (ts=1001.0) => same last future = 1002.5 => 1500 ms
    # - Row 2 (ts=1002.5) => no future => -1
    # - Row 3 (ts=1003.0) => no future => -1

    expected_end = [2500, 1500, -1, -1]
    assert all(sub['NOVA-Time-To-Event-End'].astype(int) == expected_end), \
        "NOVA-Time-To-Event-End values are incorrect."


def test_predicted_osl(sample_dataframe):
    """
    Test that NOVA-Predicted-OSL is correctly set for LLM_START events,
    and that it matches the 'completion_tokens' in the corresponding LLM_END.
    """
    df = LLMMetrics.compute_profiling_metrics(sample_dataframe.copy())

    # For UUID=uuid-abc: LLM_START (row 0) => LLM_END has completion_tokens=42
    row_start_abc = df[(df['UUID'] == 'uuid-abc') & (df['event_type'] == 'LLM_START')].iloc[0]
    assert row_start_abc['NOVA-Predicted-OSL'] == 42

    # For UUID=uuid-xyz: LLM_START (row 2) => no matching LLM_END => should be NaN
    row_start_xyz = df[(df['UUID'] == 'uuid-xyz') & (df['event_type'] == 'LLM_START')].iloc[0]
    assert pd.isna(row_start_xyz['NOVA-Predicted-OSL'])

    # For UUID=uuid-123: LLM_START => LLM_END has completion_tokens=100
    row_start_123 = df[(df['UUID'] == 'uuid-123') & (df['event_type'] == 'LLM_START')].iloc[0]
    assert row_start_123['NOVA-Predicted-OSL'] == 100


def test_time_to_session_end(sample_dataframe):
    """
    Test that 'NOVA-Time-To-Session-End' is computed as
    (max_ts_of_example_number - row_ts) * 1000.
    We'll check for example_number=1 and example_number=2.
    """
    df = LLMMetrics.compute_profiling_metrics(sample_dataframe.copy())

    # example_number=1 => max_ts=1003.0
    # Row timestamps => 1000.0, 1001.0, 1002.5, 1003.0
    # Differences => (1003.0 - row_ts)*1000 => 3000, 2000, 500, 0
    sub1 = df[df['example_number'] == 0].copy().sort_values('event_timestamp')
    expected_session_end_1 = [3000.0, 2000.0, 500.0, 0.0]
    computed_1 = (sub1['NOVA-Time-To-Session-End'].values).round(0)  # round for float safety
    assert all(computed_1 == expected_session_end_1), \
        f"Expected {expected_session_end_1} but got {computed_1} for example_number=1"

    # example_number=2 => max_ts=5001.0
    # Timestamps => 5000.0, 5001.0 => differences => (5001.0 - row_ts)*1000 => 1000, 0
    sub2 = df[df['example_number'] == 1].copy().sort_values('event_timestamp')
    expected_session_end_2 = [1000.0, 0.0]
    computed_2 = (sub2['NOVA-Time-To-Session-End'].values).round(0)
    assert all(computed_2 == expected_session_end_2), \
        f"Expected {expected_session_end_2} but got {computed_2} for example_number=2"
