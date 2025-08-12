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

import json
import os

import pytest

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.evaluate import EvalConfig
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType as WorkflowEventEnum
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.profiler import ProfilerConfig
from nat.profiler.data_frame_row import DataFrameRow
from nat.profiler.profile_runner import ProfilerRunner


@pytest.fixture(name="minimal_eval_config")
def minimal_eval_config_fixture(tmp_path):
    """
    Provides an EvalConfig with a writable output_dir pointing to pytest's tmp_path.
    This ensures ProfilerRunner will write JSON output files into that directory.
    """
    # Set up an EvalConfig that includes the fields ProfilerRunner relies on
    eval_config = EvalConfig()
    # Overwrite the output_dir to the temporary path
    eval_config.general.output_dir = str(tmp_path / "profiling_outputs")
    # Turn on the inference profiling
    eval_config.general.profiler = ProfilerConfig(fit_model=False)

    return eval_config


class BrokenStr:

    def __str__(self):
        raise ValueError("Broken __str__")


def test_cast_to_str_success():
    # Test that non-string values are correctly cast to string.
    row = DataFrameRow(
        event_type="test_event_success",
        event_timestamp=1234567890.0,
        example_number=42,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        llm_text_input=100,  # integer -> should become "100"
        llm_text_output=200.5,  # float -> should become "200.5"
        llm_new_token=True,  # bool -> should become "True"
        llm_name="model",
        tool_name="tool",
        function_name="func",
        function_id="1",
        parent_function_name="parent_func",
        parent_function_id="2",
        UUID="uuid",
        framework="pydantic")
    # Assert that the conversion happened correctly.
    assert isinstance(row.llm_text_input, str)
    assert row.llm_text_input == "100"
    assert isinstance(row.llm_text_output, str)
    assert row.llm_text_output == "200.5"
    assert isinstance(row.llm_new_token, str)
    assert row.llm_new_token == "True"


def test_cast_to_str_none():
    # Test that None values remain None.
    row = DataFrameRow(event_type="test_event",
                       event_timestamp=1234567890.0,
                       example_number=42,
                       prompt_tokens=10,
                       completion_tokens=20,
                       total_tokens=30,
                       llm_text_input=None,
                       llm_text_output=None,
                       llm_new_token=None,
                       llm_name="model",
                       tool_name="tool",
                       function_name="func",
                       function_id="1",
                       parent_function_name="parent_func",
                       parent_function_id="2",
                       UUID="uuid",
                       framework="pydantic")
    assert row.llm_text_input is None
    assert row.llm_text_output is None
    assert row.llm_new_token is None


def test_cast_to_str_failure():
    # Test that passing a value that fails to convert to str raises a ValueError.
    with pytest.raises(ValueError) as exc_info:
        DataFrameRow(
            event_type="test_event",
            event_timestamp=1234567890.0,
            example_number=42,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            llm_text_input=BrokenStr(),  # This should raise an error during conversion.
            llm_text_output="valid",
            llm_new_token="also valid",
            llm_name="model",
            tool_name="tool",
            function_name="func",
            function_id="1",
            parent_function_name="parent_func",
            parent_function_id="2",
            UUID="uuid",
            framework="pydantic")
    # Check that the error message contains the expected text.
    assert "Broken __str__" in str(exc_info.value)


def test_validate_assignment():
    # Test that assignment validation works as expected.
    row = DataFrameRow(event_type="test_event",
                       event_timestamp=1234567890.0,
                       example_number=42,
                       prompt_tokens=10,
                       completion_tokens=20,
                       total_tokens=30,
                       llm_text_input="initial",
                       llm_text_output="initial",
                       llm_new_token="initial",
                       llm_name="model",
                       tool_name="tool",
                       function_name="func",
                       function_id="1",
                       parent_function_name="parent_func",
                       parent_function_id="2",
                       UUID="uuid",
                       framework="pydantic")
    # When assigning a new non-string value, it should be cast to string.
    row.llm_text_input = 9876
    assert isinstance(row.llm_text_input, str)
    assert row.llm_text_input == "9876"


@pytest.mark.asyncio
async def test_average_workflow_runtime(minimal_eval_config):
    """
    Test that ProfilerRunner correctly computes average workflow runtime (difference between
    the earliest and latest event_timestamp in a request).
    We'll simulate two requests with known event times, confirm the 'mean' in
    'workflow_run_time_confidence_intervals' is correct.
    """

    # Build a DataFrame to mimic final "evaluation" dataframe that ProfilerRunner expects
    # Each row has a usage_stats list with LLM_START and LLM_END events
    # For the 1st request: Start=100.0, End=105.0 => workflow runtime=5.0
    # For the 2nd request: Start=200.0, End=206.0 => workflow runtime=6.0
    # => average run time = 5.5
    events = [
        [
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(function_name="llama-3", function_id="u1"),
                payload=IntermediateStepPayload(
                    event_type=WorkflowEventEnum.LLM_START,
                    event_timestamp=100.0,
                    framework=LLMFrameworkEnum.LANGCHAIN,
                ),
            ),
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(function_name="llama-3", function_id="u1"),
                payload=IntermediateStepPayload(
                    event_type=WorkflowEventEnum.LLM_END,
                    event_timestamp=105.0,
                    framework=LLMFrameworkEnum.LANGCHAIN,
                ),
            ),
        ],
        [
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(function_name="llama-3", function_id="u1"),
                payload=IntermediateStepPayload(
                    event_type=WorkflowEventEnum.LLM_START,
                    event_timestamp=200.0,
                    framework=LLMFrameworkEnum.LLAMA_INDEX,
                ),
            ),
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(function_name="llama-3", function_id="u1"),
                payload=IntermediateStepPayload(
                    event_type=WorkflowEventEnum.LLM_END,
                    event_timestamp=206.0,
                    framework=LLMFrameworkEnum.LLAMA_INDEX,
                ),
            ),
        ],
    ]

    # Initialize the ProfilerRunner
    runner = ProfilerRunner(minimal_eval_config.general.profiler,
                            minimal_eval_config.general.output_dir,
                            write_output=True)

    # Run
    await runner.run(events)

    # The runner writes 'inference_metrics.json' in output_dir
    # Let's parse it and check the "workflow_run_time_confidence_intervals" "mean"
    metrics_path = os.path.join(minimal_eval_config.general.output_dir, "inference_optimization.json")
    assert os.path.exists(metrics_path), "ProfilerRunner did not produce an simple_inference_metrics.json file."

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Grab the 90/95/99 intervals object for workflow run time
    wflow_stats = metrics["confidence_intervals"].get("workflow_run_time_confidence_intervals", {})
    # The 'mean' should be 5.5
    assert abs(wflow_stats.get("mean", -1) - 5.5) < 1e-6, \
        f"Expected mean workflow runtime=5.5, got {wflow_stats.get('mean')}"


@pytest.mark.asyncio
async def test_average_llm_latency(minimal_eval_config):
    """
    Test that ProfilerRunner correctly computes average LLM latency (LLM_END - LLM_START).
    We'll put different frameworks in usage_stats (langchain, llama_index).
    We'll simulate a distinct latency per request, confirm the result is correct.
    """

    # 1st request: LLM_START=50.0, LLM_END=55.5 => latency=5.5
    # 2nd request: LLM_START=60.0, LLM_END=66.0 => latency=6.0
    # => average latency across requests = 5.75           }]

    events = [
        [
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(function_name="llama-3", function_id="u1"),
                payload=IntermediateStepPayload(
                    event_type=WorkflowEventEnum.LLM_START,
                    event_timestamp=50.0,
                    framework=LLMFrameworkEnum.LANGCHAIN,
                ),
            ),
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(function_name="llama-3", function_id="u1"),
                payload=IntermediateStepPayload(
                    event_type=WorkflowEventEnum.LLM_END,
                    event_timestamp=55.5,
                    framework=LLMFrameworkEnum.LANGCHAIN,
                ),
            ),
        ],
        [
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(function_name="llama-3", function_id="u1"),
                payload=IntermediateStepPayload(
                    event_type=WorkflowEventEnum.LLM_START,
                    event_timestamp=60.0,
                    framework=LLMFrameworkEnum.LLAMA_INDEX,
                ),
            ),
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(function_name="llama-3", function_id="u1"),
                payload=IntermediateStepPayload(
                    event_type=WorkflowEventEnum.LLM_END,
                    event_timestamp=66.0,
                    framework=LLMFrameworkEnum.LLAMA_INDEX,
                ),
            ),
        ],
    ]

    runner = ProfilerRunner(minimal_eval_config.general.profiler,
                            minimal_eval_config.general.output_dir,
                            write_output=True)
    await runner.run(events)

    metrics_path = os.path.join(minimal_eval_config.general.output_dir, "inference_optimization.json")
    assert os.path.exists(metrics_path), "ProfilerRunner did not produce an simple_inference_metrics.json file."

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    llm_stats = metrics["confidence_intervals"].get("llm_latency_confidence_intervals", {})
    # We expect the average = (5.5 + 6.0) / 2 = 5.75
    computed_mean = llm_stats.get("mean", -1)
    assert (abs(computed_mean - 5.75) < 1e-6), f"Expected mean=5.75 for LLM latency, got {computed_mean}"
