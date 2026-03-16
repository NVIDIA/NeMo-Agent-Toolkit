# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for profiler ATIF path — uses ATIF (Trajectory) internally."""

import math
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from nat.data_models.atif import Agent
from nat.data_models.atif import Metrics
from nat.data_models.atif import Observation
from nat.data_models.atif import ObservationResult
from nat.data_models.atif import Step
from nat.data_models.atif import ToolCall
from nat.data_models.atif import Trajectory
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.profiler import ProfilerConfig
from nat.data_models.token_usage import TokenUsageBaseModel
from nat.plugins.profiler.atif_dataframe import create_dataframe_from_atif
from nat.plugins.profiler.atif_dataframe import dataframe_to_profiler_traces
from nat.plugins.profiler.profile_runner import ProfilerRunner
from nat.utils.atif_converter import IntermediateStepToATIFConverter


def _make_intermediate_step(
    event_type: IntermediateStepType,
    *,
    input_data: str | None = None,
    output_data: str | None = None,
    name: str | None = None,
    usage: UsageInfo | None = None,
) -> IntermediateStep:
    """Create a minimal IntermediateStep for testing."""
    return IntermediateStep(
        parent_id="root",
        function_ancestry=InvocationNode(function_id="root", function_name="root"),
        payload=IntermediateStepPayload(
            event_type=event_type,
            event_timestamp=1700000000.0,
            name=name,
            data=StreamEventData(input=input_data, output=output_data),
            usage_info=usage,
        ),
    )


def _make_usage(prompt: int = 50, completion: int = 10) -> UsageInfo:
    """Create UsageInfo with token counts."""
    return UsageInfo(token_usage=TokenUsageBaseModel(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    ), )


@pytest.fixture
def profiler_config():
    """Minimal profiler config."""
    return ProfilerConfig()


@pytest.fixture
def temp_output_dir():
    """Temporary directory for profiler output."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.mark.asyncio
async def test_profiler_accepts_intermediate_steps_converts_to_atif(profiler_config, temp_output_dir):
    """Profiler accepts list[list[IntermediateStep]], converts to ATIF internally."""
    steps = [
        _make_intermediate_step(
            IntermediateStepType.WORKFLOW_START,
            input_data="Hello",
        ),
        _make_intermediate_step(
            IntermediateStepType.LLM_END,
            name="gpt-4",
            output_data="Hi there!",
            usage=_make_usage(50, 10),
        ),
        _make_intermediate_step(
            IntermediateStepType.WORKFLOW_END,
            output_data="Hi there!",
        ),
    ]
    all_steps = [steps]

    runner = ProfilerRunner(profiler_config, temp_output_dir, write_output=True)
    result = await runner.run(all_steps)

    assert result is not None
    assert result.llm_latency_ci is not None


@pytest.mark.asyncio
async def test_profiler_accepts_atif_trajectories_directly(profiler_config, temp_output_dir):
    """Profiler accepts list[Trajectory] and uses ATIF internally."""
    traj = Trajectory(
        agent=Agent(name="test", version="0.0.0"),
        steps=[
            Step(
                step_id=1,
                source="user",
                message="What is 2+2?",
                timestamp="2024-01-01T12:00:00+00:00",
            ),
            Step(
                step_id=2,
                source="agent",
                message="The answer is 4",
                timestamp="2024-01-01T12:00:01+00:00",
                model_name="gpt-4",
                metrics=Metrics(prompt_tokens=100, completion_tokens=20),
            ),
        ],
    )
    trajectories = [traj]

    runner = ProfilerRunner(profiler_config, temp_output_dir, write_output=True)
    result = await runner.run(trajectories)

    assert result is not None
    assert result.llm_latency_ci is not None


@pytest.mark.asyncio
async def test_profiler_intermediate_and_atif_produce_same_structure(profiler_config, temp_output_dir):
    """IntermediateStep and Trajectory inputs both complete; ATIF used internally."""
    ist_steps = [
        _make_intermediate_step(
            IntermediateStepType.WORKFLOW_START,
            input_data="Hi",
        ),
        _make_intermediate_step(
            IntermediateStepType.LLM_END,
            name="gpt-4",
            output_data="Hello!",
            usage=_make_usage(30, 5),
        ),
        _make_intermediate_step(
            IntermediateStepType.WORKFLOW_END,
            output_data="Hello!",
        ),
    ]
    to_atif = IntermediateStepToATIFConverter()
    atif_traj = to_atif.convert(ist_steps)

    runner = ProfilerRunner(profiler_config, temp_output_dir, write_output=False)

    result_ist = await runner.run([[s for s in ist_steps]])
    result_atif = await runner.run([atif_traj])

    assert result_ist is not None
    assert result_atif is not None
    assert result_ist.llm_latency_ci is not None
    assert result_atif.llm_latency_ci is not None


def test_profiler_preserves_function_ancestry_from_atif():
    """create_dataframe_from_atif preserves function_id/parent_function_id from step.extra."""
    traj = Trajectory(
        agent=Agent(name="test", version="0.0.0"),
        steps=[
            Step(
                step_id=1,
                source="user",
                message="Hi",
                timestamp="2024-01-01T12:00:00+00:00",
                extra={
                    "function_id": "workflow-123",
                    "function_name": "my_workflow",
                    "parent_function_id": "root",
                    "parent_function_name": "",
                },
            ),
            Step(
                step_id=2,
                source="agent",
                message="Hello!",
                timestamp="2024-01-01T12:00:01+00:00",
                model_name="gpt-4",
                metrics=Metrics(prompt_tokens=50, completion_tokens=10),
                extra={
                    "function_id": "workflow-123",
                    "function_name": "my_workflow",
                    "parent_function_id": "",
                    "parent_function_name": "",
                    "framework": "langchain",
                },
            ),
        ],
    )
    df = create_dataframe_from_atif([traj])
    assert "function_id" in df.columns
    assert (df["function_id"] == "workflow-123").all()
    assert "framework" in df.columns
    llm_rows = df[df["event_type"] == IntermediateStepType.LLM_END]
    assert (llm_rows["framework"] == "langchain").all()


def test_create_dataframe_from_atif_emits_tool_end_rows():
    """create_dataframe_from_atif emits TOOL_END rows when agent has tool_calls + observation."""
    traj = Trajectory(
        agent=Agent(name="test", version="0.0.0"),
        steps=[
            Step(
                step_id=1,
                source="user",
                message="Calculate 2+2",
                timestamp="2024-01-01T12:00:00+00:00",
            ),
            Step(
                step_id=2,
                source="agent",
                message="",
                timestamp="2024-01-01T12:00:01+00:00",
                tool_calls=[
                    ToolCall(tool_call_id="tc-1", function_name="calculator", arguments={"expr": "2+2"}),
                    ToolCall(tool_call_id="tc-2", function_name="validator", arguments={}),
                ],
                observation=Observation(results=[
                    ObservationResult(source_call_id="tc-1", content="4"),
                    ObservationResult(source_call_id="tc-2", content="ok"),
                ]),
            ),
        ],
    )
    df = create_dataframe_from_atif([traj])
    tool_rows = df[df["event_type"] == IntermediateStepType.TOOL_END]
    assert len(tool_rows) == 2
    assert list(tool_rows["tool_name"]) == ["calculator", "validator"]
    assert list(tool_rows["llm_text_output"]) == ["4", "ok"]


def test_create_dataframe_from_atif_includes_cached_tokens_in_total():
    """create_dataframe_from_atif includes cached_tokens in total_tokens for LLM_END rows."""
    traj = Trajectory(
        agent=Agent(name="test", version="0.0.0"),
        steps=[
            Step(
                step_id=1,
                source="user",
                message="Hi",
                timestamp="2024-01-01T12:00:00+00:00",
            ),
            Step(
                step_id=2,
                source="agent",
                message="Hello!",
                timestamp="2024-01-01T12:00:01+00:00",
                model_name="gpt-4",
                metrics=Metrics(
                    prompt_tokens=100,
                    completion_tokens=20,
                    cached_tokens=50,
                ),
            ),
        ],
    )
    df = create_dataframe_from_atif([traj])
    llm_end_rows = df[df["event_type"] == IntermediateStepType.LLM_END]
    assert len(llm_end_rows) == 1
    # total = prompt + completion + cached = 100 + 20 + 50 = 170
    assert llm_end_rows["total_tokens"].iloc[0] == 170


def test_create_dataframe_from_atif_emits_workflow_end_for_message_only_agent():
    """create_dataframe_from_atif emits WORKFLOW_END when agent has message but no metrics/tools."""
    traj = Trajectory(
        agent=Agent(name="test", version="0.0.0"),
        steps=[
            Step(
                step_id=1,
                source="user",
                message="Say hello",
                timestamp="2024-01-01T12:00:00+00:00",
            ),
            Step(
                step_id=2,
                source="agent",
                message="Hello!",
                timestamp="2024-01-01T12:00:01+00:00",  # No metrics, no tool_calls
            ),
        ],
    )
    df = create_dataframe_from_atif([traj])
    assert IntermediateStepType.WORKFLOW_END in df["event_type"].values
    wf_end = df[df["event_type"] == IntermediateStepType.WORKFLOW_END]
    assert len(wf_end) == 1
    assert wf_end["llm_text_output"].iloc[0] == "Hello!"


def test_dataframe_to_profiler_traces():
    """dataframe_to_profiler_traces returns (traces_dict, traces_obj) with correct structure."""
    traj = Trajectory(
        agent=Agent(name="test", version="0.0.0"),
        steps=[
            Step(
                step_id=1,
                source="user",
                message="Hi",
                timestamp="2024-01-01T12:00:00+00:00",
            ),
            Step(
                step_id=2,
                source="agent",
                message="Hello!",
                timestamp="2024-01-01T12:00:01+00:00",
                model_name="gpt-4",
                metrics=Metrics(prompt_tokens=50, completion_tokens=10),
            ),
        ],
    )
    df = create_dataframe_from_atif([traj])
    traces_dict, traces_obj = dataframe_to_profiler_traces(df)
    assert len(traces_dict) == 1
    assert len(traces_obj) == 1
    steps_dict = traces_dict[0]
    assert len(steps_dict) >= 2  # WORKFLOW_START + LLM_START + LLM_END
    step = steps_dict[0]
    assert "event_timestamp" in step
    assert "event_type" in step
    assert "function_name" in step
    assert "function_ancestry" in step
    assert "usage_info" in step
    total = step["usage_info"]["token_usage"]["total_tokens"]
    assert total in (0, 60) or math.isnan(total)  # 0, 60, or NaN for non-LLM rows
    obj = traces_obj[0][0]
    assert hasattr(obj, "event_type")
    assert hasattr(obj, "function_ancestry")


def test_dataframe_to_profiler_traces_empty():
    """dataframe_to_profiler_traces returns ([], []) for empty DataFrame."""
    traces_dict, traces_obj = dataframe_to_profiler_traces(pd.DataFrame())
    assert traces_dict == []
    assert traces_obj == []
