# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

import pytest

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.invocation_node import InvocationNode
from nat.plugins.phoenix.scripts.export_trajectory_to_phoenix.atif_trajectory_exporter import ATIFTrajectorySpanExporter
from nat.utils.atif_converter import IntermediateStepToATIFConverter


def _trajectory(root_parent_id: str | None) -> dict[str, Any]:
    """User question, an LLM step that calls one tool, and the final LLM answer.

    Every step and tool call names ``root_parent_id`` as its parent.
    """

    def ancestry(function_id: str, function_name: str) -> dict[str, Any]:
        return {
            "function_id": function_id,
            "function_name": function_name,
            "parent_id": root_parent_id,
            "parent_name": None if root_parent_id is None else "my_agent",
        }

    return {
        "session_id":
            "session-1",
        "agent": {
            "name": "my_agent"
        },
        "steps": [
            {
                "step_id": 1, "source": "user", "message": "What is 7 squared?"
            },
            {
                "step_id": 2,
                "source": "agent",
                "message": "",
                "extra": {
                    "ancestry": ancestry("llm-1", "llm"),
                    "invocation": {
                        "start_timestamp": 1.0, "end_timestamp": 2.0
                    },
                },
                "tool_calls": [{
                    "tool_call_id": "call-1",
                    "function_name": "pow",
                    "arguments": {
                        "base": 7, "exp": 2
                    },
                    "extra": {
                        "ancestry": ancestry("tool-1", "pow"),
                        "invocation": {
                            "start_timestamp": 2.0, "end_timestamp": 3.0
                        },
                    },
                }],
                "observation": {
                    "results": [{
                        "content": "49"
                    }]
                },
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "7 squared is 49.",
                "extra": {
                    "ancestry": ancestry("llm-2", "llm"),
                    "invocation": {
                        "start_timestamp": 3.0, "end_timestamp": 4.0
                    },
                },
            },
        ],
    }


@pytest.mark.parametrize("root_parent_id", ["root", "agent-1", None])
def test_top_level_spans_are_children_of_the_workflow_span(root_parent_id: str | None) -> None:
    """``agent-1`` is the root agent scope of an ATOF-converted trajectory, which is not a span."""
    spans = ATIFTrajectorySpanExporter().convert(_trajectory(root_parent_id))

    root, *children = spans
    assert root.parent is None
    assert [span.name for span in children] == ["llm", "pow", "llm"]
    for span in children:
        assert span.parent is not None, f"{span.name} has no parent"
        assert span.parent.context.span_id == root.context.span_id


def test_tool_span_keeps_its_llm_parent() -> None:
    """A tool call whose parent is a span of the trajectory stays under that span."""
    trajectory = _trajectory("root")
    tool_ancestry = trajectory["steps"][1]["tool_calls"][0]["extra"]["ancestry"]
    tool_ancestry["parent_id"] = "llm-1"
    tool_ancestry["parent_name"] = "llm"
    spans = ATIFTrajectorySpanExporter().convert(trajectory)

    root, llm_1, tool, llm_2 = spans
    assert [span.name for span in (llm_1, tool, llm_2)] == ["llm", "pow", "llm"]
    assert root.parent is None
    assert tool.parent.context.span_id == llm_1.context.span_id
    for span in (llm_1, llm_2):
        assert span.parent.context.span_id == root.context.span_id


def test_tool_metadata_on_the_step_gives_tool_spans() -> None:
    """IntermediateStepToATIFConverter puts tool ancestry and invocations on the step, not on each tool call."""
    trajectory = _trajectory("root")
    agent_step = trajectory["steps"][1]
    tool_call_extra = agent_step["tool_calls"][0].pop("extra")
    agent_step["extra"]["tool_ancestry"] = [tool_call_extra["ancestry"]]
    agent_step["extra"]["tool_invocations"] = [tool_call_extra["invocation"]]

    spans = ATIFTrajectorySpanExporter().convert(trajectory)

    root, _, tool, _ = spans
    assert tool.name == "pow"
    assert tool.parent.context.span_id == root.context.span_id
    assert (tool.start_time, tool.end_time) == (2_000_000_000, 3_000_000_000)


def test_trajectory_from_the_intermediate_step_converter_has_tool_spans() -> None:

    def step(event_type: IntermediateStepType, name: str, offset: float, **data: Any) -> IntermediateStep:
        # END events also carry their start time, half a second earlier.
        span_start = {"span_event_timestamp": 99.5 + offset} if event_type.endswith("_END") else {}
        payload = IntermediateStepPayload(event_type=event_type,
                                          event_timestamp=100.0 + offset,
                                          name=name,
                                          data=StreamEventData(**data),
                                          **span_start)
        return IntermediateStep(parent_id="root",
                                function_ancestry=InvocationNode(function_name="my_agent", function_id="fn-1"),
                                payload=payload)

    intermediate_steps = [
        step(IntermediateStepType.WORKFLOW_START, "my_agent", 0.0, input="What is 7 squared?"),
        step(IntermediateStepType.LLM_END, "llm", 1.0, output="I need to call pow"),
        step(IntermediateStepType.TOOL_END, "pow", 2.0, input={
            "base": 7, "exp": 2
        }, output="49"),
        step(IntermediateStepType.LLM_END, "llm", 3.0, output="7 squared is 49."),
        step(IntermediateStepType.WORKFLOW_END, "my_agent", 4.0, output="7 squared is 49."),
    ]
    trajectory = IntermediateStepToATIFConverter().convert(intermediate_steps)

    spans = ATIFTrajectorySpanExporter().convert(trajectory.model_dump(mode="json", exclude_none=True))

    assert "pow" in [span.name for span in spans]


def test_workflow_span_covers_a_trajectory_that_ends_on_a_tool_call() -> None:
    trajectory = _trajectory("root")
    # Without the final answer, the pow call (2.0 to 3.0) is the last thing in the trajectory.
    del trajectory["steps"][2:]

    root, _, tool = ATIFTrajectorySpanExporter().convert(trajectory)

    assert tool.end_time == 3_000_000_000
    assert root.end_time == tool.end_time
