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

from nat.plugins.phoenix.scripts.export_trajectory_to_phoenix.atif_trajectory_exporter import ATIFTrajectorySpanExporter


def _trajectory(root_parent_id: str) -> dict[str, Any]:
    """User question, an LLM step that calls one tool, and the final LLM answer.

    Every step and tool call names ``root_parent_id`` as its parent.
    """

    def ancestry(function_id: str, function_name: str) -> dict[str, Any]:
        return {
            "function_id": function_id,
            "function_name": function_name,
            "parent_id": root_parent_id,
            "parent_name": "my_agent",
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


@pytest.mark.parametrize("root_parent_id", ["root", "agent-1"])
def test_top_level_spans_are_children_of_the_workflow_span(root_parent_id: str) -> None:
    """``agent-1`` is the root agent scope of an ATOF-converted trajectory, which is not a span."""
    spans = ATIFTrajectorySpanExporter().convert(_trajectory(root_parent_id))

    root, *children = spans
    assert root.parent is None
    assert [span.name for span in children] == ["llm", "pow", "llm"]
    for span in children:
        assert span.parent is not None, f"{span.name} has no parent"
        assert span.parent.context.span_id == root.context.span_id
