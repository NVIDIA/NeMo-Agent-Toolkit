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

from nat.sdk import Agent
from nat.sdk import Event
from nat.sdk import Tool
from nat.sdk import ToolResult


@pytest.fixture
def simple_tool() -> Tool:

    async def _add(a: int, b: int) -> ToolResult:
        return ToolResult(output=a + b)

    return Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer"
                },
                "b": {
                    "type": "integer"
                },
            },
            "required": ["a", "b"],
        },
        execute=_add,
    )


@pytest.fixture
def simple_agent(simple_tool: Tool) -> Agent:
    return Agent(
        tools=[simple_tool],
        system_prompt="You are a test assistant.",
    )


@pytest.fixture
def event_collector() -> list[Event]:
    """Collects events emitted during a conversation."""
    return []
