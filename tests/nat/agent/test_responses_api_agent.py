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
# pylint: disable=not-async-context-manager,unused-argument

import pytest

from nat.agent.responses_api_agent.register import ResponsesAPIAgentWorkflowConfig
from nat.agent.responses_api_agent.register import responses_api_agent_workflow
from nat.data_models.openai_mcp import OpenAIMCPSchemaTool


class _MockBuilder:

    def __init__(self, llm, tools):
        self._llm = llm
        self._tools = tools

    async def get_llm(self, llm_name, wrapper_type):
        # match interface and avoid unused warnings
        return self._llm

    async def get_tools(self, tool_names, wrapper_type):
        # match interface and avoid unused warnings
        return self._tools


def _augment_llm_for_responses(llm):
    """Augment the mock LLM class with Responses API properties/methods."""

    klass = type(llm)
    setattr(klass, "use_responses_api", True)
    setattr(klass, "model_name", "mock-openai")

    def bind_tools(self, tools, parallel_tool_calls=False, strict=True):  # noqa: D401
        # Store on class to avoid Pydantic instance attribute restrictions
        klass = type(self)
        # Preserve previously bound tools and merge with new ones
        existing_tools = getattr(klass, "bound_tools", [])
        # Create a set to track tool identity (by id for objects, by value for dicts)
        all_tools = list(existing_tools)
        for tool in tools:
            if tool not in all_tools:
                all_tools.append(tool)
        setattr(klass, "bound_tools", all_tools)
        # Preserve True values for parallel_tool_calls and strict (once True, stays True)
        existing_parallel = getattr(klass, "bound_parallel", False)
        existing_strict = getattr(klass, "bound_strict", False)
        setattr(klass, "bound_parallel", existing_parallel or parallel_tool_calls)
        setattr(klass, "bound_strict", existing_strict or strict)
        return self

    setattr(klass, "bind_tools", bind_tools)
    return llm


def _augment_llm_without_responses(llm):
    """Augment the mock LLM class but mark it as not Responses-capable."""
    klass = type(llm)
    setattr(klass, "use_responses_api", False)
    setattr(klass, "model_name", "mock-openai")
    return llm


@pytest.fixture(name="nat_tool")
def nat_tool_fixture(mock_tool):
    return mock_tool("Tool A")


async def _consume_function_info(gen):
    """Helper to consume a single yield from the async generator and return FunctionInfo."""
    function_info = None
    async for function_info in gen:
        break
    assert function_info is not None
    return function_info


async def test_llm_requires_responses_api(mock_llm, nat_tool):
    llm = _augment_llm_without_responses(mock_llm)
    builder = _MockBuilder(llm=llm, tools=[nat_tool])
    config = ResponsesAPIAgentWorkflowConfig(llm_name="openai_llm", nat_tools=["tool_a"])  # type: ignore[list-item]

    with pytest.raises(AssertionError):
        # The assertion occurs before yielding, when validating the LLM
        async with responses_api_agent_workflow(config, builder):
            pass


async def test_binds_tools_and_runs(mock_llm, nat_tool):
    llm = _augment_llm_for_responses(mock_llm)
    mcp = OpenAIMCPSchemaTool(server_label="deepwiki", server_url="https://mcp.deepwiki.com/mcp")
    builtin = {"type": "code_interpreter", "container": {"type": "auto"}}

    builder = _MockBuilder(llm=llm, tools=[nat_tool])

    config = ResponsesAPIAgentWorkflowConfig(
        llm_name="openai_llm",
        nat_tools=["tool_a"],  # type: ignore[list-item]
        builtin_tools=[builtin],
        mcp_tools=[mcp],
        verbose=True,
        parallel_tool_calls=True,
    )

    async with responses_api_agent_workflow(config, builder) as function_info:
        # Ensure tools were bound on the LLM (nat tool + mcp + builtin)
        assert hasattr(type(llm), "bound_tools")
        bound = type(llm).bound_tools
        assert any(getattr(t, "name", None) == "Tool A" for t in bound)  # NAT tool instance
        assert builtin in bound  # Built-in tool dict
        assert mcp.model_dump() in bound  # MCP tool dict

        # Parallel flag propagated
        assert getattr(type(llm), "bound_parallel", False) is True
        assert getattr(type(llm), "bound_strict", False) is True

        # Invoke the produced function and verify output path works end-to-end
        result = await function_info.single_fn("please, mock tool call!")
        assert isinstance(result, str)
        assert result == "mock query"
