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

from unittest.mock import patch

import pytest
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.graph.graph import CompiledGraph

from nat.agent.base import AgentDecision
from nat.agent.rewoo_agent.agent import NO_INPUT_ERROR_MESSAGE
from nat.agent.rewoo_agent.agent import TOOL_NOT_FOUND_ERROR_MESSAGE
from nat.agent.rewoo_agent.agent import ReWOOAgentGraph
from nat.agent.rewoo_agent.agent import ReWOOGraphState
from nat.agent.rewoo_agent.prompt import rewoo_planner_prompt
from nat.agent.rewoo_agent.prompt import rewoo_solver_prompt
from nat.agent.rewoo_agent.register import ReWOOAgentWorkflowConfig


async def test_state_schema():
    state = ReWOOGraphState()

    assert isinstance(state.task, HumanMessage)
    assert isinstance(state.plan, AIMessage)
    assert isinstance(state.steps, AIMessage)
    assert isinstance(state.intermediate_results, dict)
    assert isinstance(state.result, AIMessage)


@pytest.fixture(name='mock_config_rewoo_agent', scope="module")
def mock_config():
    return ReWOOAgentWorkflowConfig(tool_names=["mock_tool_A", "mock_tool_B"], llm_name="llm", verbose=True)


def test_rewoo_init(mock_config_rewoo_agent, mock_llm, mock_tool):
    tools = [mock_tool('mock_tool_A'), mock_tool('mock_tool_B')]
    planner_prompt = rewoo_planner_prompt
    solver_prompt = rewoo_solver_prompt
    agent = ReWOOAgentGraph(llm=mock_llm,
                            planner_prompt=planner_prompt,
                            solver_prompt=solver_prompt,
                            tools=tools,
                            detailed_logs=mock_config_rewoo_agent.verbose)
    assert isinstance(agent, ReWOOAgentGraph)
    assert agent.llm == mock_llm
    assert agent.solver_prompt == solver_prompt
    assert agent.tools == tools
    assert agent.detailed_logs == mock_config_rewoo_agent.verbose


@pytest.fixture(name='mock_rewoo_agent', scope="module")
def mock_agent(mock_config_rewoo_agent, mock_llm, mock_tool):
    tools = [mock_tool('mock_tool_A'), mock_tool('mock_tool_B')]
    planner_prompt = rewoo_planner_prompt
    solver_prompt = rewoo_solver_prompt
    agent = ReWOOAgentGraph(llm=mock_llm,
                            planner_prompt=planner_prompt,
                            solver_prompt=solver_prompt,
                            tools=tools,
                            detailed_logs=mock_config_rewoo_agent.verbose)
    return agent


async def test_build_graph(mock_rewoo_agent):
    graph = await mock_rewoo_agent.build_graph()
    assert isinstance(graph, CompiledGraph)
    assert list(graph.nodes.keys()) == ['__start__', 'planner', 'executor', 'solver']
    assert graph.builder.edges == {('planner', 'executor'), ('__start__', 'planner'), ('solver', '__end__')}
    assert set(graph.builder.branches.get('executor').get('conditional_edge').ends.keys()) == {
        AgentDecision.TOOL, AgentDecision.END
    }


async def test_planner_node_no_input(mock_rewoo_agent):
    state = await mock_rewoo_agent.planner_node(ReWOOGraphState())
    assert state["result"] == NO_INPUT_ERROR_MESSAGE


async def test_conditional_edge_no_input(mock_rewoo_agent):
    # if the state.steps is empty, the conditional_edge should return END
    decision = await mock_rewoo_agent.conditional_edge(ReWOOGraphState())
    assert decision == AgentDecision.END


def _create_step_info(plan: str, placeholder: str, tool: str, tool_input: str | dict) -> dict:
    return {"plan": plan, "evidence": {"placeholder": placeholder, "tool": tool, "tool_input": tool_input}}


async def test_conditional_edge_decisions(mock_rewoo_agent):
    mock_state = ReWOOGraphState(task=HumanMessage(content="This is a task"),
                                 plan=AIMessage(content="This is the plan"),
                                 steps=AIMessage(content=[
                                     _create_step_info("step1", "#E1", "mock_tool_A", "arg1, arg2"),
                                     _create_step_info("step2", "#E2", "mock_tool_B", "arg3, arg4"),
                                     _create_step_info("step3", "#E3", "mock_tool_A", "arg5, arg6")
                                 ]))
    decision = await mock_rewoo_agent.conditional_edge(mock_state)
    assert decision == AgentDecision.TOOL

    mock_state.intermediate_results = {
        '#E1': ToolMessage(content="result1", tool_call_id="mock_tool_A")
    }  # Added tool_call_id)}
    decision = await mock_rewoo_agent.conditional_edge(mock_state)
    assert decision == AgentDecision.TOOL

    # Now all the steps have been executed and generated intermediate results
    mock_state.intermediate_results = {
        '#E1': ToolMessage(content="result1", tool_call_id="mock_tool_A"),
        '#E2': ToolMessage(content="result2", tool_call_id="mock_tool_B"),
        '#E3': ToolMessage(content="result3", tool_call_id="mock_tool_A")
    }
    decision = await mock_rewoo_agent.conditional_edge(mock_state)
    assert decision == AgentDecision.END


async def test_executor_node_with_not_configured_tool(mock_rewoo_agent):
    tool_not_configured = 'Tool not configured'
    mock_state = ReWOOGraphState(
        task=HumanMessage(content="This is a task"),
        plan=AIMessage(content="This is the plan"),
        steps=AIMessage(content=[
            _create_step_info("step1", "#E1", "mock_tool_A", "arg1, arg2"),
            _create_step_info("step2", "#E2", tool_not_configured, "arg3, arg4")
        ]),
        intermediate_results={"#E1": ToolMessage(content="result1", tool_call_id="mock_tool_A")})
    state = await mock_rewoo_agent.executor_node(mock_state)
    assert isinstance(state, dict)
    configured_tool_names = ['mock_tool_A', 'mock_tool_B']
    assert state["intermediate_results"]["#E2"].content == TOOL_NOT_FOUND_ERROR_MESSAGE.format(
        tool_name=tool_not_configured, tools=configured_tool_names)


async def test_executor_node_parse_input(mock_rewoo_agent):
    from nat.agent.base import AGENT_LOG_PREFIX
    with patch('nat.agent.rewoo_agent.agent.logger.debug') as mock_logger_debug:
        # Test with dict as tool input
        mock_state = ReWOOGraphState(
            task=HumanMessage(content="This is a task"),
            plan=AIMessage(content="This is the plan"),
            steps=AIMessage(content=[
                _create_step_info(
                    "step1",
                    "#E1",
                    "mock_tool_A", {
                        "query": "What is the capital of France?", "input_metadata": {
                            "entities": ["France", "Paris"]
                        }
                    })
            ]),
            intermediate_results={})
        await mock_rewoo_agent.executor_node(mock_state)
        mock_logger_debug.assert_any_call("%s Tool input is already a dictionary. Use the tool input as is.",
                                          AGENT_LOG_PREFIX)

        # Test with valid JSON as tool input
        mock_state = ReWOOGraphState(
            task=HumanMessage(content="This is a task"),
            plan=AIMessage(content="This is the plan"),
            steps=AIMessage(content=[
                _create_step_info(
                    "step1",
                    "#E1",
                    "mock_tool_A",
                    '{"query": "What is the capital of France?", "input_metadata": {"entities": ["France", "Paris"]}}')
            ]),
            intermediate_results={})
        await mock_rewoo_agent.executor_node(mock_state)
        mock_logger_debug.assert_any_call("%s Successfully parsed structured tool input", AGENT_LOG_PREFIX)

        # Test with string with single quote as tool input
        mock_state.steps = AIMessage(
            content=[_create_step_info("step1", "#E1", "mock_tool_A", "{'arg1': 'arg_1', 'arg2': 'arg_2'}")])
        mock_state.intermediate_results = {}
        await mock_rewoo_agent.executor_node(mock_state)
        mock_logger_debug.assert_any_call(
            "%s Successfully parsed structured tool input after replacing single quotes with double quotes",
            AGENT_LOG_PREFIX)

        # Test with string that cannot be parsed as a JSON as tool input
        mock_state.steps = AIMessage(content=[_create_step_info("step1", "#E1", "mock_tool_A", "arg1, arg2")])
        mock_state.intermediate_results = {}
        await mock_rewoo_agent.executor_node(mock_state)
        mock_logger_debug.assert_any_call("%s Unable to parse structured tool input. Using raw tool input as is.",
                                          AGENT_LOG_PREFIX)


async def test_executor_node_handle_input_types(mock_rewoo_agent):
    # mock_tool returns the input query as is.
    # The executor_node should maintain the output type the same as the input type.

    mock_state = ReWOOGraphState(task=HumanMessage(content="This is a task"),
                                 plan=AIMessage(content="This is the plan"),
                                 steps=AIMessage(content=[
                                     _create_step_info("step1", "#E1", "mock_tool_A", "This is a string query"),
                                     _create_step_info("step2", "#E2", "mock_tool_B", "arg3, arg4")
                                 ]),
                                 intermediate_results={})
    await mock_rewoo_agent.executor_node(mock_state)
    assert isinstance(mock_state.intermediate_results["#E1"].content, str)
    # Call executor node again to make sure the intermediate result is correctly processed in the next step
    await mock_rewoo_agent.executor_node(mock_state)
    assert isinstance(mock_state.intermediate_results["#E2"].content, str)

    mock_state = ReWOOGraphState(
        task=HumanMessage(content="This is a task"),
        plan=AIMessage(content="This is the plan"),
        steps=AIMessage(content=[
            _create_step_info("step1",
                              "#E1",
                              "mock_tool_A", {"query": {
                                  "data": "This is a dict query", "metadata": {
                                      "key": "value"
                                  }
                              }}),
            _create_step_info("step2", "#E2", "mock_tool_B", {"query": "#E1"})
        ]),
        intermediate_results={})
    await mock_rewoo_agent.executor_node(mock_state)
    # The actual behavior is that dict input gets converted to string representation
    # and stored as string content in ToolMessage
    assert isinstance(mock_state.intermediate_results["#E1"].content, str)
    # Call executor node again to make sure the intermediate result is correctly processed in the next step
    await mock_rewoo_agent.executor_node(mock_state)
    assert isinstance(mock_state.intermediate_results["#E2"].content, str)


async def test_executor_node_should_not_be_invoked_after_all_steps_executed(mock_rewoo_agent):
    mock_state = ReWOOGraphState(task=HumanMessage(content="This is a task"),
                                 plan=AIMessage(content="This is the plan"),
                                 steps=AIMessage(content=[
                                     _create_step_info("step1", "#E1", "mock_tool_A", "arg1, arg2"),
                                     _create_step_info("step2", "#E2", "mock_tool_B", "arg3, arg4"),
                                     _create_step_info("step3", "#E3", "mock_tool_A", "arg5, arg6")
                                 ]),
                                 intermediate_results={
                                     '#E1': ToolMessage(content='result1', tool_call_id='mock_tool_A'),
                                     '#E2': ToolMessage(content='result2', tool_call_id='mock_tool_B'),
                                     '#E3': ToolMessage(content='result3', tool_call_id='mock_tool_A')
                                 })
    # After executing all the steps, the executor_node should not be invoked
    with pytest.raises(RuntimeError):
        await mock_rewoo_agent.executor_node(mock_state)


def test_validate_planner_prompt_no_input():
    mock_prompt = ''
    with pytest.raises(ValueError):
        ReWOOAgentGraph.validate_planner_prompt(mock_prompt)


def test_validate_planner_prompt_no_tools():
    mock_prompt = '{tools}'
    with pytest.raises(ValueError):
        ReWOOAgentGraph.validate_planner_prompt(mock_prompt)


def test_validate_planner_prompt_no_tool_names():
    mock_prompt = '{tool_names}'
    with pytest.raises(ValueError):
        ReWOOAgentGraph.validate_planner_prompt(mock_prompt)


def test_validate_planner_prompt():
    mock_prompt = '{tools} {tool_names}'
    assert ReWOOAgentGraph.validate_planner_prompt(mock_prompt)


def test_validate_solver_prompt_no_input():
    mock_prompt = ''
    with pytest.raises(ValueError):
        ReWOOAgentGraph.validate_solver_prompt(mock_prompt)


def test_validate_solver_prompt():
    mock_prompt = 'solve the problem'
    assert ReWOOAgentGraph.validate_solver_prompt(mock_prompt)
