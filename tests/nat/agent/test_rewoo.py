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
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.state import CompiledStateGraph

from nat.agent.base import AgentDecision
from nat.agent.rewoo_agent.agent import NO_INPUT_ERROR_MESSAGE
from nat.agent.rewoo_agent.agent import TOOL_NOT_FOUND_ERROR_MESSAGE
from nat.agent.rewoo_agent.agent import ReWOOAgentGraph
from nat.agent.rewoo_agent.agent import ReWOOEvidence
from nat.agent.rewoo_agent.agent import ReWOOGraphState
from nat.agent.rewoo_agent.agent import ReWOOPlanStep
from nat.agent.rewoo_agent.register import ReWOOAgentWorkflowConfig


async def test_state_schema():
    state = ReWOOGraphState()

    assert isinstance(state.messages, list)
    assert isinstance(state.task, HumanMessage)
    assert isinstance(state.plan, AIMessage)
    assert isinstance(state.steps, AIMessage)
    # New fields for parallel execution
    assert isinstance(state.evidence_map, dict)
    assert isinstance(state.execution_levels, list)
    assert isinstance(state.current_level, int)
    assert state.current_level == 0
    assert isinstance(state.intermediate_results, dict)
    assert isinstance(state.result, AIMessage)


@pytest.fixture(name='mock_config_rewoo_agent', scope="module")
def mock_config():
    return ReWOOAgentWorkflowConfig(tool_names=["mock_tool_A", "mock_tool_B"], llm_name="llm",
                                    verbose=True)  # type: ignore


def test_rewoo_init(mock_config_rewoo_agent, mock_llm, mock_tool):
    from nat.agent.rewoo_agent.prompt import PLANNER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import PLANNER_USER_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_USER_PROMPT

    tools = [mock_tool('mock_tool_A'), mock_tool('mock_tool_B')]
    planner_prompt = ChatPromptTemplate([("system", PLANNER_SYSTEM_PROMPT), ("user", PLANNER_USER_PROMPT)])
    solver_prompt = ChatPromptTemplate([("system", SOLVER_SYSTEM_PROMPT), ("user", SOLVER_USER_PROMPT)])
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
    from nat.agent.rewoo_agent.prompt import PLANNER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import PLANNER_USER_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_USER_PROMPT

    tools = [mock_tool('mock_tool_A'), mock_tool('mock_tool_B')]
    planner_prompt = ChatPromptTemplate([("system", PLANNER_SYSTEM_PROMPT), ("user", PLANNER_USER_PROMPT)])
    solver_prompt = ChatPromptTemplate([("system", SOLVER_SYSTEM_PROMPT), ("user", SOLVER_USER_PROMPT)])
    agent = ReWOOAgentGraph(llm=mock_llm,
                            planner_prompt=planner_prompt,
                            solver_prompt=solver_prompt,
                            tools=tools,
                            detailed_logs=mock_config_rewoo_agent.verbose)
    return agent


async def test_build_graph(mock_rewoo_agent):
    graph = await mock_rewoo_agent.build_graph()
    assert isinstance(graph, CompiledStateGraph)
    assert list(graph.nodes.keys()) == ['__start__', 'planner', 'executor', 'solver']
    assert graph.builder.edges == {('planner', 'executor'), ('__start__', 'planner'), ('solver', '__end__')}
    executor_branches = graph.builder.branches.get('executor')
    if executor_branches:
        conditional_edge = executor_branches.get('conditional_edge')
        if conditional_edge and hasattr(conditional_edge, 'ends') and conditional_edge.ends:
            assert set(conditional_edge.ends.keys()) == {AgentDecision.TOOL, AgentDecision.END}


async def test_planner_node_no_input(mock_rewoo_agent):
    state = await mock_rewoo_agent.planner_node(ReWOOGraphState())
    assert state["result"] == NO_INPUT_ERROR_MESSAGE


async def test_conditional_edge_no_input(mock_rewoo_agent):
    # if the state.steps is empty, the conditional_edge should return END
    decision = await mock_rewoo_agent.conditional_edge(ReWOOGraphState())
    assert decision == AgentDecision.END


def _create_step_info(plan: str, placeholder: str, tool: str, tool_input: str | dict) -> ReWOOPlanStep:
    evidence = ReWOOEvidence(placeholder=placeholder, tool=tool, tool_input=tool_input)
    return ReWOOPlanStep(plan=plan, evidence=evidence)


def _create_mock_state_with_parallel_data(steps: list[ReWOOPlanStep],
                                          intermediate_results: dict | None = None) -> ReWOOGraphState:
    """
    Create a mock ReWOOGraphState with proper evidence_map and execution_levels for testing parallel execution.
    """
    if intermediate_results is None:
        intermediate_results = {}

    # Parse dependencies and create execution levels like the agent does
    evidence_map, execution_levels = ReWOOAgentGraph._parse_planner_dependencies(steps)

    return ReWOOGraphState(
        task=HumanMessage(content="This is a task"),
        plan=AIMessage(content="This is the plan"),
        steps=AIMessage(content=""),  # steps are handled via evidence_map now
        evidence_map=evidence_map,
        execution_levels=execution_levels,
        current_level=0,
        intermediate_results=intermediate_results or {})


async def test_conditional_edge_decisions(mock_rewoo_agent):
    # Create steps without dependencies (parallel execution)
    steps = [
        _create_step_info("step1", "#E1", "mock_tool_A", "arg1, arg2"),
        _create_step_info("step2", "#E2", "mock_tool_B", "arg3, arg4"),
        _create_step_info("step3", "#E3", "mock_tool_A", "arg5, arg6")
    ]

    # Initially no results - should continue with execution
    mock_state = _create_mock_state_with_parallel_data(steps)
    decision = await mock_rewoo_agent.conditional_edge(mock_state)
    assert decision == AgentDecision.TOOL

    # Partially completed level - should continue with execution
    mock_state.intermediate_results = {'#E1': ToolMessage(content="result1", tool_call_id="mock_tool_A")}
    decision = await mock_rewoo_agent.conditional_edge(mock_state)
    assert decision == AgentDecision.TOOL

    # All steps in current level completed - should end
    mock_state.intermediate_results = {
        '#E1': ToolMessage(content="result1", tool_call_id="mock_tool_A"),
        '#E2': ToolMessage(content="result2", tool_call_id="mock_tool_B"),
        '#E3': ToolMessage(content="result3", tool_call_id="mock_tool_A")
    }
    decision = await mock_rewoo_agent.conditional_edge(mock_state)
    assert decision == AgentDecision.END


async def test_executor_node_with_not_configured_tool(mock_rewoo_agent):
    tool_not_configured = 'Tool not configured'
    steps = [
        _create_step_info("step1", "#E1", "mock_tool_A", "arg1, arg2"),
        _create_step_info("step2", "#E2", tool_not_configured, "arg3, arg4")
    ]

    # Create state with first tool already completed, second tool not configured
    intermediate_results = {"#E1": ToolMessage(content="result1", tool_call_id="mock_tool_A")}
    mock_state = _create_mock_state_with_parallel_data(steps, intermediate_results)

    state = await mock_rewoo_agent.executor_node(mock_state)
    assert isinstance(state, dict)
    configured_tool_names = ['mock_tool_A', 'mock_tool_B']
    assert state["intermediate_results"]["#E2"].content == TOOL_NOT_FOUND_ERROR_MESSAGE.format(
        tool_name=tool_not_configured, tools=configured_tool_names)


async def test_executor_node_parse_input(mock_rewoo_agent):
    from nat.agent.base import AGENT_LOG_PREFIX
    with patch('nat.agent.rewoo_agent.agent.logger.debug') as mock_logger_debug:
        # Test with dict as tool input
        steps = [
            _create_step_info(
                "step1",
                "#E1",
                "mock_tool_A", {
                    "query": "What is the capital of France?", "input_metadata": {
                        "entities": ["France", "Paris"]
                    }
                })
        ]
        mock_state = _create_mock_state_with_parallel_data(steps)
        await mock_rewoo_agent.executor_node(mock_state)
        mock_logger_debug.assert_any_call("%s Tool input is already a dictionary. Use the tool input as is.",
                                          AGENT_LOG_PREFIX)

        # Test with valid JSON as tool input
        steps = [
            _create_step_info(
                "step1",
                "#E1",
                "mock_tool_A",
                '{"query": "What is the capital of France?", "input_metadata": {"entities": ["France", "Paris"]}}')
        ]
        mock_state = _create_mock_state_with_parallel_data(steps)
        await mock_rewoo_agent.executor_node(mock_state)
        mock_logger_debug.assert_any_call("%s Successfully parsed structured tool input", AGENT_LOG_PREFIX)

        # Test with string with single quote as tool input
        steps = [_create_step_info("step1", "#E1", "mock_tool_A", "{'arg1': 'arg_1', 'arg2': 'arg_2'}")]
        mock_state = _create_mock_state_with_parallel_data(steps)
        await mock_rewoo_agent.executor_node(mock_state)
        mock_logger_debug.assert_any_call(
            "%s Successfully parsed structured tool input after replacing single quotes with double quotes",
            AGENT_LOG_PREFIX)

        # Test with string that cannot be parsed as a JSON as tool input
        steps = [_create_step_info("step1", "#E1", "mock_tool_A", "arg1, arg2")]
        mock_state = _create_mock_state_with_parallel_data(steps)
        await mock_rewoo_agent.executor_node(mock_state)
        mock_logger_debug.assert_any_call("%s Unable to parse structured tool input. Using raw tool input as is.",
                                          AGENT_LOG_PREFIX)


async def test_executor_node_handle_input_types(mock_rewoo_agent):
    # mock_tool returns the input query as is.
    # The executor_node should maintain the output type the same as the input type.

    # Test with string inputs (parallel execution - both tools run at once)
    steps = [
        _create_step_info("step1", "#E1", "mock_tool_A", "This is a string query"),
        _create_step_info("step2", "#E2", "mock_tool_B", "arg3, arg4")
    ]
    mock_state = _create_mock_state_with_parallel_data(steps)

    result = await mock_rewoo_agent.executor_node(mock_state)
    # Update state with results
    mock_state.intermediate_results.update(result["intermediate_results"])

    assert isinstance(mock_state.intermediate_results["#E1"].content, str)
    assert isinstance(mock_state.intermediate_results["#E2"].content, str)

    # Test with dict inputs and dependencies
    steps = [
        _create_step_info("step1",
                          "#E1",
                          "mock_tool_A", {"query": {
                              "data": "This is a dict query", "metadata": {
                                  "key": "value"
                              }
                          }}),
        _create_step_info("step2", "#E2", "mock_tool_B", {"query": "#E1"})
    ]
    mock_state = _create_mock_state_with_parallel_data(steps)

    # First execution - should run #E1 only (no dependencies)
    result = await mock_rewoo_agent.executor_node(mock_state)
    mock_state.intermediate_results.update(result["intermediate_results"])
    assert isinstance(mock_state.intermediate_results["#E1"].content, list)

    # Second execution - level 0 is complete, should move to level 1
    result = await mock_rewoo_agent.executor_node(mock_state)
    if "current_level" in result:
        mock_state.current_level = result["current_level"]

    # Third execution - now execute level 1 (#E2)
    result = await mock_rewoo_agent.executor_node(mock_state)
    if "intermediate_results" in result:
        mock_state.intermediate_results.update(result["intermediate_results"])
        assert isinstance(mock_state.intermediate_results["#E2"].content, list)
    else:
        # If no intermediate_results returned, #E2 should already be there
        assert "#E2" in mock_state.intermediate_results
        assert isinstance(mock_state.intermediate_results["#E2"].content, list)


async def test_executor_node_should_not_be_invoked_after_all_steps_executed(mock_rewoo_agent):
    steps = [
        _create_step_info("step1", "#E1", "mock_tool_A", "arg1, arg2"),
        _create_step_info("step2", "#E2", "mock_tool_B", "arg3, arg4"),
        _create_step_info("step3", "#E3", "mock_tool_A", "arg5, arg6")
    ]

    intermediate_results = {
        '#E1': ToolMessage(content='result1', tool_call_id='mock_tool_A'),
        '#E2': ToolMessage(content='result2', tool_call_id='mock_tool_B'),
        '#E3': ToolMessage(content='result3', tool_call_id='mock_tool_A')
    }

    mock_state = _create_mock_state_with_parallel_data(steps, intermediate_results)
    # Set current_level to beyond available levels to simulate all complete
    mock_state.current_level = len(mock_state.execution_levels)

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


def test_additional_planner_instructions_are_appended():
    """Test that additional planner instructions are properly appended to the base planner prompt."""
    from nat.agent.rewoo_agent.prompt import PLANNER_SYSTEM_PROMPT

    base_prompt = PLANNER_SYSTEM_PROMPT
    additional_instructions = "\n\nAdditional instruction: Always consider performance implications."

    # Test with additional instructions
    planner_system_prompt_with_additional = base_prompt + additional_instructions
    assert additional_instructions in planner_system_prompt_with_additional
    assert base_prompt in planner_system_prompt_with_additional

    # Verify the prompt still validates
    assert ReWOOAgentGraph.validate_planner_prompt(planner_system_prompt_with_additional)

    # Test that we can create a valid ChatPromptTemplate with additional instructions
    from nat.agent.rewoo_agent.prompt import PLANNER_USER_PROMPT
    planner_prompt = ChatPromptTemplate([("system", planner_system_prompt_with_additional),
                                         ("user", PLANNER_USER_PROMPT)])
    assert isinstance(planner_prompt, ChatPromptTemplate)


def test_additional_solver_instructions_are_appended():
    """Test that additional solver instructions are properly appended to the base solver prompt."""
    from nat.agent.rewoo_agent.prompt import SOLVER_SYSTEM_PROMPT

    base_prompt = SOLVER_SYSTEM_PROMPT
    additional_instructions = "\n\nAdditional instruction: Provide concise answers."

    # Test with additional instructions
    solver_system_prompt_with_additional = base_prompt + additional_instructions
    assert additional_instructions in solver_system_prompt_with_additional
    assert base_prompt in solver_system_prompt_with_additional

    # Verify the prompt still validates
    assert ReWOOAgentGraph.validate_solver_prompt(solver_system_prompt_with_additional)

    # Test that we can create a valid ChatPromptTemplate with additional instructions
    from nat.agent.rewoo_agent.prompt import SOLVER_USER_PROMPT
    solver_prompt = ChatPromptTemplate([("system", solver_system_prompt_with_additional), ("user", SOLVER_USER_PROMPT)])
    assert isinstance(solver_prompt, ChatPromptTemplate)


def test_prompt_validation_with_additional_instructions():
    """Test that prompt validation still works correctly when additional instructions are provided."""
    from nat.agent.rewoo_agent.prompt import PLANNER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_SYSTEM_PROMPT

    # Test planner prompt validation with additional instructions
    base_planner_prompt = PLANNER_SYSTEM_PROMPT
    additional_planner_instructions = "\n\nAdditional instruction: Be thorough in planning."
    combined_planner_prompt = base_planner_prompt + additional_planner_instructions

    # Should still be valid because it contains required variables
    assert ReWOOAgentGraph.validate_planner_prompt(combined_planner_prompt)

    # Test with additional instructions that break validation
    broken_additional_instructions = "\n\nThis breaks {tools} formatting"
    # Create a prompt that's missing required variables due to override
    broken_planner_prompt = "This is a custom prompt without required variables" + broken_additional_instructions
    with pytest.raises(ValueError):
        ReWOOAgentGraph.validate_planner_prompt(broken_planner_prompt)

    # Test solver prompt validation with additional instructions
    base_solver_prompt = SOLVER_SYSTEM_PROMPT
    additional_solver_instructions = "\n\nAdditional instruction: Be concise."
    combined_solver_prompt = base_solver_prompt + additional_solver_instructions

    # Should still be valid
    assert ReWOOAgentGraph.validate_solver_prompt(combined_solver_prompt)


# Tests for tool_call_max_retries option


def test_rewoo_agent_tool_call_max_retries_initialization(mock_llm, mock_tool):
    """Test that ReWOO agent initializes with tool_call_max_retries parameter."""
    from nat.agent.rewoo_agent.prompt import PLANNER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import PLANNER_USER_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_USER_PROMPT

    tools = [mock_tool('test_tool')]
    planner_prompt = ChatPromptTemplate([("system", PLANNER_SYSTEM_PROMPT), ("user", PLANNER_USER_PROMPT)])
    solver_prompt = ChatPromptTemplate([("system", SOLVER_SYSTEM_PROMPT), ("user", SOLVER_USER_PROMPT)])

    # Test default value
    agent = ReWOOAgentGraph(llm=mock_llm, planner_prompt=planner_prompt, solver_prompt=solver_prompt, tools=tools)
    assert agent.tool_call_max_retries == 3

    # Test custom value
    agent_custom = ReWOOAgentGraph(llm=mock_llm,
                                   planner_prompt=planner_prompt,
                                   solver_prompt=solver_prompt,
                                   tools=tools,
                                   tool_call_max_retries=5)
    assert agent_custom.tool_call_max_retries == 5


async def test_executor_node_passes_max_retries_to_call_tool(mock_rewoo_agent):
    """Test that executor_node passes the correct max_retries value to _call_tool."""
    from unittest.mock import AsyncMock

    # Mock the _call_tool method
    original_call_tool = mock_rewoo_agent._call_tool
    mock_rewoo_agent._call_tool = AsyncMock(return_value=ToolMessage(content="success", tool_call_id="mock_tool_A"))

    # Create test state
    steps = [_create_step_info("test step", "#E1", "mock_tool_A", "test input")]
    mock_state = _create_mock_state_with_parallel_data(steps)

    # Execute the node
    await mock_rewoo_agent.executor_node(mock_state)

    # Verify _call_tool was called with correct max_retries parameter
    mock_rewoo_agent._call_tool.assert_called_once()
    call_kwargs = mock_rewoo_agent._call_tool.call_args.kwargs
    assert 'max_retries' in call_kwargs
    assert call_kwargs['max_retries'] == mock_rewoo_agent.tool_call_max_retries

    # Restore original method
    mock_rewoo_agent._call_tool = original_call_tool


def test_rewoo_config_tool_call_max_retries():
    """Test that ReWOOAgentWorkflowConfig includes tool_call_max_retries field."""

    # Test default value
    config = ReWOOAgentWorkflowConfig(tool_names=["test_tool"], llm_name="test_llm")  # type: ignore
    assert hasattr(config, 'tool_call_max_retries')
    assert config.tool_call_max_retries == 3

    # Test custom value
    config_custom = ReWOOAgentWorkflowConfig(tool_names=["test_tool"], llm_name="test_llm",
                                             tool_call_max_retries=7)  # type: ignore
    assert config_custom.tool_call_max_retries == 7


def test_json_output_parsing_valid_format():
    """Test that the planner can parse valid JSON output correctly."""
    import json

    # Test with valid JSON matching the expected format
    valid_json_output = json.dumps([{
        "plan": "Calculate the result of 2023 minus 25.",
        "evidence": {
            "placeholder": "#E1", "tool": "calculator_subtract", "tool_input": [2023, 25]
        }
    },
                                    {
                                        "plan": "Search for information about the result.",
                                        "evidence": {
                                            "placeholder": "#E2",
                                            "tool": "internet_search",
                                            "tool_input": "What happened in year #E1"
                                        }
                                    }])

    # Test that the parsing method works correctly
    parsed_output = ReWOOAgentGraph._parse_planner_output(valid_json_output)
    assert isinstance(parsed_output, list)
    assert len(parsed_output) == 2

    # Verify the structure of parsed content
    first_step = parsed_output[0]
    assert isinstance(first_step, ReWOOPlanStep)
    assert first_step.plan == "Calculate the result of 2023 minus 25."
    assert first_step.evidence.placeholder == "#E1"
    assert first_step.evidence.tool == "calculator_subtract"
    assert first_step.evidence.tool_input == [2023, 25]


def test_json_output_parsing_invalid_format():
    """Test that the planner handles invalid JSON output correctly."""

    # Test with invalid JSON
    invalid_json_output = "This is not valid JSON"
    with pytest.raises(ValueError, match="The output of planner is invalid JSON format"):
        ReWOOAgentGraph._parse_planner_output(invalid_json_output)

    # Test with malformed JSON
    malformed_json = '{"plan": "incomplete json"'
    with pytest.raises(ValueError, match="The output of planner is invalid JSON format"):
        ReWOOAgentGraph._parse_planner_output(malformed_json)

    # Test with empty string
    with pytest.raises(ValueError, match="The output of planner is invalid JSON format"):
        ReWOOAgentGraph._parse_planner_output("")


def test_json_output_parsing_with_string_tool_input():
    """Test parsing JSON output with string tool inputs."""
    import json

    # Test with string tool input
    json_with_string_input = json.dumps([{
        "plan": "Search for the capital of France",
        "evidence": {
            "placeholder": "#E1", "tool": "search_tool", "tool_input": "What is the capital of France?"
        }
    }])

    parsed_output = ReWOOAgentGraph._parse_planner_output(json_with_string_input)
    assert isinstance(parsed_output[0].evidence.tool_input, str)


def test_json_output_parsing_with_dict_tool_input():
    """Test parsing JSON output with dictionary tool inputs."""
    import json

    # Test with dict tool input
    json_with_dict_input = json.dumps([{
        "plan": "Query database for user information",
        "evidence": {
            "placeholder": "#E1",
            "tool": "database_query",
            "tool_input": {
                "table": "users", "filter": {
                    "active": True
                }
            }
        }
    }])

    parsed_output = ReWOOAgentGraph._parse_planner_output(json_with_dict_input)
    assert isinstance(parsed_output[0].evidence.tool_input, dict)
    assert parsed_output[0].evidence.tool_input["table"] == "users"


def test_edge_cases_empty_additional_instructions():
    """Test edge cases with empty additional instructions."""
    from nat.agent.rewoo_agent.prompt import PLANNER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_SYSTEM_PROMPT

    # Test empty string additional instructions
    base_planner_prompt = PLANNER_SYSTEM_PROMPT
    empty_additional_instructions = ""
    combined_planner_prompt = base_planner_prompt + empty_additional_instructions

    # Should still be valid
    assert ReWOOAgentGraph.validate_planner_prompt(combined_planner_prompt)
    assert combined_planner_prompt == base_planner_prompt

    # Test None additional instructions (simulating config.additional_instructions being None)
    # In the actual register.py, None would not be concatenated
    assert ReWOOAgentGraph.validate_planner_prompt(base_planner_prompt)

    # Test for solver prompt as well
    base_solver_prompt = SOLVER_SYSTEM_PROMPT
    combined_solver_prompt = base_solver_prompt + empty_additional_instructions
    assert ReWOOAgentGraph.validate_solver_prompt(combined_solver_prompt)
    assert combined_solver_prompt == base_solver_prompt


def test_edge_cases_whitespace_additional_instructions():
    """Test edge cases with whitespace-only additional instructions."""
    from nat.agent.rewoo_agent.prompt import PLANNER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_SYSTEM_PROMPT

    # Test whitespace-only additional instructions
    whitespace_instructions = "   \n\t  "

    planner_prompt_with_whitespace = PLANNER_SYSTEM_PROMPT + whitespace_instructions
    assert ReWOOAgentGraph.validate_planner_prompt(planner_prompt_with_whitespace)

    solver_prompt_with_whitespace = SOLVER_SYSTEM_PROMPT + whitespace_instructions
    assert ReWOOAgentGraph.validate_solver_prompt(solver_prompt_with_whitespace)


def test_placeholder_replacement_functionality():
    """Test the placeholder replacement functionality with various data types."""

    # Test string replacement
    tool_input = "Search for information about #E1 in the year #E1"
    placeholder = "#E1"
    tool_output = "1998"

    result = ReWOOAgentGraph._replace_placeholder(placeholder, tool_input, tool_output)
    assert result == "Search for information about 1998 in the year 1998"

    # Test dict replacement - exact match
    tool_input_dict = {"query": "#E1", "year": "#E1"}
    result_dict = ReWOOAgentGraph._replace_placeholder(placeholder, tool_input_dict, tool_output)
    assert isinstance(result_dict, dict)
    assert result_dict["query"] == "1998"
    assert result_dict["year"] == "1998"

    # Test dict replacement - partial match in string value
    tool_input_dict2 = {"query": "What happened in #E1?", "metadata": {"source": "test"}}
    result_dict2 = ReWOOAgentGraph._replace_placeholder(placeholder, tool_input_dict2, tool_output)
    assert isinstance(result_dict2, dict)
    assert result_dict2["query"] == "What happened in 1998?"
    assert result_dict2["metadata"]["source"] == "test"

    # Test with complex tool output (dict)
    complex_output = {"result": "France", "confidence": 0.95}
    tool_input = "The capital of the country in #E1"
    result = ReWOOAgentGraph._replace_placeholder("#E1", tool_input, complex_output)
    expected = f"The capital of the country in {str(complex_output)}"
    assert result == expected


def test_tool_input_parsing_edge_cases():
    """Test edge cases in tool input parsing."""

    # Test with valid JSON string
    json_string = '{"key": "value", "number": 42}'
    result = ReWOOAgentGraph._parse_tool_input(json_string)
    assert isinstance(result, dict)
    assert result["key"] == "value"
    assert result["number"] == 42

    # Test with single quotes that get converted
    single_quote_json = "{'key': 'value', 'number': 42}"
    result = ReWOOAgentGraph._parse_tool_input(single_quote_json)
    assert isinstance(result, dict)
    assert result["key"] == "value"

    # Test with raw string that can't be parsed
    raw_string = "just a plain string"
    result = ReWOOAgentGraph._parse_tool_input(raw_string)
    assert result == raw_string

    # Test with dict input (should return as-is)
    dict_input = {"already": "a dict"}
    result = ReWOOAgentGraph._parse_tool_input(dict_input)
    assert result is dict_input

    # Test with malformed JSON
    malformed_json = '{"incomplete": json'
    result = ReWOOAgentGraph._parse_tool_input(malformed_json)
    assert result == malformed_json  # Should fall back to raw string


def test_configuration_integration_with_additional_instructions():
    """Test integration with ReWOOAgentWorkflowConfig for additional instructions."""

    # Test config with additional planner instructions
    config = ReWOOAgentWorkflowConfig(
        tool_names=["test_tool"],  # type: ignore
        llm_name="test_llm",  # type: ignore
        additional_planner_instructions="Be extra careful with planning.")
    assert config.additional_planner_instructions == "Be extra careful with planning."

    # Test config with additional solver instructions
    config_solver = ReWOOAgentWorkflowConfig(
        tool_names=["test_tool"],  # type: ignore
        llm_name="test_llm",  # type: ignore
        additional_solver_instructions="Provide detailed explanations.")
    assert config_solver.additional_solver_instructions == "Provide detailed explanations."

    # Test config with both
    config_both = ReWOOAgentWorkflowConfig(
        tool_names=["test_tool"],  # type: ignore
        llm_name="test_llm",  # type: ignore
        additional_planner_instructions="Plan carefully.",
        additional_solver_instructions="Solve thoroughly.")
    assert config_both.additional_planner_instructions == "Plan carefully."
    assert config_both.additional_solver_instructions == "Solve thoroughly."

    # Test that the validation_alias for additional_planner_instructions works
    # We can't directly test the alias in the constructor since it's used at validation time
    # But we can verify that both field names exist and work correctly
    assert hasattr(config_both, 'additional_planner_instructions')
    assert hasattr(config_both, 'additional_solver_instructions')
    assert config_both.additional_planner_instructions == "Plan carefully."
    assert config_both.additional_solver_instructions == "Solve thoroughly."


# Tests for raise_tool_call_error option


def test_rewoo_config_raise_tool_call_error():
    """Test that ReWOOAgentWorkflowConfig includes raise_tool_call_error field with correct default."""

    # Test default value
    config = ReWOOAgentWorkflowConfig(tool_names=["test_tool"], llm_name="test_llm")  # type: ignore
    assert hasattr(config, 'raise_tool_call_error')
    assert config.raise_tool_call_error is True

    # Test custom value (False)
    config_false = ReWOOAgentWorkflowConfig(tool_names=["test_tool"], llm_name="test_llm",
                                            raise_tool_call_error=False)  # type: ignore
    assert config_false.raise_tool_call_error is False

    # Test custom value (True explicitly)
    config_true = ReWOOAgentWorkflowConfig(tool_names=["test_tool"], llm_name="test_llm",
                                           raise_tool_call_error=True)  # type: ignore
    assert config_true.raise_tool_call_error is True


def test_rewoo_agent_raise_tool_call_error_initialization(mock_llm, mock_tool):
    """Test that ReWOO agent initializes with raise_tool_call_error parameter."""
    from nat.agent.rewoo_agent.prompt import PLANNER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import PLANNER_USER_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_USER_PROMPT

    tools = [mock_tool('test_tool')]
    planner_prompt = ChatPromptTemplate([("system", PLANNER_SYSTEM_PROMPT), ("user", PLANNER_USER_PROMPT)])
    solver_prompt = ChatPromptTemplate([("system", SOLVER_SYSTEM_PROMPT), ("user", SOLVER_USER_PROMPT)])

    # Test default value (True)
    agent = ReWOOAgentGraph(llm=mock_llm, planner_prompt=planner_prompt, solver_prompt=solver_prompt, tools=tools)
    assert agent.raise_tool_call_error is True

    # Test custom value (False)
    agent_false = ReWOOAgentGraph(llm=mock_llm,
                                  planner_prompt=planner_prompt,
                                  solver_prompt=solver_prompt,
                                  tools=tools,
                                  raise_tool_call_error=False)
    assert agent_false.raise_tool_call_error is False

    # Test custom value (True explicitly)
    agent_true = ReWOOAgentGraph(llm=mock_llm,
                                 planner_prompt=planner_prompt,
                                 solver_prompt=solver_prompt,
                                 tools=tools,
                                 raise_tool_call_error=True)
    assert agent_true.raise_tool_call_error is True


async def test_executor_node_raise_tool_call_error_true_behavior(mock_llm, mock_tool):
    """Test that executor_node raises RuntimeError when raise_tool_call_error=True and tool fails."""
    from unittest.mock import AsyncMock

    from nat.agent.rewoo_agent.prompt import PLANNER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import PLANNER_USER_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_USER_PROMPT

    # Create a mock tool that will fail
    failing_tool = mock_tool('failing_tool')
    tools = [failing_tool]
    planner_prompt = ChatPromptTemplate([("system", PLANNER_SYSTEM_PROMPT), ("user", PLANNER_USER_PROMPT)])
    solver_prompt = ChatPromptTemplate([("system", SOLVER_SYSTEM_PROMPT), ("user", SOLVER_USER_PROMPT)])

    # Create agent with raise_tool_call_error=True (default)
    agent = ReWOOAgentGraph(llm=mock_llm,
                            planner_prompt=planner_prompt,
                            solver_prompt=solver_prompt,
                            tools=tools,
                            raise_tool_call_error=True)

    # Mock _call_tool to return an error status
    error_tool_message = ToolMessage(content="Tool call failed after all retry attempts. Last error: Connection failed",
                                     tool_call_id="failing_tool",
                                     status="error")
    agent._call_tool = AsyncMock(return_value=error_tool_message)

    # Create test state
    steps = [_create_step_info("test step", "#E1", "failing_tool", "test input")]
    mock_state = _create_mock_state_with_parallel_data(steps)

    # Should raise RuntimeError when tool fails and raise_tool_call_error=True
    with pytest.raises(RuntimeError, match="Tool call failed"):
        await agent.executor_node(mock_state)


async def test_executor_node_raise_tool_call_error_false_behavior(mock_llm, mock_tool):
    """Test that executor_node continues with error message when raise_tool_call_error=False and tool fails."""
    from unittest.mock import AsyncMock

    from nat.agent.rewoo_agent.prompt import PLANNER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import PLANNER_USER_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_USER_PROMPT

    # Create a mock tool that will fail
    failing_tool = mock_tool('failing_tool')
    tools = [failing_tool]
    planner_prompt = ChatPromptTemplate([("system", PLANNER_SYSTEM_PROMPT), ("user", PLANNER_USER_PROMPT)])
    solver_prompt = ChatPromptTemplate([("system", SOLVER_SYSTEM_PROMPT), ("user", SOLVER_USER_PROMPT)])

    # Create agent with raise_tool_call_error=False
    agent = ReWOOAgentGraph(llm=mock_llm,
                            planner_prompt=planner_prompt,
                            solver_prompt=solver_prompt,
                            tools=tools,
                            raise_tool_call_error=False)

    # Mock _call_tool to return an error status
    error_message = "Tool call failed after all retry attempts. Last error: Connection failed"
    error_tool_message = ToolMessage(content=error_message, tool_call_id="failing_tool", status="error")
    agent._call_tool = AsyncMock(return_value=error_tool_message)

    # Create test state
    steps = [_create_step_info("test step", "#E1", "failing_tool", "test input")]
    mock_state = _create_mock_state_with_parallel_data(steps)

    # Should not raise exception when tool fails and raise_tool_call_error=False
    result = await agent.executor_node(mock_state)

    # Should return intermediate_results with the error message
    assert isinstance(result, dict)
    assert "intermediate_results" in result
    intermediate_results = result["intermediate_results"]
    assert isinstance(intermediate_results, dict)
    assert "#E1" in intermediate_results
    assert intermediate_results["#E1"].content == error_message
    assert intermediate_results["#E1"].status == "error"


async def test_executor_node_raise_tool_call_error_success_case(mock_llm, mock_tool):
    """Test that executor_node behaves normally when tool succeeds, regardless of raise_tool_call_error setting."""
    from unittest.mock import AsyncMock

    from nat.agent.rewoo_agent.prompt import PLANNER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import PLANNER_USER_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_SYSTEM_PROMPT
    from nat.agent.rewoo_agent.prompt import SOLVER_USER_PROMPT

    # Create a mock tool that will succeed
    success_tool = mock_tool('success_tool')
    tools = [success_tool]
    planner_prompt = ChatPromptTemplate([("system", PLANNER_SYSTEM_PROMPT), ("user", PLANNER_USER_PROMPT)])
    solver_prompt = ChatPromptTemplate([("system", SOLVER_SYSTEM_PROMPT), ("user", SOLVER_USER_PROMPT)])

    # Test with both True and False settings
    for raise_error_setting in [True, False]:
        agent = ReWOOAgentGraph(llm=mock_llm,
                                planner_prompt=planner_prompt,
                                solver_prompt=solver_prompt,
                                tools=tools,
                                raise_tool_call_error=raise_error_setting)

        # Mock _call_tool to return a successful response (no status field means success)
        success_tool_message = ToolMessage(content="Success result", tool_call_id="success_tool")
        agent._call_tool = AsyncMock(return_value=success_tool_message)

        # Create test state
        steps = [_create_step_info("test step", "#E1", "success_tool", "test input")]
        mock_state = _create_mock_state_with_parallel_data(steps)

        # Should work normally for successful tool calls regardless of setting
        result = await agent.executor_node(mock_state)

        assert isinstance(result, dict)
        assert "intermediate_results" in result
        intermediate_results = result["intermediate_results"]
        assert isinstance(intermediate_results, dict)
        assert "#E1" in intermediate_results
        assert intermediate_results["#E1"].content == "Success result"
        assert (not hasattr(intermediate_results["#E1"], 'status') or intermediate_results["#E1"].status != "error")


# Tests for new parallel execution functionality


def test_dependency_parsing_sequential():
    """Test dependency parsing for sequential execution."""
    steps = [
        _create_step_info("step1", "#E1", "tool_A", "input1"),
        _create_step_info("step2", "#E2", "tool_B", "#E1"),
        _create_step_info("step3", "#E3", "tool_C", "#E2")
    ]

    evidence_map, execution_levels = ReWOOAgentGraph._parse_planner_dependencies(steps)

    # Should have 3 levels for sequential execution
    assert len(execution_levels) == 3
    assert execution_levels[0] == ["#E1"]
    assert execution_levels[1] == ["#E2"]
    assert execution_levels[2] == ["#E3"]

    # Check evidence map
    assert len(evidence_map) == 3
    assert "#E1" in evidence_map
    assert "#E2" in evidence_map
    assert "#E3" in evidence_map


def test_dependency_parsing_parallel():
    """Test dependency parsing for parallel execution."""
    steps = [
        _create_step_info("step1", "#E1", "tool_A", "input1"),
        _create_step_info("step2", "#E2", "tool_B", "input2"),
        _create_step_info("step3", "#E3", "tool_C", {"combine": ["#E1", "#E2"]})
    ]

    evidence_map, execution_levels = ReWOOAgentGraph._parse_planner_dependencies(steps)

    # Should have 2 levels: E1 and E2 in parallel, then E3
    assert len(execution_levels) == 2
    assert set(execution_levels[0]) == {"#E1", "#E2"}
    assert execution_levels[1] == ["#E3"]

    # Check evidence map
    assert len(evidence_map) == 3


def test_dependency_parsing_complex():
    """Test dependency parsing for complex dependency graph."""
    steps = [
        _create_step_info("step1", "#E1", "tool_A", "input1"),
        _create_step_info("step2", "#E2", "tool_B", "input2"),
        _create_step_info("step3", "#E3", "tool_C", "#E1"),
        _create_step_info("step4", "#E4", "tool_D", "#E2"),
        _create_step_info("step5", "#E5", "tool_E", {"inputs": ["#E3", "#E4"]})
    ]

    evidence_map, execution_levels = ReWOOAgentGraph._parse_planner_dependencies(steps)

    # Should have 3 levels: [E1,E2], [E3,E4], [E5]
    assert len(execution_levels) == 3
    assert set(execution_levels[0]) == {"#E1", "#E2"}
    assert set(execution_levels[1]) == {"#E3", "#E4"}
    assert execution_levels[2] == ["#E5"]


def test_dependency_parsing_circular_error():
    """Test that circular dependencies are detected."""
    steps = [_create_step_info("step1", "#E1", "tool_A", "#E2"), _create_step_info("step2", "#E2", "tool_B", "#E1")]

    with pytest.raises(ValueError, match="Circular dependency detected"):
        ReWOOAgentGraph._parse_planner_dependencies(steps)


def test_get_current_level_status():
    """Test the _get_current_level_status method."""
    steps = [
        _create_step_info("step1", "#E1", "tool_A", "input1"), _create_step_info("step2", "#E2", "tool_B", "input2")
    ]

    state = _create_mock_state_with_parallel_data(steps)

    # Initially at level 0, not complete
    current_level, level_complete = ReWOOAgentGraph._get_current_level_status(state)
    assert current_level == 0
    assert level_complete is False

    # Add one result - still not complete
    state.intermediate_results["#E1"] = ToolMessage(content="result1", tool_call_id="tool_A")
    current_level, level_complete = ReWOOAgentGraph._get_current_level_status(state)
    assert current_level == 0
    assert level_complete is False

    # Add second result - now complete
    state.intermediate_results["#E2"] = ToolMessage(content="result2", tool_call_id="tool_B")
    current_level, level_complete = ReWOOAgentGraph._get_current_level_status(state)
    assert current_level == 0
    assert level_complete is True

    # Move to next level (beyond available levels)
    state.current_level = 1
    current_level, level_complete = ReWOOAgentGraph._get_current_level_status(state)
    assert current_level == -1
    assert level_complete is True


async def test_parallel_execution_flow(mock_rewoo_agent):
    """Test the full parallel execution flow."""
    # Create steps that can be executed in parallel
    steps = [
        _create_step_info("step1", "#E1", "mock_tool_A", "input1"),
        _create_step_info("step2", "#E2", "mock_tool_B", "input2")
    ]

    state = _create_mock_state_with_parallel_data(steps)

    # Execute first time - should process both tools in parallel
    result = await mock_rewoo_agent.executor_node(state)

    # Should return intermediate results for both tools
    assert "intermediate_results" in result
    assert "#E1" in result["intermediate_results"]
    assert "#E2" in result["intermediate_results"]

    # Update state with results
    state.intermediate_results.update(result["intermediate_results"])

    # Check conditional edge - should be END now
    decision = await mock_rewoo_agent.conditional_edge(state)
    assert decision == AgentDecision.END
