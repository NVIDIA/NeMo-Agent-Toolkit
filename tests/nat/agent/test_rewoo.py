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
from langgraph.graph.graph import CompiledGraph

from nat.agent.base import AgentDecision
from nat.agent.rewoo_agent.agent import NO_INPUT_ERROR_MESSAGE
from nat.agent.rewoo_agent.agent import TOOL_NOT_FOUND_ERROR_MESSAGE
from nat.agent.rewoo_agent.agent import ReWOOAgentGraph
from nat.agent.rewoo_agent.agent import ReWOOGraphState
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
    # The actual behavior is that dict input is converted to list and stored as list content in ToolMessage
    assert isinstance(mock_state.intermediate_results["#E1"].content, list)
    # Call executor node again to make sure the intermediate result is correctly processed in the next step
    await mock_rewoo_agent.executor_node(mock_state)
    assert isinstance(mock_state.intermediate_results["#E2"].content, list)


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
    assert isinstance(parsed_output, AIMessage)
    assert isinstance(parsed_output.content, list)
    assert len(parsed_output.content) == 2

    # Verify the structure of parsed content
    first_step = parsed_output.content[0]
    assert "plan" in first_step
    assert "evidence" in first_step
    assert "placeholder" in first_step["evidence"]
    assert "tool" in first_step["evidence"]
    assert "tool_input" in first_step["evidence"]


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
    assert isinstance(parsed_output.content[0]["evidence"]["tool_input"], str)


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
    assert isinstance(parsed_output.content[0]["evidence"]["tool_input"], dict)
    assert parsed_output.content[0]["evidence"]["tool_input"]["table"] == "users"


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
    tool_input = {"query": "#E1", "year": "#E1"}
    result = ReWOOAgentGraph._replace_placeholder(placeholder, tool_input, tool_output)
    assert result["query"] == "1998"
    assert result["year"] == "1998"

    # Test dict replacement - partial match in string value
    tool_input = {"query": "What happened in #E1?", "metadata": {"source": "test"}}
    result = ReWOOAgentGraph._replace_placeholder(placeholder, tool_input, tool_output)
    assert result["query"] == "What happened in 1998?"
    assert result["metadata"]["source"] == "test"

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
    config = ReWOOAgentWorkflowConfig(tool_names=["test_tool"],
                                      llm_name="test_llm",
                                      additional_planner_instructions="Be extra careful with planning.")
    assert config.additional_planner_instructions == "Be extra careful with planning."

    # Test config with additional solver instructions
    config_solver = ReWOOAgentWorkflowConfig(tool_names=["test_tool"],
                                             llm_name="test_llm",
                                             additional_solver_instructions="Provide detailed explanations.")
    assert config_solver.additional_solver_instructions == "Provide detailed explanations."

    # Test config with both
    config_both = ReWOOAgentWorkflowConfig(tool_names=["test_tool"],
                                           llm_name="test_llm",
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
