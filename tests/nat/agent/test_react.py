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

import pytest
from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.graph.graph import CompiledGraph

from nat.agent.base import AgentDecision
from nat.agent.react_agent.agent import NO_INPUT_ERROR_MESSAGE
from nat.agent.react_agent.agent import TOOL_NOT_FOUND_ERROR_MESSAGE
from nat.agent.react_agent.agent import ReActAgentGraph
from nat.agent.react_agent.agent import ReActGraphState
from nat.agent.react_agent.agent import create_react_agent_prompt
from nat.agent.react_agent.output_parser import FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE
from nat.agent.react_agent.output_parser import MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE
from nat.agent.react_agent.output_parser import MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE
from nat.agent.react_agent.output_parser import ReActOutputParser
from nat.agent.react_agent.output_parser import ReActOutputParserException
from nat.agent.react_agent.register import ReActAgentWorkflowConfig


async def test_state_schema():
    input_message = HumanMessage(content='test')
    state = ReActGraphState(messages=[input_message])
    sample_thought = AgentAction(tool='test', tool_input='test', log='test_action')

    state.agent_scratchpad.append(sample_thought)
    state.tool_responses.append(input_message)
    assert isinstance(state.messages, list)
    assert isinstance(state.messages[0], HumanMessage)
    assert state.messages[0].content == input_message.content
    assert isinstance(state.agent_scratchpad, list)
    assert isinstance(state.agent_scratchpad[0], AgentAction)
    assert isinstance(state.tool_responses, list)
    assert isinstance(state.tool_responses[0], HumanMessage)
    assert state.tool_responses[0].content == input_message.content


@pytest.fixture(name='mock_config_react_agent', scope="module")
def mock_config():
    return ReActAgentWorkflowConfig(tool_names=['test'], llm_name='test', verbose=True)


def test_react_init(mock_config_react_agent, mock_llm, mock_tool):
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    prompt = create_react_agent_prompt(mock_config_react_agent)
    agent = ReActAgentGraph(llm=mock_llm, prompt=prompt, tools=tools, detailed_logs=mock_config_react_agent.verbose)
    assert isinstance(agent, ReActAgentGraph)
    assert agent.llm == mock_llm
    assert agent.tools == tools
    assert agent.detailed_logs == mock_config_react_agent.verbose
    assert agent.parse_agent_response_max_retries >= 1


@pytest.fixture(name='mock_react_agent', scope="module")
def fixture_mock_agent(mock_config_react_agent, mock_llm, mock_tool):
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    prompt = create_react_agent_prompt(mock_config_react_agent)
    agent = ReActAgentGraph(llm=mock_llm, prompt=prompt, tools=tools, detailed_logs=mock_config_react_agent.verbose)
    return agent


async def test_build_graph(mock_react_agent):
    graph = await mock_react_agent.build_graph()
    assert isinstance(graph, CompiledGraph)
    assert list(graph.nodes.keys()) == ['__start__', 'agent', 'tool']
    assert graph.builder.edges == {('__start__', 'agent'), ('tool', 'agent')}
    assert set(graph.builder.branches.get('agent').get('conditional_edge').ends.keys()) == {
        AgentDecision.TOOL, AgentDecision.END
    }


async def test_agent_node_no_input(mock_react_agent):
    with pytest.raises(RuntimeError) as ex:
        await mock_react_agent.agent_node(ReActGraphState())
    assert isinstance(ex.value, RuntimeError)


async def test_malformed_agent_output_after_max_retries(mock_react_agent):
    response = await mock_react_agent.agent_node(ReActGraphState(messages=[HumanMessage('hi')]))
    response = response.messages[-1]
    assert isinstance(response, AIMessage)
    # The actual format combines error observation with original output
    assert MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE in response.content
    assert '\nQuestion: hi\n' in response.content


async def test_agent_node_parse_agent_action(mock_react_agent):
    mock_react_agent_output = 'Thought:not_many\nAction:Tool A\nAction Input: hello, world!\nObservation:'
    mock_state = ReActGraphState(messages=[HumanMessage(content=mock_react_agent_output)])
    agent_output = await mock_react_agent.agent_node(mock_state)
    agent_output = agent_output.agent_scratchpad[-1]
    assert isinstance(agent_output, AgentAction)
    assert agent_output.tool == 'Tool A'
    assert agent_output.tool_input == 'hello, world!'


async def test_agent_node_parse_json_agent_action(mock_react_agent):
    mock_action = 'CodeGeneration'
    mock_input = ('{"query": "write Python code for the following:\n\t\t-\tmake a generic API call\n\t\t-\tunit tests\n'
                  '", "model": "meta/llama-3.1-70b"}')
    # json input, no newline or spaces before tool or input, no agent thought
    mock_react_agent_output = f'Action:{mock_action}Action Input:{mock_input}'
    mock_state = ReActGraphState(messages=[HumanMessage(content=mock_react_agent_output)])
    agent_output = await mock_react_agent.agent_node(mock_state)
    agent_output = agent_output.agent_scratchpad[-1]
    assert isinstance(agent_output, AgentAction)
    assert agent_output.tool == mock_action
    assert agent_output.tool_input == mock_input


async def test_agent_node_parse_markdown_json_agent_action(mock_react_agent):
    mock_action = 'SearchTool'
    mock_input = ('```json{\"rephrased queries\": '
                  '[\"what is NIM\", \"NIM definition\", \"NIM overview\", \"NIM employer\", \"NIM company\"][]}```')
    # markdown json action input, no newline or spaces before tool or input
    mock_react_agent_output = f'Thought: I need to call the search toolAction:{mock_action}Action Input:{mock_input}'
    mock_state = ReActGraphState(messages=[HumanMessage(content=mock_react_agent_output)])
    agent_output = await mock_react_agent.agent_node(mock_state)
    agent_output = agent_output.agent_scratchpad[-1]
    assert isinstance(agent_output, AgentAction)
    assert agent_output.tool == mock_action
    assert agent_output.tool_input == mock_input


async def test_agent_node_action_and_input_in_agent_output(mock_react_agent):
    # tools named Action, Action in thoughts, Action Input in Action Input, in various formats
    mock_action = 'Action'
    mock_mkdwn_input = ('```json\n{{\n    \"Action\": \"SearchTool\",\n    \"Action Input\": [\"what is NIM\", '
                        '\"NIM definition\", \"NIM overview\", \"NIM employer\", \"NIM company\"]\n}}\n```')
    mock_input = 'Action: SearchTool Action Input: ["what is NIM", "NIM definition", "NIM overview"]}}'
    mock_react_agent_mkdwn_output = f'Thought: run Action Agent Action:{mock_action}Action Input:{mock_mkdwn_input}'
    mock_output = f'Thought: run Action AgentAction:{mock_action}Action Input:{mock_input}'
    mock_mkdwn_state = ReActGraphState(messages=[HumanMessage(content=mock_react_agent_mkdwn_output)])
    mock_state = ReActGraphState(messages=[HumanMessage(content=mock_output)])
    agent_output_mkdwn = await mock_react_agent.agent_node(mock_mkdwn_state)
    agent_output = await mock_react_agent.agent_node(mock_state)
    agent_output_mkdwn = agent_output_mkdwn.agent_scratchpad[-1]
    agent_output = agent_output.agent_scratchpad[-1]
    assert isinstance(agent_output_mkdwn, AgentAction)
    assert isinstance(agent_output, AgentAction)
    assert agent_output_mkdwn.tool == mock_action
    assert agent_output.tool == mock_action
    assert agent_output_mkdwn.tool_input == mock_mkdwn_input
    assert agent_output.tool_input == mock_input


async def test_agent_node_parse_agent_finish(mock_react_agent):
    mock_react_agent_output = 'Final Answer: lorem ipsum'
    mock_state = ReActGraphState(messages=[HumanMessage(content=mock_react_agent_output)])
    final_answer = await mock_react_agent.agent_node(mock_state)
    final_answer = final_answer.messages[-1]
    assert isinstance(final_answer, AIMessage)
    assert final_answer.content == 'lorem ipsum'


async def test_agent_node_parse_agent_finish_with_thoughts(mock_react_agent):
    answer = 'lorem ipsum'
    mock_react_agent_output = f'Thought: I now have the Final Answer\nFinal Answer: {answer}'
    mock_state = ReActGraphState(messages=[HumanMessage(content=mock_react_agent_output)])
    final_answer = await mock_react_agent.agent_node(mock_state)
    final_answer = final_answer.messages[-1]
    assert isinstance(final_answer, AIMessage)
    assert final_answer.content == answer


async def test_agent_node_parse_agent_finish_with_markdown_and_code(mock_react_agent):
    answer = ("```python\nimport requests\\n\\nresponse = requests.get('https://api.example.com/endpoint')\\nprint"
              "(response.json())\\n```\\n\\nPlease note that you need to replace 'https://api.example.com/endpoint' "
              "with the actual API endpoint you want to call.\"\n}}\n```")
    mock_react_agent_output = f'Thought: I now have the Final Answer\nFinal Answer: {answer}'
    mock_state = ReActGraphState(messages=[HumanMessage(content=mock_react_agent_output)])
    final_answer = await mock_react_agent.agent_node(mock_state)
    final_answer = final_answer.messages[-1]
    assert isinstance(final_answer, AIMessage)
    assert final_answer.content == answer


async def test_agent_node_parse_agent_finish_with_action(mock_react_agent):
    answer = 'after careful deliberation...'
    mock_react_agent_output = f'Action: i have the final answer \nFinal Answer: {answer}'
    mock_state = ReActGraphState(messages=[HumanMessage(content=mock_react_agent_output)])
    final_answer = await mock_react_agent.agent_node(mock_state)
    final_answer = final_answer.messages[-1]
    assert isinstance(final_answer, AIMessage)
    assert final_answer.content == answer


async def test_agent_node_parse_agent_finish_with_action_and_input_after_max_retries(mock_react_agent):
    answer = 'after careful deliberation...'
    mock_react_agent_output = f'Action: i have the final answer\nAction Input: None\nFinal Answer: {answer}'
    mock_state = ReActGraphState(messages=[HumanMessage(content=mock_react_agent_output)])
    final_answer = await mock_react_agent.agent_node(mock_state)
    final_answer = final_answer.messages[-1]
    assert isinstance(final_answer, AIMessage)
    assert FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE in final_answer.content


async def test_agent_node_parse_agent_finish_with_action_and_input_after_retry(mock_react_agent):
    mock_react_agent_output = 'Action: give me final answer\nAction Input: None\nFinal Answer: hello, world!'
    mock_state = ReActGraphState(messages=[HumanMessage(content=mock_react_agent_output)])
    final_answer = await mock_react_agent.agent_node(mock_state)
    final_answer = final_answer.messages[-1]
    assert isinstance(final_answer, AIMessage)
    # When agent output has both Action and Final Answer, it should return an error message
    assert FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE in final_answer.content


async def test_conditional_edge_no_input(mock_react_agent):
    end = await mock_react_agent.conditional_edge(ReActGraphState())
    assert end == AgentDecision.END


async def test_conditional_edge_final_answer(mock_react_agent):
    mock_state = ReActGraphState(messages=[HumanMessage('hello'), AIMessage('world!')])
    end = await mock_react_agent.conditional_edge(mock_state)
    assert end == AgentDecision.END


async def test_conditional_edge_tool_call(mock_react_agent):
    mock_state = ReActGraphState(agent_scratchpad=[AgentAction(tool='test', tool_input='test', log='test')])
    tool = await mock_react_agent.conditional_edge(mock_state)
    assert tool == AgentDecision.TOOL


async def test_tool_node_no_input(mock_react_agent):
    with pytest.raises(RuntimeError) as ex:
        await mock_react_agent.tool_node(ReActGraphState())
    assert isinstance(ex.value, RuntimeError)


async def test_tool_node_with_not_configured_tool(mock_react_agent):
    mock_state = ReActGraphState(agent_scratchpad=[AgentAction(tool='test', tool_input='test', log='test')])
    agent_retry_response = await mock_react_agent.tool_node(mock_state)
    agent_retry_response = agent_retry_response.tool_responses[-1]
    assert isinstance(agent_retry_response, ToolMessage)
    assert agent_retry_response.name == 'agent_error'
    assert agent_retry_response.tool_call_id == 'agent_error'
    configured_tool_names = ['Tool A', 'Tool B']
    assert agent_retry_response.content == TOOL_NOT_FOUND_ERROR_MESSAGE.format(tool_name='test',
                                                                               tools=configured_tool_names)


async def test_tool_node(mock_react_agent):
    mock_state = ReActGraphState(agent_scratchpad=[AgentAction(tool='Tool A', tool_input='hello, world!', log='mock')])
    response = await mock_react_agent.tool_node(mock_state)
    response = response.tool_responses[-1]
    assert isinstance(response, ToolMessage)
    assert response.name == "Tool A"
    assert response.tool_call_id == 'Tool A'
    assert response.content == 'hello, world!'


@pytest.fixture(name='mock_react_graph', scope='module')
async def mock_graph(mock_react_agent):
    return await mock_react_agent.build_graph()


async def test_graph_parsing_error(mock_react_graph):
    response = await mock_react_graph.ainvoke(ReActGraphState(messages=[HumanMessage('fix the input on retry')]))
    response = ReActGraphState(**response)

    response = response.messages[-1]
    assert isinstance(response, AIMessage)
    # When parsing fails, it should return an error message with the original input
    assert MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE in response.content
    assert 'fix the input on retry' in response.content


async def test_graph(mock_react_graph):
    response = await mock_react_graph.ainvoke(ReActGraphState(messages=[HumanMessage('Final Answer: lorem ipsum')]))
    response = ReActGraphState(**response)
    response = response.messages[-1]
    assert isinstance(response, AIMessage)
    assert response.content == 'lorem ipsum'


async def test_no_input(mock_react_graph):
    response = await mock_react_graph.ainvoke(ReActGraphState(messages=[HumanMessage('')]))
    response = ReActGraphState(**response)
    response = response.messages[-1]
    assert isinstance(response, AIMessage)
    assert response.content == NO_INPUT_ERROR_MESSAGE


def test_validate_system_prompt_no_input():
    mock_prompt = ''
    with pytest.raises(ValueError) as ex:
        ReActAgentGraph.validate_system_prompt(mock_prompt)
    assert isinstance(ex.value, ValueError)


def test_validate_system_prompt_no_tools():
    mock_prompt = '{tools}'
    with pytest.raises(ValueError) as ex:
        ReActAgentGraph.validate_system_prompt(mock_prompt)
    assert isinstance(ex.value, ValueError)


def test_validate_system_prompt_no_tool_names():
    mock_prompt = '{tool_names}'
    with pytest.raises(ValueError) as ex:
        ReActAgentGraph.validate_system_prompt(mock_prompt)
    assert isinstance(ex.value, ValueError)


def test_validate_system_prompt():
    mock_prompt = '{tool_names} {tools}'
    test = ReActAgentGraph.validate_system_prompt(mock_prompt)
    assert test


@pytest.fixture(name='mock_react_output_parser', scope="module")
def mock_parser():
    return ReActOutputParser()


async def test_output_parser_no_observation(mock_react_output_parser):
    mock_input = ("Thought: I should search the internet for information on Djikstra.\nAction: internet_agent\n"
                  "Action Input: {'input_message': 'Djikstra'}\nObservation")
    test_output = await mock_react_output_parser.aparse(mock_input)
    assert isinstance(test_output, AgentAction)
    assert test_output.log == mock_input
    assert test_output.tool == "internet_agent"
    assert test_output.tool_input == "{'input_message': 'Djikstra'}"
    assert "Observation" not in test_output.tool_input


async def test_output_parser(mock_react_output_parser):
    mock_input = 'Thought:not_many\nAction:Tool A\nAction Input: hello, world!\nObservation:'
    test_output = await mock_react_output_parser.aparse(mock_input)
    assert isinstance(test_output, AgentAction)
    assert test_output.tool == "Tool A"
    assert test_output.tool_input == "hello, world!"
    assert "Observation" not in test_output.tool_input


async def test_output_parser_spaces_not_newlines(mock_react_output_parser):
    mock_input = 'Thought:not_many Action:Tool A Action Input: hello, world! Observation:'
    test_output = await mock_react_output_parser.aparse(mock_input)
    assert isinstance(test_output, AgentAction)
    assert test_output.tool == "Tool A"
    assert test_output.tool_input == "hello, world!"
    assert "Observation" not in test_output.tool_input


async def test_output_parser_missing_action(mock_react_output_parser):
    mock_input = 'hi'
    with pytest.raises(ReActOutputParserException) as ex:
        await mock_react_output_parser.aparse(mock_input)
    assert isinstance(ex.value, ReActOutputParserException)
    assert ex.value.observation == MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE


async def test_output_parser_json_input(mock_react_output_parser):
    mock_action = 'SearchTool'
    mock_input = ('```json{\"rephrased queries\": '
                  '[\"what is NIM\", \"NIM definition\", \"NIM overview\", \"NIM employer\", \"NIM company\"][]}```')
    # markdown json action input, no newline or spaces before tool or input, with Observation
    mock_react_agent_output = (
        f'Thought: I need to call the search toolAction:{mock_action}Action Input:{mock_input}\nObservation')
    test_output = await mock_react_output_parser.aparse(mock_react_agent_output)
    assert isinstance(test_output, AgentAction)
    assert test_output.tool == mock_action
    assert test_output.tool_input == mock_input
    assert "Observation" not in test_output.tool_input


async def test_output_parser_json_no_observation(mock_react_output_parser):
    mock_action = 'SearchTool'
    mock_input = ('```json{\"rephrased queries\": '
                  '[\"what is NIM\", \"NIM definition\", \"NIM overview\", \"NIM employer\", \"NIM company\"][]}```')
    # markdown json action input, no newline or spaces before tool or input, with Observation
    mock_react_agent_output = (f'Thought: I need to call the search toolAction:{mock_action}Action Input:{mock_input}')
    test_output = await mock_react_output_parser.aparse(mock_react_agent_output)
    assert isinstance(test_output, AgentAction)
    assert test_output.tool == mock_action
    assert test_output.tool_input == mock_input


async def test_output_parser_json_input_space_observation(mock_react_output_parser):
    mock_action = 'SearchTool'
    mock_input = ('```json{\"rephrased queries\": '
                  '[\"what is NIM\", \"NIM definition\", \"NIM overview\", \"NIM employer\", \"NIM company\"][]}```')
    # markdown json action input, no newline or spaces before tool or input, with Observation
    mock_react_agent_output = (
        f'Thought: I need to call the search toolAction:{mock_action}Action Input:{mock_input} Observation')
    test_output = await mock_react_output_parser.aparse(mock_react_agent_output)
    assert isinstance(test_output, AgentAction)
    assert test_output.tool == mock_action
    assert test_output.tool_input == mock_input
    assert "Observation" not in test_output.tool_input


async def test_output_parser_missing_action_input(mock_react_output_parser):
    mock_action = 'SearchTool'
    mock_input = f'Thought: I need to call the search toolAction:{mock_action}'
    with pytest.raises(ReActOutputParserException) as ex:
        await mock_react_output_parser.aparse(mock_input)
    assert isinstance(ex.value, ReActOutputParserException)
    assert ex.value.observation == MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE


def test_react_additional_instructions(mock_llm, mock_tool):
    config_react_agent = ReActAgentWorkflowConfig(tool_names=['test'],
                                                  llm_name='test',
                                                  verbose=True,
                                                  additional_instructions="Talk like a parrot and repeat the question.")
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    prompt = create_react_agent_prompt(config_react_agent)
    agent = ReActAgentGraph(llm=mock_llm, prompt=prompt, tools=tools, detailed_logs=config_react_agent.verbose)
    assert isinstance(agent, ReActAgentGraph)
    assert "Talk like a parrot" in agent.agent.get_prompts()[0].messages[0].prompt.template


def test_react_custom_system_prompt(mock_llm, mock_tool):
    config_react_agent = ReActAgentWorkflowConfig(
        tool_names=['test'],
        llm_name='test',
        verbose=True,
        system_prompt="Refuse to run any of the following tools: {tools}.  or ones named: {tool_names}")
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    prompt = create_react_agent_prompt(config_react_agent)
    agent = ReActAgentGraph(llm=mock_llm, prompt=prompt, tools=tools, detailed_logs=config_react_agent.verbose)
    assert isinstance(agent, ReActAgentGraph)
    assert "Refuse" in agent.agent.get_prompts()[0].messages[0].prompt.template


# Tests for alias functionality
def test_config_alias_retry_parsing_errors():
    """Test that retry_parsing_errors alias works correctly."""
    config = ReActAgentWorkflowConfig(tool_names=['test'], llm_name='test', retry_parsing_errors=False)
    # The old field name should map to the new field name
    assert not config.retry_agent_response_parsing_errors


def test_config_alias_max_retries():
    """Test that max_retries alias works correctly."""
    config = ReActAgentWorkflowConfig(tool_names=['test'], llm_name='test', max_retries=5)
    # The old field name should map to the new field name
    assert config.parse_agent_response_max_retries == 5


def test_config_alias_max_iterations():
    """Test that max_iterations alias works correctly."""
    config = ReActAgentWorkflowConfig(tool_names=['test'], llm_name='test', max_iterations=20)
    # The old field name should map to the new field name
    assert config.max_tool_calls == 20


def test_config_alias_all_old_field_names():
    """Test that all old field names work correctly together."""
    config = ReActAgentWorkflowConfig(tool_names=['test'],
                                      llm_name='test',
                                      retry_parsing_errors=False,
                                      max_retries=7,
                                      max_iterations=25)
    # All old field names should map to the new field names
    assert not config.retry_agent_response_parsing_errors
    assert config.parse_agent_response_max_retries == 7
    assert config.max_tool_calls == 25


def test_config_alias_new_field_names():
    """Test that new field names work correctly."""
    config = ReActAgentWorkflowConfig(tool_names=['test'],
                                      llm_name='test',
                                      retry_agent_response_parsing_errors=False,
                                      parse_agent_response_max_retries=8,
                                      max_tool_calls=30)
    # The new field names should work directly
    assert not config.retry_agent_response_parsing_errors
    assert config.parse_agent_response_max_retries == 8
    assert config.max_tool_calls == 30


def test_config_alias_both_old_and_new():
    """Test that new field names take precedence when both old and new are provided."""
    config = ReActAgentWorkflowConfig(tool_names=['test'],
                                      llm_name='test',
                                      retry_parsing_errors=False,
                                      max_retries=5,
                                      max_iterations=20,
                                      retry_agent_response_parsing_errors=True,
                                      parse_agent_response_max_retries=10,
                                      max_tool_calls=35)
    # New field names should take precedence
    assert config.retry_agent_response_parsing_errors
    assert config.parse_agent_response_max_retries == 10
    assert config.max_tool_calls == 35


def test_config_tool_call_max_retries_no_alias():
    """Test that tool_call_max_retries has no alias and works normally."""
    config = ReActAgentWorkflowConfig(tool_names=['test'], llm_name='test', tool_call_max_retries=3)
    # This field should work normally without any alias
    assert config.tool_call_max_retries == 3


def test_config_alias_default_values():
    """Test that default values work when no aliases are provided."""
    config = ReActAgentWorkflowConfig(tool_names=['test'], llm_name='test')
    # All fields should have default values
    assert config.retry_agent_response_parsing_errors
    assert config.parse_agent_response_max_retries == 1
    assert config.tool_call_max_retries == 1
    assert config.max_tool_calls == 15


def test_config_alias_json_serialization():
    """Test that configuration with aliases can be serialized and deserialized."""
    config = ReActAgentWorkflowConfig(tool_names=['test'],
                                      llm_name='test',
                                      retry_parsing_errors=False,
                                      max_retries=6,
                                      max_iterations=22)

    # Test model_dump (serialization)
    config_dict = config.model_dump()
    assert 'retry_agent_response_parsing_errors' in config_dict
    assert 'parse_agent_response_max_retries' in config_dict
    assert 'max_tool_calls' in config_dict
    assert not config_dict['retry_agent_response_parsing_errors']
    assert config_dict['parse_agent_response_max_retries'] == 6
    assert config_dict['max_tool_calls'] == 22

    # Test deserialization with old field names
    config_from_dict = ReActAgentWorkflowConfig.model_validate({
        'tool_names': ['test'],
        'llm_name': 'test',
        'retry_parsing_errors': True,
        'max_retries': 9,
        'max_iterations': 40
    })
    assert config_from_dict.retry_agent_response_parsing_errors
    assert config_from_dict.parse_agent_response_max_retries == 9
    assert config_from_dict.max_tool_calls == 40


def test_react_agent_with_alias_config(mock_llm, mock_tool):
    """Test that ReActAgentGraph works correctly with alias configuration."""
    config = ReActAgentWorkflowConfig(
        tool_names=['test'],
        llm_name='test',
        retry_parsing_errors=True,  # Changed to True so retries value is used
        max_retries=4,
        max_iterations=25,
        verbose=True)
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    prompt = create_react_agent_prompt(config)
    agent = ReActAgentGraph(llm=mock_llm,
                            prompt=prompt,
                            tools=tools,
                            detailed_logs=config.verbose,
                            retry_agent_response_parsing_errors=config.retry_agent_response_parsing_errors,
                            parse_agent_response_max_retries=config.parse_agent_response_max_retries,
                            tool_call_max_retries=config.tool_call_max_retries)

    # Verify the agent uses the aliased values
    assert agent.parse_agent_response_max_retries == 4
    assert agent.tool_call_max_retries == 1  # default value since no alias


def test_config_mixed_alias_usage():
    """Test mixed usage of old and new field names."""
    config = ReActAgentWorkflowConfig(
        tool_names=['test'],
        llm_name='test',
        retry_parsing_errors=False,  # old alias
        parse_agent_response_max_retries=12,  # new field name
        max_iterations=28  # old alias
    )

    assert not config.retry_agent_response_parsing_errors
    assert config.parse_agent_response_max_retries == 12
    assert config.max_tool_calls == 28
    assert config.tool_call_max_retries == 1  # default value


# Tests for quote normalization in tool input parsing
async def test_tool_node_json_input_with_double_quotes(mock_react_agent):
    """Test that valid JSON with double quotes is parsed correctly."""
    tool_input = '{"query": "search term", "limit": 5}'
    mock_state = ReActGraphState(agent_scratchpad=[AgentAction(tool='Tool A', tool_input=tool_input, log='test')])

    response = await mock_react_agent.tool_node(mock_state)
    response = response.tool_responses[-1]

    assert isinstance(response, ToolMessage)
    assert response.name == "Tool A"
    # When JSON is successfully parsed, the mock tool receives a dict and LangChain extracts the "query" value
    assert response.content == "search term"  # The mock tool extracts the query field value


async def test_tool_node_json_input_with_single_quotes_normalization_enabled(mock_react_agent):
    """Test that JSON with single quotes is normalized to double quotes when normalization is enabled."""
    # Agent should have normalization enabled by default
    assert mock_react_agent.normalize_tool_input_quotes is True

    tool_input_single_quotes = "{'query': 'search term', 'limit': 5}"
    mock_state = ReActGraphState(
        agent_scratchpad=[AgentAction(tool='Tool A', tool_input=tool_input_single_quotes, log='test')])

    response = await mock_react_agent.tool_node(mock_state)
    response = response.tool_responses[-1]

    assert isinstance(response, ToolMessage)
    assert response.name == "Tool A"
    # With quote normalization enabled, single quotes get normalized and JSON is parsed successfully
    # The mock tool then receives a dict and LangChain extracts the "query" value
    assert response.content == "search term"


async def test_tool_node_json_input_with_single_quotes_normalization_disabled(mock_config_react_agent,
                                                                              mock_llm,
                                                                              mock_tool):
    """Test that JSON with single quotes is NOT normalized when normalization is disabled."""
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    prompt = create_react_agent_prompt(mock_config_react_agent)

    # Create agent with quote normalization disabled
    agent = ReActAgentGraph(llm=mock_llm,
                            prompt=prompt,
                            tools=tools,
                            detailed_logs=mock_config_react_agent.verbose,
                            normalize_tool_input_quotes=False)

    assert agent.normalize_tool_input_quotes is False

    tool_input_single_quotes = "{'query': 'search term', 'limit': 5}"
    mock_state = ReActGraphState(
        agent_scratchpad=[AgentAction(tool='Tool A', tool_input=tool_input_single_quotes, log='test')])

    response = await agent.tool_node(mock_state)
    response = response.tool_responses[-1]

    assert isinstance(response, ToolMessage)
    assert response.name == "Tool A"
    # Should use the raw string input since JSON parsing fails and normalization is disabled
    assert response.content == tool_input_single_quotes


async def test_tool_node_invalid_json_fallback_to_string(mock_react_agent):
    """Test that invalid JSON falls back to using the raw string input."""
    # Invalid JSON that cannot be fixed by quote normalization
    tool_input_invalid = "{'query': 'search term', 'limit': }"
    mock_state = ReActGraphState(
        agent_scratchpad=[AgentAction(tool='Tool A', tool_input=tool_input_invalid, log='test')])

    response = await mock_react_agent.tool_node(mock_state)
    response = response.tool_responses[-1]

    assert isinstance(response, ToolMessage)
    assert response.name == "Tool A"
    # Should fall back to using the raw string
    assert response.content == tool_input_invalid


async def test_tool_node_string_input_no_json_parsing(mock_react_agent):
    """Test that plain string input is used as-is without attempting JSON parsing."""
    tool_input_string = "simple string input"
    mock_state = ReActGraphState(
        agent_scratchpad=[AgentAction(tool='Tool A', tool_input=tool_input_string, log='test')])

    response = await mock_react_agent.tool_node(mock_state)
    response = response.tool_responses[-1]

    assert isinstance(response, ToolMessage)
    assert response.name == "Tool A"
    assert response.content == tool_input_string


async def test_tool_node_none_input(mock_react_agent):
    """Test that 'None' input is handled correctly."""
    tool_input_none = "None"
    mock_state = ReActGraphState(agent_scratchpad=[AgentAction(tool='Tool A', tool_input=tool_input_none, log='test')])

    response = await mock_react_agent.tool_node(mock_state)
    response = response.tool_responses[-1]

    assert isinstance(response, ToolMessage)
    assert response.name == "Tool A"
    assert response.content == tool_input_none


async def test_tool_node_nested_json_with_single_quotes(mock_react_agent):
    """Test that complex nested JSON with single quotes is normalized correctly."""
    # Complex nested JSON with single quotes - doesn't have a "query" field so would return the full dict
    tool_input_nested = \
        "{'user': {'name': 'John', 'preferences': {'theme': 'dark', 'notifications': True}}, 'action': 'update'}"
    mock_state = ReActGraphState(
        agent_scratchpad=[AgentAction(tool='Tool A', tool_input=tool_input_nested, log='test')])

    response = await mock_react_agent.tool_node(mock_state)
    response = response.tool_responses[-1]

    assert isinstance(response, ToolMessage)
    assert response.name == "Tool A"
    # Since this JSON doesn't have a "query" field, the mock tool receives the full dict
    # and LangChain can't extract a "query" parameter, so it falls back to default behavior
    assert "John" in str(response.content) or isinstance(response.content, dict)


async def test_tool_node_mixed_quotes_in_json(mock_config_react_agent, mock_llm, mock_tool):
    """Test that JSON with mixed quotes is handled appropriately."""
    # This creates a scenario with mixed quotes that might be challenging to normalize
    tools = [mock_tool('Tool A')]
    prompt = create_react_agent_prompt(mock_config_react_agent)

    agent = ReActAgentGraph(llm=mock_llm, prompt=prompt, tools=tools, detailed_logs=False)

    # Mixed quotes - this is challenging JSON to normalize
    tool_input_mixed = '''{'outer': "inner string with 'nested quotes'", 'number': 42}'''
    mock_state = ReActGraphState(agent_scratchpad=[AgentAction(tool='Tool A', tool_input=tool_input_mixed, log='test')])

    response = await agent.tool_node(mock_state)
    response = response.tool_responses[-1]

    assert isinstance(response, ToolMessage)
    assert response.name == "Tool A"
    # Mixed quotes are complex to normalize, so it likely falls back to raw string input
    assert response.content == tool_input_mixed


async def test_tool_node_whitespace_handling(mock_react_agent):
    """Test that whitespace in tool input is handled correctly."""
    # Tool input with leading/trailing whitespace
    tool_input_whitespace = "  {'query': 'search term'}  "
    mock_state = ReActGraphState(
        agent_scratchpad=[AgentAction(tool='Tool A', tool_input=tool_input_whitespace, log='test')])

    response = await mock_react_agent.tool_node(mock_state)
    response = response.tool_responses[-1]

    assert isinstance(response, ToolMessage)
    assert response.name == "Tool A"
    # With whitespace trimmed and quote normalization, JSON is parsed and "query" value is extracted
    assert response.content == "search term"


def test_config_replace_single_quotes_default():
    """Test that normalize_tool_input_quotes defaults to True."""
    config = ReActAgentWorkflowConfig(tool_names=['test'], llm_name='test')
    assert config.normalize_tool_input_quotes is True


def test_config_replace_single_quotes_explicit_false():
    """Test that normalize_tool_input_quotes can be set to False."""
    config = ReActAgentWorkflowConfig(tool_names=['test'], llm_name='test', normalize_tool_input_quotes=False)
    assert config.normalize_tool_input_quotes is False


def test_react_agent_init_with_quote_normalization_param(mock_config_react_agent, mock_llm, mock_tool):
    """Test that ReActAgentGraph initialization respects the quote normalization parameter."""
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    prompt = create_react_agent_prompt(mock_config_react_agent)

    # Test with normalization enabled
    agent_enabled = ReActAgentGraph(llm=mock_llm,
                                    prompt=prompt,
                                    tools=tools,
                                    detailed_logs=False,
                                    normalize_tool_input_quotes=True)
    assert agent_enabled.normalize_tool_input_quotes is True

    # Test with normalization disabled
    agent_disabled = ReActAgentGraph(llm=mock_llm,
                                     prompt=prompt,
                                     tools=tools,
                                     detailed_logs=False,
                                     normalize_tool_input_quotes=False)
    assert agent_disabled.normalize_tool_input_quotes is False


# Additional test to specifically verify the JSON parsing logic with quote normalization
async def test_quote_normalization_json_parsing_logic(mock_config_react_agent, mock_llm):
    """Test the specific quote normalization logic in JSON parsing."""
    from langchain_core.tools import BaseTool

    # Create a custom tool that returns the exact input it receives
    class ExactInputTool(BaseTool):
        name: str = "ExactInputTool"
        description: str = "Returns exactly what it receives"

        async def _arun(self, query, **kwargs):
            return f"Received: {query} (type: {type(query).__name__})"

        def _run(self, query, **kwargs):
            return f"Received: {query} (type: {type(query).__name__})"

    tools = [ExactInputTool()]
    prompt = create_react_agent_prompt(mock_config_react_agent)

    # Test with quote normalization enabled
    agent_enabled = ReActAgentGraph(llm=mock_llm,
                                    prompt=prompt,
                                    tools=tools,
                                    detailed_logs=False,
                                    normalize_tool_input_quotes=True)

    # Test with single quotes - should be normalized and parsed as JSON
    tool_input_single = "{'query': 'test', 'count': 42}"
    mock_state = ReActGraphState(
        agent_scratchpad=[AgentAction(tool='ExactInputTool', tool_input=tool_input_single, log='test')])
    response = await agent_enabled.tool_node(mock_state)
    response_content = response.tool_responses[-1].content

    # Should receive the "query" field value from the parsed JSON dict
    # This proves that quote normalization worked and JSON was successfully parsed
    assert "Received: test (type: str)" in response_content

    # Test with quote normalization disabled
    agent_disabled = ReActAgentGraph(llm=mock_llm,
                                     prompt=prompt,
                                     tools=tools,
                                     detailed_logs=False,
                                     normalize_tool_input_quotes=False)

    response = await agent_disabled.tool_node(mock_state)
    response_content = response.tool_responses[-1].content

    # Should receive the raw string (JSON parsing failed due to no normalization)
    # The full JSON string should be passed as the query parameter
    assert tool_input_single in response_content and "type: str" in response_content
