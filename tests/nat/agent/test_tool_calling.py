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
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode

from nat.agent.base import AgentDecision
from nat.agent.tool_calling_agent.agent import ToolCallAgentGraph
from nat.agent.tool_calling_agent.agent import ToolCallAgentGraphState
from nat.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig


async def test_state_schema():
    input_message = HumanMessage(content='test')
    state = ToolCallAgentGraphState(messages=[input_message])
    assert isinstance(state.messages, list)

    # pylint: disable=unsubscriptable-object
    assert isinstance(state.messages[0], HumanMessage)
    assert state.messages[0].content == input_message.content
    with pytest.raises(AttributeError) as ex:
        await state.agent_scratchpad
    assert isinstance(ex.value, AttributeError)


@pytest.fixture(name='mock_config_tool_calling_agent', scope="module")
def mock_config():
    return ToolCallAgentWorkflowConfig(tool_names=['test'], llm_name='test', verbose=True)


def test_tool_calling_agent_init(mock_config_tool_calling_agent, mock_llm, mock_tool):
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    agent = ToolCallAgentGraph(llm=mock_llm, tools=tools, detailed_logs=mock_config_tool_calling_agent.verbose)
    assert isinstance(agent, ToolCallAgentGraph)
    assert agent.llm == mock_llm
    assert agent.tools == tools
    assert agent.detailed_logs == mock_config_tool_calling_agent.verbose
    assert isinstance(agent.tool_caller, ToolNode)
    assert list(agent.tool_caller.tools_by_name.keys()) == ['Tool A', 'Tool B']


@pytest.fixture(name='mock_tool_agent', scope="module")
def mock_agent(mock_config_tool_calling_agent, mock_tool, mock_llm):
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    agent = ToolCallAgentGraph(llm=mock_llm, tools=tools, detailed_logs=mock_config_tool_calling_agent.verbose)
    return agent


async def test_build_graph(mock_tool_agent):
    graph = await mock_tool_agent.build_graph()
    assert isinstance(graph, CompiledGraph)
    assert list(graph.nodes.keys()) == ['__start__', 'agent', 'tool']
    assert graph.builder.edges == {('__start__', 'agent'), ('tool', 'agent')}
    assert set(graph.builder.branches.get('agent').get('conditional_edge').ends.keys()) == {
        AgentDecision.TOOL, AgentDecision.END
    }


async def test_agent_node_no_input(mock_tool_agent):
    with pytest.raises(RuntimeError) as ex:
        await mock_tool_agent.agent_node(ToolCallAgentGraphState())
    assert isinstance(ex.value, RuntimeError)


async def test_agent_node(mock_tool_agent):
    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='please, mock tool call!')])
    response = await mock_tool_agent.agent_node(mock_state)
    response = response.messages[-1]
    assert isinstance(response, AIMessage)
    assert response.content == 'mock tool call'


async def test_conditional_edge_no_input(mock_tool_agent):
    end = await mock_tool_agent.conditional_edge(ToolCallAgentGraphState())
    assert end == AgentDecision.END


async def test_conditional_edge_final_answer(mock_tool_agent):
    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='hello, world!')])
    end = await mock_tool_agent.conditional_edge(mock_state)
    assert end == AgentDecision.END


async def test_conditional_edge_tool_call(mock_tool_agent):
    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='', tool_calls={'mock': True})])
    tool = await mock_tool_agent.conditional_edge(mock_state)
    assert tool == AgentDecision.TOOL


async def test_tool_node_no_input(mock_tool_agent):
    with pytest.raises(IndexError) as ex:
        await mock_tool_agent.tool_node(ToolCallAgentGraphState())
    assert isinstance(ex.value, IndexError)


async def test_tool_node_final_answer(mock_tool_agent):
    message = AIMessage(content='mock tool call',
                        response_metadata={"mock_llm_response": True},
                        tool_calls=[{
                            "name": "Tool A",
                            "args": {
                                "query": "mock query"
                            },
                            "id": "Tool A",
                            "type": "tool_call",
                        }])
    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='hello, world!')])
    mock_state.messages.append(message)  # pylint: disable=no-member
    response = await mock_tool_agent.tool_node(mock_state)
    response = response.messages[-1]
    assert isinstance(response, ToolMessage)
    assert response.content == 'mock query'
    assert response.name == 'Tool A'


@pytest.fixture(name="mock_tool_graph", scope="module")
async def mock_graph(mock_tool_agent):
    return await mock_tool_agent.build_graph()


async def test_graph(mock_tool_graph):
    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='please, mock tool call!')])
    response = await mock_tool_graph.ainvoke(mock_state)
    response = ToolCallAgentGraphState(**response)
    response = response.messages[-1]  # pylint: disable=unsubscriptable-object
    assert isinstance(response, AIMessage)
    assert response.content == 'mock query'
