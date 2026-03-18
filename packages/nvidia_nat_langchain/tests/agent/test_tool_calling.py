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

import datetime
import json
import typing
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ChatResponseChunkChoice
from nat.data_models.api_server import ChoiceDelta
from nat.data_models.api_server import ChoiceDeltaToolCall
from nat.data_models.api_server import ChoiceDeltaToolCallFunction
from nat.plugins.langchain.agent.base import AgentDecision
from nat.plugins.langchain.agent.tool_calling_agent.agent import ToolCallAgentGraph
from nat.plugins.langchain.agent.tool_calling_agent.agent import ToolCallAgentGraphState
from nat.plugins.langchain.agent.tool_calling_agent.agent import create_tool_calling_agent_prompt
from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig
from nat.plugins.langchain.agent.tool_calling_agent.register import TruncationRetryConfig


def test_truncation_retry_config_rejects_both_strategies():
    """Setting both token_increment and token_scaling must raise ValidationError."""
    with pytest.raises(Exception, match="token_increment or token_scaling"):
        TruncationRetryConfig(max_retries=2, token_increment=1024, token_scaling=1.5)


def test_truncation_retry_config_defaults_to_increment_when_neither_set():
    """max_retries > 0 with neither set defaults token_increment to 1024."""
    config = TruncationRetryConfig(max_retries=2, token_increment=None, token_scaling=None)
    assert config.token_increment == 1024
    assert config.token_scaling is None


def test_truncation_retry_config_accepts_scaling_only():
    """Setting only token_scaling (without token_increment) must succeed."""
    config = TruncationRetryConfig(max_retries=3, token_scaling=1.5)
    assert config.token_scaling == 1.5
    assert config.token_increment is None


def test_truncation_retry_config_accepts_increment_only():
    """Setting only token_increment (without token_scaling) must succeed."""
    config = TruncationRetryConfig(max_retries=3, token_increment=512)
    assert config.token_increment == 512
    assert config.token_scaling is None


async def test_state_schema():
    input_message = HumanMessage(content='test')
    state = ToolCallAgentGraphState(messages=[input_message])
    assert isinstance(state.messages, list)

    assert isinstance(state.messages[0], HumanMessage)
    assert state.messages[0].content == input_message.content
    with pytest.raises(AttributeError) as ex:
        await state.agent_scratchpad
    assert isinstance(ex.value, AttributeError)


@pytest.fixture(name='mock_config_tool_calling_agent', scope="module")
def mock_config():
    return ToolCallAgentWorkflowConfig(tool_names=['test'], llm_name='test', verbose=True)


def test_tool_calling_config_prompt(mock_config_tool_calling_agent):
    config = mock_config_tool_calling_agent
    prompt = create_tool_calling_agent_prompt(config)
    assert prompt is None


def test_tool_calling_config_prompt_w_system_prompt():
    system_prompt = "test prompt"
    config = ToolCallAgentWorkflowConfig(tool_names=['test'],
                                         llm_name='test',
                                         verbose=True,
                                         system_prompt=system_prompt)
    prompt = create_tool_calling_agent_prompt(config)
    assert prompt is system_prompt


def test_tool_calling_config_prompt_w_additional_instructions():
    additional_instructions = "test additional instructions"
    config = ToolCallAgentWorkflowConfig(tool_names=['test'],
                                         llm_name='test',
                                         verbose=True,
                                         additional_instructions=additional_instructions)
    prompt = create_tool_calling_agent_prompt(config)
    assert prompt.strip() == additional_instructions.strip()


def test_tool_calling_agent_init(mock_config_tool_calling_agent, mock_llm, mock_tool):
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    agent = ToolCallAgentGraph(llm=mock_llm, tools=tools, detailed_logs=mock_config_tool_calling_agent.verbose)
    assert isinstance(agent, ToolCallAgentGraph)
    assert agent.llm == mock_llm
    assert agent.tools == tools
    assert agent.detailed_logs == mock_config_tool_calling_agent.verbose
    assert isinstance(agent.tool_caller, ToolNode)
    assert list(agent.tool_caller.tools_by_name.keys()) == ['Tool A', 'Tool B']


def test_tool_calling_agent_init_w_prompt(mock_config_tool_calling_agent, mock_llm, mock_tool):
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    prompt = "If a tool is available to help answer the question, use it to answer the question."
    agent = ToolCallAgentGraph(llm=mock_llm,
                               tools=tools,
                               detailed_logs=mock_config_tool_calling_agent.verbose,
                               prompt=prompt)
    assert isinstance(agent, ToolCallAgentGraph)
    assert agent.llm == mock_llm
    assert agent.tools == tools
    assert agent.detailed_logs == mock_config_tool_calling_agent.verbose
    assert isinstance(agent.tool_caller, ToolNode)
    assert list(agent.tool_caller.tools_by_name.keys()) == ['Tool A', 'Tool B']
    output_messages = agent.agent.steps[0].invoke({"messages": []})
    assert output_messages[0].content == prompt


async def test_tool_calling_agent_with_conversation_history(mock_config_tool_calling_agent, mock_llm, mock_tool):
    """
    Test that the tool calling agent with a conversation history will keep the conversation history.
    """
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    prompt = "If a tool is available to help answer the question, use it to answer the question."
    agent = ToolCallAgentGraph(llm=mock_llm,
                               tools=tools,
                               detailed_logs=mock_config_tool_calling_agent.verbose,
                               prompt=prompt)
    assert isinstance(agent, ToolCallAgentGraph)
    assert agent.llm == mock_llm
    assert agent.tools == tools
    assert agent.detailed_logs == mock_config_tool_calling_agent.verbose
    assert isinstance(agent.tool_caller, ToolNode)
    assert list(agent.tool_caller.tools_by_name.keys()) == ['Tool A', 'Tool B']
    messages = [
        HumanMessage(content='please, mock tool call!'),
        AIMessage(content='mock tool call'),
        HumanMessage(content='please, mock a different tool call!')
    ]
    state = ToolCallAgentGraphState(messages=messages)
    graph = await agent.build_graph()
    state = await graph.ainvoke(state, config={'recursion_limit': 5})
    state = ToolCallAgentGraphState(**state)
    # history preserved in order
    assert [type(m) for m in state.messages[:3]] == [type(m) for m in messages]
    assert [m.content for m in state.messages[:3]] == [m.content for m in messages]
    # exactly one new AI message appended for this scenario
    assert len(state.messages) == len(messages) + 1
    assert isinstance(state.messages[-1], AIMessage)


def test_tool_calling_agent_init_w_return_direct(mock_config_tool_calling_agent, mock_llm, mock_tool):
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    return_direct_tools = [tools[0]]
    agent = ToolCallAgentGraph(llm=mock_llm,
                               tools=tools,
                               detailed_logs=mock_config_tool_calling_agent.verbose,
                               return_direct=return_direct_tools)
    assert isinstance(agent, ToolCallAgentGraph)
    assert agent.llm == mock_llm
    assert agent.tools == tools
    assert agent.detailed_logs == mock_config_tool_calling_agent.verbose
    assert isinstance(agent.tool_caller, ToolNode)
    assert list(agent.tool_caller.tools_by_name.keys()) == ['Tool A', 'Tool B']
    assert agent.return_direct == ['Tool A']


@pytest.fixture(name='mock_tool_agent', scope="module")
def mock_agent(mock_config_tool_calling_agent, mock_tool, mock_llm):
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    agent = ToolCallAgentGraph(llm=mock_llm, tools=tools, detailed_logs=mock_config_tool_calling_agent.verbose)
    return agent


@pytest.fixture(name='mock_tool_agent_with_return_direct', scope="module")
def mock_agent_with_return_direct(mock_config_tool_calling_agent, mock_tool, mock_llm):
    tools = [mock_tool('Tool A'), mock_tool('Tool B')]
    agent = ToolCallAgentGraph(llm=mock_llm,
                               tools=tools,
                               detailed_logs=mock_config_tool_calling_agent.verbose,
                               return_direct=[tools[0]])
    return agent


async def test_build_graph(mock_tool_agent):
    graph = await mock_tool_agent.build_graph()
    assert isinstance(graph, CompiledStateGraph)
    assert list(graph.nodes.keys()) == ['__start__', 'agent', 'tool']
    assert graph.builder.edges == {('__start__', 'agent'), ('tool', 'agent')}
    assert set(graph.builder.branches.get('agent').get('conditional_edge').ends.keys()) == {
        AgentDecision.TOOL, AgentDecision.END
    }


async def test_build_graph_with_return_direct(mock_tool_agent_with_return_direct):
    graph = await mock_tool_agent_with_return_direct.build_graph()
    assert isinstance(graph, CompiledStateGraph)
    assert list(graph.nodes.keys()) == ['__start__', 'agent', 'tool']
    assert graph.builder.edges == {('__start__', 'agent')}
    assert set(graph.builder.branches.get('agent').get('conditional_edge').ends.keys()) == {
        AgentDecision.TOOL, AgentDecision.END
    }
    tool_branches = graph.builder.branches.get('tool')
    assert tool_branches is not None
    assert 'tool_conditional_edge' in tool_branches
    assert set(tool_branches.get('tool_conditional_edge').ends.keys()) == {AgentDecision.END, AgentDecision.TOOL}


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


async def test_tool_conditional_edge_no_return_direct(mock_tool_agent):
    message = ToolMessage(content='mock tool response', name='Tool A', tool_call_id='Tool A')
    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='test'), message])
    decision = await mock_tool_agent.tool_conditional_edge(mock_state)
    assert decision == AgentDecision.TOOL


async def test_tool_conditional_edge_return_direct_match(mock_tool_agent_with_return_direct):
    message = ToolMessage(content='mock tool response', name='Tool A', tool_call_id='Tool A')
    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='test'), message])
    decision = await mock_tool_agent_with_return_direct.tool_conditional_edge(mock_state)
    assert decision == AgentDecision.END


async def test_tool_conditional_edge_return_direct_no_match(mock_tool_agent_with_return_direct):
    message = ToolMessage(content='mock tool response', name='Tool B', tool_call_id='Tool B')
    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='test'), message])
    decision = await mock_tool_agent_with_return_direct.tool_conditional_edge(mock_state)
    assert decision == AgentDecision.TOOL


async def test_tool_conditional_edge_no_name_attribute(mock_tool_agent_with_return_direct):
    message = AIMessage(content='mock response')
    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='test'), message])
    decision = await mock_tool_agent_with_return_direct.tool_conditional_edge(mock_state)
    assert decision == AgentDecision.TOOL


async def test_tool_conditional_edge_empty_messages(mock_tool_agent_with_return_direct):
    mock_state = ToolCallAgentGraphState(messages=[])
    decision = await mock_tool_agent_with_return_direct.tool_conditional_edge(mock_state)
    assert decision == AgentDecision.TOOL


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
    mock_state.messages.append(message)
    response = await mock_tool_agent.tool_node(mock_state)
    response = response.messages[-1]
    assert isinstance(response, ToolMessage)
    assert response.content == 'mock query'
    assert response.name == 'Tool A'


@pytest.fixture(name="mock_tool_graph", scope="module")
async def mock_graph(mock_tool_agent):
    return await mock_tool_agent.build_graph()


@pytest.fixture(name="mock_tool_graph_with_return_direct", scope="module")
async def mock_graph_with_return_direct(mock_tool_agent_with_return_direct):
    return await mock_tool_agent_with_return_direct.build_graph()


async def test_graph(mock_tool_graph):
    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='please, mock tool call!')])
    response = await mock_tool_graph.ainvoke(mock_state)
    response = ToolCallAgentGraphState(**response)
    response = response.messages[-1]
    assert isinstance(response, AIMessage)
    assert response.content == 'mock query'


async def test_graph_with_return_direct(mock_tool_graph_with_return_direct):
    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='please, mock tool call!')])
    response = await mock_tool_graph_with_return_direct.ainvoke(mock_state)
    response = ToolCallAgentGraphState(**response)
    last_message = response.messages[-1]
    assert isinstance(last_message, ToolMessage)
    assert last_message.name == 'Tool A'


async def test_graph_astream_yields_message_chunks(mock_tool_graph):
    """Test that graph.astream with stream_mode='messages' yields message chunks from the agent node.

    This validates the streaming path used by _stream_fn in register.py. With a real LLM the chunks
    will be AIMessageChunk; the mock LLM produces AIMessage which LangGraph may wrap differently,
    so we accept any BaseMessage subclass from the agent node.
    """
    from langchain_core.messages import BaseMessage

    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='please, mock tool call!')])
    agent_messages = []
    async for msg, metadata in mock_tool_graph.astream(
            mock_state, config={'recursion_limit': 5}, stream_mode="messages"):
        if isinstance(msg, BaseMessage) and metadata.get("langgraph_node") == "agent":
            agent_messages.append(msg)

    assert len(agent_messages) > 0, "Expected at least one message from the agent node via stream_mode='messages'"
    combined_content = "".join(m.content for m in agent_messages if m.content)
    assert len(combined_content) > 0, "Expected non-empty content from streamed agent messages"


def test_tool_call_chunk_serialization():
    """Test that ChatResponseChunk with tool_calls in ChoiceDelta serializes to OpenAI-compatible SSE format."""
    chunk = ChatResponseChunk(
        id="test-chunk-id",
        choices=[
            ChatResponseChunkChoice(
                index=0,
                delta=ChoiceDelta(tool_calls=[
                    ChoiceDeltaToolCall(index=0,
                                        id="call_abc123",
                                        type="function",
                                        function=ChoiceDeltaToolCallFunction(
                                            name="test_tool",
                                            arguments="",
                                        ))
                ]),
                finish_reason=None,
            )
        ],
        created=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
    )

    sse_data = chunk.get_stream_data()
    assert sse_data.startswith("data: ")
    assert sse_data.endswith("\n\n")

    payload = json.loads(sse_data[len("data: "):])
    assert payload["id"] == "test-chunk-id"
    assert len(payload["choices"]) == 1

    delta = payload["choices"][0]["delta"]
    assert "tool_calls" in delta
    assert len(delta["tool_calls"]) == 1

    tc = delta["tool_calls"][0]
    assert tc["index"] == 0
    assert tc["id"] == "call_abc123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "test_tool"
    assert tc["function"]["arguments"] == ""


@pytest.fixture(name="error_mock_llm")
def fixture_error_mock_llm():
    """Mock LLM with a settable ``max_tokens`` field for truncation retry tests."""
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGeneration
    from langchain_core.outputs import ChatResult

    class _ErrorMockLLM(BaseChatModel):
        max_tokens: int | None = None

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: typing.Any,
        ) -> ChatResult:
            msg = AIMessage(content=messages[-1].content, response_metadata={"mock_llm_response": True})
            return ChatResult(generations=[ChatGeneration(message=msg)])

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: typing.Any,
        ) -> ChatResult:
            msg = AIMessage(content=messages[-1].content, response_metadata={"mock_llm_response": True})
            return ChatResult(generations=[ChatGeneration(message=msg)])

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self) -> str:
            return "error-mock-llm"

    return _ErrorMockLLM()


def _make_agent(llm, mock_tool, **kwargs) -> ToolCallAgentGraph:
    """Helper to build a ``ToolCallAgentGraph`` with a single tool and custom params."""
    return ToolCallAgentGraph(llm=llm, tools=[mock_tool("T")], **kwargs)


async def test_validate_truncation_raises_when_disabled(error_mock_llm, mock_tool):
    """finish_reason=length with max_truncation_retries=0 must raise RuntimeError with token info."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0)
    response = AIMessage(
        content="partial output here",
        response_metadata={
            "finish_reason": "length", "model_name": "test-model"
        },
        usage_metadata={
            "output_tokens": 100, "input_tokens": 50, "total_tokens": 150
        },
    )
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with pytest.raises(RuntimeError, match="truncated") as exc_info:
        await agent._validate_llm_response(response, state)

    msg: str = str(exc_info.value)
    assert "model=test-model" in msg
    assert "output_tokens=100" in msg
    assert "input_tokens=50" in msg
    assert "total_tokens=150" in msg
    assert "partial output here" in msg


async def test_validate_truncation_delegates_to_retry(error_mock_llm, mock_tool):
    """finish_reason=length with max_truncation_retries>0 delegates to _retry_on_truncation."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=2, truncation_scaling_fn=lambda c: c + 512)
    truncated = AIMessage(content="partial", response_metadata={"finish_reason": "length"})
    good = AIMessage(content="complete", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_retry_on_truncation", new_callable=AsyncMock, return_value=good) as mock_retry:
        result = await agent._validate_llm_response(truncated, state)

    mock_retry.assert_awaited_once()
    assert result.content == "complete"


async def test_validate_empty_response_raises_when_disabled(error_mock_llm, mock_tool):
    """Empty response with max_empty_response_retries=0 must raise RuntimeError."""
    agent = _make_agent(error_mock_llm, mock_tool, max_empty_response_retries=0)
    empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with pytest.raises(RuntimeError, match="empty response"):
        await agent._validate_llm_response(empty, state)


async def test_validate_empty_response_delegates_to_retry(error_mock_llm, mock_tool):
    """Empty response with retries>0 delegates to _retry_on_empty_response."""
    agent = _make_agent(error_mock_llm, mock_tool, max_empty_response_retries=2)
    empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    good = AIMessage(content="actual answer", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_retry_on_empty_response", new_callable=AsyncMock, return_value=good) as mock_retry:
        result = await agent._validate_llm_response(empty, state)

    mock_retry.assert_awaited_once()
    assert result.content == "actual answer"


def test_get_token_usage_from_openai_response_metadata(error_mock_llm, mock_tool):
    """Falls back to response_metadata['usage'] (OpenAI format) when usage_metadata is absent."""
    agent = _make_agent(error_mock_llm, mock_tool)
    response = AIMessage(
        content="hi",
        response_metadata={"usage": {
            "prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20
        }},
    )
    usage = agent._get_token_usage(response)
    assert usage["input_tokens"] == 5
    assert usage["output_tokens"] == 15
    assert usage["total_tokens"] == 20


async def test_retry_on_truncation_succeeds(error_mock_llm, mock_tool):
    """Truncation retry succeeds when the retried LLM call finishes normally."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=3, truncation_scaling_fn=lambda c: c + 512)
    first_response = AIMessage(
        content="partial",
        response_metadata={"finish_reason": "length"},
        usage_metadata={
            "output_tokens": 100, "input_tokens": 50, "total_tokens": 150
        },
    )
    good = AIMessage(content="complete", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=good):
        result = await agent._retry_on_truncation(first_response, state)

    assert result.content == "complete"
    assert result.response_metadata["finish_reason"] == "stop"
    assert agent._current_max_tokens == 612
    assert agent._truncation_retries_remaining == 2


async def test_retry_on_truncation_exhausted(error_mock_llm, mock_tool):
    """All truncation retries exhausted raises RuntimeError."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=2, truncation_scaling_fn=lambda c: c + 512)
    first_response = AIMessage(
        content="partial",
        response_metadata={"finish_reason": "length"},
        usage_metadata={
            "output_tokens": 100, "input_tokens": 50, "total_tokens": 150
        },
    )
    still_truncated = AIMessage(
        content="partial",
        response_metadata={"finish_reason": "length"},
        usage_metadata={
            "output_tokens": 200, "input_tokens": 50, "total_tokens": 250
        },
    )
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=still_truncated):
        with pytest.raises(RuntimeError, match="LLM output still truncated after 2 retries"):
            await agent._retry_on_truncation(first_response, state)


async def test_retry_on_truncation_increments_from_usage(error_mock_llm, mock_tool):
    """When max_tokens is not configured, the base is taken from usage_metadata output_tokens."""
    increment: int = 256
    agent = _make_agent(
        error_mock_llm,
        mock_tool,
        max_truncation_retries=3,
        truncation_scaling_fn=lambda c: c + increment,
    )
    assert agent.llm.max_tokens is None
    first_response = AIMessage(
        content="partial",
        response_metadata={"finish_reason": "length"},
        usage_metadata={
            "output_tokens": 100, "input_tokens": 50, "total_tokens": 150
        },
    )
    still_truncated = AIMessage(content="partial", response_metadata={"finish_reason": "length"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    observed_max_tokens: list[int | None] = []

    async def _capture_and_invoke(s):
        observed_max_tokens.append(getattr(agent.bound_llm, "kwargs", {}).get("max_tokens"))
        return still_truncated

    with patch.object(agent, "_invoke_llm", side_effect=_capture_and_invoke):
        with pytest.raises(RuntimeError, match="LLM output still truncated after 3 retries"):
            await agent._retry_on_truncation(first_response, state)

    assert observed_max_tokens == [356, 612, 868]


async def test_retry_on_truncation_increments_from_configured_max_tokens(error_mock_llm, mock_tool):
    """When max_tokens is already configured on the LLM, it is used as the base instead of usage data."""
    increment: int = 512
    error_mock_llm.max_tokens = 50
    agent = _make_agent(
        error_mock_llm,
        mock_tool,
        max_truncation_retries=2,
        truncation_scaling_fn=lambda c: c + increment,
    )
    first_response = AIMessage(
        content="partial",
        response_metadata={"finish_reason": "length"},
        usage_metadata={
            "output_tokens": 48, "input_tokens": 30, "total_tokens": 78
        },
    )
    still_truncated = AIMessage(content="partial", response_metadata={"finish_reason": "length"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    observed_max_tokens: list[int | None] = []

    async def _capture_and_invoke(s):
        observed_max_tokens.append(getattr(agent.bound_llm, "kwargs", {}).get("max_tokens"))
        return still_truncated

    with patch.object(agent, "_invoke_llm", side_effect=_capture_and_invoke):
        with pytest.raises(RuntimeError, match="LLM output still truncated after 2 retries"):
            await agent._retry_on_truncation(first_response, state)

    assert observed_max_tokens == [562, 1074]


async def test_retry_on_truncation_persists_across_calls(error_mock_llm, mock_tool):
    """Retries and max_tokens carry forward across multiple truncation events."""
    agent = _make_agent(
        error_mock_llm,
        mock_tool,
        max_truncation_retries=4,
        truncation_scaling_fn=lambda c: c + 100,
    )
    truncated = AIMessage(
        content="partial",
        response_metadata={"finish_reason": "length"},
        usage_metadata={
            "output_tokens": 200, "input_tokens": 50, "total_tokens": 250
        },
    )
    good = AIMessage(content="ok", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=good):
        await agent._retry_on_truncation(truncated, state)

    assert agent._current_max_tokens == 300
    assert agent._truncation_retries_remaining == 3

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=good):
        await agent._retry_on_truncation(truncated, state)

    assert agent._current_max_tokens == 400
    assert agent._truncation_retries_remaining == 2

    still_truncated = AIMessage(
        content="partial",
        response_metadata={"finish_reason": "length"},
        usage_metadata={
            "output_tokens": 400, "input_tokens": 50, "total_tokens": 450
        },
    )
    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=still_truncated):
        with pytest.raises(RuntimeError, match="LLM output still truncated after 4 retries"):
            await agent._retry_on_truncation(truncated, state)

    assert agent._truncation_retries_remaining == 0


async def test_retry_on_empty_succeeds(error_mock_llm, mock_tool):
    """Empty-response retry succeeds when the retried LLM call returns content."""
    agent = _make_agent(error_mock_llm, mock_tool, max_empty_response_retries=3)
    first_meta: dict = {"finish_reason": "stop"}
    good = AIMessage(content="actual answer", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=good):
        result = await agent._retry_on_empty_response(state, first_meta)

    assert result.content == "actual answer"


async def test_retry_on_empty_exhausted(error_mock_llm, mock_tool):
    """All empty-response retries exhausted raises RuntimeError."""
    agent = _make_agent(error_mock_llm, mock_tool, max_empty_response_retries=2)
    first_meta: dict = {"finish_reason": "stop"}
    still_empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=still_empty):
        with pytest.raises(RuntimeError, match="empty responses after 2 retries"):
            await agent._retry_on_empty_response(state, first_meta)
