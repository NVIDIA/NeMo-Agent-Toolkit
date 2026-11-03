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
import logging
import typing
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ChatResponseChunkChoice
from nat.data_models.api_server import ChoiceDelta
from nat.data_models.api_server import ChoiceDeltaToolCall
from nat.data_models.api_server import ChoiceDeltaToolCallFunction
from nat.plugins.langchain.agent.base import TRUNCATED_TOOL_CALL_ARGS
from nat.plugins.langchain.agent.base import AgentDecision
from nat.plugins.langchain.agent.base import _incomplete_tool_call_args
from nat.plugins.langchain.agent.tool_calling_agent.agent import ToolCallAgentGraph
from nat.plugins.langchain.agent.tool_calling_agent.agent import ToolCallAgentGraphState
from nat.plugins.langchain.agent.tool_calling_agent.agent import _truncation_reason
from nat.plugins.langchain.agent.tool_calling_agent.agent import create_tool_calling_agent_prompt
from nat.plugins.langchain.agent.tool_calling_agent.agent import extract_token_usage
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


async def test_agent_node_logs_reasoning_content_without_message_repr(mock_tool_agent, caplog):
    mock_state = ToolCallAgentGraphState(messages=[HumanMessage(content='needs a tool')])
    mock_response = AIMessage(content="\n",
                              additional_kwargs={"reasoning_content": "I should call Tool A."},
                              tool_calls=[{
                                  "name": "Tool A",
                                  "args": {
                                      "query": "mock query"
                                  },
                                  "id": "Tool A",
                                  "type": "tool_call",
                              }])

    with patch.object(mock_tool_agent, '_invoke_llm', new_callable=AsyncMock) as mock_invoke_llm:
        mock_invoke_llm.return_value = mock_response

        with caplog.at_level(logging.INFO, logger="nat.plugins.langchain.agent.tool_calling_agent.agent"):
            await mock_tool_agent.agent_node(mock_state)

    assert "I should call Tool A." in caplog.text
    assert "additional_kwargs" not in caplog.text
    assert "tool_calls=[" not in caplog.text


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


async def test_stream_fn_no_duplicate_content(mock_tool_graph):
    """Regression: streaming must not duplicate the previous assistant message as a final chunk.

    When stream=true, _stream_fn uses graph.astream(stream_mode="messages") which emits
    both AIMessageChunk (incremental tokens) and AIMessage (state update). Accepting
    AIMessage causes the accumulated response to appear twice in the output. The fix
    filters to AIMessageChunk only. This test exercises the same graph.astream path and
    asserts that the filtering logic in _stream_fn would prevent duplicates.
    """
    from langchain_core.messages import AIMessageChunk

    prior_reply = "Hi there!"
    mock_state = ToolCallAgentGraphState(messages=[
        HumanMessage(content="hello"),
        AIMessage(content=prior_reply),
        HumanMessage(content="what can you do?"),
    ])

    chunk_contents = []
    full_contents = []
    async for msg, metadata in mock_tool_graph.astream(
            mock_state, config={"recursion_limit": 5}, stream_mode="messages"):
        if metadata.get("langgraph_node") != "agent":
            continue
        if isinstance(msg, AIMessageChunk) and isinstance(msg.content, str) and msg.content:
            chunk_contents.append(msg.content)
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            full_contents.append(msg.content)

    chunk_response = "".join(chunk_contents)
    full_response = "".join(full_contents)

    assert prior_reply in full_response, ("AIMessage state update with prior reply should appear in unfiltered stream")
    assert prior_reply not in chunk_response, (
        f"AIMessageChunk-only stream must not contain prior assistant reply: {chunk_response!r}")


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


@pytest.fixture(params=[
    pytest.param(({
        "model_name": "test-model", TRUNCATED_TOOL_CALL_ARGS: ["T"]
    },
                  "tool call arguments are not complete JSON (T)"),
                 id="cut_off_tool_call_args"),
    pytest.param(({
        "model_name": "test-model", "status": "incomplete", "incomplete_details": {
            "reason": "max_output_tokens"
        }
    },
                  "status='incomplete', reason='max_output_tokens'"),
                 id="responses_incomplete"),
])
def truncation_case(request):
    """The two ways a reply can carry a truncation verdict, shared by the tests that need them.

    The repo's tests take shared parameters from a fixture rather than repeating a ``parametrize`` block.
    """
    return request.param


_CUT_TOOL_CALL = {
    "name": "T", "args": {
        "path": "report.txt", "content": "line 01\nline"
    }, "id": "call_1", "type": "tool_call"
}


async def test_validate_cut_off_tool_call_raises(error_mock_llm, mock_tool, truncation_case):
    """A tool call from a reply that did not end normally must not reach the tool node."""
    response_metadata, reason = truncation_case
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0)
    response = AIMessage(content="", tool_calls=[_CUT_TOOL_CALL], response_metadata=response_metadata)
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with pytest.raises(RuntimeError, match="truncated") as exc_info:
        await agent._validate_llm_response(response, state)

    assert reason in str(exc_info.value)


async def test_validate_cut_off_tool_call_delegates_to_retry(error_mock_llm, mock_tool):
    """A Responses-API incomplete reply uses the same truncation retry path as finish_reason=length."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=1, truncation_scaling_fn=lambda c: c + 512)
    truncated = AIMessage(content="", tool_calls=[_CUT_TOOL_CALL], response_metadata={"status": "incomplete"})
    good = AIMessage(content="complete", response_metadata={"status": "completed"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_retry_on_truncation", new_callable=AsyncMock, return_value=good) as mock_retry:
        result = await agent._validate_llm_response(truncated, state)

    mock_retry.assert_awaited_once()
    assert result.content == "complete"


async def test_empty_response_retry_is_checked_for_truncation(error_mock_llm, mock_tool, truncation_case):
    """A cut-off reply returned by the empty-response retry must not reach the tool node either."""
    response_metadata, reason = truncation_case
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0, max_empty_response_retries=1)
    empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    cut_off = AIMessage(content="", tool_calls=[_CUT_TOOL_CALL], response_metadata=response_metadata)
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_retry_on_empty_response", new_callable=AsyncMock, return_value=cut_off) as mock_retry:
        with pytest.raises(RuntimeError, match="truncated") as exc_info:
            await agent._validate_llm_response(empty, state)

    mock_retry.assert_awaited_once()
    assert reason in str(exc_info.value)


async def test_empty_response_retry_delegates_truncation_to_retry(error_mock_llm, mock_tool):
    """A cut-off reply from the empty-response retry uses the same truncation retry path."""
    agent = _make_agent(error_mock_llm,
                        mock_tool,
                        max_truncation_retries=1,
                        max_empty_response_retries=1,
                        truncation_scaling_fn=lambda c: c + 512)
    empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    cut_off = AIMessage(content="", tool_calls=[_CUT_TOOL_CALL], response_metadata={"status": "incomplete"})
    good = AIMessage(content="complete", response_metadata={"status": "completed"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_retry_on_empty_response", new_callable=AsyncMock,
                      return_value=cut_off), patch.object(agent,
                                                          "_retry_on_truncation",
                                                          new_callable=AsyncMock,
                                                          return_value=good) as mock_retry:
        result = await agent._validate_llm_response(empty, state)

    mock_retry.assert_awaited_once()
    assert result.content == "complete"


class _ChunkStream:
    """Stand-in for ``prompt | llm``: replays the chunks a provider stream would emit."""

    def __init__(self, chunks: list[AIMessageChunk]):
        self._chunks = chunks

    @property
    def first(self) -> "_ChunkStream":
        return self

    async def astream(self, inputs: typing.Any) -> typing.AsyncIterator[AIMessageChunk]:
        """Replay the recorded chunks as an async stream."""
        for chunk in self._chunks:
            yield chunk


async def test_agent_node_raises_when_stream_ends_mid_tool_call(error_mock_llm, mock_tool):
    """Chunks accumulated into a tool call whose stream ended mid arguments must raise, not call the tool."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0)
    agent.agent = _ChunkStream([
        AIMessageChunk(content=""),
        AIMessageChunk(content="", response_metadata={
            "model_provider": "openai", "model_name": "test-model"
        }),
        AIMessageChunk(content="",
                       tool_call_chunks=[{
                           "name": "T",
                           "args": '{"path": "report.txt", "content": "line 01',
                           "id": "call_1",
                           "index": 0
                       }]),
    ])
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with pytest.raises(RuntimeError, match="tool call arguments are not complete JSON"):
        await agent.agent_node(state)

    # The reply never reached the state, so no tool call can be scheduled from it
    assert len(state.messages) == 1


def _stream(*calls: tuple[str, str], metadata: dict | None = None) -> "_ChunkStream":
    """Chunks a provider stream would emit for the given ``(name, raw_argument_string)`` tool calls."""
    chunks: list[AIMessageChunk] = [AIMessageChunk(content="")]
    if metadata is not None:
        chunks.append(AIMessageChunk(content="", response_metadata=metadata))
    for index, (name, raw_args) in enumerate(calls):
        chunks.append(
            AIMessageChunk(content="",
                           tool_call_chunks=[{
                               "name": name, "args": raw_args, "id": f"call_{index}", "index": index
                           }]))
    return _ChunkStream(chunks)


_COMPLETE_ARGS: str = '{"path": "report.txt", "content": "line 01\\nline 02\\n"}'


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({}, id="no_metadata_at_all"),
        pytest.param({"model_name": "test-model"}, id="no_finish_reason"),
        pytest.param({"finish_reason": "stop_sequence"}, id="unknown_finish_reason"),
    ],
)
async def test_stream_runs_a_complete_tool_call_whatever_the_finish_reason(error_mock_llm, mock_tool, metadata):
    """The false rejection Will Killian flagged: a provider that omits or renames finish_reason must still work.

    Nothing about the arguments is wrong, so the reply runs the tool like any other.
    """
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0)
    agent.agent = _stream(("T", _COMPLETE_ARGS), metadata=metadata)
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    await agent.agent_node(state)

    assert state.messages[-1].tool_calls[0]["name"] == "T"
    assert state.messages[-1].tool_calls[0]["args"]["content"].startswith("line 01")


def test_incomplete_tool_call_args_skips_parsed_arguments():
    """Arguments that arrive already parsed are not a raw string, so there is nothing to validate."""
    chunk = SimpleNamespace(tool_call_chunks=[{"name": "T", "args": {"path": "r.txt"}}, {"name": "T", "args": None}])

    assert _incomplete_tool_call_args(chunk) == []


async def test_stream_with_cut_off_arguments_among_complete_calls_raises(error_mock_llm, mock_tool):
    """Each tool call is judged on its own: one cut-off call rejects the reply even beside a complete one."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0)
    agent.agent = _stream(("T", _COMPLETE_ARGS), ("T", '{"path": "second.txt", "content": "line 01'), metadata={})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with pytest.raises(RuntimeError, match=r"tool call arguments are not complete JSON \(T\)"):
        await agent.agent_node(state)


@pytest.mark.parametrize("raw_args", ["", "{}"])
async def test_stream_runs_a_tool_call_that_takes_no_arguments(error_mock_llm, mock_tool, raw_args):
    """No arguments is a complete argument string, not a truncated one."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0)
    agent.agent = _stream(("T", raw_args), metadata={"finish_reason": "tool_calls"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    await agent.agent_node(state)

    assert state.messages[-1].tool_calls[0]["name"] == "T"


async def test_non_streamed_reply_is_not_checked(error_mock_llm, mock_tool):
    """A reply that never streamed has no raw chunks to inspect, so its behaviour is unchanged."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0)
    agent.agent = _ChunkStream([AIMessage(content="", tool_calls=[_CUT_TOOL_CALL])])  # type: ignore[list-item]
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    await agent.agent_node(state)

    assert state.messages[-1].tool_calls == [_CUT_TOOL_CALL]


async def test_nim_stream_with_cut_off_arguments_raises(error_mock_llm, mock_tool):
    """NIM leaves response_metadata empty, so the argument string is the only evidence there is."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0)
    agent.agent = _stream(("T", '{"path": "report.txt", "content": "line 01'), metadata={})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with pytest.raises(RuntimeError, match="tool call arguments are not complete JSON"):
        await agent.agent_node(state)


def _cut_off_chunk(raw_args: str = '{"path": "report.txt", "content": "line 01') -> AIMessageChunk:
    """The reply ``_invoke_llm`` really returns: an accumulated chunk, not a converted message."""
    return AIMessageChunk(content="", tool_call_chunks=[{"name": "T", "args": raw_args, "id": "call_1", "index": 0}])


async def test_empty_response_retry_is_converted_before_the_truncation_check(error_mock_llm, mock_tool):
    """A cut-off tool call returned by the empty-response retry must still be caught.

    ``_retry_on_empty_response`` hands back the accumulated chunk. The argument verdict lives on the converted
    message, so without a conversion the check here sees only finish_reason and does nothing.
    """
    agent = _make_agent(error_mock_llm, mock_tool, max_empty_response_retries=1)
    empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=_cut_off_chunk()):
        with pytest.raises(RuntimeError, match="tool call arguments are not complete JSON"):
            await agent._validate_llm_response(empty, state)


async def test_empty_retry_then_truncation_retry_returning_empty_raises(error_mock_llm, mock_tool):
    """empty reply → truncation retry → still empty: the empty reply must not reach the tool node."""
    agent = _make_agent(error_mock_llm,
                        mock_tool,
                        max_empty_response_retries=1,
                        max_truncation_retries=1,
                        truncation_scaling_fn=lambda c: c + 512)
    empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    cut_off = AIMessage(content="", tool_calls=[_CUT_TOOL_CALL], response_metadata={"finish_reason": "length"})
    still_empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_retry_on_empty_response", new_callable=AsyncMock, return_value=cut_off):
        with patch.object(agent, "_retry_on_truncation", new_callable=AsyncMock, return_value=still_empty):
            with pytest.raises(RuntimeError, match="LLM returned an empty response"):
                await agent._validate_llm_response(empty, state)


async def test_truncation_retry_returning_an_empty_reply_raises(error_mock_llm, mock_tool):
    """A truncation retry that comes back with nothing usable is the same dead end as an exhausted
    empty-response retry, and must not reach the tool node as a successful reply."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=1, truncation_scaling_fn=lambda c: c + 512)
    cut_off = AIMessage(content="", tool_calls=[_CUT_TOOL_CALL], response_metadata={"status": "incomplete"})
    still_empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_retry_on_truncation", new_callable=AsyncMock, return_value=still_empty):
        with pytest.raises(RuntimeError, match="LLM returned an empty response"):
            await agent._validate_llm_response(cut_off, state)


@pytest.mark.parametrize(
    "response_metadata",
    [
        pytest.param({"finish_reason": "tool_calls"}, id="chat_tool_calls"),
        pytest.param({"status": "completed"}, id="responses_completed"),
    ],
)
async def test_validate_complete_tool_call_passes(error_mock_llm, mock_tool, response_metadata):
    """Tool calls from a reply that ended normally are returned unchanged."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0)
    response = AIMessage(content="", tool_calls=[_CUT_TOOL_CALL], response_metadata=response_metadata)
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    result = await agent._validate_llm_response(response, state)

    assert result.tool_calls == response.tool_calls


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(AIMessage(content="answer", response_metadata={"model_provider": "openai"}),
                     id="openai_text_without_finish_reason"),
        pytest.param(AIMessage(content="", tool_calls=[_CUT_TOOL_CALL], response_metadata={}),
                     id="tool_call_without_provider_metadata"),
        pytest.param(AIMessage(content="",
                               tool_calls=[_CUT_TOOL_CALL],
                               response_metadata={
                                   "model_provider": "anthropic", "stop_reason": "tool_use"
                               }),
                     id="other_provider_tool_call"),
    ],
)
async def test_validate_reply_without_openai_terminator_passes(error_mock_llm, mock_tool, response):
    """Only OpenAI-wire tool calls are checked for a missing terminator; text replies and other providers pass."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0)
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    result = await agent._validate_llm_response(response, state)

    assert result is response


async def test_validate_text_reply_without_finish_reason_passes(error_mock_llm, mock_tool):
    """A text reply is never judged by ``finish_reason``: a provider that omits it must not be rejected."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0)
    response = AIMessage(content='Action: T\nAction Input: {"path": "report.txt"}',
                         response_metadata={"model_provider": "openai"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    result = await agent._validate_llm_response(response, state)

    assert result is response


async def test_validate_nim_reply_without_metadata_passes(error_mock_llm, mock_tool):
    """ChatNVIDIA (NIM) leaves response_metadata empty; a complete tool call must still run.

    This is the false rejection the provider allowlist caused: no finish_reason is reported and nothing about the
    arguments is wrong, so there is nothing to reject.
    """
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=0)
    response = AIMessage(content="", tool_calls=[_CUT_TOOL_CALL], response_metadata={})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    result = await agent._validate_llm_response(response, state)

    assert result is response


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


async def test_retry_on_truncation_converts_a_streamed_retry(error_mock_llm, mock_tool):
    """The retry's streamed chunk is converted before it leaves this method.

    ``agent_node`` converts the first reply, but the retry runs afterwards. An unconverted
    chunk reaches state without ``additional_kwargs["tool_calls"]``, the OpenAI wire form
    that later model turns read.
    """
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=2, truncation_scaling_fn=lambda c: c + 512)
    first_response = AIMessage(content="partial",
                               response_metadata={"finish_reason": "length"},
                               usage_metadata={
                                   "output_tokens": 100, "input_tokens": 50, "total_tokens": 150
                               })
    retry_chunk = AIMessageChunk(content="",
                                 tool_calls=[_CUT_TOOL_CALL],
                                 response_metadata={"finish_reason": "tool_calls"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=retry_chunk):
        result = await agent._retry_on_truncation(first_response, state)

    assert not isinstance(result, AIMessageChunk)
    assert result.tool_calls
    assert result.additional_kwargs["tool_calls"][0]["id"] == "call_1"


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


async def test_empty_replies_exhaust_the_configured_budget(error_mock_llm, mock_tool):
    """The caller owns the empty-response budget: N configured retries allow exactly N extra requests."""
    agent = _make_agent(error_mock_llm, mock_tool, max_empty_response_retries=2)
    still_empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=still_empty) as mock_invoke:
        with pytest.raises(RuntimeError, match="empty response"):
            await agent._validate_llm_response(still_empty, state)

    assert mock_invoke.await_count == 2


async def test_empty_reply_after_a_truncation_retry_uses_the_empty_budget(error_mock_llm, mock_tool):
    """A truncation retry that comes back empty is retried again, not raised on the spot."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=1, max_empty_response_retries=1)
    truncated = AIMessage(content="", response_metadata={"finish_reason": "length"})
    empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    good = AIMessage(content="final answer", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, side_effect=[empty, good]) as mock_invoke:
        result = await agent._validate_llm_response(truncated, state)

    assert result.content == "final answer"
    # One truncation retry, then one empty-response retry.
    assert mock_invoke.await_count == 2


async def test_truncation_retry_that_stays_empty_stops_at_the_configured_budget(error_mock_llm, mock_tool):
    """Each dead end spends its own budget once; the error comes after both are spent."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=1, max_empty_response_retries=1)
    truncated = AIMessage(content="", response_metadata={"finish_reason": "length"})
    empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=empty) as mock_invoke:
        with pytest.raises(RuntimeError, match="empty response"):
            await agent._validate_llm_response(truncated, state)

    assert mock_invoke.await_count == 2


async def test_no_empty_budget_raises_as_soon_as_the_truncation_retry_is_empty(error_mock_llm, mock_tool):
    """With max_empty_response_retries=0 the behaviour is unchanged: no second retry is attempted."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=1, max_empty_response_retries=0)
    truncated = AIMessage(content="", response_metadata={"finish_reason": "length"})
    empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=empty) as mock_invoke:
        with pytest.raises(RuntimeError, match="empty response"):
            await agent._validate_llm_response(truncated, state)

    assert mock_invoke.await_count == 1


async def test_alternating_empty_and_truncated_replies_stop(error_mock_llm, mock_tool):
    """Empty and truncated replies alternate: the budgets are consumed, never reset, so it terminates."""
    agent = _make_agent(error_mock_llm, mock_tool, max_truncation_retries=2, max_empty_response_retries=2)
    truncated = AIMessage(content="", response_metadata={"finish_reason": "length"})
    empty = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent,
                      "_invoke_llm",
                      new_callable=AsyncMock,
                      side_effect=[empty, truncated, empty, truncated, empty, truncated]) as mock_invoke:
        with pytest.raises(RuntimeError):
            await agent._validate_llm_response(truncated, state)

    # 1 + max_truncation_retries + max_empty_response_retries is the hard ceiling.
    assert mock_invoke.await_count <= 1 + 2 + 2


async def test_retry_on_empty_returns_a_truncated_reply(error_mock_llm, mock_tool, truncation_case):
    """A retry that comes back cut off is returned, not discarded as another empty reply.

    A Responses reply with ``status="incomplete"`` has neither text nor tool calls, so the
    empty-response loop would otherwise treat it as empty, exhaust its retries and report
    "empty responses" while the truncation path never sees it.
    """
    response_metadata, reason = truncation_case
    agent = _make_agent(error_mock_llm, mock_tool, max_empty_response_retries=3)
    first_meta: dict = {"finish_reason": "stop"}
    cut_off = AIMessage(content="", tool_calls=[_CUT_TOOL_CALL], response_metadata=response_metadata)
    state = ToolCallAgentGraphState(messages=[HumanMessage(content="test")])

    with patch.object(agent, "_invoke_llm", new_callable=AsyncMock, return_value=cut_off) as mock_invoke:
        result = await agent._retry_on_empty_response(state, first_meta)

    assert result is cut_off
    # The first retry is enough: the reply is not retried until the budget is gone.
    mock_invoke.assert_awaited_once()
    assert reason in _truncation_reason(result)


def test_extract_token_usage_carries_langchain_cached_tokens():
    """LangChain reports the cache as `input_token_details.cache_read` and it must survive.

    The extractor rebuilds the UsageMetadata from three scalars, so a details dict it does not
    copy is a details dict the caller never sees - which is how the cached count was being lost.
    """
    message = AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_token_details": {
                "cache_read": 80
            },
        },
    )
    usage = extract_token_usage(message)
    assert usage is not None
    assert usage["input_token_details"]["cache_read"] == 80


def test_extract_token_usage_carries_openai_cached_tokens():
    """OpenAI names the same number differently, and both must land on one key."""
    message = AIMessage(
        content="x",
        response_metadata={
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {
                    "cached_tokens": 80
                },
            }
        },
    )
    usage = extract_token_usage(message)
    assert usage is not None
    assert usage["input_token_details"]["cache_read"] == 80


def test_extract_token_usage_omits_cached_tokens_when_unreported():
    """Absent and zero are different facts, and only one of them is a claim about the cache."""
    message = AIMessage(
        content="x",
        response_metadata={"usage": {
            "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120
        }},
    )
    usage = extract_token_usage(message)
    assert usage is not None
    assert "input_token_details" not in usage


def test_extract_token_usage_returns_none_without_any_usage():
    """No usage at all stays None, so a caller can tell it from a genuine zero."""
    assert extract_token_usage(AIMessage(content="x")) is None


async def _run_tool_calling_agent(tool_calls_before_answer: int | None,
                                  max_iterations: int,
                                  stream: bool = False,
                                  script: list[str] | None = None) -> tuple[str, list[str], list[str]]:
    """Run the agent with a model that requests the tool `tool_calls_before_answer` times (always if None).

    `script` sets each model turn instead: "tool" (a valid tool call), "bad_tool" (a tool call whose arguments
    don't parse, so it is never run), "fail" (the model request raises before its first token); turns past the
    script answer. A stream that raises keeps the chunks sent before the error.

    Returns the text output, the tool's calls, and the ids of the tool calls streamed to the client.
    """
    from collections.abc import Iterator

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGeneration
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.outputs import ChatResult
    from langchain_core.tools import tool

    from nat.plugins.langchain.agent.tool_calling_agent.register import tool_calling_agent_workflow

    calls = []

    @tool
    def counter(q: str) -> str:
        """Counts its calls."""
        calls.append(q)
        return f"call {len(calls)}"

    class ScriptedToolCallLLM(BaseChatModel):
        turns: int = 0

        @property
        def _llm_type(self) -> str:
            return "scripted-tool-call"

        def bind_tools(self, tools: typing.Any, **kwargs: typing.Any) -> "ScriptedToolCallLLM":
            return self

        def _next_turn(self) -> AIMessageChunk:
            self.turns += 1
            if script is not None:
                kind = script[self.turns - 1] if self.turns <= len(script) else "answer"
            else:
                done = tool_calls_before_answer is not None and self.turns > tool_calls_before_answer
                kind = "answer" if done else "tool"
            if kind == "fail":
                raise RuntimeError("model request failed")
            if kind == "answer":
                return AIMessageChunk(content="all done")
            # "not json" can't be parsed even partially, so it becomes an invalid tool call and no tool runs;
            # the text keeps the turn from counting as an empty response, so the graph ends normally
            args = '{"q": "x"}' if kind == "tool" else "not json"
            return AIMessageChunk(content="" if kind == "tool" else "no valid call",
                                  tool_call_chunks=[{
                                      "name": "counter", "args": args, "id": f"call-{self.turns}", "index": 0
                                  }])

        def _generate(self,
                      messages: typing.Any,
                      stop: typing.Any = None,
                      run_manager: typing.Any = None,
                      **kwargs: typing.Any) -> ChatResult:
            return ChatResult(generations=[ChatGeneration(message=self._next_turn())])

        def _stream(self,
                    messages: typing.Any,
                    stop: typing.Any = None,
                    run_manager: typing.Any = None,
                    **kwargs: typing.Any) -> Iterator[ChatGenerationChunk]:
            yield ChatGenerationChunk(message=self._next_turn())

    builder = AsyncMock()
    builder.get_llm = AsyncMock(return_value=ScriptedToolCallLLM())
    builder.get_tools = AsyncMock(return_value=[counter])
    config = ToolCallAgentWorkflowConfig(tool_names=["counter"], llm_name="llm", max_iterations=max_iterations)
    streamed_tool_call_ids = []
    async with tool_calling_agent_workflow(config, builder) as function_info:
        if stream:
            from nat.data_models.api_server import ChatRequestOrMessage

            chunks = []
            try:
                async for chunk in function_info.stream_fn(ChatRequestOrMessage(input_message="hi")):
                    chunks.append(chunk)
            except RuntimeError:
                pass
            result = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
            deltas = [chunk.choices[0].delta for chunk in chunks if chunk.choices]
            streamed_tool_call_ids = [tool_call.id for delta in deltas for tool_call in delta.tool_calls or []]
        else:
            result = await function_info.single_fn("hi")
    return result, calls, streamed_tool_call_ids


@pytest.mark.parametrize("max_iterations", [1, 2, 5])
async def test_tool_calling_agent_never_exceeds_max_iterations(max_iterations):
    """`max_iterations` is the maximum number of tool calls the agent may perform."""
    result, calls, _ = await _run_tool_calling_agent(None, max_iterations)
    assert len(calls) == max_iterations
    assert "could not produce a final answer" in result


@pytest.mark.parametrize("max_iterations", [1, 2, 5])
async def test_tool_calling_agent_can_use_all_max_iterations_then_answer(max_iterations):
    result, calls, _ = await _run_tool_calling_agent(max_iterations, max_iterations)
    assert len(calls) == max_iterations
    assert result == "all done"


@pytest.mark.parametrize("max_iterations", [1, 2, 5])
async def test_tool_calling_agent_stream_never_exceeds_max_iterations(max_iterations):
    """The client is only told about tool calls the agent runs, not the one rejected at the limit."""
    result, calls, streamed_tool_call_ids = await _run_tool_calling_agent(None, max_iterations, stream=True)
    assert len(calls) == max_iterations
    assert streamed_tool_call_ids == [f"call-{turn}" for turn in range(1, max_iterations + 1)]
    assert "could not produce a final answer" in result


@pytest.mark.parametrize("max_iterations", [1, 2, 5])
async def test_tool_calling_agent_stream_can_use_all_max_iterations_then_answer(max_iterations):
    result, calls, streamed_tool_call_ids = await _run_tool_calling_agent(max_iterations, max_iterations, stream=True)
    assert len(calls) == max_iterations
    assert streamed_tool_call_ids == [f"call-{turn}" for turn in range(1, max_iterations + 1)]
    assert result == "all done"


async def test_tool_calling_agent_stream_keeps_tool_call_when_next_model_request_fails():
    """A tool call that ran reaches the client even if the next model request fails before its first token."""
    _, calls, streamed_tool_call_ids = await _run_tool_calling_agent(None, 2, stream=True, script=["tool", "fail"])
    assert len(calls) == 1
    assert streamed_tool_call_ids == ["call-1"]


async def test_tool_calling_agent_stream_drops_tool_call_that_never_ran():
    """Tool-call chunks for a call the graph never runs (its arguments don't parse) are not sent to the client."""
    result, calls, streamed_tool_call_ids = await _run_tool_calling_agent(None, 2, stream=True, script=["bad_tool"])
    assert result == "no valid call"  # the graph ended normally, without the tool node
    assert calls == []
    assert streamed_tool_call_ids == []


async def test_tool_calling_agent_returns_a_string_for_a_chat_request():
    """The non-streaming response stays a plain string, even when the model reports token usage."""
    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration
    from langchain_core.outputs import ChatResult
    from langchain_core.tools import tool

    from nat.data_models.api_server import ChatRequestOrMessage
    from nat.plugins.langchain.agent.tool_calling_agent.register import tool_calling_agent_workflow

    @tool
    def counter(q: str) -> str:
        """Counts its calls."""
        return "counted"

    class UsageReportingLLM(BaseChatModel):
        turns: int = 0

        @property
        def _llm_type(self) -> str:
            return "usage-reporting"

        def bind_tools(self, tools: typing.Any, **kwargs: typing.Any) -> "UsageReportingLLM":
            return self

        def _generate(self,
                      messages: typing.Any,
                      stop: typing.Any = None,
                      run_manager: typing.Any = None,
                      **kwargs: typing.Any) -> ChatResult:
            self.turns += 1
            if self.turns == 1:
                message = AIMessage(content="",
                                    tool_calls=[{
                                        "name": "counter", "args": {
                                            "q": "x"
                                        }, "id": "call-1"
                                    }],
                                    usage_metadata={
                                        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120
                                    })
            else:
                message = AIMessage(content="all done",
                                    usage_metadata={
                                        "input_tokens": 150, "output_tokens": 30, "total_tokens": 180
                                    })
            return ChatResult(generations=[ChatGeneration(message=message)])

    builder = AsyncMock()
    builder.get_llm = AsyncMock(return_value=UsageReportingLLM())
    builder.get_tools = AsyncMock(return_value=[counter])
    config = ToolCallAgentWorkflowConfig(tool_names=["counter"], llm_name="llm")
    request = ChatRequestOrMessage(messages=[{"role": "user", "content": "hi"}])
    async with tool_calling_agent_workflow(config, builder) as function_info:
        response = await function_info.single_fn(request)
    assert response == "all done"
