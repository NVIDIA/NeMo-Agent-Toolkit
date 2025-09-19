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

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from nat.control_flow.router_agent.agent import RouterAgentGraph
from nat.control_flow.router_agent.agent import RouterAgentGraphState
from nat.control_flow.router_agent.agent import create_router_agent_prompt
from nat.control_flow.router_agent.prompt import SYSTEM_PROMPT
from nat.control_flow.router_agent.prompt import USER_PROMPT
from nat.control_flow.router_agent.register import RouterAgentWorkflowConfig


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name: str, description: str = "Mock tool"):
        super().__init__(name=name, description=description)

    def _run(self, *args, **kwargs):
        return f"Mock response from {self.name}"

    async def _arun(self, *args, **kwargs):
        return f"Mock async response from {self.name}"


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def mock_branches():
    """Create mock branches for testing."""
    return [
        MockTool("calculator_tool", "Performs mathematical calculations"),
        MockTool("weather_service", "Provides weather information"),
        MockTool("email_tool", "Sends emails")
    ]


@pytest.fixture
def mock_prompt():
    """Create a mock prompt for testing."""
    return ChatPromptTemplate([("system", SYSTEM_PROMPT), ("user", USER_PROMPT)])


@pytest.fixture
def router_agent(mock_llm, mock_branches, mock_prompt):
    """Create a RouterAgentGraph instance for testing."""
    return RouterAgentGraph(llm=mock_llm,
                            branches=mock_branches,
                            prompt=mock_prompt,
                            max_router_retries=3,
                            detailed_logs=True)


@pytest.fixture
def mock_config():
    """Create a mock RouterAgentWorkflowConfig for testing."""
    config = Mock(spec=RouterAgentWorkflowConfig)
    config.system_prompt = None
    config.user_prompt = None
    return config


class TestRouterAgentGraphState:
    """Test RouterAgentGraphState schema and initialization."""

    def test_state_schema_initialization(self):
        """Test that RouterAgentGraphState initializes with correct defaults."""
        state = RouterAgentGraphState()

        assert isinstance(state.messages, list)
        assert len(state.messages) == 0
        assert isinstance(state.forward_message, BaseMessage)
        assert isinstance(state.forward_message, HumanMessage)
        assert state.forward_message.content == ""
        assert state.chosen_branch == ""

    def test_state_schema_with_values(self):
        """Test RouterAgentGraphState initialization with provided values."""
        messages = [HumanMessage(content="test")]
        relay_message = HumanMessage(content="relay test")
        chosen_branch = "calculator_tool"

        state = RouterAgentGraphState(messages=messages, forward_message=relay_message, chosen_branch=chosen_branch)

        assert state.messages == messages
        assert state.forward_message == relay_message
        assert state.chosen_branch == chosen_branch


class TestRouterAgentGraph:
    """Test RouterAgentGraph initialization and core functionality."""

    def test_initialization(self, mock_llm, mock_branches, mock_prompt):
        """Test RouterAgentGraph initialization."""
        agent = RouterAgentGraph(llm=mock_llm,
                                 branches=mock_branches,
                                 prompt=mock_prompt,
                                 max_router_retries=5,
                                 detailed_logs=True,
                                 log_response_max_chars=500)

        assert agent.llm == mock_llm
        assert agent._branches == mock_branches
        assert len(agent._branches_dict) == 3
        assert "calculator_tool" in agent._branches_dict
        assert "weather_service" in agent._branches_dict
        assert "email_tool" in agent._branches_dict
        assert agent.max_router_retries == 5
        assert agent.detailed_logs is True
        assert agent.log_response_max_chars == 500

    def test_get_branch(self, router_agent):
        """Test _get_branch method."""
        # Test existing branch
        branch = router_agent._get_branch("calculator_tool")
        assert branch is not None
        assert branch.name == "calculator_tool"

        # Test non-existing branch
        branch = router_agent._get_branch("non_existing_tool")
        assert branch is None

    @pytest.mark.asyncio
    async def test_agent_node_successful_branch_selection(self, router_agent):
        """Test agent_node successfully selects a branch."""
        # Mock LLM response that contains a branch name
        mock_response = AIMessage(content="calculator_tool")

        state = RouterAgentGraphState(forward_message=HumanMessage(content="Calculate 2+2"),
                                      messages=[HumanMessage(content="Previous message")])

        with patch.object(router_agent, '_get_chat_history', return_value="chat history"):
            with patch.object(router_agent, '_call_llm', return_value=mock_response) as mock_call_llm:
                result_state = await router_agent.agent_node(state)

        assert result_state.chosen_branch == "calculator_tool"
        assert len(result_state.messages) == 2  # Previous + new response
        assert result_state.messages[-1] == mock_response
        mock_call_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_node_retry_on_no_branch_selected(self, router_agent):
        """Test agent_node retries when no branch is selected."""
        # First two calls return responses without branch names
        # Third call returns a valid branch name
        mock_responses = [
            AIMessage(content="I'm thinking about this..."),
            AIMessage(content="Let me consider the options..."),
            AIMessage(content="calculator_tool")
        ]

        state = RouterAgentGraphState(forward_message=HumanMessage(content="Calculate 2+2"))

        with patch.object(router_agent, '_get_chat_history', return_value=""):
            with patch.object(router_agent, '_call_llm', side_effect=mock_responses) as mock_call_llm:
                result_state = await router_agent.agent_node(state)

        assert result_state.chosen_branch == "calculator_tool"
        assert mock_call_llm.call_count == 3

    @pytest.mark.asyncio
    async def test_agent_node_max_retries_exceeded(self, router_agent):
        """Test agent_node raises error when max retries exceeded."""
        # All calls return responses without branch names
        mock_response = AIMessage(content="I don't know")

        state = RouterAgentGraphState(forward_message=HumanMessage(content="Calculate 2+2"))

        with patch.object(router_agent, '_get_chat_history', return_value=""):
            with patch.object(router_agent, '_call_llm', return_value=mock_response) as mock_call_llm:
                with pytest.raises(RuntimeError, match="Router Agent failed to choose a branch"):
                    await router_agent.agent_node(state)

        assert mock_call_llm.call_count == 3  # max_router_retries

    @pytest.mark.asyncio
    async def test_agent_node_llm_exception(self, router_agent):
        """Test agent_node handles LLM exceptions."""
        state = RouterAgentGraphState(forward_message=HumanMessage(content="Calculate 2+2"))

        with patch.object(router_agent, '_get_chat_history', return_value=""):
            with patch.object(router_agent, '_call_llm', side_effect=Exception("LLM error")):
                with pytest.raises(Exception, match="LLM error"):
                    await router_agent.agent_node(state)

    @pytest.mark.asyncio
    async def test_branch_node_successful_execution(self, router_agent):
        """Test branch_node successfully executes a tool."""
        mock_tool_response = ToolMessage(content="Result: 4", tool_call_id="test")

        state = RouterAgentGraphState(forward_message=HumanMessage(content="Calculate 2+2"),
                                      chosen_branch="calculator_tool",
                                      messages=[HumanMessage(content="Previous message")])

        with patch.object(router_agent, '_call_tool', return_value=mock_tool_response) as mock_call_tool:
            result_state = await router_agent.branch_node(state)

        mock_call_tool.assert_called_once()
        assert len(result_state.messages) == 2  # Previous + tool response
        assert result_state.messages[-1] == mock_tool_response

    @pytest.mark.asyncio
    async def test_branch_node_empty_chosen_branch(self, router_agent):
        """Test branch_node raises error when chosen_branch is empty."""
        state = RouterAgentGraphState(forward_message=HumanMessage(content="Calculate 2+2"), chosen_branch="")

        with pytest.raises(RuntimeError, match="Router Agent failed to choose a branch"):
            await router_agent.branch_node(state)

    @pytest.mark.asyncio
    async def test_branch_node_invalid_branch(self, router_agent):
        """Test branch_node raises error when chosen_branch doesn't exist."""
        state = RouterAgentGraphState(forward_message=HumanMessage(content="Calculate 2+2"),
                                      chosen_branch="non_existing_tool")

        with pytest.raises(ValueError, match="Tool not found in config file"):
            await router_agent.branch_node(state)

    @pytest.mark.asyncio
    async def test_branch_node_tool_execution_exception(self, router_agent):
        """Test branch_node handles tool execution exceptions."""
        state = RouterAgentGraphState(forward_message=HumanMessage(content="Calculate 2+2"),
                                      chosen_branch="calculator_tool")

        with patch.object(router_agent, '_call_tool', side_effect=Exception("Tool error")):
            with pytest.raises(Exception, match="Tool error"):
                await router_agent.branch_node(state)

    @pytest.mark.asyncio
    async def test_build_graph(self, router_agent):
        """Test graph building and compilation."""
        with patch('nat.control_flow.router_agent.agent.StateGraph') as mock_state_graph:
            mock_graph_instance = Mock()
            mock_compiled_graph = Mock(spec=CompiledStateGraph)
            mock_graph_instance.compile.return_value = mock_compiled_graph
            mock_state_graph.return_value = mock_graph_instance

            result = await router_agent.build_graph()

            # Verify graph construction
            mock_state_graph.assert_called_once_with(RouterAgentGraphState)
            mock_graph_instance.add_node.assert_any_call("agent", router_agent.agent_node)
            mock_graph_instance.add_node.assert_any_call("branch", router_agent.branch_node)
            mock_graph_instance.add_edge.assert_called_once_with("agent", "branch")
            mock_graph_instance.set_entry_point.assert_called_once_with("agent")
            mock_graph_instance.compile.assert_called_once()

            assert result == mock_compiled_graph
            assert router_agent.graph == mock_compiled_graph

    @pytest.mark.asyncio
    async def test_build_graph_exception(self, router_agent):
        """Test build_graph handles exceptions."""
        with patch('nat.control_flow.router_agent.agent.StateGraph', side_effect=Exception("Graph error")):
            with pytest.raises(Exception, match="Graph error"):
                await router_agent.build_graph()


class TestPromptValidation:
    """Test prompt validation methods."""

    def test_validate_system_prompt_valid(self):
        """Test validate_system_prompt with valid prompt."""
        valid_prompt = "System prompt with {branches} and {branch_names}"
        assert RouterAgentGraph.validate_system_prompt(valid_prompt) is True

    def test_validate_system_prompt_missing_branches(self):
        """Test validate_system_prompt with missing {branches}."""
        invalid_prompt = "System prompt with {branch_names} only"
        assert RouterAgentGraph.validate_system_prompt(invalid_prompt) is False

    def test_validate_system_prompt_missing_branch_names(self):
        """Test validate_system_prompt with missing {branch_names}."""
        invalid_prompt = "System prompt with {branches} only"
        assert RouterAgentGraph.validate_system_prompt(invalid_prompt) is False

    def test_validate_system_prompt_missing_both(self):
        """Test validate_system_prompt with missing both variables."""
        invalid_prompt = "System prompt without required variables"
        assert RouterAgentGraph.validate_system_prompt(invalid_prompt) is False

    def test_validate_user_prompt_valid(self):
        """Test validate_user_prompt with valid prompt."""
        valid_prompt = "User prompt with {chat_history} and {request}"
        assert RouterAgentGraph.validate_user_prompt(valid_prompt) is True

    def test_validate_user_prompt_missing_chat_history(self):
        """Test validate_user_prompt with missing {chat_history}."""
        invalid_prompt = "User prompt with {request} only"
        assert RouterAgentGraph.validate_user_prompt(invalid_prompt) is False

    def test_validate_user_prompt_empty(self):
        """Test validate_user_prompt with empty prompt."""
        assert RouterAgentGraph.validate_user_prompt("") is False

    def test_validate_user_prompt_none(self):
        """Test validate_user_prompt with None prompt."""
        assert RouterAgentGraph.validate_user_prompt(None) is False


class TestCreateRouterAgentPrompt:
    """Test create_router_agent_prompt function."""

    def test_create_prompt_default_prompts(self, mock_config):
        """Test create_router_agent_prompt with default prompts."""
        mock_config.system_prompt = None
        mock_config.user_prompt = None

        prompt = create_router_agent_prompt(mock_config)

        assert isinstance(prompt, ChatPromptTemplate)
        assert len(prompt.messages) == 2
        assert prompt.messages[0].prompt.template == SYSTEM_PROMPT
        assert prompt.messages[1].prompt.template == USER_PROMPT

    def test_create_prompt_custom_prompts(self, mock_config):
        """Test create_router_agent_prompt with custom prompts."""
        custom_system = "Custom system with {branches} and {branch_names}"
        custom_user = "Custom user with {chat_history} and {request}"

        mock_config.system_prompt = custom_system
        mock_config.user_prompt = custom_user

        prompt = create_router_agent_prompt(mock_config)

        assert isinstance(prompt, ChatPromptTemplate)
        assert prompt.messages[0].prompt.template == custom_system
        assert prompt.messages[1].prompt.template == custom_user

    def test_create_prompt_invalid_system_prompt(self, mock_config):
        """Test create_router_agent_prompt with invalid system prompt."""
        mock_config.system_prompt = "Invalid system prompt"
        mock_config.user_prompt = None

        with pytest.raises(ValueError, match="Invalid system_prompt"):
            create_router_agent_prompt(mock_config)

    def test_create_prompt_invalid_user_prompt(self, mock_config):
        """Test create_router_agent_prompt with invalid user prompt."""
        mock_config.system_prompt = None
        mock_config.user_prompt = "Invalid user prompt"

        with pytest.raises(ValueError, match="Invalid user_prompt"):
            create_router_agent_prompt(mock_config)


class TestRouterAgentIntegration:
    """Integration tests for RouterAgentGraph."""

    @pytest.mark.asyncio
    async def test_full_workflow_success(self, router_agent):
        """Test complete workflow from agent_node to branch_node."""
        # Setup state
        state = RouterAgentGraphState(forward_message=HumanMessage(content="Calculate 2+2"))

        # Mock agent_node to select a branch
        mock_agent_response = AIMessage(content="calculator_tool")

        # Mock branch_node tool execution
        mock_tool_response = ToolMessage(content="Result: 4", tool_call_id="test")

        with patch.object(router_agent, '_get_chat_history', return_value=""):
            with patch.object(router_agent, '_call_llm', return_value=mock_agent_response):
                with patch.object(router_agent, '_call_tool', return_value=mock_tool_response):
                    # Execute agent_node
                    state = await router_agent.agent_node(state)
                    assert state.chosen_branch == "calculator_tool"

                    # Execute branch_node
                    state = await router_agent.branch_node(state)
                    assert len(state.messages) == 2
                    assert state.messages[-1] == mock_tool_response

    def test_agent_initialization_with_different_configs(self, mock_llm, mock_branches):
        """Test agent initialization with various configurations."""
        prompt = ChatPromptTemplate([("system", SYSTEM_PROMPT), ("user", USER_PROMPT)])

        # Test with minimal config
        agent1 = RouterAgentGraph(llm=mock_llm, branches=mock_branches, prompt=prompt)
        assert agent1.max_router_retries == 3
        assert agent1.detailed_logs is False
        assert agent1.log_response_max_chars == 1000

        # Test with custom config
        agent2 = RouterAgentGraph(llm=mock_llm,
                                  branches=mock_branches,
                                  prompt=prompt,
                                  max_router_retries=5,
                                  detailed_logs=True,
                                  log_response_max_chars=2000)
        assert agent2.max_router_retries == 5
        assert agent2.detailed_logs is True
        assert agent2.log_response_max_chars == 2000

    def test_branch_selection_case_insensitive(self, router_agent):
        """Test that branch selection is case insensitive."""
        # Test various case combinations
        test_cases = [("CALCULATOR_TOOL", "calculator_tool"), ("Calculator_Tool", "calculator_tool"),
                      ("weather_SERVICE", "weather_service"), ("EMAIL_tool", "email_tool")]

        for response_content, expected_branch in test_cases:
            state = RouterAgentGraphState()
            state.messages = [AIMessage(content=response_content)]

            # Simulate the branch selection logic from agent_node
            for branch in router_agent._branches:
                if branch.name.lower() in response_content.lower():
                    state.chosen_branch = branch.name
                    break

            assert state.chosen_branch == expected_branch

            # Simulate the branch selection logic from agent_node
            for branch in router_agent._branches:
                if branch.name.lower() in response_content.lower():
                    state.chosen_branch = branch.name
                    break

            assert state.chosen_branch == expected_branch

            # Simulate the branch selection logic from agent_node
            for branch in router_agent._branches:
                if branch.name.lower() in response_content.lower():
                    state.chosen_branch = branch.name
                    break

            assert state.chosen_branch == expected_branch

            # Simulate the branch selection logic from agent_node
            for branch in router_agent._branches:
                if branch.name.lower() in response_content.lower():
                    state.chosen_branch = branch.name
                    break

            assert state.chosen_branch == expected_branch
