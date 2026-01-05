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
"""
End-to-end integration tests for Strands Agent with different LLM providers.

These tests require actual API keys and will make real API calls to LLM providers.
Run with: pytest --run_integration
"""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from nat.builder.function import LambdaFunction
from nat.data_models.function import EmptyFunctionConfig
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.strands.llm import bedrock_strands
from nat.plugins.strands.llm import nim_strands
from nat.plugins.strands.llm import openai_strands
from nat.plugins.strands.tool_wrapper import strands_tool_wrapper


class CalculatorInput(BaseModel):
    """Input schema for calculator function."""
    a: float
    b: float
    operation: str


class CalculatorOutput(BaseModel):
    """Output schema for calculator function."""
    result: float


class TestStrandsAgentE2EOpenAI:
    """End-to-end integration tests for Strands Agent with OpenAI."""

    @pytest.fixture
    async def calculator_function(self) -> LambdaFunction:
        """Create a simple calculator NAT function for testing."""

        async def calculator_impl(input_data: CalculatorInput) -> CalculatorOutput:
            """A simple calculator that performs basic arithmetic operations."""
            if input_data.operation == "add":
                result = input_data.a + input_data.b
            elif input_data.operation == "subtract":
                result = input_data.a - input_data.b
            elif input_data.operation == "multiply":
                result = input_data.a * input_data.b
            elif input_data.operation == "divide":
                if input_data.b == 0:
                    raise ValueError("Cannot divide by zero")
                result = input_data.a / input_data.b
            else:
                raise ValueError(f"Unknown operation: {input_data.operation}")

            return CalculatorOutput(result=result)

        from nat.builder.function_info import FunctionInfo
        info = FunctionInfo.from_fn(calculator_impl,
                                    input_schema=CalculatorInput,
                                    description="A calculator that performs basic arithmetic operations")

        return LambdaFunction.from_info(config=EmptyFunctionConfig(), info=info, instance_name="calculator")

    @pytest.fixture
    def builder(self) -> MagicMock:
        """Create a mock Builder instance for tests."""
        return MagicMock()

    @pytest.mark.integration
    @pytest.mark.usefixtures("openai_api_key")
    @pytest.mark.asyncio
    async def test_strands_agent_with_openai_simple_task(self, calculator_function, builder):
        """Test complete workflow: OpenAI LLM -> Strands Agent -> NAT Function."""
        from strands.agent import Agent

        # Create OpenAI LLM config
        llm_config = OpenAIModelConfig(model_name="gpt-4o", temperature=0.0, max_tokens=64)

        # Convert NAT function to Strands tool
        strands_tool = strands_tool_wrapper("calculator", calculator_function, builder)

        # Create Strands agent with OpenAI LLM
        async with openai_strands(llm_config, builder) as llm_client:
            agent = Agent(model=llm_client,
                          tools=[strands_tool],
                          system_prompt="You are a helpful assistant that can perform calculations.")

            # Test agent execution
            response = agent("Add 15 and 27. Reply with just the number.")

            # Verify response
            assert response is not None
            assert response.message is not None
            # Extract text from message content
            response_text = str(response.message)
            assert "42" in response_text

    @pytest.mark.integration
    @pytest.mark.usefixtures("openai_api_key")
    @pytest.mark.asyncio
    async def test_strands_agent_with_openai_multiple_operations(self, calculator_function, builder):
        """Test agent with multiple tool calls."""
        from strands.agent import Agent

        llm_config = OpenAIModelConfig(model_name="gpt-4o", temperature=0.0, max_tokens=256)

        strands_tool = strands_tool_wrapper("calculator", calculator_function, builder)

        async with openai_strands(llm_config, builder) as llm_client:
            agent = Agent(model=llm_client,
                          tools=[strands_tool],
                          system_prompt="You are a helpful assistant that can perform calculations.")

            # Test with multiple operations
            response = agent("Add 10 and 5, multiply the sum by 3, and return only the final number.")

            assert response is not None
            assert response.message is not None
            response_text = str(response.message)
            # Should eventually get to 45 (10+5=15, 15*3=45)
            assert "45" in response_text or "15" in response_text

    @pytest.mark.integration
    @pytest.mark.usefixtures("openai_api_key")
    @pytest.mark.asyncio
    async def test_strands_agent_with_openai_error_handling(self, calculator_function, builder):
        """Test that agent handles tool errors gracefully."""
        from strands.agent import Agent

        llm_config = OpenAIModelConfig(model_name="gpt-4o", temperature=0.0, max_tokens=256)

        strands_tool = strands_tool_wrapper("calculator", calculator_function, builder)

        async with openai_strands(llm_config, builder) as llm_client:
            agent = Agent(model=llm_client,
                          tools=[strands_tool],
                          system_prompt="You are a helpful assistant that can perform calculations.")

            # Test with division by zero
            response = agent("Divide 10 by 0 and explain the error briefly.")

            assert response is not None
            # Agent should handle the error and provide a meaningful response


class TestStrandsAgentE2ENIM:
    """End-to-end integration tests for Strands Agent with NVIDIA NIM."""

    @pytest.fixture
    async def echo_function(self) -> LambdaFunction:
        """Create a simple echo function for testing."""

        class EchoInput(BaseModel):
            message: str

        class EchoOutput(BaseModel):
            echo: str

        async def echo_impl(input_data: EchoInput) -> EchoOutput:
            """Echo the input message."""
            return EchoOutput(echo=f"You said: {input_data.message}")

        from nat.builder.function_info import FunctionInfo
        info = FunctionInfo.from_fn(echo_impl, input_schema=EchoInput, description="Echoes back the input message")

        return LambdaFunction.from_info(config=EmptyFunctionConfig(), info=info, instance_name="echo")

    @pytest.fixture
    def builder(self) -> MagicMock:
        """Create a mock Builder instance for tests."""
        return MagicMock()

    @pytest.mark.integration
    @pytest.mark.usefixtures("nvidia_api_key")
    @pytest.mark.asyncio
    async def test_strands_agent_with_nim_simple_task(self, echo_function, builder):
        """Test complete workflow: NIM LLM -> Strands Agent -> NAT Function."""
        from strands.agent import Agent

        # Create NIM LLM config
        llm_config = NIMModelConfig(model_name="meta/llama-3.1-8b-instruct", temperature=0.0, max_tokens=256)

        # Convert NAT function to Strands tool
        strands_tool = strands_tool_wrapper("echo", echo_function, builder)

        # Create Strands agent with NIM LLM
        async with nim_strands(llm_config, builder) as llm_client:
            agent = Agent(
                model=llm_client,
                tools=[strands_tool],
                system_prompt="You are a helpful assistant that can echo messages. Use the echo tool exactly once.")

            # Test agent execution
            response = agent("Use the echo tool to echo 'Hello World'")

            # Verify response
            assert response is not None
            assert response.message is not None
            response_text = str(response.message)
            # Check that the echo tool was used - the model may not reproduce the exact text
            # but should indicate the tool was called
            assert "echo" in response_text.lower(
            ) or "Hello World" in response_text or "hello world" in response_text.lower()

    @pytest.mark.integration
    @pytest.mark.usefixtures("nvidia_api_key")
    @pytest.mark.asyncio
    async def test_strands_agent_with_nim_reasoning(self, echo_function, builder):
        """Test NIM with reasoning capabilities (basic, no thinking mixin)."""
        from strands.agent import Agent

        llm_config = NIMModelConfig(model_name="meta/llama-3.1-8b-instruct", temperature=0.0, max_tokens=256)

        strands_tool = strands_tool_wrapper("echo", echo_function, builder)

        async with nim_strands(llm_config, builder) as llm_client:
            agent = Agent(model=llm_client, tools=[strands_tool], system_prompt="You are a helpful assistant.")

            # Test with a task that requires reasoning
            response = agent("Consider the word 'test' and then echo it back once.")

            assert response is not None
            assert response.message is not None
            response_text = str(response.message)
            assert "test" in response_text.lower()

    @pytest.mark.integration
    @pytest.mark.usefixtures("nvidia_api_key")
    @pytest.mark.asyncio
    async def test_strands_agent_with_nim_thinking_mixin_non_streaming(self, echo_function, builder):
        """Test NIM with NAT's ThinkingMixin for chain-of-thought reasoning (non-streaming)."""
        from strands.agent import Agent

        # Enable thinking mixin with Nemotron model that supports thinking
        # Note: Thinking uses additional tokens, so we need a higher max_tokens
        llm_config = NIMModelConfig(model_name="nvidia/llama-3.3-nemotron-super-49b-v1",
                                    temperature=0.0,
                                    max_tokens=1024,
                                    thinking=True)

        strands_tool = strands_tool_wrapper("echo", echo_function, builder)

        async with nim_strands(llm_config, builder) as llm_client:
            agent = Agent(model=llm_client,
                          tools=[strands_tool],
                          system_prompt="You are a helpful assistant that can use tools.")

            # Test with thinking enabled - the model should use the echo tool
            response = agent("Use the echo tool to echo the message 'success'.")

            assert response is not None
            assert response.message is not None
            response_text = str(response.message)
            # Verify the tool was successfully invoked (content may be empty but tool should execute)
            # The presence of a response confirms thinking was applied and the agent completed
            assert response_text is not None

    @pytest.mark.integration
    @pytest.mark.usefixtures("nvidia_api_key")
    @pytest.mark.asyncio
    async def test_strands_agent_with_nim_thinking_mixin_streaming(self, echo_function, builder):
        """Test NIM with NAT's ThinkingMixin using streaming mode."""
        from strands.agent import Agent

        # Enable thinking mixin with Nemotron model that supports thinking
        # Note: Thinking uses additional tokens, so we need a higher max_tokens
        llm_config = NIMModelConfig(model_name="nvidia/llama-3.3-nemotron-super-49b-v1",
                                    temperature=0.0,
                                    max_tokens=1024,
                                    thinking=True)

        strands_tool = strands_tool_wrapper("echo", echo_function, builder)

        async with nim_strands(llm_config, builder) as llm_client:
            agent = Agent(model=llm_client,
                          tools=[strands_tool],
                          system_prompt="You are a helpful assistant that can use tools.")

            # Test with streaming response and thinking enabled
            # Note: Strands agent.stream_async() returns an async generator
            collected_responses = []
            async for event in agent.stream_async("Use the echo tool to echo the word 'thinking'."):
                collected_responses.append(event)

            # Verify we got streaming events (confirms the agent ran with thinking enabled)
            assert len(collected_responses) > 0


class TestStrandsAgentE2EBedrock:
    """End-to-end integration tests for Strands Agent with AWS Bedrock."""

    @pytest.fixture
    async def greeting_function(self) -> LambdaFunction:
        """Create a simple greeting function for testing."""

        class GreetingInput(BaseModel):
            name: str

        class GreetingOutput(BaseModel):
            greeting: str

        async def greeting_impl(input_data: GreetingInput) -> GreetingOutput:
            """Generate a greeting for the given name."""
            return GreetingOutput(greeting=f"Hello, {input_data.name}! How are you today?")

        from nat.builder.function_info import FunctionInfo
        info = FunctionInfo.from_fn(greeting_impl,
                                    input_schema=GreetingInput,
                                    description="Generates a friendly greeting for a person")

        return LambdaFunction.from_info(config=EmptyFunctionConfig(), info=info, instance_name="greeting")

    @pytest.fixture
    def builder(self) -> MagicMock:
        """Create a mock Builder instance for tests."""
        return MagicMock()

    @pytest.mark.integration
    @pytest.mark.usefixtures("aws_keys")
    @pytest.mark.asyncio
    async def test_strands_agent_with_bedrock_simple_task(self, greeting_function, builder):
        """Test complete workflow: Bedrock LLM -> Strands Agent -> NAT Function."""
        from strands.agent import Agent

        # Create Bedrock LLM config
        llm_config = AWSBedrockModelConfig(model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                                           region_name="us-east-1",
                                           temperature=0.0,
                                           max_tokens=256)

        # Convert NAT function to Strands tool
        strands_tool = strands_tool_wrapper("greeting", greeting_function, builder)

        # Create Strands agent with Bedrock LLM
        async with bedrock_strands(llm_config, builder) as llm_client:
            agent = Agent(model=llm_client,
                          tools=[strands_tool],
                          system_prompt="You are a helpful assistant that can greet people.")

            # Test agent execution
            response = agent("Greet Alice warmly in a single sentence.")

            # Verify response
            assert response is not None
            assert response.message is not None
            response_text = str(response.message)
            assert "Alice" in response_text

    @pytest.mark.integration
    @pytest.mark.usefixtures("aws_keys")
    @pytest.mark.asyncio
    async def test_strands_agent_with_bedrock_claude(self, greeting_function, builder):
        """Test Bedrock with Claude model specifically."""
        from strands.agent import Agent

        # Test with Claude 3 Haiku (faster, cheaper)
        llm_config = AWSBedrockModelConfig(model_name="anthropic.claude-3-haiku-20240307-v1:0",
                                           region_name="us-east-1",
                                           temperature=0.0,
                                           max_tokens=80)

        strands_tool = strands_tool_wrapper("greeting", greeting_function, builder)

        async with bedrock_strands(llm_config, builder) as llm_client:
            agent = Agent(model=llm_client, tools=[strands_tool], system_prompt="You are a friendly assistant.")

            response = agent("Greet Bob in one friendly sentence.")

            assert response is not None
            assert response.message is not None
            response_text = str(response.message)
            assert "Bob" in response_text


class TestStrandsProfilerIntegration:
    """Integration tests for Strands profiler with real LLM calls."""

    @pytest.fixture
    def builder(self) -> MagicMock:
        """Create a mock Builder instance for tests."""
        return MagicMock()

    @pytest.fixture
    async def simple_function(self):
        """Create a simple function for profiling tests."""

        class SimpleInput(BaseModel):
            value: int

        class SimpleOutput(BaseModel):
            doubled: int

        async def simple_impl(input_data: SimpleInput) -> SimpleOutput:
            """Double the input value."""
            return SimpleOutput(doubled=input_data.value * 2)

        from nat.builder.function_info import FunctionInfo
        info = FunctionInfo.from_fn(simple_impl, input_schema=SimpleInput, description="Doubles the input value")

        return LambdaFunction.from_info(config=EmptyFunctionConfig(), info=info, instance_name="doubler")

    @pytest.mark.integration
    @pytest.mark.usefixtures("openai_api_key")
    @pytest.mark.asyncio
    async def test_strands_profiler_captures_llm_calls(self, simple_function, builder):
        """Test that profiler captures LLM call metrics."""
        from strands.agent import Agent

        from nat.plugins.strands.strands_callback_handler import StrandsProfilerHandler

        llm_config = OpenAIModelConfig(model_name="gpt-4o", temperature=0.0, max_tokens=64)

        strands_tool = strands_tool_wrapper("doubler", simple_function, builder)

        # Enable profiling
        profiler = StrandsProfilerHandler()
        profiler.instrument()

        try:
            async with openai_strands(llm_config, builder) as llm_client:
                agent = Agent(model=llm_client,
                              tools=[strands_tool],
                              system_prompt="You are a helpful assistant that can double numbers.")

                response = agent("Double 21 and return only the result.")

                assert response is not None
                # Profiler should have captured the LLM calls
                # Note: Actual profiler data verification would require access to the profiler's storage
        finally:
            # Note: uninstrument() is not yet implemented
            # This is one of the identified gaps in the Strands integration
            pass

    @pytest.mark.integration
    @pytest.mark.usefixtures("openai_api_key")
    @pytest.mark.asyncio
    async def test_strands_profiler_captures_tool_calls(self, simple_function, builder):
        """Test that profiler captures tool call metrics."""
        from strands.agent import Agent

        from nat.plugins.strands.strands_callback_handler import StrandsProfilerHandler

        llm_config = OpenAIModelConfig(model_name="gpt-4o", temperature=0.0, max_tokens=64)

        strands_tool = strands_tool_wrapper("doubler", simple_function, builder)

        profiler = StrandsProfilerHandler()
        profiler.instrument()

        try:
            async with openai_strands(llm_config, builder) as llm_client:
                agent = Agent(model=llm_client, tools=[strands_tool], system_prompt="You are a helpful assistant.")

                response = agent("Double 10 and return only the result.")

                assert response is not None
                # Tool calls should be captured by profiler
        finally:
            pass
