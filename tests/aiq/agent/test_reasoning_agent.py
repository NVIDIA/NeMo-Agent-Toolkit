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

from contextlib import AsyncExitStack
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

#
# The "build_reasoning_function" to be tested:
#
from aiq.agent.reasoning_agent.reasoning_agent import ReasoningFunctionConfig
from aiq.agent.reasoning_agent.reasoning_agent import build_reasoning_function
from aiq.builder.builder import Builder
from aiq.builder.function import Function
from aiq.builder.function import LambdaFunction
from aiq.builder.function_info import FunctionInfo
from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.function import FunctionBaseConfig

#############################
# EXAMPLE MOCK CLASSES
#############################


class DummyConfig(FunctionBaseConfig, name="dummy"):
    pass


#############################
# HELPER for mocking LLM streaming
#############################


def _fake_llm_stream(prompt: str, *args, **kwargs):
    """
    A stub for simulating an LLM streaming multiple tokens. This is the side_effect used in
    mock_llm.ainvoke_stream. It must directly return an async generator object, not a coroutine.
    """

    async def _gen():
        yield MagicMock(content="PretendLLMResponsePart1")
        yield MagicMock(content="PretendLLMResponsePart2")

    return _gen()


#############################
# Minimal stand-ins for the function "build_reasoning_function" augments
#############################


class MockAugmentedFunction(Function[str, str, str]):
    """
    A minimal stand-in for the function that 'build_reasoning_function' will augment.
    This example returns single output (no streaming).
    """

    def __init__(self, config: FunctionBaseConfig, description: str = "some tool description"):
        super().__init__(
            config=config,
            description=description,
            input_schema=None,  # Let base class auto-generate
            single_output_schema=None,
            streaming_output_schema=None,
            converters=[])
        # For test usage, let’s say we store a bool for streaming
        self._has_streaming = False

    @property
    def has_streaming_output(self) -> bool:
        return self._has_streaming

    async def _ainvoke(self, value: str) -> str:
        return f"AugmentedResult: {value}"

    async def _astream(self, value: str):
        # We won't exercise streaming in this example mock
        yield f"AugmentedStreamResult: {value}"


class MockStreamingAugmentedFunction(MockAugmentedFunction):
    """
    A minimal stand-in for a function that DOES have streaming output.
    """

    def __init__(self, config: FunctionBaseConfig, description: str = "some streaming tool desc"):
        super().__init__(config, description)
        self._has_streaming = True
        self._input_schema = AIQChatRequest

    async def _astream(self, value: AIQChatRequest):
        yield f"AugmentedStreamChunk1: {value}"
        yield f"AugmentedStreamChunk2: {value}"


@pytest_asyncio.fixture(name="fake_builder")
async def fake_builder_fixture() -> Builder:
    """
    A fixture that returns a mock `Builder` with get_llm and get_function replaced by MagicMocks.
    We'll use these to ensure we do not call real LLM or real functions.
    """
    builder = MagicMock(spec=Builder)

    async def _get_llm(name, wrapper_type):
        # Return a MagicMock that we can patch at the method level if needed
        mock_llm = MagicMock(name=f"LLM_{name}")
        # For streaming calls, we might patch mock_llm.ainvoke_stream
        # so it yields data
        # Here we rely on the side_effect to produce an async generator
        mock_llm.ainvoke_stream = MagicMock(side_effect=_fake_llm_stream)
        mock_llm.ainvoke = AsyncMock()
        return mock_llm

    builder.get_llm.side_effect = _get_llm

    def _get_function(name: str):
        # Return a mock augmented function
        # We can configure it to be streaming or not in each test
        # For now, default to a non-streaming MockAugmentedFunction
        return MockAugmentedFunction(DummyConfig())

    builder.get_function.side_effect = _get_function

    # get_function_dependencies is used just for referencing tool names, etc
    class FakeDeps:
        functions = ["SomeTool"]

    builder.get_function_dependencies.return_value = FakeDeps()

    return builder


#############################
# ACTUAL TESTS
#############################


@pytest.mark.asyncio
async def test_build_reasoning_function_happy_path(fake_builder):
    """
    Test that build_reasoning_function returns a FunctionInfo with a single-fn
    if the augmented function is non-streaming.
    """

    # Mock the augmented function to have a description
    def mock_get_function(name: str):
        # Return a non-streaming function with a description
        return MockAugmentedFunction(config=DummyConfig(), description="I am described!")

    fake_builder.get_function.side_effect = mock_get_function

    # Patch the LLM so it doesn't do real calls
    # We patch the place where the code calls llm.ainvoke_stream(...) inside build_reasoning_function
    mock_llm = MagicMock()
    mock_llm.ainvoke_stream = MagicMock(side_effect=_fake_llm_stream)
    fake_builder.get_llm.return_value = mock_llm

    # Setup config
    config = ReasoningFunctionConfig(
        llm_name="test_llm",
        augmented_fn="my_augmented_fn",  # we'll see get_function("my_augmented_fn") => mock
        verbose=True)

    # Now call the function we want to test
    reasoning_info = await AsyncExitStack().enter_async_context(build_reasoning_function(config, fake_builder))

    # Confirm it returns a FunctionInfo
    assert isinstance(reasoning_info, FunctionInfo)

    # Confirm we can create a real Function from it
    fn = LambdaFunction.from_info(config=config, info=reasoning_info)
    assert fn.has_single_output
    assert fn.has_streaming_output

    # Now let's test that calling the function triggers the expected LLM usage
    output = await fn.ainvoke("Test input")
    assert "AugmentedResult:" in output


@pytest.mark.asyncio
async def test_build_reasoning_function_streaming_with_chat_request(fake_builder):
    """
    If the augmented function has streaming output, the resulting FunctionInfo should have a stream_fn,
    and we test that streaming logic calls LLM in the background, then calls the augmented function in streaming mode.

    We also test that the connector can convert to the AIQChatRequest if the target requires it.
    """

    # Return a streaming augmented function
    def mock_get_function(name: str):
        return MockStreamingAugmentedFunction(config=DummyConfig(), description="I am streaming described!")

    fake_builder.get_function.side_effect = mock_get_function

    # Patch the LLM
    mock_llm = MagicMock()
    mock_llm.ainvoke_stream = MagicMock(side_effect=_fake_llm_stream)
    fake_builder.get_llm.return_value = mock_llm

    # Setup config
    config = ReasoningFunctionConfig(llm_name="fake_streaming_llm",
                                     augmented_fn="some_stream_augmented_fn",
                                     verbose=True)

    # Now call the function we want to test
    reasoning_info = await AsyncExitStack().enter_async_context(build_reasoning_function(config, fake_builder))
    fn = LambdaFunction.from_info(config=config, info=reasoning_info)
    assert fn.has_streaming_output
    assert fn.has_single_output  # Because the augmented function supports both method

    # calling astream
    chunks = []
    async for chunk in fn.astream("User wants to do something"):
        chunks.append(chunk)

    # We got the "AugmentedStreamChunk*" from the augmented function
    assert len(chunks) == 2
    assert all("AugmentedStreamChunk" in c for c in chunks)


@pytest.mark.asyncio
async def test_build_reasoning_function_no_augmented_function_description(fake_builder):
    """
    If the augmented function is missing a description, build_reasoning_function should raise ValueError.
    """

    def mock_get_function(name: str):
        # Return a function with an empty description
        return MockAugmentedFunction(config=DummyConfig(), description="")

    fake_builder.get_function.side_effect = mock_get_function

    config = ReasoningFunctionConfig(llm_name="test_llm", augmented_fn="fn_missing_desc", verbose=True)
    with pytest.raises(ValueError, match="does not have a description"):
        await AsyncExitStack().enter_async_context(build_reasoning_function(config, fake_builder))


@pytest.mark.asyncio
async def test_build_reasoning_function_augmented_fn_not_found(fake_builder):
    """
    If the builder cannot find the augmented function at all (None returned),
    we should see a KeyError or similar. We'll mock get_function to raise.
    """

    def mock_get_function(name: str):
        raise ValueError("No function with that name")

    fake_builder.get_function.side_effect = mock_get_function

    config = ReasoningFunctionConfig(llm_name="test_llm", augmented_fn="definitely_not_exists", verbose=True)
    with pytest.raises(ValueError, match="No function with that name"):
        await AsyncExitStack().enter_async_context(build_reasoning_function(config, fake_builder))


@pytest.mark.asyncio
async def test_build_reasoning_function_no_llm_found(fake_builder):
    """
    If builder.get_llm raises an error indicating no LLM found, we ensure the
    final build fails with that error.
    """

    def mock_get_llm(name, wrapper_type):
        raise RuntimeError("No LLM with that name found")

    fake_builder.get_llm.side_effect = mock_get_llm

    config = ReasoningFunctionConfig(llm_name="unknown_llm", augmented_fn="my_augmented_fn", verbose=True)
    # If no LLM is found, we can't proceed
    with pytest.raises(RuntimeError, match="No LLM with that name found"):
        await AsyncExitStack().enter_async_context(build_reasoning_function(config, fake_builder))


@pytest.mark.asyncio
async def test_build_reasoning_function_prompt_contains_tools(fake_builder):
    """
    We check that the final LLM call includes the tool names in the prompt, ensuring
    the code merges them in. We'll do so by capturing the call args to `ainvoke_stream`.
    """

    # We'll mock an augmented function with a valid description
    def mock_get_function(name: str):
        return MockAugmentedFunction(config=DummyConfig(), description="I am described!")

    fake_builder.get_function.side_effect = mock_get_function

    # The builder says we have 2 tools
    class FakeDeps:
        functions = ["ToolA", "ToolB"]

    fake_builder.get_function_dependencies.return_value = FakeDeps()

    mock_llm = MagicMock()

    # We'll capture the prompt used
    def side_effect_for_llm_stream(prompt: str, *args, **kwargs):
        # check that it has "ToolA" & "ToolB"
        assert "ToolA" in prompt
        assert "ToolB" in prompt
        return _fake_llm_stream(prompt, *args, **kwargs)

    mock_llm.ainvoke_stream.side_effect = side_effect_for_llm_stream
    fake_builder.get_llm.return_value = mock_llm

    config = ReasoningFunctionConfig(llm_name="test_llm", augmented_fn="my_augmented_fn", verbose=True)
    reasoning_info = await AsyncExitStack().enter_async_context(build_reasoning_function(config, fake_builder))

    # We'll create the function and invoke it so that the code does an llm.ainvoke_stream
    fn = LambdaFunction.from_info(config=config, info=reasoning_info)
    # This triggers the side_effect check
    output = await fn.ainvoke("Testing tool mention.")
    # If we got here, it means the prompt had "ToolA" and "ToolB" and didn't fail.
    assert "AugmentedResult:" in output


@pytest.mark.asyncio
async def test_build_reasoning_function_prompt_includes_input(fake_builder):
    """
    Ensure that the final prompt sent to the LLM includes the user input. We'll
    check the call argument to `ainvoke_stream`.
    """

    def mock_get_function(name: str):
        return MockAugmentedFunction(config=DummyConfig(), description="some tool desc")

    fake_builder.get_function.side_effect = mock_get_function

    # We'll check the argument in the side_effect
    def side_effect_llm_stream(prompt: str, *args, **kwargs):
        # The user input for the function invocation is "HelloUserInput"
        assert "HelloUserInput" in prompt
        return _fake_llm_stream(prompt, *args, **kwargs)

    mock_llm = MagicMock()
    mock_llm.ainvoke_stream.side_effect = side_effect_llm_stream
    fake_builder.get_llm.return_value = mock_llm

    config = ReasoningFunctionConfig(llm_name="test_llm_2", augmented_fn="augfn_check_prompt", verbose=True)
    reasoning_info = await AsyncExitStack().enter_async_context(build_reasoning_function(config, fake_builder))

    fn = LambdaFunction.from_info(config=config, info=reasoning_info)
    # The user input is "HelloUserInput"
    res = await fn.ainvoke("HelloUserInput")
    assert "AugmentedResult:" in res


@pytest.mark.asyncio
async def test_build_reasoning_function_handles_empty_tool_list(fake_builder):
    """
    If the function dependencies say there are no tools, we ensure it won't error
    but just produce a simpler LLM prompt. We'll verify the code doesn't break.
    """

    # We'll mock an augmented function with a valid description
    def mock_get_function(name: str):
        return MockAugmentedFunction(config=DummyConfig(), description="Description present")

    fake_builder.get_function.side_effect = mock_get_function

    # The builder says we have no tools
    class FakeDeps:
        functions = []

    fake_builder.get_function_dependencies.return_value = FakeDeps()

    mock_llm = MagicMock()
    mock_llm.ainvoke_stream.side_effect = _fake_llm_stream
    fake_builder.get_llm.return_value = mock_llm

    config = ReasoningFunctionConfig(llm_name="test_llm_empty_tools", augmented_fn="my_augmented_fn", verbose=True)
    # Just ensure no error is thrown
    reasoning_info = await AsyncExitStack().enter_async_context(build_reasoning_function(config, fake_builder))
    fn = LambdaFunction.from_info(config=config, info=reasoning_info)
    output = await fn.ainvoke("No tools scenario")
    # All good if we got a normal result
    assert "AugmentedResult:" in output
