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

from collections.abc import AsyncGenerator

import pytest
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.context import ContextState
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.observability.exporter_manager import ExporterManager
from nat.runtime.runner import Runner


class DummyConfig(FunctionBaseConfig, name="dummy_runner"):
    pass


class SingleOutputConfig(FunctionBaseConfig, name="single_output_runner"):
    pass


class StreamOutputConfig(FunctionBaseConfig, name="stream_output_runner"):
    pass


@pytest.fixture(scope="module", autouse=True)
async def _register_single_output_fn():

    @register_function(config_type=SingleOutputConfig)
    async def register(config: SingleOutputConfig, b: Builder):

        async def _inner(message: str) -> str:
            return message + "!"

        yield _inner


@pytest.fixture(scope="module", autouse=True)
async def _register_stream_output_fn():

    @register_function(config_type=StreamOutputConfig)
    async def register(config: StreamOutputConfig, b: Builder):

        async def _inner_stream(message: str) -> AsyncGenerator[str]:
            yield message + "!"

        yield _inner_stream


async def test_runner_result_successful_type_conversion():
    """Test that Runner.result() successfully converts output when compatible to_type is provided."""

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="test_function", config=SingleOutputConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            # Test successful conversion to compatible type
            result = await runner.result(to_type=str)
            assert result == "test!"

            # Test successful conversion without to_type
            async with Runner(input_message="test2",
                              entry_fn=entry_fn,
                              context_state=context_state,
                              exporter_manager=exporter_manager) as runner2:
                result2 = await runner2.result()
                assert result2 == "test2!"


async def test_runner_result_type_conversion_failure():
    """Test that Runner.result() raises ValueError when output cannot be converted to specified to_type."""

    class UnconvertibleOutput(BaseModel):
        value: str

    class IncompatibleType(BaseModel):
        different_field: int

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _inner(message: str) -> UnconvertibleOutput:
            return UnconvertibleOutput(value=message + "!")

        yield _inner

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="test_function", config=DummyConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            # Verify normal operation works
            result = await runner.result(to_type=UnconvertibleOutput)
            assert result.value == "test!"

        # Test that conversion to incompatible type raises ValueError
        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            with pytest.raises(ValueError, match="Cannot convert type .* to .* No match found"):
                await runner.result(to_type=IncompatibleType)


async def test_runner_result_primitive_type_conversion_failure():
    """Test that Runner.result() raises ValueError when primitive output cannot be converted to incompatible type."""

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="test_function", config=SingleOutputConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            # Verify normal operation works
            result = await runner.result(to_type=str)
            assert result == "test!"

        # Test that conversion to incompatible type raises ValueError
        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            with pytest.raises(ValueError, match="Cannot convert type .* to .* No match found"):
                await runner.result(to_type=dict)


async def test_runner_result_stream_successful_type_conversion():
    """Test that Runner.result_stream() successfully converts output when compatible to_type is provided."""

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="test_function", config=StreamOutputConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            # Test successful conversion to compatible type
            result = None
            async for output in runner.result_stream(to_type=str):
                result = output
            assert result == "test!"

        async with Runner(input_message="test2",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            # Test successful conversion without to_type
            result2 = None
            async for output in runner.result_stream():
                result2 = output
            assert result2 == "test2!"


async def test_runner_result_stream_type_conversion_failure():
    """Test that Runner.result_stream() raises ValueError when output cannot be converted to specified to_type."""

    class UnconvertibleOutput(BaseModel):
        value: str

    class IncompatibleType(BaseModel):
        different_field: int

    @register_function(config_type=DummyConfig)
    async def _register(config: DummyConfig, b: Builder):

        async def _stream_inner(message: str) -> AsyncGenerator[UnconvertibleOutput]:
            yield UnconvertibleOutput(value=message + "!")

        yield _stream_inner

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="test_function", config=DummyConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            # Verify normal operation works
            result = None
            async for output in runner.result_stream(to_type=UnconvertibleOutput):
                result = output
            assert result is not None and result.value == "test!"

        # Test that conversion to incompatible type raises ValueError during streaming
        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            with pytest.raises(ValueError, match="Cannot convert type .* to .* No match found"):
                async for output in runner.result_stream(to_type=IncompatibleType):
                    pass  # The exception should be raised during the first iteration


async def test_runner_result_stream_primitive_type_conversion_failure():
    """
    Test that Runner.result_stream() raises ValueError when primitive output cannot
    be converted to incompatible type.
    """

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="test_function", config=StreamOutputConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            # Verify normal operation works
            result = None
            async for output in runner.result_stream(to_type=str):
                result = output
            assert result == "test!"

        # Test that conversion to incompatible type raises ValueError during streaming
        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            with pytest.raises(ValueError, match="Cannot convert type .* to .* No match found"):
                async for output in runner.result_stream(to_type=dict):
                    pass  # The exception should be raised during the first iteration


async def test_runner_state_management():
    """Test that Runner properly manages state transitions during execution."""

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="test_function", config=SingleOutputConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        runner = Runner(input_message="test",
                        entry_fn=entry_fn,
                        context_state=context_state,
                        exporter_manager=exporter_manager)

        # Test that runner cannot be used outside of async context
        with pytest.raises(ValueError, match="Cannot run the workflow without entering the context"):
            await runner.result()

        # Test successful execution within context
        async with runner:
            result = await runner.result()
            assert result == "test!"
