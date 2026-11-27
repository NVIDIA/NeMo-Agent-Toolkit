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

import contextvars
from collections.abc import AsyncGenerator

import pytest
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.interactive import HumanPromptText
from nat.data_models.interactive import HumanResponseText
from nat.observability.exporter_manager import ExporterManager
from nat.runtime.runner import Runner
from nat.runtime.session import SessionManager


class DummyConfig(FunctionBaseConfig, name="dummy_runner"):
    pass


class SingleOutputConfig(FunctionBaseConfig, name="single_output_runner"):
    pass


class StreamOutputConfig(FunctionBaseConfig, name="stream_output_runner"):
    pass


class RequestContextConfig(FunctionBaseConfig, name="request_context_runner"):
    pass


class PerUserRequestContextConfig(FunctionBaseConfig, name="per_user_request_context_runner"):
    pass


class RequestContextInput(BaseModel):
    message: str


class RequestContextOutput(BaseModel):
    conversation_id: str | None
    user_message_id: str | None
    human_response: str
    build_var: str | None


# Stands in for a framework context variable set while the workflow is built.
_build_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("test_runner_build_var", default=None)


async def _read_request_context(message: RequestContextInput) -> RequestContextOutput:
    context = Context.get()
    response = await context.user_interaction_manager.prompt_user_input(HumanPromptText(text="continue?"))
    return RequestContextOutput(conversation_id=context.conversation_id,
                                user_message_id=context.user_message_id,
                                human_response=response.content.text,
                                build_var=_build_var.get())


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


@pytest.fixture(scope="module", autouse=True)
async def _register_request_context_fns():

    @register_function(config_type=RequestContextConfig)
    async def register(config: RequestContextConfig, b: Builder):
        yield _read_request_context

    @register_per_user_function(config_type=PerUserRequestContextConfig,
                                input_type=RequestContextInput,
                                single_output_type=RequestContextOutput)
    async def register_per_user(config: PerUserRequestContextConfig, b: Builder):
        yield _read_request_context


def _answer_with(text: str):

    async def _callback(prompt):
        return HumanResponseText(text=text)

    return _callback


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


async def test_runner_aexit_raises_on_incomplete_clean_exit():
    """Test that Runner raises ValueError when exited cleanly without completing the workflow."""

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="test_function", config=SingleOutputConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        with pytest.raises(ValueError, match="Cannot exit the context without completing the workflow"):
            async with Runner(input_message="test",
                              entry_fn=entry_fn,
                              context_state=context_state,
                              exporter_manager=exporter_manager):
                pass  # exit without calling result()


async def test_runner_aexit_allows_cancelled_error_to_propagate():
    """Test that Runner does not mask CancelledError with a ValueError on exit."""
    import asyncio

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="test_function", config=SingleOutputConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        with pytest.raises(asyncio.CancelledError):
            async with Runner(input_message="test",
                              entry_fn=entry_fn,
                              context_state=context_state,
                              exporter_manager=exporter_manager):
                raise asyncio.CancelledError()


async def test_runner_workflow_replacement_handoff():
    """Test the workflow-replacement handoff path tied to the message_handler regression.

    Cancelling the first in-flight Runner via task.cancel() must not mask CancelledError
    with a ValueError (runner.py fix), and the immediately-following second Runner must
    run to completion on the same context_state/exporter_manager.
    """
    import asyncio

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="test_function", config=SingleOutputConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        # Simulate message_handler cancelling an in-flight task when a new message arrives.
        async def _first_workflow():
            async with Runner(input_message="first",
                              entry_fn=entry_fn,
                              context_state=context_state,
                              exporter_manager=exporter_manager):
                await asyncio.sleep(0)  # yield so external cancel can be delivered

        first_task = asyncio.create_task(_first_workflow())
        await asyncio.sleep(0)  # let the task enter the Runner context
        first_task.cancel()

        # CancelledError must propagate cleanly — not be masked by ValueError.
        with pytest.raises(asyncio.CancelledError):
            await first_task

        # Handoff: second Runner starts immediately on the same context and runs to completion.
        async with Runner(input_message="second",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner2:
            result = await runner2.result()
            assert result == "second!"


async def test_runner_does_not_restore_request_context_saved_at_build():
    """The workflow's saved build context restores framework variables but not the request's ContextState."""

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="test_function", config=RequestContextConfig())
        context_state = ContextState()

        def _build_context() -> contextvars.Context:
            _build_var.set("set at build")
            context_state.conversation_id.set("build-conversation")
            context_state.user_message_id.set("build-message")
            context_state.user_input_callback.set(_answer_with("build callback"))
            return contextvars.copy_context()

        saved_context = contextvars.copy_context().run(_build_context)

        conversation_token = context_state.conversation_id.set("request-conversation")
        message_token = context_state.user_message_id.set("request-message")
        callback_token = context_state.user_input_callback.set(_answer_with("request callback"))
        try:
            async with Runner(input_message=RequestContextInput(message="hi"),
                              entry_fn=entry_fn,
                              context_state=context_state,
                              exporter_manager=ExporterManager(),
                              saved_context=saved_context) as runner:
                result = await runner.result(to_type=RequestContextOutput)
        finally:
            context_state.user_input_callback.reset(callback_token)
            context_state.user_message_id.reset(message_token)
            context_state.conversation_id.reset(conversation_token)

        assert result == RequestContextOutput(conversation_id="request-conversation",
                                              user_message_id="request-message",
                                              human_response="request callback",
                                              build_var="set at build")


async def test_per_user_workflow_uses_the_context_of_each_request():
    """A per-user workflow is built inside its user's first request; later requests must not see that request."""

    config = Config(workflow=PerUserRequestContextConfig())
    async with WorkflowBuilder.from_config(config) as builder:
        session_manager = await SessionManager.create(config=config, shared_builder=builder)
        try:
            for i in range(1, 4):
                async with session_manager.session(user_id="alice",
                                                   conversation_id=f"conversation-{i}",
                                                   user_message_id=f"message-{i}",
                                                   user_input_callback=_answer_with(f"connection-{i}")) as session:
                    async with session.run(RequestContextInput(message="hi")) as runner:
                        result = await runner.result(to_type=RequestContextOutput)

                assert result == RequestContextOutput(conversation_id=f"conversation-{i}",
                                                      user_message_id=f"message-{i}",
                                                      human_response=f"connection-{i}",
                                                      build_var=None)
        finally:
            await session_manager.shutdown()
