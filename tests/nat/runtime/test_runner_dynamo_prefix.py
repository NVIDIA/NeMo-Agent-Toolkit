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
"""Tests for DynamoPrefixContext integration in the Runner class.

These tests verify that the Runner properly sets and clears the DynamoPrefixContext
for KV cache optimization when using Dynamo LLM backends.
"""

from collections.abc import AsyncGenerator

import pytest

from nat.builder.builder import Builder
from nat.builder.context import ContextState
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.llm.dynamo_llm import DynamoPrefixContext
from nat.observability.exporter_manager import ExporterManager
from nat.runtime.runner import Runner


class SingleOutputConfig(FunctionBaseConfig, name="single_output_dynamo_test"):
    pass


class StreamOutputConfig(FunctionBaseConfig, name="stream_output_dynamo_test"):
    pass


class CaptureConfig(FunctionBaseConfig, name="capture_dynamo_prefix"):
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


@pytest.fixture(autouse=True)
def clean_dynamo_context():
    """Ensure DynamoPrefixContext is clean before and after each test."""
    DynamoPrefixContext.clear()
    yield
    DynamoPrefixContext.clear()


async def test_runner_result_sets_dynamo_prefix_context():
    """Test that Runner.result() sets DynamoPrefixContext with unique prefix ID."""
    captured_prefix_ids = []

    @register_function(config_type=CaptureConfig)
    async def _register(config: CaptureConfig, b: Builder):

        async def _capture(message: str) -> str:
            # Capture the prefix ID during execution
            prefix_id = DynamoPrefixContext.get()
            captured_prefix_ids.append(prefix_id)
            return message

        yield _capture

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="capture_fn", config=CaptureConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            await runner.result()

        # Verify prefix ID was set during execution
        assert len(captured_prefix_ids) == 1
        assert captured_prefix_ids[0] is not None
        assert captured_prefix_ids[0].startswith("nat-workflow-")


async def test_runner_result_clears_dynamo_prefix_context_after_completion():
    """Test that Runner.result() clears DynamoPrefixContext after workflow completes."""
    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="test_fn", config=SingleOutputConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            await runner.result()

        # Verify prefix ID is cleared after execution
        assert DynamoPrefixContext.get() is None


async def test_runner_result_clears_dynamo_prefix_context_on_error():
    """Test that Runner.result() clears DynamoPrefixContext even when workflow fails."""

    class ErrorConfig(FunctionBaseConfig, name="error_dynamo_test"):
        pass

    @register_function(config_type=ErrorConfig)
    async def _register(config: ErrorConfig, b: Builder):

        async def _error(message: str) -> str:
            raise ValueError("Simulated error")

        yield _error

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="error_fn", config=ErrorConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            with pytest.raises(ValueError, match="Simulated error"):
                await runner.result()

        # Verify prefix ID is cleared even after error
        assert DynamoPrefixContext.get() is None


async def test_runner_result_different_invocations_get_unique_prefix_ids():
    """Test that different workflow invocations get unique prefix IDs."""
    captured_prefix_ids = []

    class CaptureConfig2(FunctionBaseConfig, name="capture_dynamo_prefix2"):
        pass

    @register_function(config_type=CaptureConfig2)
    async def _register(config: CaptureConfig2, b: Builder):

        async def _capture(message: str) -> str:
            prefix_id = DynamoPrefixContext.get()
            captured_prefix_ids.append(prefix_id)
            return message

        yield _capture

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="capture_fn", config=CaptureConfig2())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        # Run workflow multiple times
        for i in range(3):
            async with Runner(input_message=f"test{i}",
                              entry_fn=entry_fn,
                              context_state=context_state,
                              exporter_manager=exporter_manager) as runner:
                await runner.result()

        # Each invocation should have a unique prefix ID
        assert len(captured_prefix_ids) == 3
        assert len(set(captured_prefix_ids)) == 3  # All unique


async def test_runner_result_stream_sets_dynamo_prefix_context():
    """Test that Runner.result_stream() sets DynamoPrefixContext with unique prefix ID."""
    captured_prefix_ids = []

    class StreamCaptureConfig(FunctionBaseConfig, name="stream_capture_dynamo"):
        pass

    @register_function(config_type=StreamCaptureConfig)
    async def _register(config: StreamCaptureConfig, b: Builder):

        async def _capture_stream(message: str) -> AsyncGenerator[str]:
            prefix_id = DynamoPrefixContext.get()
            captured_prefix_ids.append(prefix_id)
            yield message

        yield _capture_stream

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="stream_capture_fn", config=StreamCaptureConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            async for _ in runner.result_stream():
                pass

        # Verify prefix ID was set during execution
        assert len(captured_prefix_ids) == 1
        assert captured_prefix_ids[0] is not None
        assert captured_prefix_ids[0].startswith("nat-workflow-")


async def test_runner_result_stream_clears_dynamo_prefix_context_after_completion():
    """Test that Runner.result_stream() clears DynamoPrefixContext after workflow completes."""
    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="stream_fn", config=StreamOutputConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            async for _ in runner.result_stream():
                pass

        # Verify prefix ID is cleared after execution
        assert DynamoPrefixContext.get() is None


async def test_runner_result_stream_clears_dynamo_prefix_context_on_error():
    """Test that Runner.result_stream() clears DynamoPrefixContext even when workflow fails."""

    class StreamErrorConfig(FunctionBaseConfig, name="stream_error_dynamo_test"):
        pass

    @register_function(config_type=StreamErrorConfig)
    async def _register(config: StreamErrorConfig, b: Builder):

        async def _error_stream(message: str) -> AsyncGenerator[str]:
            raise ValueError("Simulated stream error")
            yield message  # Make it a generator

        yield _error_stream

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="stream_error_fn", config=StreamErrorConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            with pytest.raises(ValueError, match="Simulated stream error"):
                async for _ in runner.result_stream():
                    pass

        # Verify prefix ID is cleared even after error
        assert DynamoPrefixContext.get() is None


async def test_runner_prefix_id_based_on_workflow_run_id():
    """Test that the prefix ID is based on the workflow_run_id."""
    captured_prefix_id = None

    class PrefixCheckConfig(FunctionBaseConfig, name="prefix_check_dynamo"):
        pass

    @register_function(config_type=PrefixCheckConfig)
    async def _register(config: PrefixCheckConfig, b: Builder):

        async def _check(message: str) -> str:
            nonlocal captured_prefix_id
            captured_prefix_id = DynamoPrefixContext.get()
            return message

        yield _check

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="prefix_check_fn", config=PrefixCheckConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        async with Runner(input_message="test",
                          entry_fn=entry_fn,
                          context_state=context_state,
                          exporter_manager=exporter_manager) as runner:
            await runner.result()

        # The prefix ID should be in the expected format
        assert captured_prefix_id is not None
        assert captured_prefix_id.startswith("nat-workflow-")
        # Verify the UUID portion is valid (36 chars with hyphens)
        uuid_part = captured_prefix_id[len("nat-workflow-"):]
        assert len(uuid_part) == 36


async def test_runner_uses_existing_workflow_run_id_for_prefix():
    """Test that Runner uses existing workflow_run_id when set externally."""
    captured_prefix_id = None
    preset_run_id = "preset-external-run-id-12345"

    class ExternalIdConfig(FunctionBaseConfig, name="external_id_dynamo"):
        pass

    @register_function(config_type=ExternalIdConfig)
    async def _register(config: ExternalIdConfig, b: Builder):

        async def _check(message: str) -> str:
            nonlocal captured_prefix_id
            captured_prefix_id = DynamoPrefixContext.get()
            return message

        yield _check

    async with WorkflowBuilder() as builder:
        entry_fn = await builder.add_function(name="external_id_fn", config=ExternalIdConfig())

        context_state = ContextState()
        exporter_manager = ExporterManager()

        # Pre-set the workflow_run_id
        token = context_state.workflow_run_id.set(preset_run_id)
        try:
            async with Runner(input_message="test",
                              entry_fn=entry_fn,
                              context_state=context_state,
                              exporter_manager=exporter_manager) as runner:
                await runner.result()
        finally:
            context_state.workflow_run_id.reset(token)

        # The prefix ID should use the pre-set workflow_run_id
        assert captured_prefix_id == f"nat-workflow-{preset_run_id}"
