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

import asyncio
import logging
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.runtime.session import SessionManager
from nat.utils.type_converter import GlobalTypeConverter


class MockExceptionWorkflowConfig(FunctionBaseConfig, name="mock_exception_agent"):
    """
    Mock workflow config that creates a function which throws exceptions for testing error handling.
    """
    exception_type: str = Field(default="ValueError", description="Type of exception to throw")
    exception_message: str = Field(default="Test exception", description="Exception message")
    should_handle_error: bool = Field(default=False, description="Whether to handle the error internally")
    description: str = Field(default="Mock Exception Workflow", description="The description of this function's use.")
    use_openai_api: bool = Field(default=False, description="Use OpenAI API for input/output types")


class MockExceptionWorkflowConfigNoHandling(FunctionBaseConfig, name="mock_exception_agent_no_handling"):
    """
    Mock workflow config without any error handling in _response_fn().
    """
    exception_type: str = Field(default="ValueError", description="Type of exception to throw")
    exception_message: str = Field(default="Test exception", description="Exception message")
    description: str = Field(default="Mock Exception Workflow No Handling", description="The description")
    use_openai_api: bool = Field(default=False, description="Use OpenAI API for input/output types")


@register_function(config_type=MockExceptionWorkflowConfig, framework_wrappers=[])
async def mock_exception_workflow(config: MockExceptionWorkflowConfig, builder: Builder):
    """Mock workflow that throws exceptions to test error handling."""

    async def _response_fn(input_message: ChatRequest) -> ChatResponse:
        try:
            # Create the specified exception type
            if config.exception_type == "RuntimeError":
                raise RuntimeError(config.exception_message)
            elif config.exception_type == "ValueError":
                raise ValueError(config.exception_message)
            elif config.exception_type == "TypeError":
                raise TypeError(config.exception_message)
            elif config.exception_type == "KeyError":
                raise KeyError(config.exception_message)
            else:
                raise ValueError(config.exception_message)
        except Exception as ex:
            if config.should_handle_error:
                # Test scenario where function handles its own errors
                logging.error("Mock workflow caught exception: %s", str(ex))
                return ChatResponse.from_string(f"Handled error: {str(ex)}")
            else:
                # Re-raise to test upstream error handling
                raise

    async def _str_response_fn(input_message: str) -> str:
        oai_input = GlobalTypeConverter.get().try_convert(input_message, to_type=ChatRequest)
        oai_output = await _response_fn(oai_input)
        return GlobalTypeConverter.get().try_convert(oai_output, to_type=str)

    if config.use_openai_api:
        yield FunctionInfo.from_fn(_response_fn, description=config.description)
    else:
        yield FunctionInfo.from_fn(_str_response_fn, description=config.description)


@register_function(config_type=MockExceptionWorkflowConfigNoHandling, framework_wrappers=[])
async def mock_exception_workflow_no_handling(config: MockExceptionWorkflowConfigNoHandling, builder: Builder):
    """Mock workflow that throws exceptions WITHOUT any error handling in _response_fn()."""

    async def _response_fn(input_message: ChatRequest) -> ChatResponse:
        # NO try/catch - this will test what happens when function doesn't handle errors
        if config.exception_type == "RuntimeError":
            raise RuntimeError(config.exception_message)
        elif config.exception_type == "ValueError":
            raise ValueError(config.exception_message)
        elif config.exception_type == "TypeError":
            raise TypeError(config.exception_message)
        elif config.exception_type == "KeyError":
            raise KeyError(config.exception_message)
        else:
            raise ValueError(config.exception_message)

    async def _str_response_fn(input_message: str) -> str:
        oai_input = GlobalTypeConverter.get().try_convert(input_message, to_type=ChatRequest)
        oai_output = await _response_fn(oai_input)
        return GlobalTypeConverter.get().try_convert(oai_output, to_type=str)

    if config.use_openai_api:
        yield FunctionInfo.from_fn(_response_fn, description=config.description)
    else:
        yield FunctionInfo.from_fn(_str_response_fn, description=config.description)


class TestErrorHandling:
    """Test suite for error handling at various levels of the NAT system."""

    @pytest.fixture
    def mock_config_with_error_handling(self):
        """Config for workflow that handles its own errors."""
        return MockExceptionWorkflowConfig(exception_type="ValueError",
                                           exception_message="Test ValueError from mock workflow",
                                           should_handle_error=True,
                                           use_openai_api=True)

    @pytest.fixture
    def mock_config_no_error_handling(self):
        """Config for workflow that does NOT handle errors in _response_fn()."""
        return MockExceptionWorkflowConfigNoHandling(exception_type="RuntimeError",
                                                     exception_message="Unhandled RuntimeError from mock workflow",
                                                     use_openai_api=True)

    @pytest.fixture
    def mock_config_string_api_no_handling(self):
        """Config for workflow with string API that throws exceptions."""
        return MockExceptionWorkflowConfigNoHandling(exception_type="ValueError",
                                                     exception_message="String API test exception",
                                                     use_openai_api=False)

    async def test_function_with_internal_error_handling(self, mock_config_with_error_handling, caplog):
        """Test that functions with internal error handling work correctly."""

        # Load the workflow using WorkflowBuilder
        config = Config(workflow=mock_config_with_error_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)

            # Create test input
            test_input = ChatRequest.from_string("test input")

            # Run the workflow and verify it handles the error internally
            async with session_manager.run(test_input) as runner:
                result = await runner.result()

            # Should receive the handled error message
            assert "Handled error: Test ValueError from mock workflow" in str(result)

            # Check that error was logged
            assert any("Mock workflow caught exception" in record.message for record in caplog.records)

    async def test_function_without_error_handling_logs_properly(self, mock_config_no_error_handling, caplog):
        """Test that unhandled exceptions in _response_fn() are logged with helpful information."""

        # Set up logging capture
        config = Config(workflow=mock_config_no_error_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)

            test_input = ChatRequest.from_string("test input")

            with caplog.at_level(logging.ERROR):
                # The error should propagate up from the function
                with pytest.raises(RuntimeError, match="Unhandled RuntimeError from mock workflow"):
                    async with session_manager.run(test_input) as runner:
                        await runner.result()

            # Check that Runner logged the error with helpful context
            runner_error_logs = [
                record for record in caplog.records
                if record.levelno == logging.ERROR and "Error running workflow" in record.message
            ]
            assert len(runner_error_logs) > 0

            # The log should contain information about the error
            error_log = runner_error_logs[0]
            assert "Error running workflow" in error_log.message
            # Note: Runner uses logger.error() not logger.exception(), so exc_info is None
            # But the error message contains the exception details

    async def test_error_propagation_to_session_manager(self, mock_config_no_error_handling):
        """Test that errors propagate correctly to SessionManager and can be caught."""

        config = Config(workflow=mock_config_no_error_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)
            test_input = ChatRequest.from_string("test input")

            # Test that we can catch the error at the session manager level
            error_caught = False
            try:
                async with session_manager.run(test_input) as runner:
                    await runner.result()
            except RuntimeError as e:
                error_caught = True
                assert "Unhandled RuntimeError from mock workflow" in str(e)

            assert error_caught, "Exception should have propagated to session manager level"

    async def test_error_propagation_with_runner_try_catch(self, mock_config_no_error_handling, caplog):
        """Test that errors can be caught around the runner with try/catch."""

        config = Config(workflow=mock_config_no_error_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)
            test_input = ChatRequest.from_string("test input")

            # Test catching error around runner operations
            error_details = None
            try:
                async with session_manager.run(test_input) as runner:
                    result = await runner.result()
            except Exception as e:
                error_details = {
                    'type': type(e).__name__, 'message': str(e), 'function_scope': 'mock_exception_agent_no_handling'
                }

            # Verify we caught the error with proper details
            assert error_details is not None
            assert error_details['type'] == 'RuntimeError'
            assert "Unhandled RuntimeError from mock workflow" in error_details['message']
            assert error_details['function_scope'] == 'mock_exception_agent_no_handling'

    async def test_string_api_error_handling(self, mock_config_string_api_no_handling):
        """Test error handling with string API (non-OpenAI API)."""

        config = Config(workflow=mock_config_string_api_no_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)
            test_input = "test string input"

            # Error should propagate through string API as well
            with pytest.raises(ValueError, match="String API test exception"):
                async with session_manager.run(test_input) as runner:
                    await runner.result()

    async def test_response_helpers_error_handling(self, mock_config_no_error_handling):
        """Test error handling in the response helpers that SessionManager uses."""
        from nat.front_ends.fastapi.response_helpers import generate_single_response

        config = Config(workflow=mock_config_no_error_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)

            test_input = ChatRequest.from_string("test input")

            # Test that the response helpers also propagate errors correctly
            with pytest.raises(RuntimeError, match="Unhandled RuntimeError from mock workflow"):
                await generate_single_response(test_input, session_manager)

    async def test_multiple_exception_types(self, caplog):
        """Test handling of different exception types."""

        exception_scenarios = [
            ("ValueError", "Test ValueError", False),
            ("RuntimeError", "Test RuntimeError", False),
            ("TypeError", "Test TypeError", False),
            ("KeyError", "Test KeyError", False),
        ]

        for exc_type, exc_msg, should_handle in exception_scenarios:
            mock_config = MockExceptionWorkflowConfigNoHandling(exception_type=exc_type,
                                                                exception_message=exc_msg,
                                                                use_openai_api=True)

            config = Config(workflow=mock_config)
            async with WorkflowBuilder.from_config(config=config) as workflow_builder:
                workflow = workflow_builder.build()
                session_manager = SessionManager(workflow)
                test_input = ChatRequest.from_string("test input")

                # Each exception type should propagate correctly
                if exc_type == "RuntimeError":
                    exception_class = RuntimeError
                elif exc_type == "ValueError":
                    exception_class = ValueError
                elif exc_type == "TypeError":
                    exception_class = TypeError
                elif exc_type == "KeyError":
                    exception_class = KeyError
                else:
                    exception_class = ValueError

                with pytest.raises(exception_class, match=exc_msg):
                    async with session_manager.run(test_input) as runner:
                        await runner.result()

    async def test_logging_contains_function_context(self, mock_config_no_error_handling, caplog):
        """Test that error logs contain helpful context about function name and scope."""

        config = Config(workflow=mock_config_no_error_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)
            test_input = ChatRequest.from_string("test input")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(RuntimeError):
                    async with session_manager.run(test_input) as runner:
                        await runner.result()

            # Verify logging contains helpful context
            error_logs = [record for record in caplog.records if record.levelno == logging.ERROR]
            assert len(error_logs) > 0

            # Check for workflow/runner error logging
            runner_logs = [log for log in error_logs if "Error running workflow" in log.message]
            assert len(runner_logs) > 0

            # The log record contains the error message (runner uses logger.error, not logger.exception)
            runner_log = runner_logs[0]
            assert "RuntimeError" in runner_log.getMessage() or "Unhandled RuntimeError" in runner_log.getMessage()
            # The function-level logging does contain more detailed error info

    async def test_concurrent_error_handling(self, mock_config_no_error_handling):
        """Test that error handling works correctly with concurrent workflow runs."""

        config = Config(workflow=mock_config_no_error_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow, max_concurrency=3)

            async def run_single_request():
                test_input = ChatRequest.from_string("concurrent test")
                with pytest.raises(RuntimeError, match="Unhandled RuntimeError from mock workflow"):
                    async with session_manager.run(test_input) as runner:
                        await runner.result()

            # Run multiple concurrent requests - all should fail with the same error
            tasks = [run_single_request() for _ in range(3)]
            await asyncio.gather(*tasks)

    async def test_error_with_different_input_types(self, mock_config_string_api_no_handling):
        """Test error handling with different input types (string vs ChatRequest)."""

        config = Config(workflow=mock_config_string_api_no_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)

            # Test with string input (uses string API)
            test_string_input = "test string"
            with pytest.raises(ValueError, match="String API test exception"):
                async with session_manager.run(test_string_input) as runner:
                    await runner.result()

            # Test with ChatRequest input (should also work)
            test_chat_input = ChatRequest.from_string("test chat")
            with pytest.raises(ValueError, match="String API test exception"):
                async with session_manager.run(test_chat_input) as runner:
                    await runner.result()

    async def test_error_message_preservation(self, caplog):
        """Test that original error messages are preserved through the error handling chain."""

        custom_error_message = "Very specific error message that should be preserved"
        mock_config = MockExceptionWorkflowConfigNoHandling(exception_type="ValueError",
                                                            exception_message=custom_error_message,
                                                            use_openai_api=True)

        config = Config(workflow=mock_config)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)
            test_input = ChatRequest.from_string("test")

            # The exact error message should be preserved
            with pytest.raises(ValueError) as exc_info:
                async with session_manager.run(test_input) as runner:
                    await runner.result()

            assert custom_error_message in str(exc_info.value)

    def test_workflow_config_validation(self):
        """Test that our mock workflow configs are properly validated."""

        # Valid config should work
        valid_config = MockExceptionWorkflowConfig(exception_type="ValueError", exception_message="test message")
        assert valid_config.exception_type == "ValueError"
        assert valid_config.exception_message == "test message"
        assert not valid_config.should_handle_error  # Default

        # Config without error handling should also work
        no_handle_config = MockExceptionWorkflowConfigNoHandling(exception_type="RuntimeError",
                                                                 exception_message="runtime error test")
        assert no_handle_config.exception_type == "RuntimeError"
        assert no_handle_config.exception_message == "runtime error test"
