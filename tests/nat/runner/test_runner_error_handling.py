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
from pydantic import Field

from nat.builder.builder import Builder
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
    handle_error: bool = Field(default=False, description="Whether to handle the error internally")
    description: str = Field(default="Mock Exception Workflow", description="The description of this function's use.")
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
            if config.handle_error:
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


class TestErrorHandling:
    """Test suite for error handling at various levels of the NAT system."""

    @pytest.fixture
    def mock_config_with_error_handling(self):
        """Config for workflow that handles its own errors."""
        return MockExceptionWorkflowConfig(exception_type="ValueError",
                                           exception_message="Test ValueError from mock workflow",
                                           handle_error=True,
                                           use_openai_api=True)

    @pytest.fixture
    def mock_config_no_error_handling(self):
        """Config for workflow that does NOT handle errors in _response_fn()."""
        return MockExceptionWorkflowConfig(exception_type="RuntimeError",
                                           exception_message="Unhandled RuntimeError from mock workflow",
                                           handle_error=False,
                                           use_openai_api=True)

    @pytest.fixture
    def mock_config_string_api_no_handling(self):
        """Config for workflow with string API that throws exceptions."""
        return MockExceptionWorkflowConfig(exception_type="ValueError",
                                           exception_message="String API test exception",
                                           handle_error=False,
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
        """Test that unhandled exceptions in _response_fn() are logged with comprehensive helpful information."""

        # Set up logging capture
        config = Config(workflow=mock_config_no_error_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)

            test_input = ChatRequest.from_string("test input for error logging")

            with caplog.at_level(logging.ERROR):
                # The error should propagate up from the function
                with pytest.raises(RuntimeError, match="Unhandled RuntimeError from mock workflow"):
                    async with session_manager.run(test_input) as runner:
                        await runner.result()

            # Comprehensive log analysis
            self._verify_comprehensive_error_logging(caplog.records, test_input)

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

    async def test_multiple_exception_types_with_detailed_logging(self, caplog):
        """Test handling of different exception types and verify comprehensive logging for each."""

        exception_scenarios = [
            ("ValueError", "Test ValueError for comprehensive logging", False),
            ("RuntimeError", "Test RuntimeError for comprehensive logging", False),
            ("TypeError", "Test TypeError for comprehensive logging", False),
            ("KeyError", "Test KeyError for comprehensive logging", False),
        ]

        for exc_type, exc_msg, should_handle in exception_scenarios:
            with caplog.at_level(logging.ERROR):
                # Clear previous log records
                caplog.clear()

                mock_config = MockExceptionWorkflowConfig(exception_type=exc_type,
                                                          exception_message=exc_msg,
                                                          handle_error=should_handle,
                                                          use_openai_api=True)

                config = Config(workflow=mock_config)
                async with WorkflowBuilder.from_config(config=config) as workflow_builder:
                    workflow = workflow_builder.build()
                    session_manager = SessionManager(workflow)
                    test_input = ChatRequest.from_string(f"testing {exc_type} error logging")

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

                # Verify logging for this specific exception type
                self._verify_exception_type_logging(caplog.records, exc_type, exc_msg, test_input)

    async def test_logging_contains_function_context(self, mock_config_no_error_handling, caplog):
        """Test that error logs contain comprehensive context about function name, scope, and execution details."""

        config = Config(workflow=mock_config_no_error_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)
            test_input = ChatRequest.from_string("context test input with specific content")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(RuntimeError):
                    async with session_manager.run(test_input) as runner:
                        await runner.result()

            # Perform detailed log analysis
            self._verify_function_context_logging(caplog.records, test_input)

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
        mock_config = MockExceptionWorkflowConfig(exception_type="ValueError",
                                                  exception_message=custom_error_message,
                                                  handle_error=False,
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
        assert not valid_config.handle_error  # Default

        # Config without error handling should also work
        no_handle_config = MockExceptionWorkflowConfig(exception_type="RuntimeError",
                                                       exception_message="runtime error test",
                                                       handle_error=False)
        assert no_handle_config.exception_type == "RuntimeError"
        assert no_handle_config.exception_message == "runtime error test"
        assert not no_handle_config.handle_error

    def _verify_comprehensive_error_logging(self, log_records, test_input):
        """Helper method to verify comprehensive error logging information."""

        error_logs = [record for record in log_records if record.levelno == logging.ERROR]
        assert len(error_logs) >= 2, f"Expected at least 2 error log records, got {len(error_logs)}"

        # 1. Function-level error log (more detailed)
        function_error_logs = [log for log in error_logs if "Error with ainvoke in function" in log.message]
        assert len(function_error_logs) > 0, "Should have function-level error log"

        function_log = function_error_logs[0]

        # Verify function log contains comprehensive information
        assert "nat.builder.function" in function_log.name, f"Expected function logger, got {function_log.name}"
        assert function_log.filename == "function.py", f"Expected function.py, got {function_log.filename}"
        assert function_log.levelno == logging.ERROR, f"Expected ERROR level, got {function_log.levelno}"

        # Check that the log message contains detailed input context
        log_message = function_log.getMessage()
        assert "Error with ainvoke in function with input:" in log_message
        assert "test input for error logging" in log_message, "Should contain input content"
        assert "messages=" in log_message, "Should contain message structure info"
        assert "Unhandled RuntimeError from mock workflow" in log_message, "Should contain original error"

        # 2. Runner-level error log (higher level)
        runner_error_logs = [log for log in error_logs if "Error running workflow" in log.message]
        assert len(runner_error_logs) > 0, "Should have runner-level error log"

        runner_log = runner_error_logs[0]

        # Verify runner log contains essential information
        assert "nat.runtime.runner" in runner_log.name, f"Expected runner logger, got {runner_log.name}"
        assert runner_log.filename == "runner.py", f"Expected runner.py, got {runner_log.filename}"
        assert runner_log.levelno == logging.ERROR, f"Expected ERROR level, got {runner_log.levelno}"

        runner_message = runner_log.getMessage()
        assert "Error running workflow:" in runner_message
        assert "Unhandled RuntimeError from mock workflow" in runner_message, "Should preserve original error message"

    def _verify_function_context_logging(self, log_records, test_input):
        """Helper method to verify function context and scope information in logs."""

        error_logs = [record for record in log_records if record.levelno == logging.ERROR]

        # Find the function-level error log
        function_logs = [log for log in error_logs if "Error with ainvoke in function" in log.message]
        assert len(function_logs) > 0, "Should have detailed function error log"

        function_log = function_logs[0]
        log_message = function_log.getMessage()

        # Verify comprehensive function context information
        context_checks = {
            "Module/Logger Name":
                "nat.builder.function" in function_log.name,
            "File Name":
                function_log.filename == "function.py",
            "Function Scope":
                "Error with ainvoke in function" in log_message,
            "Input Content":
                "context test input with specific content" in log_message,
            "Input Structure":
                "messages=" in log_message,
            "Error Type":
                "RuntimeError" in log_message,
            "Original Error Message":
                "Unhandled RuntimeError from mock workflow" in log_message,
            "Detailed Input Context":
                any(field in log_message for field in ["frequency_penalty", "max_tokens", "temperature"]),
        }

        failed_checks = [name for name, passed in context_checks.items() if not passed]
        assert len(failed_checks) == 0, f"Failed context checks: {failed_checks}"

        # Verify log record has proper metadata
        assert function_log.levelname == "ERROR"
        assert function_log.lineno > 0, "Should have line number information"
        assert function_log.pathname.endswith("function.py"), "Should have correct file path"

        # Check runner-level logging provides workflow context
        runner_logs = [log for log in error_logs if "Error running workflow" in log.message]
        assert len(runner_logs) > 0, "Should have runner-level error log"

        runner_log = runner_logs[0]
        assert "nat.runtime.runner" in runner_log.name, "Should identify runner as error source"
        assert runner_log.filename == "runner.py", "Should identify runner file"

    def _verify_exception_type_logging(self, log_records, exc_type, exc_msg, test_input):
        """Helper method to verify comprehensive logging for specific exception types."""

        error_logs = [record for record in log_records if record.levelno == logging.ERROR]
        assert len(error_logs) >= 1, f"Should have error logs for {exc_type}"

        # Find function-level error logs
        function_logs = [log for log in error_logs if "Error with ainvoke in function" in log.message]

        if len(function_logs) > 0:
            function_log = function_logs[0]
            log_message = function_log.getMessage()

            # Verify exception-specific details are logged
            exception_checks = {
                f"{exc_type} in message": exc_type in log_message or exc_msg in log_message,
                "Input content preserved": f"testing {exc_type} error logging" in log_message,  # Fixed case sensitivity
                "Function context": "Error with ainvoke in function" in log_message,
                "Module identification": "nat.builder.function" in function_log.name,
                "Proper log level": function_log.levelname == "ERROR",
                "File information": function_log.filename == "function.py",
                "Line number": function_log.lineno > 0,
            }

            failed_checks = [name for name, passed in exception_checks.items() if not passed]
            assert len(failed_checks) == 0, f"Failed checks for {exc_type}: {failed_checks}. Log: {log_message}"

    async def test_error_logging_with_detailed_input_context(self, mock_config_no_error_handling, caplog):
        """Test that error logs preserve detailed input context and parameters."""

        config = Config(workflow=mock_config_no_error_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)

            # Create input with specific identifiable content
            test_input = ChatRequest.from_string("Complex input with unique identifier: TEST_ID_12345")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(RuntimeError):
                    async with session_manager.run(test_input) as runner:
                        await runner.result()

            # Find detailed function error log
            function_logs = [
                log for log in caplog.records
                if log.levelno == logging.ERROR and "Error with ainvoke in function" in log.message
            ]
            assert len(function_logs) > 0, "Should have detailed function error log"

            function_log = function_logs[0]
            log_message = function_log.getMessage()

            # Verify comprehensive input context preservation
            input_context_checks = {
                "Unique input identifier":
                    "TEST_ID_12345" in log_message,
                "Input structure":
                    "messages=" in log_message,
                "Message content":
                    "Complex input with unique identifier" in log_message,
                "Chat parameters":
                    any(param in log_message for param in ["frequency_penalty", "max_tokens", "temperature", "model"]),
                "Input format details":
                    any(detail in log_message for detail in ["role=", "content=", "user"]),
            }

            failed_checks = [name for name, passed in input_context_checks.items() if not passed]
            assert len(failed_checks) == 0, f"Failed input context checks: {failed_checks}"

    async def test_error_logging_source_identification(self, mock_config_no_error_handling, caplog):
        """Test that error logs clearly identify the source module, function, and execution path."""

        config = Config(workflow=mock_config_no_error_handling)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)
            test_input = ChatRequest.from_string("source identification test")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(RuntimeError):
                    async with session_manager.run(test_input) as runner:
                        await runner.result()

            error_logs = [record for record in caplog.records if record.levelno == logging.ERROR]

            # Verify we have logs from different parts of the system
            log_sources = {}
            for log in error_logs:
                log_sources[log.name] = log

            # Check expected log sources
            expected_sources = ["nat.builder.function", "nat.runtime.runner"]
            found_sources = [source for source in expected_sources if source in log_sources]
            assert len(found_sources) >= 1, f"Should have logs from expected sources. Found: {list(log_sources.keys())}"

            # Verify each log source provides appropriate information
            if "nat.builder.function" in log_sources:
                func_log = log_sources["nat.builder.function"]
                assert "Error with ainvoke" in func_log.getMessage()
                assert func_log.filename == "function.py"

            if "nat.runtime.runner" in log_sources:
                runner_log = log_sources["nat.runtime.runner"]
                assert "Error running workflow" in runner_log.getMessage()
                assert runner_log.filename == "runner.py"

    async def test_error_logging_preserves_exception_hierarchy(self, caplog):
        """Test that error logging preserves exception type hierarchy and details."""

        # Test with custom exception message that includes special characters and details
        custom_message = "Custom error: Failed operation [ID: abc123] with params {retry: 3, timeout: 30}"

        mock_config = MockExceptionWorkflowConfig(exception_type="ValueError",
                                                  exception_message=custom_message,
                                                  handle_error=False,
                                                  use_openai_api=True)

        config = Config(workflow=mock_config)
        async with WorkflowBuilder.from_config(config=config) as workflow_builder:
            workflow = workflow_builder.build()
            session_manager = SessionManager(workflow)
            test_input = ChatRequest.from_string("exception hierarchy test")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(ValueError, match="Custom error"):
                    async with session_manager.run(test_input) as runner:
                        await runner.result()

            # Verify exception details are preserved in logs
            error_logs = [record for record in caplog.records if record.levelno == logging.ERROR]

            # Check that complex error message is preserved exactly
            found_detailed_error = False
            for log in error_logs:
                if custom_message in log.getMessage():
                    found_detailed_error = True

                    # Verify special characters and structure are preserved
                    log_msg = log.getMessage()
                    assert "[ID: abc123]" in log_msg, "Should preserve bracketed IDs"
                    assert "{retry: 3, timeout: 30}" in log_msg, "Should preserve parameter details"
                    assert "Failed operation" in log_msg, "Should preserve operation context"
                    break

            assert found_detailed_error, "Should find log with complete custom error message"
