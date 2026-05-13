# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import typing
from unittest.mock import MagicMock

import pytest
from langchain_core.tools.base import BaseTool
from pydantic import PrivateAttr

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.data_models.component_ref import FunctionRef
from nat.plugins.langchain.control_flow.parallel_executor import ParallelExecutorConfig
from nat.plugins.langchain.control_flow.parallel_executor import UnknownParallelToolsError
from nat.plugins.langchain.control_flow.parallel_executor import parallel_execution


class MockParallelTool(BaseTool):
    """Mock tool for testing the parallel executor."""

    name: str = "mock_parallel_tool"
    description: str = "A mock parallel tool for testing"

    _response: typing.Any = PrivateAttr(default=None)
    _error: Exception | None = PrivateAttr(default=None)
    _queries: list[typing.Any] = PrivateAttr(default_factory=list)

    def __init__(self,
                 name: str,
                 response: typing.Any = None,
                 error: Exception | None = None,
                 **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.name = name
        self._response = response
        self._error = error
        self._queries = []

    async def _arun(self, query: typing.Any = None, **kwargs: typing.Any) -> typing.Any:
        self._queries.append(query)
        if self._error is not None:
            raise self._error
        return self._response

    def _run(self, query: typing.Any = None, **kwargs: typing.Any) -> typing.Any:
        self._queries.append(query)
        if self._error is not None:
            raise self._error
        return self._response

    @property
    def queries(self) -> list[typing.Any]:
        return self._queries


class TestParallelExecutorConfig:
    """Test cases for ParallelExecutorConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ParallelExecutorConfig()
        assert config.description == "Parallel Executor Workflow"
        assert config.tool_list == []
        assert not config.detailed_logs
        assert not config.return_error_on_exception

    def test_config_with_values(self) -> None:
        """Test configuration with custom values."""
        config = ParallelExecutorConfig(
            description="Parallel analysis",
            tool_list=[FunctionRef("topic_agent"), FunctionRef("risk_agent")],
            detailed_logs=True,
            return_error_on_exception=True,
        )

        assert config.description == "Parallel analysis"
        assert config.tool_list == [FunctionRef("topic_agent"), FunctionRef("risk_agent")]
        assert config.detailed_logs
        assert config.return_error_on_exception


class TestParallelExecution:
    """Test cases for parallel execution behavior."""

    @pytest.mark.asyncio
    async def test_parallel_execution_merges_branch_outputs(self) -> None:
        """Test fan-out/fan-in branch execution and appended string output."""
        builder = MagicMock(spec=Builder)
        topic_tool = MockParallelTool(name="topic_agent", response={"topic": "product"})
        risk_tool = MockParallelTool(name="risk_agent", response="low")
        builder.get_tools.return_value = [topic_tool, risk_tool]

        config = ParallelExecutorConfig(tool_list=[FunctionRef("topic_agent"), FunctionRef("risk_agent")])

        async with parallel_execution(config, builder) as function_info:
            assert isinstance(function_info, FunctionInfo)
            parallel_fn = function_info.single_fn  # type: ignore[assignment]
            result = await parallel_fn("Launch update request")  # type: ignore[misc]

        assert isinstance(result, str)
        assert "topic_agent:" in result
        assert "\"topic\": \"product\"" in result
        assert "risk_agent:" in result
        assert "low" in result
        assert topic_tool.queries == ["Launch update request"]
        assert risk_tool.queries == ["Launch update request"]

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_error(self) -> None:
        """Test validation when configured tools cannot be resolved."""
        builder = MagicMock(spec=Builder)
        builder.get_tools.return_value = [MockParallelTool(name="topic_agent", response="product")]
        config = ParallelExecutorConfig(tool_list=[FunctionRef("topic_agent"), FunctionRef("missing_tool")])

        with pytest.raises(UnknownParallelToolsError, match="missing_tool"):
            async with parallel_execution(config, builder) as _:
                pass

    @pytest.mark.asyncio
    async def test_branch_exception_raises_by_default(self) -> None:
        """Test default behavior where branch exceptions are raised."""
        builder = MagicMock(spec=Builder)
        ok_tool = MockParallelTool(name="topic_agent", response="product")
        failing_tool = MockParallelTool(name="risk_agent", error=RuntimeError("branch failed"))
        builder.get_tools.return_value = [ok_tool, failing_tool]

        config = ParallelExecutorConfig(tool_list=[FunctionRef("topic_agent"), FunctionRef("risk_agent")])

        async with parallel_execution(config, builder) as function_info:
            parallel_fn = function_info.single_fn  # type: ignore[assignment]
            with pytest.raises(RuntimeError, match="branch failed"):
                await parallel_fn("Launch update request")  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_branch_exception_returned_when_configured(self) -> None:
        """Test optional behavior where branch exceptions are returned in appended output."""
        builder = MagicMock(spec=Builder)
        ok_tool = MockParallelTool(name="topic_agent", response="product")
        failing_tool = MockParallelTool(name="risk_agent", error=RuntimeError("branch failed"))
        builder.get_tools.return_value = [ok_tool, failing_tool]

        config = ParallelExecutorConfig(
            tool_list=[FunctionRef("topic_agent"), FunctionRef("risk_agent")],
            return_error_on_exception=True,
        )

        async with parallel_execution(config, builder) as function_info:
            parallel_fn = function_info.single_fn  # type: ignore[assignment]
            result = await parallel_fn("Launch update request")  # type: ignore[misc]

        assert isinstance(result, str)
        assert "topic_agent:" in result
        assert "product" in result
        assert "risk_agent:" in result
        assert "ERROR: RuntimeError: branch failed" in result

    @pytest.mark.asyncio
    async def test_detailed_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test detailed fan-out and fan-in logs."""
        builder = MagicMock(spec=Builder)
        topic_tool = MockParallelTool(name="topic_agent", response="product")
        risk_tool = MockParallelTool(name="risk_agent", response="low")
        builder.get_tools.return_value = [topic_tool, risk_tool]

        config = ParallelExecutorConfig(
            tool_list=[FunctionRef("topic_agent"), FunctionRef("risk_agent")],
            detailed_logs=True,
        )

        with caplog.at_level(logging.INFO):
            async with parallel_execution(config, builder) as function_info:
                parallel_fn = function_info.single_fn  # type: ignore[assignment]
                await parallel_fn("Launch update request")  # type: ignore[misc]

        assert "fan-out start" in caplog.text
        assert "start branch=topic_agent" in caplog.text
        assert "start branch=risk_agent" in caplog.text
        assert "fan-in complete" in caplog.text
