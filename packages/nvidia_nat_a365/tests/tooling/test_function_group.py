# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""Unit tests for A365MCPToolingFunctionGroup aggregation logic."""

from unittest.mock import AsyncMock, Mock

import pytest

from nat.builder.function import Function, FunctionGroup
from nat.plugins.a365.tooling import A365MCPToolingConfig
from nat.plugins.a365.tooling.register import A365MCPToolingFunctionGroup


@pytest.fixture
def base_config():
    """Base config for tests."""
    return A365MCPToolingConfig(
        agentic_app_id="test-agent",
        auth_token="test-token",
    )


@pytest.fixture
def mock_function():
    """Create a mock Function."""
    func = Mock(spec=Function)
    func.name = "test_function"
    return func


@pytest.fixture
def mock_mcp_group():
    """Create a mock MCP FunctionGroup."""
    group = Mock(spec=FunctionGroup)
    return group


class TestFunctionAggregation:
    """Test function aggregation logic in A365MCPToolingFunctionGroup."""

    @pytest.mark.asyncio
    async def test_function_name_collision_overwrites(self, base_config, mock_function):
        """Test that function name collisions result in the last group's function winning."""
        # Create two groups with the same function name
        group1 = Mock(spec=FunctionGroup)
        func1 = Mock(spec=Function)
        func1.name = "collision_func"
        group1.get_all_functions = AsyncMock(return_value={"collision_func": func1})

        group2 = Mock(spec=FunctionGroup)
        func2 = Mock(spec=Function)
        func2.name = "collision_func"
        group2.get_all_functions = AsyncMock(return_value={"collision_func": func2})

        composite = A365MCPToolingFunctionGroup(config=base_config, mcp_groups=[group1, group2])

        all_functions = await composite.get_all_functions()

        # Should only have one function (last one wins)
        assert len(all_functions) == 1
        assert "collision_func" in all_functions
        # Should be func2 (from group2, the last one)
        assert all_functions["collision_func"] is func2

    @pytest.mark.asyncio
    async def test_filter_fn_propagated_to_all_groups(self, base_config):
        """Test that filter_fn parameter is passed to all MCP groups."""
        group1 = Mock(spec=FunctionGroup)
        group1.get_all_functions = AsyncMock(return_value={"func1": Mock()})

        group2 = Mock(spec=FunctionGroup)
        group2.get_all_functions = AsyncMock(return_value={"func2": Mock()})

        composite = A365MCPToolingFunctionGroup(config=base_config, mcp_groups=[group1, group2])

        # Create a filter function
        async def filter_fn(names):
            return names

        await composite.get_all_functions(filter_fn=filter_fn)

        # Verify filter_fn was passed to both groups
        group1.get_all_functions.assert_called_once_with(filter_fn=filter_fn)
        group2.get_all_functions.assert_called_once_with(filter_fn=filter_fn)

    @pytest.mark.asyncio
    async def test_filter_fn_propagated_to_accessible_functions(self, base_config):
        """Test that filter_fn is propagated to get_accessible_functions."""
        group1 = Mock(spec=FunctionGroup)
        group1.get_accessible_functions = AsyncMock(return_value={"func1": Mock()})

        composite = A365MCPToolingFunctionGroup(config=base_config, mcp_groups=[group1])

        async def filter_fn(names):
            return names

        await composite.get_accessible_functions(filter_fn=filter_fn)

        group1.get_accessible_functions.assert_called_once_with(filter_fn=filter_fn)

    @pytest.mark.asyncio
    async def test_filter_fn_propagated_to_included_functions(self, base_config):
        """Test that filter_fn is propagated to get_included_functions."""
        group1 = Mock(spec=FunctionGroup)
        group1.get_included_functions = AsyncMock(return_value={"func1": Mock()})

        composite = A365MCPToolingFunctionGroup(config=base_config, mcp_groups=[group1])

        async def filter_fn(names):
            return names

        await composite.get_included_functions(filter_fn=filter_fn)

        group1.get_included_functions.assert_called_once_with(filter_fn=filter_fn)

    @pytest.mark.asyncio
    async def test_filter_fn_propagated_to_excluded_functions(self, base_config):
        """Test that filter_fn is propagated to get_excluded_functions."""
        group1 = Mock(spec=FunctionGroup)
        group1.get_excluded_functions = AsyncMock(return_value={})

        composite = A365MCPToolingFunctionGroup(config=base_config, mcp_groups=[group1])

        async def filter_fn(names):
            return names

        await composite.get_excluded_functions(filter_fn=filter_fn)

        group1.get_excluded_functions.assert_called_once_with(filter_fn=filter_fn)

    @pytest.mark.asyncio
    async def test_empty_groups_list(self, base_config):
        """Test that empty groups list returns empty functions dict."""
        composite = A365MCPToolingFunctionGroup(config=base_config, mcp_groups=[])

        all_functions = await composite.get_all_functions()
        assert all_functions == {}

        accessible = await composite.get_accessible_functions()
        assert accessible == {}

        included = await composite.get_included_functions()
        assert included == {}

        excluded = await composite.get_excluded_functions()
        assert excluded == {}
