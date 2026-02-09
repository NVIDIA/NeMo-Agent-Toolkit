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

"""Registration tests for A365 tooling integration."""

import sys
from unittest.mock import Mock

import pytest

from nat.cli.type_registry import GlobalTypeRegistry
from nat.plugins.a365.tooling import A365MCPToolingConfig
from nat.runtime.loader import PluginTypes, discover_and_register_plugins
from nat.utils.optional_imports import OptionalImportError


@pytest.fixture(autouse=True)
def discover_plugins():
    """Discover and register all plugins before each test."""
    discover_and_register_plugins(PluginTypes.ALL)


class TestA365ToolingConfigRegistration:
    """Test that A365MCPToolingConfig is properly registered."""

    def test_config_class_registered(self):
        """Test that A365MCPToolingConfig is registered in the type registry."""
        registry = GlobalTypeRegistry.get()

        # Should be able to retrieve the function group registration
        registration = registry.get_function_group(A365MCPToolingConfig)

        assert registration is not None
        assert registration.config_type == A365MCPToolingConfig
        assert registration.full_type == "a365_mcp_tooling"

    def test_config_class_discoverable(self):
        """Test that A365MCPToolingConfig can be discovered via entry points."""
        registry = GlobalTypeRegistry.get()

        # Verify it's registered (discovery happens via autouse fixture)
        registration = registry.get_function_group(A365MCPToolingConfig)
        assert registration is not None
        assert registration.config_type == A365MCPToolingConfig

    def test_config_class_has_correct_name(self):
        """Test that A365MCPToolingConfig has the correct name attribute."""
        assert A365MCPToolingConfig.full_type == "a365_mcp_tooling"
        assert A365MCPToolingConfig.name == "a365_mcp_tooling"


class TestA365ToolingMissingMCPDependency:
    """Test handling of missing nvidia-nat-mcp dependency."""

    @pytest.mark.asyncio
    async def test_missing_mcp_dependency_raises_optional_import_error(self):
        """Test that missing nvidia-nat-mcp raises OptionalImportError with helpful message."""
        from nat.plugins.a365.tooling.register import a365_mcp_tooling_function_group

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token="test-token",
        )

        mock_builder = Mock()

        # Temporarily remove MCP modules from sys.modules to simulate missing dependency
        # Save original modules
        saved_modules = {}
        mcp_module_key = "nat.plugins.mcp.client.client_config"
        if mcp_module_key in sys.modules:
            saved_modules[mcp_module_key] = sys.modules[mcp_module_key]
            del sys.modules[mcp_module_key]

        # Also remove parent modules that might prevent re-import
        for key in list(sys.modules.keys()):
            if key.startswith("nat.plugins.mcp") and key != "nat.plugins.mcp":
                if key not in saved_modules:
                    saved_modules[key] = sys.modules[key]
                del sys.modules[key]

        try:
            with pytest.raises(OptionalImportError) as exc_info:
                await a365_mcp_tooling_function_group(config, mock_builder)

            # Verify the error message contains helpful installation instructions
            error_msg = str(exc_info.value)
            assert "nvidia-nat-mcp" in error_msg
            assert "uv pip install nvidia-nat-mcp" in error_msg or "nvidia-nat[mcp]" in error_msg
            assert "A365 tooling feature requires the MCP client functionality" in error_msg
        finally:
            # Restore modules
            sys.modules.update(saved_modules)

    @pytest.mark.asyncio
    async def test_missing_mcp_dependency_error_message_format(self):
        """Test that OptionalImportError has the correct format."""
        from nat.plugins.a365.tooling.register import a365_mcp_tooling_function_group

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token="test-token",
        )

        mock_builder = Mock()

        # Temporarily remove MCP modules from sys.modules to simulate missing dependency
        saved_modules = {}
        mcp_module_key = "nat.plugins.mcp.client.client_config"
        if mcp_module_key in sys.modules:
            saved_modules[mcp_module_key] = sys.modules[mcp_module_key]
            del sys.modules[mcp_module_key]

        for key in list(sys.modules.keys()):
            if key.startswith("nat.plugins.mcp") and key != "nat.plugins.mcp":
                if key not in saved_modules:
                    saved_modules[key] = sys.modules[key]
                del sys.modules[key]

        try:
            with pytest.raises(OptionalImportError) as exc_info:
                await a365_mcp_tooling_function_group(config, mock_builder)

            # Verify OptionalImportError structure
            assert exc_info.value.module_name == "nvidia-nat-mcp"
            assert len(exc_info.value.args) > 0
            assert "additional_message" in str(exc_info.value) or "MCP client functionality" in str(exc_info.value)
        finally:
            # Restore modules
            sys.modules.update(saved_modules)


class TestA365ToolingFunctionGroupDiscovery:
    """Test discovery and loading of the A365 tooling function group."""

    def test_function_group_discovered_via_entry_point(self):
        """Test that the function group is discoverable via entry points."""
        registry = GlobalTypeRegistry.get()

        # Verify registration exists (discovery happens via autouse fixture)
        registration = registry.get_function_group(A365MCPToolingConfig)
        assert registration is not None
        assert callable(registration.build_fn)

    def test_config_class_instantiation(self):
        """Test that A365MCPToolingConfig can be instantiated with required fields."""
        config = A365MCPToolingConfig(
            agentic_app_id="test-agent-123",
            auth_token="test-token-456",
        )

        assert config.agentic_app_id == "test-agent-123"
        assert config.auth_token == "test-token-456"

    def test_config_class_with_optional_fields(self):
        """Test that A365MCPToolingConfig accepts optional fields."""
        from datetime import timedelta

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token="test-token",
            tool_call_timeout=timedelta(seconds=120),
            reconnect_enabled=False,
            max_sessions=50,
        )
        assert config.tool_call_timeout == timedelta(seconds=120)
        assert config.reconnect_enabled is False
        assert config.max_sessions == 50

    def test_config_class_validation(self):
        """Test that A365MCPToolingConfig validates reconnect backoff values."""
        from datetime import timedelta

        # Should raise ValueError if max_backoff < initial_backoff
        with pytest.raises(ValueError, match="reconnect_max_backoff must be greater than or equal"):
            A365MCPToolingConfig(
                agentic_app_id="test-agent",
                auth_token="test-token",
                reconnect_initial_backoff=10.0,
                reconnect_max_backoff=5.0,  # Invalid: max < initial
            )

        # Should work with valid values
        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token="test-token",
            reconnect_initial_backoff=0.5,
            reconnect_max_backoff=50.0,
        )
        assert config.reconnect_initial_backoff == 0.5
        assert config.reconnect_max_backoff == 50.0
