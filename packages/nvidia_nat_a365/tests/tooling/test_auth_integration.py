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

"""Integration tests for A365 tooling integration with delegation pattern."""

import sys
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from pydantic import SecretStr

from nat.builder.function import FunctionGroup
from nat.data_models.authentication import AuthResult, BearerTokenCred, HeaderCred
from nat.data_models.component_ref import AuthenticationRef
from nat.plugins.a365.tooling import A365MCPToolingConfig
from nat.plugins.a365.tooling.register import (
    A365MCPToolingFunctionGroup,
    a365_mcp_tooling_function_group,
)
from nat.utils.optional_imports import OptionalImportError

# Skip all tests on Python 3.13 until nvidia-nat-mcp supports it
pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason="nvidia-nat-mcp does not support Python 3.13 yet. These tests require MCP functionality.",
)


@pytest.fixture
def mock_auth_provider():
    """Create a mock auth provider."""
    provider = Mock()
    provider.authenticate = AsyncMock()
    return provider


@pytest.fixture
def mock_builder(mock_auth_provider):
    """Create a mock builder with auth provider resolution."""
    builder = Mock()
    builder.get_auth_provider = AsyncMock(return_value=mock_auth_provider)
    return builder


@pytest.fixture
def mock_a365_service():
    """Create a mock A365ToolingService."""
    service = Mock()
    service.list_tool_servers = AsyncMock()
    return service


@pytest.fixture
def mock_mcp_servers():
    """Create mock MCP server configurations."""
    server1 = Mock()
    server1.mcp_server_name = "server-1"
    server1.url = "https://mcp-server-1.example.com"

    server2 = Mock()
    server2.mcp_server_name = "server-2"
    server2.url = "https://mcp-server-2.example.com"

    return [server1, server2]


@pytest.fixture
def mock_mcp_function_group():
    """Create a mock MCP FunctionGroup with functions."""
    def create_mock_group(server_name: str, tool_names: list[str]):
        """Create a mock function group for a specific server."""
        group = Mock(spec=FunctionGroup)
        # Functions are namespaced with the MCP group's instance name
        # For simplicity, we'll use "mcp_client" as the namespace
        functions = {
            f"mcp_client__{tool_name}": Mock(
                ainvoke=AsyncMock(return_value=f"result-from-{tool_name}"),
                has_single_output=True,
                input_schema=None,
                description=f"Tool {tool_name}",
                converters=None,
            )
            for tool_name in tool_names
        }
        group.get_all_functions = AsyncMock(return_value=functions)
        group.get_accessible_functions = AsyncMock(return_value=functions)
        group.get_included_functions = AsyncMock(return_value=functions)
        group.get_excluded_functions = AsyncMock(return_value={})
        return group

    return create_mock_group


@pytest.fixture
def mock_mcp_client_function_group(mock_mcp_function_group):
    """Fixture to mock mcp_client_function_group as an async context manager."""
    call_count = {"value": 0}

    async def mock_mcp_group_generator(*args, **kwargs):
        """Mock async generator that yields a function group."""
        call_count["value"] += 1
        # First server gets 2 tools, second server gets 1 tool
        tool_names = ["tool1", "tool2"] if call_count["value"] == 1 else ["tool3"]
        group = mock_mcp_function_group(f"server-{call_count['value']}", tool_names)
        yield group

    def factory(*args, **kwargs):
        return asynccontextmanager(mock_mcp_group_generator)(*args, **kwargs)

    return factory


@pytest.fixture
def base_config():
    """Base config for tests."""
    return A365MCPToolingConfig(
        agentic_app_id="test-agent",
        auth_token=AuthenticationRef("test_auth"),
    )


@contextmanager
def patch_services(mock_a365_service, mock_mcp_client_function_group):
    """Context manager to patch A365 and MCP services."""
    import nat.plugins.mcp.client.client_impl

    mock_service_class = Mock(return_value=mock_a365_service)
    # Patch A365ToolingService where it's imported in register.py
    # Since the import happens inside the function, we patch the source module
    with patch("nat.plugins.a365.tooling.A365ToolingService", new=mock_service_class):
        with patch.object(
            nat.plugins.mcp.client.client_impl,
            "mcp_client_function_group",
            side_effect=mock_mcp_client_function_group,
        ) as mock_mcp_patched:
            yield mock_mcp_patched


class TestDelegationPattern:
    """Test that the composite group correctly delegates to MCP groups."""

    @pytest.mark.asyncio
    async def test_composite_group_delegates_to_mcp_groups(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
        base_config,
    ):
        """Test that composite group aggregates functions from all MCP groups."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))]
        )
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            async with a365_mcp_tooling_function_group(base_config, mock_builder) as composite_group:
                assert isinstance(composite_group, A365MCPToolingFunctionGroup)

                # Get all functions - should aggregate from both MCP groups
                all_functions = await composite_group.get_all_functions()
                # First server: tool1, tool2; Second server: tool3
                assert len(all_functions) == 3
                assert "mcp_client__tool1" in all_functions
                assert "mcp_client__tool2" in all_functions
                assert "mcp_client__tool3" in all_functions

    @pytest.mark.asyncio
    async def test_get_accessible_functions_delegates(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
        base_config,
    ):
        """Test that get_accessible_functions delegates to MCP groups."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))]
        )
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            async with a365_mcp_tooling_function_group(base_config, mock_builder) as composite_group:
                accessible = await composite_group.get_accessible_functions()
                assert len(accessible) == 3
                # Verify it called get_accessible_functions on each MCP group
                # (The mock returns the same as get_all_functions, but in real usage
                # it would respect include/exclude)

    @pytest.mark.asyncio
    async def test_get_included_functions_delegates(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
        base_config,
    ):
        """Test that get_included_functions delegates to MCP groups."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))]
        )
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            async with a365_mcp_tooling_function_group(base_config, mock_builder) as composite_group:
                included = await composite_group.get_included_functions()
                assert len(included) == 3

    @pytest.mark.asyncio
    async def test_get_excluded_functions_delegates(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
        base_config,
    ):
        """Test that get_excluded_functions delegates to MCP groups."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))]
        )
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            async with a365_mcp_tooling_function_group(base_config, mock_builder) as composite_group:
                excluded = await composite_group.get_excluded_functions()
                # Mock groups return empty excluded, so should be empty
                assert len(excluded) == 0


class TestTokenExtraction:
    """Test token extraction from auth provider credentials."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "credentials,expected_token",
        [
            (
                [BearerTokenCred(token=SecretStr("test-bearer-token-123"))],
                "test-bearer-token-123",
            ),
            (
                [HeaderCred(name="Authorization", value=SecretStr("Bearer test-header-token-456"))],
                "test-header-token-456",
            ),
            (
                [HeaderCred(name="Authorization", value=SecretStr("CustomScheme token-without-bearer"))],
                "CustomScheme token-without-bearer",
            ),
        ],
    )
    async def test_token_extraction(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
        base_config,
        credentials,
        expected_token,
    ):
        """Test extracting token from various credential types."""
        mock_auth_provider.authenticate.return_value = AuthResult(credentials=credentials)
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            async with a365_mcp_tooling_function_group(base_config, mock_builder):
                mock_auth_provider.authenticate.assert_called_once()
                call_args = mock_a365_service.list_tool_servers.call_args
                assert call_args.kwargs["auth_token"] == expected_token

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "credentials,error_match",
        [
            (
                [HeaderCred(name="X-Custom-Header", value=SecretStr("custom-value"))],
                "No bearer token found",
            ),
            (
                [],
                "No credentials available",
            ),
        ],
    )
    async def test_token_extraction_errors(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_client_function_group,
        base_config,
        credentials,
        error_match,
    ):
        """Test error handling when token cannot be extracted."""
        mock_auth_provider.authenticate.return_value = AuthResult(credentials=credentials)
        # Return empty list so code can proceed past service discovery
        mock_a365_service.list_tool_servers.return_value = []

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            with pytest.raises(ValueError, match=error_match):
                async with a365_mcp_tooling_function_group(base_config, mock_builder):
                    pass


class TestAuthProviderPriority:
    """Test auth provider priority logic for MCP servers."""

    @pytest.mark.asyncio
    async def test_per_server_override_takes_priority(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
    ):
        """Test that per-server override takes priority over gateway auth."""
        from nat.plugins.mcp.client.client_config import MCPClientConfig

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("gateway-token"))]
        )
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        override_auth_provider = Mock()
        override_auth_provider.authenticate = AsyncMock()
        mock_builder.get_auth_provider = AsyncMock(
            side_effect=lambda ref: (
                override_auth_provider if str(ref) == "override_auth" else mock_auth_provider
            )
        )

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token=AuthenticationRef("gateway_auth"),
            server_auth_providers={"server-1": AuthenticationRef("override_auth")},
        )

        with patch_services(mock_a365_service, mock_mcp_client_function_group) as mock_mcp_patched:
            async with a365_mcp_tooling_function_group(config, mock_builder):
                assert mock_mcp_patched.call_count == 2

                first_mcp_config: MCPClientConfig = mock_mcp_patched.call_args_list[0][0][0]
                assert str(first_mcp_config.server.auth_provider) == "override_auth"

                second_mcp_config: MCPClientConfig = mock_mcp_patched.call_args_list[1][0][0]
                assert str(second_mcp_config.server.auth_provider) == "gateway_auth"

    @pytest.mark.asyncio
    async def test_gateway_auth_used_when_no_override(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
    ):
        """Test that gateway auth is used when no per-server override."""
        from nat.plugins.mcp.client.client_config import MCPClientConfig

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("gateway-token"))]
        )
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token=AuthenticationRef("gateway_auth"),
        )

        with patch_services(mock_a365_service, mock_mcp_client_function_group) as mock_mcp_patched:
            async with a365_mcp_tooling_function_group(config, mock_builder):
                assert mock_mcp_patched.call_count == 2
                for call in mock_mcp_patched.call_args_list:
                    server_mcp_config: MCPClientConfig = call[0][0]
                    assert str(server_mcp_config.server.auth_provider) == "gateway_auth"

    @pytest.mark.asyncio
    async def test_string_token_no_auth_for_servers(
        self, mock_builder, mock_a365_service, mock_mcp_servers, mock_mcp_client_function_group
    ):
        """Test that string token doesn't pass auth to MCP servers."""
        from nat.plugins.mcp.client.client_config import MCPClientConfig

        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token="string-token-123",
        )

        with patch_services(mock_a365_service, mock_mcp_client_function_group) as mock_mcp_patched:
            async with a365_mcp_tooling_function_group(config, mock_builder):
                call_args = mock_a365_service.list_tool_servers.call_args
                assert call_args.kwargs["auth_token"] == "string-token-123"

                assert mock_mcp_patched.call_count == 2
                for call in mock_mcp_patched.call_args_list:
                    server_mcp_config: MCPClientConfig = call[0][0]
                    assert server_mcp_config.server.auth_provider is None


class TestUserContext:
    """Test user context handling for OAuth flows."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("has_context", [True, False])
    async def test_user_id_passed_to_authenticate(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
        base_config,
        has_context,
    ):
        """Test that user_id from context is passed to authenticate()."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))]
        )
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        if has_context:
            test_user_id = f"user-{uuid4()}"
            mock_context = Mock()
            mock_context.user_id = test_user_id
            expected_user_id = test_user_id
        else:
            mock_context = Mock()
            mock_context.user_id = None
            expected_user_id = None

        with patch("nat.builder.context.Context") as mock_context_class:
            mock_context_class.get.return_value = mock_context

            with patch_services(mock_a365_service, mock_mcp_client_function_group):
                async with a365_mcp_tooling_function_group(base_config, mock_builder):
                    mock_auth_provider.authenticate.assert_called_once()
                    call_kwargs = mock_auth_provider.authenticate.call_args.kwargs
                    assert call_kwargs["user_id"] == expected_user_id


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_servers_list_returns_empty_group(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_client_function_group,
        base_config,
    ):
        """Test that empty servers list returns empty composite group."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))]
        )
        mock_a365_service.list_tool_servers.return_value = []

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            async with a365_mcp_tooling_function_group(base_config, mock_builder) as result:
                assert isinstance(result, A365MCPToolingFunctionGroup)
                all_functions = await result.get_all_functions()
                assert len(all_functions) == 0

    @pytest.mark.asyncio
    async def test_server_without_url_is_skipped(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_client_function_group,
        base_config,
    ):
        """Test that servers without URLs are skipped."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))]
        )

        server_with_url = Mock()
        server_with_url.mcp_server_name = "server-with-url"
        server_with_url.url = "https://mcp-server.example.com"

        server_without_url = Mock()
        server_without_url.mcp_server_name = "server-without-url"
        server_without_url.url = None

        mock_a365_service.list_tool_servers.return_value = [server_with_url, server_without_url]

        with patch_services(mock_a365_service, mock_mcp_client_function_group) as mock_mcp_patched:
            async with a365_mcp_tooling_function_group(base_config, mock_builder):
                assert mock_mcp_patched.call_count == 1
                call_args = mock_mcp_patched.call_args_list[0][0][0]
                # URL is a Pydantic HttpUrl object, which normalizes URLs (adds trailing slash)
                url_str = str(call_args.server.url).rstrip("/")
                assert url_str == "https://mcp-server.example.com"

    @pytest.mark.asyncio
    async def test_mcp_client_registration_failure_is_handled(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        base_config,
        mock_mcp_function_group,
    ):
        """Test that MCP client registration failures are handled gracefully."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))]
        )
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        call_count = {"value": 0}

        async def mock_mcp_group_generator(*args, **kwargs):
            """Mock that fails first time, succeeds second time."""
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise Exception("Connection failed")

            group = mock_mcp_function_group("server-2", ["tool3"])
            yield group

        def mock_factory(*args, **kwargs):
            return asynccontextmanager(mock_mcp_group_generator)(*args, **kwargs)

        import nat.plugins.a365.tooling
        import nat.plugins.mcp.client.client_impl

        mock_service_class = Mock(return_value=mock_a365_service)
        with patch.object(nat.plugins.a365.tooling, "A365ToolingService", new=mock_service_class):
            with patch.object(
                nat.plugins.mcp.client.client_impl, "mcp_client_function_group", side_effect=mock_factory
            ):
                async with a365_mcp_tooling_function_group(base_config, mock_builder) as result:
                    assert isinstance(result, A365MCPToolingFunctionGroup)
                    all_functions = await result.get_all_functions()
                    # Should only have tools from successful server
                    assert len(all_functions) == 1
                    assert "mcp_client__tool3" in all_functions

    @pytest.mark.asyncio
    async def test_tool_overrides_conversion(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
    ):
        """Test that tool_overrides dict is converted to MCPToolOverrideConfig."""
        from nat.plugins.mcp.client.client_config import MCPClientConfig, MCPToolOverrideConfig

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))]
        )
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token=AuthenticationRef("test_auth"),
            tool_overrides={
                "calculator_add": {
                    "alias": "add_numbers",
                    "description": "Add two numbers together",
                },
                "calculator_subtract": {
                    "alias": "subtract_numbers",
                },
            },
        )

        with patch_services(mock_a365_service, mock_mcp_client_function_group) as mock_mcp_patched:
            async with a365_mcp_tooling_function_group(config, mock_builder):
                assert mock_mcp_patched.call_count == 2
                for call in mock_mcp_patched.call_args_list:
                    mcp_config: MCPClientConfig = call[0][0]
                    assert mcp_config.tool_overrides is not None
                    assert "calculator_add" in mcp_config.tool_overrides
                    assert "calculator_subtract" in mcp_config.tool_overrides

                    add_override = mcp_config.tool_overrides["calculator_add"]
                    assert isinstance(add_override, MCPToolOverrideConfig)
                    assert add_override.alias == "add_numbers"
                    assert add_override.description == "Add two numbers together"
