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

import logging
import sys
from contextlib import asynccontextmanager
from contextlib import contextmanager
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch
from uuid import uuid4

import pytest
from pydantic import SecretStr

from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import BearerTokenCred
from nat.data_models.authentication import HeaderCred
from nat.data_models.component_ref import AuthenticationRef
from nat.plugins.a365.exceptions import A365ConfigurationError
from nat.plugins.a365.tooling import A365MCPToolingConfig
from nat.plugins.a365.tooling.register import A365MCPToolingFunctionGroup
from nat.plugins.a365.tooling.register import a365_mcp_tooling_function_group

# Skip all tests on Python 3.13 until nvidia-nat-mcp supports it
pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason="nvidia-nat-mcp does not support Python 3.13 yet. These tests require MCP functionality.",
)


@pytest.fixture(name="mock_auth_provider")
def mock_auth_provider_fixture():
    """Create a mock auth provider."""
    provider = Mock()
    provider.authenticate = AsyncMock()
    return provider


@pytest.fixture(name="mock_builder")
def mock_builder_fixture(mock_auth_provider):
    """Create a mock builder with auth provider resolution."""
    builder = Mock()
    builder.get_auth_provider = AsyncMock(return_value=mock_auth_provider)
    return builder


@pytest.fixture(name="mock_a365_service")
def mock_a365_service_fixture():
    """Create a mock A365ToolingService."""
    service = Mock()
    service.list_tool_servers = AsyncMock()
    return service


@pytest.fixture(name="mock_mcp_servers")
def mock_mcp_servers_fixture():
    """Create mock MCP server configurations."""
    server1 = Mock()
    server1.mcp_server_name = "server-1"
    server1.url = "https://mcp-server-1.example.com"

    server2 = Mock()
    server2.mcp_server_name = "server-2"
    server2.url = "https://mcp-server-2.example.com"

    return [server1, server2]


@pytest.fixture(name="mock_mcp_function_group")
def mock_mcp_function_group_fixture():
    """Create a mock MCP FunctionGroup with functions."""

    def create_mock_group(server_name: str, tool_names: list[str]):
        """Create a mock function group for a specific server."""
        group = Mock(spec=FunctionGroup)
        # Functions are namespaced with the MCP group's instance name
        # For simplicity, we'll use "mcp_client" as the namespace
        functions = {
            f"mcp_client__{tool_name}":
                Mock(
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


@pytest.fixture(name="mock_mcp_client_function_group")
def mock_mcp_client_function_group_fixture(mock_mcp_function_group):
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


@pytest.fixture(name="base_config")
def base_config_fixture():
    """Base config for tests."""
    return A365MCPToolingConfig(
        agentic_app_id="test-agent",
        auth_token=AuthenticationRef("test_auth"),
    )


@contextmanager
def patch_services(mock_a365_service, mock_mcp_client_function_group):
    """Context manager to patch A365 and MCP services."""
    try:
        import nat.plugins.mcp.client.client_impl
    except ImportError:
        pytest.skip("nvidia-nat-mcp not installed, skipping MCP-dependent tests")

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
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
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


class TestTokenExtraction:
    """Test token extraction from auth provider credentials."""

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
        from nat.plugins.a365.exceptions import A365AuthenticationError

        mock_auth_provider.authenticate.return_value = AuthResult(credentials=credentials)
        # Return empty list so code can proceed past service discovery
        mock_a365_service.list_tool_servers.return_value = []

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            with pytest.raises(A365AuthenticationError, match=error_match):
                async with a365_mcp_tooling_function_group(base_config, mock_builder):
                    pass


class TestAuthProviderPriority:
    """Test auth provider priority logic for MCP servers."""

    async def test_per_server_override_takes_priority(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
    ):
        """Test that per-server override takes priority over gateway auth."""
        try:
            from nat.plugins.mcp.client.client_config import MCPClientConfig
        except ImportError:
            pytest.skip("nvidia-nat-mcp not installed")

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("gateway-token"))])
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        override_auth_provider = Mock()
        override_auth_provider.authenticate = AsyncMock()
        mock_builder.get_auth_provider = AsyncMock(
            side_effect=lambda ref: (override_auth_provider if str(ref) == "override_auth" else mock_auth_provider))

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

    async def test_gateway_auth_used_when_no_override(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
    ):
        """Test that gateway auth is used when no per-server override."""
        try:
            from nat.plugins.mcp.client.client_config import MCPClientConfig
        except ImportError:
            pytest.skip("nvidia-nat-mcp not installed")

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("gateway-token"))])
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

    async def test_string_token_no_auth_for_servers(self,
                                                    mock_builder,
                                                    mock_a365_service,
                                                    mock_mcp_servers,
                                                    mock_mcp_client_function_group):
        """Test that string token doesn't pass auth to MCP servers."""
        try:
            from nat.plugins.mcp.client.client_config import MCPClientConfig
        except ImportError:
            pytest.skip("nvidia-nat-mcp not installed")

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
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
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
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
        mock_a365_service.list_tool_servers.return_value = []

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            async with a365_mcp_tooling_function_group(base_config, mock_builder) as result:
                assert isinstance(result, A365MCPToolingFunctionGroup)
                all_functions = await result.get_all_functions()
                assert len(all_functions) == 0

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
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])

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
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
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
        try:
            import nat.plugins.mcp.client.client_impl
        except ImportError:
            pytest.skip("nvidia-nat-mcp not installed")

        mock_service_class = Mock(return_value=mock_a365_service)
        with patch.object(nat.plugins.a365.tooling, "A365ToolingService", new=mock_service_class):
            with patch.object(nat.plugins.mcp.client.client_impl, "mcp_client_function_group",
                              side_effect=mock_factory):
                async with a365_mcp_tooling_function_group(base_config, mock_builder) as result:
                    assert isinstance(result, A365MCPToolingFunctionGroup)
                    all_functions = await result.get_all_functions()
                    # Should only have tools from successful server
                    assert len(all_functions) == 1
                    assert "mcp_client__tool3" in all_functions

    async def test_tool_overrides_conversion(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
    ):
        """Test that tool_overrides dict is converted to MCPToolOverrideConfig."""
        try:
            from nat.plugins.mcp.client.client_config import MCPClientConfig
            from nat.plugins.mcp.client.client_config import MCPToolOverrideConfig
        except ImportError:
            pytest.skip("nvidia-nat-mcp not installed")

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
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

    async def test_tool_overrides_invalid_config_raises_error(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
    ):
        """Test that invalid tool_overrides raises A365ConfigurationError.

        Note: Since Pydantic validates tool_overrides at config creation time,
        we test the error handling by directly patching the conversion to raise ValidationError.
        """
        try:
            from pydantic import ValidationError

            # Presence-probe for nvidia-nat-mcp; the symbol itself is only referenced
            # below via patch("...MCPToolOverrideConfig", ...), so silence F401.
            from nat.plugins.mcp.client.client_config import MCPToolOverrideConfig  # noqa: F401
        except ImportError:
            pytest.skip("nvidia-nat-mcp not installed")

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token=AuthenticationRef("test_auth"),
            tool_overrides={
                "calculator_add": {
                    "alias": "valid_alias",
                    "description": "valid description",
                },
            },
        )

        # Patch the conversion to raise ValidationError, simulating invalid MCPToolOverrideConfig data
        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            # Patch MCPToolOverrideConfig where it's imported inside the function
            with patch("nat.plugins.mcp.client.client_config.MCPToolOverrideConfig",
                       side_effect=ValidationError.from_exception_data("MCPToolOverrideConfig",
                                                                       [{
                                                                           "type": "string_type",
                                                                           "loc": ("alias", ),
                                                                           "msg": "Input should be a valid string",
                                                                           "input": None
                                                                       }])):
                with pytest.raises(A365ConfigurationError, match="Invalid tool_overrides configuration"):
                    async with a365_mcp_tooling_function_group(config, mock_builder):
                        pass

    async def test_auth_provider_resolution_failure(
        self,
        mock_builder,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
    ):
        """Test that auth provider resolution failure is handled."""

        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        # Make get_auth_provider raise an exception
        mock_builder.get_auth_provider = AsyncMock(side_effect=Exception("Auth provider not found"))

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token=AuthenticationRef("test_auth"),
        )

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            # Should propagate the exception (or wrap it appropriately)
            # The actual behavior depends on how NAT handles this, but we should test it
            with pytest.raises(Exception, match="Auth provider not found"):
                async with a365_mcp_tooling_function_group(config, mock_builder):
                    pass

    async def test_non_auth_error_raises_a365_sdk_error(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_client_function_group,
        base_config,
    ):
        """Test that non-authentication errors from service.list_tool_servers() raise A365SDKError."""
        from nat.plugins.a365.exceptions import A365SDKError

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])

        # Make list_tool_servers raise a non-auth error (e.g., connection error)
        mock_a365_service.list_tool_servers = AsyncMock(side_effect=ConnectionError("Connection refused"))

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            with pytest.raises(A365SDKError, match="Failed to discover MCP servers"):
                async with a365_mcp_tooling_function_group(base_config, mock_builder):
                    pass

    async def test_all_config_fields_passed_to_mcp_client_config(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
    ):
        """Test that all config fields are correctly passed to MCPClientConfig."""
        try:
            from datetime import timedelta

            from nat.plugins.mcp.client.client_config import MCPClientConfig
        except ImportError:
            pytest.skip("nvidia-nat-mcp not installed")

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        # Create config with custom values for all fields
        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token=AuthenticationRef("test_auth"),
            tool_call_timeout=timedelta(seconds=90),
            auth_flow_timeout=timedelta(seconds=600),
            reconnect_enabled=False,
            reconnect_max_attempts=5,
            reconnect_initial_backoff=1.0,
            reconnect_max_backoff=100.0,
            session_aware_tools=False,
            max_sessions=200,
            session_idle_timeout=timedelta(hours=2),
        )

        with patch_services(mock_a365_service, mock_mcp_client_function_group) as mock_mcp_patched:
            async with a365_mcp_tooling_function_group(config, mock_builder):
                # Verify MCPClientConfig was created with correct values
                assert mock_mcp_patched.call_count > 0
                mcp_config: MCPClientConfig = mock_mcp_patched.call_args_list[0][0][0]

                # Verify all fields are passed correctly
                assert mcp_config.tool_call_timeout == timedelta(seconds=90)
                assert mcp_config.auth_flow_timeout == timedelta(seconds=600)
                assert mcp_config.reconnect_enabled is False
                assert mcp_config.reconnect_max_attempts == 5
                assert mcp_config.reconnect_initial_backoff == 1.0
                assert mcp_config.reconnect_max_backoff == 100.0
                assert mcp_config.session_aware_tools is False
                assert mcp_config.max_sessions == 200
                assert mcp_config.session_idle_timeout == timedelta(hours=2)


# -----------------------------------------------------------------------------
# Tests for the per-server registration error policy (C5) and the supporting
# observability surfaces: skipped_servers metadata, server_auth_providers
# unused-key warnings (M11), unnamed-server fallback names (M12), and tool-name
# collision warnings (M13). The C6 classifier is exercised via an isinstance
# branch test and a fallback-substring test.
# -----------------------------------------------------------------------------


def _make_alternating_failing_factory(mock_mcp_function_group, fail_indices: list[int]):
    """Build a mock ``mcp_client_function_group`` factory that fails on the given call indices.

    ``fail_indices`` is 1-based to match the call_count semantics already used in the file.
    """
    call_count = {"value": 0}

    async def gen(*_args, **_kwargs):
        call_count["value"] += 1
        if call_count["value"] in fail_indices:
            raise Exception(f"Simulated registration failure on call {call_count['value']}")
        group = mock_mcp_function_group(f"server-{call_count['value']}", [f"tool{call_count['value']}"])
        yield group

    def factory(*args, **kwargs):
        return asynccontextmanager(gen)(*args, **kwargs)

    return factory


class TestServerRegistrationErrorPolicy:
    """Verify the ``on_server_registration_error`` policy field behaves correctly."""

    async def test_fail_fast_raises_a365_sdk_error_on_first_server_failure(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_function_group,
    ):
        """Under ``fail_fast``, a single per-server failure aborts the whole registration."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token=AuthenticationRef("test_auth"),
            on_server_registration_error="fail_fast",
        )

        factory = _make_alternating_failing_factory(mock_mcp_function_group, fail_indices=[1])

        import nat.plugins.a365.tooling
        try:
            import nat.plugins.mcp.client.client_impl
        except ImportError:
            pytest.skip("nvidia-nat-mcp not installed")

        mock_service_class = Mock(return_value=mock_a365_service)
        with patch.object(nat.plugins.a365.tooling, "A365ToolingService", new=mock_service_class):
            with patch.object(nat.plugins.mcp.client.client_impl, "mcp_client_function_group", side_effect=factory):
                from nat.plugins.a365.exceptions import A365SDKError
                with pytest.raises(A365SDKError, match="policy=fail_fast"):
                    async with a365_mcp_tooling_function_group(config, mock_builder):
                        pass  # pragma: no cover

    async def test_skip_with_warning_continues_and_records_skipped_servers(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_function_group,
        caplog,
    ):
        """Under the default ``skip_with_warning``, failed servers are logged and recorded."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token=AuthenticationRef("test_auth"),
            # Default policy; specified explicitly to make intent clear.
            on_server_registration_error="skip_with_warning",
        )

        factory = _make_alternating_failing_factory(mock_mcp_function_group, fail_indices=[1])

        import nat.plugins.a365.tooling
        try:
            import nat.plugins.mcp.client.client_impl
        except ImportError:
            pytest.skip("nvidia-nat-mcp not installed")

        mock_service_class = Mock(return_value=mock_a365_service)
        with patch.object(nat.plugins.a365.tooling, "A365ToolingService", new=mock_service_class):
            with patch.object(nat.plugins.mcp.client.client_impl, "mcp_client_function_group", side_effect=factory):
                with caplog.at_level(logging.WARNING, logger="nat.plugins.a365.tooling.register"):
                    async with a365_mcp_tooling_function_group(config, mock_builder) as result:
                        assert isinstance(result, A365MCPToolingFunctionGroup)
                        # One server (server-1) failed and was skipped; one survived.
                        all_functions = await result.get_all_functions()
                        assert len(all_functions) == 1
                        # Skipped-server metadata exposes the failure for monitoring.
                        assert len(result.skipped_servers) == 1
                        skipped_name, skipped_err = result.skipped_servers[0]
                        assert "Simulated registration failure" in skipped_err

        # WARN log emitted for the skipped server.
        assert any("after registration failure (policy=skip_with_warning)" in r.getMessage()
                   for r in caplog.records), \
            f"Expected skip_with_warning log, got: {[r.getMessage() for r in caplog.records]}"

    async def test_skip_silently_continues_without_warning(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_function_group,
        caplog,
    ):
        """Under ``skip_silently``, failures don't produce WARN logs but still record metadata."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token=AuthenticationRef("test_auth"),
            on_server_registration_error="skip_silently",
        )

        factory = _make_alternating_failing_factory(mock_mcp_function_group, fail_indices=[1])

        import nat.plugins.a365.tooling
        try:
            import nat.plugins.mcp.client.client_impl
        except ImportError:
            pytest.skip("nvidia-nat-mcp not installed")

        mock_service_class = Mock(return_value=mock_a365_service)
        with patch.object(nat.plugins.a365.tooling, "A365ToolingService", new=mock_service_class):
            with patch.object(nat.plugins.mcp.client.client_impl, "mcp_client_function_group", side_effect=factory):
                with caplog.at_level(logging.WARNING, logger="nat.plugins.a365.tooling.register"):
                    async with a365_mcp_tooling_function_group(config, mock_builder) as result:
                        assert len(result.skipped_servers) == 1

        # No WARN log naming the skipped server (DEBUG-only under this policy).
        skip_logs = [r for r in caplog.records if "after registration failure" in r.getMessage()]
        assert not skip_logs, (f"skip_silently should not emit WARN logs, got: {[r.getMessage() for r in skip_logs]}")


class TestServerAuthProvidersHygiene:
    """M11: case-insensitive lookup + warn on unused override keys."""

    async def test_server_auth_providers_match_case_insensitively(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
    ):
        """A YAML override key with different casing still matches the discovered server name."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        # mock_mcp_servers has servers named "server-1" and "server-2". Use a mixed-case key.
        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token=AuthenticationRef("gateway_auth"),
            server_auth_providers={"Server-1": "per_server_auth"},
        )

        with patch_services(mock_a365_service, mock_mcp_client_function_group) as mock_mcp_patched:
            async with a365_mcp_tooling_function_group(config, mock_builder):
                # Two MCPClientConfigs constructed; the first one (server-1) should carry the
                # per-server override, the second (server-2) should fall back to the gateway auth.
                server1_config = mock_mcp_patched.call_args_list[0][0][0]
                server2_config = mock_mcp_patched.call_args_list[1][0][0]
                assert str(server1_config.server.auth_provider) == "per_server_auth"
                assert str(server2_config.server.auth_provider) == "gateway_auth"

    async def test_server_auth_providers_warns_on_unused_override_keys(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_servers,
        mock_mcp_client_function_group,
        caplog,
    ):
        """Override entries that don't match any discovered server should warn the operator."""
        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
        mock_a365_service.list_tool_servers.return_value = mock_mcp_servers

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token=AuthenticationRef("test_auth"),
            server_auth_providers={
                "server-1": "real_override",  # matches a discovered server
                "decommissioned-server": "stale_override",  # no match -> warn
                "another-ghost": "another_stale",  # no match -> warn
            },
        )

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            with caplog.at_level(logging.WARNING, logger="nat.plugins.a365.tooling.register"):
                async with a365_mcp_tooling_function_group(config, mock_builder):
                    pass

        unused_warnings = [r for r in caplog.records if "references unknown MCP servers" in r.getMessage()]
        assert len(unused_warnings) == 1, (
            f"Expected one unused-keys warning, got: {[r.getMessage() for r in unused_warnings]}")
        warning_msg = unused_warnings[0].getMessage()
        # Both stale keys should be named in the warning.
        assert "decommissioned-server" in warning_msg
        assert "another-ghost" in warning_msg
        # The valid key should NOT appear in the unused-keys warning.
        assert "real_override" not in warning_msg


class TestUnnamedServerFallback:
    """M12: derive a deterministic display name from the URL when ``mcp_server_name`` is absent."""

    async def test_unnamed_server_uses_url_hostname_in_logs_and_overrides(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_client_function_group,
        caplog,
    ):
        """Two unnamed servers with distinct hostnames are distinguishable in logs and overrides."""
        # Build two mock servers with no mcp_server_name and distinct hostnames.
        unnamed_a = Mock()
        unnamed_a.mcp_server_name = None
        unnamed_a.url = "https://alpha.example.com/mcp"

        unnamed_b = Mock()
        unnamed_b.mcp_server_name = None
        unnamed_b.url = "https://beta.example.com/mcp"

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])
        mock_a365_service.list_tool_servers.return_value = [unnamed_a, unnamed_b]

        # An override that targets one of the unnamed servers by its derived display name.
        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token=AuthenticationRef("gateway_auth"),
            server_auth_providers={"unknown:alpha.example.com": "alpha_specific_auth"},
        )

        with patch_services(mock_a365_service, mock_mcp_client_function_group) as mock_mcp_patched:
            with caplog.at_level(logging.INFO, logger="nat.plugins.a365.tooling.register"):
                async with a365_mcp_tooling_function_group(config, mock_builder):
                    pass

        # Verify the override targeted the right server (M11 cooperation: lookup by derived name).
        a_config = mock_mcp_patched.call_args_list[0][0][0]
        b_config = mock_mcp_patched.call_args_list[1][0][0]
        assert str(a_config.server.auth_provider) == "alpha_specific_auth"
        assert str(b_config.server.auth_provider) == "gateway_auth"

        # Logs name the servers distinctly so ops can tell them apart.
        info_logs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("unknown:alpha.example.com" in m for m in info_logs)
        assert any("unknown:beta.example.com" in m for m in info_logs)


class TestToolNameCollisionWarning:
    """M13: warn when two MCP groups expose the same function name."""

    async def test_get_all_functions_warns_on_collision(self, caplog):
        """Two MCP groups exposing the same tool name should produce a WARN; later wins."""
        from nat.plugins.a365.tooling.register import A365MCPToolingFunctionGroup

        # Two groups, each claiming a tool with the same name -- distinct underlying functions.
        fn_a = Mock(spec=Function)
        fn_a.name = "from_group_a"
        group_a = Mock(spec=FunctionGroup)
        group_a.get_all_functions = AsyncMock(return_value={"shared_tool": fn_a})

        fn_b = Mock(spec=Function)
        fn_b.name = "from_group_b"
        group_b = Mock(spec=FunctionGroup)
        group_b.get_all_functions = AsyncMock(return_value={"shared_tool": fn_b})

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token="test-token",
        )
        composite = A365MCPToolingFunctionGroup(
            config=config,
            mcp_groups=[group_a, group_b],
        )

        with caplog.at_level(logging.WARNING, logger="nat.plugins.a365.tooling.register"):
            merged = await composite.get_all_functions()

        # Later definition wins (preserves prior dict.update semantics).
        assert merged["shared_tool"] is fn_b
        # Exactly one collision warning fired, naming the offending tool name.
        collisions = [r for r in caplog.records if "Tool name collision" in r.getMessage()]
        assert len(collisions) == 1, (f"Expected one collision warning, got: {[r.getMessage() for r in collisions]}")
        assert "'shared_tool'" in collisions[0].getMessage()

    async def test_no_warning_when_same_function_instance_in_multiple_groups(self, caplog):
        """If the same Function instance appears under the same name in two groups, no warning."""
        from nat.plugins.a365.tooling.register import A365MCPToolingFunctionGroup

        shared_fn = Mock(spec=Function)
        shared_fn.name = "shared_fn"
        group_a = Mock(spec=FunctionGroup)
        group_a.get_all_functions = AsyncMock(return_value={"tool": shared_fn})
        group_b = Mock(spec=FunctionGroup)
        group_b.get_all_functions = AsyncMock(return_value={"tool": shared_fn})

        config = A365MCPToolingConfig(
            agentic_app_id="test-agent",
            auth_token="test-token",
        )
        composite = A365MCPToolingFunctionGroup(
            config=config,
            mcp_groups=[group_a, group_b],
        )

        with caplog.at_level(logging.WARNING, logger="nat.plugins.a365.tooling.register"):
            await composite.get_all_functions()

        collisions = [r for r in caplog.records if "Tool name collision" in r.getMessage()]
        assert not collisions, (
            f"Identical Function instances should not warn, got: {[r.getMessage() for r in collisions]}")


class TestDiscoveryErrorClassifier:
    """C6: classify gateway discovery errors via aiohttp.ClientResponseError status first, substring as fallback."""

    async def test_aiohttp_401_classifies_as_authentication_error(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_client_function_group,
        base_config,
    ):
        """A ``ClientResponseError(status=401)`` from the gateway raises A365AuthenticationError."""
        from aiohttp import ClientResponseError

        from nat.plugins.a365.exceptions import A365AuthenticationError

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])

        err = ClientResponseError(
            request_info=Mock(),
            history=(),
            status=401,
            message="Unauthorized",
        )
        mock_a365_service.list_tool_servers = AsyncMock(side_effect=err)

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            with pytest.raises(A365AuthenticationError, match="HTTP 401"):
                async with a365_mcp_tooling_function_group(base_config, mock_builder):
                    pass

    async def test_aiohttp_500_classifies_as_sdk_error(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_client_function_group,
        base_config,
    ):
        """A ``ClientResponseError(status=500)`` raises A365SDKError, not an auth error."""
        from aiohttp import ClientResponseError

        from nat.plugins.a365.exceptions import A365SDKError

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])

        err = ClientResponseError(
            request_info=Mock(),
            history=(),
            status=500,
            message="Internal Server Error",
        )
        mock_a365_service.list_tool_servers = AsyncMock(side_effect=err)

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            with pytest.raises(A365SDKError, match="HTTP 500"):
                async with a365_mcp_tooling_function_group(base_config, mock_builder):
                    pass

    async def test_unclassified_exception_falls_back_to_sdk_error_with_warning(
        self,
        mock_builder,
        mock_auth_provider,
        mock_a365_service,
        mock_mcp_client_function_group,
        base_config,
        caplog,
    ):
        """An unknown exception type without auth keywords falls back to A365SDKError + WARN."""
        from nat.plugins.a365.exceptions import A365SDKError

        mock_auth_provider.authenticate.return_value = AuthResult(
            credentials=[BearerTokenCred(token=SecretStr("test-token"))])

        # A custom exception type with no auth-related keywords in the message.
        class WeirdGatewayError(Exception):
            pass

        mock_a365_service.list_tool_servers = AsyncMock(side_effect=WeirdGatewayError("internal mystery"))

        with patch_services(mock_a365_service, mock_mcp_client_function_group):
            with caplog.at_level(logging.WARNING, logger="nat.plugins.a365.tooling.register"):
                with pytest.raises(A365SDKError):
                    async with a365_mcp_tooling_function_group(base_config, mock_builder):
                        pass

        # The WARN should name the unclassified exception type so ops know the classifier is missing it.
        unclassified = [r for r in caplog.records if "unclassified exception type" in r.getMessage()]
        assert len(unclassified) == 1
        assert "WeirdGatewayError" in unclassified[0].getMessage()
