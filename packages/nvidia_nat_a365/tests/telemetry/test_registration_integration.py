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

"""End-to-end integration tests for A365 telemetry exporter registration."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.data_models.authentication import AuthResult, BearerTokenCred
from nat.data_models.component_ref import AuthenticationRef
from nat.plugins.a365.exceptions import A365AuthenticationError
from nat.plugins.a365.telemetry.register import a365_telemetry_exporter
from nat.plugins.a365.telemetry.register import A365TelemetryExporter


class TestRegistrationIntegration:
    """End-to-end tests for a365_telemetry_exporter registration function."""

    @pytest.fixture
    def mock_auth_provider(self):
        """Create a mock auth provider."""
        provider = Mock()
        provider.authenticate = AsyncMock()
        return provider

    @pytest.fixture
    def mock_builder(self, mock_auth_provider):
        """Create a mock builder."""
        builder = Mock(spec=Builder)
        builder.get_auth_provider = AsyncMock(return_value=mock_auth_provider)
        return builder

    @pytest.fixture
    def config(self):
        """Create A365TelemetryExporter config."""
        return A365TelemetryExporter(
            agent_id="test-agent-123",
            tenant_id="test-tenant-456",
            token_resolver=AuthenticationRef("test_auth"),
        )

    @pytest.mark.asyncio
    async def test_registration_creates_exporter_successfully(
        self, config, mock_builder, mock_auth_provider
    ):
        """Test that registration function creates exporter with correct config."""
        from pydantic import SecretStr

        # Setup auth provider to return valid credentials
        cred = BearerTokenCred(token=SecretStr("test_token_123"))
        auth_result = Mock(spec=AuthResult)
        auth_result.credentials = [cred]
        auth_result.token_expires_at = None
        mock_auth_provider.authenticate.return_value = auth_result

        with patch("nat.builder.context.Context") as mock_context_class:
            mock_context = Mock(spec=Context)
            mock_context.user_id = "test_user"
            mock_context_class.get.return_value = mock_context

            with patch("nat.plugins.a365.telemetry.a365_exporter.Agent365Exporter"):
                async with a365_telemetry_exporter(config, mock_builder) as exporter:
                    # Verify exporter was created
                    assert exporter is not None
                    assert exporter._agent_id == "test-agent-123"
                    assert exporter._tenant_id == "test-tenant-456"
                    assert exporter._token_resolver is not None
                    assert exporter._auth_provider == mock_auth_provider
                    assert exporter._token_cache is not None

                    # Verify auth provider was resolved
                    mock_builder.get_auth_provider.assert_called_once_with(config.token_resolver)

    @pytest.mark.asyncio
    async def test_registration_passes_all_config_to_exporter(
        self, config, mock_builder, mock_auth_provider
    ):
        """Test that all config fields are passed to exporter."""
        from pydantic import SecretStr

        config.cluster_category = "dev"
        config.use_s2s_endpoint = True
        config.suppress_invoke_agent_input = True
        config.batch_size = 200
        config.flush_interval = 10.0

        cred = BearerTokenCred(token=SecretStr("test_token"))
        auth_result = Mock(spec=AuthResult)
        auth_result.credentials = [cred]
        auth_result.token_expires_at = None
        mock_auth_provider.authenticate.return_value = auth_result

        with patch("nat.builder.context.Context") as mock_context_class:
            mock_context_class.get.return_value.user_id = "test_user"

            with patch("nat.plugins.a365.telemetry.a365_exporter.Agent365Exporter"):
                async with a365_telemetry_exporter(config, mock_builder) as exporter:
                    # Verify A365-specific config fields
                    assert exporter._cluster_category == "dev"
                    assert exporter._use_s2s_endpoint is True
                    assert exporter._suppress_invoke_agent_input is True
                    
                    # Batch config (batch_size, flush_interval) is passed to parent class
                    # and used in processors - not stored as instance attributes
                    # The fact that exporter was created successfully verifies config was valid

    @pytest.mark.asyncio
    async def test_registration_handles_auth_provider_resolution_failure(
        self, config, mock_builder
    ):
        """Test that registration handles auth provider resolution failure."""
        mock_builder.get_auth_provider.side_effect = ValueError("Auth provider not found")

        with pytest.raises(ValueError, match="Auth provider not found"):
            async with a365_telemetry_exporter(config, mock_builder):
                pass

    @pytest.mark.asyncio
    async def test_registration_handles_authentication_failure(
        self, config, mock_builder, mock_auth_provider
    ):
        """Test that registration handles authentication failure."""
        mock_auth_provider.authenticate.side_effect = A365AuthenticationError("Auth failed")

        with patch("nat.builder.context.Context") as mock_context_class:
            mock_context_class.get.return_value.user_id = "test_user"

            with pytest.raises(A365AuthenticationError, match="Auth failed"):
                async with a365_telemetry_exporter(config, mock_builder):
                    pass

    @pytest.mark.asyncio
    async def test_registration_handles_no_credentials(
        self, config, mock_builder, mock_auth_provider
    ):
        """Test that registration handles case when auth provider returns no credentials."""
        auth_result = Mock(spec=AuthResult)
        auth_result.credentials = []
        mock_auth_provider.authenticate.return_value = auth_result

        with patch("nat.builder.context.Context") as mock_context_class:
            mock_context_class.get.return_value.user_id = "test_user"

            with pytest.raises(A365AuthenticationError, match="No credentials available"):
                async with a365_telemetry_exporter(config, mock_builder):
                    pass

    @pytest.mark.asyncio
    async def test_registration_is_context_manager(self, config, mock_builder, mock_auth_provider):
        """Test that registration function works as async context manager."""
        from pydantic import SecretStr

        cred = BearerTokenCred(token=SecretStr("test_token"))
        auth_result = Mock(spec=AuthResult)
        auth_result.credentials = [cred]
        auth_result.token_expires_at = None
        mock_auth_provider.authenticate.return_value = auth_result

        with patch("nat.builder.context.Context") as mock_context_class:
            mock_context_class.get.return_value.user_id = "test_user"

            with patch("nat.plugins.a365.telemetry.a365_exporter.Agent365Exporter"):
                # Should work as context manager
                async with a365_telemetry_exporter(config, mock_builder) as exporter:
                    assert exporter is not None

                # After context exit, exporter should be cleaned up
                # (actual cleanup is handled by A365OtelExporter's context manager)
