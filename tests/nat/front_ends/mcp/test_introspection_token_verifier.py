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
"""Tests for IntrospectionTokenVerifier configuration parsing and validation."""

from unittest.mock import MagicMock

import pytest

from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
from nat.front_ends.mcp.introspection_token_verifier import IntrospectionTokenVerifier


class TestJwtVsOpaqueValidationCapability:
    """Test that IntrospectionTokenVerifier correctly validates JWT vs opaque token capability."""

    @pytest.mark.asyncio
    async def test_valid_jwt_capability_creates_bearer_validator(self):
        """Test config with JWT capability (issuer) creates BearerTokenValidator."""
        from nat.front_ends.mcp.mcp_front_end_plugin import MCPFrontEndPlugin

        plugin = MagicMock()
        plugin.front_end_config = MagicMock()
        plugin.front_end_config.auth_provider = True

        # Config that supports _verify_jwt_token (has issuer)
        auth_config = OAuth2AuthCodeFlowProviderConfig(
            client_id="test-client",
            client_secret="test-secret",
            authorization_url="https://auth.test/authorize",
            token_url="https://auth.test/token",
            redirect_uri="https://app.test/callback",
            authorization_kwargs={
                "issuer": "https://jwt-issuer.test"  # Enables JWT verification
            })

        factory_method = MCPFrontEndPlugin._create_token_verifier
        verifier = await factory_method(plugin, auth_config)

        # Verify JWT capability was recognized and BearerTokenValidator created
        assert isinstance(verifier, IntrospectionTokenVerifier)
        assert hasattr(verifier, '_bearer_token_validator')
        assert verifier._bearer_token_validator is not None
        assert verifier.issuer == "https://jwt-issuer.test"

    @pytest.mark.asyncio
    async def test_valid_jwt_capability_with_discovery_creates_bearer_validator(self):
        """Test config with JWT capability (discovery_url) creates BearerTokenValidator."""
        from nat.front_ends.mcp.mcp_front_end_plugin import MCPFrontEndPlugin

        plugin = MagicMock()
        plugin.front_end_config = MagicMock()
        plugin.front_end_config.auth_provider = True

        # Config that supports _verify_jwt_token (has discovery_url)
        auth_config = OAuth2AuthCodeFlowProviderConfig(
            client_id="test-client",
            client_secret="test-secret",
            authorization_url="https://auth.test/authorize",
            token_url="https://auth.test/token",
            redirect_uri="https://app.test/callback",
            authorization_kwargs={
                "discovery_url": "https://issuer.test/.well-known/openid-configuration"  # Enables JWT verification
            })

        factory_method = MCPFrontEndPlugin._create_token_verifier
        verifier = await factory_method(plugin, auth_config)

        # Verify JWT capability was recognized and BearerTokenValidator created
        assert isinstance(verifier, IntrospectionTokenVerifier)
        assert hasattr(verifier, '_bearer_token_validator')
        assert verifier._bearer_token_validator is not None
        assert verifier.discovery_url == "https://issuer.test/.well-known/openid-configuration"

    @pytest.mark.asyncio
    async def test_valid_opaque_capability_creates_bearer_validator(self):
        """Test config with opaque token capability creates BearerTokenValidator."""
        from nat.front_ends.mcp.mcp_front_end_plugin import MCPFrontEndPlugin

        plugin = MagicMock()
        plugin.front_end_config = MagicMock()
        plugin.front_end_config.auth_provider = True

        # Config that supports _verify_opaque_token (has introspection_endpoint + credentials)
        auth_config = OAuth2AuthCodeFlowProviderConfig(
            client_id="opaque-client",
            client_secret="opaque-secret",
            authorization_url="https://auth.test/authorize",
            token_url="https://auth.test/token",
            redirect_uri="https://app.test/callback",
            authorization_kwargs={
                "introspection_endpoint": "https://auth.test/introspect"  # Enables opaque verification
            })

        factory_method = MCPFrontEndPlugin._create_token_verifier
        verifier = await factory_method(plugin, auth_config)

        # Verify opaque capability was recognized and BearerTokenValidator created
        assert isinstance(verifier, IntrospectionTokenVerifier)
        assert hasattr(verifier, '_bearer_token_validator')
        assert verifier._bearer_token_validator is not None
        assert verifier.introspection_endpoint == "https://auth.test/introspect"
        assert verifier.client_id == "opaque-client"
        assert verifier.client_secret == "opaque-secret"

    @pytest.mark.asyncio
    async def test_invalid_no_verification_capability_fails_validation(self):
        """Test config with no JWT or opaque capability fails before BearerTokenValidator creation."""
        from nat.front_ends.mcp.mcp_front_end_plugin import MCPFrontEndPlugin

        plugin = MagicMock()
        plugin.front_end_config = MagicMock()
        plugin.front_end_config.auth_provider = True

        # Config that supports NEITHER _verify_jwt_token NOR _verify_opaque_token
        auth_config = OAuth2AuthCodeFlowProviderConfig(
            client_id="test-client",
            client_secret="test-secret",
            authorization_url="https://auth.test/authorize",
            token_url="https://auth.test/token",
            redirect_uri="https://app.test/callback",
            authorization_kwargs={
                "jwks_uri": "https://keys.test/jwks.json"  # Not sufficient - needs issuer or introspection_endpoint
            })

        factory_method = MCPFrontEndPlugin._create_token_verifier
        verifier = await factory_method(plugin, auth_config)

        # Verify validation failed and NO BearerTokenValidator was created
        assert isinstance(verifier, IntrospectionTokenVerifier)
        assert not hasattr(verifier, '_bearer_token_validator')
        # Config was parsed but lacks capability for either verification method

    @pytest.mark.asyncio
    async def test_invalid_opaque_missing_credentials_fails_validation(self):
        """Test opaque config missing credentials fails before BearerTokenValidator creation."""
        from nat.front_ends.mcp.mcp_front_end_plugin import MCPFrontEndPlugin

        plugin = MagicMock()
        plugin.front_end_config = MagicMock()
        plugin.front_end_config.auth_provider = True

        # Config has introspection_endpoint but missing client credentials for opaque verification
        auth_config = OAuth2AuthCodeFlowProviderConfig(
            client_id="",  # Missing required credential
            client_secret="test-secret",
            authorization_url="https://auth.test/authorize",
            token_url="https://auth.test/token",
            redirect_uri="https://app.test/callback",
            authorization_kwargs={
                "introspection_endpoint": "https://auth.test/introspect"  # Would enable opaque, but missing creds
            })

        factory_method = MCPFrontEndPlugin._create_token_verifier
        verifier = await factory_method(plugin, auth_config)

        # Verify validation failed due to missing credentials and NO BearerTokenValidator was created
        assert isinstance(verifier, IntrospectionTokenVerifier)
        assert not hasattr(verifier, '_bearer_token_validator')
        assert verifier.introspection_endpoint == "https://auth.test/introspect"  # Config was parsed
        assert verifier.client_id == ""  # But credentials insufficient for opaque verification

    @pytest.mark.asyncio
    async def test_dual_capability_config_creates_bearer_validator(self):
        """Test config with BOTH JWT and opaque capability creates BearerTokenValidator."""
        from nat.front_ends.mcp.mcp_front_end_plugin import MCPFrontEndPlugin

        plugin = MagicMock()
        plugin.front_end_config = MagicMock()
        plugin.front_end_config.auth_provider = True

        # Config that supports BOTH _verify_jwt_token AND _verify_opaque_token
        auth_config = OAuth2AuthCodeFlowProviderConfig(
            client_id="dual-client",
            client_secret="dual-secret",
            authorization_url="https://auth.test/authorize",
            token_url="https://auth.test/token",
            redirect_uri="https://app.test/callback",
            authorization_kwargs={
                "issuer": "https://jwt-issuer.test",  # Enables JWT verification
                "introspection_endpoint": "https://auth.test/introspect"  # Enables opaque verification
            })

        factory_method = MCPFrontEndPlugin._create_token_verifier
        verifier = await factory_method(plugin, auth_config)

        # Verify dual capability was recognized and BearerTokenValidator created
        assert isinstance(verifier, IntrospectionTokenVerifier)
        assert hasattr(verifier, '_bearer_token_validator')
        assert verifier._bearer_token_validator is not None
        assert verifier.issuer == "https://jwt-issuer.test"
        assert verifier.introspection_endpoint == "https://auth.test/introspect"
        # BearerTokenValidator can handle both JWT and opaque tokens with this config
