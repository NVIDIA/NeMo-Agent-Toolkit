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

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest
from oauth2_mock_server import OAuth2Flow
from oauth2_mock_server import OAuth2FlowTester

from aiq.authentication.oauth2.auth_code_grant_client import AuthCodeGrantClient
from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig


@pytest.fixture
def auth_code_grant_config():
    """Create a test AuthCodeGrantConfig instance."""
    return AuthCodeGrantConfig(client_server_url="https://test.com",
                               authorization_url="https://test.com/auth",
                               authorization_token_url="https://test.com/token",
                               consent_prompt_key="test_key_secure",
                               client_secret="test_secret_secure_16_chars_minimum",
                               client_id="test_client",
                               audience="test_audience",
                               scope=["test_scope"])


@pytest.fixture
def auth_code_grant_client(auth_code_grant_config):
    """Create a test AuthCodeGrantClient instance."""
    return AuthCodeGrantClient(config=auth_code_grant_config, config_name="test_config")


# ========== CREDENTIALS VALIDATION ==========


async def test_auth_code_grant_valid_credentials(auth_code_grant_client: AuthCodeGrantClient):
    """Test Auth Code Grant Flow credentials."""
    # Update the config with valid credentials
    auth_code_grant_client._config.access_token = "valid_token"
    auth_code_grant_client._config.access_token_expires_in = (datetime.now(timezone.utc) + timedelta(hours=1))

    # Return True if access token is present and not expired.
    result = await auth_code_grant_client.validate_credentials()
    assert result is True


async def test_auth_code_grant_credentials_expired(auth_code_grant_client: AuthCodeGrantClient):
    """Test Auth Code Grant Flow credentials with expired access token."""
    # Update the config with expired credentials
    auth_code_grant_client._config.access_token = "expired_token"
    auth_code_grant_client._config.access_token_expires_in = (datetime.now(timezone.utc) - timedelta(hours=1))

    # Return False if access token is expired.
    result = await auth_code_grant_client.validate_credentials()
    assert result is False


async def test_auth_code_grant_no_access_token(auth_code_grant_client: AuthCodeGrantClient):
    """Test Auth Code Grant Flow credentials without access token."""
    # Ensure no access token is set
    auth_code_grant_client._config.access_token = None

    # Return False if access token is missing.
    result = await auth_code_grant_client.validate_credentials()
    assert result is False


# ========== OAUTH2 FLOW TESTING ==========


async def test_oauth2_full_flow():
    """Test the complete OAuth2 authorization code flow."""
    # Create OAuth2FlowTester instance with a minimal config and client for testing
    minimal_config = AuthCodeGrantConfig(client_server_url="https://test.com",
                                         authorization_url="https://test.com/auth",
                                         authorization_token_url="https://test.com/token",
                                         consent_prompt_key="test_key_secure",
                                         client_secret="test_secret_secure_16_chars_minimum",
                                         client_id="test_client",
                                         audience="test_audience",
                                         scope=["test_scope"])

    auth_client = AuthCodeGrantClient(config=minimal_config, config_name="test_config")

    tester = OAuth2FlowTester(oauth_client=auth_client, flow=OAuth2Flow.AUTHORIZATION_CODE)

    # Run the complete OAuth2 flow test suite
    assert await tester.run() is True
