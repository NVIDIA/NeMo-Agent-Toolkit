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

import base64

from aiq.authentication.api_key.api_key_client import APIKeyClient
from aiq.authentication.api_key.api_key_config import APIKeyConfig
from aiq.authentication.interfaces import AUTHORIZATION_HEADER
from aiq.data_models.authentication import AuthenticatedContext
from aiq.data_models.authentication import CredentialLocation
from aiq.data_models.authentication import HeaderAuthScheme


async def test_api_key_client_creation():
    """Test creating API key client with configuration"""
    config = APIKeyConfig(raw_key="test_api_key_12345", header_name="X-API-Key", header_prefix="Bearer")

    client = APIKeyClient(config=config, config_name="test_api_key")

    assert client is not None
    assert client.config_name == "test_api_key"
    assert client._config == config


async def test_api_key_header_construction_x_api_key():
    """Test API key header construction with X-API-Key scheme"""
    config = APIKeyConfig(raw_key="test_api_key_12345",
                          header_name="CUSTOM-HEADER-NAME",
                          header_prefix="CUSTOM-HEADER-PREFIX")

    client = APIKeyClient(config=config, config_name="test_api_key")

    # Test header construction with X-API-Key scheme
    authentication_context: AuthenticatedContext | None = await client.construct_authentication_context(
        credential_location=CredentialLocation.HEADER, header_scheme=HeaderAuthScheme.X_API_KEY)

    assert authentication_context is not None
    assert authentication_context.headers is not None
    assert HeaderAuthScheme.X_API_KEY.value.lower() in authentication_context.headers
    assert authentication_context.headers[HeaderAuthScheme.X_API_KEY.value.lower()] == "test_api_key_12345"


async def test_api_key_header_construction_bearer():
    """Test API key header construction with Bearer scheme"""
    config = APIKeyConfig(raw_key="bearer_token_12345",
                          header_name="CUSTOM-HEADER-NAME",
                          header_prefix="CUSTOM-HEADER-PREFIX")

    client = APIKeyClient(config=config, config_name="test_bearer")

    # Test header construction with Bearer scheme
    authentication_context: AuthenticatedContext | None = await client.construct_authentication_context(
        credential_location=CredentialLocation.HEADER, header_scheme=HeaderAuthScheme.BEARER)

    assert authentication_context is not None
    assert authentication_context.headers is not None
    assert AUTHORIZATION_HEADER.lower() in authentication_context.headers
    assert HeaderAuthScheme.BEARER.value in authentication_context.headers[AUTHORIZATION_HEADER.lower()]
    assert "bearer_token_12345" in authentication_context.headers[AUTHORIZATION_HEADER.lower()]


async def test_api_key_header_construction_basic():
    """Test API key client with basic authentication"""
    username = "testuser"
    password = "testpass123456"
    config = APIKeyConfig(raw_key="dummy_key_12345",
                          username=username,
                          password=password,
                          header_name="Authorization",
                          header_prefix="Basic")

    client = APIKeyClient(config=config, config_name="test_basic")

    # Test header construction with Basic scheme
    authentication_context: AuthenticatedContext | None = await client.construct_authentication_context(
        credential_location=CredentialLocation.HEADER, header_scheme=HeaderAuthScheme.BASIC)

    token_key: str = f"{username}:{password}"
    encoded_key: str = base64.b64encode(token_key.encode("utf-8")).decode("utf-8")

    assert authentication_context is not None
    assert authentication_context.headers is not None
    assert AUTHORIZATION_HEADER.lower() in authentication_context.headers
    assert "Basic" in authentication_context.headers[AUTHORIZATION_HEADER.lower()]
    assert authentication_context.headers[AUTHORIZATION_HEADER.lower()] == f"Basic {encoded_key}"

    decoded_cred = base64.b64decode(encoded_key.encode("utf-8")).decode("utf-8")

    decoded_username, decoded_password = decoded_cred.split(":", 1)

    assert decoded_username == username
    assert decoded_password == password


async def test_api_key_header_construction_custom():
    """Test API key header construction with custom scheme"""
    config = APIKeyConfig(raw_key="custom_api_key_67890",
                          header_name="CUSTOM-HEADER-NAME",
                          header_prefix="CUSTOM-HEADER-PREFIX")

    client = APIKeyClient(config=config, config_name="test_custom")

    # Test header construction with custom scheme
    authentication_context: AuthenticatedContext | None = await client.construct_authentication_context(
        credential_location=CredentialLocation.HEADER, header_scheme=HeaderAuthScheme.CUSTOM)

    assert authentication_context is not None
    assert authentication_context.headers is not None
    assert "CUSTOM-HEADER-NAME".lower() in authentication_context.headers
    assert "CUSTOM-HEADER-PREFIX" in authentication_context.headers["CUSTOM-HEADER-NAME".lower()]
    assert "custom_api_key_67890" in authentication_context.headers["CUSTOM-HEADER-NAME".lower()]


async def test_api_key_validation_valid():
    """Test API key validation with valid key"""
    config = APIKeyConfig(raw_key="test_api_key_12345", header_name="X-API-Key", header_prefix="Bearer")

    client = APIKeyClient(config=config, config_name="test_api_key")

    # Test credential validation
    is_valid = await client.validate_credentials()

    # API key should be valid.
    assert is_valid is True


async def test_api_key_validation_invalid():
    """Test API key validation with valid key"""
    config = APIKeyConfig(raw_key="test_api_key_12345", header_name="X-API-Key", header_prefix="Bearer")

    client: APIKeyClient = APIKeyClient(config=config, config_name="test_api_key")

    client._config.raw_key = ""

    # Test credential validation
    is_valid = await client.validate_credentials()

    # API key should not be valid.
    assert is_valid is False


async def test_api_key_client_integration():
    """Authentication Client Integration tests."""
    from test_custom_authentication_client import AuthenticationClientTester

    client = APIKeyClient(config=APIKeyConfig(raw_key="test_api_key_12345",
                                              header_name="X-API-Key",
                                              header_prefix="Bearer"),
                          config_name="test_api_key")

    tester = AuthenticationClientTester(auth_client=client)

    # Run the complete Authentication Client integration test suite
    assert await tester.run() is True
