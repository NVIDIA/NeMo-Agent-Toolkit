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

import httpx

from aiq.authentication.api_key.api_key_config import APIKeyConfig
from aiq.authentication.api_key.api_key_manager import APIKeyManager
from aiq.authentication.interfaces import AUTHORIZATION_HEADER
from aiq.data_models.authentication import HeaderAuthScheme


async def test_api_key_manager_creation():
    """Test creating API key manager with configuration"""
    config = APIKeyConfig(raw_key="test_api_key_12345", header_name="X-API-Key", header_prefix="Bearer")

    manager = APIKeyManager(config=config, config_name="test_api_key")

    assert manager is not None
    assert manager.config_name == "test_api_key"
    assert manager._config == config


async def test_api_key_header_construction_x_api_key():
    """Test API key header construction with X-API-Key scheme"""
    config = APIKeyConfig(raw_key="test_api_key_12345",
                          header_name="CUSTOM-HEADER-NAME",
                          header_prefix="CUSTOM-HEADER-PREFIX")

    manager = APIKeyManager(config=config, config_name="test_api_key")

    # Test header construction with X-API-Key scheme
    headers: httpx.Headers | None = await manager.construct_authentication_header(HeaderAuthScheme.X_API_KEY)

    assert headers is not None
    assert HeaderAuthScheme.X_API_KEY.value in headers
    assert headers[HeaderAuthScheme.X_API_KEY.value] == "test_api_key_12345"


async def test_api_key_header_construction_bearer():
    """Test API key header construction with Bearer scheme"""
    config = APIKeyConfig(raw_key="bearer_token_12345",
                          header_name="CUSTOM-HEADER-NAME",
                          header_prefix="CUSTOM-HEADER-PREFIX")

    manager = APIKeyManager(config=config, config_name="test_bearer")

    # Test header construction with Bearer scheme
    headers: httpx.Headers | None = await manager.construct_authentication_header(HeaderAuthScheme.BEARER)

    assert headers is not None
    assert AUTHORIZATION_HEADER in headers
    assert HeaderAuthScheme.BEARER.value in headers[AUTHORIZATION_HEADER]
    assert "bearer_token_12345" in headers[AUTHORIZATION_HEADER]


async def test_api_key_header_construction_basic():
    """Test API key manager with basic authentication"""
    username = "testuser"
    password = "testpass123456"
    config = APIKeyConfig(raw_key="dummy_key_12345",
                          username=username,
                          password=password,
                          header_name="Authorization",
                          header_prefix="Basic")

    manager = APIKeyManager(config=config, config_name="test_basic")

    # Test header construction with Basic scheme
    headers = await manager.construct_authentication_header(HeaderAuthScheme.BASIC)

    token_key: str = f"{username}:{password}"
    encoded_key: str = base64.b64encode(token_key.encode("utf-8")).decode("utf-8")

    assert headers is not None
    assert AUTHORIZATION_HEADER in headers
    assert "Basic" in headers[AUTHORIZATION_HEADER]
    assert headers[AUTHORIZATION_HEADER] == f"Basic {encoded_key}"

    decoded_cred = base64.b64decode(encoded_key.encode("utf-8")).decode("utf-8")

    decoded_username, decoded_password = decoded_cred.split(":", 1)

    assert decoded_username == username
    assert decoded_password == password


async def test_api_key_header_construction_custom():
    """Test API key header construction with custom scheme"""
    config = APIKeyConfig(raw_key="custom_api_key_67890",
                          header_name="CUSTOM-HEADER-NAME",
                          header_prefix="CUSTOM-HEADER-PREFIX")

    manager = APIKeyManager(config=config, config_name="test_custom")

    # Test header construction with custom scheme
    headers: httpx.Headers | None = await manager.construct_authentication_header(HeaderAuthScheme.CUSTOM)

    assert headers is not None
    assert "CUSTOM-HEADER-NAME" in headers
    assert "CUSTOM-HEADER-PREFIX" in headers["CUSTOM-HEADER-NAME"]
    assert "custom_api_key_67890" in headers["CUSTOM-HEADER-NAME"]


async def test_api_key_validation_valid():
    """Test API key validation with valid key"""
    config = APIKeyConfig(raw_key="test_api_key_12345", header_name="X-API-Key", header_prefix="Bearer")

    manager = APIKeyManager(config=config, config_name="test_api_key")

    # Test credential validation
    is_valid = await manager.validate_credentials()

    # API key should be valid.
    assert is_valid is True


async def test_api_key_validation_invalid():
    """Test API key validation with valid key"""
    config = APIKeyConfig(raw_key="test_api_key_12345", header_name="X-API-Key", header_prefix="Bearer")

    manager: APIKeyManager = APIKeyManager(config=config, config_name="test_api_key")

    manager._config.raw_key = ""

    # Test credential validation
    is_valid = await manager.validate_credentials()

    # API key should not be valid.
    assert is_valid is False
