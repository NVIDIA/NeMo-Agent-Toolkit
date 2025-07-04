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

import pytest

from aiq.authentication.api_key.api_key_config import APIKeyConfig
from aiq.authentication.exceptions.api_key_exceptions import APIKeyFieldError
from aiq.authentication.exceptions.api_key_exceptions import HeaderNameFieldError
from aiq.authentication.exceptions.api_key_exceptions import HeaderPrefixFieldError


async def test_api_key_config_creation():
    """Test creating API key configuration with valid parameters"""
    config = APIKeyConfig(raw_key="test_api_key_12345", header_name="X-API-Key", header_prefix="Bearer")

    assert config.raw_key == "test_api_key_12345"
    assert config.header_name == "X-API-Key"
    assert config.header_prefix == "Bearer"


@pytest.mark.parametrize(
    "valid_raw_key",
    [
        "abc12345",  # simple alphanumeric
        "TOKEN_ABC123XYZ",  # uppercase with underscore
        "apiKey-9999-8888",  # mixed with dashes
        "a1b2c3d4e5",  # minimum 10 characters
        "SuperSecureKey99",  # camelCase
        "token1234567890",  # long numeric key
        "K3yWithSymbols!@",  # symbols are allowed unless restricted explicitly
    ])
async def test_raw_key_field_validation(valid_raw_key):
    """Test valid raw_key values"""
    # Should not raise APIKeyFieldError
    APIKeyConfig(raw_key=valid_raw_key, header_name="X-API-Key", header_prefix="Bearer")


@pytest.mark.parametrize(
    "invalid_raw_key",
    [
        "",  # empty string
        "  ",  # whitespace only
        " abc12345",  # leading space
        "abc12345 ",  # trailing space
        "\tabc12345",  # tab character (leading)
        "abc\n12345",  # contains newline
        "abc",  # too short (less than 8 chars)
        "validKey\t",  # trailing tab
    ])
async def test_raw_key_field_invalidation(invalid_raw_key):
    """Test API key configuration validation with various invalid raw_key inputs."""
    # Should raise APIKeyFieldError
    with pytest.raises(APIKeyFieldError):
        APIKeyConfig(raw_key=invalid_raw_key, header_name="X-API-Key", header_prefix="Bearer")


@pytest.mark.parametrize(
    "valid_header_name",
    [
        "X-API-Key",  # standard API key header
        "Authorization",  # standard auth header
        "X-Auth-Token",  # custom auth token
        "X-Custom-Header",  # custom header
        "API-Key",  # simple API key
        "Bearer-Token",  # bearer token header
        "Content-Type",  # standard content type header
        "X-Request-ID",  # request ID header
    ])
async def test_header_name_field_validation(valid_header_name):
    """Test valid header_name values"""
    # Should not raise HeaderNameFieldError
    APIKeyConfig(raw_key="test12345", header_name=valid_header_name, header_prefix="Bearer")


@pytest.mark.parametrize(
    "invalid_header_name",
    [
        "",  # empty string
        "  ",  # whitespace only
        " X-API-Key",  # leading space
        "X-API-Key ",  # trailing space
        "\tX-API-Key",  # tab character (leading)
        "X-API\nKey",  # contains newline
        "X-API-Key\t",  # trailing tab
        "X API Key"  # spaces in middle
    ])
async def test_header_name_field_invalidation(invalid_header_name):
    """Test header_name field validation with various invalid inputs."""
    # Should raise HeaderNameFieldError
    with pytest.raises(HeaderNameFieldError):
        APIKeyConfig(raw_key="test12345", header_name=invalid_header_name, header_prefix="Bearer")


@pytest.mark.parametrize(
    "valid_header_prefix",
    [
        "Bearer",  # standard bearer prefix
        "Token",  # simple token prefix
        "API-Key",  # API key prefix
        "Basic",  # basic auth prefix
        "Custom",  # custom prefix
        "JWT",  # JWT token prefix
        "Key",  # simple key prefix
        "Auth"  # auth prefix
    ])
async def test_header_prefix_field_validation(valid_header_prefix):
    """Test valid header_prefix values"""
    # Should not raise HeaderPrefixFieldError
    APIKeyConfig(raw_key="test12345", header_name="X-API-Key", header_prefix=valid_header_prefix)


@pytest.mark.parametrize(
    "invalid_header_prefix",
    [
        "",  # empty string
        "  ",  # whitespace only
        " Bearer",  # leading space
        "Bearer ",  # trailing space
        "\tBearer",  # tab character (leading)
        "Bear\ner",  # contains newline
        "Bearer\t",  # trailing tab
        "Bear er"  # spaces in middle
    ])
async def test_header_prefix_field_invalidation(invalid_header_prefix):
    """Test header_prefix field validation with various invalid inputs."""
    # Should raise HeaderPrefixFieldError
    with pytest.raises(HeaderPrefixFieldError):
        APIKeyConfig(raw_key="test12345", header_name="X-API-Key", header_prefix=invalid_header_prefix)
