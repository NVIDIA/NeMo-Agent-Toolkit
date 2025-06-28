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

# ========== API_KEY VALIDATION ==========


def test_api_key_valid():
    """Test valid api_key field validation."""

    valid_api_keys = [
        "test_api_key_12345",  # Standard API key (18 chars)
        "1234567890123456",  # Exactly 16 characters (minimum)
        "a" * 32,  # Long API key (32 chars)
        "sk-1234567890abcdef1234567890abcdef",  # OpenAI-style key (35 chars)
        "AIzaSyDummy_Api_Key_1234567890123456",  # Google-style key (39 chars)
        "Bearer_token_123456789012345678901234",  # Bearer-style token (36 chars)
        "ghp_1234567890abcdef1234567890abcdef12345678",  # GitHub-style token (40 chars)
    ]

    for api_key in valid_api_keys:
        config = APIKeyConfig(api_key=api_key, header_name="Authorization", header_prefix="Bearer")
        assert config.api_key == api_key


def test_api_key_invalid():
    """Test invalid api_key field invalidation."""
    from aiq.authentication.exceptions.api_key_exceptions import APIKeyFieldError

    invalid_api_keys = [
        "",  # Empty key (value_missing)
        "short",  # Too short - 5 chars (too_short)
        "1234567890123",  # Too short - 13 chars (too_short)
        "  valid_api_key_123456  ",  # Leading/trailing whitespace (whitespace_found)
        " api_key_123456",  # Leading whitespace (whitespace_found)
        "api_key_123456 ",  # Trailing whitespace (whitespace_found)
        "api key with spaces 123456",  # Internal whitespace (whitespace_found)
    ]

    for api_key in invalid_api_keys:
        with pytest.raises(APIKeyFieldError):
            APIKeyConfig(api_key=api_key, header_name="Authorization", header_prefix="Bearer")


# ========== HEADER_NAME VALIDATION ==========


def test_header_name_valid():
    """Test valid header_name field validation."""

    valid_header_names = [
        "Authorization",  # Standard HTTP header
        "X-API-Key",  # Custom header with hyphens
        "Content-Type",  # Standard header with hyphen
        "User-Agent",  # Standard user agent header
        "X-Custom-Header-123",  # Custom header with numbers
        "APIKey",  # Simple alphanumeric header
        "X-Auth-Token",  # Authentication token header
        "X-RapidAPI-Key",  # RapidAPI style header
        "Ocp-Apim-Subscription-Key",  # Azure API Management style
    ]

    for header_name in valid_header_names:
        config = APIKeyConfig(api_key="test_api_key_123456", header_name=header_name, header_prefix="Bearer")
        assert config.header_name == header_name


def test_header_name_invalid():
    """Test invalid header_name field invalidation."""
    from aiq.authentication.exceptions.api_key_exceptions import HeaderNameFieldError

    invalid_header_names = [
        "",  # Empty header name (value_missing)
        "Invalid Header",  # Contains space (invalid_format)
        "Header@Name",  # Contains @ symbol (invalid_format)
        "Header_Name!",  # Contains exclamation mark (invalid_format)
        "Header.Name",  # Contains dot (invalid_format)
        "Header:Name",  # Contains colon (invalid_format)
        "Header/Name",  # Contains slash (invalid_format)
        "Header=Name",  # Contains equals (invalid_format)
        "Header?Name",  # Contains question mark (invalid_format)
        "Header[Name]",  # Contains brackets (invalid_format)
        "Header{Name}",  # Contains braces (invalid_format)
        "Header(Name)",  # Contains parentheses (invalid_format)
        "Héader-Name",  # Contains non-ASCII character (invalid_format)
        "Header\tName",  # Contains tab (invalid_format)
        "Header\nName",  # Contains newline (invalid_format)
    ]

    for header_name in invalid_header_names:
        with pytest.raises(HeaderNameFieldError):
            APIKeyConfig(api_key="test_api_key_123456", header_name=header_name, header_prefix="Bearer")


# ========== HEADER_PREFIX VALIDATION ==========


def test_header_prefix_valid():
    """Test valid header_prefix field validation."""

    valid_header_prefixes = [
        "Bearer",  # Standard Bearer token
        "Basic",  # Basic authentication
        "Token",  # Generic token prefix
        "JWT",  # JSON Web Token
        "API",  # API prefix
        "Key",  # Simple key prefix
        "OAuth",  # OAuth prefix
        "Bot",  # Bot token prefix (Discord style)
        "SSWS",  # Okta style prefix
        "sk",  # Short prefix (Stripe style)
        "pk",  # Public key prefix
        "APIKEY",  # All caps API key prefix
        "Bearer123",  # Bearer with numbers
        "Token456",  # Token with numbers
    ]

    for header_prefix in valid_header_prefixes:
        config = APIKeyConfig(api_key="test_api_key_123456", header_name="Authorization", header_prefix=header_prefix)
        assert config.header_prefix == header_prefix


def test_header_prefix_invalid():
    """Test invalid header_prefix field invalidation."""
    from aiq.authentication.exceptions.api_key_exceptions import HeaderPrefixFieldError

    invalid_header_prefixes = [
        "",  # Empty prefix (value_missing)
        "Bearer Token",  # Contains space (contains_whitespace)
        "Basic Auth",  # Contains space (contains_whitespace)
        "API Key",  # Contains space (contains_whitespace)
        "Bearer\tToken",  # Contains tab (contains_whitespace)
        "Bearer\nToken",  # Contains newline (contains_whitespace)
        "Bearer-Token",  # Contains hyphen (invalid_format)
        "Bearer_Token",  # Contains underscore (invalid_format)
        "Bearer.Token",  # Contains dot (invalid_format)
        "Bearer@Token",  # Contains @ symbol (invalid_format)
        "Bearer#Token",  # Contains hash (invalid_format)
        "Bearer$Token",  # Contains dollar sign (invalid_format)
        "Bearer%Token",  # Contains percent (invalid_format)
        "Bearer&Token",  # Contains ampersand (invalid_format)
        "Bearer*Token",  # Contains asterisk (invalid_format)
        "Bearer+Token",  # Contains plus (invalid_format)
        "Bearer=Token",  # Contains equals (invalid_format)
        "Bearer!Token",  # Contains exclamation (invalid_format)
        "Bearer?Token",  # Contains question mark (invalid_format)
        "Bearer/Token",  # Contains slash (invalid_format)
        "Bearer\\Token",  # Contains backslash (invalid_format)
        "Bearer|Token",  # Contains pipe (invalid_format)
        "Bearer:Token",  # Contains colon (invalid_format)
        "Bearer;Token",  # Contains semicolon (invalid_format)
        "Bearer<Token",  # Contains less than (invalid_format)
        "Bearer>Token",  # Contains greater than (invalid_format)
        "Bearer[Token]",  # Contains brackets (invalid_format)
        "Bearer{Token}",  # Contains braces (invalid_format)
        "Bearer(Token)",  # Contains parentheses (invalid_format)
        "Bearer\"Token\"",  # Contains quotes (invalid_format)
        "Bearer'Token'",  # Contains single quotes (invalid_format)
        "Bearer`Token`",  # Contains backticks (invalid_format)
        "Bearer~Token",  # Contains tilde (invalid_format)
    ]

    for header_prefix in invalid_header_prefixes:
        with pytest.raises(HeaderPrefixFieldError):
            APIKeyConfig(api_key="test_api_key_123456", header_name="Authorization", header_prefix=header_prefix)
