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

from aiq.authentication.exceptions.request_exceptions import BaseUrlValidationError
from aiq.authentication.exceptions.request_exceptions import HTTPHeaderValidationError
from aiq.authentication.exceptions.request_exceptions import HTTPMethodValidationError
from aiq.authentication.exceptions.request_exceptions import QueryParameterValidationError
from aiq.authentication.request_manager import RequestManager


@pytest.fixture
def request_manager():
    return RequestManager()


async def test_validate_base_url_valid(request_manager: RequestManager):
    """Test that the valid base URLs does NOT raise BaseUrlValidationError."""

    valid_urls = [
        "https://example.com/path",
        "http://example.com/path",
        "https://example.com:8080/path",
        "https://example.com/path?query=value"
    ]
    for url in valid_urls:
        try:
            request_manager._validate_base_url(url)
        except BaseUrlValidationError as e:
            pytest.fail(f"Valid URL '{url}' incorrectly raised BaseUrlValidationError: {e}")


async def test_validate_base_url_invalid(request_manager: RequestManager):
    """Test that the invalid base URLs raise BaseUrlValidationError."""

    invalid_urls = ["example.com/path", "ftp://example.com/path", "https:\\example.com"]

    for url in invalid_urls:
        with pytest.raises(BaseUrlValidationError):
            request_manager._validate_base_url(url)


async def test_validate_http_method_valid(request_manager: RequestManager):
    """Test that the valid HTTP methods does NOT raise ValueError."""

    valid_methods = ["GET", "POST", "PoSt", "PUT", "DELETE", "get", "post", "put", "delete"]

    for method in valid_methods:
        request_manager._validate_http_method(method)


async def test_validate_http_method_invalid(request_manager: RequestManager):
    """Test that the invalid HTTP methods raise ValueError."""

    invalid_methods = ["INVALID", "FOO", "BAR"]

    for method in invalid_methods:
        with pytest.raises(HTTPMethodValidationError):
            request_manager._validate_http_method(method)


async def test_validate_headers_valid(request_manager: RequestManager):
    """Test that the valid headers does NOT raise HeaderValidationError."""

    valid_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer token",
        "X-Custom-Header": "value",
        "AuthorizationTest": "Basic token"
    }
    request_manager._validate_headers(valid_headers)


async def test_validate_headers_invalid(request_manager: RequestManager):
    """Test that the invalid HTTP headers raise HeaderValidationError."""

    invalid_headers = [
        {
            "Invalid@Header": "value"  # Invalid character in header name
        },
        {
            "Valid-Header": "value\nwith\nnewlines"  # Newlines in header value
        },
        {
            "": "value"  # Empty header name
        },
        {
            "Header With Spaces": "value"  # Space in header name
        },
        {
            "X-Valid-Header": ""  # Empty header value
        },
        {
            "X-Valid-Header": None  # None header value
        },
        {
            "X-Valid-Header": "value\x00with\x00nulls"  # Null characters in header value
        },
        {
            "X-Valid-Header": "value\r\nX-Injected: hacked"  # CRLF injection in header value
        },
        {
            "X-Valid-Header": "value\nX-Injected: hacked"  # LF injection in header value
        },
        {
            "X-Valid-Header\nX-Injected": "value"  # LF injection in header name
        },
        {
            "Content-Type: application/json": "value"  # Colon in header name
        },
        {
            None: "value"  # None as header name
        },
        {
            123: "value"  # Non-string header name
        },
        {
            "X-Valid-Header": "value\r\n"  # Trailing CRLF injection in header value
        },
        {
            "X-Valid-Header": "value\r"  # Lone CR injection
        },
        {
            "X-Valid-Header": "value\n"  # Lone LF injection
        },
        {
            "X-Valid-Header": "value\r\n"  # Trailing CRLF
        },
        {
            "X-Valid-Header": "value\nX-Another-Header: value"  # Header splitting via value
        },
    ]
    for headers in invalid_headers:
        with pytest.raises((HTTPHeaderValidationError, ValueError, TypeError)):
            request_manager._validate_headers(headers)


async def test_validate_query_parameters_valid(request_manager: RequestManager):
    """Test that the valid query parameters do NOT raise QueryParameterValidationError."""

    valid_query_params = {"key1": "value1", "key2": "value2", "special": "!@#$%^&*()"}

    request_manager._validate_query_parameters(valid_query_params)


async def test_validate_query_parameters_invalid(request_manager: RequestManager):
    """Test that the invalid query parameters raise QueryParameterValidationError."""

    invalid_query_params = [
        {
            "": "value"  # Empty key
        },
        {
            "key": ""  # Empty value
        },
        {
            " key": "value"  # Leading space in key
        },
        {
            "key ": "value"  # Trailing space in key
        },
        {
            " key ": "value"  # Leading and trailing spaces in key
        },
        {
            # Newline in key (potential header injection)
            "key\nInjected": "value"
        },
        {
            # Newline in value (potential header injection)
            "key": "value\nInjected"
        },
        {
            "key": "value\r\nAnother-Header: hacked"  # CRLF injection in value
        },
        {
            "key\r\nAnother-Header": "value"  # CRLF injection in key
        },
        {
            None: "value"  # None key
        },
        {
            "key": None  # None value
        },
        {
            123: "value"  # Non-string key
        }
    ]
    for query_params in invalid_query_params:
        with pytest.raises((QueryParameterValidationError, ValueError)):
            request_manager._validate_query_parameters(query_params)
