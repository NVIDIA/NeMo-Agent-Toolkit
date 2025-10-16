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
from pydantic import ValidationError

from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
from nat.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig


def test_mcp_front_end_config_default_values():
    """Test that the default values are set correctly."""
    config = MCPFrontEndConfig()

    assert config.name == "NeMo Agent Toolkit MCP"
    assert config.host == "localhost"
    assert config.port == 9901
    assert config.debug is False
    assert config.log_level == "INFO"
    assert isinstance(config.tool_names, list)
    assert len(config.tool_names) == 0


def test_mcp_front_end_config_custom_values():
    """Test that custom values are set correctly."""
    config = MCPFrontEndConfig(name="Custom MCP Server",
                               host="0.0.0.0",
                               port=8080,
                               debug=True,
                               log_level="DEBUG",
                               tool_names=["test_tool", "another_tool"])

    assert config.name == "Custom MCP Server"
    assert config.host == "0.0.0.0"
    assert config.port == 8080
    assert config.debug is True
    assert config.log_level == "DEBUG"
    assert config.tool_names == ["test_tool", "another_tool"]


def test_mcp_front_end_config_port_validation():
    """Test port validation (must be between 0 and 65535)."""
    # Valid port number
    config = MCPFrontEndConfig(port=8080)
    assert config.port == 8080

    # Invalid port number (too large)
    with pytest.raises(ValidationError):
        MCPFrontEndConfig(port=70000)

    # Invalid port number (negative)
    with pytest.raises(ValidationError):
        MCPFrontEndConfig(port=-1)


def test_mcp_front_end_config_from_dict():
    """Test creating config from a dictionary."""
    config_dict = {
        "name": "Dict Config",
        "host": "127.0.0.1",
        "port": 5000,
        "debug": True,
        "log_level": "WARNING",
        "tool_names": ["tool1", "tool2", "tool3"]
    }

    config = MCPFrontEndConfig(**config_dict)

    assert config.name == "Dict Config"
    assert config.host == "127.0.0.1"
    assert config.port == 5000
    assert config.debug is True
    assert config.log_level == "WARNING"
    assert config.tool_names == ["tool1", "tool2", "tool3"]


def test_security_warning_non_localhost_without_auth(caplog):
    """Test that a warning is logged when binding to non-localhost without authentication."""
    config = MCPFrontEndConfig(host="192.168.1.100", port=9901)  # noqa: F841
    # Check that a warning was logged
    assert any("without authentication" in record.message for record in caplog.records)
    assert any("192.168.1.100" in record.message for record in caplog.records)


def test_no_security_warning_localhost_without_auth(caplog):
    """Test that no warning is logged when binding to localhost without authentication."""
    config = MCPFrontEndConfig(host="localhost", port=9901)  # noqa: F841
    # Check that no security warning was logged
    assert not any("without authentication" in record.message for record in caplog.records)


def test_no_security_warning_with_auth(caplog):
    """Test that no warning is logged when authentication is configured for non-localhost."""
    auth_config = OAuth2ResourceServerConfig(issuer_url="https://example.com/oauth2")
    config = MCPFrontEndConfig(host="192.168.1.100", port=9901, server_auth=auth_config)  # noqa: F841
    # Check that no warning about missing authentication was logged
    assert not any("without authentication" in record.message for record in caplog.records)


def test_security_warning_sse_with_auth(caplog):
    """Test that a warning is logged when SSE transport is used with authentication configured."""
    auth_config = OAuth2ResourceServerConfig(issuer_url="https://example.com/oauth2")
    config = MCPFrontEndConfig(transport="sse", server_auth=auth_config)  # noqa: F841
    # Check that a warning was logged about SSE not supporting auth
    assert any("SSE transport does not support authentication" in record.message for record in caplog.records)
    assert any("server_auth will be ignored" in record.message for record in caplog.records)


def test_security_warning_sse_non_localhost(caplog):
    """Test that a warning is logged when SSE transport is used on non-localhost without auth."""
    config = MCPFrontEndConfig(transport="sse", host="192.168.1.100")  # noqa: F841
    # Check that a warning was logged about SSE lacking authentication
    assert any("SSE transport does not support authentication" in record.message for record in caplog.records)
    assert any("not recommended for production" in record.message for record in caplog.records)


def test_no_security_warning_sse_localhost(caplog):
    """Test that minimal warnings are logged when SSE transport is used on localhost."""

    # Check that no critical security warnings were logged (SSE on localhost is acceptable for dev)
    assert not any("not recommended for production" in record.message for record in caplog.records)


def test_no_security_warning_streamable_http_with_auth(caplog):
    """Test that no warning is logged when streamable-http is used with authentication."""
    auth_config = OAuth2ResourceServerConfig(issuer_url="https://example.com/oauth2")
    config = MCPFrontEndConfig(transport="streamable-http", host="192.168.1.100", server_auth=auth_config)  # noqa: F841
    # Check that no warnings were logged (this is the recommended configuration)
    assert not any("WARNING" in record.levelname for record in caplog.records)
