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

from enum import Enum


class MCPErrorCategory(str, Enum):
    """Categories of MCP errors for structured handling."""
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    SSL = "ssl"
    AUTHENTICATION = "authentication"
    TOOL_NOT_FOUND = "tool_not_found"
    PROTOCOL = "protocol"
    UNKNOWN = "unknown"


class MCPError(Exception):
    """Base exception for MCP-related errors."""

    def __init__(self,
                 message: str,
                 url: str,
                 category: MCPErrorCategory = MCPErrorCategory.UNKNOWN,
                 suggestions: list[str] | None = None,
                 original_exception: Exception | None = None):
        super().__init__(message)
        self.url = url
        self.category = category
        self.suggestions = suggestions or []
        self.original_exception = original_exception


class MCPConnectionError(MCPError):
    """Exception for MCP connection failures."""

    def __init__(self, url: str, original_exception: Exception | None = None):
        super().__init__(f"Unable to connect to MCP server at {url}",
                         url=url,
                         category=MCPErrorCategory.CONNECTION,
                         suggestions=[
                             "Please ensure the MCP server is running and accessible",
                             "Check if the URL and port are correct"
                         ],
                         original_exception=original_exception)


class MCPTimeoutError(MCPError):
    """Exception for MCP timeout errors."""

    def __init__(self, url: str, original_exception: Exception | None = None):
        super().__init__(f"Connection timed out to MCP server at {url}",
                         url=url,
                         category=MCPErrorCategory.TIMEOUT,
                         suggestions=[
                             "The server may be overloaded or network is slow",
                             "Try again in a moment or check network connectivity"
                         ],
                         original_exception=original_exception)


class MCPSSLError(MCPError):
    """Exception for MCP SSL/TLS errors."""

    def __init__(self, url: str, original_exception: Exception | None = None):
        super().__init__(f"SSL/TLS error connecting to {url}",
                         url=url,
                         category=MCPErrorCategory.SSL,
                         suggestions=[
                             "Check if the server requires HTTPS or has valid certificates",
                             "Try using HTTP instead of HTTPS if appropriate"
                         ],
                         original_exception=original_exception)


class MCPRequestError(MCPError):
    """Exception for MCP request errors."""

    def __init__(self, url: str, original_exception: Exception | None = None):
        message = f"Request failed to MCP server at {url}"
        if original_exception:
            message += f": {original_exception}"

        super().__init__(message,
                         url=url,
                         category=MCPErrorCategory.PROTOCOL,
                         suggestions=["Check the server URL format and network settings"],
                         original_exception=original_exception)


class MCPToolNotFoundError(MCPError):
    """Exception for when a specific MCP tool is not found."""

    def __init__(self, tool_name: str, url: str, original_exception: Exception | None = None):
        super().__init__(f"Tool '{tool_name}' not available at {url}",
                         url=url,
                         category=MCPErrorCategory.TOOL_NOT_FOUND,
                         suggestions=[
                             "Use 'aiq info mcp --detail' to see available tools",
                             "Check that the tool name is spelled correctly"
                         ],
                         original_exception=original_exception)


class MCPAuthenticationError(MCPError):
    """Exception for MCP authentication failures."""

    def __init__(self, url: str, original_exception: Exception | None = None):
        super().__init__(f"Authentication failed when connecting to MCP server at {url}",
                         url=url,
                         category=MCPErrorCategory.AUTHENTICATION,
                         suggestions=[
                             "Check if the server requires authentication credentials",
                             "Verify that your credentials are correct and not expired"
                         ],
                         original_exception=original_exception)


class MCPProtocolError(MCPError):
    """Exception for MCP protocol-related errors."""

    def __init__(self, url: str, message: str = "Protocol error", original_exception: Exception | None = None):
        super().__init__(f"{message} (MCP server at {url})",
                         url=url,
                         category=MCPErrorCategory.PROTOCOL,
                         suggestions=[
                             "Check that the MCP server is running and accessible at this URL",
                             "Verify the server supports the expected MCP protocol version"
                         ],
                         original_exception=original_exception)
