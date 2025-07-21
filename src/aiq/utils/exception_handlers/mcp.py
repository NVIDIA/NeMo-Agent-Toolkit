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

import logging
import ssl
from collections.abc import Callable
from functools import wraps
from typing import Any

import httpx

from aiq.tool.mcp.exceptions import MCPAuthenticationError
from aiq.tool.mcp.exceptions import MCPConnectionError
from aiq.tool.mcp.exceptions import MCPError
from aiq.tool.mcp.exceptions import MCPProtocolError
from aiq.tool.mcp.exceptions import MCPRequestError
from aiq.tool.mcp.exceptions import MCPSSLError
from aiq.tool.mcp.exceptions import MCPTimeoutError
from aiq.tool.mcp.exceptions import MCPToolNotFoundError

logger = logging.getLogger(__name__)


def _extract_url(args: tuple, kwargs: dict[str, Any], url_param: str, func_name: str) -> str:
    """Extract URL from function arguments using clean fallback chain.

    Args:
        args: Function positional arguments
        kwargs: Function keyword arguments
        url_param (str): Parameter name containing the URL
        func_name (str): Function name for logging

    Returns:
        URL string or "unknown" if extraction fails
    """
    # Try keyword arguments first
    if url_param in kwargs:
        return kwargs[url_param]

    # Try self attribute (e.g., self.url)
    if args and hasattr(args[0], url_param):
        return getattr(args[0], url_param)

    # Try common case: url as second parameter after self
    if len(args) > 1 and url_param == "url":
        return args[1]

    # Fallback with warning
    logger.warning("Could not extract URL for error handling in %s", func_name)
    return "unknown"


def extract_primary_exception(exceptions: list[Exception]) -> Exception:
    """Extract the most relevant exception from a group.

    Prioritizes connection errors over others for better user experience.

    Args:
        exceptions (list[Exception]): List of exceptions from ExceptionGroup

    Returns:
        Most relevant exception for user feedback
    """
    # Prioritize connection errors
    for exc in exceptions:
        if isinstance(exc, (httpx.ConnectError, ConnectionError)):
            return exc

    # Then timeout errors
    for exc in exceptions:
        if isinstance(exc, httpx.TimeoutException):
            return exc

    # Then SSL errors
    for exc in exceptions:
        if isinstance(exc, ssl.SSLError):
            return exc

    # Fall back to first exception
    return exceptions[0]


def convert_to_mcp_error(e: Exception, url: str) -> MCPError:
    """Convert single exception to appropriate MCPError.

    Args:
        e (Exception): Single exception to convert
        url (str): MCP server URL for context

    Returns:
        Appropriate MCPError subclass
    """
    match e:
        case httpx.ConnectError() | ConnectionError():
            return MCPConnectionError(url, e)
        case httpx.TimeoutException():
            return MCPTimeoutError(url, e)
        case ssl.SSLError():
            return MCPSSLError(url, e)
        case httpx.RequestError():
            return MCPRequestError(url, e)
        case ValueError() if "Tool" in str(e) and "not available" in str(e):
            # Extract tool name from error message if possible
            tool_name = str(e).split("Tool ")[1].split(" not available")[0] if "Tool " in str(e) else "unknown"
            return MCPToolNotFoundError(tool_name, url, e)
        case _:
            # Handle TaskGroup error message specifically
            if "unhandled errors in a TaskGroup" in str(e):
                return MCPProtocolError(url, "Failed to connect to MCP server", e)
            if "unauthorized" in str(e).lower() or "forbidden" in str(e).lower():
                return MCPAuthenticationError(url, e)
            return MCPError(f"Unexpected error: {e}", url, original_exception=e)


def handle_mcp_exceptions(url_param: str = "url"):
    """Decorator that handles exceptions and converts them to MCPErrors.

    This decorator wraps MCP client methods and converts low-level exceptions
    to structured MCPError instances with helpful user guidance.

    Args:
        url_param (str): Name of the parameter or attribute containing the MCP server URL

    Returns:
        Decorator function

    Example:
        @handle_mcp_exceptions("url")
        async def get_tools(self, url: str):
            # Method implementation
            pass

        @handle_mcp_exceptions("url")  # Uses self.url
        async def get_tool(self):
            # Method implementation
            pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:

        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except MCPError:
                # Re-raise MCPErrors as-is
                raise
            except Exception as e:
                url = _extract_url(args, kwargs, url_param, func.__name__)

                # Handle ExceptionGroup by extracting most relevant exception
                if isinstance(e, ExceptionGroup):
                    primary_exception = extract_primary_exception(list(e.exceptions))
                    mcp_error = convert_to_mcp_error(primary_exception, url)
                else:
                    mcp_error = convert_to_mcp_error(e, url)

                raise mcp_error from e

        return wrapper

    return decorator


def mcp_exception_handler(func: Callable[..., Any]) -> Callable[..., Any]:
    """Simplified decorator for methods that have self.url attribute.

    This is a convenience decorator that assumes the URL is available as self.url.
    Follows the same pattern as schema_exception_handler in this directory.

    Args:
        func (Callable[..., Any]): The function to decorate

    Returns:
        Callable[..., Any]: Decorated function
    """
    return handle_mcp_exceptions("url")(func)
