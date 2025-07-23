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
import sys
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


def format_mcp_error(error: MCPError, include_traceback: bool = False) -> None:
    """Format MCP errors for CLI display with structured logging and user guidance.

    Logs structured error information for debugging and displays user-friendly
    error messages with actionable suggestions to stderr.

    Args:
        error (MCPError): MCPError instance containing message, url, category, suggestions, and original_exception
        include_traceback (bool, optional): Whether to include the traceback in the error message. Defaults to False.
    """
    # Log structured error information for debugging
    logger.error("MCP operation failed: %s", error, exc_info=include_traceback)

    # Display user-friendly suggestions
    for suggestion in error.suggestions:
        print(f"  → {suggestion}", file=sys.stderr)


def _extract_url(args: tuple, kwargs: dict[str, Any], url_param: str, func_name: str) -> str:
    """Extract URL from function arguments using clean fallback chain.

    Args:
        args: Function positional arguments
        kwargs: Function keyword arguments
        url_param (str): Parameter name containing the URL
        func_name (str): Function name for logging

    Returns:
        str: URL string or "unknown" if extraction fails
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
        Exception: Most relevant exception for user feedback
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


def convert_to_mcp_error(exception: Exception, url: str) -> MCPError:
    """Convert single exception to appropriate MCPError.

    Args:
        exception (Exception): Single exception to convert
        url (str): MCP server URL for context

    Returns:
        MCPError: Appropriate MCPError subclass
    """
    match exception:
        case httpx.ConnectError() | ConnectionError():
            return MCPConnectionError(url, exception)
        case httpx.TimeoutException():
            return MCPTimeoutError(url, exception)
        case ssl.SSLError():
            return MCPSSLError(url, exception)
        case httpx.RequestError():
            return MCPRequestError(url, exception)
        case ValueError() if "Tool" in str(exception) and "not available" in str(exception):
            # Extract tool name from error message if possible
            tool_name = str(exception).split("Tool ")[1].split(" not available")[0] if "Tool " in str(
                exception) else "unknown"
            return MCPToolNotFoundError(tool_name, url, exception)
        case _:
            # Handle TaskGroup error message specifically
            if "unhandled errors in a TaskGroup" in str(exception):
                return MCPProtocolError(url, "Failed to connect to MCP server", exception)
            if "unauthorized" in str(exception).lower() or "forbidden" in str(exception).lower():
                return MCPAuthenticationError(url, exception)
            return MCPError(f"Unexpected error: {exception}", url, original_exception=exception)


def handle_mcp_exceptions(url_param: str = "url") -> Callable[..., Any]:
    """Decorator that handles exceptions and converts them to MCPErrors.

    This decorator wraps MCP client methods and converts low-level exceptions
    to structured MCPError instances with helpful user guidance.

    Args:
        url_param (str): Name of the parameter or attribute containing the MCP server URL

    Returns:
        Callable[..., Any]: Decorated function

    Example:
        .. code-block:: python

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
                if isinstance(e, ExceptionGroup):  # noqa: F821
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
