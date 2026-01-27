#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Compatibility helpers for FastMCP runtime selection."""

from __future__ import annotations

import importlib.metadata
import inspect
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP as FastMCPType
else:
    FastMCPType: TypeAlias = Any


def get_fastmcp_class() -> type[FastMCPType]:
    """Resolve the FastMCP class for the available runtime."""
    try:
        from fastmcp import FastMCP as FastMCPClass
    except ImportError:
        from mcp.server.fastmcp import FastMCP as FastMCPClass
    return FastMCPClass


def build_fastmcp_server(**kwargs: Any) -> FastMCPType:
    """Build a FastMCP server with supported keyword arguments."""
    fastmcp_class = get_fastmcp_class()
    signature = inspect.signature(fastmcp_class.__init__)
    allowed = set(signature.parameters.keys())
    allowed.discard("self")
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in allowed and value is not None}
    return fastmcp_class(**filtered_kwargs)


def get_fastmcp_version() -> str | None:
    """Return the installed FastMCP or MCP version string."""
    for package_name in ("fastmcp", "mcp"):
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def is_fastmcp3() -> bool:
    """Return True when the runtime is FastMCP 3.x or newer."""
    version = get_fastmcp_version()
    if not version:
        return False
    return _parse_major(version) >= 3


def _parse_major(version: str) -> int:
    digits: list[str] = []
    for char in version:
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    if not digits:
        return 0
    return int("".join(digits))
