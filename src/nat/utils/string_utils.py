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

from typing import Any

from pydantic import BaseModel


def convert_to_str(value: Any) -> str:
    """
    Convert a value to a string representation.
    Handles various types including lists, dictionaries, and other objects.
    """
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return ", ".join(map(str, value))
    elif isinstance(value, BaseModel):
        return value.model_dump_json(exclude_none=True, exclude_unset=True)
    elif isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    elif hasattr(value, '__str__'):
        return str(value)
    else:
        raise ValueError(f"Unsupported type for conversion to string: {type(value)}")


def truncate_string(text: str | None, max_length: int = 100) -> str | None:
    """
    Truncate a string to a maximum length, adding ellipsis if truncated.

    Args:
        text: The text to truncate (can be None)
        max_length: Maximum allowed length (default: 100)

    Returns:
        The truncated text with ellipsis if needed, or None if input was None
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def sanitize_tool_name_for_openai(tool_name: str) -> str:
    """
    Sanitize a tool name to comply with OpenAI's tool name validation rules.

    OpenAI requires tool names to match the pattern: ^[a-zA-Z0-9_-]+$
    This function replaces invalid characters (like dots, spaces, etc.) with underscores.

    Args:
        tool_name: The original tool name (may contain dots, spaces, or other invalid characters)

    Returns:
        A sanitized tool name that complies with OpenAI's validation rules

    Examples:
        >>> sanitize_tool_name_for_openai("mcp_client.my_tool")
        'mcp_client_my_tool'
        >>> sanitize_tool_name_for_openai("my-tool")
        'my-tool'
        >>> sanitize_tool_name_for_openai("tool with spaces")
        'tool_with_spaces'
    """
    import re
    # Replace any character that is not a letter, digit, underscore, or hyphen with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', tool_name)
    # Ensure the name doesn't start or end with a hyphen (OpenAI requirement)
    sanitized = sanitized.strip('-')
    # Replace multiple consecutive underscores with a single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized