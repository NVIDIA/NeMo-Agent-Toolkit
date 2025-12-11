# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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
"""Common utilities shared across all parser implementations."""

import json
from typing import Any

from nat.data_models.intermediate_step import IntermediateStep


def extract_content(data: Any) -> str:
    """Extract string content from various data formats.

    This is a shared utility used by all parser implementations.

    Args:
        data: The data to extract content from. Can be a string, dict, list,
              or object with content/text attributes.

    Returns:
        The extracted content as a string.
    """
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        # Try common content fields
        for key in ["content", "text", "message", "output"]:
            if key in data:
                return str(data[key])
        # Check for blocks format
        if "blocks" in data:
            blocks = data["blocks"]
            if isinstance(blocks, list):
                return ''.join(block.get('text', '') if isinstance(block, dict) else str(block) for block in blocks)
        # Fallback to JSON representation
        return json.dumps(data)
    elif isinstance(data, list):
        # Join list items if they're strings
        if all(isinstance(item, str) for item in data):
            return "\n".join(data)
        # Otherwise convert to JSON
        return json.dumps(data)
    elif hasattr(data, 'content'):
        return str(data.content)
    elif hasattr(data, 'text'):
        return str(data.text)
    else:
        return str(data)


def parse_generic_message(message: IntermediateStep) -> dict:
    """Parse messages that don't fit standard patterns.

    This is a shared utility used by all parser implementations for handling
    event types that don't have specialized parsers.

    Args:
        message: An IntermediateStep object representing a message.

    Returns:
        A dictionary with 'role' and 'content' keys.
    """
    result = {"role": "user"}  # Default to user role

    # Try to extract content from various fields
    if message.data:
        if message.data.output:
            result["content"] = extract_content(message.data.output)
        elif message.data.input:
            result["content"] = extract_content(message.data.input)
        elif message.data.chunk:
            result["content"] = extract_content(message.data.chunk)
        else:
            result["content"] = ""
    else:
        result["content"] = ""

    return result
