# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""File operation tools for sandbox."""

import logging
from pathlib import PurePosixPath
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field

from nat_sandbox_agent.tools.sandbox.executor import SandboxToolExecutor

logger = logging.getLogger(__name__)

# Allowed base directories for file operations
ALLOWED_BASE_PATHS = ("/workspace",)


def _validate_path(path: str) -> str:
    """Validate and normalize a file path to prevent path traversal attacks.

    Args:
        path: The path to validate.

    Returns:
        The normalized path.

    Raises:
        ValueError: If the path is outside allowed directories.
    """
    # Normalize the path (resolve .. and .)
    normalized = str(PurePosixPath(path))

    # Check if the path starts with an allowed base
    if not any(normalized.startswith(base) for base in ALLOWED_BASE_PATHS):
        raise ValueError(
            f"Path '{path}' is outside allowed directories. "
            f"Allowed: {ALLOWED_BASE_PATHS}"
        )

    # Additional check: ensure no path traversal escapes after normalization
    # by verifying the resolved path still starts with allowed base
    parts = PurePosixPath(normalized).parts
    resolved_parts = []
    for part in parts:
        if part == "..":
            if resolved_parts:
                resolved_parts.pop()
        elif part != ".":
            resolved_parts.append(part)

    resolved = "/" + "/".join(resolved_parts[1:]) if resolved_parts else "/"

    if not any(resolved.startswith(base) for base in ALLOWED_BASE_PATHS):
        raise ValueError(
            f"Path '{path}' resolves to '{resolved}' which is outside allowed directories."
        )

    return normalized


class FileReadInput(BaseModel):
    """Input schema for file reading."""

    path: str = Field(
        description="Path to the file to read in the sandbox."
    )


class FileWriteInput(BaseModel):
    """Input schema for file writing."""

    path: str = Field(
        description="Path where the file should be written."
    )
    content: str = Field(
        description="Content to write to the file."
    )


async def read_file(
    executor: SandboxToolExecutor,
    path: str,
) -> dict[str, Any]:
    """Read a file from the sandbox.

    Args:
        executor: The sandbox executor instance.
        path: Path to the file.

    Returns:
        Dict with status and content or error message.
    """
    logger.info(f"Reading file: {path}")

    try:
        # Validate path to prevent traversal attacks
        validated_path = _validate_path(path)
        content = await executor.sandbox.read_file(validated_path)
        return {
            "status": "success",
            "content": executor.truncate(content),
            "path": validated_path,
        }
    except ValueError as e:
        return {
            "status": "error",
            "error": str(e),
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "error": f"File not found: {path}",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


async def write_file(
    executor: SandboxToolExecutor,
    path: str,
    content: str,
) -> dict[str, Any]:
    """Write content to a file in the sandbox.

    Args:
        executor: The sandbox executor instance.
        path: Path to the file.
        content: Content to write.

    Returns:
        Dict with status and path.
    """
    logger.info(f"Writing file: {path} ({len(content)} chars)")

    try:
        # Validate path to prevent traversal attacks
        validated_path = _validate_path(path)
        await executor.sandbox.write_file(validated_path, content)
        return {
            "status": "success",
            "path": validated_path,
            "size": len(content),
        }
    except ValueError as e:
        return {
            "status": "error",
            "error": str(e),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def create_file_read_tool(executor: SandboxToolExecutor) -> StructuredTool:
    """Create the file read tool.

    Args:
        executor: The sandbox executor instance.

    Returns:
        LangChain StructuredTool instance.
    """
    return StructuredTool.from_function(
        coroutine=lambda path: read_file(executor, path),
        name="file_read",
        description=(
            "Read the contents of a file from the sandbox. "
            "Returns the file content as text."
        ),
        args_schema=FileReadInput,
    )


def create_file_write_tool(executor: SandboxToolExecutor) -> StructuredTool:
    """Create the file write tool.

    Args:
        executor: The sandbox executor instance.

    Returns:
        LangChain StructuredTool instance.
    """
    return StructuredTool.from_function(
        coroutine=lambda path, content: write_file(executor, path, content),
        name="file_write",
        description=(
            "Write content to a file in the sandbox. "
            "Parent directories are created automatically if needed."
        ),
        args_schema=FileWriteInput,
    )
