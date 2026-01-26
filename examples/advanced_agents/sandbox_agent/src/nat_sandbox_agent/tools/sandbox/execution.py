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

"""Shell and Python execution tools for sandbox."""

import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field

from nat_sandbox_agent.sandbox.base import DEFAULT_SCRIPT_PATH
from nat_sandbox_agent.sandbox.base import WORKSPACE_ROOT
from nat_sandbox_agent.tools.sandbox.executor import SandboxToolExecutor

logger = logging.getLogger(__name__)

class ShellInput(BaseModel):
    """Input schema for shell command execution."""

    command: str = Field(
        description="The shell command to execute in the sandbox."
    )
    working_dir: str = Field(
        default=WORKSPACE_ROOT,
        description="Working directory for the command.",
    )

class PythonInput(BaseModel):
    """Input schema for Python code execution."""

    code: str = Field(
        description="Python code to execute in the sandbox."
    )

async def execute_shell(
    executor: SandboxToolExecutor,
    command: str,
    working_dir: str = WORKSPACE_ROOT,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Execute a shell command in the sandbox.

    Args:
        executor: The sandbox executor instance.
        command: The shell command to execute.
        working_dir: Working directory for the command.
        timeout: Optional timeout override.

    Returns:
        Dict with status, stdout, stderr, and exit_code.
    """
    logger.info(f"Executing shell command: {command[:20]}... ({len(command)} chars)")

    result = await executor.sandbox.run_command(
        command=command,
        working_dir=working_dir,
        timeout=timeout or executor.default_timeout,
    )

    return {
        "status": "success" if result.success else "error",
        "stdout": executor.truncate(result.stdout),
        "stderr": executor.truncate(result.stderr),
        "exit_code": result.exit_code,
    }

async def execute_python(
    executor: SandboxToolExecutor,
    code: str,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Execute Python code in the sandbox.

    Args:
        executor: The sandbox executor instance.
        code: Python code to execute.
        timeout: Optional timeout override.

    Returns:
        Dict with status, stdout, stderr, and generated files.
    """
    logger.info(f"Executing Python code ({len(code)} chars)")

    # Write code to a temp file for better error messages
    try:
        await executor.sandbox.write_file(DEFAULT_SCRIPT_PATH, code)
    except Exception as e:
        logger.exception("Failed to write script file")
        return {
            "status": "error",
            "stdout": "",
            "stderr": executor.truncate(str(e)),
            "generated_files": [],
        }

    result = await executor.sandbox.run_command(
        command=f"cd {WORKSPACE_ROOT} && python3 {DEFAULT_SCRIPT_PATH}",
        timeout=timeout or executor.default_timeout,
    )

    # List any generated files in output directory
    generated_files = await executor.list_generated_files()

    return {
        "status": "success" if result.success else "error",
        "stdout": executor.truncate(result.stdout),
        "stderr": executor.truncate(result.stderr),
        "generated_files": generated_files,
    }

def create_shell_tool(executor: SandboxToolExecutor) -> StructuredTool:
    """Create the shell command tool.

    Args:
        executor: The sandbox executor instance.

    Returns:
        LangChain StructuredTool instance.
    """
    return StructuredTool.from_function(
        coroutine=lambda command, working_dir=WORKSPACE_ROOT: execute_shell(
            executor, command, working_dir
        ),
        name="shell",
        description=(
            "Execute bash/shell commands for SYSTEM OPERATIONS: "
            "file management (ls, cp, mv, rm, mkdir, chmod), "
            "package installation (pip install, apt-get), "
            "downloads (curl, wget), "
            "process management (ps, kill), "
            "git operations. "
            "Do NOT use for data processing - use python instead."
        ),
        args_schema=ShellInput,
    )

def create_python_tool(executor: SandboxToolExecutor) -> StructuredTool:
    """Create the Python execution tool.

    Args:
        executor: The sandbox executor instance.

    Returns:
        LangChain StructuredTool instance.
    """
    return StructuredTool.from_function(
        coroutine=lambda code: execute_python(executor, code),
        name="python",
        description=(
            "Execute Python code for DATA PROCESSING and COMPUTATION: "
            "data analysis (pandas, numpy), "
            "calculations and math, "
            "file parsing (JSON, CSV, XML), "
            "API calls (requests), "
            "text processing (regex). "
            "Generated files should be saved to /workspace/output/. "
            "Do NOT use for simple system commands - use shell instead."
        ),
        args_schema=PythonInput,
    )
