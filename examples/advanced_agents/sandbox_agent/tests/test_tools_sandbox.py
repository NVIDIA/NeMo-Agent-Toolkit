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
"""Tests for sandbox-side tools (shell, python, file_read, file_write, web_browse)."""

from unittest.mock import AsyncMock

import pytest

from nat_sandbox_agent.sandbox.base import CommandResult
from nat_sandbox_agent.tools.sandbox import SandboxToolExecutor
from nat_sandbox_agent.tools.sandbox.execution import create_python_tool
from nat_sandbox_agent.tools.sandbox.execution import create_shell_tool
from nat_sandbox_agent.tools.sandbox.execution import execute_python
from nat_sandbox_agent.tools.sandbox.execution import execute_shell
from nat_sandbox_agent.tools.sandbox.file_ops import create_file_read_tool
from nat_sandbox_agent.tools.sandbox.file_ops import create_file_write_tool
from nat_sandbox_agent.tools.sandbox.file_ops import read_file
from nat_sandbox_agent.tools.sandbox.file_ops import write_file


class TestSandboxToolExecutor:
    """Tests for SandboxToolExecutor."""

    def test_init_with_defaults(self, mock_sandbox):
        """Test executor initialization with default values."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        assert executor.sandbox is mock_sandbox
        assert executor.max_output_chars == 16000
        assert executor.default_timeout == 120

    def test_init_with_custom_values(self, mock_sandbox):
        """Test executor initialization with custom values."""
        executor = SandboxToolExecutor(
            sandbox=mock_sandbox,
            max_output_chars=5000,
            default_timeout=60,
        )

        assert executor.max_output_chars == 5000
        assert executor.default_timeout == 60

    def test_truncate_short_text(self, mock_sandbox):
        """Test that short text is not truncated."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox, max_output_chars=100)

        result = executor.truncate("Short text")

        assert result == "Short text"

    def test_truncate_long_text(self, mock_sandbox):
        """Test that long text is truncated with indicator."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox, max_output_chars=20)

        long_text = "A" * 100
        result = executor.truncate(long_text)

        assert len(result) < 100
        assert "truncated" in result
        assert "100 total chars" in result

    @pytest.mark.asyncio
    async def test_list_generated_files_success(self, mock_sandbox):
        """Test listing generated files using shell command."""
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=0, stdout="output.txt\ndata.json\n", stderr=""))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        files = await executor.list_generated_files()

        assert files == ["/workspace/output/output.txt", "/workspace/output/data.json"]
        mock_sandbox.run_command.assert_called_once_with(
            "ls -1 /workspace/output", timeout=120
        )

    @pytest.mark.asyncio
    async def test_list_generated_files_handles_exception(self, mock_sandbox):
        """Test that list_generated_files handles exceptions gracefully."""
        mock_sandbox.run_command = AsyncMock(side_effect=Exception("Directory not found"))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        files = await executor.list_generated_files()

        assert files == []

    @pytest.mark.asyncio
    async def test_list_generated_files_empty_on_error(self, mock_sandbox):
        """Test that list_generated_files returns empty list on command failure."""
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=1, stdout="", stderr="ls: cannot access"))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        files = await executor.list_generated_files()

        assert files == []


class TestShellTool:
    """Tests for shell command execution tool."""

    @pytest.mark.asyncio
    async def test_execute_shell_success(self, mock_sandbox):
        """Test successful shell command execution."""
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=0, stdout="file1.txt\nfile2.py", stderr=""))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await execute_shell(executor, "ls -la", "/workspace")

        assert result["status"] == "success"
        assert "file1.txt" in result["stdout"]
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_execute_shell_failure(self, mock_sandbox):
        """Test shell command failure."""
        mock_sandbox.run_command = AsyncMock(
            return_value=CommandResult(exit_code=1, stdout="", stderr="Command not found"))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await execute_shell(executor, "invalid_command")

        assert result["status"] == "error"
        assert result["exit_code"] == 1
        assert "Command not found" in result["stderr"]

    @pytest.mark.asyncio
    async def test_execute_shell_respects_working_dir(self, mock_sandbox):
        """Test that working directory is passed correctly."""
        mock_sandbox.run_command = AsyncMock(return_value=CommandResult(exit_code=0, stdout="", stderr=""))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        await execute_shell(executor, "pwd", working_dir="/custom/dir")

        mock_sandbox.run_command.assert_called_once()
        call_kwargs = mock_sandbox.run_command.call_args[1]
        assert call_kwargs["working_dir"] == "/custom/dir"

    @pytest.mark.asyncio
    async def test_execute_shell_truncates_output(self, mock_sandbox):
        """Test that long output is truncated."""
        long_output = "X" * 20000
        mock_sandbox.run_command = AsyncMock(return_value=CommandResult(exit_code=0, stdout=long_output, stderr=""))
        executor = SandboxToolExecutor(sandbox=mock_sandbox, max_output_chars=100)

        result = await execute_shell(executor, "cat bigfile.txt")

        assert len(result["stdout"]) < 20000
        assert "truncated" in result["stdout"]

    def test_create_shell_tool_returns_structured_tool(self, mock_sandbox):
        """Test that create_shell_tool creates a StructuredTool."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox)
        tool = create_shell_tool(executor)

        assert tool.name == "shell"
        assert "bash" in tool.description.lower() or "shell" in tool.description.lower()


class TestPythonTool:
    """Tests for Python code execution tool."""

    @pytest.mark.asyncio
    async def test_execute_python_success(self, mock_sandbox):
        """Test successful Python code execution."""
        mock_sandbox.write_file = AsyncMock()
        # First call is for listing files, second is for running the script
        mock_sandbox.run_command = AsyncMock(side_effect=[
            CommandResult(exit_code=0, stdout="42", stderr=""),  # python execution
            CommandResult(exit_code=0, stdout="result.txt\n", stderr=""),  # ls for generated files
        ])
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await execute_python(executor, "print(6 * 7)")

        assert result["status"] == "success"
        assert "42" in result["stdout"]
        assert "generated_files" in result

    @pytest.mark.asyncio
    async def test_execute_python_writes_script(self, mock_sandbox):
        """Test that Python code is written to script file."""
        mock_sandbox.write_file = AsyncMock()
        mock_sandbox.run_command = AsyncMock(side_effect=[
            CommandResult(exit_code=0, stdout="", stderr=""),  # python execution
            CommandResult(exit_code=0, stdout="", stderr=""),  # ls for generated files
        ])
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        code = "print('hello')"
        await execute_python(executor, code)

        mock_sandbox.write_file.assert_called_once()
        call_args = mock_sandbox.write_file.call_args
        assert call_args[0][0] == "/workspace/temp/_script.py"
        assert call_args[0][1] == code

    @pytest.mark.asyncio
    async def test_execute_python_error(self, mock_sandbox):
        """Test Python execution with error."""
        mock_sandbox.write_file = AsyncMock()
        mock_sandbox.run_command = AsyncMock(side_effect=[
            CommandResult(
                exit_code=1,
                stdout="",
                stderr="NameError: name 'undefined_var' is not defined",
            ),  # python execution error
            CommandResult(exit_code=0, stdout="", stderr=""),  # ls for generated files
        ])
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await execute_python(executor, "print(undefined_var)")

        assert result["status"] == "error"
        assert "NameError" in result["stderr"]

    def test_create_python_tool_returns_structured_tool(self, mock_sandbox):
        """Test that create_python_tool creates a StructuredTool."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox)
        tool = create_python_tool(executor)

        assert tool.name == "python"
        assert "python" in tool.description.lower()


class TestFileReadTool:
    """Tests for file read tool."""

    @pytest.mark.asyncio
    async def test_read_file_success(self, mock_sandbox):
        """Test successful file read."""
        mock_sandbox.read_file = AsyncMock(return_value="File content here")
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await read_file(executor, "/workspace/test.txt")

        assert result["status"] == "success"
        assert result["content"] == "File content here"
        assert result["path"] == "/workspace/test.txt"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, mock_sandbox):
        """Test file not found error."""
        mock_sandbox.read_file = AsyncMock(side_effect=FileNotFoundError("No such file"))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await read_file(executor, "/workspace/nonexistent.txt")

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_read_file_other_error(self, mock_sandbox):
        """Test handling of other errors."""
        mock_sandbox.read_file = AsyncMock(side_effect=PermissionError("Access denied"))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await read_file(executor, "/workspace/protected.txt")

        assert result["status"] == "error"
        assert "Access denied" in result["error"]

    @pytest.mark.asyncio
    async def test_read_file_truncates_long_content(self, mock_sandbox):
        """Test that long file content is truncated."""
        long_content = "X" * 20000
        mock_sandbox.read_file = AsyncMock(return_value=long_content)
        executor = SandboxToolExecutor(sandbox=mock_sandbox, max_output_chars=100)

        result = await read_file(executor, "/workspace/bigfile.txt")

        assert result["status"] == "success"
        assert len(result["content"]) < 20000
        assert "truncated" in result["content"]

    @pytest.mark.asyncio
    async def test_read_file_path_traversal_blocked(self, mock_sandbox):
        """Test that path traversal attempts are blocked."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await read_file(executor, "/etc/passwd")

        assert result["status"] == "error"
        assert "outside allowed directories" in result["error"]

    def test_create_file_read_tool_returns_structured_tool(self, mock_sandbox):
        """Test that create_file_read_tool creates a StructuredTool."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox)
        tool = create_file_read_tool(executor)

        assert tool.name == "file_read"
        assert "read" in tool.description.lower()


class TestFileWriteTool:
    """Tests for file write tool."""

    @pytest.mark.asyncio
    async def test_write_file_success(self, mock_sandbox):
        """Test successful file write."""
        mock_sandbox.write_file = AsyncMock()
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await write_file(executor, "/workspace/output.txt", "Hello World")

        assert result["status"] == "success"
        assert result["path"] == "/workspace/output.txt"
        assert result["size"] == 11

        mock_sandbox.write_file.assert_called_once_with("/workspace/output.txt", "Hello World")

    @pytest.mark.asyncio
    async def test_write_file_error(self, mock_sandbox):
        """Test file write error handling."""
        mock_sandbox.write_file = AsyncMock(side_effect=PermissionError("Cannot write to directory"))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await write_file(executor, "/workspace/readonly/file.txt", "content")

        assert result["status"] == "error"
        assert "Cannot write" in result["error"]

    @pytest.mark.asyncio
    async def test_write_file_path_traversal_blocked(self, mock_sandbox):
        """Test that path traversal attempts are blocked."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        result = await write_file(executor, "/etc/passwd", "content")

        assert result["status"] == "error"
        assert "outside allowed directories" in result["error"]

    def test_create_file_write_tool_returns_structured_tool(self, mock_sandbox):
        """Test that create_file_write_tool creates a StructuredTool."""
        executor = SandboxToolExecutor(sandbox=mock_sandbox)
        tool = create_file_write_tool(executor)

        assert tool.name == "file_write"
        assert "write" in tool.description.lower()


class TestWebBrowseTool:
    """Tests for web browse tool."""

    @pytest.mark.asyncio
    async def test_web_browse_success(self, mock_sandbox):
        """Test successful web browse."""
        import json

        mock_result = json.dumps({
            "status": "success",
            "url": "https://example.com",
            "title": "Example Domain",
            "content": "This is example content.",
        })
        mock_sandbox.write_file = AsyncMock()
        mock_sandbox.run_command = AsyncMock(return_value=CommandResult(exit_code=0, stdout=mock_result, stderr=""))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        from nat_sandbox_agent.tools.sandbox.browser import web_browse

        result = await web_browse(executor, "https://example.com")

        assert result["status"] == "success"
        assert result["title"] == "Example Domain"
        assert "example content" in result["content"]
        # Verify script was written
        mock_sandbox.write_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_web_browse_with_selector(self, mock_sandbox):
        """Test web browse with CSS selector."""
        import json

        mock_result = json.dumps({
            "status": "success",
            "url": "https://example.com",
            "title": "Example",
            "content": "Selected content only",
        })
        mock_sandbox.write_file = AsyncMock()
        mock_sandbox.run_command = AsyncMock(return_value=CommandResult(exit_code=0, stdout=mock_result, stderr=""))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        from nat_sandbox_agent.tools.sandbox.browser import web_browse

        result = await web_browse(executor, "https://example.com", selector="article")

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_web_browse_error(self, mock_sandbox):
        """Test web browse error handling."""
        mock_sandbox.write_file = AsyncMock()
        mock_sandbox.run_command = AsyncMock(return_value=CommandResult(exit_code=1, stdout="", stderr="Network error"))
        executor = SandboxToolExecutor(sandbox=mock_sandbox)

        from nat_sandbox_agent.tools.sandbox.browser import web_browse

        result = await web_browse(executor, "https://invalid-url.test")

        assert result["status"] == "error"

    def test_create_web_browse_tool_returns_structured_tool(self, mock_sandbox):
        """Test that create_web_browse_tool creates a StructuredTool."""
        from nat_sandbox_agent.tools.sandbox.browser import create_web_browse_tool

        executor = SandboxToolExecutor(sandbox=mock_sandbox)
        tool = create_web_browse_tool(executor)

        assert tool.name == "web_browse"
        assert "browse" in tool.description.lower() or "web" in tool.description.lower()
