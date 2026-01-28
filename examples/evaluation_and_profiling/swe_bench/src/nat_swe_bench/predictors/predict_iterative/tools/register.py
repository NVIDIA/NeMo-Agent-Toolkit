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

# Register all the tools needed by the full predictor without loading the dependencies.
import logging
import typing

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class GitRepoToolConfig(FunctionBaseConfig, name="git_repo_tool"):
    """Configuration for git repository management tool."""
    _type: typing.Literal["git_repo_tool"] = "git_repo_tool"
    workspace_dir: str = "./.workspace"  # Base directory for cloning repositories
    cleanup_on_exit: bool = True  # Whether to clean up repos after use


@register_function(config_type=GitRepoToolConfig)
async def git_repo_tool(tool_config: GitRepoToolConfig, builder: Builder):
    """Git repository management tool for SWE Bench.

    Args:
        tool_config: Configuration for the git tool.
        builder: NAT builder instance.

    Yields:
        FunctionInfo for the git_operations function.
    """
    import json

    from .git_tool import RepoManager
    repo_manager = RepoManager(tool_config.workspace_dir)

    async def git_operations(args_str: str) -> str:
        """Perform git operations based on JSON input.

        Args:
            args_str: JSON string with 'operation' and operation-specific parameters.
                      Supported operations:
                      - setup: requires 'repo_url', 'base_commit', optional 'instance_id'
                      - cleanup: no additional parameters

        Returns:
            For 'setup': the repository path as a string.
            For 'cleanup': "Cleanup complete".

        Raises:
            ValueError: If JSON is invalid or operation is unknown.
        """
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}") from e

        operation = args.get('operation')

        if operation == "setup":
            if 'repo_url' not in args or 'base_commit' not in args:
                raise ValueError("setup operation requires 'repo_url' and 'base_commit'")
            # instance_id is optional - when provided, creates isolated workspace per instance
            instance_id = args.get('instance_id')
            context = await repo_manager.setup_repository(
                args['repo_url'], args['base_commit'], instance_id
            )
            return str(context.repo_path)

        if operation == "cleanup":
            await repo_manager.cleanup()
            return "Cleanup complete"

        raise ValueError(f"Unknown operation: {operation}. Supported: 'setup', 'cleanup'")

    try:
        yield FunctionInfo.from_fn(git_operations,
                                   description="Git repository management tool that accepts JSON string arguments")
    finally:
        if tool_config.cleanup_on_exit:
            try:
                await repo_manager.cleanup()
                logger.info("Workspace cleanup completed successfully")
            except Exception as e:
                logger.error("Workspace cleanup failed: %s", e, exc_info=True)
                # Don't raise - allow graceful degradation
