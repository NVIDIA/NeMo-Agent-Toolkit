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

# Register all the tools needed by the full predictor without loading the dependencies.
import typing

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig


class GitRepoToolConfig(FunctionBaseConfig, name="git_repo_tool"):
    """Configuration for git repository management tool."""
    _type: typing.Literal["git_repo_tool"] = "git_repo_tool"
    workspace_dir: str = "./.workspace"  # Base directory for cloning repositories
    cleanup_on_exit: bool = True  # Whether to clean up repos after use


@register_function(config_type=GitRepoToolConfig)
async def git_repo_tool(tool_config: GitRepoToolConfig, builder: Builder):
    """Git repository management tool for SWE Bench."""
    import json

    from .git_tool import RepoManager
    repo_manager = RepoManager(tool_config.workspace_dir)

    # Simple async function that accepts a JSON string
    async def git_operations(args_str: str) -> str:
        args = json.loads(args_str)
        operation = args.get('operation')

        if operation == "setup":
            context = await repo_manager.setup_repository(args['repo_url'], args['base_commit'])
            return str(context.repo_path)

        if operation == "cleanup":
            await repo_manager.cleanup()
            return "Cleanup complete"

        raise ValueError(f"Unknown operation: {operation}")

    try:
        yield FunctionInfo.from_fn(git_operations,
                                   description="Git repository management tool that accepts JSON string arguments")
    finally:
        if tool_config.cleanup_on_exit:
            await repo_manager.cleanup()
