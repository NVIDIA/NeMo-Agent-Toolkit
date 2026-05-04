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
"""
Per-user workflow for the per-user workflow example.

This workflow demonstrates how to create a per-user workflow that uses
per-user functions. The workflow itself is per-user, meaning each user
gets their own workflow instance with isolated state.
"""

import logging

from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.function import FunctionBaseConfig
from nat_per_user_workflow.per_user_functions import NoteInput
from nat_per_user_workflow.per_user_functions import NoteOutput
from nat_per_user_workflow.per_user_functions import PreferenceInput
from nat_per_user_workflow.per_user_functions import PreferenceOutput

logger = logging.getLogger(__name__)


# ============= Schemas =============
class UserAssistantInput(BaseModel):
    """Input for the user assistant workflow."""
    command: str = Field(description="Command to execute: 'note', 'pref', 'stats', or 'help'")
    action: str = Field(default="", description="Action for the command")
    param1: str = Field(default="", description="First parameter (key/content)")
    param2: str = Field(default="", description="Second parameter (value)")


class UserAssistantOutput(BaseModel):
    """Output from the user assistant workflow."""
    success: bool = Field(description="Whether the command succeeded")
    message: str = Field(description="Result message")
    data: dict = Field(default_factory=dict, description="Additional data from the command")


# ============= Config =============
class PerUserAssistantConfig(FunctionBaseConfig, name="per_user_assistant"):
    """Configuration for the per-user assistant workflow."""
    notepad_name: str = Field(default="notepad", description="Name of the notepad function")
    preferences_name: str = Field(default="preferences", description="Name of the preferences function")


# ============= Per-User Workflow =============
@register_per_user_function(config_type=PerUserAssistantConfig,
                            input_type=UserAssistantInput,
                            single_output_type=UserAssistantOutput)
async def per_user_assistant_workflow(config: PerUserAssistantConfig, builder: Builder):
    """
    A per-user assistant workflow that combines notepad and preferences.

    This workflow is per-user, meaning each user gets their own instance.
    It orchestrates calls to per-user functions (notepad, preferences) and
    provides a unified interface for users.

    Commands:
    - note add <content>: Add a note
    - note list: List all notes
    - note clear: Clear all notes
    - note count: Count notes
    - pref set <key> <value>: Set a preference
    - pref get <key>: Get a preference
    - pref list: List all preferences
    - help: Show help message
    """
    # Get per-user functions
    notepad_fn = await builder.get_function(config.notepad_name)
    preferences_fn = await builder.get_function(config.preferences_name)

    # Track session stats (also per-user state)
    session_stats = {"commands_executed": 0}

    logger.info("Creating new per-user assistant workflow instance")

    async def _assistant(inp: UserAssistantInput) -> UserAssistantOutput:
        session_stats["commands_executed"] += 1
        command = inp.command.lower()

        if command == "help":
            return UserAssistantOutput(success=True,
                                       message="""
Available commands:
- note add <content>: Add a note
- note list: List all notes
- note clear: Clear all notes
- note count: Count notes
- pref set <key> <value>: Set a preference
- pref get <key>: Get a preference
- pref list: List all preferences
- stats: Show session statistics
- help: Show this help message
                """.strip(),
                                       data={"commands_executed": session_stats["commands_executed"]})

        elif command == "stats":
            return UserAssistantOutput(
                success=True,
                message=f"Session statistics: {session_stats['commands_executed']} commands executed",
                data={"commands_executed": session_stats["commands_executed"]})

        elif command == "note":
            note_input = NoteInput(action=inp.action, content=inp.param1)
            result = await notepad_fn.ainvoke(note_input, to_type=NoteOutput)

            return UserAssistantOutput(success=result.success,
                                       message=result.message,
                                       data={
                                           "notes": result.notes,
                                           "count": result.count,
                                           "commands_executed": session_stats["commands_executed"]
                                       })

        elif command == "pref":
            pref_input = PreferenceInput(action=inp.action, key=inp.param1, value=inp.param2)
            result = await preferences_fn.ainvoke(pref_input, to_type=PreferenceOutput)

            return UserAssistantOutput(success=result.success,
                                       message=result.message,
                                       data={
                                           "value": result.value,
                                           "preferences": result.preferences,
                                           "commands_executed": session_stats["commands_executed"]
                                       })

        else:
            return UserAssistantOutput(success=False,
                                       message=f"Unknown command: {command}. Use 'help' to see available commands.",
                                       data={"commands_executed": session_stats["commands_executed"]})

    yield FunctionInfo.from_fn(_assistant)
