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
Per-user functions for the per-user workflow example.

This module demonstrates how to create functions with per-user state using
the @register_per_user_function decorator. Each user gets their own isolated
instance of the function with separate state.
"""

import logging
from datetime import datetime

from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


# ============= Schemas =============
class NoteInput(BaseModel):
    """Input for note operations."""
    action: str = Field(description="Action to perform: 'add', 'list', 'clear', or 'count'")
    content: str = Field(default="", description="Note content (for 'add' action)")


class NoteOutput(BaseModel):
    """Output from note operations."""
    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Result message")
    notes: list[str] = Field(default_factory=list, description="List of notes (for 'list' action)")
    count: int = Field(default=0, description="Number of notes")


class PreferenceInput(BaseModel):
    """Input for preference operations."""
    action: str = Field(description="Action to perform: 'set', 'get', or 'list'")
    key: str = Field(default="", description="Preference key")
    value: str = Field(default="", description="Preference value (for 'set' action)")


class PreferenceOutput(BaseModel):
    """Output from preference operations."""
    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Result message")
    value: str = Field(default="", description="Preference value (for 'get' action)")
    preferences: dict[str, str] = Field(default_factory=dict, description="All preferences (for 'list' action)")


# ============= Configs =============
class PerUserNotepadConfig(FunctionBaseConfig, name="per_user_notepad"):
    """Configuration for the per-user notepad function."""
    max_notes: int = Field(default=100, description="Maximum number of notes per user")


class PerUserPreferencesConfig(FunctionBaseConfig, name="per_user_preferences"):
    """Configuration for the per-user preferences function."""
    default_preferences: dict[str, str] = Field(default_factory=lambda: {
        "theme": "light", "language": "en"
    },
                                                description="Default preferences for new users")


# ============= Per-User Functions =============
@register_per_user_function(config_type=PerUserNotepadConfig, input_type=NoteInput, single_output_type=NoteOutput)
async def per_user_notepad(config: PerUserNotepadConfig, builder: Builder):
    """
    A per-user notepad that stores notes separately for each user.

    Each user gets their own isolated notepad - notes added by one user
    are not visible to other users.
    """
    # This state is unique per user - created fresh for each user
    user_notes: list[dict[str, str]] = []

    logger.info(f"Creating new notepad instance (max_notes={config.max_notes})")

    async def _notepad(inp: NoteInput) -> NoteOutput:
        action = inp.action.lower()

        if action == "add":
            if not inp.content:
                return NoteOutput(success=False, message="Content is required for 'add' action", count=len(user_notes))
            if len(user_notes) >= config.max_notes:
                return NoteOutput(success=False,
                                  message=f"Maximum notes ({config.max_notes}) reached",
                                  count=len(user_notes))

            user_notes.append({"content": inp.content, "timestamp": datetime.now().isoformat()})
            return NoteOutput(success=True, message="Note added successfully", count=len(user_notes))

        elif action == "list":
            notes_content = [note["content"] for note in user_notes]
            return NoteOutput(success=True,
                              message=f"Found {len(user_notes)} notes",
                              notes=notes_content,
                              count=len(user_notes))

        elif action == "clear":
            count = len(user_notes)
            user_notes.clear()
            return NoteOutput(success=True, message=f"Cleared {count} notes", count=0)

        elif action == "count":
            return NoteOutput(success=True, message=f"You have {len(user_notes)} notes", count=len(user_notes))

        else:
            return NoteOutput(success=False,
                              message=f"Unknown action: {action}. Use 'add', 'list', 'clear', or 'count'",
                              count=len(user_notes))

    yield FunctionInfo.from_fn(_notepad)


@register_per_user_function(config_type=PerUserPreferencesConfig,
                            input_type=PreferenceInput,
                            single_output_type=PreferenceOutput)
async def per_user_preferences(config: PerUserPreferencesConfig, builder: Builder):
    """
    A per-user preferences store.

    Each user gets their own isolated preferences - settings changed by one user
    do not affect other users.
    """
    # This state is unique per user - initialized with defaults
    user_preferences: dict[str, str] = dict(config.default_preferences)

    logger.info(f"Creating new preferences instance with defaults: {user_preferences}")

    async def _preferences(inp: PreferenceInput) -> PreferenceOutput:
        action = inp.action.lower()

        if action == "set":
            if not inp.key:
                return PreferenceOutput(success=False, message="Key is required for 'set' action")
            user_preferences[inp.key] = inp.value
            return PreferenceOutput(success=True,
                                    message=f"Preference '{inp.key}' set to '{inp.value}'",
                                    preferences=user_preferences)

        elif action == "get":
            if not inp.key:
                return PreferenceOutput(success=False, message="Key is required for 'get' action")
            value = user_preferences.get(inp.key, "")
            found = inp.key in user_preferences
            return PreferenceOutput(
                success=found,
                message=f"Preference '{inp.key}' = '{value}'" if found else f"Preference '{inp.key}' not found",
                value=value)

        elif action == "list":
            return PreferenceOutput(success=True,
                                    message=f"Found {len(user_preferences)} preferences",
                                    preferences=user_preferences)

        else:
            return PreferenceOutput(success=False, message=f"Unknown action: {action}. Use 'set', 'get', or 'list'")

    yield FunctionInfo.from_fn(_preferences)
