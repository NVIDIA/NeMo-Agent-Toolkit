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

from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from nat.data_models.skill import Skill
from nat.sdk.tool.tool import Tool
from nat.workspace.types import WorkspaceBase


class Agent(BaseModel):
    """Immutable agent configuration.

    An agent defines *what* an AI assistant is (tools, skills,
    system prompt, workspace) but carries no runtime state.  Pair with a
    :class:`~nat.sdk.conversation.conversation.Conversation` to execute.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    # Capabilities
    tools: list[Tool] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    skill_discovery_dirs: tuple[str | Path, ...] = Field(default=())
    workspace: WorkspaceBase | None = Field(default=None)

    # System prompt
    system_prompt: str = ""
    system_prompt_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Execution limits
    max_iterations: int = 50

    def render_system_prompt(self, **extra_kwargs: Any) -> str:
        """Render the system prompt template with kwargs.

        Merges ``system_prompt_kwargs`` with any additional keyword
        arguments provided at call time.
        """
        if not self.system_prompt:
            return ""
        kwargs = {**self.system_prompt_kwargs, **extra_kwargs}
        if kwargs:
            try:
                return self.system_prompt.format(**kwargs)
            except KeyError:
                return self.system_prompt
        return self.system_prompt
