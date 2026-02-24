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

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ToolCall(BaseModel):
    """A tool invocation requested by an LLM."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_openai_dict(self) -> dict[str, Any]:
        """Serialize to the OpenAI chat completion tool_call format."""
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


class Message(BaseModel):
    """A transport-agnostic message in a conversation.

    Represents system, user, assistant, and tool messages.  Assistant messages
    may optionally carry ``tool_calls``.  Tool result messages carry a
    ``tool_call_id`` linking them back to the originating call.
    """

    model_config = ConfigDict(frozen=True)

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None  # set for role="tool" messages

    def to_openai_dict(self) -> dict[str, Any]:
        """Serialize to the OpenAI chat completion message format."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}

        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai_dict() for tc in self.tool_calls]

        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id

        return msg


class TokenUsage(BaseModel):
    """Token counts for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0


class LLMResponse(BaseModel):
    """The result of an LLM completion call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: Message
    usage: TokenUsage | None = None
    model: str | None = None
    finish_reason: str | None = None
    raw: Any = Field(default=None, exclude=True)
