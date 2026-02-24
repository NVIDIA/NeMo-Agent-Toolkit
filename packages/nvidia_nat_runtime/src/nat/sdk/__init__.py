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
"""NAT SDK — a self-contained toolkit for building agents.

Usage::

    from nat.sdk import Agent, Conversation, Tool, tool

    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    agent = Agent(tools=[add], system_prompt="You are a helpful assistant.")
    conv = Conversation(agent=agent, client=my_llm_client)
    response = await conv.send_message("What is 2 + 3?")
"""

# Agent
# Skills
from nat.data_models.skill import Skill

# Workspace
from nat.data_models.workspace import ActionRequest
from nat.data_models.workspace import ActionResult
from nat.data_models.workspace import ActionStatus
from nat.sdk.agent.agent import Agent
from nat.sdk.agent.state import AgentState
from nat.sdk.agent.state import AgentStatus

# Conversation
from nat.sdk.conversation.conversation import Conversation
from nat.sdk.conversation.state import ConversationState
from nat.sdk.conversation.state import ConversationStatus
from nat.sdk.conversation.state import UsageStats

# Events
from nat.sdk.event.event import ActionEvent
from nat.sdk.event.event import ErrorEvent
from nat.sdk.event.event import Event
from nat.sdk.event.event import EventSource
from nat.sdk.event.event import MessageEvent
from nat.sdk.event.event import ObservationEvent
from nat.sdk.event.event import SystemPromptEvent
from nat.sdk.event.log import EventLog

# LLM
from nat.sdk.llm.client import LLMClient
from nat.sdk.llm.message import LLMResponse
from nat.sdk.llm.message import Message
from nat.sdk.llm.message import TokenUsage
from nat.sdk.llm.message import ToolCall

# Tools
from nat.sdk.tool.result import ToolResult
from nat.sdk.tool.tool import Tool
from nat.sdk.tool.tool import tool
from nat.workspace.types import WorkspaceBase

__all__ = [
    # Agent
    "Agent",
    "AgentState",
    "AgentStatus",  # Conversation
    "Conversation",
    "ConversationState",
    "ConversationStatus",
    "UsageStats",  # Events
    "ActionEvent",
    "ErrorEvent",
    "Event",
    "EventLog",
    "EventSource",
    "MessageEvent",
    "ObservationEvent",
    "SystemPromptEvent",  # LLM
    "BuilderLLMClient",
    "LLMClient",
    "LLMResponse",
    "Message",
    "TokenUsage",
    "ToolCall",
    "create_llm_client",  # Skills
    "Skill",  # Tools
    "Tool",
    "ToolResult",
    "tool",  # Workspace
    "ActionRequest",
    "ActionResult",
    "ActionStatus",
    "WorkspaceBase",
]


def __getattr__(name: str):  # noqa: ANN001
    if name == "BuilderLLMClient":
        from nat.sdk.llm.builder_client import BuilderLLMClient

        return BuilderLLMClient
    if name == "create_llm_client":
        from nat.sdk.llm.builder_client import create_llm_client

        return create_llm_client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
