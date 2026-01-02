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

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from nat.builder.function import FunctionGroup
from nat.builder.workflow_builder import Builder
from nat.cli.register_workflow import register_per_user_function_group
from nat.plugins.a2a.client.client_base import A2ABaseClient
from nat.plugins.a2a.client.client_config import A2AClientConfig

if TYPE_CHECKING:
    from nat.authentication.interfaces import AuthProviderBase

logger = logging.getLogger(__name__)


# Input models for helper functions
class GetTaskInput(BaseModel):
    """Input for get_task function."""
    task_id: str = Field(..., description="The ID of the task to retrieve")
    history_length: int | None = Field(default=None, description="Number of history items to include")


class CancelTaskInput(BaseModel):
    """Input for cancel_task function."""
    task_id: str = Field(..., description="The ID of the task to cancel")


class SendMessageInput(BaseModel):
    """Input for send_message function."""
    query: str = Field(..., description="The query to send to the agent")
    task_id: str | None = Field(default=None, description="Optional task ID for continuation")
    context_id: str | None = Field(default=None, description="Optional context ID for session management")


class A2AClientFunctionGroup(FunctionGroup):
    """
    A minimal FunctionGroup for A2A agents.

    Exposes a simple `send_message` function to interact with A2A agents.
    """

    def __init__(self, config: A2AClientConfig, builder: Builder):
        super().__init__(config=config)
        self._builder = builder
        self._client: A2ABaseClient | None = None
        self._include_skills_in_description = config.include_skills_in_description

    async def __aenter__(self):
        """Initialize the A2A client and register functions."""
        config: A2AClientConfig = self._config  # type: ignore[assignment]
        base_url = str(config.url)

        # Get user_id from context (set by runtime for per-user function groups)
        from nat.builder.context import Context
        user_id = Context.get().user_id
        if not user_id:
            raise RuntimeError("User ID not found in context")

        # Resolve auth provider if configured
        auth_provider: AuthProviderBase | None = None
        if config.auth_provider:
            try:
                auth_provider = await self._builder.get_auth_provider(config.auth_provider)
                logger.info("Resolved authentication provider for A2A client")
            except Exception as e:
                logger.error("Failed to resolve auth provider '%s': %s", config.auth_provider, e)
                raise RuntimeError(f"Failed to resolve auth provider: {e}") from e

        # Create and initialize A2A client
        self._client = A2ABaseClient(
            base_url=base_url,
            agent_card_path=config.agent_card_path,
            task_timeout=config.task_timeout,
            streaming=config.streaming,
            auth_provider=auth_provider,
        )
        await self._client.__aenter__()

        if auth_provider:
            logger.info("Connected to A2A agent at %s with authentication (user_id: %s)", base_url, user_id)
        else:
            logger.info("Connected to A2A agent at %s (user_id: %s)", base_url, user_id)

        # Discover agent card and register functions
        self._register_functions()

        return self

    def _register_functions(self):
        """Retrieve agent card and register the three-level API: high-level, helpers, and low-level."""
        # Validate client is initialized
        if not self._client:
            raise RuntimeError("A2A client not initialized")

        # Get and validate agent card
        agent_card = self._client.agent_card
        if not agent_card:
            raise RuntimeError("Agent card not available")

        # Log agent information
        logger.info("Agent: %s v%s", agent_card.name, agent_card.version)
        if agent_card.skills:
            logger.info("Skills: %s", [skill.name for skill in agent_card.skills])

        # Register functions
        # LEVEL 1: High-level main function (LLM-friendly)
        self.add_function(
            name="call",
            fn=self._create_high_level_function(),
            description=self._format_main_function_description(agent_card),
        )

        # LEVEL 2: Standard helpers (metadata/utility)
        self.add_function(
            name="get_skills",
            fn=self._get_skills,
            description="Get the list of skills and capabilities available from this agent",
        )

        self.add_function(
            name="get_info",
            fn=self._get_agent_info,
            description="Get metadata about this agent (name, version, provider, capabilities)",
        )

        self.add_function(
            name="get_task",
            fn=self._wrap_get_task,
            description="Get the status and details of a specific task by task_id",
        )

        self.add_function(
            name="cancel_task",
            fn=self._wrap_cancel_task,
            description="Cancel a running task by task_id",
        )

        # LEVEL 3: Low-level protocol (advanced)
        self.add_function(
            name="send_message",
            fn=self._send_message_advanced,
            description=("Advanced: Send a message with full control over the A2A protocol. "
                         "Returns raw events as a list. For most use cases, prefer using the "
                         "high-level 'call()' function instead."),
        )

        self.add_function(
            name="send_message_streaming",
            fn=self._send_message_streaming,
            description=("Advanced: Send a message and stream response events as they arrive. "
                         "Yields raw events one by one. This is an async generator function. "
                         "For most use cases, prefer using the high-level 'call()' function instead."),
        )

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Clean up the A2A client."""
        if self._client:
            await self._client.__aexit__(exc_type, exc_value, traceback)
            self._client = None
            logger.info("Disconnected from A2A agent")

    def _format_main_function_description(self, agent_card) -> str:
        """Create description for the main agent function."""
        description = f"{agent_card.description}\n\n"

        # Conditionally include skills based on configuration
        if self._include_skills_in_description and agent_card.skills:
            description += "**Capabilities:**\n"
            for skill in agent_card.skills:
                description += f"\n• **{skill.name}**: {skill.description}"
                if skill.examples:
                    examples = skill.examples[:2]  # Limit to 2 examples
                    description += f"\n  Examples: {', '.join(examples)}"
            description += "\n\n"
        elif agent_card.skills:
            # Brief mention that skills are available
            description += f"**{len(agent_card.skills)} capabilities available.** "
            description += "Use get_skills() for detailed information.\n\n"

        description += "**Usage:** Send natural language queries to interact with this agent."

        return description

    def _create_high_level_function(self):
        """High-level function that simplifies the response."""

        async def high_level_fn(query: str, task_id: str | None = None, context_id: str | None = None) -> str:
            """
            Send a query to the agent and get a simple text response.

            This is the recommended method for LLM usage.
            For advanced use cases, use send_message() for raw events.
            """
            if not self._client:
                raise RuntimeError("A2A client not initialized")

            events = []
            async for event in self._client.send_message(query, task_id, context_id):
                events.append(event)

            # Extract and return just the text response using base client helper
            return self._client.extract_text_from_events(events)

        return high_level_fn

    async def _get_skills(self, params: dict | None = None) -> dict:
        """Helper function to list agent skills."""
        if not self._client or not self._client.agent_card:
            return {"skills": []}

        agent_card = self._client.agent_card
        return {
            "agent":
                agent_card.name,
            "skills": [{
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "examples": skill.examples or [],
                "tags": skill.tags or []
            } for skill in agent_card.skills]
        }

    async def _get_agent_info(self, params: dict | None = None) -> dict:
        """Helper function to get agent metadata."""
        if not self._client or not self._client.agent_card:
            return {}

        agent_card = self._client.agent_card
        return {
            "name": agent_card.name,
            "description": agent_card.description,
            "version": agent_card.version,
            "provider": agent_card.provider.model_dump() if agent_card.provider else None,
            "url": agent_card.url,
            "capabilities": {
                "streaming": agent_card.capabilities.streaming if agent_card.capabilities else False,
            },
            "num_skills": len(agent_card.skills)
        }

    async def _wrap_get_task(self, params: GetTaskInput) -> Any:
        """Wrapper for get_task that delegates to client_base."""
        if not self._client:
            raise RuntimeError("A2A client not initialized")
        return await self._client.get_task(params.task_id, params.history_length)

    async def _wrap_cancel_task(self, params: CancelTaskInput) -> Any:
        """Wrapper for cancel_task that delegates to client_base."""
        if not self._client:
            raise RuntimeError("A2A client not initialized")
        return await self._client.cancel_task(params.task_id)

    async def _send_message_advanced(self, params: SendMessageInput) -> list:
        """
        Send a message with full A2A protocol control.

        Returns: List of ClientEvent|Message objects containing:
        - Task information
        - Status updates
        - Artifact updates
        - Full message details
        """
        if not self._client:
            raise RuntimeError("A2A client not initialized")

        events = []
        async for event in self._client.send_message(params.query, params.task_id, params.context_id):
            events.append(event)
        return events

    async def _send_message_streaming(self, params: SendMessageInput) -> AsyncGenerator[Any, None]:
        """
        Send a message with full A2A protocol control and stream events.

        This is an async generator that yields events as they arrive from the agent.

        Yields: ClientEvent|Message objects containing:
        - Task information
        - Status updates
        - Artifact updates
        - Full message details
        """
        if not self._client:
            raise RuntimeError("A2A client not initialized")

        async for event in self._client.send_message_streaming(params.query, params.task_id, params.context_id):
            yield event


@register_per_user_function_group(config_type=A2AClientConfig)
async def a2a_client_function_group(config: A2AClientConfig, _builder: Builder):
    """
    Connect to an A2A agent, discover agent card and publish the primary
    agent function and helper functions. This function group is per-user,
    meaning each user gets their own isolated instance.

    This function group creates a three-level API:
    - High-level: Agent function named after the agent (e.g., dice_agent)
    - Helpers: get_skills, get_info, get_task, cancel_task
    - Low-level: send_message for advanced usage
    """
    async with A2AClientFunctionGroup(config, _builder) as group:
        yield group
