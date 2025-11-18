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
"""Worker for setting up and running the A2A server."""

import logging

import httpx

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import BasePushNotificationSender
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentSkill
from nat.builder.function import Function
from nat.builder.workflow import Workflow
from nat.data_models.config import Config
from nat.plugins.a2a.server.agent_executor_adapter import NATWorkflowAgentExecutor
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


class A2AFrontEndPluginWorker:
    """Worker that handles A2A server setup and configuration."""

    def __init__(self, config: Config):
        """Initialize the A2A worker with configuration.

        Args:
            config: The full NAT configuration
        """
        self.full_config = config
        self.front_end_config: A2AFrontEndConfig = config.general.front_end  # type: ignore

        # Max concurrency for handling A2A tasks
        # This limits how many workflow invocations can run simultaneously
        self.max_concurrency = 8  # Could be made configurable

    async def _get_all_functions(self, workflow: Workflow) -> dict[str, Function]:
        """Get all functions from the workflow.

        Args:
            workflow: The NAT workflow

        Returns:
            Dict mapping function names to Function objects
        """
        functions: dict[str, Function] = {}

        # Extract all functions from the workflow
        functions.update(workflow.functions)
        for function_group in workflow.function_groups.values():
            functions.update(await function_group.get_accessible_functions())

        return functions

    async def create_agent_card(self, workflow: Workflow) -> AgentCard:
        """Build AgentCard from configuration and workflow functions.

        Skills are auto-generated from the workflow's functions, similar to how
        MCP introspects and exposes functions as tools.

        Args:
            workflow: The NAT workflow to extract functions from

        Returns:
            AgentCard with agent metadata, capabilities, and auto-generated skills
        """
        config = self.front_end_config

        # Build capabilities
        capabilities = AgentCapabilities(
            streaming=config.capabilities.streaming,
            push_notifications=config.capabilities.push_notifications,
        )

        # Auto-generate skills from workflow functions
        functions = await self._get_all_functions(workflow)
        skills = []

        for function_name, function in functions.items():
            # Create skill from function metadata
            skill_name = function_name.replace('_', ' ').replace('.', ' - ').title()
            skill_description = function.description or f"Execute {function_name}"

            skill = AgentSkill(
                id=function_name,
                name=skill_name,
                description=skill_description,
                tags=[],  # Could be extended with function metadata
                examples=[],  # Could be extracted from function examples if available
            )
            skills.append(skill)

        logger.info("Auto-generated %d skills from workflow functions", len(skills))

        # Build agent card
        agent_url = f"http://{config.host}:{config.port}/"
        agent_card = AgentCard(
            name=config.name,
            description=config.description,
            url=agent_url,
            version=config.version,
            default_input_modes=config.default_input_modes,
            default_output_modes=config.default_output_modes,
            capabilities=capabilities,
            skills=skills,
        )

        logger.info("Created AgentCard for: %s v%s", config.name, config.version)
        logger.info("Agent URL: %s", agent_url)
        logger.info("Skills: %d", len(skills))

        return agent_card

    def create_agent_executor(self, workflow: Workflow) -> NATWorkflowAgentExecutor:
        """Create agent executor adapter for the workflow.

        This creates a SessionManager to handle concurrent A2A task requests,
        similar to how FastAPI handles multiple HTTP requests.

        Args:
            workflow: The NAT workflow to expose

        Returns:
            NATWorkflowAgentExecutor that wraps the workflow with a SessionManager
        """
        # Create SessionManager to handle concurrent requests with proper limits
        session_manager = SessionManager(
            workflow=workflow,
            max_concurrency=self.max_concurrency,
        )

        logger.info("Created SessionManager with max_concurrency=%d", self.max_concurrency)

        return NATWorkflowAgentExecutor(session_manager)

    def create_a2a_server(
        self,
        agent_card: AgentCard,
        agent_executor: NATWorkflowAgentExecutor,
    ) -> A2AStarletteApplication:
        """Create A2A server with the agent executor.

        Args:
            agent_card: The agent card describing the agent
            agent_executor: The executor that handles task processing

        Returns:
            Configured A2A Starlette application
        """
        # Create HTTP client for push notifications
        httpx_client = httpx.AsyncClient()

        # Create push notification infrastructure
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store,
        )

        # Create request handler
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender,
        )

        # Create A2A server
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        logger.info("Created A2A server with DefaultRequestHandler")

        return server
