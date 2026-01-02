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

import httpx

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import BasePushNotificationSender
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentSkill
from a2a.types import SecurityScheme
from nat.builder.function import Function
from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import WorkflowBuilder
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

        # Max concurrency for handling A2A tasks (from configuration)
        # This limits how many workflow invocations can run simultaneously
        self.max_concurrency = self.front_end_config.max_concurrency

        # HTTP client for push notifications (managed for cleanup)
        self._httpx_client: httpx.AsyncClient | None = None

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

    async def _generate_security_schemes(
            self, server_auth_config) -> tuple[dict[str, SecurityScheme], list[dict[str, list[str]]]]:
        """Generate A2A security schemes from OAuth2ResourceServerConfig.

        Args:
            server_auth_config: OAuth2ResourceServerConfig

        Returns:
            Tuple of (security_schemes dict, security requirements list)
        """
        from a2a.types import AuthorizationCodeOAuthFlow
        from a2a.types import OAuth2SecurityScheme
        from a2a.types import OAuthFlows

        # Resolve OAuth2 endpoints from configuration
        auth_url, token_url = await self._resolve_oauth_endpoints(server_auth_config)

        # Create scope descriptions
        scope_descriptions = {scope: f"Permission: {scope}" for scope in server_auth_config.scopes}

        # Build OAuth2 security scheme
        security_schemes = {
            "oauth2":
                SecurityScheme(root=OAuth2SecurityScheme(
                    type="oauth2",
                    description="OAuth 2.0 authentication required to access this agent",
                    flows=OAuthFlows(authorizationCode=AuthorizationCodeOAuthFlow(
                        authorizationUrl=auth_url,
                        tokenUrl=token_url,
                        scopes=scope_descriptions,
                    )),
                ))
        }

        # Security requirements (scopes needed)
        security = [{"oauth2": server_auth_config.scopes}]

        return security_schemes, security

    async def _resolve_oauth_endpoints(self, server_auth_config) -> tuple[str, str]:
        """Resolve authorization and token URLs from OAuth2 configuration.

        Args:
            server_auth_config: OAuth2ResourceServerConfig

        Returns:
            Tuple of (authorization_url, token_url)
        """
        import httpx

        # If discovery URL is provided, use OIDC discovery
        if server_auth_config.discovery_url:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(server_auth_config.discovery_url, timeout=5.0)
                    response.raise_for_status()
                    metadata = response.json()

                    auth_url = metadata.get("authorization_endpoint")
                    token_url = metadata.get("token_endpoint")

                    if auth_url and token_url:
                        logger.info("Resolved OAuth endpoints via discovery: %s", server_auth_config.discovery_url)
                        return auth_url, token_url
            except Exception as e:
                logger.warning("Failed to discover OAuth endpoints: %s", e)

        # Fallback: derive from issuer URL (common convention)
        issuer = server_auth_config.issuer_url.rstrip("/")
        auth_url = f"{issuer}/oauth/authorize"
        token_url = f"{issuer}/oauth/token"

        logger.info("Using derived OAuth endpoints from issuer: %s", issuer)
        return auth_url, token_url

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

        # Generate security schemes if server_auth is configured
        security_schemes = None
        security = None

        if config.server_auth:
            security_schemes, security = await self._generate_security_schemes(config.server_auth)
            logger.info(
                "Generated OAuth2 security schemes for agent (issuer=%s, scopes=%s)",
                config.server_auth.issuer_url,
                config.server_auth.scopes,
            )

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
            security_schemes=security_schemes,
            security=security,
        )

        logger.info("Created AgentCard for: %s v%s", config.name, config.version)
        logger.info("Agent URL: %s", agent_url)
        logger.info("Skills: %d", len(skills))
        if security_schemes:
            logger.info("Security: OAuth2 authentication required")

        return agent_card

    def create_agent_executor(self, workflow: Workflow, builder: WorkflowBuilder) -> NATWorkflowAgentExecutor:
        """Create agent executor adapter for the workflow.

        This creates a SessionManager to handle concurrent A2A task requests,
        similar to how FastAPI handles multiple HTTP requests.

        Args:
            workflow: The NAT workflow to expose
            builder: The workflow builder used to create the workflow

        Returns:
            NATWorkflowAgentExecutor that wraps the workflow with a SessionManager
        """
        # Create SessionManager to handle concurrent requests with proper limits
        session_manager = SessionManager(
            config=self.full_config,
            shared_builder=builder,
            shared_workflow=workflow,
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

        Note:
            The httpx client is stored in self._httpx_client for lifecycle management.
            Call cleanup() during server shutdown to properly close the client.
        """
        # Create HTTP client for push notifications and store for cleanup
        self._httpx_client = httpx.AsyncClient()

        # Create push notification infrastructure
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=self._httpx_client,
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

    async def cleanup(self) -> None:
        """Clean up resources, particularly the httpx client.

        This should be called during server shutdown to prevent connection leaks.
        """
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None
            logger.info("Closed httpx client for push notifications")
