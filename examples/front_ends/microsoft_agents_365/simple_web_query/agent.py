# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Web Query Agent — wraps the NeMo Agent Toolkit simple_web_query workflow
behind the A365 AgentInterface so it can be hosted by the generic agent host.
"""

import logging
import os
from pathlib import Path

from agent_interface import AgentInterface
from microsoft_agents.hosting.core import Authorization, TurnContext

logger = logging.getLogger(__name__)

# A365 observability (optional — gracefully degrades if not installed)
try:
    from microsoft_agents_a365.observability.core.middleware.baggage_builder import BaggageBuilder
    from microsoft_agents_a365.observability.extensions.langchain import CustomLangChainInstrumentor
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

# Default config path relative to this file
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "config.yml"


class WebQueryAgent(AgentInterface):
    """A365 agent that answers questions using NeMo Agent Toolkit's ReAct web query workflow."""

    def __init__(self):
        self._workflow_builder = None
        self._session_manager = None
        self._config = None
        self._initialized = False

    async def initialize(self) -> None:
        # Defer heavy initialization to first message so the server starts fast
        # and Azure health probes succeed within the timeout.
        logger.info("WebQueryAgent registered — workflow will initialize on first message")

    async def _ensure_workflow(self) -> None:
        """Lazily initialize the NAT workflow on first use."""
        if self._initialized:
            return

        from nat.builder.workflow_builder import WorkflowBuilder
        from nat.runtime.loader import load_config
        from nat.runtime.session import SessionManager

        # Register the webpage_query tool (not available as a PyPI package)
        import web_query_tool  # noqa: F401 — triggers @register_function decorator

        config_path = os.getenv("NAT_CONFIG_FILE", str(DEFAULT_CONFIG))
        logger.info(f"Loading NAT config from: {config_path}")

        self._config = load_config(config_path)
        self._workflow_builder = WorkflowBuilder.from_config(config=self._config)
        self._workflow_builder = await self._workflow_builder.__aenter__()
        self._session_manager = await SessionManager.create(
            config=self._config, shared_builder=self._workflow_builder
        )

        self._initialized = True

        # Enable LangChain auto-instrumentation for free traces of LLM calls + tool use
        if OBSERVABILITY_AVAILABLE:
            try:
                CustomLangChainInstrumentor()
                logger.info("LangChain auto-instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to enable LangChain instrumentation: {e}")

        logger.info("WebQueryAgent workflow initialized — ready to process messages")

    async def process_user_message(
        self,
        message: str,
        auth: Authorization,
        context: TurnContext,
        auth_handler_name: str | None = None,
    ) -> str:
        await self._ensure_workflow()

        # Extract tenant/agent IDs for observability baggage
        tenant_id = context.activity.recipient.tenant_id if context.activity.recipient else None
        agent_id = context.activity.recipient.agentic_app_id if context.activity.recipient else None

        logger.info(f"Running NAT workflow for: {message}")

        if OBSERVABILITY_AVAILABLE and tenant_id and agent_id:
            with BaggageBuilder().tenant_id(tenant_id).agent_id(agent_id).build():
                result = await self._run_workflow(message)
        else:
            result = await self._run_workflow(message)

        logger.info(f"Workflow result: {result[:200] if len(result) > 200 else result}")
        return result

    async def _run_workflow(self, message: str) -> str:
        """Execute the NAT workflow."""
        async with self._session_manager.session() as session:
            async with session.run(message) as runner:
                return await runner.result(to_type=str)

    async def cleanup(self) -> None:
        if self._session_manager:
            await self._session_manager.shutdown()
            self._session_manager = None

        if self._workflow_builder:
            await self._workflow_builder.__aexit__(None, None, None)
            self._workflow_builder = None

        logger.info("WebQueryAgent cleaned up")
