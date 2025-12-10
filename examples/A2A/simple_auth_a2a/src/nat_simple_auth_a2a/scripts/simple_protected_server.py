#!/usr/bin/env python3
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
"""
Simple protected A2A server for testing NeMo Agent toolkit CLI authentication.

This server uses simple Bearer token validation for testing purposes only.
For production OAuth setups with JWT and token exchange, see:
https://blog.christianposta.com/setting-up-a2a-oauth-user-delegation/
"""

import asyncio
import logging
import os

import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentSkill
from a2a.types import HTTPAuthSecurityScheme
from a2a.types import SecurityScheme
from a2a.utils import new_agent_text_message
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple token for testing - change using TEST_BEARER_TOKEN env var
VALID_TOKEN = os.getenv("TEST_BEARER_TOKEN", "test-token-12345")


class SimpleBearerAuthMiddleware(BaseHTTPMiddleware):
    """
    Simple Bearer token authentication middleware for testing.
    """

    async def dispatch(self, request: Request, call_next):
        """Validate Bearer token on all requests except public agent card."""
        # Allow public agent card discovery (follows A2A spec)
        # Support both current and deprecated paths
        if request.url.path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
            logger.info("✓ Public access to agent card")
            return await call_next(request)

        # Check Bearer token for all other endpoints
        auth_header = request.headers.get("Authorization", "")

        if not auth_header:
            logger.warning("✗ Missing Authorization header")
            return JSONResponse({"error": "unauthorized", "message": "Missing Authorization header"}, status_code=401)

        if not auth_header.startswith("Bearer "):
            logger.warning("✗ Invalid Authorization format")
            return JSONResponse({"error": "unauthorized", "message": "Invalid Authorization format"}, status_code=401)

        token = auth_header[7:]  # Strip "Bearer "

        # Simple token validation (for testing only)
        # Production should use JWT validation with JWKS endpoint
        if token != VALID_TOKEN:
            logger.warning(f"✗ Invalid token: {token[:10]}...")
            return JSONResponse({"error": "forbidden", "message": "Invalid Bearer token"}, status_code=403)

        logger.info(f"✓ Authentication successful with token: {token[:10]}...")

        # In production, decoded JWT claims would be attached here:
        # request.state.user_token = decoded_jwt

        return await call_next(request)


class SimpleAgentExecutor(AgentExecutor):
    """Simple agent that echoes back messages to confirm authentication."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Echo the user's message back with authentication confirmation."""
        user_input = context.get_user_input()

        # Create response
        response_text = f"✅ Authentication successful! You said: {user_input}"
        message = new_agent_text_message(
            response_text,
            context_id=context.context_id,
            task_id=context.task_id,
        )

        # Send response
        await event_queue.enqueue_event(message)
        logger.info("Sent response to authenticated user")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel method - not supported for this simple agent."""
        raise Exception("cancel not supported")


def create_agent_card(host: str = "localhost", port: int = 10001) -> AgentCard:
    """
    Create agent card with Bearer auth requirement.

    Follows the modern securitySchemes pattern from:
    https://blog.christianposta.com/setting-up-a2a-oauth-user-delegation/

    In production, you would:
    1. Specify bearerFormat="JWT"
    2. Define required scopes (for example, "agent:execute")
    3. Use OAuth 2.0 Token Exchange (RFC 8693) for delegation
    """
    return AgentCard(
        name="Simple Protected Test Agent",
        description="A minimal A2A agent protected with Bearer token authentication for testing NAT CLI auth",
        url=f"http://{host}:{port}/",
        version="0.1.0",
        default_input_modes=["text", "text/plain"],
        default_output_modes=["text", "text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="echo",
                name="Echo Message",
                description="Echo back your message to confirm authentication worked",
                tags=["test", "auth"],
                examples=["Hello", "Test authentication", "Who am I?"],
            )
        ],
        # Security configuration (modern format, not deprecated 'authentication')
        security_schemes={
            "bearer_auth":
                SecurityScheme(root=HTTPAuthSecurityScheme(
                    type="http",
                    scheme="bearer",
                    # In production, add: bearerFormat="JWT"
                    description="Bearer token authentication - simplified for testing",
                ))
        },
        # Require bearer_auth for all operations
        # In production, specify scopes: [{"bearer_auth": ["agent:execute"]}]
        security=[{
            "bearer_auth": []
        }],
    )


async def main():
    """Start the protected A2A server."""
    host = os.getenv("SIMPLE_PROTECTED_SERVER_HOST", "localhost")
    port = int(os.getenv("SIMPLE_PROTECTED_SERVER_PORT", "8000"))

    # Create agent card
    agent_card = create_agent_card(host, port)

    # Create agent executor
    agent_executor = SimpleAgentExecutor()

    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
    )

    # Create A2A server
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Build app and add auth middleware
    app = server.build()
    app.add_middleware(SimpleBearerAuthMiddleware)

    # Log startup info
    logger.info("=" * 60)
    logger.info("🔒 Simple Protected A2A Server Starting")
    logger.info("=" * 60)
    logger.info(f"URL: http://{host}:{port}/")
    logger.info(f"Agent Card: http://{host}:{port}/.well-known/agent-card.json")
    logger.info(f"Valid Bearer Token: {VALID_TOKEN}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Test commands:")
    logger.info("")
    logger.info("# Should FAIL (no auth):")
    logger.info(f"  nat a2a client call --url http://{host}:{port} --message 'Hello'")
    logger.info("")
    logger.info("# Should SUCCEED (with token):")
    logger.info(f"  nat a2a client call --url http://{host}:{port} --message 'Hello' \\")
    logger.info(f"    --bearer-token '{VALID_TOKEN}'")
    logger.info("")
    logger.info("=" * 60)

    # Run server
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
