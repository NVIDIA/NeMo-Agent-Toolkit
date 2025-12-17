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

import uvicorn

from nat.builder.front_end import FrontEndBase
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig
from nat.plugins.a2a.server.front_end_plugin_worker import A2AFrontEndPluginWorker

logger = logging.getLogger(__name__)


class A2AFrontEndPlugin(FrontEndBase[A2AFrontEndConfig]):
    """A2A front end plugin implementation.

    Exposes NAT workflows as A2A-compliant remote agents that can be
    discovered and invoked by other A2A agents and clients.
    """

    async def run(self) -> None:
        """Run the A2A server.

        This method:
        1. Builds the workflow
        2. Creates the agent card from configuration
        3. Creates the agent executor adapter
        4. Sets up the A2A server
        5. Starts the server with uvicorn
        """
        # Build the workflow
        async with WorkflowBuilder.from_config(config=self.full_config) as builder:
            workflow = await builder.build()

            # Create worker instance
            worker = self._get_worker_instance()

            # Build agent card from configuration and workflow functions
            agent_card = await worker.create_agent_card(workflow)

            # Create agent executor adapter
            agent_executor = worker.create_agent_executor(workflow, builder)

            # Create A2A server
            a2a_server = worker.create_a2a_server(agent_card, agent_executor)

            # Start the server with proper cleanup
            try:
                logger.info(
                    "Starting A2A server '%s' at http://%s:%s",
                    self.front_end_config.name,
                    self.front_end_config.host,
                    self.front_end_config.port,
                )
                logger.info("Agent card available at: http://%s:%s/.well-known/agent-card.json",
                            self.front_end_config.host,
                            self.front_end_config.port)

                # Build the ASGI app
                app = a2a_server.build()

                # Add OAuth2 validation middleware if configured
                if self.front_end_config.server_auth:
                    from nat.plugins.a2a.server.oauth_middleware import OAuth2ValidationMiddleware

                    app.add_middleware(OAuth2ValidationMiddleware, config=self.front_end_config.server_auth)
                    logger.info(
                        "OAuth2 token validation enabled for A2A server (issuer=%s, scopes=%s)",
                        self.front_end_config.server_auth.issuer_url,
                        self.front_end_config.server_auth.scopes,
                    )

                # Run with uvicorn
                config = uvicorn.Config(
                    app,
                    host=self.front_end_config.host,
                    port=self.front_end_config.port,
                    log_level=self.front_end_config.log_level.lower(),
                )
                server = uvicorn.Server(config)
                await server.serve()

            except KeyboardInterrupt:
                logger.info("A2A server shutdown requested (Ctrl+C). Shutting down gracefully.")
            except Exception as e:
                logger.error("A2A server error: %s", e, exc_info=True)
                raise
            finally:
                # Ensure cleanup of resources (httpx client)
                await worker.cleanup()
                logger.info("A2A server resources cleaned up")

    def _get_worker_instance(self) -> A2AFrontEndPluginWorker:
        """Get an instance of the worker class.

        Returns:
            Worker instance configured with full config
        """
        # Check if custom worker class is specified
        if self.front_end_config.runner_class:
            module_name, class_name = self.front_end_config.runner_class.rsplit(".", 1)
            import importlib
            module = importlib.import_module(module_name)
            worker_class = getattr(module, class_name)
            return worker_class(self.full_config)

        # Use default worker
        return A2AFrontEndPluginWorker(self.full_config)
