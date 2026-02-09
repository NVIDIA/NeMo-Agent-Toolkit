# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""Microsoft Agent 365 front-end plugin implementation."""

import importlib
import logging

from nat.builder.front_end import FrontEndBase
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.a365.front_end.front_end_config import A365FrontEndConfig
from nat.plugins.a365.exceptions import A365SDKError
from nat.runtime.session import SessionManager
from nat.utils.log_levels import LOG_LEVELS
from nat.utils.log_utils import setup_logging

logger = logging.getLogger(__name__)


class A365FrontEndPlugin(FrontEndBase[A365FrontEndConfig]):
    """Microsoft Agent 365 front-end plugin.

    This plugin integrates NAT workflows with Microsoft Agent 365 hosting framework,
    allowing workflows to receive and respond to notifications from Teams, Email, and Office 365.
    """

    async def run(self) -> None:
        """Run the Microsoft Agent 365 server.
        
        This method orchestrates the workflow lifecycle:
        1. Imports and validates Microsoft Agents SDK dependencies
        2. Configures logging
        3. Builds NAT workflows and creates session managers
        4. Delegates SDK setup to worker
        5. Starts the Microsoft Agents SDK server
        6. Handles cleanup on shutdown
        """
        try:
            # Try importing start_server from hosting.aiohttp (most common)
            try:
                from microsoft_agents.hosting.aiohttp import start_server
            except ImportError:
                # Fallback to hosting.core if aiohttp version not available
                try:
                    from microsoft_agents.hosting.core.server import start_server
                except ImportError:
                    raise A365SDKError(
                        "Could not import start_server. "
                        "Install with: uv pip install microsoft-agents-hosting-aiohttp",
                        sdk_component="start_server"
                    )
        except ImportError as e:
            raise A365SDKError(
                "Microsoft Agents SDK packages are required. "
                "Install with: uv pip install microsoft-agents-activity microsoft-agents-hosting-core microsoft-agents-hosting-aiohttp",
                sdk_component="core",
                original_error=e
            ) from e

        log_level = LOG_LEVELS.get(self.front_end_config.log_level.upper(), logging.INFO)
        setup_logging(log_level)
        logger.setLevel(log_level)

        # Initialize worker and session managers (needed for cleanup in finally block)
        worker = None
        session_manager = None
        notification_session_manager = None

        try:
            async with WorkflowBuilder.from_config(config=self.full_config) as builder:
                session_manager = await SessionManager.create(
                    config=self.full_config,
                    shared_builder=builder
                )

                # Create separate session manager for notifications if configured
                notification_session_manager = session_manager
                if self.front_end_config.notification_workflow:
                    logger.info(
                        f"Creating separate session manager for notifications with entry_function='{self.front_end_config.notification_workflow}'"
                    )
                    notification_session_manager = await SessionManager.create(
                        config=self.full_config,
                        shared_builder=builder,
                        entry_function=self.front_end_config.notification_workflow
                    )

                # Get worker instance (allows for custom workers via config)
                worker = self._get_worker_instance()

                agent_app, connection_manager = await worker.create_agent_application()

                if self.front_end_config.enable_notifications:
                    await worker.setup_notification_handlers(
                        agent_app=agent_app,
                        session_manager=notification_session_manager
                    )

                await worker.setup_message_handlers(
                    agent_app=agent_app,
                    session_manager=session_manager
                )

                worker.setup_error_handlers(agent_app)

                logger.info(
                    f"Starting Microsoft Agent 365 server on "
                    f"{self.front_end_config.host}:{self.front_end_config.port}"
                )

                try:
                    await start_server(
                        agent_application=agent_app,
                        auth_configuration=connection_manager.get_default_connection_configuration(),
                        host=self.front_end_config.host,
                        port=self.front_end_config.port,
                    )
                except Exception as e:
                    error_msg = str(e).lower()
                    if "address already in use" in error_msg or "port" in error_msg:
                        raise A365SDKError(
                            f"Failed to start server: port {self.front_end_config.port} may already be in use. "
                            f"Try a different port or stop the process using this port.",
                            sdk_component="start_server",
                            original_error=e
                        ) from e
                    elif "permission" in error_msg or "access" in error_msg:
                        raise A365SDKError(
                            f"Failed to start server: insufficient permissions to bind to "
                            f"{self.front_end_config.host}:{self.front_end_config.port}",
                            sdk_component="start_server",
                            original_error=e
                        ) from e
                    else:
                        raise A365SDKError(
                            f"Failed to start server: {str(e)}",
                            sdk_component="start_server",
                            original_error=e
                        ) from e
        finally:
            if session_manager is not None:
                try:
                    await session_manager.shutdown()
                except Exception as e:
                    logger.error(f"Error cleaning up default session manager: {e}")
            
            # If notification session manager is different, clean it up too
            if notification_session_manager is not None and notification_session_manager is not session_manager:
                try:
                    await notification_session_manager.shutdown()
                except Exception as e:
                    logger.error(f"Error cleaning up notification session manager: {e}")
            
            if worker is not None:
                try:
                    await worker.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up worker: {e}")

    def _get_worker_instance(self):
        """Get an instance of the worker class.
        
        Returns:
            A365FrontEndPluginWorker instance configured with full config.
            If runner_class is specified in config, returns a custom worker instance.
        """
        from nat.plugins.a365.front_end.worker import A365FrontEndPluginWorker

        if self.front_end_config.runner_class:
            module_name, class_name = self.front_end_config.runner_class.rsplit(".", 1)
            module = importlib.import_module(module_name)
            worker_class = getattr(module, class_name)
            return worker_class(self.full_config)

        return A365FrontEndPluginWorker(self.full_config)

