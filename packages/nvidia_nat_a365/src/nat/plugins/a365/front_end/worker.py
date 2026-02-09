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

"""Worker for Microsoft Agent 365 front-end plugin.

This worker encapsulates the Microsoft Agents SDK integration logic,
allowing for extensibility and better separation of concerns.
"""

import logging
from typing import TYPE_CHECKING

from nat.data_models.common import get_secret_value
from nat.data_models.config import Config
from nat.plugins.a365.exceptions import (
    A365AuthenticationError,
    A365ConfigurationError,
    A365SDKError,
    A365WorkflowExecutionError,
)
from nat.plugins.a365.front_end.front_end_config import A365FrontEndConfig
from nat.runtime.session import SessionManager

if TYPE_CHECKING:
    from microsoft_agents.hosting.core import AgentApplication, TurnState
    from microsoft_agents.hosting.core.authentication import MsalConnectionManager

logger = logging.getLogger(__name__)


class A365FrontEndPluginWorker:
    """Worker that handles Microsoft Agents SDK setup and configuration.
    
    This class encapsulates the implementation details of integrating NAT workflows
    with the Microsoft Agents SDK, allowing for extensibility through subclassing
    and better separation of concerns from the plugin orchestration logic.
    """

    def __init__(self, config: Config):
        """Initialize the A365 worker with configuration.

        Args:
            config: The full NAT configuration
        """
        self.full_config = config
        self.front_end_config: A365FrontEndConfig = config.general.front_end  # type: ignore

    async def create_agent_application(
        self
    ) -> tuple[AgentApplication[TurnState], MsalConnectionManager]:
        """Create and initialize Microsoft Agents SDK application.
        
        Returns:
            tuple[AgentApplication[TurnState], MsalConnectionManager]: Initialized application
                and connection manager (needed for server startup)
            
        Raises:
            A365ConfigurationError: If configuration is invalid (missing fields, wrong types)
            A365SDKError: If SDK component initialization fails
        """
        from microsoft_agents.hosting.core import (
            AgentApplication,
            CloudAdapter,
            MemoryStorage,
            TurnState,
        )
        from microsoft_agents.hosting.core.app.app_error import ApplicationError
        from microsoft_agents.hosting.core.authentication import (
            Authorization,
            MsalConnectionManager,
        )

        # Set up Microsoft Agents SDK configuration
        agents_sdk_config = {
            "MicrosoftAppId": self.front_end_config.app_id,
            "MicrosoftAppPassword": get_secret_value(self.front_end_config.app_password),
        }
        if self.front_end_config.tenant_id:
            agents_sdk_config["MicrosoftAppTenantId"] = self.front_end_config.tenant_id

        # Initialize components sequentially, catching errors with context
        # This pattern matches A2A and MCP plugins: sequential initialization with
        # specific error handling for configuration vs. general SDK errors

        try:
            storage = MemoryStorage()
        except Exception as e:
            raise A365SDKError(
                f"Failed to initialize MemoryStorage: {str(e)}",
                sdk_component="MemoryStorage",
                original_error=e
            ) from e

        try:
            connection_manager = MsalConnectionManager(**agents_sdk_config)
        except (ValueError, TypeError) as e:
            # ValueError/TypeError from MsalConnectionManager initialization indicate configuration issues
            # (missing required fields, wrong parameter types, invalid values)
            raise A365ConfigurationError(
                f"Invalid configuration for MsalConnectionManager: {str(e)}. "
                f"Please check that app_id, app_password, and tenant_id are properly configured.",
                original_error=e
            ) from e
        except ApplicationError as e:
            # ApplicationError from SDK indicates missing or misconfigured SDK components
            raise A365SDKError(
                f"Failed to initialize MsalConnectionManager: {str(e)}",
                sdk_component="MsalConnectionManager",
                original_error=e
            ) from e
        except Exception as e:
            raise A365SDKError(
                f"Failed to initialize MsalConnectionManager: {str(e)}",
                sdk_component="MsalConnectionManager",
                original_error=e
            ) from e

        try:
            adapter = CloudAdapter(connection_manager=connection_manager)
        except Exception as e:
            raise A365SDKError(
                f"Failed to initialize CloudAdapter: {str(e)}",
                sdk_component="CloudAdapter",
                original_error=e
            ) from e

        try:
            authorization = Authorization(
                storage=storage,
                connection_manager=connection_manager,
                **agents_sdk_config
            )
        except (ValueError, TypeError) as e:
            # ValueError/TypeError from Authorization initialization indicate configuration issues
            # (missing storage, unrecognized auth types, missing handlers)
            raise A365ConfigurationError(
                f"Invalid configuration for Authorization: {str(e)}. "
                f"Please check that app_id, app_password, and tenant_id are properly configured.",
                original_error=e
            ) from e
        except ApplicationError as e:
            # ApplicationError from SDK indicates missing or misconfigured SDK components
            raise A365SDKError(
                f"Failed to initialize Authorization: {str(e)}",
                sdk_component="Authorization",
                original_error=e
            ) from e
        except Exception as e:
            raise A365SDKError(
                f"Failed to initialize Authorization: {str(e)}",
                sdk_component="Authorization",
                original_error=e
            ) from e

        try:
            # Create AgentApplication
            agent_app = AgentApplication[TurnState](
                storage=storage,
                adapter=adapter,
                authorization=authorization,
                **agents_sdk_config
            )
        except ApplicationError as e:
            # ApplicationError from SDK indicates missing required components (storage, adapter, auth)
            raise A365SDKError(
                f"Failed to create AgentApplication: {str(e)}",
                sdk_component="AgentApplication",
                original_error=e
            ) from e
        except (ValueError, TypeError) as e:
            # ValueError/TypeError from AgentApplication initialization indicate configuration issues
            raise A365ConfigurationError(
                f"Invalid configuration for AgentApplication: {str(e)}",
                original_error=e
            ) from e
        except RuntimeError as e:
            # RuntimeError from SDK indicates runtime issues (not typically raised during initialization)
            raise A365SDKError(
                f"Failed to create AgentApplication: {str(e)}",
                sdk_component="AgentApplication",
                original_error=e
            ) from e
        except Exception as e:
            raise A365SDKError(
                f"Failed to create AgentApplication: {str(e)}",
                sdk_component="AgentApplication",
                original_error=e
            ) from e

        return agent_app, connection_manager

    async def setup_notification_handlers(
        self,
        agent_app: AgentApplication,
        session_manager: SessionManager
    ) -> None:
        """Set up A365 notification handlers.
        
        Args:
            agent_app: The Microsoft Agents SDK AgentApplication instance
            session_manager: SessionManager for executing NAT workflows
        """
        try:
            from microsoft_agents_a365.notifications import AgentNotification
            from microsoft_agents_a365.notifications.models import AgentNotificationActivity
        except ImportError as e:
            logger.warning(
                "A365 notifications package not available. Notification handlers will be disabled. "
                f"Install with: uv pip install microsoft-agents-a365-notifications. Error: {e}"
            )
            return

        from microsoft_agents.hosting.core import TurnContext, TurnState

        notification = AgentNotification(agent_app)

        # Helper to execute NAT workflow from notification
        async def execute_workflow_from_notification(
            context: TurnContext,
            activity: AgentNotificationActivity,
            notification_type: str
        ) -> None:
            """Execute NAT workflow with notification data."""
            try:
                # Extract text/content from notification (context.activity is the Activity wrapped by AgentNotificationActivity)
                query = context.activity.text or context.activity.summary or f"Notification: {notification_type}"

                # Create input payload for workflow
                from nat.data_models.api_server import ChatRequest
                payload = ChatRequest.from_string(query)

                # Execute workflow
                async with session_manager.run(payload) as runner:
                    result = await runner.result(to_type=str)

                # Send response back
                await context.send_activity(result)

            except A365WorkflowExecutionError as e:
                # Re-raise our custom exceptions to preserve context
                logger.error(
                    f"Error executing workflow from {notification_type} notification: {e.workflow_type}",
                    exc_info=True
                )
                await context.send_activity(
                    f"I encountered an error processing the {notification_type} notification. Please try again."
                )
            except Exception as e:
                # Wrap other exceptions for better error handling
                error_msg = str(e).lower()
                logger.error(
                    f"Error executing workflow from {notification_type} notification: {type(e).__name__}",
                    exc_info=True
                )
                
                # Provide context-appropriate error message
                if "timeout" in error_msg:
                    user_message = f"The {notification_type} notification timed out. Please try again."
                elif "validation" in error_msg or "invalid" in error_msg:
                    user_message = f"Invalid input in {notification_type} notification. Please check the content and try again."
                else:
                    user_message = f"I encountered an error processing the {notification_type} notification. Please try again."
                
                await context.send_activity(user_message)

        # Email notification handler
        @notification.on_email()
        async def on_email(
            context: TurnContext,
            state: TurnState,
            activity: AgentNotificationActivity
        ):
            text = context.activity.text or context.activity.summary or ""
            logger.info(f"Received email notification: {text[:100] if text else 'No text'}")
            await execute_workflow_from_notification(context, activity, "email")

        # Word document notification handler
        @notification.on_word()
        async def on_word(
            context: TurnContext,
            state: TurnState,
            activity: AgentNotificationActivity
        ):
            text = context.activity.text or context.activity.summary or ""
            logger.info(f"Received Word notification: {text[:100] if text else 'No text'}")
            await execute_workflow_from_notification(context, activity, "Word")

        # Excel notification handler
        @notification.on_excel()
        async def on_excel(
            context: TurnContext,
            state: TurnState,
            activity: AgentNotificationActivity
        ):
            text = context.activity.text or context.activity.summary or ""
            logger.info(f"Received Excel notification: {text[:100] if text else 'No text'}")
            await execute_workflow_from_notification(context, activity, "Excel")

        # PowerPoint notification handler
        @notification.on_powerpoint()
        async def on_powerpoint(
            context: TurnContext,
            state: TurnState,
            activity: AgentNotificationActivity
        ):
            text = context.activity.text or context.activity.summary or ""
            logger.info(f"Received PowerPoint notification: {text[:100] if text else 'No text'}")
            await execute_workflow_from_notification(context, activity, "PowerPoint")

        # Lifecycle handlers
        @notification.on_user_created()
        async def on_user_created(
            context: TurnContext,
            state: TurnState,
            activity: AgentNotificationActivity
        ):
            logger.info("User created lifecycle event received")
            await execute_workflow_from_notification(context, activity, "user_created")

        @notification.on_user_deleted()
        async def on_user_deleted(
            context: TurnContext,
            state: TurnState,
            activity: AgentNotificationActivity
        ):
            logger.info("User deleted lifecycle event received")
            await execute_workflow_from_notification(context, activity, "user_deleted")

        logger.info("A365 notification handlers registered")

    async def setup_message_handler(
        self,
        agent_app: AgentApplication,
        session_manager: SessionManager
    ) -> None:
        """Set up message handler for regular chat messages.
        
        Args:
            agent_app: The Microsoft Agents SDK AgentApplication instance
            session_manager: SessionManager for executing NAT workflows
        """
        from microsoft_agents.hosting.core import TurnContext, TurnState

        @agent_app.activity("message")
        async def on_message(context: TurnContext, state: TurnState):
            """Handle regular chat messages."""
            try:
                query = context.activity.text or ""

                if not query:
                    await context.send_activity("I didn't receive any message. Please try again.")
                    return

                logger.info(f"Received message: {query[:100]}")

                # Create input payload for workflow
                from nat.data_models.api_server import ChatRequest
                payload = ChatRequest.from_string(query)

                # Execute workflow
                async with session_manager.run(payload) as runner:
                    result = await runner.result(to_type=str)

                # Send response back
                await context.send_activity(result)

            except A365WorkflowExecutionError as e:
                # Re-raise our custom exceptions to preserve context
                logger.error(
                    f"Error executing workflow from message: {e.workflow_type}",
                    exc_info=True
                )
                await context.send_activity(
                    "I encountered an error processing your message. Please try again."
                )
            except Exception as e:
                # Wrap other exceptions for better error handling
                error_msg = str(e).lower()
                logger.error(
                    f"Error handling message: {type(e).__name__}",
                    exc_info=True
                )
                
                # Provide context-appropriate error message
                if "timeout" in error_msg:
                    user_message = "Your message timed out. Please try again."
                elif "validation" in error_msg or "invalid" in error_msg:
                    user_message = "Invalid message format. Please check your input and try again."
                else:
                    user_message = "I encountered an error processing your message. Please try again."
                
                await context.send_activity(user_message)

        logger.info("Message handler registered")

    def setup_error_handler(self, agent_app: AgentApplication) -> None:
        """Set up error handler for the AgentApplication.
        
        Args:
            agent_app: The Microsoft Agents SDK AgentApplication instance
        """
        from microsoft_agents.hosting.core import TurnContext

        @agent_app.error
        async def on_error(context: TurnContext, error: Exception):
            """Handle unhandled errors in the AgentApplication."""
            # Log full error details server-side for debugging
            logger.error(
                f"Unhandled error in Agent 365 front-end: {type(error).__name__}: {error}",
                exc_info=True
            )
            
            # Provide user-friendly error message without exposing internals
            # Check for our custom exception types first for better error handling
            if isinstance(error, A365AuthenticationError):
                user_message = "Authentication failed. Please verify your credentials and try again."
            elif isinstance(error, A365SDKError):
                # SDK errors might be configuration issues
                if "port" in str(error).lower() or "address" in str(error).lower():
                    user_message = "Server configuration error. Please contact your administrator."
                else:
                    user_message = "A system error occurred. Please try again later."
            elif isinstance(error, A365WorkflowExecutionError):
                user_message = "I encountered an error processing your request. Please try again."
            else:
                # Check error message for common patterns
                error_msg = str(error).lower()
                if "authentication" in error_msg or "unauthorized" in error_msg:
                    user_message = "Authentication failed. Please verify your credentials and try again."
                elif "timeout" in error_msg:
                    user_message = "The request timed out. Please try again."
                elif "connection" in error_msg or "network" in error_msg:
                    user_message = "Connection error occurred. Please check your network connection and try again."
                else:
                    user_message = "I encountered an error processing your request. Please try again."
            
            await context.send_activity(user_message)

        logger.info("Error handler registered")

    async def cleanup(self) -> None:
        """Clean up any resources managed by the worker.
        
        Currently, the worker doesn't manage any resources that need explicit cleanup,
        but this method is provided for consistency with other workers and future extensibility.
        """
        # No resources to clean up currently, but this provides an extension point
        # for subclasses that might manage resources (e.g., HTTP clients, connections)
        pass
