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

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from nat.data_models.authentication import AuthenticatedContext, AuthFlowType
from nat.data_models.common import get_secret_value
from nat.data_models.config import Config
from nat.plugins.a365.exceptions import (
    A365AuthenticationError,
    A365ConfigurationError,
    A365SDKError,
    A365WorkflowExecutionError,
)
from nat.plugins.a365.front_end.front_end_config import A365FrontEndConfig
from nat.plugins.a365.telemetry.agentic_token_cache import cache_agentic_observability_token
from nat.plugins.a365.telemetry.turn_context import (
    A365TurnTelemetryContext,
    reset_a365_turn_telemetry_context,
    set_a365_turn_telemetry_context,
)
from nat.runtime.session import SessionManager

if TYPE_CHECKING:
    from microsoft_agents.authentication.msal import MsalConnectionManager
    from microsoft_agents.hosting.aiohttp import CloudAdapter
    from microsoft_agents.hosting.core import AgentApplication, AgentAuthConfiguration, TurnState
    from microsoft_agents.hosting.core.authorization import Connections
    from microsoft_agents.hosting.core.storage import Storage

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

    def _get_storage(self) -> Storage:
        """Get the storage instance for the AgentApplication.
        
        Uses dependency injection pattern - returns Storage Protocol implementation.
        Defaults to MemoryStorage, but can be overridden for custom storage (e.g., BlobStorage, CosmosDbStorage).
        
        Returns:
            Storage: A Storage Protocol implementation (default: MemoryStorage)
        """
        from microsoft_agents.hosting.core import MemoryStorage
        return MemoryStorage()

    def _build_connection_configurations(
        self, service_connection: "AgentAuthConfiguration"
    ) -> dict[str, "AgentAuthConfiguration"]:
        """Build SDK connection configs, including optional JWT audience aliases.

        The Microsoft Agents SDK validates inbound JWT audiences against the
        ``CLIENT_ID`` values present in ``AgentAuthConfiguration._connections``.
        We keep ``SERVICE_CONNECTION`` as the default outbound auth config and add
        alias-only connections so Bot Framework / Teams tokens with alternate
        audiences are accepted without modifying SDK internals.
        """
        from microsoft_agents.hosting.core import AgentAuthConfiguration

        connections = {"SERVICE_CONNECTION": service_connection}

        for index, audience in enumerate(self.front_end_config.allowed_audiences, start=1):
            if audience.lower() == service_connection.CLIENT_ID.lower():
                continue

            connections[f"AUDIENCE_ALIAS_{index}"] = AgentAuthConfiguration(
                client_id=audience,
                client_secret=get_secret_value(self.front_end_config.app_password),
                auth_type=service_connection.AUTH_TYPE,
                connection_name=f"AUDIENCE_ALIAS_{index}",
                tenant_id=self.front_end_config.tenant_id,
            )

        return connections

    def _get_connection_manager(self, service_connection: "AgentAuthConfiguration") -> Connections:
        """Get the connection manager instance for the AgentApplication.

        Defaults to MsalConnectionManager with a single ``SERVICE_CONNECTION`` entry
        (required by the Microsoft Agents SDK 0.8+ MSAL integration).

        Args:
            service_connection: Auth configuration for the bot's service connection.

        Returns:
            Connections: A Connections implementation (default: MsalConnectionManager)
        """
        from microsoft_agents.authentication.msal import MsalConnectionManager

        return MsalConnectionManager(
            connections_configurations=self._build_connection_configurations(service_connection)
        )

    def _load_agents_sdk_configuration(self) -> dict:
        """Load Microsoft Agents SDK config from environment variables."""
        from microsoft_agents.activity.config import load_configuration_from_env

        return load_configuration_from_env(dict(os.environ))

    async def create_agent_application(
        self,
    ) -> tuple[AgentApplication[TurnState], Connections, "CloudAdapter"]:
        """Create and initialize Microsoft Agents SDK application.

        Returns:
            Initialized ``AgentApplication``, ``Connections`` (MSAL manager), and aiohttp
            ``CloudAdapter`` (used by the HTTP server and ``AgentApplication`` options).

        Raises:
            A365ConfigurationError: If configuration is invalid (missing fields, wrong types)
            A365SDKError: If SDK component initialization fails
        """
        from microsoft_agents.hosting.aiohttp import CloudAdapter
        from microsoft_agents.hosting.core import AgentApplication, AgentAuthConfiguration, TurnState
        from microsoft_agents.hosting.core.app.app_error import ApplicationError
        from microsoft_agents.hosting.core.app.app_options import ApplicationOptions
        from microsoft_agents.hosting.core.app.oauth import Authorization
        from microsoft_agents.hosting.core.authorization.auth_types import AuthTypes

        service_connection = AgentAuthConfiguration(
            client_id=self.front_end_config.app_id,
            client_secret=get_secret_value(self.front_end_config.app_password),
            auth_type=AuthTypes.client_secret,
            connection_name="SERVICE_CONNECTION",
            tenant_id=self.front_end_config.tenant_id,
        )

        # Initialize components sequentially, catching errors with context
        # This pattern matches A2A and MCP plugins: sequential initialization with
        # specific error handling for configuration vs. general SDK errors

        # Get storage instance (uses dependency injection pattern - defaults to MemoryStorage)
        # Users can override _get_storage() in a subclass to use custom storage (e.g., BlobStorage, CosmosDbStorage)
        storage = self._get_storage()

        # Get connection manager instance (uses dependency injection pattern - defaults to MsalConnectionManager)
        # Users can override _get_connection_manager() in a subclass to use custom connection managers
        try:
            connection_manager = self._get_connection_manager(service_connection)
        except (ValueError, TypeError) as e:
            # ValueError/TypeError from connection manager initialization indicate configuration issues
            # (missing required fields, wrong parameter types, invalid values)
            raise A365ConfigurationError(
                f"Invalid configuration for connection manager: {str(e)}. "
                f"Please check that app_id, app_password, and tenant_id are properly configured.",
                original_error=e
            ) from e
        except ApplicationError as e:
            # ApplicationError from SDK indicates missing or misconfigured SDK components
            raise A365SDKError(
                f"Failed to initialize connection manager: {str(e)}",
                sdk_component="ConnectionManager",
                original_error=e
            ) from e
        except Exception as e:
            raise A365SDKError(
                f"Failed to initialize connection manager: {str(e)}",
                sdk_component="ConnectionManager",
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
            agents_sdk_config = self._load_agents_sdk_configuration()
            authorization = Authorization(
                storage=storage,
                connection_manager=connection_manager,
                **agents_sdk_config,
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
            options = ApplicationOptions(
                storage=storage,
                adapter=adapter,
                bot_app_id=self.front_end_config.app_id,
            )
            agent_app = AgentApplication[TurnState](
                options=options,
                connection_manager=connection_manager,
                authorization=authorization,
            )
            self._patch_authorization_invoke_handling(agent_app)
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

        return agent_app, connection_manager, adapter

    def _patch_authorization_invoke_handling(self, agent_app) -> None:
        """Ignore unsupported non-OAuth invoke callbacks during sign-in flows.

        The current Microsoft Agents SDK OAuth flow raises on invoke activities
        whose names are neither ``signin/verifyState`` nor
        ``signin/tokenExchange``. Teams can emit other invoke turns while a
        sign-in flow is active; those should be logged and ignored instead of
        crashing the whole turn.
        """

        original = agent_app._auth._on_turn_auth_intercept

        async def _wrapped_on_turn_auth_intercept(context, state):
            try:
                return await original(context, state)
            except ValueError as exc:
                activity = getattr(context, "activity", None)
                activity_type = getattr(activity, "type", None)
                activity_name = getattr(activity, "name", None)
                if activity_type == "invoke" and "Unknown activity type invoke" in str(exc):
                    logger.warning(
                        "Ignoring unsupported invoke during auth flow: name=%s value=%s",
                        activity_name,
                        getattr(activity, "value", None),
                    )
                    return True, None
                raise

        agent_app._auth._on_turn_auth_intercept = _wrapped_on_turn_auth_intercept

    def _extract_user_id(self, context, state) -> str | None:
        """Extract a stable user identifier from the current turn when available.

        Defaults to AAD object ID first, then channel/user ID. Subclasses can
        override if they need a different mapping.
        """
        activity = getattr(context, "activity", None)
        sender = getattr(activity, "from_property", None) or getattr(activity, "from_", None)
        if sender is None:
            return None

        aad_object_id = getattr(sender, "aad_object_id", None)
        if isinstance(aad_object_id, str) and aad_object_id:
            return aad_object_id

        sender_id = getattr(sender, "id", None)
        if isinstance(sender_id, str) and sender_id:
            return sender_id

        return None

    def _get_user_authentication_callback(self, agent_app, context, state):
        """Return an optional per-turn auth callback for delegated auth flows.

        Delegated/OBO auth is enabled only when the front-end config names a
        Microsoft Agents auth handler. Service-token / S2S integrations should
        leave that unset so no per-turn callback is injected.
        """
        auth_handler_id = self.front_end_config.observability_auth_handler_id
        if not auth_handler_id:
            return None

        from microsoft_agents_a365.runtime import get_observability_authentication_scope

        scope = get_observability_authentication_scope()
        scopes = scope if isinstance(scope, list) else [scope]
        exchange_connection = os.getenv(
            f"AGENTAPPLICATION__USERAUTHORIZATION__HANDLERS__{auth_handler_id}__SETTINGS__OBOCONNECTIONNAME"
        ) or os.getenv(
            f"AGENTAPPLICATION__USERAUTHORIZATION__HANDLERS__{auth_handler_id}__SETTINGS__AZUREBOTOAUTHCONNECTIONNAME"
        )

        async def _get_cached_bot_oauth_token():
            activity = getattr(context, "activity", None)
            connector_user_id = getattr(getattr(activity, "from_property", None), "id", None) or getattr(
                getattr(activity, "from_", None), "id", None
            )
            channel_id = getattr(activity, "channel_id", None)
            if not connector_user_id or not channel_id or not exchange_connection:
                return None

            clients = []
            for owner in (agent_app, getattr(agent_app, "_adapter", None), getattr(agent_app, "adapter", None), getattr(context, "adapter", None)):
                if owner is None:
                    continue
                for attr in ("user_token_client", "_user_token_client"):
                    client = getattr(owner, attr, None)
                    if client is not None and client not in clients:
                        clients.append(client)

            for client in clients:
                get_token = getattr(client, "get_token", None)
                if get_token is None:
                    continue
                try:
                    token_response = await get_token(connector_user_id, exchange_connection, channel_id, None)
                except TypeError:
                    token_response = await get_token(connector_user_id, exchange_connection, channel_id)
                except Exception as exc:  # pragma: no cover - defensive against SDK drift
                    logger.warning(
                        "Cached Bot OAuth token lookup failed via %s: %s",
                        type(client).__name__,
                        exc,
                    )
                    continue
                if token_response is not None and getattr(token_response, "token", None):
                    logger.info(
                        "Recovered delegated observability token from cached Bot OAuth store "
                        "(handler=%s, connection=%s, connector_user_id=%s)",
                        auth_handler_id,
                        exchange_connection,
                        connector_user_id,
                    )
                    return token_response
            return None

        async def _authenticate(_config, method: AuthFlowType) -> AuthenticatedContext:
            if method != AuthFlowType.OAUTH2_AUTHORIZATION_CODE:
                raise RuntimeError(f"Unsupported auth flow for A365 front-end: {method}")

            token_response = await agent_app.auth.exchange_token(
                context,
                scopes=scopes,
                auth_handler_id=auth_handler_id,
                exchange_connection=exchange_connection,
            )
            if token_response is None or not getattr(token_response, "token", None):
                token_response = await agent_app.auth.get_token(
                    context,
                    auth_handler_id=auth_handler_id,
                )
            if token_response is None or not getattr(token_response, "token", None):
                token_response = await _get_cached_bot_oauth_token()
            if token_response is None or not getattr(token_response, "token", None):
                logger.error(
                    "Delegated observability token missing on turn "
                    "(handler=%s, exchange_connection=%s, activity_type=%s, channel_id=%s, user_id=%s)",
                    auth_handler_id,
                    exchange_connection,
                    getattr(getattr(context, "activity", None), "type", None),
                    getattr(getattr(context, "activity", None), "channel_id", None),
                    self._extract_user_id(context, state),
                )
                raise RuntimeError("Failed to acquire delegated observability token from TurnContext")

            metadata = {}
            expiration = getattr(token_response, "expiration", None)
            if expiration:
                metadata["expires_at"] = expiration

            return AuthenticatedContext(
                headers={"Authorization": f"Bearer {token_response.token}"},
                metadata=metadata,
            )

        return _authenticate

    @staticmethod
    def _call_activity_helper(activity, helper_name: str) -> str | None:
        helper = getattr(activity, helper_name, None)
        if not callable(helper):
            return None
        try:
            value = helper()
        except Exception:  # pragma: no cover - defensive against SDK drift
            logger.debug("Failed to call Activity.%s()", helper_name, exc_info=True)
            return None
        return value if isinstance(value, str) and value else None

    async def _build_turn_telemetry_context(self, agent_app, context, state, user_authentication_callback):
        activity = getattr(context, "activity", None)
        if activity is None:
            return None

        agent_id = self._call_activity_helper(activity, "get_agentic_instance_id")
        tenant_id = self._call_activity_helper(activity, "get_agentic_tenant_id")
        agentic_user_id = self._call_activity_helper(activity, "get_agentic_user")

        token = None
        expires_at = None
        if user_authentication_callback is not None:
            try:
                auth_context = await user_authentication_callback(None, AuthFlowType.OAUTH2_AUTHORIZATION_CODE)
                headers = auth_context.headers if isinstance(auth_context.headers, dict) else {}
                metadata = auth_context.metadata if isinstance(auth_context.metadata, dict) else {}
                auth_header = headers.get("Authorization", "")
                if isinstance(auth_header, str) and auth_header.startswith("Bearer "):
                    token = auth_header[7:]
                expires_at = metadata.get("expires_at")
            except Exception as exc:
                logger.warning(
                    "Failed to acquire per-turn Agent 365 observability token "
                    "(agent_id=%s, tenant_id=%s, agentic_user_id=%s): %s",
                    agent_id,
                    tenant_id,
                    agentic_user_id,
                    exc,
                )

        if not any((agent_id, tenant_id, agentic_user_id, token)):
            return None

        token_bridge = self.front_end_config.observability_token_bridge
        if token_bridge in {"cache", "both"} and token:
            cached = cache_agentic_observability_token(
                tenant_id,
                agent_id,
                token,
                expires_at=expires_at,
                agentic_user_id=agentic_user_id,
            )
            if not cached:
                logger.warning(
                    "Could not cache agentic observability token "
                    "(agent_id=%s, tenant_id=%s, token=present)",
                    agent_id,
                    tenant_id,
                )

        context_token = token if token_bridge in {"turn_context", "both"} else None

        logger.info(
            "A365 turn telemetry context prepared "
            "(agent_id=%s, tenant_id=%s, agentic_user_id=%s, token=%s, bridge=%s)",
            agent_id,
            tenant_id,
            agentic_user_id,
            "present" if context_token else "cached" if token else "missing",
            token_bridge,
        )

        return A365TurnTelemetryContext(
            agent_id=agent_id,
            tenant_id=tenant_id,
            agentic_user_id=agentic_user_id,
            token=context_token,
            expires_at=expires_at,
        )

    async def _execute_workflow_for_turn(self, agent_app, session_manager: SessionManager, payload, context, state) -> str:
        """Execute the workflow for a turn while wiring per-user auth context."""
        user_id = self._extract_user_id(context, state)
        user_authentication_callback = self._get_user_authentication_callback(agent_app, context, state)
        turn_telemetry_context = await self._build_turn_telemetry_context(
            agent_app,
            context,
            state,
            user_authentication_callback,
        )
        turn_telemetry_token = (
            set_a365_turn_telemetry_context(turn_telemetry_context)
            if turn_telemetry_context is not None
            else None
        )

        try:
            if user_id is not None or user_authentication_callback is not None:
                async with session_manager.session(
                    user_id=user_id,
                    user_authentication_callback=user_authentication_callback,
                ) as session:
                    async with session.run(payload) as runner:
                        return await runner.result(to_type=str)

            async with session_manager.run(payload) as runner:
                return await runner.result(to_type=str)
        finally:
            if turn_telemetry_token is not None:
                reset_a365_turn_telemetry_context(turn_telemetry_token)

    async def _handle_signin_command(self, agent_app, context, state) -> bool:
        """Bootstrap the delegated auth flow for the current user when requested.

        Returns True when the current message was handled as a sign-in command and
        no further workflow execution should occur.
        """
        query = (context.activity.text or "").strip().lower()
        if query not in {"signin", "sign-in", "sign in", "login", "log in"}:
            return False

        auth_handler_id = self.front_end_config.observability_auth_handler_id
        if not auth_handler_id:
            await context.send_activity(
                "Delegated sign-in is not enabled for this bot configuration."
            )
            return True

        sign_in_response = await agent_app.auth._start_or_continue_sign_in(
            context,
            state,
            auth_handler_id=auth_handler_id,
        )
        tag = getattr(sign_in_response, "tag", None)
        if str(tag).upper().endswith("COMPLETE"):
            await context.send_activity(
                "Sign-in completed. Send your question again so I can retry observability with your delegated token."
            )
        elif str(tag).upper().endswith("FAILURE"):
            await context.send_activity(
                "Sign-in could not be completed. Please try `signin` again."
            )
        else:
            await context.send_activity(
                "A sign-in prompt was sent. Complete it in Teams, then send your question again."
            )
        return True

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

        async def execute_workflow_from_notification(
            context: TurnContext,
            state: TurnState,
            activity: AgentNotificationActivity,
            notification_type: str
        ) -> None:
            """Execute NAT workflow with notification data."""
            try:
                # Extract text/content from notification using typed properties when available
                # Email notifications have typed email data
                if activity.email and activity.email.html_body:
                    query = activity.email.html_body
                # Word/Excel/PowerPoint comments - use activity text (WpxComment doesn't contain text directly)
                elif activity.wpx_comment:
                    query = context.activity.text or context.activity.summary or f"Document comment notification"
                # Lifecycle events and other notifications - use generic activity text
                else:
                    query = context.activity.text or context.activity.summary or f"Notification: {notification_type}"

                from nat.data_models.api_server import ChatRequest
                payload = ChatRequest.from_string(query)

                result = await self._execute_workflow_for_turn(
                    agent_app=agent_app,
                    session_manager=session_manager,
                    payload=payload,
                    context=context,
                    state=state,
                )

                await context.send_activity(result)

            except A365WorkflowExecutionError as e:
                logger.error(
                    f"Error executing workflow from {notification_type} notification: {e.workflow_type}",
                    exc_info=True
                )
                await context.send_activity(
                    f"I encountered an error processing the {notification_type} notification. Please try again."
                )
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(
                    f"Error executing workflow from {notification_type} notification: {type(e).__name__}",
                    exc_info=True
                )
                
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
            await execute_workflow_from_notification(context, state, activity, "email")

        # Word document notification handler
        @notification.on_word()
        async def on_word(
            context: TurnContext,
            state: TurnState,
            activity: AgentNotificationActivity
        ):
            text = context.activity.text or context.activity.summary or ""
            logger.info(f"Received Word notification: {text[:100] if text else 'No text'}")
            await execute_workflow_from_notification(context, state, activity, "Word")

        # Excel notification handler
        @notification.on_excel()
        async def on_excel(
            context: TurnContext,
            state: TurnState,
            activity: AgentNotificationActivity
        ):
            text = context.activity.text or context.activity.summary or ""
            logger.info(f"Received Excel notification: {text[:100] if text else 'No text'}")
            await execute_workflow_from_notification(context, state, activity, "Excel")

        # PowerPoint notification handler
        @notification.on_powerpoint()
        async def on_powerpoint(
            context: TurnContext,
            state: TurnState,
            activity: AgentNotificationActivity
        ):
            text = context.activity.text or context.activity.summary or ""
            logger.info(f"Received PowerPoint notification: {text[:100] if text else 'No text'}")
            await execute_workflow_from_notification(context, state, activity, "PowerPoint")

        # Lifecycle handlers
        @notification.on_user_created()
        async def on_user_created(
            context: TurnContext,
            state: TurnState,
            activity: AgentNotificationActivity
        ):
            logger.info("User created lifecycle event received")
            await execute_workflow_from_notification(context, state, activity, "user_created")

        @notification.on_user_deleted()
        async def on_user_deleted(
            context: TurnContext,
            state: TurnState,
            activity: AgentNotificationActivity
        ):
            logger.info("User deleted lifecycle event received")
            await execute_workflow_from_notification(context, state, activity, "user_deleted")

        logger.info("A365 notification handlers registered")

    async def setup_message_handlers(
        self,
        agent_app: AgentApplication,
        session_manager: SessionManager
    ) -> None:
        """Set up message handlers for regular chat messages.
        
        Args:
            agent_app: The Microsoft Agents SDK AgentApplication instance
            session_manager: SessionManager for executing NAT workflows
        """
        from microsoft_agents.hosting.core import TurnContext, TurnState

        auth_handlers = None
        if self.front_end_config.observability_auth_handler_id:
            auth_handlers = [self.front_end_config.observability_auth_handler_id]

        activity_decorator = (
            agent_app.activity("message", auth_handlers=auth_handlers)
            if auth_handlers is not None
            else agent_app.activity("message")
        )

        @activity_decorator
        async def on_message(context: TurnContext, state: TurnState):
            """Handle regular chat messages."""
            try:
                query = context.activity.text or ""

                if not query:
                    await context.send_activity("I didn't receive any message. Please try again.")
                    return

                logger.info(f"Received message: {query[:100]}")

                if await self._handle_signin_command(agent_app, context, state):
                    return

                from nat.data_models.api_server import ChatRequest
                payload = ChatRequest.from_string(query)

                result = await self._execute_workflow_for_turn(
                    agent_app=agent_app,
                    session_manager=session_manager,
                    payload=payload,
                    context=context,
                    state=state,
                )

                await context.send_activity(result)

            except A365WorkflowExecutionError as e:
                logger.error(
                    f"Error executing workflow from message: {e.workflow_type}",
                    exc_info=True
                )
                await context.send_activity(
                    "I encountered an error processing your message. Please try again."
                )
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(
                    f"Error handling message: {type(e).__name__}",
                    exc_info=True
                )
                
                if "timeout" in error_msg:
                    user_message = "Your message timed out. Please try again."
                elif "validation" in error_msg or "invalid" in error_msg:
                    user_message = "Invalid message format. Please check your input and try again."
                else:
                    user_message = "I encountered an error processing your message. Please try again."
                
                await context.send_activity(user_message)

        logger.info("Message handlers registered")

    def setup_error_handlers(self, agent_app: AgentApplication) -> None:
        """Set up error handlers for the AgentApplication.
        
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

        logger.info("Error handlers registered")

    async def cleanup(self) -> None:
        """Clean up any resources managed by the worker.
        
        Currently, the worker doesn't manage any resources that need explicit cleanup,
        but this method is provided for consistency with other workers and future extensibility.
        """
        # No resources to clean up currently, but this provides an extension point
        # for subclasses that might manage resources (e.g., HTTP clients, connections)
        pass
