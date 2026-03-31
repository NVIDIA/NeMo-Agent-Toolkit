# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Generic Agent Host Server
A generic hosting server that can host any agent class that implements the required interface.
"""

import asyncio
import logging
import os
import socket
from os import environ

from aiohttp.web import Application, Request, Response, json_response, run_app
from aiohttp.web_middlewares import middleware as web_middleware
from dotenv import load_dotenv
from microsoft_agents.hosting.aiohttp import (
    CloudAdapter,
    jwt_authorization_middleware,
    start_agent_process,
)

# Microsoft Agents SDK imports
from microsoft_agents.hosting.core import (
    Authorization,
    AgentApplication,
    AgentAuthConfiguration,
    AuthenticationConstants,
    ClaimsIdentity,
    MemoryStorage,
    TurnContext,
    TurnState,
)

from microsoft_agents.authentication.msal import MsalConnectionManager
from microsoft_agents.activity import load_configuration_from_env, Activity

# Import our agent base class
from agent_interface import AgentInterface, check_agent_inheritance

# Configure logging
ms_agents_logger = logging.getLogger("microsoft_agents")
ms_agents_logger.addHandler(logging.StreamHandler())
ms_agents_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Notifications imports (optional)
try:
    from microsoft_agents_a365.notifications.agent_notification import (
        AgentNotification,
        AgentNotificationActivity,
        ChannelId,
    )
    from microsoft_agents_a365.notifications import EmailResponse, NotificationTypes
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False

# Observability imports (optional)
try:
    from microsoft_agents_a365.observability.core.config import configure as configure_observability
    from microsoft_agents_a365.observability.core.middleware.baggage_builder import BaggageBuilder
    from token_cache import get_cached_agentic_token, cache_agentic_token
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

# Load configuration
load_dotenv()
agents_sdk_config = load_configuration_from_env(environ)


class GenericAgentHost:
    """Generic host that can host any agent implementing the AgentInterface"""

    def __init__(self, agent_class: type[AgentInterface], *agent_args, **agent_kwargs):
        if not check_agent_inheritance(agent_class):
            raise TypeError(f"Agent class {agent_class.__name__} must inherit from AgentInterface")

        self.auth_handler_name = os.getenv("AUTH_HANDLER_NAME", "") or None
        if self.auth_handler_name:
            logger.info(f"Using auth handler: {self.auth_handler_name}")
        else:
            logger.info("No auth handler configured (AUTH_HANDLER_NAME not set)")

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.agent_kwargs = agent_kwargs
        self.agent_instance = None

        # Microsoft Agents SDK components
        self.storage = MemoryStorage()
        self.connection_manager = MsalConnectionManager(**agents_sdk_config)
        self.adapter = CloudAdapter(connection_manager=self.connection_manager)
        self.authorization = Authorization(
            self.storage, self.connection_manager, **agents_sdk_config
        )
        self.agent_app = AgentApplication[TurnState](
            storage=self.storage,
            adapter=self.adapter,
            authorization=self.authorization,
            **agents_sdk_config,
        )

        # Initialize notification support if available
        if NOTIFICATIONS_AVAILABLE:
            self.agent_notification = AgentNotification(self.agent_app)
            logger.info("Notification handlers will be registered")

        # Setup message handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup the Microsoft Agents SDK message handlers"""

        handler_config = (
            {"auth_handlers": [self.auth_handler_name]} if self.auth_handler_name else {}
        )

        async def help_handler(context: TurnContext, _: TurnState):
            welcome_message = (
                f"Welcome! I'm powered by: **{self.agent_class.__name__}**\n\n"
                "Ask me anything and I'll do my best to help!\n"
                "Type '/help' for this message."
            )
            await context.send_activity(welcome_message)
            logger.info("Sent help/welcome message")

        self.agent_app.conversation_update("membersAdded", **handler_config)(help_handler)
        self.agent_app.message("/help", **handler_config)(help_handler)

        @self.agent_app.activity("installationUpdate")
        async def on_installation_update(context: TurnContext, _: TurnState):
            action = context.activity.action
            logger.info(f"InstallationUpdate received: action={action}")
            if action == "add":
                await context.send_activity(
                    "Welcome! I'm the Hello World agent. "
                    "Send me a message and I'll echo it back. "
                    "Type 'help' to learn more."
                )
            elif action == "remove":
                await context.send_activity("Thank you for your time, I enjoyed working with you.")

        @self.agent_app.activity("conversationUpdate")
        async def on_conversation_update(context: TurnContext, _: TurnState):
            if context.activity.members_added:
                for member in context.activity.members_added:
                    if member.id != context.activity.recipient.id:
                        logger.info(f"New member added: {member.id}")
                        await context.send_activity(
                            "Welcome! I'm the Hello World agent. "
                            "Send me a message and I'll echo it back. "
                            "Type 'help' to learn more."
                        )

        @self.agent_app.activity("message", **handler_config)
        async def on_message(context: TurnContext, _: TurnState):
            try:
                if not self.agent_instance:
                    await context.send_activity("Sorry, the agent is not available.")
                    return

                user_message = context.activity.text or ""
                logger.info(f"Processing message: '{user_message}'")

                if not user_message.strip() or user_message.strip() == "/help":
                    return

                # Cache observability token for A365 exporter
                if OBSERVABILITY_AVAILABLE and self.auth_handler_name:
                    await self._cache_observability_token(context)

                await context.send_activity("Got it - working on it...")
                await context.send_activity(Activity(type="typing"))

                async def _typing_loop():
                    while True:
                        try:
                            await asyncio.sleep(4)
                            await context.send_activity(Activity(type="typing"))
                        except asyncio.CancelledError:
                            break

                typing_task = asyncio.create_task(_typing_loop())
                try:
                    response = await self.agent_instance.process_user_message(
                        user_message, self.agent_app.auth, context, self.auth_handler_name
                    )
                    await context.send_activity(response)
                    logger.info("Response sent successfully")
                finally:
                    typing_task.cancel()
                    try:
                        await typing_task
                    except asyncio.CancelledError:
                        pass

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await context.send_activity(f"Sorry, I encountered an error: {str(e)}")

        # Register notification handlers if available
        if NOTIFICATIONS_AVAILABLE:
            self._setup_notification_handlers(handler_config)

    def _setup_notification_handlers(self, handler_config):
        """Setup notification handlers for agents and msteams channels"""

        async def handle_notification_common(
            context: TurnContext,
            state: TurnState,
            notification_activity: AgentNotificationActivity,
        ):
            try:
                logger.info(f"Notification received! Type: {context.activity.type}")

                if not self.agent_instance:
                    await context.send_activity("Sorry, the agent is not available.")
                    return

                if not hasattr(self.agent_instance, "handle_agent_notification_activity"):
                    await context.send_activity("This agent doesn't support notification handling yet.")
                    return

                response = await self.agent_instance.handle_agent_notification_activity(
                    notification_activity, self.agent_app.auth, context, self.auth_handler_name
                )

                if notification_activity.notification_type == NotificationTypes.EMAIL_NOTIFICATION:
                    response_activity = EmailResponse.create_email_response_activity(response)
                    await context.send_activity(response_activity)
                    return

                await context.send_activity(response)

            except Exception as e:
                logger.error(f"Notification error: {e}")
                await context.send_activity(f"Sorry, error processing notification: {str(e)}")

        @self.agent_notification.on_agent_notification(
            channel_id=ChannelId(channel="agents", sub_channel="*"),
            **handler_config,
        )
        async def on_notification_agents(context, state, notification_activity):
            await handle_notification_common(context, state, notification_activity)

        @self.agent_notification.on_agent_notification(
            channel_id=ChannelId(channel="msteams", sub_channel="*"),
            **handler_config,
        )
        async def on_notification_msteams(context, state, notification_activity):
            await handle_notification_common(context, state, notification_activity)

        logger.info("Notification handlers registered for 'agents' and 'msteams' channels")

    async def initialize_agent(self):
        if self.agent_instance is None:
            logger.info(f"Initializing {self.agent_class.__name__}...")
            self.agent_instance = self.agent_class(*self.agent_args, **self.agent_kwargs)
            await self.agent_instance.initialize()
            logger.info(f"{self.agent_class.__name__} initialized successfully")

    async def _cache_observability_token(self, context: TurnContext) -> None:
        """Exchange and cache an agentic token for the A365 observability exporter."""
        try:
            from microsoft_agents_a365.runtime.environment_utils import (
                get_observability_authentication_scope,
            )

            tenant_id = context.activity.recipient.tenant_id if context.activity.recipient else None
            agent_id = context.activity.recipient.agentic_app_id if context.activity.recipient else None

            if not tenant_id or not agent_id:
                return

            exchange_kwargs = {}
            if self.auth_handler_name:
                exchange_kwargs["auth_handler_id"] = self.auth_handler_name

            token_response = await self.agent_app.auth.exchange_token(
                context,
                scopes=get_observability_authentication_scope(),
                **exchange_kwargs,
            )
            cache_agentic_token(tenant_id, agent_id, token_response.token)
            logger.debug(f"Cached observability token for {tenant_id}:{agent_id}")
        except Exception as e:
            logger.warning(f"Failed to cache observability token: {e}")

    def create_auth_configuration(self) -> AgentAuthConfiguration | None:
        client_id = environ.get("CLIENT_ID")
        tenant_id = environ.get("TENANT_ID")
        client_secret = environ.get("CLIENT_SECRET")

        if client_id and tenant_id and client_secret:
            logger.info("Using Client Credentials authentication")
            try:
                return AgentAuthConfiguration(
                    client_id=client_id,
                    tenant_id=tenant_id,
                    client_secret=client_secret,
                    scopes=["https://api.botframework.com/.default"],
                )
            except Exception as e:
                logger.error(f"Failed to create AgentAuthConfiguration: {e}")
                return None

        if environ.get("BEARER_TOKEN"):
            logger.info("BEARER_TOKEN present; continuing in anonymous dev mode")
        else:
            logger.warning("No authentication env vars found; running anonymous")

        return None

    def start_server(self, auth_configuration: AgentAuthConfiguration | None = None):
        async def entry_point(req: Request) -> Response:
            agent: AgentApplication = req.app["agent_app"]
            adapter: CloudAdapter = req.app["adapter"]
            return await start_agent_process(req, agent, adapter)

        async def init_app(app):
            await self.initialize_agent()

        async def health(_req: Request) -> Response:
            return json_response({
                "status": "ok",
                "agent_type": self.agent_class.__name__,
                "agent_initialized": self.agent_instance is not None,
                "auth_mode": "authenticated" if auth_configuration else "anonymous",
            })

        # Build middleware
        middlewares = []
        if auth_configuration:
            @web_middleware
            async def auth_with_exclusions(request, handler):
                path = request.path.lower()
                if path in ['/api/health', '/robots933456.txt', '/', '/privacy', '/terms']:
                    return await handler(request)
                return await jwt_authorization_middleware(request, handler)
            middlewares.append(auth_with_exclusions)

        @web_middleware
        async def anonymous_claims(request, handler):
            if not auth_configuration:
                request["claims_identity"] = ClaimsIdentity(
                    {
                        AuthenticationConstants.AUDIENCE_CLAIM: "anonymous",
                        AuthenticationConstants.APP_ID_CLAIM: "anonymous-app",
                    },
                    False,
                    "Anonymous",
                )
            return await handler(request)

        middlewares.append(anonymous_claims)
        app = Application(middlewares=middlewares)

        # Static pages for Teams manifest validation
        async def privacy_page(_req: Request) -> Response:
            return Response(
                text="<html><body><h1>Privacy Policy</h1><p>This agent collects no personal data.</p></body></html>",
                content_type="text/html",
            )

        async def terms_page(_req: Request) -> Response:
            return Response(
                text="<html><body><h1>Terms of Use</h1><p>Use this agent at your own discretion.</p></body></html>",
                content_type="text/html",
            )

        async def root_page(_req: Request) -> Response:
            return Response(
                text=f"<html><body><h1>{self.agent_class.__name__}</h1><p>Agent is running.</p></body></html>",
                content_type="text/html",
            )

        # Routes
        app.router.add_post("/api/messages", entry_point)
        app.router.add_get("/api/messages", lambda _: Response(status=200))
        app.router.add_get("/api/health", health)
        app.router.add_get("/privacy", privacy_page)
        app.router.add_get("/terms", terms_page)
        app.router.add_get("/", root_page)

        # Context
        app["agent_configuration"] = auth_configuration
        app["agent_app"] = self.agent_app
        app["adapter"] = self.agent_app.adapter
        app.on_startup.append(init_app)

        # Port configuration
        desired_port = int(environ.get("PORT", 3978))
        port = desired_port

        # Host configuration
        if "HOST" in environ:
            host = environ["HOST"]
        elif environ.get("WEBSITE_INSTANCE_ID"):
            host = "0.0.0.0"
        else:
            host = "localhost"

        # Port availability check for local dev
        if host in ("localhost", "127.0.0.1"):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex(("127.0.0.1", desired_port)) == 0:
                    logger.warning(f"Port {desired_port} in use, trying {desired_port + 1}")
                    port = desired_port + 1

        print("=" * 80)
        print(f"Generic Agent Host - {self.agent_class.__name__}")
        print("=" * 80)
        print(f"\nAuthentication: {'Enabled' if auth_configuration else 'Anonymous'}")
        print(f"Starting server on {host}:{port}")
        print(f"Bot Framework endpoint: http://{host}:{port}/api/messages")
        print(f"Health: http://{host}:{port}/api/health")
        print("Ready for testing!\n")

        try:
            run_app(app, host=host, port=port)
        except KeyboardInterrupt:
            print("\nServer stopped")

    async def cleanup(self):
        if self.agent_instance:
            await self.agent_instance.cleanup()
            logger.info("Agent cleanup completed")


def create_and_run_host(agent_class: type[AgentInterface], *agent_args, **agent_kwargs):
    """Convenience function to create and run a generic agent host."""
    try:
        if not check_agent_inheritance(agent_class):
            raise TypeError(f"Agent class {agent_class.__name__} must inherit from AgentInterface")

        # Configure observability if available and enabled
        if OBSERVABILITY_AVAILABLE:
            enable_observability = os.getenv("ENABLE_OBSERVABILITY", "false").lower() in ("true", "1", "yes")
            if enable_observability:
                service_name = os.getenv("OBSERVABILITY_SERVICE_NAME", "generic-agent-host")
                service_namespace = os.getenv("OBSERVABILITY_SERVICE_NAMESPACE", "agent365")

                def token_resolver(agent_id: str, tenant_id: str) -> str | None:
                    try:
                        return get_cached_agentic_token(tenant_id, agent_id)
                    except Exception as e:
                        logger.warning(f"Error resolving token for observability: {e}")
                        return None

                try:
                    configure_observability(
                        service_name=service_name,
                        service_namespace=service_namespace,
                        token_resolver=token_resolver,
                        cluster_category=os.getenv("PYTHON_ENVIRONMENT", "development"),
                    )
                    logger.info(f"Observability configured: {service_name} ({service_namespace})")
                except Exception as e:
                    logger.warning(f"Failed to configure observability: {e}")

        host = GenericAgentHost(agent_class, *agent_args, **agent_kwargs)
        auth_config = host.create_auth_configuration()
        host.start_server(auth_config)

    except Exception as error:
        logger.error(f"Failed to start generic agent host: {error}")
        raise error
