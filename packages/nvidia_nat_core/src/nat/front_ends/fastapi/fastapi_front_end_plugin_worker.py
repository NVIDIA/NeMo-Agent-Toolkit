# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import logging
import os
from abc import ABC
from abc import abstractmethod
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nat.builder.evaluator import EvaluatorInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.builder.workflow_builder import WorkflowEvalBuilderBase
from nat.data_models.config import Config
from nat.runtime.session import SessionManager
from nat.utils.log_utils import setup_logging

from .auth_flow_handlers.http_flow_handler import HTTPAuthenticationFlowHandler
from .auth_flow_handlers.websocket_flow_handler import FlowState
from .execution_store import ExecutionStore
from .fastapi_front_end_config import FastApiFrontEndConfig
from .message_handler import WebSocketMessageHandler
from .routes.auth import add_authorization_route
from .routes.chat import add_chat_routes
from .routes.execution import add_execution_routes
from .routes.generate import add_generate_routes
from .routes.health import add_health_route
from .routes.monitor import add_monitor_route
from .routes.static import add_static_files_route
from .routes.websocket import add_websocket_routes
from .step_adaptor import StepAdaptor
from .utils import get_config_file_path

logger = logging.getLogger(__name__)


class FastApiFrontEndPluginWorkerBase(ABC):

    def __init__(self, config: Config):
        self._config = config

        assert isinstance(config.general.front_end,
                          FastApiFrontEndConfig), ("Front end config is not FastApiFrontEndConfig")

        self._front_end_config = config.general.front_end
        self._dask_available = False
        self._job_store = None
        self._http_flow_handler: HTTPAuthenticationFlowHandler | None = HTTPAuthenticationFlowHandler()
        self._scheduler_address = os.environ.get("NAT_DASK_SCHEDULER_ADDRESS")
        self._db_url = os.environ.get("NAT_JOB_STORE_DB_URL")
        self._config_file_path = get_config_file_path()
        self._use_dask_threads = os.environ.get("NAT_USE_DASK_THREADS", "0") == "1"
        self._log_level = int(os.environ.get("NAT_FASTAPI_LOG_LEVEL", logging.INFO))
        setup_logging(self._log_level)

        if self._scheduler_address is not None:
            try:
                from nat.front_ends.fastapi.async_jobs.job_store import JobStore
                if self._db_url is None:
                    raise RuntimeError(
                        "NAT_JOB_STORE_DB_URL must be set when using Dask (configure a persistent JobStore database).")
                self._job_store = JobStore(scheduler_address=self._scheduler_address, db_url=self._db_url)
                self._dask_available = True
                logger.debug("Connected to Dask scheduler at %s", self._scheduler_address)
            except ImportError as e:
                raise RuntimeError(
                    "Dask is not available, please install it to use the FastAPI front end with Dask.") from e
            except Exception as e:
                raise RuntimeError(f"Failed to connect to Dask scheduler at {self._scheduler_address}: {e}") from e
        else:
            logger.debug("No Dask scheduler address provided, running without Dask support.")

    @property
    def config(self) -> Config:
        return self._config

    @property
    def front_end_config(self) -> FastApiFrontEndConfig:
        return self._front_end_config

    def build_app(self) -> FastAPI:

        # Create the FastAPI app and configure it
        @asynccontextmanager
        async def lifespan(starting_app: FastAPI):

            logger.debug("Starting NAT server from process %s", os.getpid())

            async with WorkflowBuilder.from_config(self.config) as builder:

                await self.configure(starting_app, builder)

                yield

            logger.debug("Closing NAT server from process %s", os.getpid())

        nat_app = FastAPI(lifespan=lifespan)

        # Configure app CORS.
        self.set_cors_config(nat_app)

        @nat_app.middleware("http")
        async def authentication_log_filter(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
            return await self._suppress_authentication_logs(request, call_next)

        return nat_app

    def set_cors_config(self, nat_app: FastAPI) -> None:
        """
        Set the cross origin resource sharing configuration.
        """
        cors_kwargs = {}

        if self.front_end_config.cors.allow_origins is not None:
            cors_kwargs["allow_origins"] = self.front_end_config.cors.allow_origins

        if self.front_end_config.cors.allow_origin_regex is not None:
            cors_kwargs["allow_origin_regex"] = self.front_end_config.cors.allow_origin_regex

        if self.front_end_config.cors.allow_methods is not None:
            cors_kwargs["allow_methods"] = self.front_end_config.cors.allow_methods

        if self.front_end_config.cors.allow_headers is not None:
            cors_kwargs["allow_headers"] = self.front_end_config.cors.allow_headers

        if self.front_end_config.cors.allow_credentials is not None:
            cors_kwargs["allow_credentials"] = self.front_end_config.cors.allow_credentials

        if self.front_end_config.cors.expose_headers is not None:
            cors_kwargs["expose_headers"] = self.front_end_config.cors.expose_headers

        if self.front_end_config.cors.max_age is not None:
            cors_kwargs["max_age"] = self.front_end_config.cors.max_age

        nat_app.add_middleware(
            CORSMiddleware,
            **cors_kwargs,
        )

    async def _suppress_authentication_logs(self, request: Request,
                                            call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """
        Intercepts authentication request and supreses logs that contain sensitive data.
        """
        from nat.utils.log_utils import LogFilter

        logs_to_suppress: list[str] = []

        if (self.front_end_config.oauth2_callback_path):
            logs_to_suppress.append(self.front_end_config.oauth2_callback_path)

        logging.getLogger("uvicorn.access").addFilter(LogFilter(logs_to_suppress))
        try:
            response = await call_next(request)
        finally:
            logging.getLogger("uvicorn.access").removeFilter(LogFilter(logs_to_suppress))

        return response

    @abstractmethod
    async def configure(self, app: FastAPI, builder: WorkflowBuilder):
        pass

    @abstractmethod
    def get_step_adaptor(self) -> StepAdaptor:
        pass


class RouteInfo(BaseModel):

    function_name: str | None


class FastApiFrontEndPluginWorker(FastApiFrontEndPluginWorkerBase):

    def __init__(self, config: Config):
        super().__init__(config)

        self._outstanding_flows: dict[str, FlowState] = {}
        self._outstanding_flows_lock = asyncio.Lock()

        # Conversation handlers for WebSocket reconnection support
        self._conversation_handlers: dict[str, WebSocketMessageHandler] = {}

        # Track session managers for each route
        self._session_managers: list[SessionManager] = []

        # Evaluator storage for single-item evaluation
        self._evaluators: dict[str, EvaluatorInfo] = {}
        self._eval_builder: WorkflowEvalBuilderBase | None = None

        # HTTP interactive execution store
        self._execution_store = ExecutionStore()

        # Re-create the HTTP flow handler with OAuth flow callbacks for interactive mode
        self._http_flow_handler = HTTPAuthenticationFlowHandler(
            add_flow_cb=self._add_flow,
            remove_flow_cb=self._remove_flow,
        )

    def get_conversation_handler(self, conversation_id: str) -> "WebSocketMessageHandler | None":
        """Get a conversation handler for reconnection support."""
        return self._conversation_handlers.get(conversation_id)

    def set_conversation_handler(self, conversation_id: str, handler: "WebSocketMessageHandler") -> None:
        """Register a conversation handler for reconnection support."""
        self._conversation_handlers[conversation_id] = handler

    def remove_conversation_handler(self, conversation_id: str) -> None:
        """Remove a conversation handler when workflow completes."""
        self._conversation_handlers.pop(conversation_id, None)

    async def initialize_evaluators(self, config: Config):
        """Initialize and store evaluators from config for single-item evaluation."""
        try:
            from nat.plugins.eval.runtime.builder import WorkflowEvalBuilder
        except ImportError:
            logger.info("Evaluation package not installed, skipping evaluator initialization")
            return

        if not config.eval or not config.eval.evaluators:
            logger.info("No evaluators configured, skipping evaluator initialization")
            return

        try:
            # Build evaluators using WorkflowEvalBuilder (same pattern as nat eval)
            # Start with registry=None and let populate_builder set everything up
            eval_builder = WorkflowEvalBuilder(
                general_config=config.general,
                eval_general_config=config.eval.general,
                registry=None,
            )
            self._eval_builder = eval_builder

            # Enter the async context and keep it alive
            await eval_builder.__aenter__()

            # Populate builder with config (this sets up LLMs, functions, etc.)
            # Skip workflow build since we already have it from the main builder
            await eval_builder.populate_builder(config, skip_workflow=True)

            # Now evaluators should be populated by populate_builder
            for name in config.eval.evaluators.keys():
                self._evaluators[name] = eval_builder.get_evaluator(name)
                logger.info("Initialized evaluator: %s", name)

            logger.info("Successfully initialized %d evaluators", len(self._evaluators))

        except Exception as e:
            logger.error("Failed to initialize evaluators: %s", e)
            # Don't fail startup, just log the error
            self._evaluators = {}

    async def _create_session_manager(self,
                                      builder: WorkflowBuilder,
                                      entry_function: str | None = None) -> SessionManager:
        """Create and register a SessionManager."""

        sm = await SessionManager.create(config=self._config, shared_builder=builder, entry_function=entry_function)
        self._session_managers.append(sm)

        return sm

    async def cleanup_session_managers(self):
        """Clean up all SessionManager resources on shutdown."""
        for sm in self._session_managers:
            try:
                await sm.shutdown()
            except Exception as e:
                logger.error(f"Error cleaning up SessionManager: {e}")

        self._session_managers.clear()
        logger.info("All SessionManagers cleaned up")

    async def cleanup_evaluators(self):
        """Clean up evaluator resources on shutdown."""
        if self._eval_builder:
            try:
                await self._eval_builder.__aexit__(None, None, None)
                logger.info("Evaluator builder context cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up evaluator builder: {e}")
            finally:
                self._eval_builder = None
                self._evaluators.clear()

    def get_step_adaptor(self) -> StepAdaptor:

        return StepAdaptor(self.front_end_config.step_adaptor)

    async def configure(self, app: FastAPI, builder: WorkflowBuilder):

        # Do things like setting the base URL and global configuration options
        app.root_path = self.front_end_config.root_path

        # Initialize evaluators for single-item evaluation
        # TODO: we need config control over this as it's not always needed
        await self.initialize_evaluators(self._config)

        # Ensure session manager resources are cleaned up when the app shuts down
        app.add_event_handler("shutdown", self.cleanup_session_managers)

        # Ensure evaluator resources are cleaned up when the app shuts down
        app.add_event_handler("shutdown", self.cleanup_evaluators)

        await self.add_routes(app, builder)

    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder):

        session_manager = await self._create_session_manager(builder)

        await add_authorization_route(self, app)
        await add_execution_routes(self, app)
        await add_monitor_route(self, app)
        await add_health_route(app)
        await add_static_files_route(self, app, builder)

        await self.add_default_route(app, session_manager)

        try:
            from nat.plugins.eval.fastapi.routes import add_evaluate_routes
            await add_evaluate_routes(self, app, session_manager=session_manager)
        except ImportError:
            logger.warning("nvidia-nat-eval is not installed; skipping evaluate routes.")

        try:
            from nat.plugins.mcp.client.fastapi_routes import add_mcp_client_tool_list_route
            await add_mcp_client_tool_list_route(app, builder, self._session_managers)
        except ImportError:
            logger.warning("nvidia-nat-mcp is not installed; skipping MCP client tool list routes.")

        disable_legacy_routes: bool = self.front_end_config.disable_legacy_routes
        enable_interactive_extensions: bool = self.front_end_config.enable_interactive_extensions

        for ep in self.front_end_config.endpoints:
            session_manager = await self._create_session_manager(builder, ep.function_name)
            await add_generate_routes(self, app, ep, session_manager, disable_legacy_routes=disable_legacy_routes)
            await add_chat_routes(self,
                                  app,
                                  ep,
                                  session_manager,
                                  enable_interactive_extensions=enable_interactive_extensions,
                                  disable_legacy_routes=disable_legacy_routes)
            await add_websocket_routes(self, app, ep, session_manager)

    async def add_default_route(self, app: FastAPI, session_manager: SessionManager):

        disable_legacy_routes: bool = self.front_end_config.disable_legacy_routes
        enable_interactive_extensions: bool = self.front_end_config.enable_interactive_extensions

        await add_generate_routes(self,
                                  app,
                                  self.front_end_config.workflow,
                                  session_manager,
                                  disable_legacy_routes=disable_legacy_routes)
        await add_chat_routes(self,
                              app,
                              self.front_end_config.workflow,
                              session_manager,
                              enable_interactive_extensions=enable_interactive_extensions,
                              disable_legacy_routes=disable_legacy_routes)
        await add_websocket_routes(self, app, self.front_end_config.workflow, session_manager)

    async def _add_flow(self, state: str, flow_state: FlowState):
        async with self._outstanding_flows_lock:
            self._outstanding_flows[state] = flow_state

    async def _remove_flow(self, state: str):
        async with self._outstanding_flows_lock:
            self._outstanding_flows.pop(state, None)


# Prevent Sphinx from documenting items not a part of the public API
__all__ = ["FastApiFrontEndPluginWorkerBase", "FastApiFrontEndPluginWorker", "RouteInfo"]
