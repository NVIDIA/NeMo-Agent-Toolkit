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
import typing
from abc import ABC
from abc import abstractmethod
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from nat.builder.eval_builder import WorkflowEvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import Usage
from nat.data_models.config import Config
from nat.eval.evaluate import EvaluationRun  # Backwards compatibility for tests
from nat.front_ends.fastapi.api_versioning import VersioningOptions
from nat.front_ends.fastapi.api_versioning import apply_version_headers
from nat.front_ends.fastapi.auth_flow_handlers.http_flow_handler import HTTPAuthenticationFlowHandler
from nat.front_ends.fastapi.auth_flow_handlers.websocket_flow_handler import FlowState
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.routes.hitl import HitlHttpHandler
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.front_ends.fastapi.utils import get_config_file_path
from nat.runtime.session import SessionManager
from nat.utils.log_utils import setup_logging

logger = logging.getLogger(__name__)

_DASK_AVAILABLE = False

try:
    from nat.front_ends.fastapi.job_store import JobInfo
    from nat.front_ends.fastapi.job_store import JobStatus
    from nat.front_ends.fastapi.job_store import JobStore
    _DASK_AVAILABLE = True
except ImportError:
    JobInfo = typing.cast(typing.Any, None)
    JobStatus = typing.cast(typing.Any, None)
    JobStore = typing.cast(typing.Any, None)


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
        self._versioning_options = VersioningOptions.from_config(self._front_end_config)
        self._hitl_http = HitlHttpHandler(self._front_end_config.hitl)
        setup_logging(self._log_level)

        if self._scheduler_address is not None:
            if not _DASK_AVAILABLE:
                raise RuntimeError("Dask is not available, please install it to use the FastAPI front end with Dask.")

            if self._db_url is None:
                raise RuntimeError(
                    "NAT_JOB_STORE_DB_URL must be set when using Dask (configure a persistent JobStore database).")

            try:
                self._job_store = JobStore(scheduler_address=self._scheduler_address, db_url=self._db_url)
                self._dask_available = True
                logger.debug("Connected to Dask scheduler at %s", self._scheduler_address)
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

            async with typing.cast(typing.AsyncContextManager[WorkflowBuilder],
                                   WorkflowBuilder.from_config(self.config)) as builder:

                await self.configure(starting_app, builder)

                yield

            logger.debug("Closing NAT server from process %s", os.getpid())

        nat_app = FastAPI(lifespan=lifespan)

        # Configure app CORS.
        self.set_cors_config(nat_app)

        @nat_app.middleware("http")
        async def authentication_log_filter(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
            return await self._suppress_authentication_logs(request, call_next)

        @nat_app.middleware("http")
        async def versioning_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
            response = await call_next(request)
            apply_version_headers(response, self._versioning_options)
            return response

        return nat_app

    def set_cors_config(self, nat_app: FastAPI) -> None:
        """
        Set the cross origin resource sharing configuration.
        """
        cors_kwargs: dict[str, typing.Any] = {}

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
            CORSMiddleware,  # type: ignore[arg-type]
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

        # Track session managers for each route
        self._session_managers: list[SessionManager] = []

        # Evaluator storage for single-item evaluation
        self._evaluators: dict[str, EvaluatorInfo] = {}
        self._eval_builder: WorkflowEvalBuilder | None = None

    async def initialize_evaluators(self, config: Config):
        """Initialize and store evaluators from config for single-item evaluation."""
        if not config.eval or not config.eval.evaluators:
            logger.info("No evaluators configured, skipping evaluator initialization")
            return

        try:
            # Build evaluators using WorkflowEvalBuilder (same pattern as nat eval)
            # Start with registry=None and let populate_builder set everything up
            self._eval_builder = WorkflowEvalBuilder(general_config=config.general,
                                                     eval_general_config=config.eval.general,
                                                     registry=None)

            # Enter the async context and keep it alive
            await self._eval_builder.__aenter__()

            # Populate builder with config (this sets up LLMs, functions, etc.)
            # Skip workflow build since we already have it from the main builder
            await self._eval_builder.populate_builder(config, skip_workflow=True)

            # Now evaluators should be populated by populate_builder
            for name in config.eval.evaluators.keys():
                self._evaluators[name] = self._eval_builder.get_evaluator(name)
                logger.info(f"Initialized evaluator: {name}")

            logger.info(f"Successfully initialized {len(self._evaluators)} evaluators")

        except Exception as e:
            logger.error(f"Failed to initialize evaluators: {e}")
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

    def _get_user_input_callback(self, request: Request | None):
        """Return HITL user callback if HTTP HITL is enabled and request is available."""
        if not (self.front_end_config.hitl.enable_http or self.front_end_config.hitl.enable_sse):
            return None
        if request is None:
            return None
        session_id = HitlHttpHandler.resolve_session_id(request)
        return self._hitl_http.user_callback(session_id)

    @staticmethod
    def _ensure_usage(result: typing.Any) -> None:
        """Ensure ChatResponse includes usage per OpenAI spec."""
        if isinstance(result, ChatResponse) and result.usage != Usage():
            result.usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def _openai_error_response(self, exc: Exception) -> JSONResponse:
        """Format errors to OpenAI-compatible structure."""
        status_code = 500
        message = str(exc)
        error_type = "server_error"

        if isinstance(exc, HTTPException):
            status_code = exc.status_code
            error_type = "invalid_request_error" if status_code < 500 else "server_error"

        return JSONResponse(status_code=status_code,
                            content={"error": {
                                "message": message, "type": error_type, "param": None, "code": None
                            }})

    def _versioned_path(self, path: str) -> str:
        """Prefix a path with /v{n} if not already versioned."""
        prefix = f"/v{self.front_end_config.versioning.version}"
        normalized = path if path.startswith("/") else f"/{path}"
        if normalized.startswith(prefix):
            return normalized
        return f"{prefix}{normalized}"

    def _register_api_route(self, app: FastAPI, *, path: str, **kwargs) -> None:
        """Register route on legacy + versioned paths depending on config."""
        if self.front_end_config.versioning.enable_legacy_routes:
            app.add_api_route(path=path, **kwargs)
        versioned = self._versioned_path(path)
        if versioned != path:
            app.add_api_route(path=versioned, **kwargs)

    def _register_websocket_route(self, app: FastAPI, *, path: str, endpoint) -> None:
        if self.front_end_config.versioning.enable_legacy_routes:
            app.add_websocket_route(path, endpoint)
        versioned = self._versioned_path(path)
        if versioned != path:
            app.add_websocket_route(versioned, endpoint)

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
        self._hitl_http.register_routes(app)

    async def add_route(self,
                        app: FastAPI,
                        endpoint: FastApiFrontEndConfig.EndpointBase,
                        session_manager: SessionManager):
        """Register workflow/chat/OpenAI endpoints for the configured endpoint."""
        from nat.front_ends.fastapi.routes.websocket import register_websocket_route
        from nat.front_ends.fastapi.routes.workflow import register_workflow_route
        await register_workflow_route(self, app, endpoint, session_manager)
        await register_websocket_route(self, app, endpoint, session_manager)

    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder):
        """Register all HTTP/SSE/WebSocket routes."""
        from nat.front_ends.fastapi.routes.auth import register_auth_routes
        from nat.front_ends.fastapi.routes.monitor import register_monitor_routes
        from nat.front_ends.fastapi.routes.static_files import register_static_file_routes

        session_manager = await self._create_session_manager(builder)

        # Register the primary workflow route
        await self.add_route(app, self.front_end_config.workflow, session_manager)

        # Register the static files route
        await register_static_file_routes(self, app, builder)

        # Register the authorization route
        await register_auth_routes(self, app)

        # Register the monitor route
        await register_monitor_routes(self, app)

        # Register the MCP client tool list route
        try:
            from nat.plugins.mcp.server.routes import register_mcp_routes
            await register_mcp_routes(self, app, builder)
        except ImportError:
            # Plugin not available; skip endpoint registration
            pass

        for ep in self.front_end_config.endpoints:
            await self.add_route(app,
                                 endpoint=ep,
                                 session_manager=await self._create_session_manager(builder, ep.function_name))

    async def add_evaluate_route(self, app: FastAPI, session_manager: SessionManager):
        """
        Compatibility wrapper to register the evaluate endpoint.
        """
        from nat.front_ends.fastapi.routes.evaluate import register_evaluate_route

        await register_evaluate_route(self, app, session_manager)

    async def add_evaluate_item_route(self, app: FastAPI, session_manager: SessionManager):
        """
        Compatibility wrapper to register the evaluate item endpoint.
        """
        from nat.front_ends.fastapi.routes.evaluate import register_evaluate_item_route

        await register_evaluate_item_route(self, app, session_manager)

    async def _add_flow(self, state: str, flow_state: FlowState):
        async with self._outstanding_flows_lock:
            self._outstanding_flows[state] = flow_state

    async def _remove_flow(self, state: str):
        async with self._outstanding_flows_lock:
            del self._outstanding_flows[state]


# Prevent Sphinx from documenting items not a part of the public API
__all__ = ["FastApiFrontEndPluginWorkerBase", "FastApiFrontEndPluginWorker", "RouteInfo", "EvaluationRun"]
