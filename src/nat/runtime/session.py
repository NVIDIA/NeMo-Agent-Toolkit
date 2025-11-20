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

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextlib import nullcontext
from datetime import datetime
from datetime import timedelta

import aiorwlock
from fastapi import WebSocket
from starlette.requests import HTTPConnection
from starlette.requests import Request

from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import PerUserWorkflowBuilder
from nat.builder.workflow_builder import SharedWorkflowBuilder
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.config import Config
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import InteractionPrompt
from nat.data_models.runtime_enum import RuntimeTypeEnum
from nat.data_models.user_workflow_data import UserWorkflowData
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.runtime.runner import Runner

logger = logging.getLogger(__name__)


class UserSession:

    def __init__(self,
                 workflow: Workflow,
                 max_concurrency: int = 8,
                 runtime_type: RuntimeTypeEnum = RuntimeTypeEnum.RUN_OR_SERVE):
        """
        The UserSession class is used to run and manage a user workflow session. It runs a workflow for a single user
        with the specified concurrency limit.

        Parameters
        ----------
        workflow : Workflow
            The workflow to run
        max_concurrency : int, optional
            The maximum number of simultaneous workflow invocations, by default 8
        """
        if workflow is None:
            raise ValueError("Workflow must be provided to initialize a UserSession")

        self._workflow: Workflow = workflow
        self._max_concurrency = max_concurrency

        if max_concurrency > 0:
            self._semaphore = asyncio.Semaphore(max_concurrency)
        else:
            # If max_concurrency is 0, then we don't need to limit the concurrency but we still need a context
            self._semaphore = nullcontext()

    @property
    def workflow(self) -> Workflow:
        "Get the user's workflow instance"
        return self._workflow

    @property
    def config(self) -> Config:
        "Get the user's workflow configuration"
        return self._workflow.config

    @asynccontextmanager
    async def run(self, message) -> AsyncGenerator[Runner, None]:
        """
        Run a workflow for a single user
        """
        async with self._semaphore:
            async with self._workflow.run(message) as runner:
                yield runner


class SessionManager:

    def __init__(self, config: Config, max_concurrency: int = 8):
        """
        The SessionManager class is for per-user workflow management.

        Accepts config and builds workflows on-demand for each user.

        Parameters
        ----------
        config : Config
            The configuration for building the workflow
        max_concurrency : int, optional
            The maximum number of simultaneous workflow invocations, by default 8
        """

        self._config = config
        self._max_concurrency = max_concurrency

        front_end_config = config.general.front_end

        if isinstance(front_end_config, FastApiFrontEndConfig):
            self._max_users = front_end_config.max_concurrent_users
            self._user_idle_timeout = front_end_config.user_idle_timeout
            self._cleanup_check_interval = front_end_config.cleanup_check_interval
            self._require_user_id = front_end_config.require_user_id
        else:
            self._max_users = 100
            self._user_idle_timeout = timedelta(minutes=30)
            self._cleanup_check_interval = timedelta(minutes=10)
            self._require_user_id = False

        # Per-user workflow registry
        self._user_workflows: dict[str, UserWorkflowData] = {}
        self._users_lock = aiorwlock.RWLock()

        # Shared builder
        self._shared_builder: SharedWorkflowBuilder | None = None
        self._shared_builder_lock = asyncio.Lock()

        # Cleanup task
        self._cleanup_task: asyncio.Task | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()

        # Context state
        self._context_state = ContextState.get()
        self._context = Context(self._context_state)

    @property
    def config(self) -> Config:
        return self._config

    @property
    def context(self) -> Context:
        """Get the context."""
        return self._context

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def max_users(self) -> int:
        return self._max_users

    @property
    def user_idle_timeout(self) -> timedelta:
        return self._user_idle_timeout

    @property
    def cleanup_check_interval(self) -> timedelta:
        return self._cleanup_check_interval

    @property
    def require_user_id(self) -> bool:
        return self._require_user_id

    @property
    def active_user_count(self) -> int:
        return len(self._user_workflows)

    @property
    def user_limit(self) -> int:
        return self._max_users

    async def get_workflow(self, user_id: str) -> Workflow:
        """
        Get the workflow for a specific user.
        """
        workflow_data = await self._get_or_create_user_workflow(user_id)
        return workflow_data.workflow

    @staticmethod
    def _truncate_user_id(user_id: str, length: int = 8) -> str:
        """
        Truncate the user_id for logging.
        """
        if len(user_id) > length:
            return user_id[:length] + "..."
        return user_id

    async def _ensure_shared_builder(self) -> SharedWorkflowBuilder:
        """
        Ensures the shared builder is created with shared components. Lazy initialization on first user request.
        """

        # Fast path: shared builderalready initialized
        if self._shared_builder is not None:
            return self._shared_builder

        # Slow path: shared builder not initialized
        async with self._shared_builder_lock:
            # Double-check after acquiring lock
            if self._shared_builder is not None:
                return self._shared_builder

            logger.info("Initializing shared builder")

            try:
                shared_builder = SharedWorkflowBuilder(general_config=self._config.general)

                await shared_builder.__aenter__()
                await shared_builder.populate_builder(self._config)

                self._shared_builder = shared_builder
                logger.info("Shared builder initialized")

                return shared_builder

            except Exception as e:
                logger.error("Error initializing shared builder: %s", e)
                raise RuntimeError(f"Shared builder initialization failed: {e}") from e

    async def _create_user_workflow(self, user_id: str) -> tuple[Workflow, PerUserWorkflowBuilder]:
        """
        Create a new workflow for a specific user.
        """
        logger.info(f"Creating workflow for user {self._truncate_user_id(user_id)}")

        try:
            # Ensure shared components are built
            shared_builder = await self._ensure_shared_builder()

            # Create per-user builder
            user_builder = PerUserWorkflowBuilder(general_config=self._config.general,
                                                  shared_builder=shared_builder,
                                                  user_id=user_id)

            await user_builder.__aenter__()
            await user_builder.populate_builder(self._config)
            workflow = await user_builder.build()

            logger.info(f"Workflow created for user {self._truncate_user_id(user_id)}")

            return workflow, user_builder

        except Exception as e:
            logger.error(f"Error creating workflow for user {self._truncate_user_id(user_id)}: {e}")
            raise RuntimeError(f"Workflow creation failed for user {user_id}: {e}") from e

    async def _get_or_create_user_workflow(self, user_id: str) -> UserWorkflowData:
        """
        Get existing workflow for user or create lazily.
        """

        # Fast path: Check if user already exists (read lock)
        async with self._users_lock.reader:
            if user_id in self._user_workflows:
                user_data = self._user_workflows[user_id]
                # no lock needed, atomic timestamp write
                user_data.last_activity = datetime.now()
                logger.debug(f"User {self._truncate_user_id(user_id)} already exists, updating last activity")
                return user_data

        # Slow path: Need to create new workflow (write lock)
        async with self._users_lock.writer:
            # Double-check after acquiring lock
            if user_id in self._user_workflows:
                user_data = self._user_workflows[user_id]
                user_data.last_activity = datetime.now()
                logger.debug(f"User {self._truncate_user_id(user_id)} already exists, updating last activity")
                return user_data

            # Check user limit
            if len(self._user_workflows) >= self._max_users:
                logger.warning(f"User limit reached ({self._max_users}), attempting to cleanup before creating \
                    workflow for {self._truncate_user_id(user_id)}")

                cleaned = await self._cleanup_inactive_users()
                logger.info(f"Cleaned up {cleaned} inactive users")

                if len(self._user_workflows) >= self._max_users:
                    logger.warning(f"User limit still reached ({self._max_users}), rejecting new user: \
                        {self._truncate_user_id(user_id)}")
                    raise RuntimeError(f"User limit reached ({self._max_users}), rejecting new user: {user_id}")

            workflow, builder = await self._create_user_workflow(user_id)
            user_data = UserWorkflowData(user_id=user_id, workflow=workflow, builder=builder, \
                last_activity=datetime.now(), ref_count=0)

            self._user_workflows[user_id] = user_data
            logger.info(f"User {self._truncate_user_id(user_id)} created")
            return user_data

    @asynccontextmanager
    async def session(self,
                      user_manager=None,
                      http_connection: HTTPConnection | None = None,
                      user_message_id: str | None = None,
                      conversation_id: str | None = None,
                      user_input_callback: Callable[[InteractionPrompt], Awaitable[HumanResponse]] = None,
                      user_authentication_callback: Callable[[AuthProviderBaseConfig, AuthFlowType],
                                                             Awaitable[AuthenticatedContext | None]] = None):

        token_user_input = None
        if user_input_callback is not None:
            token_user_input = self._context_state.user_input_callback.set(user_input_callback)

        token_user_manager = None
        if user_manager is not None:
            token_user_manager = self._context_state.user_manager.set(user_manager)

        token_user_authentication = None
        if user_authentication_callback is not None:
            token_user_authentication = self._context_state.user_auth_callback.set(user_authentication_callback)

        if isinstance(http_connection, WebSocket):
            self.set_metadata_from_websocket(http_connection, user_message_id, conversation_id)
        elif isinstance(http_connection, Request):
            self.set_metadata_from_http_request(http_connection)

        # Extract user_id from metadata
        user_id = None
        cookies = self._context.metadata.cookies
        if cookies:
            user_id = cookies.get("nat-session")

        # Validate if required
        if self._require_user_id and not user_id:
            raise ValueError("user_id is required for this session but not found in the request.")

        # Use default if not found
        if not user_id:
            user_id = "default_user"

        user_workflow_data = await self._get_or_create_user_workflow(user_id)

        # track active requests
        async with user_workflow_data.lock:
            user_workflow_data.ref_count += 1
            logger.debug(
                f"User {self._truncate_user_id(user_id)} reference count increased to {user_workflow_data.ref_count}")
        user_session = UserSession(workflow=user_workflow_data.workflow, max_concurrency=self._max_concurrency)

        try:
            yield user_session
        finally:
            async with user_workflow_data.lock:
                user_workflow_data.ref_count -= 1
                logger.debug(
                    f"User {self._truncate_user_id(user_id)} reference count decreased to {user_workflow_data.ref_count}"
                )

            if token_user_manager is not None:
                self._context_state.user_manager.reset(token_user_manager)
            if token_user_input is not None:
                self._context_state.user_input_callback.reset(token_user_input)
            if token_user_authentication is not None:
                self._context_state.user_auth_callback.reset(token_user_authentication)

    async def _cleanup_inactive_users(self) -> int:
        """
        Remove workflows for inactive users.
        """
        now = datetime.now()
        to_cleanup: list[tuple[str, UserWorkflowData]] = []

        async with self._users_lock.writer:
            for user_id, user_data in self._user_workflows.items():
                async with user_data.lock:
                    is_idle = now - user_data.last_activity > self._user_idle_timeout
                    is_inactive = user_data.ref_count <= 0

                    if is_idle and is_inactive:
                        to_cleanup.append((user_id, self._user_workflows.pop(user_id)))
                        logger.info(f"User {self._truncate_user_id(user_id)} is idle and inactive, removing")

        # Clean up outside of the lock
        for user_id, user_data in to_cleanup:
            try:
                # Exit builder context (release resources)
                if user_data.builder:
                    await user_data.builder.__aexit__(None, None, None)
                    logger.info(f"Builder exited for user {self._truncate_user_id(user_id)}")

            except Exception as e:
                logger.error(f"Error cleaning up user {self._truncate_user_id(user_id)}: {e}")

        if to_cleanup:
            logger.info(f"Cleanup completed: {len(to_cleanup)} users removed "
                        f"(remaining: {len(self._user_workflows)}/{self._max_users})")

        return len(to_cleanup)

    async def _cleanup_loop(self):
        """
        Background task that periodically cleans up inactive users. Runs until shutdown_event is set.
        """
        logger.info(f"Cleanup loop started (interval: {self._cleanup_check_interval}, "
                    f"timeout: {self._user_idle_timeout})")
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(),
                                       timeout=self._cleanup_check_interval.total_seconds())
                break

            except TimeoutError:
                try:
                    cleaned = await self._cleanup_inactive_users()
                    logger.info(f"Cleanup completed: {cleaned} users removed (remaining: \
                        {len(self._user_workflows)}/{self._max_users})")
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")

        logger.info("Cleanup loop stopped")

    def start_cleanup_loop(self):
        """
        Start the cleanup loop in a background task.
        """
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Cleanup loop started")
        else:
            logger.info("Cleanup loop already running")

    async def shutdown(self):
        """
        Gracefully shut down SessionManager.
        """

        self._shutdown_event.set()

        if self._cleanup_task and not self._cleanup_task.done():
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except TimeoutError:
                logger.warning("Cleanup task did not shut down in time, cancelling")
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

        async with self._users_lock.writer:
            users_to_cleanup = list(self._user_workflows.items())
            self._user_workflows.clear()

        for user_id, user_data in users_to_cleanup:
            try:
                if user_data.builder:
                    await user_data.builder.__aexit__(None, None, None)
                    logger.debug(f"Builder exited for user {self._truncate_user_id(user_id)}")
            except Exception as e:
                logger.error(f"Error cleaning up user {self._truncate_user_id(user_id)}: {e}")

        if self._shared_builder:
            try:
                await self._shared_builder.__aexit__(None, None, None)
                logger.debug("Shared builder exited")
            except Exception as e:
                logger.error("Error cleaning up shared builder: %s", e)
            finally:
                self._shared_builder = None

        logger.info("SessionManager shutdown complete")

    def set_metadata_from_http_request(self, request: Request) -> None:
        """
        Extracts and sets user metadata request attributes from a HTTP request.
        If request is None, no attributes are set.
        """
        self._context.metadata._request.method = getattr(request, "method", None)
        self._context.metadata._request.url_path = request.url.path
        self._context.metadata._request.url_port = request.url.port
        self._context.metadata._request.url_scheme = request.url.scheme
        self._context.metadata._request.headers = request.headers
        self._context.metadata._request.query_params = request.query_params
        self._context.metadata._request.path_params = request.path_params
        self._context.metadata._request.client_host = request.client.host
        self._context.metadata._request.client_port = request.client.port
        self._context.metadata._request.cookies = request.cookies

        if request.headers.get("conversation-id"):
            self._context_state.conversation_id.set(request.headers["conversation-id"])

        if request.headers.get("user-message-id"):
            self._context_state.user_message_id.set(request.headers["user-message-id"])

        # W3C Trace Context header: traceparent: 00-<trace-id>-<span-id>-<flags>
        traceparent = request.headers.get("traceparent")
        if traceparent:
            try:
                parts = traceparent.split("-")
                if len(parts) >= 4:
                    trace_id_hex = parts[1]
                    if len(trace_id_hex) == 32:
                        trace_id_int = uuid.UUID(trace_id_hex).int
                        self._context_state.workflow_trace_id.set(trace_id_int)
            except Exception:
                pass

        if not self._context_state.workflow_trace_id.get():
            workflow_trace_id = request.headers.get("workflow-trace-id")
            if workflow_trace_id:
                try:
                    self._context_state.workflow_trace_id.set(uuid.UUID(workflow_trace_id).int)
                except Exception:
                    pass

        workflow_run_id = request.headers.get("workflow-run-id")
        if workflow_run_id:
            self._context_state.workflow_run_id.set(workflow_run_id)

    def set_metadata_from_websocket(self,
                                    websocket: WebSocket,
                                    user_message_id: str | None,
                                    conversation_id: str | None) -> None:
        """
        Extracts and sets user metadata for WebSocket connections.
        """

        # Extract cookies from WebSocket headers (similar to HTTP request)
        if websocket and hasattr(websocket, 'scope') and 'headers' in websocket.scope:
            cookies = {}
            for header_name, header_value in websocket.scope.get('headers', []):
                if header_name == b'cookie':
                    cookie_header = header_value.decode('utf-8')
                    # Parse cookie header: "name1=value1; name2=value2"
                    for cookie in cookie_header.split(';'):
                        cookie = cookie.strip()
                        if '=' in cookie:
                            name, value = cookie.split('=', 1)
                            cookies[name.strip()] = value.strip()

            # Set cookies in metadata (same as HTTP request)
            self._context.metadata._request.cookies = cookies
            self._context_state.metadata.set(self._context.metadata)

        if conversation_id is not None:
            self._context_state.conversation_id.set(conversation_id)

        if user_message_id is not None:
            self._context_state.user_message_id.set(user_message_id)


# Compatibility aliases with previous releases
AIQSessionManager = SessionManager
