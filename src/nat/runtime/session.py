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
import typing
import uuid
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextlib import nullcontext
from datetime import datetime

from fastapi import WebSocket
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from starlette.requests import HTTPConnection
from starlette.requests import Request

from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.workflow import Workflow
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.config import Config
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import InteractionPrompt
from nat.data_models.runtime_enum import RuntimeTypeEnum

if typing.TYPE_CHECKING:
    from nat.builder.per_user_workflow_builder import PerUserWorkflowBuilder
    from nat.builder.workflow_builder import WorkflowBuilder

logger = logging.getLogger(__name__)


class PerUserBuilderInfo(BaseModel):
    """
    Container for per-user builder data with activity tracking.

    Tracks lifecycle and usage of per-user builders for automatic cleanup.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)

    builder: typing.Any = Field(description="The per-user workflow builder instance")
    workflow: typing.Any = Field(description="The cached per-user workflow instance")
    semaphore: typing.Any = Field(description="Per-user semaphore for concurrency control")
    last_activity: datetime = Field(default_factory=datetime.now,
                                    description="The timestamp of the last access to this builder")
    ref_count: int = Field(default=0, ge=0, description="The reference count of this builder")
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock, description="Lock for thread-safe ref_count updates")


class Session:
    """
    Represents an active session with access to workflow and builders.

    Each session is tied to a specific request, and provides access to the appropriate workflow
    instance (shared or per-user).

    Lifecycle:
    - Created for each request via SessionManager.session()
    - Automatically manages ref_count for per-user builder tracking
    - Cleans up context variables on exit

    Concurrency:
    - Each session has its own semaphore for concurrency control
    - For per-user workflows: each user has an independent concurrency limit
    - For shared workflows: all sessions share the SessionManager's semaphore
    """

    def __init__(self,
                 session_manager: "SessionManager",
                 workflow: Workflow,
                 semaphore: asyncio.Semaphore | nullcontext,
                 user_id: str | None = None):
        self._session_manager = session_manager
        self._workflow = workflow
        self._semaphore = semaphore
        self._user_id = user_id

    @property
    def user_id(self) -> str | None:
        return self._user_id

    @property
    def workflow(self) -> Workflow:
        return self._workflow

    @property
    def session_manager(self) -> "SessionManager":
        return self._session_manager

    @asynccontextmanager
    async def run(self, message, runtime_type: RuntimeTypeEnum = RuntimeTypeEnum.RUN_OR_SERVE):
        """
        Start a workflow run using this session's workflow.

        Args:
            message: Input message for the workflow
            runtime_type: Runtime type (defaults to SessionManager's runtime_type)

        Yields:
            Runner instance for the workflow execution
        """
        async with self._semaphore:
            async with self._workflow.run(message, runtime_type=runtime_type) as runner:
                yield runner


class SessionManager:

    def __init__(self,
                 config: Config,
                 shared_builder: "WorkflowBuilder",
                 entry_function: str | None = None,
                 shared_workflow: Workflow | None = None,
                 max_concurrency: int = 8,
                 runtime_type: RuntimeTypeEnum = RuntimeTypeEnum.RUN_OR_SERVE):
        """
        The SessionManager class is used to manage workflow builders and sessions.
        It manages workflow sessions and per-user builders with lifecycle management.

        Architecture:
        - One SessionManager per FastAPI server
        - Creates/caches PerUserWorkflowBuilder instances per user
        - Cleans up inactive builders based on timeout

        Parameters
        ----------
        config : Config
            The configuration for the workflow
        shared_builder : WorkflowBuilder
            The shared workflow builder
        entry_function : str | None, optional
            The entry function for this SessionManager's workflows, by default None
        shared_workflow : Workflow, optional
            The shared workflow, by default None
        max_concurrency : int, optional
            The maximum number of simultaneous workflow invocations, by default 8
        runtime_type : RuntimeTypeEnum, optional
            The type of runtime the session manager is operating in, by default RuntimeTypeEnum.RUN_OR_SERVE
        """

        from nat.cli.type_registry import GlobalTypeRegistry

        self._config = config
        self._max_concurrency = max_concurrency
        self._entry_function = entry_function

        # Semaphore for limiting concurrency
        if max_concurrency > 0:
            self._semaphore = asyncio.Semaphore(max_concurrency)
        else:
            # If max_concurrency is 0, then we don't need to limit the concurrency but we still need a context
            self._semaphore = nullcontext()

        self._runtime_type = runtime_type

        # Context state for per-request context variables
        self._context_state = ContextState.get()
        self._context = Context(self._context_state)

        # Track if workflow is shared or per-user
        workflow_registration = GlobalTypeRegistry.get().get_function(type(config.workflow))
        self._is_workflow_per_user = workflow_registration.is_per_user

        # Shared components
        self._shared_builder = shared_builder
        self._shared_workflow = shared_workflow

        # Per-user management
        self._per_user_builders: dict[str, PerUserBuilderInfo] = {}
        self._per_user_builders_lock = asyncio.Lock()
        self._per_user_builders_cleanup_task: asyncio.Task | None = None
        self._per_user_session_timeout = config.general.per_user_workflow_timeout
        self._per_user_session_cleanup_interval = config.general.per_user_workflow_cleanup_interval
        self._shutdown_event = asyncio.Event()

        # Cache schemas for per-user workflows
        if self._is_workflow_per_user:
            self._per_user_workflow_input_schema = workflow_registration.per_user_function_input_schema
            self._per_user_workflow_single_output_schema = workflow_registration.per_user_function_single_output_schema
            self._per_user_workflow_streaming_output_schema = \
                                                        workflow_registration.per_user_function_streaming_output_schema
        else:
            self._per_user_workflow_input_schema = None
            self._per_user_workflow_single_output_schema = None
            self._per_user_workflow_streaming_output_schema = None

    @property
    def config(self) -> Config:
        return self._config

    @property
    def workflow(self) -> Workflow:
        """
        Get workflow for backward compatibility.

        Only works for shared workflows. For per-user workflows, use session.workflow.

        Raises:
            ValueError: If workflow is per-user
        """
        if self._is_workflow_per_user:
            raise ValueError("Workflow is per-user. Access workflow through session.workflow instead.")
        if self._shared_workflow is None:
            raise ValueError("No shared workflow available")
        return self._shared_workflow

    @property
    def shared_builder(self) -> "WorkflowBuilder":
        return self._shared_builder

    @property
    def is_workflow_per_user(self) -> bool:
        return self._is_workflow_per_user

    def get_workflow_input_schema(self) -> type[BaseModel]:
        """Get workflow input schema for OpenAPI documentation."""

        if self._is_workflow_per_user:
            return self._per_user_workflow_input_schema

        return self._shared_workflow.input_schema

    def get_workflow_single_output_schema(self) -> type[BaseModel]:
        """Get workflow single output schema for OpenAPI documentation."""

        if self._is_workflow_per_user:
            return self._per_user_workflow_single_output_schema

        return self._shared_workflow.single_output_schema

    def get_workflow_streaming_output_schema(self) -> type[BaseModel]:
        """Get workflow streaming output schema for OpenAPI documentation."""

        if self._is_workflow_per_user:
            return self._per_user_workflow_streaming_output_schema

        return self._shared_workflow.streaming_output_schema

    @classmethod
    async def create(cls,
                     config: Config,
                     shared_builder: "WorkflowBuilder",
                     entry_function: str | None = None,
                     max_concurrency: int = 8,
                     runtime_type: RuntimeTypeEnum = RuntimeTypeEnum.RUN_OR_SERVE) -> "SessionManager":
        """
        Create a SessionManager. This is the preferred way to instantiate.

        Handles async workflow building and starts cleanup task if per-user.
        """
        from nat.cli.type_registry import GlobalTypeRegistry

        workflow_registration = GlobalTypeRegistry.get().get_function(type(config.workflow))

        if workflow_registration.is_per_user:
            shared_workflow = None
            logger.info(f"Workflow is per-user (entry_function={entry_function})")
        else:
            shared_workflow = await shared_builder.build(entry_function=entry_function)
            logger.info(f"Shared workflow built (entry_function={entry_function})")

        session_manager = cls(config=config,
                              shared_builder=shared_builder,
                              entry_function=entry_function,
                              shared_workflow=shared_workflow,
                              max_concurrency=max_concurrency,
                              runtime_type=runtime_type)

        # Start cleanup task for per-user workflows
        if session_manager._is_workflow_per_user:
            session_manager._per_user_builders_cleanup_task = asyncio.create_task(
                session_manager._run_periodic_cleanup())

        return session_manager

    async def _run_periodic_cleanup(self):

        logger.debug("Running periodic cleanup of per-user builders")
        while not self._shutdown_event.is_set():
            try:
                # Wait for either cleanup interval or shutdown
                await asyncio.wait_for(self._shutdown_event.wait(),
                                       timeout=self._per_user_session_cleanup_interval.total_seconds())
                # If we get here, shutdown was signaled
                break
            except TimeoutError:
                # Timeout means it's time to run cleanup
                try:
                    await self._cleanup_inactive_per_user_builders()
                except Exception:
                    logger.exception("Error during periodic cleanup")

        logger.debug("Periodic cleanup task shutting down")

    async def _cleanup_inactive_per_user_builders(self) -> int:

        now = datetime.now()
        threshold = now - self._per_user_session_timeout
        builders_to_cleanup: list[tuple[str, PerUserBuilderInfo]] = []

        # Identify builders to cleanup (under lock)
        async with self._per_user_builders_lock:
            for user_id, builder_info in list(self._per_user_builders.items()):
                if builder_info.ref_count == 0 and builder_info.last_activity < threshold:
                    # Remove from dict and add to cleanup list
                    builders_to_cleanup.append((user_id, builder_info))
                    del self._per_user_builders[user_id]
                    logger.debug(f"Marked per-user builder for user {user_id} for cleanup "
                                 f"(inactive since {builder_info.last_activity.isoformat()})")
        # Cleanup builders (outside lock to avoid blocking)
        for user_id, builder_info in builders_to_cleanup:
            try:
                await builder_info.builder.__aexit__(None, None, None)
                logger.info(f"Cleaned up inactive per-user builder for user={user_id} "
                            f"(remaining users: {len(self._per_user_builders)})")
            except Exception:
                logger.exception(f"Error cleaning up per-user builder for user {user_id}")

        return len(builders_to_cleanup)

    def _get_user_id_from_context(self) -> str | None:
        """
        Get user ID from current context.

        Extraction order:
        1. From context user_id (set from nat-session cookie)
        2. From context user_manager if set
        3. None (for shared workflow or unauthenticated access)

        """
        try:
            # Primary: Get from context user_id (already extracted from nat-session cookie)
            user_id = self._context.user_id
            if user_id:
                return user_id

            # Fallback: Get from user_manager if set
            user_manager = self._context.user_manager
            if user_manager:
                return user_manager.get_id()
            return None
        except Exception as e:
            logger.debug(f"Could not extract user_id from context: {e}")
            return None

    async def _get_or_create_per_user_builder(self, user_id: str) -> tuple["PerUserWorkflowBuilder", Workflow]:
        from nat.builder.per_user_workflow_builder import PerUserWorkflowBuilder

        async with self._per_user_builders_lock:
            if user_id in self._per_user_builders:
                builder_info = self._per_user_builders[user_id]
                builder_info.last_activity = datetime.now()

                return builder_info.builder, builder_info.workflow

            logger.info(f"Creating per-user builder for user={user_id}, entry_function={self._entry_function}")
            builder = PerUserWorkflowBuilder(user_id=user_id, shared_builder=self._shared_builder)
            # Enter the builder's context manually to avoid exiting the context manager
            # Exit the context when cleaning up the builder
            await builder.__aenter__()

            try:
                await builder.populate_builder(self._config)
                workflow = await builder.build(entry_function=self._entry_function)

                # Create per-user semaphore for concurrency control
                if self._max_concurrency > 0:
                    per_user_semaphore = asyncio.Semaphore(self._max_concurrency)
                else:
                    per_user_semaphore = nullcontext()

                builder_info = PerUserBuilderInfo(builder=builder,
                                                  workflow=workflow,
                                                  semaphore=per_user_semaphore,
                                                  last_activity=datetime.now(),
                                                  ref_count=0,
                                                  lock=asyncio.Lock())
                self._per_user_builders[user_id] = builder_info
                logger.info(
                    f"Created per-user builder for user={user_id} (total users: {len(self._per_user_builders)})")
                return builder_info.builder, builder_info.workflow
            except Exception:
                logger.exception(f"Error creating per-user builder for user {user_id}")
                try:
                    await builder.__aexit__(None, None, None)
                except Exception:
                    logger.exception("Error during builder cleanup after failed creation")
                raise

    @asynccontextmanager
    async def session(self,
                      user_id: str | None = None,
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

        if isinstance(http_connection, Request):
            self.set_metadata_from_http_request(http_connection)

        builder_info: PerUserBuilderInfo | None = None

        if self._is_workflow_per_user:
            # Resolve user_id: explicit param > context
            if user_id is None:
                user_id = self._get_user_id_from_context()
            if user_id is None:
                raise ValueError("user_id is required for per-user workflow but could not be determined. "
                                 "Ensure 'nat-session' cookie is set or pass user_id explicitly.")

            # To ensure the user_id is set in the context before the per-user builder is created
            self._context_state.user_id.set(user_id)

            # Get or create per-user builder
            logger.debug(f"Getting or creating per-user builder for user {user_id}")
            _, workflow = await self._get_or_create_per_user_builder(user_id)
            builder_info = self._per_user_builders[user_id]
            async with builder_info.lock:
                builder_info.ref_count += 1
                logger.debug(f"Incremented ref_count for user {user_id} to {builder_info.ref_count}")
            # Use per-user semaphore for concurrency control
            semaphore = builder_info.semaphore
        else:
            workflow = self._shared_workflow
            # Use shared semaphore for concurrency control
            semaphore = self._semaphore

        # TODO: this logic needs to be cleaned up since it is a duplicated setting of the user_id
        # But we need to keep it for now to maintain the token_user_id
        token_user_id = None
        if user_id is not None:
            token_user_id = self._context_state.user_id.set(user_id)

        try:
            session = Session(session_manager=self, user_id=user_id, workflow=workflow, semaphore=semaphore)

            yield session

        finally:
            if builder_info is not None:
                async with builder_info.lock:
                    builder_info.ref_count -= 1
                    builder_info.last_activity = datetime.now()

            if token_user_id is not None:
                self._context_state.user_id.reset(token_user_id)
            if token_user_manager is not None:
                self._context_state.user_manager.reset(token_user_manager)
            if token_user_input is not None:
                self._context_state.user_input_callback.reset(token_user_input)
            if token_user_authentication is not None:
                self._context_state.user_auth_callback.reset(token_user_authentication)

    @asynccontextmanager
    async def run(self, message, runtime_type: RuntimeTypeEnum = RuntimeTypeEnum.RUN_OR_SERVE):
        """
        Start a workflow run
        """
        if self._is_workflow_per_user:
            raise ValueError("Cannot use SessionManager.run() with per-user workflows. "
                             "Use 'async with session_manager.session() as session' then 'session.run()' instead.")
        async with self._semaphore:
            async with self._shared_workflow.run(message, runtime_type=runtime_type) as runner:
                yield runner

    async def shutdown(self) -> None:
        """
        Shutdown the SessionManager and cleanup resources.

        Call this when the SessionManager is no longer needed.
        """
        if self._is_workflow_per_user:
            # Shutdown cleanup task
            self._shutdown_event.set()
            if self._per_user_builders_cleanup_task:
                try:
                    await asyncio.wait_for(self._per_user_builders_cleanup_task, timeout=5.0)
                except TimeoutError:
                    logger.warning("Cleanup task did not finish in time, cancelling")
                    self._per_user_builders_cleanup_task.cancel()

            # Cleanup all per-user builders
            async with self._per_user_builders_lock:
                for user_id, builder_info in list(self._per_user_builders.items()):
                    logger.debug(f"Cleaning up per-user builder for user {user_id}")
                    try:
                        await builder_info.builder.__aexit__(None, None, None)
                    except Exception:
                        logger.exception(f"Error cleaning up builder for user {user_id}")
                self._per_user_builders.clear()

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

        # Set user_id from nat-session cookie
        if request.cookies.get("nat-session"):
            self._context_state.user_id.set(request.cookies["nat-session"])

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

            # Set user_id from nat-session cookie
            if cookies.get("nat-session"):
                self._context_state.user_id.set(cookies["nat-session"])

        if conversation_id is not None:
            self._context_state.conversation_id.set(conversation_id)

        if user_message_id is not None:
            self._context_state.user_message_id.set(user_message_id)


# Compatibility aliases with previous releases
AIQSessionManager = SessionManager
