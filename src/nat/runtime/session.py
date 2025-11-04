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
import contextvars
import logging
import re
import uuid
from collections.abc import AsyncGenerator
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
from nat.builder.workflow_builder import SharedWorkflowBuilder
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.config import Config
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import InteractionPrompt
from nat.data_models.user_workflow_data import UserWorkflowData
from nat.runtime.runner import Runner

logger = logging.getLogger(__name__)


class UserSession:

    def __init__(self, workflow: Workflow, max_concurrency: int = 8):
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

    def __init__(self,
                 config: Config,
                 max_concurrency: int = 8,
                 max_users: int = 100,
                 user_idle_timeout: timedelta = timedelta(minutes=30),
                 cleanup_check_interval: timedelta = timedelta(minutes=10),
                 require_user_id: bool = False):
        """
        The SessionManager class is for per-user workflow management.

        Accepts config and builds workflows on-demand for each user.

        Parameters
        ----------
        config : Config
            The configuration for building the workflow
        max_concurrency : int, optional
            The maximum number of simultaneous workflow invocations, by default 8
        max_users : int, optional
            The maximum number of users to support, by default 100
        user_idle_timeout : timedelta, optional
            The timeout for user inactivity, by default 30 minutes
        cleanup_check_interval : timedelta, optional
            The interval for checking for idle users, by default 10 minutes
        require_user_id : bool, optional
            Whether to require a user_id in the request, by default False
        """

        self._config = config
        self._max_concurrency = max_concurrency
        self._max_users = max_users
        self._user_idle_timeout = user_idle_timeout
        self._cleanup_check_interval = cleanup_check_interval
        self._require_user_id = require_user_id

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

    def _validate_user_id(self, user_id: str) -> str | None:
        """
        Validates user_id format.
        TODO: Implement actual validation logic.
        """
        if not user_id or not isinstance(user_id, str):
            return None

        user_id = user_id.strip()

        # Check if empty after trimming
        if len(user_id) == 0:
            return None

        # Check reasonable length (prevent memory issues)
        if len(user_id) > 256:
            logger.warning(f"user_id too long: {len(user_id)} characters")
            return None

    def _extract_user_id(self, http_connection: HTTPConnection | None = None) -> str | None:
        """
        Extracts the user ID from the HTTP connection.
        """
        if http_connection is None:
            return None

        # Extract from nat-session cookie (NAT's standard approach)
        if hasattr(http_connection, 'cookies'):
            user_id = http_connection.cookies.get('nat-session')
            if user_id:
                validated = self._validate_user_id(user_id)
                if validated:
                    logger.debug(f"Extracted user_id from nat-session cookie: {validated[:8]}...")
                    return validated

        # For WebSocket connections, extract from scope/headers
        if isinstance(http_connection, WebSocket):
            if hasattr(http_connection, 'scope') and 'headers' in http_connection.scope:
                for header_name, header_value in http_connection.scope.get('headers', []):
                    if header_name == b'cookie':
                        cookie_header = header_value.decode('utf-8')
                        # Parse cookie header for nat-session
                        for cookie in cookie_header.split(';'):
                            cookie = cookie.strip()
                            if cookie.startswith('nat-session='):
                                user_id = cookie.split('=', 1)[1]
                                validated = self._validate_user_id(user_id)
                                if validated:
                                    logger.debug(f"Extracted user_id from WebSocket cookie: {validated[:8]}...")
                                    return validated

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

        if isinstance(http_connection, Request):
            self.set_metadata_from_http_request(http_connection)

        try:
            yield self
        finally:
            if token_user_manager is not None:
                self._context_state.user_manager.reset(token_user_manager)
            if token_user_input is not None:
                self._context_state.user_input_callback.reset(token_user_input)
            if token_user_authentication is not None:
                self._context_state.user_auth_callback.reset(token_user_authentication)

    @asynccontextmanager
    async def run(self, message):
        """
        Start a workflow run
        """
        async with self._semaphore:
            # Apply the saved context
            for k, v in self._saved_context.items():
                k.set(v)

            async with self._workflow.run(message) as runner:
                yield runner

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
