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
import typing
import uuid
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextlib import nullcontext

from fastapi import WebSocket
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

_T = typing.TypeVar("_T")


class UserManagerBase:
    pass


class SessionManager:

    def __init__(self,
                 workflow: Workflow,
                 max_concurrency: int = 8,
                 runtime_type: RuntimeTypeEnum = RuntimeTypeEnum.RUN_OR_SERVE):
        """
        The SessionManager class is used to run and manage a user workflow session. It runs and manages the context,
        and configuration of a workflow with the specified concurrency.

        Parameters
        ----------
        workflow : Workflow
            The workflow to run
        max_concurrency : int, optional
            The maximum number of simultaneous workflow invocations, by default 8
        runtime_type : RuntimeTypeEnum, optional
            The type of runtime the session manager is operating in, by default RuntimeTypeEnum.RUN_OR_SERVE
        """

        if (workflow is None):
            raise ValueError("Workflow cannot be None")

        self._workflow: Workflow = workflow

        self._max_concurrency = max_concurrency
        self._context_state = ContextState.get()
        self._context = Context(self._context_state)
        self._runtime_type = runtime_type

        # We save the context because Uvicorn spawns a new process
        # for each request, and we need to restore the context vars
        self._saved_context = contextvars.copy_context()

        if (max_concurrency > 0):
            self._semaphore = asyncio.Semaphore(max_concurrency)
        else:
            # If max_concurrency is 0, then we don't need to limit the concurrency but we still need a context
            self._semaphore = nullcontext()

    @property
    def config(self) -> Config:
        return self._workflow.config

    @property
    def workflow(self) -> Workflow:
        return self._workflow

    @property
    def context(self) -> Context:
        return self._context

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
    async def run(self, message, runtime_type: RuntimeTypeEnum = RuntimeTypeEnum.RUN_OR_SERVE):
        """
        Start a workflow run
        """
        async with self._semaphore:
            # Apply the saved context
            for k, v in self._saved_context.items():
                k.set(v)

            async with self._workflow.run(message, runtime_type=runtime_type) as runner:
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
