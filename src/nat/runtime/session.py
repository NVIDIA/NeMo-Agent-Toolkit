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
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextlib import nullcontext

from starlette.requests import HTTPConnection

from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.workflow import Workflow
from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.data_models.config import Config
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import InteractionPrompt

_T = typing.TypeVar("_T")


class UserManagerBase:
    pass


class SessionManager:

    def __init__(self, workflow: Workflow, max_concurrency: int = 8):
        """
        The AIQSessionManager class is used to run and manage a user workflow session. It runs and manages the context,
        and configuration of a workflow with the specified concurrency.

        Parameters
        ----------
        workflow : Workflow
            The workflow to run
        max_concurrency : int, optional
            The maximum number of simultaneous workflow invocations, by default 8
        """

        if (workflow is None):
            raise ValueError("Workflow cannot be None")

        self._workflow: Workflow = workflow

        self._max_concurrency = max_concurrency
        self._context_state = ContextState.get()
        self._context = Context(self._context_state)

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
                      request: HTTPConnection | None = None,
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

        if conversation_id is not None and request is None:
            self._context_state.conversation_id.set(conversation_id)

        self.set_metadata_from_http_request(request)

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

    def set_metadata_from_http_request(self, request: HTTPConnection | None) -> None:
        """
        Extracts and sets user metadata request attributes from a HTTP request.
        If request is None, no attributes are set.
        """
        if request is None:
            return

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


# Compatibility aliases with previous releases
AIQSessionManager = SessionManager
