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
"""OpenAI-compatible chat route registration."""

from enum import StrEnum
from typing import Any

from fastapi import FastAPI

from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.runtime.session import SessionManager

from .common_utils import RESPONSE_500
from .common_utils import get_single_endpoint
from .common_utils import get_streaming_endpoint
from .common_utils import post_single_endpoint
from .common_utils import post_streaming_endpoint
from .v1_chat_completions import add_v1_chat_completions_route


class _ChatEndpointType(StrEnum):
    SINGLE = "single"
    STREAMING = "streaming"


class _ChatEndpointMethod(StrEnum):
    GET = "GET"
    POST = "POST"


def _add_chat_route(app: FastAPI,
                    worker: Any,
                    endpoint_path: str,
                    session_manager: SessionManager,
                    endpoint_type: _ChatEndpointType,
                    endpoint_method: _ChatEndpointMethod,
                    endpoint_description: str,
                    enable_interactive: bool):

    match endpoint_type:
        case _ChatEndpointType.SINGLE:
            if endpoint_method == _ChatEndpointMethod.GET:
                route_handler = get_single_endpoint(worker=worker,
                                                    session_manager=session_manager,
                                                    result_type=ChatResponse)
            else:
                route_handler = post_single_endpoint(worker=worker,
                                                     session_manager=session_manager,
                                                     request_type=ChatRequest,
                                                     enable_interactive=enable_interactive,
                                                     result_type=ChatResponse)
        case _ChatEndpointType.STREAMING:
            if endpoint_method == _ChatEndpointMethod.GET:
                route_handler = get_streaming_endpoint(worker=worker,
                                                       session_manager=session_manager,
                                                       streaming=True,
                                                       result_type=ChatResponseChunk,
                                                       output_type=ChatResponseChunk)
            else:
                route_handler = post_streaming_endpoint(worker=worker,
                                                        session_manager=session_manager,
                                                        request_type=ChatRequest,
                                                        enable_interactive=enable_interactive,
                                                        streaming=True,
                                                        result_type=ChatResponseChunk,
                                                        output_type=ChatResponseChunk)
        case _:
            raise ValueError(f"Unsupported chat endpoint type: {endpoint_type}")

    app.add_api_route(
        path=endpoint_path,
        endpoint=route_handler,
        methods=[endpoint_method],
        description=endpoint_description,
        responses={500: RESPONSE_500},
    )


async def add_chat_routes(
    worker: Any,
    app: FastAPI,
    endpoint: Any,
    session_manager: SessionManager,
    *,
    enable_interactive_extensions: bool = False,
    disable_legacy_routes: bool = False,
):
    """Add OpenAI-compatible chat routes for an endpoint."""
    endpoint_method = _ChatEndpointMethod(endpoint.method)
    openai_v1_path = endpoint.openai_api_v1_path
    openai_path = endpoint.openai_api_path

    # If OpenAI v1 path overlaps the legacy OpenAI-compatible path,
    # register only the v1 endpoint at that path so stream=True/False
    # is handled by a single route as intended.
    register_openai_path = bool(openai_path) and openai_path != openai_v1_path

    if register_openai_path and openai_path:
        _add_chat_route(app=app,
                        worker=worker,
                        endpoint_path=openai_path,
                        session_manager=session_manager,
                        endpoint_type=_ChatEndpointType.SINGLE,
                        endpoint_method=endpoint_method,
                        endpoint_description=endpoint.description,
                        enable_interactive=True)
        _add_chat_route(app=app,
                        worker=worker,
                        endpoint_path=f"{openai_path}/stream",
                        session_manager=session_manager,
                        endpoint_type=_ChatEndpointType.STREAMING,
                        endpoint_method=endpoint_method,
                        endpoint_description=endpoint.description,
                        enable_interactive=True)

    if not disable_legacy_routes and endpoint.legacy_openai_api_path:
        _add_chat_route(app=app,
                        worker=worker,
                        endpoint_path=endpoint.legacy_openai_api_path,
                        session_manager=session_manager,
                        endpoint_type=_ChatEndpointType.SINGLE,
                        endpoint_method=endpoint_method,
                        endpoint_description=endpoint.description,
                        enable_interactive=False)
        _add_chat_route(app=app,
                        worker=worker,
                        endpoint_path=f"{endpoint.legacy_openai_api_path}/stream",
                        session_manager=session_manager,
                        endpoint_type=_ChatEndpointType.STREAMING,
                        endpoint_method=endpoint_method,
                        endpoint_description=endpoint.description,
                        enable_interactive=False)

    if openai_v1_path:
        if endpoint_method != _ChatEndpointMethod.POST:
            raise ValueError(f"Unsupported method {endpoint.method} for {openai_v1_path}")

        await add_v1_chat_completions_route(worker,
                                            app,
                                            path=openai_v1_path,
                                            method=endpoint.method,
                                            description=endpoint.description,
                                            session_manager=session_manager,
                                            enable_interactive=enable_interactive_extensions)
