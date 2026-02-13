# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat route registration."""

from typing import Any

from fastapi import FastAPI

from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import ResponseIntermediateStep
from nat.front_ends.fastapi.routes.generate import RESPONSE_500
from nat.front_ends.fastapi.routes.generate import get_single_endpoint
from nat.front_ends.fastapi.routes.generate import get_streaming_endpoint
from nat.front_ends.fastapi.routes.generate import post_single_endpoint
from nat.front_ends.fastapi.routes.generate import post_streaming_endpoint
from nat.front_ends.fastapi.routes.v1_chat_completions import add_v1_chat_completions_route
from nat.runtime.session import SessionManager


class _ChatEndpointType(StrEnum):
    SINGLE = "single"
    STREAMING = "streaming"


class _ChatEndpointMethod(StrEnum):
    GET = "GET"
    POST = "POST"


def _add_chat_route(app: FastAPI,
                    endpoint_path: str,
                    session_manager: SessionManager,
                    endpoint_type: _ChatEndpointType,
                    endpoint_method: _ChatEndpointMethod
                    endpoint_description: str):

    method = get_single_endpoint if endpoint_method == _ChatEndpointMethod.GET else post_single_endpoint
    streaming_method = get_streaming_endpoint if endpoint_method == _ChatEndpointMethod.GET else post_streaming_endpoint

    match endpoint_type:
        case _ChatEndpointType.SINGLE:
            app.add_api_route(
                path=endpoint_path,
                endpoint=method(worker=worker, session_manager=session_manager, result_type=ChatResponse),
                description=endpoint_description,
                responses={500: RESPONSE_500},
            )
        case _ChatEndpointType.STREAMING:
            app.add_api_route(
                path=f"{endpoint_path}/stream",
                endpoint=streaming_method(worker=worker,
                                                session_manager=session_manager,
                                                streaming=True,
                                                result_type=ChatResponseChunk,
                                                output_type=ChatResponseChunk),
                description=endpoint_description,
                responses={500: RESPONSE_500},
            )


async def add_chat_routes(
    worker: Any,
    app: FastAPI,
    endpoint: Any,
    session_manager: SessionManager,
    *,
):
    """Add OpenAI-compatible chat routes for an endpoint."""
    if endpoint.openai_api_path:
        if endpoint.method == "POST":
            raise ValueError(f"Unsupported method {endpoint.method} for {endpoint.openai_api_path}")
        _add_chat_route(app=app,
                        endpoint_path=endpoint.openai_api_path,
                        session_manager=session_manager,
                        endpoint_type=_ChatEndpointType.SINGLE,
                        endpoint_method=endpoint.method,
                        endpoint_description=endpoint.description)
        _add_chat_route(app=app,
                        endpoint_path=f"{endpoint.openai_api_path}/stream",
                        session_manager=session_manager,
                        endpoint_type=_ChatEndpointType.STREAMING,
                        endpoint_method=endpoint.method,
                        endpoint_description=endpoint.description)

    if endpoint.openai_api_v1_path:
        if endpoint.method == "GET":
            raise ValueError(f"Unsupported method {endpoint.method} for {endpoint.openai_api_v1_path}")
        
        await add_v1_chat_completions_route(worker,
                                            app,
                                            path=openai_v1_path,
                                            method=endpoint.method,
                                            description=endpoint.description,
                                            session_manager=session_manager,
                                            enable_interactive=enable_interactive)
