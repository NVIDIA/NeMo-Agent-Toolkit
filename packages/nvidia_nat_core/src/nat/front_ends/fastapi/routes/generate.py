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
"""Generate route registration and handler factories."""

import logging
from enum import StrEnum
from typing import Any

from fastapi import Body
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nat.front_ends.fastapi.response_helpers import generate_streaming_response_atif_as_str
from nat.front_ends.fastapi.response_helpers import generate_streaming_response_full_as_str
from nat.runtime.session import SessionManager

from .async_generation import add_async_generation_routes
from .common_utils import RESPONSE_500
from .common_utils import _build_interactive_runner
from .common_utils import _with_annotation
from .common_utils import get_single_endpoint
from .common_utils import get_streaming_endpoint
from .common_utils import post_single_endpoint
from .common_utils import post_streaming_endpoint

logger = logging.getLogger(__name__)


def get_streaming_raw_endpoint(*,
                               session_manager: SessionManager,
                               streaming: bool,
                               result_type: type | None,
                               output_type: type | None):
    """Build a raw-streaming GET handler."""

    async def get_stream(request: Request, filter_steps: str | None = None):
        async with session_manager.session(http_connection=request) as session:
            return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                     content=generate_streaming_response_full_as_str(None,
                                                                                     session=session,
                                                                                     streaming=streaming,
                                                                                     result_type=result_type,
                                                                                     output_type=output_type,
                                                                                     filter_steps=filter_steps))

    return get_stream


def post_streaming_raw_endpoint(*,
                                worker: Any,
                                session_manager: SessionManager,
                                request_type: Any,
                                enable_interactive: bool,
                                streaming: bool,
                                result_type: type | None,
                                output_type: type | None):
    """Build a raw-streaming POST handler."""

    async def post_stream_interactive(request: Request, payload: Any = Body(), filter_steps: str | None = None):
        runner = _build_interactive_runner(worker, session_manager)
        return StreamingResponse(
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
            content=runner.streaming_generator_raw(
                payload,
                request,
                streaming=streaming,
                result_type=result_type,
                output_type=output_type,
                filter_steps=filter_steps,
            ),
        )

    async def post_stream(request: Request, payload: Any = Body(), filter_steps: str | None = None):
        async with session_manager.session(http_connection=request) as session:
            return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                     content=generate_streaming_response_full_as_str(payload,
                                                                                     session=session,
                                                                                     streaming=streaming,
                                                                                     result_type=result_type,
                                                                                     output_type=output_type,
                                                                                     filter_steps=filter_steps))

    return _with_annotation(post_stream_interactive if enable_interactive else post_stream, "payload", request_type)


def post_streaming_atif_endpoint(*,
                                 worker: Any,
                                 session_manager: SessionManager,
                                 request_type: Any,
                                 enable_interactive: bool,
                                 streaming: bool,
                                 result_type: type | None,
                                 output_type: type | None):
    """Build an experimental POST handler that streams ATIF-formatted steps."""

    async def post_stream(request: Request, payload: Any = Body()):
        async with session_manager.session(http_connection=request) as session:
            return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                     content=generate_streaming_response_atif_as_str(payload,
                                                                                     session=session,
                                                                                     streaming=streaming,
                                                                                     result_type=result_type,
                                                                                     output_type=output_type))

    return _with_annotation(post_stream, "payload", request_type)


class _GenerateEndpointType(StrEnum):
    SINGLE = "single"
    STREAMING = "streaming"
    FULL = "full"
    ATIF = "atif"


class _GenerateEndpointMethod(StrEnum):
    GET = "GET"
    POST = "POST"


def _response_for_endpoint_type(session_manager: SessionManager, endpoint_type: _GenerateEndpointType) -> type | None:
    if endpoint_type == _GenerateEndpointType.SINGLE:
        return session_manager.get_workflow_single_output_schema()
    elif endpoint_type == _GenerateEndpointType.STREAMING:
        return session_manager.get_workflow_streaming_output_schema()
    elif endpoint_type == _GenerateEndpointType.FULL:
        return session_manager.get_workflow_streaming_output_schema()
    elif endpoint_type == _GenerateEndpointType.ATIF:
        return session_manager.get_workflow_streaming_output_schema()
    else:
        return None


async def add_generate_route(
    worker: Any,
    app: FastAPI,
    session_manager: SessionManager,
    *,
    enable_interactive: bool,
    endpoint_path: str,
    endpoint_type: _GenerateEndpointType,
    endpoint_method: _GenerateEndpointMethod,
):
    """Add a generate route for an endpoint."""

    request_type = session_manager.get_workflow_input_schema()
    response_type = _response_for_endpoint_type(session_manager, endpoint_type)
    if isinstance(request_type, type) and issubclass(request_type, BaseModel):
        logger.info("Expecting generate request payloads in the following format: %s", request_type.model_fields)
    else:
        logger.warning("Generate request payloads are not a Pydantic BaseModel, skipping request validation.")

    match endpoint_type:
        case _GenerateEndpointType.SINGLE:
            if endpoint_method == _GenerateEndpointMethod.GET:
                route_handler = get_single_endpoint(worker=worker,
                                                    session_manager=session_manager,
                                                    result_type=response_type)
            else:
                route_handler = post_single_endpoint(worker=worker,
                                                     session_manager=session_manager,
                                                     request_type=request_type,
                                                     enable_interactive=enable_interactive,
                                                     result_type=response_type)
            app.add_api_route(
                path=endpoint_path,
                endpoint=route_handler,
                methods=[endpoint_method],
                response_model=response_type,
                responses={500: RESPONSE_500},
            )
        case _GenerateEndpointType.STREAMING:
            if endpoint_method == _GenerateEndpointMethod.GET:
                route_handler = get_streaming_endpoint(worker=worker,
                                                       session_manager=session_manager,
                                                       streaming=True,
                                                       result_type=response_type,
                                                       output_type=response_type)
            else:
                route_handler = post_streaming_endpoint(worker=worker,
                                                        session_manager=session_manager,
                                                        request_type=request_type,
                                                        enable_interactive=enable_interactive,
                                                        streaming=True,
                                                        result_type=response_type,
                                                        output_type=response_type)
            app.add_api_route(
                path=endpoint_path,
                endpoint=route_handler,
                methods=[endpoint_method],
                response_model=response_type,
                responses={500: RESPONSE_500},
            )
        case _GenerateEndpointType.FULL:
            if endpoint_method == _GenerateEndpointMethod.GET:
                route_handler = get_streaming_raw_endpoint(session_manager=session_manager,
                                                           streaming=True,
                                                           result_type=response_type,
                                                           output_type=response_type)
            else:
                route_handler = post_streaming_raw_endpoint(session_manager=session_manager,
                                                            worker=worker,
                                                            request_type=request_type,
                                                            enable_interactive=enable_interactive,
                                                            streaming=True,
                                                            result_type=response_type,
                                                            output_type=response_type)
            app.add_api_route(
                path=endpoint_path,
                endpoint=route_handler,
                methods=[endpoint_method],
                response_model=response_type,
                responses={500: RESPONSE_500},
                description="Stream raw intermediate steps without any step adaptor translations.\n"
                "Use filter_steps query parameter to filter steps by type (comma-separated list) or"
                " set to 'none' to suppress all intermediate steps.",
            )
        case _GenerateEndpointType.ATIF:
            route_handler = post_streaming_atif_endpoint(session_manager=session_manager,
                                                         worker=worker,
                                                         request_type=request_type,
                                                         enable_interactive=False,
                                                         streaming=True,
                                                         result_type=response_type,
                                                         output_type=response_type)
            app.add_api_route(
                path=endpoint_path,
                endpoint=route_handler,
                methods=[endpoint_method],
                response_model=response_type,
                responses={500: RESPONSE_500},
                description="Stream workflow execution as ATIF "
                "(Agent Trajectory Interchange Format) steps.\n"
                "Each SSE event is either an ATIF step object or a final trajectory summary.\n"
                "This endpoint is currently experimental and may change in future releases.",
            )
        case _:
            raise ValueError(f"Unsupported endpoint type: {endpoint_type}")


async def add_generate_routes(
    worker: Any,
    app: FastAPI,
    endpoint: Any,
    session_manager: SessionManager,
    *,
    enable_interactive: bool = True,
    disable_legacy_routes: bool = False,
):
    request_type = session_manager.get_workflow_input_schema()
    endpoint_method = _GenerateEndpointMethod(endpoint.method)

    if endpoint.path:
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=True,
                                 endpoint_path=endpoint.path,
                                 endpoint_type=_GenerateEndpointType.SINGLE,
                                 endpoint_method=endpoint_method)
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=True,
                                 endpoint_path=f"{endpoint.path}/stream",
                                 endpoint_type=_GenerateEndpointType.STREAMING,
                                 endpoint_method=endpoint_method)
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=True,
                                 endpoint_path=f"{endpoint.path}/full",
                                 endpoint_type=_GenerateEndpointType.FULL,
                                 endpoint_method=endpoint_method)
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=False,
                                 endpoint_path=f"{endpoint.path}/atif",
                                 endpoint_type=_GenerateEndpointType.ATIF,
                                 endpoint_method=endpoint_method)

    if not disable_legacy_routes and endpoint.legacy_path:
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=False,
                                 endpoint_path=endpoint.legacy_path,
                                 endpoint_type=_GenerateEndpointType.SINGLE,
                                 endpoint_method=endpoint_method)
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=False,
                                 endpoint_path=f"{endpoint.legacy_path}/stream",
                                 endpoint_type=_GenerateEndpointType.STREAMING,
                                 endpoint_method=endpoint_method)
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=False,
                                 endpoint_path=f"{endpoint.legacy_path}/full",
                                 endpoint_type=_GenerateEndpointType.FULL,
                                 endpoint_method=endpoint_method)

    await add_async_generation_routes(worker=worker,
                                      app=app,
                                      endpoint=endpoint,
                                      session_manager=session_manager,
                                      generate_body_type=request_type,
                                      response_500=RESPONSE_500,
                                      disable_legacy_routes=disable_legacy_routes)
