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
"""Shared FastAPI route helpers for HTTP generate/chat endpoints."""

import logging
from typing import Any

from fastapi import Body
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from nat.builder.context import Context
from nat.data_models.api_server import Error
from nat.data_models.api_server import ErrorTypes
from nat.data_models.interactive_http import ExecutionStatus
from nat.front_ends.fastapi.response_helpers import generate_single_response
from nat.front_ends.fastapi.response_helpers import generate_streaming_response_as_str
from nat.runtime.session import SessionManager

from .execution import build_accepted_response

logger = logging.getLogger(__name__)

RESPONSE_500 = {
    "description": "Internal Server Error",
    "content": {
        "application/json": {
            "example": {
                "detail": "Internal server error occurred"
            }
        }
    },
}


def add_context_headers_to_response(response: Response) -> None:
    """Add context-based headers to response if available."""
    observability_trace_id = Context.get().observability_trace_id
    if observability_trace_id:
        response.headers["Observability-Trace-Id"] = observability_trace_id


def _build_interactive_runner(worker: Any, session_manager: SessionManager):
    from nat.front_ends.fastapi.http_interactive_runner import HTTPInteractiveRunner

    return HTTPInteractiveRunner(
        execution_store=worker._execution_store,
        session_manager=session_manager,
        http_flow_handler=worker._http_flow_handler,
    )


def _with_annotation(handler: Any, param_name: str, annotation: Any):
    annotations = dict(getattr(handler, "__annotations__", {}))
    annotations[param_name] = annotation
    handler.__annotations__ = annotations
    return handler


def get_single_endpoint(*, worker: Any, session_manager: SessionManager, result_type: type | None):
    """Build a single-response GET handler."""
    auth_cb = worker._http_flow_handler.authenticate if worker._http_flow_handler else None

    async def get_single(response: Response, request: Request):
        response.headers["Content-Type"] = "application/json"
        async with session_manager.session(http_connection=request, user_authentication_callback=auth_cb) as session:
            try:
                result = await generate_single_response(None, session, result_type=result_type)
                add_context_headers_to_response(response)
                return result
            except Exception as exc:
                logger.exception("Unhandled workflow error")
                add_context_headers_to_response(response)
                return JSONResponse(
                    content=Error(
                        code=ErrorTypes.WORKFLOW_ERROR,
                        message=str(exc),
                        details=type(exc).__name__,
                    ).model_dump(),
                    status_code=422,
                )

    return get_single


def get_streaming_endpoint(*,
                           worker: Any,
                           session_manager: SessionManager,
                           streaming: bool,
                           result_type: type | None,
                           output_type: type | None):
    """Build a streaming GET handler."""
    auth_cb = worker._http_flow_handler.authenticate if worker._http_flow_handler else None

    async def get_stream(request: Request):
        async with session_manager.session(http_connection=request, user_authentication_callback=auth_cb) as session:
            return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                     content=generate_streaming_response_as_str(None,
                                                                                session=session,
                                                                                streaming=streaming,
                                                                                step_adaptor=worker.get_step_adaptor(),
                                                                                result_type=result_type,
                                                                                output_type=output_type))

    return get_stream


def post_single_endpoint(*,
                         worker: Any,
                         session_manager: SessionManager,
                         request_type: Any,
                         enable_interactive: bool,
                         result_type: type | None):
    """Build a single-response POST handler."""

    async def post_single_interactive(response: Response, request: Request, payload: Any = Body()):
        response.headers["Content-Type"] = "application/json"
        runner = _build_interactive_runner(worker, session_manager)
        try:
            record = await runner.start_non_streaming(
                payload=payload,
                request=request,
                result_type=result_type,
            )
            await record.first_outcome.wait()

            match record.status:
                case ExecutionStatus.COMPLETED:
                    response.status_code = 200
                    add_context_headers_to_response(response)
                    return record.result
                case ExecutionStatus.FAILED:
                    add_context_headers_to_response(response)
                    return JSONResponse(
                        content=Error(
                            code=ErrorTypes.WORKFLOW_ERROR,
                            message=record.error or "Unknown error",
                            details="ExecutionFailed",
                        ).model_dump(),
                        status_code=422,
                    )
                case _:
                    response.status_code = 202
                    return build_accepted_response(record)

        except Exception as exc:
            logger.exception("Unhandled interactive workflow error")
            add_context_headers_to_response(response)
            return JSONResponse(
                content=Error(
                    code=ErrorTypes.WORKFLOW_ERROR,
                    message=str(exc),
                    details=type(exc).__name__,
                ).model_dump(),
                status_code=500,
            )

    async def post_single(response: Response, request: Request, payload: Any = Body()):
        response.headers["Content-Type"] = "application/json"
        auth_cb = worker._http_flow_handler.authenticate if worker._http_flow_handler else None
        async with session_manager.session(http_connection=request, user_authentication_callback=auth_cb) as session:
            try:
                result = await generate_single_response(payload, session, result_type=result_type)
                add_context_headers_to_response(response)
                return result
            except Exception as exc:
                logger.exception("Unhandled workflow error")
                add_context_headers_to_response(response)
                return JSONResponse(
                    content=Error(
                        code=ErrorTypes.WORKFLOW_ERROR,
                        message=str(exc),
                        details=type(exc).__name__,
                    ).model_dump(),
                    status_code=422,
                )

    return _with_annotation(post_single_interactive if enable_interactive else post_single, "payload", request_type)


def post_streaming_endpoint(*,
                            worker: Any,
                            session_manager: SessionManager,
                            request_type: Any,
                            enable_interactive: bool,
                            streaming: bool,
                            result_type: type | None,
                            output_type: type | None):
    """Build a streaming POST handler."""

    async def post_stream_interactive(request: Request, payload: Any = Body()):
        runner = _build_interactive_runner(worker, session_manager)
        return StreamingResponse(
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
            content=runner.streaming_generator(
                payload,
                request,
                streaming=streaming,
                step_adaptor=worker.get_step_adaptor(),
                result_type=result_type,
                output_type=output_type,
            ),
        )

    async def post_stream(request: Request, payload: Any = Body()):
        auth_cb = worker._http_flow_handler.authenticate if worker._http_flow_handler else None
        async with session_manager.session(http_connection=request, user_authentication_callback=auth_cb) as session:
            return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                     content=generate_streaming_response_as_str(payload,
                                                                                session=session,
                                                                                streaming=streaming,
                                                                                step_adaptor=worker.get_step_adaptor(),
                                                                                result_type=result_type,
                                                                                output_type=output_type))

    return _with_annotation(post_stream_interactive if enable_interactive else post_stream, "payload", request_type)
