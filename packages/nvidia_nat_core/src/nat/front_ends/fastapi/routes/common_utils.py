# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

        if auth_cb is None:
            session_context = session_manager.session(http_connection=request)
        else:
            session_context = session_manager.session(http_connection=request, user_authentication_callback=auth_cb)

        async with session_context as session:
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
        if auth_cb is None:
            session_context = session_manager.session(http_connection=request)
        else:
            session_context = session_manager.session(http_connection=request, user_authentication_callback=auth_cb)

        async with session_context as session:
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
    auth_cb = worker._http_flow_handler.authenticate if worker._http_flow_handler else None

    async def post_single_interactive(response: Response, request: Request, payload: Any = Body()):
        runner = _build_interactive_runner(worker, session_manager)
        record = await runner.start_non_streaming(
            payload=payload,
            request=request,
            result_type=result_type,
        )
        await record.first_outcome.wait()

        if record.status == ExecutionStatus.COMPLETED:
            response.status_code = 200
            response.headers["Content-Type"] = "application/json"
            add_context_headers_to_response(response)
            return record.result
        if record.status == ExecutionStatus.FAILED:
            response.headers["Content-Type"] = "application/json"
            add_context_headers_to_response(response)
            return JSONResponse(
                content=Error(
                    code=ErrorTypes.WORKFLOW_ERROR,
                    message=record.error or "Unknown error",
                    details="ExecutionFailed",
                ).model_dump(),
                status_code=422,
            )

        response.status_code = 202
        response.headers["Content-Type"] = "application/json"
        return build_accepted_response(record)

    async def post_single(response: Response, request: Request, payload: Any = Body()):
        response.headers["Content-Type"] = "application/json"
        if auth_cb is None:
            session_context = session_manager.session(http_connection=request)
        else:
            session_context = session_manager.session(http_connection=request, user_authentication_callback=auth_cb)

        async with session_context as session:
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
    auth_cb = worker._http_flow_handler.authenticate if worker._http_flow_handler else None

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
        if auth_cb is None:
            session_context = session_manager.session(http_connection=request)
        else:
            session_context = session_manager.session(http_connection=request, user_authentication_callback=auth_cb)

        async with session_context as session:
            return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                     content=generate_streaming_response_as_str(payload,
                                                                                session=session,
                                                                                streaming=streaming,
                                                                                step_adaptor=worker.get_step_adaptor(),
                                                                                result_type=result_type,
                                                                                output_type=output_type))

    return _with_annotation(post_stream_interactive if enable_interactive else post_stream, "payload", request_type)
