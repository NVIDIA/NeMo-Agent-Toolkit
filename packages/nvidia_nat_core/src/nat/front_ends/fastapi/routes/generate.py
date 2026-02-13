# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate route registration and handler factories."""

import logging
from typing import Any

from fastapi import Body
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nat.builder.context import Context
from nat.data_models.api_server import Error
from nat.data_models.api_server import ErrorTypes
from nat.data_models.interactive_http import ExecutionStatus
from nat.front_ends.fastapi.routes.async_generation import add_async_generation_routes
from nat.front_ends.fastapi.routes.execution import build_accepted_response
from nat.front_ends.fastapi.response_helpers import generate_single_response
from nat.front_ends.fastapi.response_helpers import generate_streaming_response_as_str
from nat.front_ends.fastapi.response_helpers import generate_streaming_response_full_as_str
from nat.runtime.session import SessionManager

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


def get_single_endpoint(*,
                        worker: Any,
                        session_manager: SessionManager,
                        result_type: type | None):
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
            except Exception as e:
                logger.exception("Unhandled workflow error")
                add_context_headers_to_response(response)
                return JSONResponse(
                    content=Error(
                        code=ErrorTypes.WORKFLOW_ERROR,
                        message=str(e),
                        details=type(e).__name__,
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


def get_streaming_raw_endpoint(*,
                               worker: Any | None = None,
                               session_manager: SessionManager,
                               streaming: bool,
                               result_type: type | None,
                               output_type: type | None):
    """Build a raw-streaming GET handler."""

    async def get_stream(filter_steps: str | None = None):
        async with session_manager.session(http_connection=None) as session:
            return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                     content=generate_streaming_response_full_as_str(None,
                                                                                     session=session,
                                                                                     streaming=streaming,
                                                                                     result_type=result_type,
                                                                                     output_type=output_type,
                                                                                     filter_steps=filter_steps))

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
            except Exception as e:
                logger.exception("Unhandled workflow error")
                add_context_headers_to_response(response)
                return JSONResponse(
                    content=Error(
                        code=ErrorTypes.WORKFLOW_ERROR,
                        message=str(e),
                        details=type(e).__name__,
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
        async with session_manager.session(http_connection=None) as session:
            return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                     content=generate_streaming_response_full_as_str(payload,
                                                                                     session=session,
                                                                                     streaming=streaming,
                                                                                     result_type=result_type,
                                                                                     output_type=output_type,
                                                                                     filter_steps=filter_steps))

    return _with_annotation(post_stream_interactive if enable_interactive else post_stream, "payload", request_type)


class _GenerateEndpointType(StrEnum):
    SINGLE = "single"
    STREAMING = "streaming"
    FULL = "full"


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

    single_endpoint = get_single_endpoint if endpoint_method == _GenerateEndpointMethod.GET else post_single_endpoint
    streaming_endpoint = get_streaming_endpoint if endpoint_method == _GenerateEndpointMethod.GET else post_streaming_endpoint
    full_endpoint = get_streaming_raw_endpoint if endpoint_method == _GenerateEndpointMethod.GET else post_streaming_raw_endpoint

    match endpoint_type:
        case _GenerateEndpointType.SINGLE:
            app.add_api_route(
                path=endpoint_path,
                endpoint=single_endpoint(worker=worker,
                                         session_manager=session_manager,
                                         request_type=request_type,
                                         enable_interactive=enable_interactive,
                                         result_type=response_type),
                methods=[endpoint_method],
                response_model=response_type,
                responses={500: RESPONSE_500},
            )
        case _GenerateEndpointType.STREAMING:
            app.add_api_route(
                path=endpoint_path,
                endpoint=streaming_endpoint(worker=worker,
                                            session_manager=session_manager,
                                            request_type=request_type,
                                            enable_interactive=enable_interactive,
                                            streaming=True,
                                            result_type=response_type,
                                            output_type=response_type),
                methods=[endpoint_method],
                response_model=response_type,
                responses={500: RESPONSE_500},
            )
        case _GenerateEndpointType.FULL:
            app.add_api_route(
                path=endpoint_path,
                endpoint=full_endpoint(session_manager=session_manager,
                                       worker=worker,
                                       request_type=request_type,
                                       enable_interactive=enable_interactive,
                                       streaming=True,
                                       result_type=response_type,
                                       output_type=response_type),
                methods=[endpoint_method],
                response_model=response_type,
                responses={500: RESPONSE_500},
                description="Stream raw intermediate steps without any step adaptor translations.\n"
                "Use filter_steps query parameter to filter steps by type (comma-separated list) or"
                " set to 'none' to suppress all intermediate steps.",
            )
        case _:
            raise ValueError(f"Unsupported endpoint type: {endpoint_type}")


async def add_generate_routes(
    worker: Any,
    app: FastAPI,
    endpoint: Any,
    session_manager: SessionManager,
    *,
    disable_legacy_routes: bool = False,
):
    if endpoint.path:
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=True,
                                 endpoint_path=endpoint.path,
                                 endpoint_type=_GenerateEndpointType.SINGLE,
                                 endpoint_method=endpoint.method)
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=True,
                                 endpoint_path=f"{endpoint.path}/stream",
                                 endpoint_type=_GenerateEndpointType.STREAMING,
                                 endpoint_method=endpoint.method)
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=True,
                                 endpoint_path=f"{endpoint.path}/full",
                                 endpoint_type=_GenerateEndpointType.FULL,
                                 endpoint_method=endpoint.method)

    if not disable_legacy_routes and endpoint.legacy_path:
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=False,
                                 endpoint_path=endpoint.legacy_path,
                                 endpoint_type=_GenerateEndpointType.SINGLE,
                                 endpoint_method=endpoint.method)
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=False,
                                 endpoint_path=f"{endpoint.legacy_path}/stream",
                                 endpoint_type=_GenerateEndpointType.STREAMING,
                                 endpoint_method=endpoint.method)
        await add_generate_route(worker=worker,
                                 app=app,
                                 session_manager=session_manager,
                                 enable_interactive=False,
                                 endpoint_path=f"{endpoint.legacy_path}/full",
                                 endpoint_type=_GenerateEndpointType.FULL,
                                 endpoint_method=endpoint.method)

    await add_async_generation_routes(worker=worker,
                                      app=app,
                                      endpoint=endpoint,
                                      session_manager=session_manager,
                                      generate_body_type=request_type,
                                      response_500=RESPONSE_500)
