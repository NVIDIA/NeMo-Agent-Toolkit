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
"""OpenAI v1 chat completions route registration."""

import logging
from typing import Any

from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import Error
from nat.data_models.api_server import ErrorTypes
from nat.data_models.interactive_http import ExecutionStatus
from nat.front_ends.fastapi.response_helpers import generate_single_response
from nat.front_ends.fastapi.response_helpers import generate_streaming_response_as_str
from nat.runtime.session import SessionManager

from .common_utils import RESPONSE_500
from .common_utils import _build_interactive_runner
from .common_utils import add_context_headers_to_response
from .execution import build_accepted_response

logger = logging.getLogger(__name__)


def post_openai_api_compatible_endpoint(*, worker: Any, session_manager: SessionManager, enable_interactive: bool):
    """Build OpenAI Chat Completions compatible POST handler."""

    async def post_openai_api_compatible_interactive(response: Response, request: Request, payload: ChatRequest):
        stream_requested = getattr(payload, "stream", False)

        runner = _build_interactive_runner(worker, session_manager)

        if stream_requested:
            return StreamingResponse(
                headers={"Content-Type": "text/event-stream; charset=utf-8"},
                content=runner.streaming_generator(
                    payload,
                    request,
                    streaming=True,
                    step_adaptor=worker.get_step_adaptor(),
                    result_type=ChatResponseChunk,
                    output_type=ChatResponseChunk,
                ),
            )

        response.headers["Content-Type"] = "application/json"
        try:
            record = await runner.start_non_streaming(
                payload=payload,
                request=request,
                result_type=ChatResponse,
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

        except Exception as e:
            logger.exception("Unhandled interactive workflow error")
            add_context_headers_to_response(response)
            return JSONResponse(
                content=Error(
                    code=ErrorTypes.WORKFLOW_ERROR,
                    message=str(e),
                    details=type(e).__name__,
                ).model_dump(),
                status_code=500,
            )

    async def post_openai_api_compatible(response: Response, request: Request, payload: ChatRequest):
        stream_requested = getattr(payload, "stream", False)

        if stream_requested:
            async with session_manager.session(http_connection=request) as session:
                return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                         content=generate_streaming_response_as_str(
                                             payload,
                                             session=session,
                                             streaming=True,
                                             step_adaptor=worker.get_step_adaptor(),
                                             result_type=ChatResponseChunk,
                                             output_type=ChatResponseChunk))

        response.headers["Content-Type"] = "application/json"
        async with session_manager.session(http_connection=request) as session:
            try:
                result = await generate_single_response(payload, session, result_type=ChatResponse)
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

    return post_openai_api_compatible_interactive if enable_interactive else post_openai_api_compatible


async def add_v1_chat_completions_route(
    worker: Any,
    app: FastAPI,
    *,
    path: str,
    method: str,
    description: str,
    session_manager: SessionManager,
    enable_interactive: bool,
):
    """Register OpenAI v1 chat completions endpoint."""
    extra = ' with interaction support' if enable_interactive else ''
    app.add_api_route(
        path=path,
        endpoint=post_openai_api_compatible_endpoint(worker=worker,
                                                     session_manager=session_manager,
                                                     enable_interactive=enable_interactive),
        methods=[method],
        response_model=ChatResponse | ChatResponseChunk,
        description=f"{description} (OpenAI Chat Completions API compatible{extra})",
        responses={500: RESPONSE_500},
    )
