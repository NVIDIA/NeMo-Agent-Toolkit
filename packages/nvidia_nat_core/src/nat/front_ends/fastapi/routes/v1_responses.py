# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""OpenAI v1 Responses API route registration."""

import datetime
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import Error
from nat.data_models.api_server import ErrorTypes
from nat.data_models.api_server import ResponsesRequest
from nat.data_models.api_server import Usage
from nat.front_ends.fastapi.response_helpers import generate_single_response
from nat.front_ends.fastapi.response_helpers import generate_streaming_response
from nat.runtime.session import SessionManager

from .common_utils import RESPONSE_500
from .common_utils import add_context_headers_to_response

logger = logging.getLogger(__name__)


def _responses_usage(usage: Usage | None) -> dict[str, int | None]:
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _output_message(text: str, response_id: str | None = None) -> dict[str, Any]:
    return {
        "id": response_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": [],
        }],
    }


def _responses_response(response: ChatResponse, request_model: str | None) -> dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex}"
    text = ""
    if response.choices and response.choices[0].message:
        text = response.choices[0].message.content or ""

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(response.created.timestamp()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": request_model or response.model,
        "output": [_output_message(text)],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": None,
        "store": True,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": _responses_usage(response.usage),
        "user": None,
        "metadata": {},
    }


def _event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def _responses_stream(payload: ResponsesRequest,
                            request: Request,
                            worker: Any,
                            session_manager: SessionManager) -> AsyncGenerator[str]:
    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    content_index = 0
    output_index = 0
    created_at = int(datetime.datetime.now(datetime.UTC).timestamp())
    chat_request = payload.to_chat_request()

    response_base: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "model": payload.model or "unknown-model",
        "output": [],
    }
    output_item = _output_message("", item_id)
    content_part = {"type": "output_text", "text": "", "annotations": []}

    yield _event("response.created", {"type": "response.created", "response": response_base})
    yield _event("response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": output_item,
    })
    yield _event("response.content_part.added", {
        "type": "response.content_part.added",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": content_part,
    })

    text_parts: list[str] = []
    async with session_manager.session(http_connection=request) as session:
        async for item in generate_streaming_response(chat_request,
                                                      session=session,
                                                      streaming=True,
                                                      step_adaptor=worker.get_step_adaptor(),
                                                      result_type=ChatResponseChunk,
                                                      output_type=ChatResponseChunk):
            if not isinstance(item, ChatResponseChunk):
                continue
            if not item.choices:
                continue
            delta = item.choices[0].delta.content or ""
            if not delta:
                continue
            text_parts.append(delta)
            yield _event("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "delta": delta,
            })

    text = "".join(text_parts)
    content_part["text"] = text
    output_item["content"] = [content_part]
    done_response = {**response_base, "status": "completed", "output": [output_item]}

    yield _event("response.output_text.done", {
        "type": "response.output_text.done",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "text": text,
    })
    yield _event("response.content_part.done", {
        "type": "response.content_part.done",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": content_part,
    })
    yield _event("response.output_item.done", {
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": output_item,
    })
    yield _event("response.done", {"type": "response.done", "response": done_response})


def post_responses_api_endpoint(*, worker: Any, session_manager: SessionManager):
    """Build OpenAI Responses API compatible POST handler."""

    async def post_responses_api(response: Response, request: Request, payload: ResponsesRequest):
        if payload.stream:
            return StreamingResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"},
                                     content=_responses_stream(payload, request, worker, session_manager))

        response.headers["Content-Type"] = "application/json"
        async with session_manager.session(http_connection=request) as session:
            try:
                result = await generate_single_response(payload.to_chat_request(), session, result_type=ChatResponse)
                add_context_headers_to_response(response)
                return _responses_response(result, payload.model)
            except Exception as exc:
                logger.exception("Unhandled Responses API workflow error")
                add_context_headers_to_response(response)
                return JSONResponse(
                    content=Error(
                        code=ErrorTypes.WORKFLOW_ERROR,
                        message=str(exc),
                        details=type(exc).__name__,
                    ).model_dump(),
                    status_code=422,
                )

    return post_responses_api


async def add_v1_responses_route(
    worker: Any,
    app: FastAPI,
    *,
    path: str,
    method: str,
    description: str,
    session_manager: SessionManager,
):
    """Register OpenAI v1 Responses endpoint."""
    app.add_api_route(
        path=path,
        endpoint=post_responses_api_endpoint(worker=worker, session_manager=session_manager),
        methods=[method],
        description=f"{description} (OpenAI Responses API compatible)",
        responses={500: RESPONSE_500},
    )
