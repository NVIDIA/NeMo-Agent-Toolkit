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
#
# HTTP-based Human-in-the-Loop utilities (polling + SSE).
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import InteractionPrompt
from nat.front_ends.fastapi.fastapi_front_end_config import HitlHttpConfig

logger = logging.getLogger(__name__)


class HitlQueue:
    """Simple in-memory HITL queue; swappable for shared stores later."""

    def __init__(self, *, interaction_timeout_seconds: int):
        self._interaction_timeout_seconds = interaction_timeout_seconds
        self._pending_by_session: dict[str, list[InteractionPrompt]] = defaultdict(list)
        self._response_waiters: dict[str, asyncio.Future[HumanResponse]] = {}
        self._subscribers: dict[str, list[asyncio.Queue[InteractionPrompt]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def enqueue(self, session_id: str, prompt: InteractionPrompt) -> asyncio.Future[HumanResponse]:
        async with self._lock:
            self._pending_by_session[session_id].append(prompt)
            fut: asyncio.Future[HumanResponse] = asyncio.get_event_loop().create_future()
            self._response_waiters[prompt.id] = fut
            await self._notify_subscribers(session_id, prompt)
            return fut

    async def respond(self, interaction_id: str, response: HumanResponse) -> None:
        async with self._lock:
            fut = self._response_waiters.get(interaction_id)
            if fut is None:
                raise KeyError(f"Interaction {interaction_id} not found")
            if not fut.done():
                fut.set_result(response)

            # Remove from pending lists
            for prompts in self._pending_by_session.values():
                prompts[:] = [p for p in prompts if p.id != interaction_id]
            self._response_waiters.pop(interaction_id, None)

    async def pending(self, session_id: str) -> list[InteractionPrompt]:
        async with self._lock:
            return list(self._pending_by_session.get(session_id, []))

    async def subscribe(self, session_id: str) -> AsyncGenerator[InteractionPrompt, None]:
        queue: asyncio.Queue[InteractionPrompt] = asyncio.Queue()
        async with self._lock:
            self._subscribers[session_id].append(queue)

        try:
            # Emit currently pending first
            for prompt in await self.pending(session_id):
                yield prompt

            while True:
                prompt = await queue.get()
                yield prompt
        finally:
            async with self._lock:
                subs = self._subscribers.get(session_id, [])
                if queue in subs:
                    subs.remove(queue)

    async def _notify_subscribers(self, session_id: str, prompt: InteractionPrompt) -> None:
        for queue in self._subscribers.get(session_id, []):
            await queue.put(prompt)


class HitlPendingResponse(BaseModel):
    session_id: str
    pending: list[dict[str, Any]]


class HitlRespondAccepted(BaseModel):
    status: str
    session_id: str
    interaction_id: str


class HitlHttpHandler:
    """HITL HTTP surface (polling + SSE) with pluggable queue backend."""

    def __init__(self, config: HitlHttpConfig):
        self._config = config
        self._queue = HitlQueue(interaction_timeout_seconds=config.interaction_timeout_seconds)

    def user_callback(self, session_id: str) -> Callable[[InteractionPrompt], asyncio.Future[HumanResponse]]:

        async def _callback(prompt: InteractionPrompt) -> HumanResponse:
            fut = await self._queue.enqueue(session_id, prompt)
            try:
                return await asyncio.wait_for(fut, timeout=self._config.interaction_timeout_seconds)
            except TimeoutError as exc:
                raise TimeoutError(f"Timed out waiting for human response for interaction {prompt.id}") from exc

        def wrapper(prompt: InteractionPrompt) -> asyncio.Future[HumanResponse]:
            return asyncio.ensure_future(_callback(prompt))

        return wrapper

    def register_routes(self, app: FastAPI) -> None:
        """Register polling and SSE endpoints if enabled."""
        if not (self._config.enable_http or self._config.enable_sse):
            return

        def _require_session(request: Request) -> str:
            return self.resolve_session_id(request)

        if self._config.enable_http:

            # Disable response model generation to allow raw Response (204) or model.
            @app.get("/v1/hitl/pending", response_model=None)
            async def get_pending(request: Request) -> Response | HitlPendingResponse:
                session_id = _require_session(request)
                prompts = await self._queue.pending(session_id)
                if not prompts:
                    return Response(status_code=204)
                return HitlPendingResponse(session_id=session_id, pending=[p.model_dump() for p in prompts])

            @app.post("/v1/hitl/{interaction_id}/respond")
            async def post_response(interaction_id: str, payload: HumanResponse,
                                    request: Request) -> HitlRespondAccepted:
                session_id = _require_session(request)
                try:
                    await self._queue.respond(interaction_id, payload)
                except KeyError:
                    raise HTTPException(status_code=404, detail=f"Interaction {interaction_id} not found")
                return HitlRespondAccepted(status="accepted", session_id=session_id, interaction_id=interaction_id)

        if self._config.enable_sse:

            @app.get("/v1/hitl/stream")
            async def get_stream(request: Request) -> StreamingResponse:
                session_id = _require_session(request)

                async def event_stream() -> AsyncGenerator[str, None]:
                    async for prompt in self._queue.subscribe(session_id):
                        yield f"data: {json.dumps(prompt.model_dump())}\n\n"

                return StreamingResponse(event_stream(), media_type="text/event-stream")

    @staticmethod
    def resolve_session_id(request: Request) -> str:
        """Resolve a session id from query/header/cookie, or generate one."""
        session_id = request.query_params.get("session_id") or request.cookies.get("nat-session")
        if not session_id:
            session_id = request.headers.get("X-Session-Id")
        if not session_id:
            session_id = str(uuid.uuid4())
        return session_id


__all__ = ["HitlHttpHandler", "HitlQueue"]
