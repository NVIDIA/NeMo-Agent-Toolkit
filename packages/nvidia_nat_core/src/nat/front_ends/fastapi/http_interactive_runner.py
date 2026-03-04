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
"""
HTTP interactive execution runner.

Runs a workflow in a background task with HITL and OAuth callbacks that
coordinate with the :class:`ExecutionStore` so HTTP clients can interact
via polling and dedicated endpoints.
"""

import asyncio
import logging
import typing
from collections.abc import AsyncGenerator
from collections.abc import Callable

from nat.data_models.api_server import Error
from nat.data_models.api_server import ErrorTypes
from nat.data_models.api_server import ResponseSerializable
from nat.data_models.interactive import HumanPromptNotification
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import HumanResponseNotification
from nat.data_models.interactive import InteractionPrompt
from nat.data_models.interactive_http import StreamInteractionEvent
from nat.front_ends.fastapi.execution_store import ExecutionRecord
from nat.front_ends.fastapi.execution_store import ExecutionStore
from nat.front_ends.fastapi.response_helpers import generate_single_response
from nat.front_ends.fastapi.response_helpers import generate_streaming_response
from nat.front_ends.fastapi.response_helpers import generate_streaming_response_full_as_str
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.runtime.session import SessionManager

if typing.TYPE_CHECKING:
    from fastapi import Request

    from nat.front_ends.fastapi.auth_flow_handlers.http_flow_handler import HTTPAuthenticationFlowHandler

logger = logging.getLogger(__name__)

_HITL_TIMEOUT_GRACE_PERIOD_SECONDS: int = 5


class HTTPInteractiveRunner:
    """
    Coordinates running a workflow with HTTP-based HITL and OAuth.

    For **non-streaming** (single-response) endpoints:
      1. Call :meth:`start_non_streaming`.
      2. Await ``record.first_outcome`` – if the workflow finishes first, return
         200 with the result; if it needs interaction / OAuth, return 202.
      3. Client polls ``GET /executions/{id}`` and submits responses.

    For **streaming** endpoints:
      1. Call :meth:`streaming_generator` which yields SSE chunks.
      2. When the workflow needs HITL / OAuth, a special event is yielded, and
         the generator blocks until the client responds, then continues
         streaming.
    """

    def __init__(
        self,
        execution_store: ExecutionStore,
        session_manager: SessionManager,
        http_flow_handler: "HTTPAuthenticationFlowHandler",
    ) -> None:
        self._store = execution_store
        self._session_manager = session_manager
        self._http_flow_handler = http_flow_handler

    # ------------------------------------------------------------------
    # HITL callback (used as ``user_input_callback``)
    # ------------------------------------------------------------------

    def _build_hitl_callback(
        self,
        record: ExecutionRecord,
        *,
        stream_queue: asyncio.Queue[ResponseSerializable | None] | None = None,
    ):
        """
        Return an ``async def callback(prompt: InteractionPrompt) -> HumanResponse``
        suitable for ``session(..., user_input_callback=callback)``.

        When *stream_queue* is provided (streaming mode), the callback also
        pushes a :class:`StreamInteractionEvent` onto the queue so the SSE
        generator can emit it.
        """
        store = self._store

        async def _hitl_callback(prompt: InteractionPrompt) -> HumanResponse:
            # Notifications are fire-and-forget
            if isinstance(prompt.content, HumanPromptNotification):
                return HumanResponseNotification()

            interaction_id = prompt.id
            pending = await store.set_interaction_required(
                execution_id=record.execution_id,
                prompt=prompt.content,
                interaction_id=interaction_id,
            )

            response_url = f"/executions/{record.execution_id}/interactions/{interaction_id}/response"

            # In streaming mode, push an event onto the queue for SSE
            if stream_queue is not None:
                event = StreamInteractionEvent(
                    execution_id=record.execution_id,
                    interaction_id=interaction_id,
                    prompt=prompt.content,
                    response_url=response_url,
                )
                await stream_queue.put(event)

            # Block until client responds
            backend_timeout: float | None = (prompt.content.timeout + _HITL_TIMEOUT_GRACE_PERIOD_SECONDS
                                             if prompt.content.timeout is not None else None)
            try:
                human_response: HumanResponse = await asyncio.wait_for(
                    pending.future,
                    timeout=backend_timeout,
                )
            except TimeoutError:
                raise TimeoutError(
                    f"HITL prompt timed out after {prompt.content.timeout}s waiting for human response") from None

            # Transition back to running
            await store.set_running(record.execution_id)
            return human_response

        return _hitl_callback

    # ------------------------------------------------------------------
    # OAuth callback builder (wraps the flow handler)
    # ------------------------------------------------------------------

    def _build_auth_callback(
        self,
        record: ExecutionRecord,
        *,
        stream_queue: asyncio.Queue[ResponseSerializable | None] | None = None,
    ):
        """
        Return a wrapper around the HTTP flow handler's ``authenticate``
        that publishes ``oauth_required`` to the execution store (and
        optionally to the stream queue) **before** blocking on the flow
        state future.
        """
        store = self._store
        flow_handler = self._http_flow_handler

        async def _auth_callback(config, method):
            # Delegate to the flow handler which will:
            #  1. Call store.set_oauth_required (via its notification_cb)
            #  2. Push a StreamOAuthEvent onto stream_queue if provided
            #  3. Await the flow state future
            flow_handler.set_execution_context(
                execution_id=record.execution_id,
                store=store,
                stream_queue=stream_queue,
            )
            return await flow_handler.authenticate(config, method)

        return _auth_callback

    # ------------------------------------------------------------------
    # Non-streaming: run workflow as background task
    # ------------------------------------------------------------------

    async def start_non_streaming(
        self,
        payload: typing.Any,
        request: "Request",
        result_type: type | None = None,
    ) -> ExecutionRecord:
        """
        Create an execution record, start the workflow as a background task,
        and return the record immediately.

        The caller should ``await record.first_outcome.wait()`` to know when
        to return 200 (workflow done) or 202 (interaction / OAuth needed).
        """
        record = await self._store.create_execution()

        hitl_cb = self._build_hitl_callback(record)
        auth_cb = self._build_auth_callback(record)

        async def _run():
            try:
                async with self._session_manager.session(
                        http_connection=request,
                        user_input_callback=hitl_cb,
                        user_authentication_callback=auth_cb,
                ) as session:
                    result = await generate_single_response(payload, session, result_type=result_type)
                    await self._store.set_completed(record.execution_id, result)
            except Exception as exc:
                logger.exception("Interactive execution %s failed", record.execution_id)
                await self._store.set_failed(record.execution_id, str(exc))

        record.task = asyncio.create_task(_run())
        return record

    # ------------------------------------------------------------------
    # Streaming: yield SSE chunks with interaction / OAuth events
    # ------------------------------------------------------------------

    async def _streaming_generator_impl(
        self,
        request: "Request",
        *,
        workflow_gen_factory: Callable[[typing.Any], AsyncGenerator[typing.Any]],
        error_log_message: str,
        passthrough_str_items: bool = False,
    ) -> AsyncGenerator[str]:
        """Shared streaming orchestration for interactive HTTP endpoints."""
        record = await self._store.create_execution()

        # Queue used by the HITL / OAuth callbacks to inject events
        # into the stream. Auth can be required during session acquisition
        # (e.g. per-user builder / MCP), so we must consume the queue in the
        # main loop while session acquisition runs in a task.
        stream_queue: asyncio.Queue[typing.Any | None] = asyncio.Queue()

        hitl_cb = self._build_hitl_callback(record, stream_queue=stream_queue)
        auth_cb = self._build_auth_callback(record, stream_queue=stream_queue)

        async def _acquire_session_and_push_workflow() -> None:
            try:
                async with self._session_manager.session(
                        http_connection=request,
                        user_input_callback=hitl_cb,
                        user_authentication_callback=auth_cb,
                ) as session:
                    workflow_gen = workflow_gen_factory(session)
                    try:
                        async for item in workflow_gen:
                            await stream_queue.put(item)
                    except Exception as exc:
                        await stream_queue.put(
                            Error(
                                code=ErrorTypes.WORKFLOW_ERROR,
                                message=str(exc),
                                details=type(exc).__name__,
                            ))
            except Exception as exc:
                logger.exception(error_log_message)
                await stream_queue.put(
                    Error(
                        code=ErrorTypes.WORKFLOW_ERROR,
                        message=str(exc),
                        details=type(exc).__name__,
                    ))
            finally:
                await stream_queue.put(None)

        task = asyncio.create_task(_acquire_session_and_push_workflow())

        try:
            while True:
                item = await stream_queue.get()
                if item is None:
                    break
                if isinstance(item, ResponseSerializable):
                    yield item.get_stream_data()
                elif isinstance(item, Error):
                    yield f"event: error\ndata: {item.model_dump_json()}\n\n"
                    break
                elif isinstance(item, str):
                    if passthrough_str_items:
                        yield item
                    else:
                        yield f"data: {item}\n\n"
                else:
                    yield f"data: {item}\n\n"
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def streaming_generator(
        self,
        payload: typing.Any,
        request: "Request",
        *,
        streaming: bool,
        step_adaptor: StepAdaptor,
        result_type: type | None = None,
        output_type: type | None = None,
    ) -> AsyncGenerator[str]:
        """
        Async generator that yields SSE ``data:`` / ``event:`` lines.

        When the workflow pauses for interaction or OAuth, this generator
        emits a special event and then *blocks* until the client responds
        (the HTTP connection stays open).
        """
        async for chunk in self._streaming_generator_impl(
                request,
                workflow_gen_factory=lambda session: generate_streaming_response(
                    payload,
                    session=session,
                    streaming=streaming,
                    step_adaptor=step_adaptor,
                    result_type=result_type,
                    output_type=output_type, ),
                error_log_message="Interactive streaming execution failed",
                passthrough_str_items=False):
            yield chunk

    async def streaming_generator_raw(
        self,
        payload: typing.Any,
        request: "Request",
        *,
        streaming: bool,
        result_type: type | None = None,
        output_type: type | None = None,
        filter_steps: str | None = None,
    ) -> AsyncGenerator[str]:
        """
        Async generator that yields raw SSE chunks for ``/full`` style streaming.

        This uses ``generate_streaming_response_full_as_str`` so intermediate
        steps are emitted without step-adaptor translations.
        """
        async for chunk in self._streaming_generator_impl(
                request,
                workflow_gen_factory=lambda session: generate_streaming_response_full_as_str(
                    payload,
                    session=session,
                    streaming=streaming,
                    result_type=result_type,
                    output_type=output_type,
                    filter_steps=filter_steps, ),
                error_log_message="Interactive raw streaming execution failed",
                passthrough_str_items=True):
            yield chunk
