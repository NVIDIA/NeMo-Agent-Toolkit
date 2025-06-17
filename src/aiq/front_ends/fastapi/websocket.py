# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import logging
import typing
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from fastapi import WebSocket
from fastapi import WebSocketException
from starlette.endpoints import WebSocketEndpoint
from starlette.websockets import WebSocketDisconnect

from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.api_server import AIQChatResponse
from aiq.data_models.api_server import AIQChatResponseChunk
from aiq.data_models.api_server import AIQResponsePayloadOutput
from aiq.data_models.api_server import AIQResponseSerializable
from aiq.data_models.api_server import WebSocketMessageStatus
from aiq.data_models.api_server import WorkflowSchemaType
from aiq.front_ends.fastapi.message_handler import MessageHandler
from aiq.front_ends.fastapi.response_helpers import generate_streaming_response
from aiq.front_ends.fastapi.step_adaptor import StepAdaptor
from aiq.runtime.session import AIQSessionManager

logger = logging.getLogger(__name__)


class AIQWebSocket(WebSocketEndpoint):
    encoding = "json"

    def __init__(self, session_manager: AIQSessionManager, step_adaptor: StepAdaptor, *args, **kwargs):
        self._session_manager: AIQSessionManager = session_manager
        self._message_handler: MessageHandler = MessageHandler(self)
        self._process_response_event: asyncio.Event = asyncio.Event()
        self._workflow_schema_type: dict[str, Callable[..., Awaitable[Any]]] = {
            WorkflowSchemaType.GENERATE_STREAM: self.process_generate_stream,
            WorkflowSchemaType.CHAT_STREAM: self.process_chat_stream,
            WorkflowSchemaType.GENERATE: self.process_generate,
            WorkflowSchemaType.CHAT: self.process_chat
        }
        self._step_adaptor = step_adaptor
        super().__init__(*args, **kwargs)

    @property
    def workflow_schema_type(self) -> dict[str, Callable[..., Awaitable[Any]]]:
        return self._workflow_schema_type

    @property
    def process_response_event(self) -> asyncio.Event:
        return self._process_response_event

    async def on_connect(self, websocket: WebSocket):
        try:
            # Accept the websocket connection
            await websocket.accept()
            try:
                # Start background message processors
                self._message_handler.process_messages_task = asyncio.create_task(
                    self._message_handler.process_messages())

                self._message_handler.process_out_going_messages_task = asyncio.create_task(
                    self._message_handler.process_out_going_messages(websocket))

            except asyncio.CancelledError:
                pass

        except (WebSocketDisconnect, WebSocketException):
            logger.error("A WebSocket error occured during `on_connect`. Ignoring the connection.", exc_info=True)

    async def on_send(self, websocket: WebSocket, data: dict[str, str]):
        try:
            await websocket.send_json(data)
        except (WebSocketDisconnect, WebSocketException, Exception):
            logger.error("A WebSocket error occurred during `on_send`. Ignoring the connection.", exc_info=True)

    async def on_receive(self, websocket: WebSocket, data: dict[str, Any]):
        try:
            await self._message_handler.messages_queue.put(data)
        except (Exception):
            logger.error("An unxpected error occurred during `on_receive`. Ignoring the exception", exc_info=True)

    async def on_disconnect(self, websocket: WebSocket, close_code: Any):
        try:
            if self._message_handler.process_messages_task:
                self._message_handler.process_messages_task.cancel()

            if self._message_handler.process_out_going_messages_task:
                self._message_handler.process_out_going_messages_task.cancel()

            if self._message_handler.background_task:
                self._message_handler.background_task.cancel()

        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    async def _process_message(self,
                               payload: typing.Any,
                               conversation_id: str | None = None,
                               result_type: type | None = None,
                               output_type: type | None = None) -> None:

        async with self._session_manager.session(
                conversation_id=conversation_id,
                user_input_callback=self._message_handler.human_interaction) as session:

            async for value in generate_streaming_response(payload,
                                                           session_manager=session,
                                                           streaming=True,
                                                           step_adaptor=self._step_adaptor,
                                                           result_type=result_type,
                                                           output_type=output_type):

                await self._process_response_event.wait()

                if not isinstance(value, AIQResponseSerializable):
                    value = AIQResponsePayloadOutput(payload=value)

                await self._message_handler.create_websocket_message(data_model=value,
                                                                     status=WebSocketMessageStatus.IN_PROGRESS)

    async def process_generate_stream(self, payload: str, conversation_id: str) -> None:

        return await self._process_message(payload, conversation_id=conversation_id, result_type=None, output_type=None)

    async def process_chat_stream(self, payload: AIQChatRequest, conversation_id: str):

        return await self._process_message(payload,
                                           conversation_id=conversation_id,
                                           result_type=AIQChatResponse,
                                           output_type=AIQChatResponseChunk)

    async def process_generate(self, payload: typing.Any, conversation_id: str):

        return await self._process_message(payload, conversation_id=conversation_id)

    async def process_chat(self, payload: AIQChatRequest, conversation_id: str):

        return await self._process_message(payload, conversation_id=conversation_id, result_type=AIQChatResponse)
