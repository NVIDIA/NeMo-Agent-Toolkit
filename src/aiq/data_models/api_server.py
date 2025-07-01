# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import abc
import datetime
import typing
import uuid
from abc import abstractmethod
from enum import Enum

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import HttpUrl
from pydantic import conlist
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from aiq.data_models.interactive import HumanPrompt
from aiq.utils.type_converter import GlobalTypeConverter


class Request(BaseModel):
    """
    Request is a data model that represents HTTP request attributes.
    """
    model_config = ConfigDict(extra="forbid")

    method: str | None = Field(default=None,
                               description="HTTP method used for the request (e.g., GET, POST, PUT, DELETE).")
    url_path: str | None = Field(default=None, description="URL request path.")
    url_port: int | None = Field(default=None, description="URL request port number.")
    url_scheme: str | None = Field(default=None, description="URL scheme indicating the protocol (e.g., http, https).")
    headers: typing.Any | None = Field(default=None, description="HTTP headers associated with the request.")
    query_params: typing.Any | None = Field(default=None, description="Query parameters included in the request URL.")
    path_params: dict[str, str] | None = Field(default=None,
                                               description="Path parameters extracted from the request URL.")
    client_host: str | None = Field(default=None, description="Client host address from which the request originated.")
    client_port: int | None = Field(default=None, description="Client port number from which the request originated.")
    cookies: dict[str, str] | None = Field(
        default=None, description="Cookies sent with the request, stored in a dictionary-like object.")


class ChatContentType(str, Enum):
    """
    ChatContentType is an Enum that represents the type of Chat content.
    """
    TEXT = "text"
    IMAGE_URL = "image_url"
    INPUT_AUDIO = "input_audio"


class InputAudio(BaseModel):
    data: str = "default"
    format: str = "default"


class AudioContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: typing.Literal[ChatContentType.INPUT_AUDIO] = ChatContentType.INPUT_AUDIO
    input_audio: InputAudio = InputAudio()


class ImageUrl(BaseModel):
    url: HttpUrl = HttpUrl(url="http://default.com")


class ImageContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: typing.Literal[ChatContentType.IMAGE_URL] = ChatContentType.IMAGE_URL
    image_url: ImageUrl = ImageUrl()


class TextContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: typing.Literal[ChatContentType.TEXT] = ChatContentType.TEXT
    text: str = "default"


class Security(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str = "default"
    token: str = "default"


UserContent = typing.Annotated[TextContent | ImageContent | AudioContent, Discriminator("type")]


class Message(BaseModel):
    content: str | list[UserContent]
    role: str


class AIQChatRequest(BaseModel):
    """
    AIQChatRequest is a data model that represents a request to the AIQ Toolkit chat API.
    """

    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")

    messages: typing.Annotated[list[Message], conlist(Message, min_length=1)]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None

    @staticmethod
    def from_string(data: str,
                    *,
                    model: str | None = None,
                    temperature: float | None = None,
                    max_tokens: int | None = None,
                    top_p: float | None = None) -> "AIQChatRequest":

        return AIQChatRequest(messages=[Message(content=data, role="user")],
                              model=model,
                              temperature=temperature,
                              max_tokens=max_tokens,
                              top_p=top_p)

    @staticmethod
    def from_content(content: list[UserContent],
                     *,
                     model: str | None = None,
                     temperature: float | None = None,
                     max_tokens: int | None = None,
                     top_p: float | None = None) -> "AIQChatRequest":

        return AIQChatRequest(messages=[Message(content=content, role="user")],
                              model=model,
                              temperature=temperature,
                              max_tokens=max_tokens,
                              top_p=top_p)


class AIQChoiceMessage(BaseModel):
    content: str | None = None
    role: str | None = None


class AIQChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: AIQChoiceMessage
    finish_reason: typing.Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'] | None = None
    index: int
    # logprobs: AIQChoiceLogprobs | None = None


class AIQUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AIQResponseSerializable(abc.ABC):
    """
    AIQChatResponseSerializable is an abstract class that defines the interface for serializing output for the AIQ
    Toolkit chat streaming API.
    """

    @abstractmethod
    def get_stream_data(self) -> str:
        pass


class AIQResponseBaseModelOutput(BaseModel, AIQResponseSerializable):

    def get_stream_data(self) -> str:
        return f"data: {self.model_dump_json()}\n\n"


class AIQResponseBaseModelIntermediate(BaseModel, AIQResponseSerializable):

    def get_stream_data(self) -> str:
        return f"intermediate_data: {self.model_dump_json()}\n\n"


class AIQChatResponse(AIQResponseBaseModelOutput):
    """
    AIQChatResponse is a data model that represents a response from the AIQ Toolkit chat API.
    """

    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")
    id: str
    object: str
    model: str = ""
    created: datetime.datetime
    choices: list[AIQChoice]
    usage: AIQUsage | None = None

    @staticmethod
    def from_string(data: str,
                    *,
                    id_: str | None = None,
                    object_: str | None = None,
                    model: str | None = None,
                    created: datetime.datetime | None = None,
                    usage: AIQUsage | None = None) -> "AIQChatResponse":

        if id_ is None:
            id_ = str(uuid.uuid4())
        if object_ is None:
            object_ = "chat.completion"
        if model is None:
            model = ""
        if created is None:
            created = datetime.datetime.now(datetime.timezone.utc)

        return AIQChatResponse(
            id=id_,
            object=object_,
            model=model,
            created=created,
            choices=[AIQChoice(index=0, message=AIQChoiceMessage(content=data), finish_reason="stop")],
            usage=usage)


class AIQChatResponseChunk(AIQResponseBaseModelOutput):
    """
    AIQChatResponseChunk is a data model that represents a response chunk from the AIQ Toolkit chat streaming API.
    """

    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")

    id: str
    choices: list[AIQChoice]
    created: datetime.datetime
    model: str = ""
    object: str = "chat.completion.chunk"

    @staticmethod
    def from_string(data: str,
                    *,
                    id_: str | None = None,
                    created: datetime.datetime | None = None,
                    model: str | None = None,
                    object_: str | None = None) -> "AIQChatResponseChunk":

        if id_ is None:
            id_ = str(uuid.uuid4())
        if created is None:
            created = datetime.datetime.now(datetime.timezone.utc)
        if model is None:
            model = ""
        if object_ is None:
            object_ = "chat.completion.chunk"

        return AIQChatResponseChunk(
            id=id_,
            choices=[AIQChoice(index=0, message=AIQChoiceMessage(content=data), finish_reason="stop")],
            created=created,
            model=model,
            object=object_)


class AIQResponseIntermediateStep(AIQResponseBaseModelIntermediate):
    """
    AIQResponseSerializedStep is a data model that represents a serialized step in the AIQ Toolkit chat streaming API.
    """

    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")

    id: str
    parent_id: str | None = None
    type: str = "markdown"
    name: str
    payload: str


class AIQResponsePayloadOutput(BaseModel, AIQResponseSerializable):

    payload: typing.Any

    def get_stream_data(self) -> str:

        if (isinstance(self.payload, BaseModel)):
            return f"data: {self.payload.model_dump_json()}\n\n"

        return f"data: {self.payload}\n\n"


class AIQGenerateResponse(BaseModel):
    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")

    # (fixme) define the intermediate step model
    intermediate_steps: list[tuple] | None = None
    output: str
    value: str | None = "default"


class UserMessageContentRoleType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class WebSocketMessageType(str, Enum):
    """
    WebSocketMessageType is an Enum that represents WebSocket Message types.
    """
    USER_MESSAGE = "user_message"
    RESPONSE_MESSAGE = "system_response_message"
    INTERMEDIATE_STEP_MESSAGE = "system_intermediate_message"
    SYSTEM_INTERACTION_MESSAGE = "system_interaction_message"
    USER_INTERACTION_MESSAGE = "user_interaction_message"
    ERROR_MESSAGE = "error_message"


class WorkflowSchemaType(str, Enum):
    """
    WorkflowSchemaType is an Enum that represents Workkflow response types.
    """
    GENERATE_STREAM = "generate_stream"
    CHAT_STREAM = "chat_stream"
    GENERATE = "generate"
    CHAT = "chat"


class WebSocketMessageStatus(str, Enum):
    """
    WebSocketMessageStatus is an Enum that represents the status of a WebSocket message.
    """
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class UserMessages(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: UserMessageContentRoleType
    content: list[UserContent]


class UserMessageContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[UserMessages]


class User(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    email: str = "default"


class ErrorTypes(str, Enum):
    UNKNOWN_ERROR = "unknown_error"
    INVALID_MESSAGE = "invalid_message"
    INVALID_MESSAGE_TYPE = "invalid_message_type"
    INVALID_USER_MESSAGE_CONTENT = "invalid_user_message_content"
    INVALID_DATA_CONTENT = "invalid_data_content"


class Error(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorTypes = ErrorTypes.UNKNOWN_ERROR
    message: str = "default"
    details: str = "default"


class WebSocketUserMessage(BaseModel):
    """
    For more details, refer to the API documentation:
    docs/source/developer_guide/websockets.md
    """
    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")

    type: typing.Literal[WebSocketMessageType.USER_MESSAGE]
    schema_type: WorkflowSchemaType
    id: str = "default"
    conversation_id: str | None = None
    content: UserMessageContent
    user: User = User()
    security: Security = Security()
    error: Error = Error()
    schema_version: str = "1.0.0"
    timestamp: str = str(datetime.datetime.now(datetime.timezone.utc))


class WebSocketUserInteractionResponseMessage(BaseModel):
    """
    For more details, refer to the API documentation:
    docs/source/developer_guide/websockets.md
    """
    type: typing.Literal[WebSocketMessageType.USER_INTERACTION_MESSAGE]
    id: str = "default"
    thread_id: str = "default"
    content: UserMessageContent
    user: User = User()
    security: Security = Security()
    error: Error = Error()
    schema_version: str = "1.0.0"
    timestamp: str = str(datetime.datetime.now(datetime.timezone.utc))


class SystemIntermediateStepContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    payload: str


class WebSocketSystemIntermediateStepMessage(BaseModel):
    """
    For more details, refer to the API documentation:
    docs/source/developer_guide/websockets.md
    """
    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")

    type: typing.Literal[WebSocketMessageType.INTERMEDIATE_STEP_MESSAGE]
    id: str = "default"
    thread_id: str | None = "default"
    parent_id: str = "default"
    intermediate_parent_id: str | None = "default"
    update_message_id: str | None = "default"
    content: SystemIntermediateStepContent
    status: WebSocketMessageStatus
    timestamp: str = str(datetime.datetime.now(datetime.timezone.utc))


class SystemResponseContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = None


class WebSocketSystemResponseTokenMessage(BaseModel):
    """
    For more details, refer to the API documentation:
    docs/source/developer_guide/websockets.md
    """
    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")

    type: typing.Literal[WebSocketMessageType.RESPONSE_MESSAGE, WebSocketMessageType.ERROR_MESSAGE]
    id: str | None = "default"
    thread_id: str | None = "default"
    parent_id: str = "default"
    content: SystemResponseContent | Error | AIQGenerateResponse
    status: WebSocketMessageStatus
    timestamp: str = str(datetime.datetime.now(datetime.timezone.utc))

    @field_validator("content")
    @classmethod
    def validate_content_by_type(cls, value: SystemResponseContent | Error | AIQGenerateResponse, info: ValidationInfo):
        if info.data.get("type") == WebSocketMessageType.ERROR_MESSAGE and not isinstance(value, Error):
            raise ValueError(f"Field: content must be 'Error' when type is {WebSocketMessageType.ERROR_MESSAGE}")

        if info.data.get("type") == WebSocketMessageType.RESPONSE_MESSAGE and not isinstance(
                value, (SystemResponseContent, AIQGenerateResponse)):
            raise ValueError(
                f"Field: content must be 'SystemResponseContent' when type is {WebSocketMessageType.RESPONSE_MESSAGE}")
        return value


class WebSocketSystemInteractionMessage(BaseModel):
    """
    For more details, refer to the API documentation:
    docs/source/developer_guide/websockets.md
    """
    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")

    type: typing.Literal[
        WebSocketMessageType.SYSTEM_INTERACTION_MESSAGE] = WebSocketMessageType.SYSTEM_INTERACTION_MESSAGE
    id: str | None = "default"
    thread_id: str | None = "default"
    parent_id: str = "default"
    content: HumanPrompt
    status: WebSocketMessageStatus
    timestamp: str = str(datetime.datetime.now(datetime.timezone.utc))


# ======== AIQGenerateResponse Converters ========


def _generate_response_to_str(response: AIQGenerateResponse) -> str:
    return response.output


GlobalTypeConverter.register_converter(_generate_response_to_str)


def _generate_response_to_chat_response(response: AIQGenerateResponse) -> AIQChatResponse:
    data = response.output

    # Simulate usage
    prompt_tokens = 0
    usage = AIQUsage(prompt_tokens=prompt_tokens,
                     completion_tokens=len(data.split()),
                     total_tokens=prompt_tokens + len(data.split()))

    # Build and return the response
    return AIQChatResponse.from_string(data, usage=usage)


GlobalTypeConverter.register_converter(_generate_response_to_chat_response)


# ======== AIQChatRequest Converters ========
def _aiq_chat_request_to_string(data: AIQChatRequest) -> str:
    if isinstance(data.messages[-1].content, str):
        return data.messages[-1].content
    return str(data.messages[-1].content)


GlobalTypeConverter.register_converter(_aiq_chat_request_to_string)


def _string_to_aiq_chat_request(data: str) -> AIQChatRequest:
    return AIQChatRequest.from_string(data, model="")


GlobalTypeConverter.register_converter(_string_to_aiq_chat_request)


# ======== AIQChatResponse Converters ========
def _aiq_chat_response_to_string(data: AIQChatResponse) -> str:
    return data.choices[0].message.content or ""


GlobalTypeConverter.register_converter(_aiq_chat_response_to_string)


def _string_to_aiq_chat_response(data: str) -> AIQChatResponse:
    '''Converts a string to an AIQChatResponse object'''

    # Simulate usage
    prompt_tokens = 0
    usage = AIQUsage(prompt_tokens=prompt_tokens,
                     completion_tokens=len(data.split()),
                     total_tokens=prompt_tokens + len(data.split()))

    # Build and return the response
    return AIQChatResponse.from_string(data, usage=usage)


GlobalTypeConverter.register_converter(_string_to_aiq_chat_response)


def _chat_response_to_chat_response_chunk(data: AIQChatResponse) -> AIQChatResponseChunk:

    return AIQChatResponseChunk(id=data.id, choices=data.choices, created=data.created, model=data.model)


GlobalTypeConverter.register_converter(_chat_response_to_chat_response_chunk)


# ======== AIQChatResponseChunk Converters ========
def _aiq_chat_response_chunk_to_string(data: AIQChatResponseChunk) -> str:
    return data.choices[0].message.content or ""


GlobalTypeConverter.register_converter(_aiq_chat_response_chunk_to_string)


def _string_to_aiq_chat_response_chunk(data: str) -> AIQChatResponseChunk:
    '''Converts a string to an AIQChatResponseChunk object'''

    # Build and return the response
    return AIQChatResponseChunk.from_string(data)


GlobalTypeConverter.register_converter(_string_to_aiq_chat_response_chunk)
