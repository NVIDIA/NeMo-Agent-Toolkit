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
from pydantic import field_serializer
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo

from nat.data_models.interactive import HumanPrompt
from nat.utils.type_converter import GlobalTypeConverter


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


class ChatRequest(BaseModel):
    """
    ChatRequest is a data model that represents a request to the AIQ Toolkit chat API.
    Fully compatible with OpenAI Chat Completions API specification.
    """

    # Required fields
    messages: typing.Annotated[list[Message], conlist(Message, min_length=1)]

    # Optional fields (OpenAI Chat Completions API compatible)
    model: str | None = Field(default=None, description="name of the model to use")
    frequency_penalty: float | None = Field(default=0.0,
                                            description="Penalty for new tokens based on frequency in text")
    logit_bias: dict[str, float] | None = Field(default=None,
                                                description="Modify likelihood of specified tokens appearing")
    logprobs: bool | None = Field(default=None, description="Whether to return log probabilities")
    top_logprobs: int | None = Field(default=None, description="Number of most likely tokens to return")
    max_tokens: int | None = Field(default=None, description="Maximum number of tokens to generate")
    n: int | None = Field(default=1, description="Number of chat completion choices to generate")
    presence_penalty: float | None = Field(default=0.0, description="Penalty for new tokens based on presence in text")
    response_format: dict[str, typing.Any] | None = Field(default=None, description="Response format specification")
    seed: int | None = Field(default=None, description="Random seed for deterministic sampling")
    service_tier: typing.Literal["auto", "default"] | None = Field(default=None,
                                                                   description="Service tier for the request")
    stream: bool | None = Field(default=False, description="Whether to stream partial message deltas")
    stream_options: dict[str, typing.Any] | None = Field(default=None, description="Options for streaming")
    temperature: float | None = Field(default=1.0, description="Sampling temperature between 0 and 2")
    top_p: float | None = Field(default=None, description="Nucleus sampling parameter")
    tools: list[dict[str, typing.Any]] | None = Field(default=None, description="List of tools the model may call")
    tool_choice: str | dict[str, typing.Any] | None = Field(default=None, description="Controls which tool is called")
    parallel_tool_calls: bool | None = Field(default=True, description="Whether to enable parallel function calling")
    user: str | None = Field(default=None, description="Unique identifier representing end-user")

    model_config = ConfigDict(extra="allow",
                              json_schema_extra={
                                  "example": {
                                      "model": "nvidia/nemotron",
                                      "messages": [{
                                          "role": "user", "content": "who are you?"
                                      }],
                                      "temperature": 0.7,
                                      "stream": False
                                  }
                              })

    @staticmethod
    def from_string(data: str,
                    *,
                    model: str | None = None,
                    temperature: float | None = None,
                    max_tokens: int | None = None,
                    top_p: float | None = None) -> "ChatRequest":

        return ChatRequest(messages=[Message(content=data, role="user")],
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
                     top_p: float | None = None) -> "ChatRequest":

        return ChatRequest(messages=[Message(content=content, role="user")],
                           model=model,
                           temperature=temperature,
                           max_tokens=max_tokens,
                           top_p=top_p)


class ChoiceMessage(BaseModel):
    content: str | None = None
    role: str | None = None


class ChoiceDelta(BaseModel):
    """Delta object for streaming responses (OpenAI-compatible)"""
    content: str | None = None
    role: str | None = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: ChoiceMessage | None = None
    delta: ChoiceDelta | None = None
    finish_reason: typing.Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'] | None = None
    index: int
    # logprobs: AIQChoiceLogprobs | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseSerializable(abc.ABC):
    """
    AIQChatResponseSerializable is an abstract class that defines the interface for serializing output for the AIQ
    Toolkit chat streaming API.
    """

    @abstractmethod
    def get_stream_data(self) -> str:
        pass


class ResponseBaseModelOutput(BaseModel, ResponseSerializable):

    def get_stream_data(self) -> str:
        return f"data: {self.model_dump_json()}\n\n"


class ResponseBaseModelIntermediate(BaseModel, ResponseSerializable):

    def get_stream_data(self) -> str:
        return f"intermediate_data: {self.model_dump_json()}\n\n"


class ChatResponse(ResponseBaseModelOutput):
    """
    ChatResponse is a data model that represents a response from the AIQ Toolkit chat API.
    Fully compatible with OpenAI Chat Completions API specification.
    """

    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")
    id: str
    object: str = "chat.completion"
    model: str = ""
    created: datetime.datetime
    choices: list[Choice]
    usage: Usage | None = None
    system_fingerprint: str | None = None
    service_tier: typing.Literal["scale", "default"] | None = None

    @field_serializer('created')
    def serialize_created(self, created: datetime.datetime) -> int:
        """Serialize datetime to Unix timestamp for OpenAI compatibility"""
        return int(created.timestamp())

    @staticmethod
    def from_string(data: str,
                    *,
                    id_: str | None = None,
                    object_: str | None = None,
                    model: str | None = None,
                    created: datetime.datetime | None = None,
                    usage: Usage | None = None) -> "ChatResponse":

        if id_ is None:
            id_ = str(uuid.uuid4())
        if object_ is None:
            object_ = "chat.completion"
        if model is None:
            model = ""
        if created is None:
            created = datetime.datetime.now(datetime.timezone.utc)

        return ChatResponse(id=id_,
                            object=object_,
                            model=model,
                            created=created,
                            choices=[Choice(index=0, message=ChoiceMessage(content=data), finish_reason="stop")],
                            usage=usage)


class ChatResponseChunk(ResponseBaseModelOutput):
    """
    ChatResponseChunk is a data model that represents a response chunk from the AIQ Toolkit chat streaming API.
    Fully compatible with OpenAI Chat Completions API specification.
    """

    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")

    id: str
    choices: list[Choice]
    created: datetime.datetime
    model: str = ""
    object: str = "chat.completion.chunk"
    system_fingerprint: str | None = None
    service_tier: typing.Literal["scale", "default"] | None = None
    usage: Usage | None = None

    @field_serializer('created')
    def serialize_created(self, created: datetime.datetime) -> int:
        """Serialize datetime to Unix timestamp for OpenAI compatibility"""
        return int(created.timestamp())

    @staticmethod
    def from_string(data: str,
                    *,
                    id_: str | None = None,
                    created: datetime.datetime | None = None,
                    model: str | None = None,
                    object_: str | None = None) -> "ChatResponseChunk":

        if id_ is None:
            id_ = str(uuid.uuid4())
        if created is None:
            created = datetime.datetime.now(datetime.timezone.utc)
        if model is None:
            model = ""
        if object_ is None:
            object_ = "chat.completion.chunk"

        return ChatResponseChunk(id=id_,
                                 choices=[Choice(index=0, message=ChoiceMessage(content=data), finish_reason="stop")],
                                 created=created,
                                 model=model,
                                 object=object_)

    @staticmethod
    def create_streaming_chunk(content: str,
                               *,
                               id_: str | None = None,
                               created: datetime.datetime | None = None,
                               model: str | None = None,
                               role: str | None = None,
                               finish_reason: str | None = None,
                               usage: Usage | None = None,
                               system_fingerprint: str | None = None) -> "ChatResponseChunk":
        """Create an OpenAI-compatible streaming chunk"""
        if id_ is None:
            id_ = str(uuid.uuid4())
        if created is None:
            created = datetime.datetime.now(datetime.timezone.utc)
        if model is None:
            model = ""

        delta = ChoiceDelta(content=content, role=role) if content is not None or role is not None else ChoiceDelta()

        return ChatResponseChunk(id=id_,
                                 choices=[Choice(index=0, message=None, delta=delta, finish_reason=finish_reason)],
                                 created=created,
                                 model=model,
                                 object="chat.completion.chunk",
                                 usage=usage,
                                 system_fingerprint=system_fingerprint)


class ResponseIntermediateStep(ResponseBaseModelIntermediate):
    """
    ResponseSerializedStep is a data model that represents a serialized step in the AIQ Toolkit chat streaming API.
    """

    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")

    id: str
    parent_id: str | None = None
    type: str = "markdown"
    name: str
    payload: str


class ResponsePayloadOutput(BaseModel, ResponseSerializable):

    payload: typing.Any

    def get_stream_data(self) -> str:

        if (isinstance(self.payload, BaseModel)):
            return f"data: {self.payload.model_dump_json()}\n\n"

        return f"data: {self.payload}\n\n"


class GenerateResponse(BaseModel):
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
    conversation_id: str | None = None
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
    conversation_id: str | None = None
    content: SystemResponseContent | Error | GenerateResponse
    status: WebSocketMessageStatus
    timestamp: str = str(datetime.datetime.now(datetime.timezone.utc))

    @field_validator("content")
    @classmethod
    def validate_content_by_type(cls, value: SystemResponseContent | Error | GenerateResponse, info: ValidationInfo):
        if info.data.get("type") == WebSocketMessageType.ERROR_MESSAGE and not isinstance(value, Error):
            raise ValueError(f"Field: content must be 'Error' when type is {WebSocketMessageType.ERROR_MESSAGE}")

        if info.data.get("type") == WebSocketMessageType.RESPONSE_MESSAGE and not isinstance(
                value, (SystemResponseContent, GenerateResponse)):
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
    conversation_id: str | None = None
    content: HumanPrompt
    status: WebSocketMessageStatus
    timestamp: str = str(datetime.datetime.now(datetime.timezone.utc))


# ======== AIQGenerateResponse Converters ========


def _generate_response_to_str(response: GenerateResponse) -> str:
    return response.output


GlobalTypeConverter.register_converter(_generate_response_to_str)


def _generate_response_to_chat_response(response: GenerateResponse) -> ChatResponse:
    data = response.output

    # Simulate usage
    prompt_tokens = 0
    usage = Usage(prompt_tokens=prompt_tokens,
                  completion_tokens=len(data.split()),
                  total_tokens=prompt_tokens + len(data.split()))

    # Build and return the response
    return ChatResponse.from_string(data, usage=usage)


GlobalTypeConverter.register_converter(_generate_response_to_chat_response)


# ======== AIQChatRequest Converters ========
def _aiq_chat_request_to_string(data: ChatRequest) -> str:
    if isinstance(data.messages[-1].content, str):
        return data.messages[-1].content
    return str(data.messages[-1].content)


GlobalTypeConverter.register_converter(_aiq_chat_request_to_string)


def _string_to_aiq_chat_request(data: str) -> ChatRequest:
    return ChatRequest.from_string(data, model="")


GlobalTypeConverter.register_converter(_string_to_aiq_chat_request)


# ======== AIQChatResponse Converters ========
def _aiq_chat_response_to_string(data: ChatResponse) -> str:
    return data.choices[0].message.content or ""


GlobalTypeConverter.register_converter(_aiq_chat_response_to_string)


def _string_to_aiq_chat_response(data: str) -> ChatResponse:
    '''Converts a string to an AIQChatResponse object'''

    # Simulate usage
    prompt_tokens = 0
    usage = Usage(prompt_tokens=prompt_tokens,
                  completion_tokens=len(data.split()),
                  total_tokens=prompt_tokens + len(data.split()))

    # Build and return the response
    return ChatResponse.from_string(data, usage=usage)


GlobalTypeConverter.register_converter(_string_to_aiq_chat_response)


def _chat_response_to_chat_response_chunk(data: ChatResponse) -> ChatResponseChunk:
    # Preserve original message structure for backward compatibility
    return ChatResponseChunk(id=data.id, choices=data.choices, created=data.created, model=data.model)


GlobalTypeConverter.register_converter(_chat_response_to_chat_response_chunk)


# ======== ChatResponseChunk Converters ========
def _chat_response_chunk_to_string(data: ChatResponseChunk) -> str:
    if data.choices and len(data.choices) > 0:
        choice = data.choices[0]
        if choice.delta and choice.delta.content:
            return choice.delta.content
        if choice.message and choice.message.content:
            return choice.message.content
    return ""


GlobalTypeConverter.register_converter(_chat_response_chunk_to_string)


def _string_to_aiq_chat_response_chunk(data: str) -> ChatResponseChunk:
    '''Converts a string to an AIQChatResponseChunk object'''

    # Build and return the response
    return ChatResponseChunk.from_string(data)


GlobalTypeConverter.register_converter(_string_to_aiq_chat_response_chunk)


# ======== AINodeMessageChunk Converters ========
def _ai_message_chunk_to_aiq_chat_response_chunk(data) -> ChatResponseChunk:
    '''Converts LangChain AINodeMessageChunk to AIQChatResponseChunk'''
    content = ""
    if hasattr(data, 'content') and data.content is not None:
        content = str(data.content)
    elif hasattr(data, 'text') and data.text is not None:
        content = str(data.text)
    elif hasattr(data, 'message') and data.message is not None:
        content = str(data.message)

    return ChatResponseChunk.create_streaming_chunk(content=content, role="assistant", finish_reason=None)


# Compatibility aliases with previous releases
AIQChatRequest = ChatRequest
AIQChoiceMessage = ChoiceMessage
AIQChoiceDelta = ChoiceDelta
AIQChoice = Choice
AIQUsage = Usage
AIQResponseSerializable = ResponseSerializable
AIQResponseBaseModelOutput = ResponseBaseModelOutput
AIQResponseBaseModelIntermediate = ResponseBaseModelIntermediate
AIQChatResponse = ChatResponse
AIQChatResponseChunk = ChatResponseChunk
AIQResponseIntermediateStep = ResponseIntermediateStep
AIQResponsePayloadOutput = ResponsePayloadOutput
AIQGenerateResponse = GenerateResponse
