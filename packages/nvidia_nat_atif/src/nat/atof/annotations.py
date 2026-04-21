# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Annotation schemas for LLM request/response payloads.

Standalone Pydantic models for structured LLM request and response data.
These appear on ``ScopeStartEvent.annotated_request`` and
``ScopeEndEvent.annotated_response`` when ``scope_type == "llm"``.

See ``atof-schema-profiles.md`` §4 for the canonical shapes.
"""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

# ---------------------------------------------------------------------------
# Request-side types (Section 5.1-5.6)
# ---------------------------------------------------------------------------


class ContentPart(BaseModel):
    """A single content part within a multimodal message (Section 5.3)."""

    type: Literal["text"] = "text"
    text: str

    model_config = ConfigDict(extra="forbid")


class RequestToolCall(BaseModel):
    """A tool call in the assistant message's ``tool_calls`` array (Section 5.4).

    Note: ``function.arguments`` is a JSON **string** on the request side
    (OpenAI wire convention).
    """

    id: str
    type: str = "function"
    function: RequestFunctionCall

    model_config = ConfigDict(extra="forbid")


class RequestFunctionCall(BaseModel):
    """Function call within a request-side ToolCall."""

    name: str
    arguments: str = Field(description="Raw JSON string per OpenAI convention")

    model_config = ConfigDict(extra="forbid")


class ToolDefinition(BaseModel):
    """A tool/function schema available to the model (Section 5.5)."""

    type: str = "function"
    function: FunctionDefinition

    model_config = ConfigDict(extra="forbid")


class FunctionDefinition(BaseModel):
    """Function definition within a ToolDefinition."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class GenerationParams(BaseModel):
    """Normalized generation parameters (Section 5.6)."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None

    model_config = ConfigDict(extra="forbid")


class ToolChoiceFunction(BaseModel):
    """Forces a specific function by name (Section 5.11)."""

    type: str = "function"
    function: ToolChoiceFunctionName

    model_config = ConfigDict(extra="forbid")


class ToolChoiceFunctionName(BaseModel):
    """The name component of a specific tool choice."""

    name: str

    model_config = ConfigDict(extra="forbid")


class Message(BaseModel):
    """A message in the conversation history (Section 5.2).

    Discriminated by ``role``. Optional fields are present only for certain roles.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[RequestToolCall] | None = None
    tool_call_id: str | None = None

    model_config = ConfigDict(extra="allow")


class AnnotatedLLMRequest(BaseModel):
    """Structured, decoded view of an LLM request (atof-schema-profiles.md §4.1)."""

    messages: list[Message]
    model: str | None = None
    params: GenerationParams | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | ToolChoiceFunction | None = None

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Response-side types (Section 5.7-5.10)
# ---------------------------------------------------------------------------


class ResponseToolCall(BaseModel):
    """A tool call in the response's ``tool_calls`` array (Section 5.8).

    Note: ``arguments`` is a parsed JSON **object** on the response side
    (normalized by the producer), NOT a string.
    """

    id: str
    name: str
    arguments: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class Usage(BaseModel):
    """Token usage statistics (Section 5.9)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None

    model_config = ConfigDict(extra="forbid")


class ApiSpecificResponse(BaseModel):
    """Provider-specific response fields (Section 5.10).

    Discriminated by ``api`` field.
    """

    api: str
    # OpenAI Chat
    logprobs: Any | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
    # OpenAI Responses
    output_items: list[Any] | None = None
    status: str | None = None
    incomplete_details: Any | None = None
    # Anthropic Messages
    stop_sequence: str | None = None
    content_blocks: list[Any] | None = None
    # Custom
    api_name: str | None = None
    data: Any | None = None

    model_config = ConfigDict(extra="allow")


class AnnotatedLLMResponse(BaseModel):
    """Structured, decoded view of an LLM response (atof-schema-profiles.md §4.2)."""

    id: str | None = None
    model: str | None = None
    message: str | list[ContentPart] | None = None
    tool_calls: list[ResponseToolCall] | None = None
    finish_reason: str | dict[str, str] | None = None
    usage: Usage | None = None
    api_specific: ApiSpecificResponse | None = None

    model_config = ConfigDict(extra="allow")
