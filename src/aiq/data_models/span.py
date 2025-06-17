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

import logging
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    LLM = "LLM"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    UNKNOWN = "UNKNOWN"


EVENT_TYPE_TO_SPAN_KIND_MAP = {
    "LLM_START": SpanKind.LLM,
    "LLM_END": SpanKind.LLM,
    "LLM_NEW_TOKEN": SpanKind.LLM,
    "TOOL_START": SpanKind.TOOL,
    "TOOL_END": SpanKind.TOOL,
    "FUNCTION_START": SpanKind.CHAIN,
    "FUNCTION_END": SpanKind.CHAIN,
}


def event_type_to_span_kind(event_type: str) -> SpanKind:
    return EVENT_TYPE_TO_SPAN_KIND_MAP.get(event_type, SpanKind.UNKNOWN)


class SpanAttributes(Enum):
    AIQ_SPAN_KIND = "aiq.span.kind"
    INPUT_VALUE = "input.value"
    INPUT_MIME_TYPE = "input.mime_type"
    LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
    LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
    LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
    OUTPUT_VALUE = "output.value"
    OUTPUT_MIME_TYPE = "output.mime_type"
    AIQ_USAGE_NUM_LLM_CALLS = "aiq.usage.num_llm_calls"
    AIQ_USAGE_SECONDS_BETWEEN_CALLS = "aiq.usage.seconds_between_calls"
    AIQ_USAGE_TOKEN_COUNT_PROMPT = "aiq.usage.token_count.prompt"
    AIQ_USAGE_TOKEN_COUNT_COMPLETION = "aiq.usage.token_count.completion"
    AIQ_USAGE_TOKEN_COUNT_TOTAL = "aiq.usage.token_count.total"
    AIQ_EVENT_TYPE = "aiq.event_type"


class MimeTypes(Enum):
    TEXT = "text/plain"
    JSON = "application/json"


class SpanStatusCode(Enum):
    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


class SpanEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class SpanStatus(BaseModel):
    code: SpanStatusCode = SpanStatusCode.OK
    message: str | None = None


class SpanContext(BaseModel):
    trace_id: int = uuid.uuid4().int & ((1 << 128) - 1)
    span_id: int = uuid.uuid4().int & ((1 << 64) - 1)


class Span(BaseModel):
    name: str
    context: SpanContext | None = None
    parent: "Span | None" = None
    start_time: float = Field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)
    status: SpanStatus = Field(default_factory=SpanStatus)

    @field_validator('context', mode='before')
    @classmethod
    def set_default_context(cls, v: SpanContext | None) -> SpanContext:
        if v is None:
            return SpanContext()
        return v

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        if attributes is None:
            attributes = {}
        self.events = self.events + [SpanEvent(name=name, attributes=attributes)]

    def end(self, end_time: float | None = None) -> None:
        if end_time is None:
            end_time = time.time()
        self.end_time = end_time
