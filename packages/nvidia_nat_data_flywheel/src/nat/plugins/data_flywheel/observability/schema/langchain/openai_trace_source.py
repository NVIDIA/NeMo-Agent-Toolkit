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

import logging
from typing import Any
from typing import Literal
from typing import TypeVar

from pydantic import BaseModel
from pydantic import field_validator

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import ToolSchema
from nat.plugins.data_flywheel.observability.schema.langchain.openai_message import OpenAIMessage
from nat.plugins.data_flywheel.observability.schema.provider import Provider
from nat.plugins.data_flywheel.observability.schema.trace_source_base import TraceSourceBase
from nat.plugins.data_flywheel.observability.utils.deserialize import deserialize_span_attribute

ProviderT = TypeVar("ProviderT")

logger = logging.getLogger(__name__)


class OpenAIMetadata(BaseModel):
    tools_schema: list[ToolSchema] | None = None
    chat_responses: list[dict[str, Any]] | None = None


class OpenAITraceSourceBase(TraceSourceBase[Literal[LLMFrameworkEnum.LANGCHAIN], ProviderT]):
    framework: Literal[LLMFrameworkEnum.LANGCHAIN] = LLMFrameworkEnum.LANGCHAIN
    input_value: list[OpenAIMessage]
    metadata: OpenAIMetadata

    @field_validator("input_value", mode="before")
    @classmethod
    def validate_input_value(cls, v: Any) -> list[OpenAIMessage]:
        if v is None:
            raise ValueError("Input value is required")

        # Handle string input (JSON string)
        if isinstance(v, str):
            v = deserialize_span_attribute(v)

        # Handle dict input (single message)
        if isinstance(v, dict):
            v = [v]

        # Validate list of messages
        if isinstance(v, list):
            validated_messages = []
            for msg in v:
                if isinstance(msg, dict):
                    validated_messages.append(OpenAIMessage(**msg))
                elif isinstance(msg, OpenAIMessage):
                    validated_messages.append(msg)
                else:
                    raise ValueError(f"Invalid message format: {msg}")
            return validated_messages

        raise ValueError(f"Invalid input_value format: {v}")

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_tools_schema(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}

        if isinstance(v, str):
            metadata = deserialize_span_attribute(v)
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid metadata format: {metadata}")
            return metadata

        raise ValueError(f"Invalid metadata format: {v}")


class OpenAITraceSource(OpenAITraceSourceBase[Literal[Provider.OPENAI]]):
    provider: Literal[Provider.OPENAI] = Provider.OPENAI
