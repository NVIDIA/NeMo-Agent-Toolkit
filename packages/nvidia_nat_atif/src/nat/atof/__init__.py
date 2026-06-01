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
"""Pydantic models for the Agentic Trajectory Observability Format (ATOF).

ATOF is a JSON-Lines wire format for agent runtime event streams. These
models define the two event kinds (``ScopeEvent``, ``MarkEvent``), the
behavioral flag enum (``Flags``), and the canonical ``category``
vocabulary (``Category``).

See ``atof-event-format.md`` for the core wire format. For payload
extraction, see :mod:`nat.atof.extractors` (schema-map-driven LLM
extractors for OpenAI, Anthropic, and Gemini). For the open question
of how producers should *declare* their schemas to consumers (a future
spec revision), see the DESIGN NOTE block at the top of
:mod:`nat.atof.schemas`.
"""

from nat.atof.category import Category
from nat.atof.events import Event
from nat.atof.events import MarkEvent
from nat.atof.events import ScopeEvent
from nat.atof.extractors import ANTHROPIC_MESSAGES_V1_MAP
from nat.atof.extractors import GEMINI_GENERATE_CONTENT_V1_MAP
from nat.atof.extractors import LLM_EXTRACTOR_REGISTRY
from nat.atof.extractors import MARK_EXTRACTOR_REGISTRY
from nat.atof.extractors import OPENAI_CHAT_COMPLETIONS_V1_MAP
from nat.atof.extractors import TOOL_EXTRACTOR_REGISTRY
from nat.atof.extractors import LlmPayloadExtractor
from nat.atof.extractors import MarkPayloadExtractor
from nat.atof.extractors import SchemaMap
from nat.atof.extractors import SchemaMapLlmExtractor
from nat.atof.extractors import ToolPayloadExtractor
from nat.atof.extractors import register_anthropic_messages_v1
from nat.atof.extractors import register_gemini_generate_content_v1
from nat.atof.extractors import register_llm_extractor
from nat.atof.extractors import register_mark_extractor
from nat.atof.extractors import register_tool_extractor
from nat.atof.flags import Flags
from nat.atof.io import read_jsonl
from nat.atof.io import write_jsonl
from nat.atof.schemas import ANTHROPIC_MESSAGES_V1
from nat.atof.schemas import GEMINI_GENERATE_CONTENT_V1
from nat.atof.schemas import SCHEMA_REGISTRY
from nat.atof.schemas import lookup_schema
from nat.atof.schemas import register_schema

__all__ = [
    "ANTHROPIC_MESSAGES_V1",
    "ANTHROPIC_MESSAGES_V1_MAP",
    "GEMINI_GENERATE_CONTENT_V1",
    "GEMINI_GENERATE_CONTENT_V1_MAP",
    "LLM_EXTRACTOR_REGISTRY",
    "MARK_EXTRACTOR_REGISTRY",
    "OPENAI_CHAT_COMPLETIONS_V1_MAP",
    "SCHEMA_REGISTRY",
    "TOOL_EXTRACTOR_REGISTRY",
    "Category",
    "Event",
    "Flags",
    "LlmPayloadExtractor",
    "MarkEvent",
    "MarkPayloadExtractor",
    "SchemaMap",
    "SchemaMapLlmExtractor",
    "ScopeEvent",
    "ToolPayloadExtractor",
    "lookup_schema",
    "read_jsonl",
    "register_anthropic_messages_v1",
    "register_gemini_generate_content_v1",
    "register_llm_extractor",
    "register_mark_extractor",
    "register_schema",
    "register_tool_extractor",
    "write_jsonl",
]
