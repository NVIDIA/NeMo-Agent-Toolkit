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


"""Registered JSON Schemas for validating ATOF ``event.data`` payloads.

The ATOF envelope carries an optional ``data_schema = {name, version}``
identifier declaring the shape of ``event.data``. Spec §2 leaves schema
validation to the consumer.

This module maintains a process-wide registry keyed on
``(name, version) -> JSON Schema dict`` and ships one built-in schema:

- ``openai/chat-completions@1`` — permissive shape check for LLM
  scope-start and scope-end payloads; accepts any object carrying at
  least one of the extractable top-level keys: ``messages``, ``content``,
  ``tool_calls``, ``choices``.

External producers register their own schemas via :func:`register_schema`:

    from nat.atof.schemas import register_schema

    register_schema("myco/my-payload", "1", {
        "type": "object",
        "required": ["myco_field"],
    })

Consumers validate an event by looking up the schema and calling
:func:`jsonschema.validate`. The ATOF→ATIF converter wires this into
its pre-pass and raises ``DataSchemaViolationError`` on failure.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: dict[tuple[str, str], dict[str, Any]] = {}


def register_schema(name: str, version: str, schema: dict[str, Any]) -> None:
    """Register a JSON Schema for ATOF events whose ``data_schema`` matches
    ``{name, version}``.

    Overwrites any existing entry with the same key.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("schema name must be a non-empty string")
    if not isinstance(version, str) or not version:
        raise ValueError("schema version must be a non-empty string")
    if not isinstance(schema, dict):
        raise ValueError("schema must be a JSON Schema dict")
    SCHEMA_REGISTRY[(name, version)] = schema


def lookup_schema(name: str, version: str) -> dict[str, Any] | None:
    """Return the registered schema for ``(name, version)`` or ``None``."""
    return SCHEMA_REGISTRY.get((name, version))


# ---------------------------------------------------------------------------
# Built-in schemas
# ---------------------------------------------------------------------------

# Permissive schema covering both OpenAI chat-completions REQUEST shapes
# (``messages`` at top level, or nested under ``content.messages``) and
# RESPONSE shapes (``content`` string, ``tool_calls`` array, or the full
# ``choices[0].message`` structure). Validates only the top-level shape
# boundary — payloads carrying recognizable keys pass, payloads using
# foreign conventions (Anthropic ``input``/``output_blocks``, Gemini
# ``candidates``, etc.) fail.
OPENAI_CHAT_COMPLETIONS_V1: dict[str, Any] = {
    "$schema":
        "https://json-schema.org/draft/2020-12/schema",
    "$id":
        "openai/chat-completions@1",
    "title":
        "OpenAI chat-completions payload (request or response, permissive)",
    "type":
        "object",
    "anyOf": [
        {
            "type": "object", "required": ["messages"]
        },
        {
            "type": "object",
            "required": ["content"],
            "properties": {
                "content": {
                    "oneOf": [
                        {
                            "type": "string"
                        },
                        {
                            "type": "object", "required": ["messages"]
                        },
                    ],
                },
            },
        },
        {
            "type": "object", "required": ["tool_calls"]
        },
        {
            "type": "object", "required": ["choices"]
        },
    ],
}

register_schema("openai/chat-completions", "1", OPENAI_CHAT_COMPLETIONS_V1)

__all__ = [
    "OPENAI_CHAT_COMPLETIONS_V1",
    "SCHEMA_REGISTRY",
    "lookup_schema",
    "register_schema",
]
