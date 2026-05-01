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

DESIGN NOTE: Producer-Declared Schema Discovery (Future)
========================================================

Today, registering a non-default schema/extractor is a consumer-side
concern: the consumer calls :func:`register_schema` and
:func:`nat.atof.extractors.register_llm_extractor` (or one of the
``register_*_v1()`` convenience helpers) **before** invoking the
converter. The producer declares ``data_schema = {name, version}`` per
event but offers no mechanism to *deliver* the schema or extractor logic
along with the stream. This works fine when the consumer knows the
producer in advance (the ATOF v0.1 expectation) but becomes friction
once a single consumer wants to ingest streams from multiple producers
without prior coordination — e.g. a forensics tool replaying old
trajectories from a producer it has never seen.

Three design options are on the table for a future ATOF revision; none
are implemented yet. Captured here so the next iteration doesn't
relitigate the trade-off space:

(A) **Stream-level schema manifest** — Reserve the first line of the
    JSONL stream for a non-event manifest:

        {"type": "atof_schema_manifest",
         "schemas": [{"name": ..., "version": ..., "json_schema": {...},
                      "extractor_plugin": "anthropic.messages.v1"}]}

    Consumers parse the manifest, register declared schemas + extractor
    plugins, then process events normally. **Pros**: backward-compat
    (consumers ignore unknown first line), explicit, easy to ship.
    **Cons**: requires a new wire-format reservation; ``extractor_plugin``
    references opaque code (security and trust concerns).

(B) **ATOF-native metadata on root scope-start** — Embed the manifest
    in ``metadata._atof_schemas`` on the root agent ScopeStart event.
    Already-permitted by spec §2.1 (open metadata). **Pros**: no wire
    format change, zero-overhead for streams that don't use it.
    **Cons**: late discovery (consumer can't pre-register before seeing
    events), and requires every producer to remember this convention.

(C) **Out-of-band manifest file** — Ship a sidecar manifest alongside
    the JSONL (e.g. ``trajectory.jsonl`` + ``trajectory.manifest.json``).
    Consumers load both. **Pros**: clean separation; schemas can be
    versioned and signed independently. **Cons**: two-file coupling is
    fragile; transport-level constraints (logs systems, kafka) often
    drop sidecars.

Recommendation when the work is taken up: prototype (A) first — it's
the least invasive and is self-documenting in the stream itself.
Decline (C) unless storage transports demand it. (B) is a cheap
fallback if (A) hits backward-compat blockers.

This block is the architectural commitment record. Update it when the
decision is made; do not expand the registry/helpers in this module
without a corresponding spec amendment.
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

# ---------------------------------------------------------------------------
# Opt-in built-in schemas (NOT auto-registered)
# ---------------------------------------------------------------------------
# These constants ship with the package but are NOT installed into
# SCHEMA_REGISTRY at import time. Consumers register them through the
# pairing helpers in :mod:`nat.atof.extractors` (e.g.
# ``register_anthropic_messages_v1()``), which install both the JSON Schema
# and the matching LLM extractor atomically. This keeps the default
# registry minimal — only providers the consumer has opted into appear
# in lookups, so a stray ``data_schema`` referencing an unregistered
# provider falls through to the converter's "schema not registered"
# warning rather than passing validation but failing extraction.

# Permissive schema covering Anthropic Messages REQUEST and RESPONSE
# shapes. Request shape: top-level ``messages`` array (each carrying
# ``role`` + ``content``, where content is either a string or a list of
# typed content blocks). Response shape: top-level ``content`` array of
# typed blocks plus ``role: "assistant"``.
ANTHROPIC_MESSAGES_V1: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "anthropic/messages@1",
    "title": "Anthropic Messages API payload (request or response, permissive)",
    "type": "object",
    "anyOf": [
        {
            "type": "object", "required": ["messages"]
        },
        {
            "type": "object", "required": ["content"]
        },
    ],
}

# Permissive schema covering Gemini ``generateContent`` REQUEST and
# RESPONSE shapes. Request: top-level ``contents`` array (each entry has
# ``role`` ∈ {user, model} + ``parts`` array where parts are
# polymorphic — ``{text}``, ``{functionCall}``, or ``{functionResponse}``).
# Response: top-level ``candidates`` array (each candidate's
# ``content.parts`` follows the same part shape).
GEMINI_GENERATE_CONTENT_V1: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "gemini/generate-content@1",
    "title": "Gemini generateContent payload (request or response, permissive)",
    "type": "object",
    "anyOf": [
        {
            "type": "object", "required": ["contents"]
        },
        {
            "type": "object", "required": ["candidates"]
        },
    ],
}

__all__ = [
    "ANTHROPIC_MESSAGES_V1",
    "GEMINI_GENERATE_CONTENT_V1",
    "OPENAI_CHAT_COMPLETIONS_V1",
    "SCHEMA_REGISTRY",
    "lookup_schema",
    "register_schema",
]
