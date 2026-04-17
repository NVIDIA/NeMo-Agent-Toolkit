<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Agentic Trajectory Observability Format (ATOF) Specification

**Status:** Active  
**Version:** 0.1  
**Date:** 2026-04-12  
**Reference Implementation:** `src/nat/atof/`

---

## 1. Overview

ATOF (Agentic Trajectory Observability Format) is the wire format for agent runtime subscriber callbacks. Events represent the lifecycle of scopes, LLM calls, and tool invocations within the agent runtime. Subscribers receive events in real time as the runtime executes agent workflows.

Transport is JSON-Lines: each event is one JSON object per line. The `kind` field is the outer discriminator. Valid `kind` values are:

- `"ScopeStart"` — a scope was opened
- `"ScopeEnd"` — a scope was closed
- `"LLMStart"` — an LLM call began
- `"LLMEnd"` — an LLM call completed
- `"ToolStart"` — a tool invocation began
- `"ToolEnd"` — a tool invocation completed
- `"Mark"` — a named checkpoint was emitted

**Wire envelope shape:**

```json
{"kind": "LLMStart", "uuid": "...", "parent_uuid": "...", "timestamp": "...", "name": "...", "attributes": [], "input": {...}, "model_name": "nvidia/nemotron-3-super-v3", "data": null, "metadata": null}
```

ATOF events are the raw, un-merged observations from the runtime. A separate conversion layer (see Section 7) maps them to ATIF steps.

---

## 2. Common Event Fields

All seven event types share these six fields.


| Field         | Type                  | Required | Description                                                                                                                                                                 |
| ------------- | --------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `parent_uuid` | string (UUID) or null | No       | UUID of the scope that was on top of the stack when this handle was created. Null only on the root scope. Following `parent_uuid` links upward reconstructs the call graph. |
| `uuid`        | string (UUID)         | Yes      | Unique identifier for this handle. The matching Start and End events for the same handle carry the same `uuid`.                                                             |
| `timestamp`   | string (RFC 3339)     | Yes      | Wall-clock time when this event was emitted. Start and End events for the same handle have different timestamps.                                                            |
| `name`        | string                | Yes      | Human-readable label for this handle — e.g., `"my_agent"`, `"calculator__add"`, `"nvidia/nemotron-3-super-v3"`.                                                             |
| `data`        | object or null        | No       | Application-specific JSON payload attached by the caller.                                                                                                                   |
| `metadata`    | object or null        | No       | Tracing and correlation metadata — e.g., `{"trace_id": "...", "span_id": "..."}`.                                                                                           |


---

## 3. Event Types

### 3.1 ScopeStartEvent

Emitted when a new scope is pushed onto the scope stack.


| Field         | Type                  | Required | Description                                                                                                                                               |
| ------------- | --------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `parent_uuid` | string (UUID) or null | No       | See Section 2.                                                                                                                                            |
| `uuid`        | string (UUID)         | Yes      | See Section 2.                                                                                                                                            |
| `timestamp`   | string (RFC 3339)     | Yes      | See Section 2.                                                                                                                                            |
| `name`        | string                | Yes      | See Section 2.                                                                                                                                            |
| `data`        | object or null        | No       | See Section 2.                                                                                                                                            |
| `metadata`    | object or null        | No       | See Section 2.                                                                                                                                            |
| `attributes`  | array of strings      | Yes      | Behavioral flag names, canonical form (lowercase, sorted, deduplicated). Canonical values: `"parallel"`, `"relocatable"`. Empty array when no flags are set. See Section 4 for flag descriptions and the extensibility rule. |
| `scope_type`  | string (enum)         | Yes      | One of: `"agent"`, `"function"`, `"tool"`, `"llm"`, `"retriever"`, `"embedder"`, `"reranker"`, `"guardrail"`, `"evaluator"`, `"custom"`, `"unknown"`.     |
| `input`       | any or null           | No       | Optional sanitized input payload handed to the scope at entry (post request-sanitize guardrails). Omitted or null when the scope has no meaningful input, or the emitter does not capture one. |


### 3.2 ScopeEndEvent

Emitted when a scope is popped from the scope stack. Fields mirror `ScopeStartEvent` except that `input` is replaced by an optional `output`.


| Field         | Type                  | Required | Description                                                                                                                                                           |
| ------------- | --------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `parent_uuid` | string (UUID) or null | No       | See Section 2.                                                                                                                                                        |
| `uuid`        | string (UUID)         | Yes      | Same value as the matching `ScopeStartEvent`.                                                                                                                         |
| `timestamp`   | string (RFC 3339)     | Yes      | See Section 2.                                                                                                                                                        |
| `name`        | string                | Yes      | Same value as the matching `ScopeStartEvent`.                                                                                                                         |
| `data`        | object or null        | No       | See Section 2.                                                                                                                                                        |
| `metadata`    | object or null        | No       | See Section 2.                                                                                                                                                        |
| `attributes`  | array of strings      | Yes      | Same value as the matching `ScopeStartEvent`. See Section 4.                                                                                                          |
| `scope_type`  | string (enum)         | Yes      | Same value as the matching `ScopeStartEvent`.                                                                                                                         |
| `output`      | any or null           | No       | Optional sanitized output payload produced by the scope at exit (post response-sanitize guardrails). Omitted or null when the scope has no meaningful return value, or the emitter does not capture one. |


### 3.3 LLMStartEvent

Emitted when an LLM call begins. Carries the post request-sanitize guardrail payload (guardrails have already run on the request before the event is emitted).


| Field               | Type                  | Required | Description                                                                                                                                                                                        |
| ------------------- | --------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `parent_uuid`       | string (UUID) or null | No       | See Section 2.                                                                                                                                                                                     |
| `uuid`              | string (UUID)         | Yes      | See Section 2. Stable across the matching `LLMStartEvent` and `LLMEndEvent`.                                                                                                                       |
| `timestamp`         | string (RFC 3339)     | Yes      | See Section 2.                                                                                                                                                                                     |
| `name`              | string                | Yes      | See Section 2.                                                                                                                                                                                     |
| `data`              | object or null        | No       | See Section 2.                                                                                                                                                                                     |
| `metadata`          | object or null        | No       | See Section 2.                                                                                                                                                                                     |
| `attributes`        | array of strings      | Yes      | LLM behavioral flag names, canonical form (lowercase, sorted, deduplicated). Canonical values: `"stateless"`, `"streaming"`. Empty array when no flags are set. See Section 4.                     |
| `input`             | any or null           | No       | Sanitized LLM request payload (post request-sanitize guardrails). Typically an object with `messages`, `model`, `tools`, and other parameters.                                                     |
| `model_name`        | string or null        | No       | Model identifier set by the caller — e.g., `"nvidia/nemotron-3-super-v3"`.                                                                                                                         |
| `annotated_request` | object or null        | No       | Structured decoded form of `input`. Present only when a codec (e.g., `OpenAIChatCodec`) is registered on this LLM call. Omitted from serialized JSON when null. See Section 5 for the full schema. |


### 3.4 LLMEndEvent

Emitted when an LLM call completes. Carries the post-guardrail-sanitized response payload.


| Field                | Type                  | Required | Description                                                                                                                                                                       |
| -------------------- | --------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `parent_uuid`        | string (UUID) or null | No       | See Section 2.                                                                                                                                                                    |
| `uuid`               | string (UUID)         | Yes      | Same value as the matching `LLMStartEvent`.                                                                                                                                       |
| `timestamp`          | string (RFC 3339)     | Yes      | See Section 2.                                                                                                                                                                    |
| `name`               | string                | Yes      | Same value as the matching `LLMStartEvent`.                                                                                                                                       |
| `data`               | object or null        | No       | See Section 2.                                                                                                                                                                    |
| `metadata`           | object or null        | No       | See Section 2.                                                                                                                                                                    |
| `attributes`         | array of strings      | Yes      | Same value as the matching `LLMStartEvent`. See Section 4.                                                                                                                        |
| `output`             | any or null           | No       | Sanitized LLM response payload (post response-sanitize guardrails).                                                                                                               |
| `model_name`         | string or null        | No       | Same value as on the matching `LLMStartEvent`.                                                                                                                                    |
| `annotated_response` | object or null        | No       | Structured decoded form of `output`. Present only when a response codec is active and decode succeeds. Omitted from serialized JSON when null. See Section 5 for the full schema. |


### 3.5 ToolStartEvent

Emitted when a tool invocation begins.


| Field          | Type                  | Required | Description                                                                                                                                        |
| -------------- | --------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `parent_uuid`  | string (UUID) or null | No       | See Section 2.                                                                                                                                     |
| `uuid`         | string (UUID)         | Yes      | See Section 2. Stable across the matching `ToolStartEvent` and `ToolEndEvent`.                                                                     |
| `timestamp`    | string (RFC 3339)     | Yes      | See Section 2.                                                                                                                                     |
| `name`         | string                | Yes      | The tool function name — e.g., `"calculator__add"`.                                                                                                |
| `data`         | object or null        | No       | See Section 2.                                                                                                                                     |
| `metadata`     | object or null        | No       | See Section 2.                                                                                                                                     |
| `attributes`   | array of strings      | Yes      | Tool behavioral flag names, canonical form (lowercase, sorted, deduplicated). Canonical values: `"local"`. Empty array when no flags are set. See Section 4. |
| `input`        | any or null           | No       | Sanitized tool input arguments (post request-sanitize guardrails).                                                                                 |
| `tool_call_id` | string or null        | No       | Correlation ID from the LLM's tool-call response — e.g., `"call_abc123"` (OpenAI convention). Null for tools invoked outside an LLM tool-use flow. |


### 3.6 ToolEndEvent

Emitted when a tool invocation completes.


| Field          | Type                  | Required | Description                                                                                      |
| -------------- | --------------------- | -------- | ------------------------------------------------------------------------------------------------ |
| `parent_uuid`  | string (UUID) or null | No       | See Section 2.                                                                                   |
| `uuid`         | string (UUID)         | Yes      | Same value as the matching `ToolStartEvent`.                                                     |
| `timestamp`    | string (RFC 3339)     | Yes      | See Section 2.                                                                                   |
| `name`         | string                | Yes      | Same value as the matching `ToolStartEvent`.                                                     |
| `data`         | object or null        | No       | See Section 2.                                                                                   |
| `metadata`     | object or null        | No       | See Section 2.                                                                                   |
| `attributes`   | array of strings      | Yes      | Same value as the matching `ToolStartEvent`. See Section 4.                                      |
| `output`       | any or null           | No       | Sanitized tool result (post response-sanitize guardrails). Null if the tool raised an exception. |
| `tool_call_id` | string or null        | No       | Same value as on the matching `ToolStartEvent`.                                                  |


### 3.7 MarkEvent

Emitted when the application records a named checkpoint in the event stream. Mark is the simplest event type — it has no `attributes`, `scope_type`, `input`, `output`, `model_name`, or `tool_call_id`.


| Field         | Type                  | Required | Description                                                                                                                                |
| ------------- | --------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `parent_uuid` | string (UUID) or null | No       | See Section 2.                                                                                                                             |
| `uuid`        | string (UUID)         | Yes      | See Section 2.                                                                                                                             |
| `timestamp`   | string (RFC 3339)     | Yes      | See Section 2.                                                                                                                             |
| `name`        | string                | Yes      | Name of the marker checkpoint.                                                                                                             |
| `data`        | object or null        | No       | Application-specific payload for this milestone. When present, this value becomes the `message` field of the resulting ATIF `system` step. |
| `metadata`    | object or null        | No       | See Section 2.                                                                                                                             |


---

## 4. Attribute Types

Attribute fields serialize on the wire as a JSON array of lowercase string flag names. The canonical form is sorted and deduplicated — producers MUST emit attributes in lexicographic order with no duplicates, and consumers SHOULD treat the array as an unordered set. Empty `[]` means no flags are set.

**Extensibility.** The flag sets below are the names recognized by this version of the spec, but the set is implementation-defined and open-ended. Implementations MAY emit additional flag names (for vendor extensions or experimental features); to avoid collisions, non-canonical flags SHOULD be namespaced with a dotted prefix — for example, `"nvidia.speculative"`. Consumers MUST preserve unknown flag strings when re-emitting events and MUST NOT treat unknown flags as an error.

### 4.1 ScopeAttributes

Used in `ScopeStartEvent.attributes` and `ScopeEndEvent.attributes`.


| Flag            | Description                                                                 |
| --------------- | --------------------------------------------------------------------------- |
| `"parallel"`    | The scope may execute concurrently with siblings on the scope stack.        |
| `"relocatable"` | The scope may be moved across async task boundaries without losing context. |


Example: a scope that is both parallel and relocatable serializes as `"attributes": ["parallel", "relocatable"]`.

### 4.2 LLMAttributes

Used in `LLMStartEvent.attributes` and `LLMEndEvent.attributes`.


| Flag          | Description                                                                       |
| ------------- | --------------------------------------------------------------------------------- |
| `"stateless"` | The LLM call does not maintain conversation state across invocations.             |
| `"streaming"` | The LLM response is delivered as a stream of chunks rather than a single payload. |


### 4.3 ToolAttributes

Used in `ToolStartEvent.attributes` and `ToolEndEvent.attributes`.


| Flag      | Description                                                                   |
| --------- | ----------------------------------------------------------------------------- |
| `"local"` | The tool executes in the same process as the runtime (not via a remote call). |


---

## 5. Codec Schemas

Codec fields appear on `LLMStartEvent.annotated_request` and `LLMEndEvent.annotated_response`. Both are optional — present only when a codec (e.g., `OpenAIChatCodec`) is registered on the LLM call. Omitted from serialized JSON when null.

### 5.1 AnnotatedLLMRequest

A structured, codec-decoded view of the LLM request. Unmodeled provider-specific keys are flattened into the top level (`serde(flatten)`).


| Field         | Type                            | Required | Description                                                                                                                      |
| ------------- | ------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `messages`    | array of Message                | Yes      | Conversation history. Each entry is discriminated by `role`. See Section 5.2.                                                    |
| `model`       | string or null                  | No       | Model identifier requested by the caller — e.g., `"nvidia/nemotron-3-super-v3"`. Omitted when null.                              |
| `params`      | object or null                  | No       | Normalized generation parameters. See Section 5.6. Omitted when null.                                                            |
| `tools`       | array of ToolDefinition or null | No       | Tool and function schemas available to the model. See Section 5.5. Omitted when null.                                            |
| `tool_choice` | string or object or null        | No       | Tool selection control: `"auto"`, `"none"`, `"required"`, or a `ToolChoiceFunction` object. Omitted when null. See Section 5.11. |

Any additional provider-specific keys not listed above appear as top-level fields in the serialized JSON (`serde(flatten)`). These are preserved for lossless round-trip through codec encode/decode but are not part of the normalized schema.

### 5.2 Message (discriminated by `role`)

Each message in the `messages` array has a `role` field that acts as a discriminator. Additional fields depend on role.


| Role          | Additional Fields                                                                                                                                                             |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"system"`    | `content`: string or array of ContentPart (see 5.3); `name`: string or null (omitted when null)                                                                               |
| `"user"`      | `content`: string or array of ContentPart; `name`: string or null (omitted when null)                                                                                         |
| `"assistant"` | `content`: string, array of ContentPart, or null (omitted when null); `tool_calls`: array of ToolCall or null (omitted when null); `name`: string or null (omitted when null) |
| `"tool"`      | `content`: string or array of ContentPart; `tool_call_id`: string (required)                                                                                                  |


### 5.3 ContentPart (discriminated by `type`)

Used inside `messages[*].content` when the content is an array (multimodal). The `type` field is the discriminator.


| Type     | Additional Fields                              |
| -------- | ---------------------------------------------- |
| `"text"` | `text`: string — the text content of this part |


### 5.4 ToolCall (request-side)

Represents a tool call in the `assistant` message's `tool_calls` array. Note that `arguments` is a JSON string on the request side (OpenAI wire convention).


| Field      | Type   | Required | Description                                                                                             |
| ---------- | ------ | -------- | ------------------------------------------------------------------------------------------------------- |
| `id`       | string | Yes      | Unique identifier for this tool call.                                                                   |
| `type`     | string | Yes      | Type of call — typically `"function"`.                                                                  |
| `function` | object | Yes      | A `FunctionCall` object with `name` (string) and `arguments` (string — raw JSON per OpenAI convention). |


**Key distinction:** `ToolCall.function.arguments` here is a **JSON string** (e.g., `"{\"a\": 3, \"b\": 4}"`). The response-side `ResponseToolCall.arguments` (Section 5.8) is a **parsed JSON object**.

### 5.5 ToolDefinition

Describes a tool or function available to the model.


| Field      | Type   | Required | Description                                                                                                                                                                     |
| ---------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`     | string | Yes      | Type of tool — typically `"function"`.                                                                                                                                          |
| `function` | object | Yes      | A `FunctionDefinition` object with: `name` (string, required), `description` (string or null, omitted when null), `parameters` (JSON Schema object or null, omitted when null). |


### 5.6 GenerationParams

Normalized generation parameters, shared across providers.


| Field         | Type                    | Required | Description                                                  |
| ------------- | ----------------------- | -------- | ------------------------------------------------------------ |
| `temperature` | number or null          | No       | Sampling temperature. Omitted when null.                     |
| `max_tokens`  | integer or null         | No       | Maximum number of tokens to generate. Omitted when null.     |
| `top_p`       | number or null          | No       | Nucleus sampling probability threshold. Omitted when null.   |
| `stop`        | array of string or null | No       | Stop sequences that terminate generation. Omitted when null. |


### 5.7 AnnotatedLLMResponse

A structured, codec-decoded view of the LLM response. Unmodeled top-level fields are flattened into the top level (`serde(flatten)`).


| Field           | Type                                   | Required | Description                                                                                                                 |
| --------------- | -------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------- |
| `id`            | string or null                         | No       | Response ID from the provider — e.g., `"chatcmpl-abc123"`. Omitted when null.                                               |
| `model`         | string or null                         | No       | Model that actually served the request (may differ from the requested model). Omitted when null.                            |
| `message`       | string or array of ContentPart or null | No       | The assistant's response content. Omitted when null.                                                                        |
| `tool_calls`    | array of ResponseToolCall or null      | No       | Tool calls requested by the model, normalized across APIs. Omitted when null. See Section 5.8.                              |
| `finish_reason` | string or null                         | No       | Normalized stop reason. One of: `"complete"`, `"length"`, `"tool_use"`, `"content_filter"`, `"unknown"`. Omitted when null. |
| `usage`         | object or null                         | No       | Token usage statistics. See Section 5.9. Omitted when null.                                                                 |
| `api_specific`  | object or null                         | No       | Provider-specific fields that cannot be normalized. See Section 5.10. Omitted when null.                                    |

Any additional provider-specific keys not listed above appear as top-level fields in the serialized JSON (`serde(flatten)`). These are preserved for lossless round-trip through codec encode/decode but are not part of the normalized schema.

### 5.8 ResponseToolCall (response-side)

Represents a tool call in the response's `tool_calls` array. Note that `arguments` is parsed JSON on the response side (codec-normalized).


| Field       | Type   | Required | Description                                                                                     |
| ----------- | ------ | -------- | ----------------------------------------------------------------------------------------------- |
| `id`        | string | Yes      | Unique identifier for this tool call.                                                           |
| `name`      | string | Yes      | The function or tool name.                                                                      |
| `arguments` | object | Yes      | Parsed JSON arguments — **NOT a string**. Codecs parse OpenAI's string arguments during decode. |


**Key distinction:** Request-side `ToolCall.function.arguments` is a JSON string. Response-side `ResponseToolCall.arguments` is a parsed JSON object. This asymmetry matches the OpenAI API convention but is normalized by codecs on the response side.

### 5.9 Usage

Token usage statistics from the LLM API response. All fields are optional because not every provider reports every counter.


| Field                | Type            | Required | Description                                     |
| -------------------- | --------------- | -------- | ----------------------------------------------- |
| `prompt_tokens`      | integer or null | No       | Tokens consumed by the prompt input.            |
| `completion_tokens`  | integer or null | No       | Tokens generated in the completion output.      |
| `total_tokens`       | integer or null | No       | Total tokens (prompt plus completion).          |
| `cache_read_tokens`  | integer or null | No       | Tokens served from the provider's prompt cache. |
| `cache_write_tokens` | integer or null | No       | Tokens written to the provider's prompt cache.  |


### 5.10 ApiSpecificResponse (discriminated by `api`)

Holds provider-specific response fields that cannot be normalized across APIs. The `api` field is the discriminator.


| `api` value            | Additional Fields                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| `"openai_chat"`        | `logprobs` (object or null), `system_fingerprint` (string or null), `service_tier` (string or null) |
| `"openai_responses"`   | `output_items` (array or null), `status` (string or null), `incomplete_details` (object or null)    |
| `"anthropic_messages"` | `stop_sequence` (string or null), `content_blocks` (array or null)                                  |
| `"custom"`             | `api_name` (string — the custom API identifier), `data` (any object — opaque API-specific payload)  |


### 5.11 ToolChoice Serialization

The `tool_choice` field on `AnnotatedLLMRequest` has four possible wire forms:

| Value | JSON | Meaning |
| ----- | ---- | ------- |
| Auto | `"auto"` | Model decides whether to call a tool |
| None | `"none"` | Model must not call any tool |
| Required | `"required"` | Model must call some tool (but can pick which) |
| Specific | `{"type": "function", "function": {"name": "calculator__add"}}` | Model must call this specific function |

The first three serialize as plain JSON strings. The `Specific` variant serializes as a `ToolChoiceFunction` object with fields: `type` (string, typically `"function"`) and `function` (object with `name` string).

---

## 6. Event Stream Semantics

### 6.1 Timestamp Ordering

Events are emitted in wall-clock order. However, delivery order from subscriber callbacks may differ for concurrent operations. Consumers MUST sort by `timestamp` before processing. `AtifExporter.events_to_steps()` always sorts collected events by timestamp as its first step.

### 6.2 Scope Nesting and parent_uuid

The runtime maintains a scope stack per async task. The `parent_uuid` of any event is the UUID of the scope that was on top of the stack when the handle was created. Following `parent_uuid` links upward reconstructs the full call graph.

The root scope has `parent_uuid = null`. This is the only event in a well-formed stream that may have a null `parent_uuid` (once the root scope is established).

### 6.3 Start/End Pairing

Every Start event is paired with exactly one End event sharing the same `uuid`. End events always arrive after their matching Start events in wall-clock order. All child events (events whose `parent_uuid` equals this scope's `uuid`) will have been emitted before the parent's End event fires.

### 6.4 UUID Uniqueness

Each handle (scope, tool invocation, LLM call) receives a unique UUID at creation time. The `uuid` is stable across the Start and End events for the same handle, enabling correlation. In the Rust implementation, UUIDs are v7 (time-ordered).

### 6.5 ID Relationships

Four distinct identifier types appear in ATOF events. They serve different purposes and belong to different namespaces. The following diagram shows how they relate within a single tool-call cycle (EXMP-01):

```text
┌─ ScopeStart ──────────────────────────────────────────────── ScopeEnd -─┐
│  uuid: "scope-001"                                                      │
│  parent_uuid: null                                                      │
│                                                                         │
│  ┌─ LLMStart ─────────────────────-─ LLMEnd ─┐                          │
│  │  uuid: "llm-001"                          │                          │
│  │  parent_uuid: "scope-001"  ───────────────┼──► scope graph edge      │
│  │                                           │                          │
│  │  output.tool_calls[0].id ─────────────────┼──► "call_calc_001"       │
│  │  annotated_response.id ───────────────────┼──► "chatcmpl-abc123"     │
│  └───────────────────────────────────────────┘                          │
│                                                                         │
│  ┌─ ToolStart ──────────────────── ToolEnd ──┐                          │
│  │  uuid: "tool-001"          (≠ "llm-001")  │  ← own handle identity   │
│  │  parent_uuid: "scope-001"  ───────────────┼──► scope graph edge      │
│  │  tool_call_id: "call_calc_001" ───────────┼──► LLM correlation       │
│  └───────────────────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘

uuid           → "Who am I?"          (handle identity, Start=End)
parent_uuid    → "Who created me?"    (scope stack lineage, call graph)
tool_call_id   → "Which LLM request?" (LLM↔tool correlation, OpenAI convention)
response.id    → "Which API call?"    (provider billing/tracking, ephemeral)
```

**Key distinctions:**

- `uuid` and `parent_uuid` are agent runtime identifiers. Every event has them. They form the scope graph.
- `tool_call_id` is an LLM-provider identifier that bridges the LLM's tool-call request (`LLMEnd.output.tool_calls[].id`) with the tool execution (`ToolStart.tool_call_id`). It is null for tools invoked outside an LLM tool-use flow.
- `annotated_response.id` is a provider tracking identifier (e.g., OpenAI's `chatcmpl-*`). It has no relationship to the other three ID types and exists only inside codec-decoded responses.

---

## 7. Canonical ATOF-to-ATIF Mapping

This section formalizes the mapping from ATOF events to ATIF steps. The Toolkit implementation is in [`src/nat/atof/scripts/atof_to_atif_converter.py`](src/nat/atof/scripts/atof_to_atif_converter.py).


| ATOF Event         | ATIF Step            | ATIF `source` | Content Mapping                                                                                                                                                                                                                                                                                                                                          |
| ------------------ | -------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LLMStart`         | user step            | `"user"`      | `step.message` = `messages` array extracted from `input` (stripping `model`, `tools`, and other LLM config fields). `step.timestamp` = event timestamp.                                                                                                                                                                                                  |
| `LLMEnd`           | agent step           | `"agent"`     | `step.message` = extracted `content` from response output. `step.tool_calls` = promoted from response `tool_calls` array (with arguments parsed from JSON string to JSON object). `step.metrics` = extracted from `token_usage`. `step.extra.ancestry`, `invocation`, `tool_ancestry` are deferred — written when the next `LLMStart` arrives (see 7.1). |
| `ToolStart`        | embedded in agent step | `"agent"`   | Already captured in the preceding `LLMEnd` agent step's `tool_calls[]` array. `ToolStart` adds no new information — the function name, arguments, and `tool_call_id` are all present in the LLM response.                                                                                                                                                |
| `ToolEnd`          | buffered observation | —             | Buffered as an `AtifObservationResult`. Consecutive `ToolEnd` events within the same turn are merged into a single `system` step. `source_call_id` is correlated using two strategies: (a) explicit `tool_call_id` on the event, (b) function-name lookup against the preceding `LLMEnd`'s promoted `tool_calls`.                                        |
| `Mark` (with data) | system step          | `"system"`    | `step.message` = event `data`.                                                                                                                                                                                                                                                                                                                           |
| `ScopeStart`       | embedded in `parent_uuid` graph | —  | Structural event used for scope-graph reconstruction (Section 6.2). No ATIF step is produced; ancestry information is captured in `step.extra.ancestry` and `step.extra.tool_ancestry` on the agent step.                                                                                                                                                |
| `ScopeEnd`         | embedded in `parent_uuid` graph | —  | Structural event closing the scope opened by the matching `ScopeStart`. No ATIF step is produced; timing information is captured in `step.extra.invocation` on the agent step.                                                                                                                                                                           |


### 7.1 Observation Flush Rule

Buffered `ToolEnd` observations are flushed (emitted as a single `system` step) when any of the following conditions occur:

- The next `LLMStart` event arrives (closing the previous turn).
- The next `Mark` event arrives.
- End of event stream.

Consecutive `ToolEnd` events within the same turn are merged into a single observation step with multiple results.

### 7.2 Key Design Decisions

**ToolStart is skipped.** Tool calls are promoted from the `LLMEnd` response, which is the authoritative source. Recording `ToolStart` separately would duplicate entries already in the agent step's `tool_calls` array.

**Consecutive ToolEnd events merge.** When an LLM dispatches multiple tools in parallel, their observations arrive as separate `ToolEnd` events but belong to a single logical response turn. The exporter merges them into one `system` step with multiple observation results.

**Two-strategy tool_call_id correlation.** (a) Explicit: `ToolEnd.tool_call_id` matches a `tool_call_id` in the preceding `LLMEnd`'s promoted `tool_calls` — used directly as `source_call_id`. (b) Function-name fallback: if `tool_call_id` is absent, `ToolEnd.name` is matched against the `function_name` of the preceding `LLMEnd`'s promoted `tool_calls`.

**Deferred step.extra.** When `LLMEnd` arrives, `step.extra` cannot be written immediately because `ToolEnd` ancestry records accumulate afterward. The exporter defers writing `step.extra` (ancestry, invocation, tool_ancestry) until the next `LLMStart` arrives — or at end of stream — when all tool ancestry is known.

**Timestamp representations.** ATIF `step.timestamp` is ISO 8601 UTC (e.g., `"2026-01-01T00:00:00Z"`). `AtifInvocationInfo.start_timestamp` and `end_timestamp` (inside `step.extra.invocation`) are epoch seconds (float, e.g., `1735689600.0`). These are distinct representations for distinct purposes.

---

## 8. What ATOF Is Not

ATOF events are not ATIF steps. The distinctions are structural:


| Property          | ATOF events                                                                | ATIF steps                                                                  |
| ----------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Start/End pairing | Un-merged: `LLMStart` and `LLMEnd` are separate events                     | Merged: a single agent step captures the full LLM call                      |
| Sequencing        | No `step_id`; ordered only by `timestamp`                                  | Sequential 1-based `step_id`; no gaps allowed                               |
| Source field      | No `source` discriminator                                                  | Required field: `"user"`, `"agent"`, or `"system"`                          |
| Tool ancestry     | No per-step `tool_ancestry`; only `parent_uuid` for scope-graph navigation | `step.extra.tool_ancestry[]` aligns by index with `step.tool_calls[]`       |
| Observations      | `ToolEnd` events carry raw output independently                            | Consecutive `ToolEnd` events merged into a single `system` step             |
| Computed fields   | None                                                                       | `step_id` assigned sequentially; `final_metrics` computed from step metrics |
| Scope events      | `ScopeStart`/`ScopeEnd` exist in the stream                                | Scope events produce no ATIF steps                                          |


**ATOF does not have:**

- `step_id`
- `source` field
- Merged observations
- `tool_ancestry` per step
- `schema_version`
- Sequential guarantees beyond timestamp ordering

---

## 9. EXMP-01: Simple Tool Call

A minimal 6-event stream illustrating one complete tool call cycle. Each line is one JSON object.

```jsonl
{"kind":"ScopeStart","uuid":"scope-agent-001","parent_uuid":null,"timestamp":"2026-01-01T00:00:00Z","name":"simple_calculator_agent","scope_type":"agent","attributes":[],"data":null,"metadata":null}
{"kind":"LLMStart","uuid":"llm-001","parent_uuid":"scope-agent-001","timestamp":"2026-01-01T00:00:01Z","name":"nvidia/nemotron-3-super-v3","attributes":[],"input":{"messages":[{"role":"user","content":"What is 3 + 4?"}],"model":"nvidia/nemotron-3-super-v3","tools":[{"type":"function","function":{"name":"calculator__add","description":"Add two numbers","parameters":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}}}}}]},"model_name":"nvidia/nemotron-3-super-v3","data":null,"metadata":null}
{"kind":"LLMEnd","uuid":"llm-001","parent_uuid":"scope-agent-001","timestamp":"2026-01-01T00:00:02Z","name":"nvidia/nemotron-3-super-v3","attributes":[],"output":{"content":"The result of 3 + 4 is 7.","tool_calls":[{"id":"call_calc_001","type":"function","function":{"name":"calculator__add","arguments":"{\"a\": 3, \"b\": 4}"}}]},"model_name":"nvidia/nemotron-3-super-v3","data":null,"metadata":null}
{"kind":"ToolStart","uuid":"tool-001","parent_uuid":"scope-agent-001","timestamp":"2026-01-01T00:00:03Z","name":"calculator__add","attributes":[],"input":{"a":3,"b":4},"tool_call_id":"call_calc_001","data":null,"metadata":null}
{"kind":"ToolEnd","uuid":"tool-001","parent_uuid":"scope-agent-001","timestamp":"2026-01-01T00:00:04Z","name":"calculator__add","attributes":[],"output":7,"tool_call_id":"call_calc_001","data":null,"metadata":null}
{"kind":"ScopeEnd","uuid":"scope-agent-001","parent_uuid":null,"timestamp":"2026-01-01T00:00:05Z","name":"simple_calculator_agent","scope_type":"agent","attributes":[],"data":null,"metadata":null}
```

**Resulting ATIF output:** From this 6-event stream, the converter produces 3 ATIF steps: (1) a user step from `LLMStart` with `message = [{"role": "user", "content": "What is 3 + 4?"}]`; (2) an agent step from `LLMEnd` with `message = "The result of 3 + 4 is 7."` and promoted `tool_calls`; (3) a system step from the buffered `ToolEnd` observation with `source_call_id = "call_calc_001"` and `content = "7"` (stringified from the integer output). The observation flush occurs at end of stream (no subsequent `LLMStart`).

**Note on event ordering:** `LLMEnd` arrives at `t=02` before `ToolStart` at `t=03`. This is the correct order: the LLM decides to call the tool (emitting `LLMEnd` with `tool_calls` in `output`), then the runtime dispatches the tool call. The exporter sorts by timestamp, so the ordering `LLMEnd → ToolStart → ToolEnd` is required for correct correlation.

---

## 10. Reference Implementations

### 10.1 NeMo Agent Toolkit Python (this package)

The Toolkit-native ATOF-to-ATIF converter is in [`src/nat/atof/scripts/atof_to_atif_converter.py`](src/nat/atof/scripts/atof_to_atif_converter.py). It implements the same accumulator pattern as the Rust reference:

1. Sorts all events by timestamp.
2. Runs a pre-pass to build `uuid → name` and `uuid → start_timestamp` maps for ancestry and invocation-info resolution.
3. Iterates through sorted events, emitting ATIF steps per the mapping in Section 7.
4. Implements the deferred `step.extra` write pattern: agent step extra is written when the next `LLMStart` arrives (or at end of stream), after all `ToolEnd` ancestry records have been accumulated.
5. Sorts `tool_ancestry` and `tool_invocations` by `tool_call_id` declaration order from `LLMEnd` (not by `ToolEnd` arrival order), ensuring index alignment with `tool_calls[]` even for concurrent tool execution.
6. Unwraps the transport envelope (`{"content": ..., "headers": ...}`) from `LLMRequest` payloads before extracting `messages` for the user step.

| Module | Entry Point | Description |
| ------ | ----------- | ----------- |
| `nat.atof.scripts.atof_to_atif_converter` | `convert(events) → Trajectory` | Convert typed Event list to ATIF Trajectory |
| `nat.atof.scripts.atof_to_atif_converter` | `convert_file(path) → Trajectory` | Read JSONL file and convert |
| `nat.atof.io` | `read_jsonl(path) → list[Event]` | Parse ATOF JSONL to typed events |
| `nat.atof.io` | `write_jsonl(events, path)` | Serialize events to JSONL |


