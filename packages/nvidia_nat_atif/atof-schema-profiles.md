# ATOF Schema Profiles

**Version:** 0.1
**Companion to:** `[atof-event-format.md](./atof-event-format.md)` (ATOF core wire format)

---

## 1. Purpose

ATOF Core (`atof-event-format.md`) carries provider-specific payloads as opaque JSON in `input` and `output`. For consumers that want structured access — replay tools, evaluators, audit pipelines — **schema profiles** define canonical structured shapes that a producer can decode raw provider JSON into and attach as `annotated_request` on `ScopeStartEvent` and `annotated_response` on `ScopeEndEvent` (applicable when `scope_type == "llm"` or `scope_type == "tool"`).

This document defines:

- The **schema identifier** (`schema: {name, version}`) convention
- The **canonical schema registry** — spec-defined schema IDs that consumers can expect
- **Reference JSON Schemas** describing each profile's structured shape (as out-of-band validation contracts — NOT required on the wire)

Schema profiles are the **tier-3 enrichment layer** of ATOF's three-tier producer model (see core §1.1). They are entirely optional: a producer without a schema emits `schema: null` and `annotated_request: null`; a consumer without schema knowledge ignores the field entirely.

## 2. Scope

This document covers:

- The schema identifier wire format
- Canonical schema IDs for OpenAI Chat Completions, Anthropic Messages, NVIDIA NIM, and the passthrough schema
- Structured shapes (`AnnotatedLlmRequest`, `AnnotatedLlmResponse`, `AnnotatedToolInvocation`) and their JSON Schemas
- Schema versioning and registry rules

This document does NOT cover:

- The ATOF wire format — see `atof-event-format.md`
- ATIF conversion — see `atof-to-atif-converter.md`
- Producer-side decoding implementation (typed adapter patterns in specific languages) — out of scope for this spec; consult producer-runtime documentation

## 3. Schema Identifier

The `schema` field on `ScopeStartEvent` and `ScopeEndEvent` (when `scope_type` is `"llm"` or `"tool"`) is an object:

```json
{"name": "openai/chat-completions", "version": "v1"}
```


| Field     | Type   | Required | Description                                                       |
| --------- | ------ | -------- | ----------------------------------------------------------------- |
| `name`    | string | Yes      | Canonical schema ID — lowercase, slashed namespace + name.        |
| `version` | string | Yes      | Schema version — `"v1"`, `"v2"`, ... Independent of ATOF version. |


**When present**, `schema` declares which registered schema shapes the `annotated_request` / `annotated_response` payload. Consumers MAY use the schema ID to look up the structured shape in a registry and dispatch accordingly.

**When absent** (`schema: null`), the event is tier-1 or tier-2 — no structured schema annotation is carried, and any `annotated_request` / `annotated_response` values on the event MUST be null.

### 3.1 Schema ID namespace

Schema IDs follow the `<vendor>/<schema_name>` format:

- `default/` — ATOF-spec-defined schemas. Stable; changes are MAJOR-bumped.
- `openai/` — OpenAI provider schemas.
- `anthropic/` — Anthropic provider schemas.
- `nvidia/` — NVIDIA provider schemas (NIM, Nemotron, ...).
- `<vendor>/` — any other vendor. Vendors SHOULD namespace their schemas to avoid collisions.

**Reserved schema IDs.** `default/passthrough` and `default/opaque` are reserved. `default/passthrough` is a no-op schema that declares `annotated_request == input` (structured shape identical to raw shape). `default/opaque` is the explicit signal that the producer does not decode the payload (equivalent to `schema: null`, but machine-declared for audit trails).

### 3.2 Schema versioning

Schema versions are INDEPENDENT of the ATOF `atof_version`. An ATOF v0.1 stream can carry any schema version. A schema version bump (`v1` → `v2`) signals a breaking change to the structured shape for that specific schema; consumers dispatch on `schema.version` within each schema namespace.

A stream MAY contain events referencing multiple schemas (an agent using both OpenAI and Anthropic models would emit events with two different `schema.name` values).

## 4. Structured Shapes (High-Level)

Three structured shapes are canonical across schema profiles:

- `**AnnotatedLlmRequest`** — structured view of an LLM call's request (messages, model, params, tools, tool_choice).
- `**AnnotatedLlmResponse**` — structured view of an LLM call's response (choices, usage, finish_reason, tool_calls).
- `**AnnotatedToolInvocation**` — structured view of a tool call's arguments and result (only relevant when tools are invoked through a structured protocol that carries more than raw JSON).

The `**default/passthrough` schema** declares these shapes as the ATOF-canonical form. Provider schemas (`openai/chat-completions`, `anthropic/messages`, …) MAY extend or specialize them.

### 4.1 `AnnotatedLlmRequest` (canonical)

```
{
  "messages": [Message, ...],             # required
  "model": string | null,                 # optional
  "params": GenerationParams | null,      # optional
  "tools": [ToolDefinition, ...] | null,  # optional
  "tool_choice": ToolChoice | null,       # optional
  ...provider extras (flattened)          # preserved verbatim
}
```

`Message`:

```
{"role": "system",    "content": MessageContent, "name": string | null}
{"role": "user",      "content": MessageContent, "name": string | null}
{"role": "assistant", "content": MessageContent | null, "tool_calls": [ToolCall, ...] | null, "name": string | null}
{"role": "tool",      "content": MessageContent, "tool_call_id": string}
```

`MessageContent` is either a plain text `string` or an array of typed content parts (`[{"type": "text", "text": "..."}, ...]`) for multimodal support.

`GenerationParams` captures sampling / generation knobs: `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `response_format`, `reasoning_effort`, `parallel_tool_calls`, `stream`, `user`, `modalities`. All optional.

`ToolDefinition`: `{"type": "function", "function": {"name": "...", "description": "...", "parameters": <JSON Schema>}}`. Mirrors OpenAI Chat Completions tool definition format.

`ToolChoice`: `"none" | "auto" | "required" | {"type": "function", "function": {"name": "..."}}`.

**Extra fields.** Any provider-specific fields not modeled above are preserved verbatim (serde flatten on the producer side). Consumers MUST preserve unknown keys on re-emission.

### 4.2 `AnnotatedLlmResponse` (canonical)

```
{
  "choices": [Choice, ...],               # required
  "model": string | null,                 # optional — actually-used model
  "usage": Usage | null,                  # optional
  "id": string | null,                    # optional — provider response ID
  "system_fingerprint": string | null,    # optional
  "service_tier": string | null,          # optional
  ...provider extras (flattened)          # preserved verbatim
}
```

`Choice`:

```
{
  "index": int,
  "message": {
    "role": "assistant",
    "content": string | null,
    "refusal": string | null,
    "tool_calls": [ToolCall, ...] | null,
    "audio": {...} | null
  },
  "finish_reason": "stop" | "length" | "tool_calls" | "content_filter" | "refusal" | null,
  "logprobs": {...} | null
}
```

`Usage`:

```
{
  "prompt_tokens": int | null,
  "completion_tokens": int | null,
  "total_tokens": int | null,
  "prompt_tokens_details": {"cached_tokens": int | null, "audio_tokens": int | null} | null,
  "completion_tokens_details": {"reasoning_tokens": int | null, "accepted_prediction_tokens": int | null, "rejected_prediction_tokens": int | null, "audio_tokens": int | null} | null
}
```

`ToolCall`: `{"id": string, "type": "function", "function": {"name": string, "arguments": object | string}}`.

### 4.3 `AnnotatedToolInvocation` (canonical)

```
{
  "tool_call_id": string | null,  # correlation ID from the LLM tool-call response
  "function_name": string | null, # tool/function name
  "arguments": object | null,     # parsed JSON argument dict
  "result": any | null,           # tool return value (only on ScopeEnd with scope_type="tool")
  "tool_type": string | null,     # "function" | "web_search" | "file_search" | "code_interpreter" | "computer" | "mcp" | "retrieval" | "custom"
  "tool_provider": string | null, # "langchain" | "mcp:filesystem" | "native" | ...
  "parent_tool_call_id": string | null  # for nested tools
}
```

## 5. Canonical Schema Registry

The following schema IDs are spec-defined. Reference JSON Schemas live in the `atof-profile/` directory alongside this document.


| Schema ID                 | Version | Purpose                                                                   | Schema file                                              |
| ------------------------- | ------- | ------------------------------------------------------------------------- | -------------------------------------------------------- |
| `default/passthrough`     | v1      | Structured shape IS the raw shape; no decoding.                           | (no schema — shape = raw `input`/`output`)               |
| `default/opaque`          | v1      | Explicit signal: producer does not decode. Annotations MUST be null.      | (no schema)                                              |
| `openai/chat-completions` | v1      | OpenAI Chat Completions API request/response.                             | (future: `atof-profile/openai-chat-completions-v1.json`) |
| `anthropic/messages`      | v1      | Anthropic Messages API request/response.                                  | (future: `atof-profile/anthropic-messages-v1.json`)      |
| `nvidia/llm`              | v1      | NVIDIA-namespaced LLM profile (canonical ATOF shape + NVIDIA extensions). | `atof-profile/atof-profile-nvidia_llm_v1.json`           |
| `nvidia/tools`            | v1      | NVIDIA-namespaced tool invocation profile.                                | `atof-profile/atof-profile-nvidia_tools_v1.json`         |


`**nvidia/llm` and `nvidia/tools`.** These schemas describe NVIDIA-specific LLM and tool event shapes. The `nvidia/llm` schema declares the expected shape for `annotated_request` / `annotated_response` on an NVIDIA LLM scope, extending the canonical `AnnotatedLlmRequest` / `AnnotatedLlmResponse` with NVIDIA-specific fields (provider, completion_id, system_fingerprint, full usage breakdown, etc.).

### 5.1 Schema file format

Each schema's reference JSON Schema is stored under `atof-profile/` with filename convention `<vendor>-<schema_name>-v<N>.json` (e.g., `openai-chat-completions-v1.json`). Current filenames use underscores for filesystem friendliness (`atof-profile-nvidia_llm_v1.json`); this is cosmetic and the logical schema ID uses slashes.

Each schema:

- Declares `$id` matching the schema ID (e.g., `"$id": "nvidia/llm.v1"`)
- Uses JSON Schema Draft 2020-12
- Sets `additionalProperties: true` at every level — vendors MAY extend
- All fields optional — producers emit only what they know

### 5.2 When to add a schema

A new schema ID is appropriate when:

- The provider API has a distinct request/response envelope shape that producers in the ecosystem regularly emit
- The shape is stable enough to warrant a versioned schema
- At least one concrete producer is using it (not speculative)

Proposed new schemas go through spec review. Schema IDs are effectively a public contract — renaming or removing one is a MAJOR-level break.

## 6. Schema Resolution Protocol

When a consumer wants structured access to `annotated_request` or `annotated_response`, it resolves the schema via a four-priority chain. Each priority is independently optional; events and streams missing higher-priority sources fall through cleanly to lower-priority ones, ending at opaque pass-through if nothing resolves.

### 6.1 The Four-Priority Chain


| Priority                         | Source                                           | Condition                                                                                                                         | Behavior                                                                                               |
| -------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **1. Per-event inline**          | `event.schema["$schema"]`                        | Event has `schema` with an inline `$schema` body                                                                                  | Use the inline body directly; no lookup needed                                                         |
| **2. StreamHeader registry**     | `stream_header.schemas[canonical_id]["$schema"]` | Event has `schema` (name + version) and the stream's `StreamHeaderEvent` registry contains a matching entry with inline `$schema` | Use the header's schema body                                                                           |
| **3. Consumer-bundled registry** | `consumer.local_registry[canonical_id]`          | Event has `schema` (name + version); neither priorities 1 nor 2 provided a body                                                   | Look up the canonical ID in the consumer's locally-bundled schema library                              |
| **4. Opaque pass-through**       | —                                                | Event has no `schema` field at all, OR priorities 1-3 all failed                                                                  | Preserve `annotated_request` / `annotated_response` verbatim as opaque dicts; no structured validation |


**Key semantics:**

- The **canonical ID** is the string `{schema.name}.v{schema.version}` — e.g., `"openai/chat-completions.v1"`, `"nvidia/llm.v1"`. Matching across the four priorities is by this canonical ID.
- The `schema` declaration on the event is what **triggers** the resolution chain. Events with no `schema` field skip priorities 1-3 entirely and land at priority 4.
- An event MAY declare a schema ID without inline body (`schema: {"name": "...", "version": "..."}`) — the chain will find the body via priority 2 or 3.
- Consumers MUST NOT reject events at priority 4. Opaque pass-through is always legal.

### 6.2 Canonical ID Naming Convention

Canonical schema IDs follow the `{vendor}/{schema_name}.v{MAJOR}` format (see §3.1). The `.v{MAJOR}` suffix is part of the matching key, so bumping a schema from `v1` → `v2` creates a distinct registry entry; events declaring `v1` continue to resolve to the v1 body.

The priority-3 local registry (consumer-bundled) uses the **same keying convention**, so an event declaring `schema: {"name": "openai/chat-completions", "version": "v1"}` looks up `"openai/chat-completions.v1"` at every priority level. Consumer registries that follow this convention are automatically compatible with the resolution chain.

### 6.3 Resolution Algorithm

```python
def resolve_annotation_schema(event, stream_header, consumer_registry):
    """Return the JSON Schema for validating an event's annotated_* payload,
    or None for opaque pass-through (priority 4).
    """
    if event.get("schema") is None:
        return None  # Priority 4: no schema declared → opaque

    schema_ref = event["schema"]
    canonical_id = f"{schema_ref['name']}.v{schema_ref['version']}"

    # Priority 1: inline schema body on the event itself
    inline = schema_ref.get("$schema")
    if inline is not None:
        return inline

    # Priority 2: StreamHeader registry with inline body
    if stream_header is not None:
        header_schemas = stream_header.get("schemas", {})
        entry = header_schemas.get(canonical_id)
        if entry is not None and entry.get("$schema") is not None:
            return entry["$schema"]

    # Priority 3: consumer-bundled canonical registry
    if canonical_id in consumer_registry:
        return consumer_registry[canonical_id]

    # Priority 4: opaque
    return None
```

### 6.4 Interaction with `scope_type`

When schema resolution succeeds (priorities 1-3), validation is grounded in the schema body and `scope_type` becomes a lightweight label — the schema is the source of truth for payload shape.

When resolution falls through to opaque (priority 4), `scope_type` does more semantic heavy-lifting: it is the only spec-defined signal about what kind of work the event represents. A consumer inspecting an opaque event relies on `scope_type == "llm"` or `scope_type == "tool"` conventions to interpret the raw `input` / `output`.

### 6.5 Consumer Resilience

- **Unknown canonical ID (priorities 2-3 both miss):** preserve `annotated_`* verbatim; fall through to priority 4; MUST NOT reject the event.
- `**StreamHeader` missing but events declare schemas:** priority 2 is skipped silently; chain proceeds to priorities 3-4 as usual.
- `**StreamHeader` present but declares an unknown schema (empty `{}`, no inline body, consumer registry doesn't have it):** consumer MAY log a manifest-warning at stream open; events using that schema still fall through to priority 4.
- **Schema declared without inline body anywhere in the resolution chain:** priority 4 (opaque). The `annotated_`* payload is preserved, but not validated.

## 7. Producer Guidance

### 7.1 Emitting tier-1 events

Tier-1 producers don't have a schema. Emit `ScopeStart` / `ScopeEnd` with:

```json
{"kind":"ScopeStart", "scope_type":"llm", "schema":null, "input":<raw_provider_json>, "annotated_request":null, ...}
```

Consumers fall back to the raw payload.

### 7.2 Emitting tier-2 events

Tier-2 producers know the event is an LLM call and can provide `profile.model_name` + `attributes`, but don't decode the provider shape. Same as tier 1 — `schema: null`, `annotated_request: null`.

### 7.3 Emitting tier-3 events

Tier-3 producers have a schema registered. For every LLM call:

1. Decode the raw provider JSON into the schema's structured shape.
2. Attach the decoded object as `annotated_request` on `ScopeStartEvent` (`scope_type: "llm"`).
3. Declare the schema via `schema: {name: "<id>", version: "v<N>"}`.
4. Preserve the raw provider JSON in `input` alongside the decoded structure.
5. Repeat for the response on `ScopeEndEvent` (`annotated_response`).

The raw `input`/`output` is always preserved for round-trip fidelity. The annotation is a consumer convenience, not a replacement.

### 7.4 Validation (optional)

Producer-side validation: producers MAY validate decoded structures against the reference schema before emitting. The reference implementation decodes via type system and skips runtime JSON-Schema validation.

Consumer-side validation: consumers that care about schema conformance MAY validate `annotated_request` / `annotated_response` against the reference schema. Validation failures SHOULD be logged but MUST NOT cause events to be dropped (replay-for-inspection requires lossless propagation).

## 8. Examples

### 8.1 Tier-3 OpenAI Chat Completions call

```jsonl
{"kind":"ScopeStart","atof_version":"0.1","uuid":"llm-005","parent_uuid":"agent-005","timestamp":"2026-01-05T00:00:01Z","name":"gpt-4.1","attributes":[],"scope_type":"llm","profile":{"model_name":"gpt-4.1"},"schema":{"name":"openai/chat-completions","version":"v1"},"input":{"model":"gpt-4.1","messages":[{"role":"user","content":"hello"}],"temperature":0.7},"annotated_request":{"model":"gpt-4.1","messages":[{"role":"user","content":"hello"}],"params":{"temperature":0.7}},"data":null,"metadata":null}
{"kind":"ScopeEnd","atof_version":"0.1","uuid":"llm-005","parent_uuid":"agent-005","timestamp":"2026-01-05T00:00:03Z","name":"gpt-4.1","attributes":[],"scope_type":"llm","profile":{"model_name":"gpt-4.1"},"schema":{"name":"openai/chat-completions","version":"v1"},"output":{"id":"chatcmpl-abc","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10}},"annotated_response":{"id":"chatcmpl-abc","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10}},"status":"ok","error":null,"data":null,"metadata":null}
```

The `input` and `annotated_request` happen to be identical here because the raw OpenAI request already matches the canonical shape. For providers whose wire format differs (e.g., Anthropic, NIM), the producer decodes into the canonical structure and `annotated_request` diverges from `input`.

### 8.2 Mixed-schema stream

An agent using both an OpenAI model for planning and a local NIM model for generation emits events with different `schema.name` values in the same stream. Consumers dispatch per-event.

```jsonl
{"kind":"ScopeStart","scope_type":"llm","schema":{"name":"openai/chat-completions","version":"v1"},...}
{"kind":"ScopeEnd","scope_type":"llm","schema":{"name":"openai/chat-completions","version":"v1"},...}
{"kind":"ScopeStart","scope_type":"llm","schema":{"name":"nvidia/llm","version":"v1"},...}
{"kind":"ScopeEnd","scope_type":"llm","schema":{"name":"nvidia/llm","version":"v1"},...}
```

## 9. Versioning and Evolution

This document tracks the ATOF spec version. Schema registry entries have their own versioning independent of ATOF:

- Adding a new schema (new `name`) is a MINOR change to this document.
- Bumping a schema's version (`openai/chat-completions` v1 → v2) is a MINOR change — the old version remains valid and consumers dispatch on `schema.version`.
- Removing a schema is a MAJOR change.
- Changing the canonical structured shape (`AnnotatedLlmRequest` structure) is a MAJOR change to the ATOF spec.

## 10. Future Work (roadmap)

### 10.1 `scope_type` profiles (priority 3.5)

The current resolution chain validates `annotated_request` / `annotated_response` when a schema is declared. When schema resolution falls through to priority 4 (opaque), `annotated_*` payloads are preserved but not validated — and `scope_type` becomes the only semantic signal about the event's shape.

A future revision MAY introduce **scope_type profiles**: canonical default schemas keyed by `scope_type` that provide basic structural validation for opaque events. Sketch:

- Spec defines canonical default schemas for common scope types — e.g., `default/llm.v1`, `default/tool.v1` — describing the minimum structural shape a consumer can expect for an opaque `scope_type == "llm"` or `scope_type == "tool"` event (shape of `input`, `output`, conventional fields).
- Consumers bundle these defaults alongside explicit schemas.
- A new "priority 3.5" slots in between priority 3 and priority 4: when no schema is declared but the event carries a known `scope_type`, the consumer MAY apply the corresponding scope_type profile as a structural-validation fallback.
- Consumers retain discretion — scope_type profiles are opt-in, never mandated by the spec.

This closes the "opaque but not unknown" gap: a tier-1 producer that emits `scope_type: "llm"` with no schema can still give downstream consumers a typed view if both sides agree on the scope_type profile convention.

Not in v0.1 scope. Tracked for a future MINOR-level revision once concrete consumer demand surfaces.

---

*Last updated: 2026-04-15 alongside ATOF v0.1.*