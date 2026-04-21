# ATOF Schema Profiles

**Version:** 0.1  
**Companion to:** `[atof-event-format.md](./atof-event-format.md)` (ATOF core wire format)

---

## 1. Purpose

This document is a guided walkthrough of the reference examples under `packages/nvidia_nat_atif/examples/atof_to_atif/`. Each example demonstrates one of ATOF's three producer enrichment tiers (`atof-event-format.md` §1.1). Reading the examples with this doc open is the fastest way to understand what `profile`, `schema`, and `annotated_`* look like on the wire, and how consumers use the optional schema layer.

Scope of this document:

- Walk through EXMP-01, EXMP-02, EXMP-03 event streams, and the EXMP-03b error-recovery variant.
- Specify the `schema` identifier contract and namespace conventions.
- Specify the four-priority schema resolution chain consumers use to find schema bodies.
- State what producers at each tier must emit.

Not in scope:

- Defining canonical annotation shapes. ATOF does not define payload shape — each schema ID does, independently. When this doc needs a concrete example shape, it uses `openai/chat-completions.v1` with OpenAI's native wire format ([OpenAI Chat Completions reference](https://developers.openai.com/api/docs/guides/completions)).
- Shipping specific profile files. This package does not ship a schema registry in v0.1 — schema identity is carried on the wire (on `schema` fields and optionally in `StreamHeader.schemas`), and consumers bring their own bodies.
- ATOF → ATIF conversion — see `examples/atof_to_atif/README.md`.

## 2. Running the examples

Two scripts drive the walkthrough:

```bash
cd packages/nvidia_nat_atif
python examples/atof_to_atif/generate_examples.py   # writes 4 .jsonl streams
python examples/atof_to_atif/convert_to_atif.py     # converts each to ATIF JSON
```

Outputs land in `examples/atof_to_atif/output/`:


| Scenario | ATOF file            | ATIF file           | Events | ATIF steps | Tier | Purpose                                                                   |
| -------- | -------------------- | ------------------- | ------ | ---------- | ---- | ------------------------------------------------------------------------- |
| EXMP-01  | `exmp01_atof.jsonl`  | `exmp01_atif.json`  | 8      | 4          | 1    | Opaque pass-through; no StreamHeader, `scope_type: "unknown"` throughout. |
| EXMP-02  | `exmp02_atof.jsonl`  | `exmp02_atif.json`  | 9      | 5          | 2    | Classified `scope_type` + `profile`; no schema.                           |
| EXMP-03  | `exmp03_atof.jsonl`  | `exmp03_atif.json`  | 9      | 5          | 3    | Adds `openai/chat-completions.v1` schema + annotations.                   |
| EXMP-03b | `exmp03b_atof.jsonl` | `exmp03b_atif.json` | 7      | 3          | 2    | Variant: tier-2 tool timeout + parent recovery.                           |


EXMP-01 → 02 → 03 walks the three producer enrichment tiers in order. EXMP-03b is a side variant showing error-recovery semantics at tier-2; not part of the main tier progression. EXMP-02, EXMP-03, and EXMP-03b each open with a `StreamHeaderEvent` at position 0 (spec §3.4 — MUST be first when present); EXMP-01 deliberately omits the header because tier-1 producers have nothing to declare.

## 3. EXMP-01 — tier-1 raw pass-through

A calculator-shaped workflow where the producer cannot classify any scope. Every scope carries `scope_type: "unknown"`, `profile: null`, `schema: null`, and opaque raw JSON in `input` / `output`. Eight events total — 4 `ScopeStart` / `ScopeEnd` pairs with **no `StreamHeader`**.

What this example demonstrates:

- **No `StreamHeader`.** Tier-1 producers have nothing to declare — no schema registry, no stream-level metadata. The header is optional per spec §3.4, so omitting it is the correct tier-1 wire shape. Consumers see no leading header and know immediately that the 4-priority resolution chain (§7.1) will fall straight through to priority-4 (opaque).
- **Every lifecycle event** carries `scope_type: "unknown"`, `profile: null`, `schema: null`, `annotated_request: null`, `annotated_response: null`. The producer knows timing and payloads but has no semantic classification.
- **Raw `input` / `output`** carry opaque provider JSON. The stream is a lossless record of what happened; consumers can replay it without provider-specific parsing.
- **ATIF conversion** maps each opaque `ScopeEnd` to a `source: "system"` step via the converter's generic fall-through (see `examples/atof_to_atif/README.md` → "Conversion reference"). The four inner + outer `ScopeEnd` events yield a 4-step trajectory. `Trajectory.agent.name` uses the outermost root scope's `name` (`"opaque_workflow"`) since no `scope_type: "agent"` event is present — a tier-1 fallback built into the reference converter.

Representative opaque ScopeStart (event 2):

```json
{"kind":"ScopeStart","atof_version":"0.1","uuid":"inner-001","parent_uuid":"root-001","timestamp":"2026-01-01T00:00:01Z","name":"provider_callback_1","attributes":[],"scope_type":"unknown","input":{"raw_payload":"<provider request 1>"}}
```

Representative opaque ScopeEnd (event 3):

```json
{"kind":"ScopeEnd","atof_version":"0.1","uuid":"inner-001","parent_uuid":"root-001","timestamp":"2026-01-01T00:00:02Z","name":"provider_callback_1","attributes":[],"scope_type":"unknown","output":{"raw_payload":"<provider response 1: tool invocation>"},"status":"ok"}
```

**When to emit events like this:** a runtime wrapping a third-party framework whose callback fires a raw blob the wrapper can't classify. The stream preserves timing, parent linkage, and raw payloads; consumers can still reconstruct call-graph ancestry and replay the run, even though the ATIF trajectory consists of opaque system steps rather than structured user/agent/observation turns.

**Producer requirements (tier-1):**

- **Do not emit a `StreamHeader`** — tier-1 has no schema metadata to declare.
- Emit `scope_type: "unknown"` on every `ScopeStart` / `ScopeEnd`.
- Emit `profile: null`, `schema: null`, `annotated_request: null`, `annotated_response: null`.
- Preserve the raw provider payload in `input` / `output`.

File: `examples/atof_to_atif/output/exmp01_atof.jsonl`

## 4. EXMP-02 — tier-2 semantic-tagged

Same calculator workflow as EXMP-01, but now the producer classifies every scope. Workflow: `agent → llm (decides) → tool → llm (answers) → agent done`. Nine events.

What this example demonstrates:

- `**StreamHeader**` (event 0) carries `schemas: {}` — a minimal manifest. Tier-2 producers MAY emit an empty header as a forward-looking "I know about the schema layer but am not using it" signal, or omit it entirely. EXMP-02 emits it; EXMP-01 (tier-1) omitted it entirely.
- **LLM scopes** carry `profile.model_name: "gpt-4.1"` — tells consumers "this is a gpt-4.1 call" without decoding the payload shape.
- **Tool scope** carries `profile.tool_call_id: "call_abc"` — the correlation ID the LLM returned when it requested the tool call.
- `**schema: null`** on every event — no schema declared, so `annotated_request` and `annotated_response` are null (§7 rule).
- **Raw `input` / `output`** still carries the full provider JSON — tier-1 fidelity is preserved even at tier-2.
- **ATIF conversion** produces a 5-step rich trajectory: user → agent (with `tool_calls`) → system (observation) → user → agent. `Trajectory.agent.name == "calculator_agent"` from the `scope_type: "agent"` scope.

Representative LLM ScopeStart (event 2):

```json
{"kind":"ScopeStart","atof_version":"0.1","uuid":"llm-001","parent_uuid":"agent-001","timestamp":"2026-01-02T00:00:01Z","name":"gpt-4.1","attributes":[],"scope_type":"llm","profile":{"model_name":"gpt-4.1"},"input":{"messages":[{"role":"user","content":"What is 3 + 4?"}]}}
```

Representative tool ScopeStart (event 4):

```json
{"kind":"ScopeStart","atof_version":"0.1","uuid":"tool-001","parent_uuid":"agent-001","timestamp":"2026-01-02T00:00:03Z","name":"calculator__add","attributes":[],"scope_type":"tool","profile":{"tool_call_id":"call_abc"},"input":{"a":3,"b":4}}
```

**When to emit events like this:** native runtimes (NAT, LangChain, LlamaIndex wrappers) that classify work at the hook site but don't decode provider API shapes. Consumers get enough to know *what kind of work ran* without having to parse provider-specific structure.

**Producer requirements (tier-2):**

- Everything in tier-1 (raw payloads preserved, timing + parent linkage).
- Populate `scope_type` with a value from the closed vocabulary (`atof-event-format.md` §4): `"agent"`, `"llm"`, `"tool"`, `"retriever"`, `"embedder"`, `"reranker"`, `"guardrail"`, `"evaluator"`, `"function"`, or `"custom"` (with `profile.subtype`).
- Populate applicable `profile` keys — `profile.model_name` for llm, `profile.tool_call_id` for tool, `profile.subtype` for custom.
- Keep `schema: null` and `annotated_*: null` unless you are a tier-3 producer.
- The `StreamHeader` is optional; emit `schemas: {}` as a "schema-layer aware but unused" signal, or omit the header entirely.

File: `examples/atof_to_atif/output/exmp02_atof.jsonl`

## 5. EXMP-03 — tier-3 with `openai/chat-completions.v1`

Same workflow as EXMP-02, but every LLM event carries the tier-3 schema layer (`schema: {name, version}` plus structured `annotated_request` / `annotated_response`). Tool events stay tier-2 to show that the schema layer is per-event and selective.

### 5.1 The StreamHeader declares the schema registry

Event 0 carries the registry inline, with a full JSON Schema body:

```json
{
  "kind": "StreamHeader",
  "atof_version": "0.1",
  "uuid": "hdr-003",
  "timestamp": "2026-01-03T00:00:00Z",
  "name": "exmp03_header",
  "schemas": {
    "openai/chat-completions.v1": {
      "$schema": {
        "$id": "openai/chat-completions.v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": true
      }
    }
  }
}
```

- **Key:** `"openai/chat-completions.v1"` — the canonical form `{name}.{version}` with `version` carrying its own `v` prefix (§7).
- **Value:** a `SchemaEntry` object whose `$schema` is the JSON Schema body consumers use to validate `annotated_`* payloads at priority 2 of the resolution chain (§7.1).
- An empty entry (`{}`) would be a manifest declaration only — the resolution chain would fall through to priority 3 (consumer-bundled registry).

### 5.2 Each LLM ScopeStart references the schema

Event 2 (first LLM turn, ScopeStart):

```json
{
  "kind": "ScopeStart",
  "atof_version": "0.1",
  "uuid": "llm-005",
  "parent_uuid": "agent-003",
  "timestamp": "2026-01-03T00:00:01Z",
  "name": "gpt-4.1",
  "attributes": [],
  "scope_type": "llm",
  "profile": {"model_name": "gpt-4.1"},
  "schema": {"name": "openai/chat-completions", "version": "v1"},
  "input": {
    "model": "gpt-4.1",
    "messages": [{"role": "user", "content": "What is 3 + 4?"}],
    "tools": [{"type": "function", "function": {"name": "calculator__add", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}}}],
    "temperature": 0.7,
    "max_tokens": 1024
  },
  "annotated_request": {
    "model": "gpt-4.1",
    "messages": [{"role": "user", "content": "What is 3 + 4?"}],
    "tools": [{"type": "function", "function": {"name": "calculator__add", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}}}],
    "temperature": 0.7,
    "max_tokens": 1024
  }
}
```

Key points about `annotated_request`:

- Uses OpenAI Chat Completions' **native wire shape** — the same shape you'd POST to `/v1/chat/completions`.
- `model`, `messages` required.
- Sampling params (`temperature`, `max_tokens`, `top_p`, `stop`, `seed`, `frequency_penalty`, `presence_penalty`, `response_format`, `tool_choice`, `parallel_tool_calls`, etc.) are **top-level** — not wrapped under a `params` object.
- `tools` is an array of `{type: "function", function: {name, description?, parameters}}`.
- `input` (raw) and `annotated_request` (decoded) happen to be identical here because the provider's request already matches the canonical shape. For providers whose wire format differs (Anthropic, NIM), `annotated_request` would diverge from `input`.

### 5.3 Each LLM ScopeEnd carries the response

Event 3 (first LLM turn, ScopeEnd):

```json
{
  "kind": "ScopeEnd",
  "atof_version": "0.1",
  "uuid": "llm-005",
  "parent_uuid": "agent-003",
  "timestamp": "2026-01-03T00:00:02Z",
  "name": "gpt-4.1",
  "attributes": [],
  "scope_type": "llm",
  "profile": {"model_name": "gpt-4.1"},
  "schema": {"name": "openai/chat-completions", "version": "v1"},
  "status": "ok",
  "output": {
    "id": "chatcmpl-exmp03-001",
    "model": "gpt-4.1",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "calculator__add", "arguments": "{\"a\":3,\"b\":4}"}}]}, "finish_reason": "tool_calls"}],
    "usage": {"prompt_tokens": 84, "completion_tokens": 18, "total_tokens": 102}
  },
  "annotated_response": {
    "id": "chatcmpl-exmp03-001",
    "model": "gpt-4.1",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "calculator__add", "arguments": "{\"a\":3,\"b\":4}"}}]}, "finish_reason": "tool_calls"}],
    "usage": {"prompt_tokens": 84, "completion_tokens": 18, "total_tokens": 102}
  }
}
```

Key points about `annotated_response`:

- Uses OpenAI's **native response shape**. `id`, `model`, `choices`, `usage` at the top level.
- `choices[*].message.tool_calls[*].function.arguments` is a **JSON string** (`"{\"a\":3,\"b\":4}"`), per OpenAI's wire convention — not a parsed object. Consumers that want typed args parse the string themselves.
- `finish_reason: "tool_calls"` signals the model wants to invoke a tool before completing.
- `usage` comes directly from the provider: `prompt_tokens`, `completion_tokens`, `total_tokens`. If the provider returned nested breakdowns (`prompt_tokens_details`, `completion_tokens_details`), they'd pass through verbatim.

### 5.4 Tool events stay tier-2 in this example

Events 4-5 are the tool invocation. They carry `profile.tool_call_id: "call_abc"` but `schema: null` — the example deliberately demonstrates that the schema layer is per-event. A producer MAY mix tier-3 LLM events with tier-2 tool events in the same stream.

**When to emit events like this:** producer runtimes that wrap a known provider API (OpenAI Chat, Anthropic Messages, NVIDIA NIM) and can decode the wire format at the hook site. Consumers that understand the schema get typed access to messages, tool calls, and usage without per-provider parsing. Consumers that don't understand the schema still have the raw `input` / `output` for fallback.

**Producer requirements (tier-3):**

- Everything in tier-2 (typed `scope_type`, populated `profile`, raw payloads preserved).
- Decode the provider payload into the shape declared by the chosen schema ID.
- Declare `schema: {name, version}` on each `ScopeStart` / `ScopeEnd` that carries an annotation.
- Populate `annotated_request` on `ScopeStart` and `annotated_response` on `ScopeEnd`.
- Keep raw `input` and `output` unchanged for round-trip fidelity.
- Optionally include an inline `$schema` body on the event (priority 1) or declare it in the `StreamHeader.schemas` registry (priority 2). The schema layer is per-event — tier-3 producers MAY emit tier-2 events in the same stream for scopes they don't decode.

File: `examples/atof_to_atif/output/exmp03_atof.jsonl`

---

## 6. EXMP-03b — tier-2 with error recovery (variant)

A variant of EXMP-03's structural shape that illustrates error-recovery semantics rather than schema annotation. Workflow: `agent → llm (plans) → tool (times out) → agent (recovers)`. Seven events. Tier-2 shape (no schema layer) — sits alongside the main tier progression as a side example.

What this example demonstrates:

- **Tool ScopeEnd** (event 5) carries `status: "error"` with structured `error: {type: "TimeoutError", message: "request timed out after 5s"}` — per spec §5.1.
- **Agent ScopeEnd** (event 6) carries `status: "ok"` with a graceful user-facing message in `output` — per spec §5.3, parents may catch child errors and complete normally.
- Each scope reports its own terminal status; parent status does not inherit child status.

Tool ScopeEnd (event 5):

```json
{"kind":"ScopeEnd","atof_version":"0.1","uuid":"tool-002","parent_uuid":"agent-002","timestamp":"2026-01-04T00:00:08Z","name":"web_search","attributes":[],"scope_type":"tool","profile":{"tool_call_id":"call_xyz"},"status":"error","error":{"message":"request timed out after 5s","type":"TimeoutError"}}
```

Agent ScopeEnd (event 6):

```json
{"kind":"ScopeEnd","atof_version":"0.1","uuid":"agent-002","parent_uuid":null,"timestamp":"2026-01-04T00:00:10Z","name":"search_agent","attributes":[],"scope_type":"agent","profile":null,"output":"Sorry — the search service is temporarily unavailable. Please try again shortly.","status":"ok"}
```

**When to emit events like this:** any time a runtime handles mid-flight failures. Consumers replaying the stream can see that the child failed, the parent caught and recovered, and the user-facing output reflects the recovery path.

EXMP-03b follows the same tier-2 producer requirements as EXMP-02 (§4); it adds no new producer rules, just exercises the `status` / `error` semantics from `atof-event-format.md` §5.

File: `examples/atof_to_atif/output/exmp03b_atof.jsonl`

---

## 7. The `schema` field

`ScopeStart` and `ScopeEnd` events whose `scope_type` is `"llm"` or `"tool"` MAY carry a `schema` field.


| Field     | Type   | Required | Description                                                                                                                                                                                                   |
| --------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`    | string | Yes      | Canonical schema ID, `<vendor>/<schema_name>` form.                                                                                                                                                           |
| `version` | string | Yes      | Schema version. MUST start with `v` followed by MAJOR (e.g., `v1`, `v2`) and MAY include `.MINOR` (e.g., `v1.0`, `v1.1`, `v2.3`). Minor is optional — `v1` and `v1.0` are equivalent for resolution purposes. |
| `$schema` | object | No       | Optional inline schema body. When present, triggers priority-1 resolution (§7.1).                                                                                                                             |


**Canonical ID form:** `{schema.name}.{schema.version}` — concatenation with a single `.` separator. Version already carries its `v` prefix. This ID is the key used in `StreamHeader.schemas`, in consumer-local registries, and across all four resolution priorities. Examples:

- `openai/chat-completions.v1`
- `nvidia/llm.v1`
- `anthropic/messages.v1`

**Version semantics:**

- **MAJOR bump** (e.g., `v1` → `v2.0`) signals a breaking change to the structured shape — renamed or removed required fields, semantically incompatible field types, etc. Consumers dispatch on MAJOR and MUST NOT treat different MAJORs as interchangeable.
- **MINOR bump** (e.g., `v1.0` → `v1.1`) is backward-compatible — new optional fields, extended enums, additional nested objects. A consumer that only knows `v1.0` should accept `v1.1` payloads and ignore unknown fields.
- Minor is optional. `v1` is a valid version string by itself; consumers treat `v1` and `v1.0` as equivalent.

**Namespace conventions:**

- `default/` — ATOF-spec-defined utility IDs. Reserved: `default/passthrough` (annotation shape equals raw shape), `default/opaque` (explicit no-decode).
- `openai/`, `anthropic/`, `nvidia/` — provider-specific. When a schema describes a provider's native API shape, place it under the provider's namespace.
- Any other vendor prefix is allowed; vendors SHOULD namespace to avoid collisions.
- The `<vendor>/<schema_name>` form is case-sensitive. Use lowercase by convention.

This v0.1 package does not ship specific schema files. Schema identity is carried on the wire, and consumers bring their own bodies (local registry or inline via `StreamHeader.schemas`).

**Rules:**

- If `schema` is `null`, `annotated_request` and `annotated_response` MUST be `null`.
- If `schema` is present, `annotated_`* MAY be present and SHOULD follow the schema.
- Unknown schema names MUST NOT cause event rejection — preserve and pass through.

### 7.1 Resolving the schema body

Consumers that need structured access to `annotated_*` resolve a schema body by checking, in order: (1) the event's inline `$schema`, (2) the matching entry in `StreamHeader.schemas`, (3) the consumer's local bundled registry, (4) nothing — treat the annotation as opaque and preserve it verbatim. Missing `StreamHeader` or unresolved IDs at any priority fall through to the next; opaque resolution is always legal and MUST NOT cause event rejection. EXMP-03 is a priority-2 walkthrough — event 2 declares `schema: {name: "openai/chat-completions", version: "v1"}` without an inline body, and the `StreamHeader.schemas["openai/chat-completions.v1"]["$schema"]` body is what a consumer would use to validate the LLM events' `annotated_request` / `annotated_response`.

**Consumer requirements:**

- Resolve schema bodies via the four-priority chain above.
- Validate `annotated_`* against the resolved body when validation is enabled and a body was found. Validation is opt-in.
- Never reject events for unknown schema IDs.
- Preserve unknown fields and unknown annotation keys on re-emission.
- Fall back to opaque handling (priority 4) when resolution fails.

**Validation policy:** producer-side and consumer-side JSON Schema validation are both optional. Validation failures SHOULD be logged but MUST NOT cause event drops — replay-for-inspection requires lossless propagation. The v0.1 reference ATOF→ATIF converter (`src/nat/atof/scripts/atof_to_atif_converter.py`) does not validate annotations against schemas; schema resolution and validation are consumer-side concerns.

---

*Last updated: 2026-04-21 alongside ATOF v0.1.*