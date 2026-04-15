# Agentic Trajectory Observability Format (ATOF) Specification — Core

**Status:** Active
**Version:** 0.3
**Date:** 2026-04-15
**NeMo Agent Toolkit Reference Implementation:** `src/nat/atof/`
**NeMo-Flow Reference Implementation:** `crates/core/src/api/event.rs`

**Companion documents:**

- `[atof-codec-profiles.md](./atof-codec-profiles.md)` — codec identifiers and out-of-band JSON Schema reference shapes for structured LLM request/response and tool invocation payloads.
- `[atof-to-atif-converter.md](./atof-to-atif-converter.md)` — normative mapping from an ATOF event stream to an ATIF trajectory.

---

## 1. Overview

ATOF (Agentic Trajectory Observability Format) is the wire format for agent runtime subscriber callbacks. Events represent the lifecycle of scopes — composable units of agent work — within the runtime. Subscribers receive events in real time as the runtime executes agent workflows.

**Primary purpose: replay-for-inspection.** An ATOF event stream MUST carry enough information to reconstruct what happened in an agent run — identity, call graph, LLM messages in/out, tool calls and results, status — so that humans and tools can debug, audit, and evaluate the run post-hoc. ATIF conversion and observability export are consumer-specific downstream layers, not core goals.

Transport is JSON Lines: one JSON object per line. The `kind` field at the top of every event is the primary discriminator. ATOF v0.3 defines **seven event kinds**:

- `"ScopeStart"` — a generic scope was opened (for anything outside LLM/Tool)
- `"ScopeEnd"` — a generic scope was closed
- `"LlmStart"` — an LLM call span was opened
- `"LlmEnd"` — an LLM call span was closed
- `"ToolStart"` — a tool invocation span was opened
- `"ToolEnd"` — a tool invocation span was closed
- `"Mark"` — a point-in-time checkpoint was recorded

LLM and Tool are **first-class event kinds**, not `scope_type` values on a generic event. The generic `ScopeStart`/`ScopeEnd` pair exists for everything else (agents, functions, retrievers, rerankers, guardrails, evaluators, vendor-defined custom categories, and "I don't know what this is").

**Wire envelope example:**

```json
{"kind":"LlmStart","schema_version":"0.3","uuid":"...","parent_uuid":"...","timestamp":"...","name":"gpt-4.1","attributes":["streaming"],"input":{...},"model_name":"gpt-4.1","annotated_request":null,"codec":null,"data":null,"metadata":null}
```

### 1.1 Three Producer Enrichment Tiers

ATOF is designed for progressive enrichment at the producer's discretion. A producer emits what it knows; absent fields are legal everywhere except where noted.

| Tier | Producer knows | Wire shape | Use case |
|------|---|---|---|
| **1. Raw pass-through** | nothing semantic — just a payload | event kind + envelope + opaque `input`/`output` JSON; other typed fields are `null` | runtime wrapping third-party frameworks where callback provides a blob, not a classification |
| **2. Semantic-tagged** | the kind of work (LLM, tool, specific scope type) | typed event kind + populated typed fields (`model_name`, `tool_call_id`, `scope_type`, `attributes`) | native runtime (NeMo-Flow) emitting its own events; langchain wrappers that can classify at the hook site |
| **3. Codec-annotated** | the structured shape of the payload (messages, params, tool defs, …) | tier-2 plus `codec: {name, version}` identifier and optional `annotated_request`/`annotated_response` structured representation | producer has a registered codec (OpenAI Chat, Anthropic Messages, NVIDIA NIM, …) that decodes raw provider JSON into a canonical structure |

**Design principle:** Tier 1 must always work. A consumer that doesn't understand tier-2 or tier-3 enrichment MUST still preserve the event verbatim. Consumers SHOULD NOT reject events whose `scope_type` or `codec` name they don't recognize — unknown values are forward-compat extensions, not errors.

### 1.2 The Three Structured Fields at a Glance

Beyond the base envelope (`kind`, `uuid`, `parent_uuid`, `timestamp`, `name`, `schema_version`), every event carries three structured fields:

|                       | Spec-governed shape (varies by kind) | Opaque to ATOF     |
| --------------------- | ------------------------------------ | ------------------ |
| **About the work**    | `input` (Start), `output` (End), `attributes`, kind-specific fields (`model_name`, `tool_call_id`, `scope_type`, …) | —                  |
| **About the context** | —                                    | `data`, `metadata` |

- `input` / `output` — raw payload (post-guardrail). Opaque to the ATOF core; structure MAY be defined by a codec (tier-3) and carried as `annotated_request` / `annotated_response`.
- `attributes` — behavioral flag array. Per-event-kind vocabulary (see §2.1 and each event's §3 subsection).
- Kind-specific typed fields — `model_name`, `annotated_request`, `annotated_response` on LLM events; `tool_call_id` on tool events; `scope_type`, `subtype` on generic scope events. All optional.
- `data` — application-defined payload. Opaque to ATOF. Consumers MUST NOT dispatch on `data` contents.
- `metadata` — tracing/correlation envelope (`trace_id`, `span_id`, etc.).

---

## 2. Base Event Envelope

Every event carries these six envelope fields. Specific event kinds add fields on top.

| Field            | Type                                    | Required | Description                                                                                                 |
| ---------------- | --------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------- |
| `kind`           | string                                  | Yes      | Event kind discriminator. One of: `"ScopeStart"`, `"ScopeEnd"`, `"LlmStart"`, `"LlmEnd"`, `"ToolStart"`, `"ToolEnd"`, `"Mark"`. |
| `schema_version` | string                                  | Yes      | ATOF protocol version, `"MAJOR.MINOR"` (e.g., `"0.3"`). See §6.6.                                            |
| `uuid`           | string (UUID)                           | Yes      | Unique identifier for this event or span. For `*Start`/`*End` pairs, the Start and End share a `uuid`.       |
| `parent_uuid`    | string (UUID) or null                   | No       | UUID of the containing scope when this event was emitted. Null only for root scope events and unparented `Mark` events. |
| `timestamp`      | string (RFC 3339) or integer (epoch µs) | Yes      | Wall-clock time the event was emitted. See §6.1.                                                             |
| `name`           | string                                  | Yes      | Human-readable label — e.g., `"my_agent"`, `"calculator__add"`, `"gpt-4.1"`.                                 |
| `data`           | object or null                          | No       | Application-defined payload. Opaque to ATOF.                                                                 |
| `metadata`       | object or null                          | No       | Tracing/correlation envelope — e.g., `{"trace_id": "...", "span_id": "..."}`.                                |

### 2.1 `attributes` — behavioral flag array

`attributes` is a cross-cutting field on all six lifecycle events (`ScopeStart`/`ScopeEnd`/`LlmStart`/`LlmEnd`/`ToolStart`/`ToolEnd`). `MarkEvent` does NOT carry `attributes`.

| Field        | Type             | Required | Description                                                                                                 |
| ------------ | ---------------- | -------- | ----------------------------------------------------------------------------------------------------------- |
| `attributes` | array of strings | Yes      | Canonical lowercase flag names (sorted, deduplicated). Empty array `[]` when no flags are set. |

Producers MUST emit `attributes` in lexicographic order with no duplicates. Consumers SHOULD treat the array as an unordered set and MUST preserve unknown flag names when re-emitting. Unknown flags SHOULD NOT be treated as errors.

**Per-event-kind flag vocabulary:**

`ScopeStart` / `ScopeEnd`:

| Flag | Meaning (when present) |
|------|---|
| `"parallel"` | Scope executes concurrently with sibling scopes under the same parent. |
| `"relocatable"` | Scope may be moved across async task boundaries without losing context. |

`LlmStart` / `LlmEnd`:

| Flag | Meaning (when present) |
|------|---|
| `"stateless"` | Scope does not maintain state between invocations. |
| `"streaming"` | Scope produces its output incrementally as chunks. |

`ToolStart` / `ToolEnd`:

| Flag | Meaning (when present) |
|------|---|
| `"local"` | Tool executes in the same process as the runtime (not dispatched to a remote service). |

**Flag extensibility.** Implementations MAY emit additional flag names for vendor extensions; non-canonical flags SHOULD be namespaced with a dotted prefix — for example, `"nvidia.speculative"`. Consumers MUST preserve unknown flag strings and MUST NOT reject events carrying them.

---

## 3. Event Kinds

### 3.1 `ScopeStartEvent`

Emitted when a generic scope (not an LLM or tool call) is pushed onto the active scope stack.

| Field            | Type                                    | Required | Description                                                                                             |
| ---------------- | --------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------- |
| `kind`           | string                                  | Yes      | Literal `"ScopeStart"`.                                                                                 |
| `schema_version` | string                                  | Yes      | See §2.                                                                                                 |
| `uuid`           | string (UUID)                           | Yes      | See §2.                                                                                                 |
| `parent_uuid`    | string (UUID) or null                   | No       | See §2. Null on the root scope.                                                                         |
| `timestamp`      | string or integer                       | Yes      | See §2.                                                                                                 |
| `name`           | string                                  | Yes      | See §2.                                                                                                 |
| `attributes`     | array of strings                        | Yes      | See §2.1. Vocabulary: `"parallel"`, `"relocatable"`.                                                     |
| `scope_type`     | string                                  | Yes      | Semantic category. See §4.                                                                              |
| `subtype`        | string or null                          | No       | Free-form vendor name. REQUIRED when `scope_type == "custom"`; SHOULD be omitted otherwise. See §4.2.   |
| `input`          | any or null                             | No       | Post-guardrail input payload. Opaque to ATOF. When a codec is registered, the structured form lives in `annotated_request` on the companion `LlmStart` event; generic scopes do NOT carry codec annotations. |
| `data`           | object or null                          | No       | See §2.                                                                                                 |
| `metadata`       | object or null                          | No       | See §2.                                                                                                 |

### 3.2 `ScopeEndEvent`

Emitted when a generic scope is popped from the active scope stack. Paired 1:1 with `ScopeStart` by `uuid`.

| Field            | Type                                    | Required | Description                                                                                             |
| ---------------- | --------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------- |
| `kind`           | string                                  | Yes      | Literal `"ScopeEnd"`.                                                                                   |
| `schema_version` | string                                  | Yes      | See §2.                                                                                                 |
| `uuid`           | string (UUID)                           | Yes      | Same UUID as the paired `ScopeStart`.                                                                   |
| `parent_uuid`    | string (UUID) or null                   | No       | Same as the paired `ScopeStart`.                                                                        |
| `timestamp`      | string or integer                       | Yes      | See §2. Differs from the `ScopeStart` timestamp (End occurs later).                                     |
| `name`           | string                                  | Yes      | Same as the paired `ScopeStart`.                                                                        |
| `attributes`     | array of strings                        | Yes      | Same as the paired `ScopeStart`.                                                                        |
| `scope_type`     | string                                  | Yes      | Same as the paired `ScopeStart`.                                                                        |
| `subtype`        | string or null                          | No       | Same as the paired `ScopeStart`.                                                                        |
| `output`         | any or null                             | No       | Post-guardrail output payload. Opaque to ATOF.                                                          |
| `status`         | string (enum)                           | Yes      | Terminal outcome. One of: `"ok"`, `"error"`, `"cancelled"`. See §5.                                     |
| `error`          | object or null                          | No       | Structured error info when `status == "error"`. See §5.1.                                              |
| `data`           | object or null                          | No       | See §2.                                                                                                 |
| `metadata`       | object or null                          | No       | See §2.                                                                                                 |

### 3.3 `LlmStartEvent`

Emitted when an LLM call span is opened.

| Field               | Type                                    | Required | Description                                                                                             |
| ------------------- | --------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------- |
| `kind`              | string                                  | Yes      | Literal `"LlmStart"`.                                                                                   |
| `schema_version`    | string                                  | Yes      | See §2.                                                                                                 |
| `uuid`              | string (UUID)                           | Yes      | See §2.                                                                                                 |
| `parent_uuid`       | string (UUID) or null                   | No       | See §2.                                                                                                 |
| `timestamp`         | string or integer                       | Yes      | See §2.                                                                                                 |
| `name`              | string                                  | Yes      | Usually the model identifier or a runtime-assigned span name — e.g., `"gpt-4.1"`, `"nvidia/nemotron-3"`. |
| `attributes`        | array of strings                        | Yes      | See §2.1. Vocabulary: `"stateless"`, `"streaming"`.                                                      |
| `model_name`        | string or null                          | No       | Normalized model identifier. MAY duplicate `name`; present for explicit structured access.              |
| `codec`             | object or null                          | No       | Codec identifier `{name: string, version: string}` declaring which codec shaped `annotated_request`. See `atof-codec-profiles.md`. |
| `input`             | any or null                             | No       | Raw provider request payload (post-guardrail). Opaque to ATOF core.                                     |
| `annotated_request` | object or null                          | No       | Structured codec-decoded view of the request (`messages`, `model`, `params`, `tools`, `tool_choice`, …). Shape declared by `codec`. See `atof-codec-profiles.md`. |
| `data`              | object or null                          | No       | See §2.                                                                                                 |
| `metadata`          | object or null                          | No       | See §2.                                                                                                 |

### 3.4 `LlmEndEvent`

Emitted when an LLM call span is closed. Paired 1:1 with `LlmStart` by `uuid`.

| Field                | Type                                    | Required | Description                                                                                             |
| -------------------- | --------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------- |
| `kind`               | string                                  | Yes      | Literal `"LlmEnd"`.                                                                                     |
| `schema_version`     | string                                  | Yes      | See §2.                                                                                                 |
| `uuid`               | string (UUID)                           | Yes      | Same UUID as the paired `LlmStart`.                                                                     |
| `parent_uuid`        | string (UUID) or null                   | No       | Same as the paired `LlmStart`.                                                                          |
| `timestamp`          | string or integer                       | Yes      | See §2.                                                                                                 |
| `name`               | string                                  | Yes      | Same as the paired `LlmStart`.                                                                          |
| `attributes`         | array of strings                        | Yes      | Same as the paired `LlmStart`.                                                                          |
| `model_name`         | string or null                          | No       | Same as the paired `LlmStart`, or the actually-used model if different (e.g., after provider routing).   |
| `codec`              | object or null                          | No       | Codec identifier for `annotated_response`. Usually the same codec as the paired `LlmStart`.              |
| `output`             | any or null                             | No       | Raw provider response payload (post-guardrail). Opaque to ATOF core.                                    |
| `annotated_response` | object or null                          | No       | Structured codec-decoded view of the response (`choices`, `usage`, `finish_reason`, `tool_calls`, …). Shape declared by `codec`. |
| `status`             | string (enum)                           | Yes      | Terminal outcome. One of: `"ok"`, `"error"`, `"cancelled"`. See §5.                                     |
| `error`              | object or null                          | No       | See §5.1.                                                                                               |
| `data`               | object or null                          | No       | See §2.                                                                                                 |
| `metadata`           | object or null                          | No       | See §2.                                                                                                 |

### 3.5 `ToolStartEvent`

Emitted when a tool invocation span is opened.

| Field            | Type                                    | Required | Description                                                                                             |
| ---------------- | --------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------- |
| `kind`           | string                                  | Yes      | Literal `"ToolStart"`.                                                                                  |
| `schema_version` | string                                  | Yes      | See §2.                                                                                                 |
| `uuid`           | string (UUID)                           | Yes      | See §2.                                                                                                 |
| `parent_uuid`    | string (UUID) or null                   | No       | Usually the UUID of the LLM scope that requested the tool call.                                         |
| `timestamp`      | string or integer                       | Yes      | See §2.                                                                                                 |
| `name`           | string                                  | Yes      | Tool/function identifier — e.g., `"calculator__add"`, `"web_search"`.                                   |
| `attributes`     | array of strings                        | Yes      | See §2.1. Vocabulary: `"local"`.                                                                         |
| `tool_call_id`   | string or null                          | No       | Provider-assigned correlation ID from the upstream LLM tool-call response — e.g., `"call_abc123"`.      |
| `input`          | any or null                             | No       | Tool arguments (post-guardrail). SHOULD be the JSON arguments dict.                                     |
| `data`           | object or null                          | No       | See §2.                                                                                                 |
| `metadata`       | object or null                          | No       | See §2.                                                                                                 |

### 3.6 `ToolEndEvent`

Emitted when a tool invocation span is closed. Paired 1:1 with `ToolStart` by `uuid`.

| Field            | Type                                    | Required | Description                                                                                             |
| ---------------- | --------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------- |
| `kind`           | string                                  | Yes      | Literal `"ToolEnd"`.                                                                                    |
| `schema_version` | string                                  | Yes      | See §2.                                                                                                 |
| `uuid`           | string (UUID)                           | Yes      | Same UUID as the paired `ToolStart`.                                                                    |
| `parent_uuid`    | string (UUID) or null                   | No       | Same as the paired `ToolStart`.                                                                         |
| `timestamp`      | string or integer                       | Yes      | See §2.                                                                                                 |
| `name`           | string                                  | Yes      | Same as the paired `ToolStart`.                                                                         |
| `attributes`     | array of strings                        | Yes      | Same as the paired `ToolStart`.                                                                         |
| `tool_call_id`   | string or null                          | No       | Same as the paired `ToolStart`.                                                                         |
| `output`         | any or null                             | No       | Tool return value (post-guardrail).                                                                     |
| `status`         | string (enum)                           | Yes      | Terminal outcome. One of: `"ok"`, `"error"`, `"cancelled"`. See §5.                                     |
| `error`          | object or null                          | No       | See §5.1.                                                                                               |
| `data`           | object or null                          | No       | See §2.                                                                                                 |
| `metadata`       | object or null                          | No       | See §2.                                                                                                 |

### 3.7 `MarkEvent`

Emitted as a point-in-time checkpoint. Unpaired (no Start/End semantics).

| Field            | Type                                    | Required | Description                                                                                             |
| ---------------- | --------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------- |
| `kind`           | string                                  | Yes      | Literal `"Mark"`.                                                                                       |
| `schema_version` | string                                  | Yes      | See §2.                                                                                                 |
| `uuid`           | string (UUID)                           | Yes      | See §2.                                                                                                 |
| `parent_uuid`    | string (UUID) or null                   | No       | See §2.                                                                                                 |
| `timestamp`      | string or integer                       | Yes      | See §2.                                                                                                 |
| `name`           | string                                  | Yes      | Label for the checkpoint — e.g., `"workflow_start"`, `"retry_attempt_2"`.                               |
| `data`           | object or null                          | No       | Optional checkpoint payload.                                                                            |
| `metadata`       | object or null                          | No       | See §2.                                                                                                 |

`MarkEvent` does NOT carry `attributes`, `status`, `input`, or `output`. It is deliberately minimal — a named timestamp with optional data.

---

## 4. `scope_type` Vocabulary

`scope_type` on `ScopeStart`/`ScopeEnd` classifies the kind of work the generic scope represents. The canonical vocabulary is a closed set of lowercase strings:

| `scope_type` value | Meaning                                                                       |
| ------------------ | ----------------------------------------------------------------------------- |
| `"agent"`          | Top-level agent or workflow scope.                                            |
| `"function"`       | Generic function or application step.                                         |
| `"retriever"`      | Retrieval step (document search, index lookup).                               |
| `"embedder"`       | Embedding-generation step.                                                    |
| `"reranker"`       | Result reranking step.                                                        |
| `"guardrail"`      | Guardrail or validation step.                                                 |
| `"evaluator"`      | Evaluation or scoring step.                                                   |
| `"custom"`         | Vendor-defined custom category. REQUIRES `subtype` to name the vendor scope.  |
| `"unknown"`        | Producer does not know or cannot classify the scope.                          |

**`"llm"` and `"tool"` are reserved and SHOULD NOT be emitted on `ScopeStart`/`ScopeEnd`.** Use the dedicated `LlmStart`/`LlmEnd` and `ToolStart`/`ToolEnd` event kinds instead. A generic scope with `scope_type: "llm"` or `"tool"` is legal but strongly discouraged — it indicates the producer is emitting LLM/tool work through the wrong event kind and downstream consumers MAY dispatch it less efficiently.

### 4.1 `"unknown"` is the tier-1 escape hatch

Producers that have a payload but no classification (the tier-1 pass-through case from §1.1) emit `scope_type: "unknown"`. This is ALWAYS valid. Consumers SHOULD NOT reject events with `scope_type: "unknown"`.

### 4.2 `subtype` when `scope_type == "custom"`

When `scope_type == "custom"`, the event MUST carry a `subtype: string` field naming the vendor scope category. `subtype` SHOULD follow a dotted-namespace convention to avoid collisions — for example:

- `"nvidia.speculative_decode"`
- `"langchain.memory_retrieval"`
- `"internal.audit_gate"`

When `scope_type != "custom"`, `subtype` SHOULD be absent or null. Consumers SHOULD preserve `subtype` verbatim on re-emission.

### 4.3 Extensibility

The `scope_type` enum is closed but `"custom"` + `subtype` provides unbounded vendor expressiveness. ATOF v0.3 reserves the right to promote frequently-used `subtype` values into first-class `scope_type` vocabulary entries in future versions.

---

## 5. Status and Error Semantics

### 5.1 Terminal status

Every `*EndEvent` (`ScopeEnd`, `LlmEnd`, `ToolEnd`) carries a required `status` field valued in:

- `"ok"` — the scope returned normally.
- `"error"` — the scope raised an exception or equivalent failure.
- `"cancelled"` — the scope was terminated before completion (timeout, parent cancel, explicit cancel).

`MarkEvent` does not carry `status` — marks are not lifecycle events.

When `status == "error"`, an optional `error` field MAY carry structured error info:

| Field       | Type                           | Description                                                                              |
| ----------- | ------------------------------ | ---------------------------------------------------------------------------------------- |
| `message`   | string                         | Human-readable error description.                                                        |
| `type`      | string or null                 | Error class or category — e.g., `"TimeoutError"`, `"ValidationError"`, `"RateLimited"`.  |
| `traceback` | string or null                 | Optional stack trace or debug trace.                                                     |

When `status != "error"`, `error` SHOULD be absent or null.

### 5.2 Cascading cancellation

When a parent cancels its children (e.g., due to parent timeout or parent error recovery), each child emits its own `*EndEvent` with `status == "cancelled"` before the parent emits its `*EndEvent`. §6.3 still holds: all child events precede the parent's `*EndEvent` in wall-clock order. The parent's own `status` reflects the parent's outcome:

- `"cancelled"` if the parent was itself cancelled from above.
- `"error"` if the parent raised (possibly because a child's failure propagated).
- `"ok"` if the parent chose to cancel its children as normal control flow and then completed.

### 5.3 Each scope reports its own terminal status

The `status` on an `*EndEvent` describes the outcome of THAT scope, not its children or its parent. A parent whose child errored MAY itself report `status == "ok"` if the parent caught and handled the child's error.

### 5.4 Dangling scopes

If the runtime dies before emitting a paired `*EndEvent`, no event appears in the stream. §6.3's pairing guarantee is contingent on orderly shutdown. Consumers that detect an unpaired `*Start` after the stream ends MAY synthesize an `*End` with `status == "cancelled"` for downstream processing; such synthetic events are out of scope for ATOF Core.

---

## 6. Event Stream Semantics

### 6.1 Timestamp Format and Ordering

**Accepted forms.** Every event's `timestamp` carries one of two interchangeable forms:

- **RFC 3339 string** (e.g., `"2026-01-01T00:00:00.123456Z"`) — human-readable, interoperable with general-purpose date libraries, default choice for debug and log-tailing contexts. MUST end with `Z` or an explicit UTC offset.
- **Integer epoch microseconds UTC** (e.g., `1767225600123456`) — fast to parse (~15× faster than RFC 3339), ~50% smaller on the wire, safe in JSON numbers through year 2255. Chosen for high-throughput streams and columnar-storage pipelines.

Emitters choose per event. A single stream MAY contain events in both forms.

**Why microseconds and not nanoseconds.** JSON numbers are IEEE 754 doubles with 53 bits of integer precision (~9 × 10¹⁵). Nanoseconds since epoch for 2026 is ~1.76 × 10¹⁸ — exceeds safe integer range. Microseconds fits safely and remains precise enough for agent-scope event correlation.

**Ordering.** Events are emitted in wall-clock order. Delivery from subscriber callbacks MAY arrive out-of-order for concurrent operations. Consumers MUST sort by `timestamp` before processing. When sorting a mixed-format stream, consumers MUST normalize both forms to a common representation (typically integer microseconds) before comparison — lexicographic string vs integer comparison is undefined.

**ATIF compatibility.** ATIF requires timestamps as ISO 8601 strings. RFC 3339 is a strict subset of ISO 8601, so the ATOF → ATIF converter forwards the RFC 3339 string form unchanged as a zero-cost pass-through; only the integer microsecond form is serialized to an RFC 3339 string before emitting ATIF.

### 6.2 Scope Nesting and `parent_uuid`

The runtime maintains a scope stack per async task. The `parent_uuid` of any event is the UUID of the scope that was on top of the stack when the handle was created. Following `parent_uuid` links upward reconstructs the full call graph.

The root scope has `parent_uuid = null`. This is the only Start event in a well-formed stream that may have a null `parent_uuid` (once the root scope is established). `MarkEvent`s MAY carry `parent_uuid = null` when emitted outside any scope.

### 6.3 Start/End Pairing

Every `ScopeStartEvent` is paired with exactly one `ScopeEndEvent` sharing the same `uuid`. The same rule applies to `LlmStart`/`LlmEnd` and `ToolStart`/`ToolEnd`. The `*EndEvent` is always emitted strictly after the `*StartEvent` (strict, not non-strict — `ts_micros(End) > ts_micros(Start)`).

`MarkEvent`s have no paired event — they are single-shot.

### 6.4 UUID Uniqueness

Each span receives a unique UUID at creation time. The `uuid` is stable across the Start and End events for the same span. In the Rust reference implementation, UUIDs are v7 (time-ordered).

### 6.5 ID Relationships

Three distinct identifier namespaces appear in an ATOF stream:

- **`uuid` / `parent_uuid`** — runtime identifiers attached to every event. Form the scope graph.
- **`tool_call_id`** (on `ToolStart`/`ToolEnd`) — an LLM-provider identifier that bridges an LLM's tool-call response with the resulting tool execution. Null when the tool was not invoked via an LLM tool-use flow.
- **Codec-decoded response IDs** (e.g., `chatcmpl-*` inside a decoded LLM response body, under `annotated_response`) — provider tracking identifiers. Opaque to ATOF Core; see `atof-codec-profiles.md`.

### 6.6 Schema Version and Negotiation

Every event carries a required `schema_version` field, formatted `"MAJOR.MINOR"` — e.g., `"0.3"`. This section defines when producers bump the version and how consumers dispatch on it.

**Reading rules.** Consumers SHOULD accept any `0.Y` event as ATOF-v0-family. Major-version bumps (`1.0`, `2.0`) MAY introduce breaking changes; consumers that want forward compat MUST dispatch on the major version and fail fast on unknown majors.

**Mixed-version streams.** A single stream MAY contain events at different minor versions (`0.3` and `0.4`). Consumers MUST NOT reject a stream because it contains newer minor versions than expected; unknown fields are preserved per §2.

**When to bump.**

- Bump **MINOR** when adding new optional fields, new flag vocabulary, new `scope_type` values, new codec IDs, or new `attributes` flags. Backward-compatible.
- Bump **MAJOR** when renaming or removing required fields, changing `kind` discriminator values, changing `status` enum values, or altering pairing semantics. Breaking.

v0.2 → v0.3 is a MAJOR break, not a minor bump (see Appendix A "Migration from v0.2").

---

## 7. What ATOF Is Not

- **Not ATIF.** ATIF is a higher-level trajectory format with computed ancestry, merged observations, sequenced step_ids, and turn-based structure. ATOF events are the raw observations ATIF is built from. See `atof-to-atif-converter.md`.
- **Not a metrics format.** Token counts, latency budgets, cost attribution — those live in codec-annotated structures (`annotated_response.usage`) or in downstream aggregation. ATOF does not normalize or roll up metrics.
- **Not a trace format.** ATOF is compatible with distributed tracing (subscribers can export to OpenTelemetry via `metadata.trace_id`/`metadata.span_id`) but is not itself an OTLP-equivalent wire format.
- **Not a schema-validated contract.** ATOF's contract is "emit valid events" at the envelope level. Structured shapes (LLM messages, tool arguments) are codec-shaped, not ATOF-shaped. Codecs declare shapes; ATOF carries them.
- **Not a replay executor.** An ATOF stream lets you reconstruct what happened. It does not provide the mechanism to re-run it — that's a separate layer built on top.

---

## 8. Examples

### 8.1 EXMP-01: Simple Tool Call

Single-LLM turn with one tool call. Tier-2 enrichment (semantic tagging, no codec).

```jsonl
{"kind":"ScopeStart","schema_version":"0.3","uuid":"agent-001","parent_uuid":null,"timestamp":"2026-01-01T00:00:00Z","name":"calculator_agent","attributes":[],"scope_type":"agent","input":"What is 3+4?","data":null,"metadata":null}
{"kind":"LlmStart","schema_version":"0.3","uuid":"llm-001","parent_uuid":"agent-001","timestamp":"2026-01-01T00:00:01Z","name":"gpt-4.1","attributes":[],"model_name":"gpt-4.1","codec":null,"input":{"messages":[{"role":"user","content":"What is 3+4?"}]},"annotated_request":null,"data":null,"metadata":null}
{"kind":"LlmEnd","schema_version":"0.3","uuid":"llm-001","parent_uuid":"agent-001","timestamp":"2026-01-01T00:00:02Z","name":"gpt-4.1","attributes":[],"model_name":"gpt-4.1","codec":null,"output":{"content":"","tool_calls":[{"id":"call_abc","name":"calculator__add","arguments":{"a":3,"b":4}}]},"annotated_response":null,"status":"ok","error":null,"data":null,"metadata":null}
{"kind":"ToolStart","schema_version":"0.3","uuid":"tool-001","parent_uuid":"agent-001","timestamp":"2026-01-01T00:00:03Z","name":"calculator__add","attributes":["local"],"tool_call_id":"call_abc","input":{"a":3,"b":4},"data":null,"metadata":null}
{"kind":"ToolEnd","schema_version":"0.3","uuid":"tool-001","parent_uuid":"agent-001","timestamp":"2026-01-01T00:00:04Z","name":"calculator__add","attributes":["local"],"tool_call_id":"call_abc","output":{"result":7},"status":"ok","error":null,"data":null,"metadata":null}
{"kind":"LlmStart","schema_version":"0.3","uuid":"llm-002","parent_uuid":"agent-001","timestamp":"2026-01-01T00:00:05Z","name":"gpt-4.1","attributes":[],"model_name":"gpt-4.1","codec":null,"input":{"messages":[{"role":"user","content":"What is 3+4?"},{"role":"assistant","tool_calls":[...]},{"role":"tool","tool_call_id":"call_abc","content":"7"}]},"annotated_request":null,"data":null,"metadata":null}
{"kind":"LlmEnd","schema_version":"0.3","uuid":"llm-002","parent_uuid":"agent-001","timestamp":"2026-01-01T00:00:06Z","name":"gpt-4.1","attributes":[],"model_name":"gpt-4.1","codec":null,"output":{"content":"3+4=7"},"annotated_response":null,"status":"ok","error":null,"data":null,"metadata":null}
{"kind":"ScopeEnd","schema_version":"0.3","uuid":"agent-001","parent_uuid":null,"timestamp":"2026-01-01T00:00:07Z","name":"calculator_agent","attributes":[],"scope_type":"agent","output":"3+4=7","status":"ok","error":null,"data":null,"metadata":null}
```

### 8.2 EXMP-02: Tool Error with Parent Recovery

LLM calls a tool that fails; agent catches and reports success.

```jsonl
{"kind":"ScopeStart","schema_version":"0.3","uuid":"agent-002","parent_uuid":null,"timestamp":"2026-01-02T00:00:00Z","name":"search_agent","attributes":[],"scope_type":"agent","input":"Find quantum news.","data":null,"metadata":null}
{"kind":"LlmStart","schema_version":"0.3","uuid":"llm-003","parent_uuid":"agent-002","timestamp":"2026-01-02T00:00:01Z","name":"gpt-4.1","attributes":[],"model_name":"gpt-4.1","codec":null,"input":{...},"annotated_request":null,"data":null,"metadata":null}
{"kind":"LlmEnd","schema_version":"0.3","uuid":"llm-003","parent_uuid":"agent-002","timestamp":"2026-01-02T00:00:02Z","name":"gpt-4.1","attributes":[],"model_name":"gpt-4.1","codec":null,"output":{"tool_calls":[{"id":"call_xyz","name":"web_search","arguments":{"q":"quantum"}}]},"annotated_response":null,"status":"ok","error":null,"data":null,"metadata":null}
{"kind":"ToolStart","schema_version":"0.3","uuid":"tool-002","parent_uuid":"agent-002","timestamp":"2026-01-02T00:00:03Z","name":"web_search","attributes":[],"tool_call_id":"call_xyz","input":{"q":"quantum"},"data":null,"metadata":null}
{"kind":"ToolEnd","schema_version":"0.3","uuid":"tool-002","parent_uuid":"agent-002","timestamp":"2026-01-02T00:00:08Z","name":"web_search","attributes":[],"tool_call_id":"call_xyz","output":null,"status":"error","error":{"message":"request timed out after 5s","type":"TimeoutError","traceback":null},"data":null,"metadata":null}
{"kind":"ScopeEnd","schema_version":"0.3","uuid":"agent-002","parent_uuid":null,"timestamp":"2026-01-02T00:00:10Z","name":"search_agent","attributes":[],"scope_type":"agent","output":"Unable to search; service timed out.","status":"ok","error":null,"data":null,"metadata":null}
```

Note: tool failed (`status: "error"`), but parent agent caught and reported `status: "ok"` with a helpful output — demonstrates §5.3.

### 8.3 EXMP-03: Tier-3 Codec Annotation

Same simple tool call as EXMP-01, but producer has an OpenAI Chat codec registered. `codec` is declared; `annotated_request`/`annotated_response` carry the structured shape.

```jsonl
{"kind":"LlmStart","schema_version":"0.3","uuid":"llm-004","parent_uuid":"agent-003","timestamp":"2026-01-03T00:00:01Z","name":"gpt-4.1","attributes":[],"model_name":"gpt-4.1","codec":{"name":"openai/chat-completions","version":"v1"},"input":{...raw OpenAI request JSON...},"annotated_request":{"messages":[{"role":"user","content":"What is 3+4?"}],"model":"gpt-4.1","params":{"temperature":0.7,"max_tokens":1024},"tools":[{"type":"function","function":{"name":"calculator__add","parameters":{...}}}]},"data":null,"metadata":null}
```

The `codec` field declares which codec's shape `annotated_request` conforms to; the shape itself is defined in `atof-codec-profiles.md`.

---

## 9. Design Rationale

**Why seven event kinds instead of three?** LLM and Tool calls are the dominant event types in any agentic system. Making them first-class event kinds (rather than `scope_type` values under `ScopeStart`) simplifies both producer ergonomics (typed builder with `model_name`/`tool_call_id` as typed fields) and consumer dispatch (one `match kind` handles the common cases without a nested scope_type switch). The generic `ScopeStart`/`ScopeEnd` pair remains for the long tail (agent, retriever, custom, …).

**Why string arrays for `attributes` and not bitfields?** Per Q14: JSONL round-trip robustness. Strings deduplicate obviously, sort naturally, and are legible in log inspection. A bitfield integer saves bytes but costs every consumer a decode step and obscures the wire format.

**Why closed `scope_type` enum with `custom`+`subtype`?** Q2's "producer may not know what scope this is" requires an `unknown` variant; extensibility requires either open strings or a `custom`+`subtype` pair. Open strings historically led to fragmentation (each vendor picks their own name and consumers can't safely assume vocabulary). Closed enum + namespaced `subtype` preserves extensibility without surrendering consumer dispatch cleanliness.

**Why drop the profile-contract design from v0.2?** Three reasons: (1) Q2 — producers often lack the schema body at emit time, making `$schema` an unreliable dispatch key; (2) NeMo-Flow runtime already expresses structured payloads via the codec layer (`AnnotatedLlmRequest`/`Response`), which is a stronger per-language mechanism than wire-level schema declarations; (3) the cost of StreamHeaderEvent + inline/header/opaque mode resolution never paid for itself because most consumers fall back to opaque anyway.

**Why optional codec identifier instead of required?** Tier 1 (raw pass-through) must always be legal. Requiring `codec` would break tier-1 producers. Optional `codec` is opt-in for tier-3 without taxing tier-1 or tier-2.

**Why required `status` on `*EndEvent`?** Post-hoc inspection (Q1) requires reliable failure detection. Making `status` optional would force downstream tools to infer error state from data shape — brittle and incompatible across providers. A required `status` enum is a tiny wire cost and a large correctness win.

**Why `schema_version` as required `"MAJOR.MINOR"` string?** Forward-compat negotiation needs to be machine-parseable but also human-inspectable. `"0.3"` equality is a one-line check for same-version; split-on-dot handles major/minor comparisons when needed.

**Why not inline error traces?** Tracebacks can be large and often sensitive. The `error` sub-object makes `traceback` optional; producers that want audit-grade error info emit it, producers that value privacy or stream size omit it.

---

## 10. Reference Implementations

- **Python (consumer + test-producer):** `src/nat/atof/` in `nvidia_nat_atif`. Pydantic models per event kind with `model_config = ConfigDict(extra="allow")` for lossless pass-through.
- **Rust (primary producer):** `crates/core/src/api/event.rs` in `NeMo-Flow`. Seven struct types + `#[flatten]`'d `BaseEvent` + `typed_builder::TypedBuilder` for ergonomic construction. `Event` enum with `#[serde(tag = "kind")]` for wire-format alignment.
- **Bindings:** Python/Go/Node.js/WASM bindings in NeMo-Flow re-export the Rust event types via language-idiomatic wrappers.

See `atof-codec-profiles.md` for codec registry and structured payload shapes, and `atof-to-atif-converter.md` for the normative ATOF → ATIF conversion.

---

## Appendix A: Migration from v0.2

ATOF v0.3 is a **major break** from v0.2. There is no transitional shim; v0.2 and v0.3 streams cannot be mixed in the same pipeline. Producers upgrade atomically; consumers dispatch on `schema_version` at the major level if they need to handle both.

Concrete changes:

| v0.2 construct | v0.3 replacement |
|---|---|
| Single `ScopeStart`/`ScopeEnd` + `scope_type` discriminator | 7 event kinds: `ScopeStart`/`ScopeEnd` + dedicated `LlmStart`/`LlmEnd`/`ToolStart`/`ToolEnd` |
| `profile: { $schema, $version, $mode, ...vendor fields }` | Typed fields directly on event (`model_name`, `tool_call_id`); optional `annotated_request`/`annotated_response` for tier-3 |
| `StreamHeaderEvent` (4th event kind) | Removed entirely. Codec vocabulary lives out-of-band in `atof-codec-profiles.md`. |
| `flags: list[str]` (shared vocabulary across scope types) | `attributes: list[str]` (per-event-kind vocabulary); see §2.1 |
| `scope_type` open string (any non-empty string) | Closed enum + `custom`+`subtype` for vendor extensions; `unknown` for tier-1 |
| No required `status` on `ScopeEndEvent` (implicit) | Required `status` on all `*EndEvent` kinds, per §5 |
| No `subtype` field | Required when `scope_type == "custom"` |
| Attribute vocabulary (`stateful`, `remote`) | Reverts to runtime words: `stateless`, `local`. Still lowercase string arrays on the wire. |

Migration path: regenerate streams under v0.3. The three EXMP JSONL files in `examples/atof_to_atif/output/` are rebuilt clean from the v0.3 `generate_examples.py`.
