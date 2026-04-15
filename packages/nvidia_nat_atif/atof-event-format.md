# Agentic Trajectory Observability Format (ATOF) Specification — Core

**Version:** 0.1  
**NeMo Agent Toolkit Reference Implementation:** `src/nat/atof/`

**Companion documents:**

- `[atof-codec-profiles.md](./atof-codec-profiles.md)` — codec identifiers and out-of-band JSON Schema reference shapes for structured LLM request/response and tool invocation payloads.
- `[atof-to-atif-converter.md](./atof-to-atif-converter.md)` — normative mapping from an ATOF event stream to an ATIF trajectory.

---

## 1. Overview

ATOF (Agentic Trajectory Observability Format) is the wire format for agent runtime subscriber callbacks. Events represent the lifecycle of scopes — composable units of agent work — within the runtime. Subscribers receive events in real time as the runtime executes agent workflows.

**Primary purpose:** lossless replay for inspection and evaluation. An ATOF event stream MUST carry enough information to reconstruct what happened in an agent run — identity, call graph, LLM messages in/out, tool calls and results, status — so that humans and tools can debug, audit, and evaluate the run post-hoc.

Transport is JSON Lines: one JSON object per line. The `kind` field at the top of every event is the primary discriminator. ATOF v0.1 defines **four event kinds**:

- `"StreamHeader"` — optional metadata carrier; MUST be the first event when present. Declares a codec registry for the stream (§3.4, §5.5).
- `"ScopeStart"` — a scope was opened
- `"ScopeEnd"` — a scope was closed
- `"Mark"` — a point-in-time checkpoint was recorded

`StreamHeader` is structural, not lifecycle — it carries no `scope_type` or `status` and participates in no Start/End pairing. `ScopeStart`/`ScopeEnd`/`Mark` are the lifecycle events proper.

What *kind of work* a scope represents — an LLM call, a tool invocation, an agent turn, a retriever lookup, a vendor extension — is carried by the `scope_type` field on `ScopeStart`/`ScopeEnd`. Kind-specific typed fields (`model_name`, `tool_call_id`, codec annotations) live directly on `ScopeStart`/`ScopeEnd` and are null for scope types that don't need them.

**Wire envelope example:**

```json
{"kind":"ScopeStart","schema_version":"0.1","uuid":"...","parent_uuid":"...","timestamp":"...","name":"gpt-4.1","attributes":["streaming"],"scope_type":"llm","subtype":null,"model_name":"gpt-4.1","tool_call_id":null,"codec":null,"input":{...},"annotated_request":null,"data":null,"metadata":null}
```

### 1.1 Three Producer Enrichment Tiers

ATOF is designed for progressive enrichment at the producer's discretion. A producer emits what it knows; absent fields are legal everywhere except where noted.


| Tier                    | Producer knows                                                       | Wire shape                                                                                                                      | Use case                                                                                                                                   |
| ----------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **1. Raw pass-through** | nothing semantic — just a payload                                    | event kind + envelope + opaque `input`/`output` JSON; `scope_type: "unknown"`; other typed fields are `null`                    | runtime wrapping third-party frameworks where callback provides a blob, not a classification                                               |
| **2. Semantic-tagged**  | the kind of work (LLM, tool, specific scope type)                    | typed event kind + populated `scope_type` + kind-appropriate typed fields (`model_name`, `tool_call_id`, `attributes`)          | native agent runtimes emitting their own events; framework wrappers that can classify at the hook site                                     |
| **3. Codec-annotated**  | the structured shape of the payload (messages, params, tool defs, …) | tier-2 plus `codec: {name, version}` identifier and optional `annotated_request`/`annotated_response` structured representation | producer has a registered codec (OpenAI Chat, Anthropic Messages, NVIDIA NIM, …) that decodes raw provider JSON into a canonical structure |


**Design principle:** Tier 1 must always work. A consumer that doesn't understand tier-2 or tier-3 enrichment MUST still preserve the event verbatim. Consumers SHOULD NOT reject events whose `scope_type` or `codec` name they don't recognize — unknown values are forward-compat extensions, not errors.

### 1.2 The Three Structured Fields at a Glance

Beyond the base envelope (`kind`, `uuid`, `parent_uuid`, `timestamp`, `name`, `schema_version`), every lifecycle event carries three structured fields:


|                       | Spec-governed shape                                                                                                                                               | Opaque to ATOF     |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **About the work**    | `input` (ScopeStart), `output` (ScopeEnd), `attributes`, `scope_type`, `subtype`, `model_name`, `tool_call_id`, `codec`, `annotated_request`/`annotated_response` | —                  |
| **About the context** | —                                                                                                                                                                 | `data`, `metadata` |


- `input` / `output` — raw payload (post-guardrail). Opaque to the ATOF core; structure MAY be defined by a codec (tier-3) and carried as `annotated_request` / `annotated_response`.
- `attributes` — behavioral flag array. Vocabulary is shared across scope types (see §2.1); per-flag applicability is documented with each flag.
- `scope_type` — semantic category of the scope. Closed enum (see §4).
- `subtype` — free-form vendor name when `scope_type == "custom"`.
- `model_name` — LLM model identifier. Populated when `scope_type == "llm"`; null otherwise.
- `tool_call_id` — tool-call correlation ID. Populated when `scope_type == "tool"`; null otherwise.
- `codec`, `annotated_request`, `annotated_response` — tier-3 codec annotations. Optional; see `atof-codec-profiles.md`.
- `data` — application-defined payload. Opaque to ATOF. Consumers MUST NOT dispatch on `data` contents.
- `metadata` — tracing/correlation envelope (`trace_id`, `span_id`, etc.).

---

## 2. Base Event Envelope

Every event carries these six envelope fields. `ScopeStart`/`ScopeEnd` add fields on top; `Mark` adds only `data` / `metadata`; `StreamHeader` adds the `codecs` registry (§3.4).


| Field            | Type                                    | Required | Description                                                                                                             |
| ---------------- | --------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------- |
| `kind`           | string                                  | Yes      | Event kind discriminator. One of: `"ScopeStart"`, `"ScopeEnd"`, `"Mark"`, `"StreamHeader"`.                             |
| `schema_version` | string                                  | Yes      | ATOF protocol version, `"MAJOR.MINOR"` (e.g., `"0.1"`). See §6.6.                                                       |
| `uuid`           | string (UUID)                           | Yes      | Unique identifier for this event or span. For `ScopeStart`/`ScopeEnd` pairs, the Start and End share a `uuid`.          |
| `parent_uuid`    | string (UUID) or null                   | No       | UUID of the containing scope when this event was emitted. Null only for root scope events and unparented `Mark` events. |
| `timestamp`      | string (RFC 3339) or integer (epoch µs) | Yes      | Wall-clock time the event was emitted. See §6.1.                                                                        |
| `name`           | string                                  | Yes      | Human-readable label — e.g., `"my_agent"`, `"calculator__add"`, `"gpt-4.1"`.                                            |
| `data`           | object or null                          | No       | Application-defined payload. Opaque to ATOF.                                                                            |
| `metadata`       | object or null                          | No       | Tracing/correlation envelope — e.g., `{"trace_id": "...", "span_id": "..."}`.                                           |


### 2.1 `attributes` — behavioral flag array

`attributes` is a cross-cutting field on `ScopeStart` and `ScopeEnd`. `Mark` does NOT carry `attributes`.


| Field        | Type             | Required | Description                                                                                    |
| ------------ | ---------------- | -------- | ---------------------------------------------------------------------------------------------- |
| `attributes` | array of strings | Yes      | Canonical lowercase flag names (sorted, deduplicated). Empty array `[]` when no flags are set. |


Producers MUST emit `attributes` in lexicographic order with no duplicates. Consumers SHOULD treat the array as an unordered set and MUST preserve unknown flag names when re-emitting. Unknown flags SHOULD NOT be treated as errors.

**Canonical flag vocabulary** (shared across all scope_types; individual flag applicability noted):


| Flag            | Applies when                     | Meaning (when present)                                                                                           |
| --------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `"parallel"`    | any `scope_type`                 | Scope executes concurrently with sibling scopes under the same parent.                                           |
| `"relocatable"` | any `scope_type`                 | Scope may be moved across async task boundaries (e.g., between threads or event loops) without losing context.   |
| `"stateless"`   | `scope_type == "llm"` primarily  | Scope does not maintain state between invocations — same inputs produce the same behavior regardless of history. |
| `"streaming"`   | `scope_type == "llm"` primarily  | Scope produces its output incrementally as chunks, rather than as a single payload at exit.                      |
| `"local"`       | `scope_type == "tool"` primarily | Tool executes in the same process as the runtime, not dispatched to a remote service.                            |


**Why defaults are "absence":** Each flag describes the exceptional case. Absence means the default applies — serial (not parallel), pinned (not relocatable), stateful (not stateless), single-payload (not streaming), remote (not local).

**Flag extensibility.** Implementations MAY emit additional flag names for vendor extensions; non-canonical flags SHOULD be namespaced with a dotted prefix — for example, `"nvidia.speculative"`. Consumers MUST preserve unknown flag strings and MUST NOT reject events carrying them.

**Streaming + terminal status.** For scopes carrying the `"streaming"` flag: if the scope terminates with `status == "error"` or `status == "cancelled"` (§5.1), the `output` on `ScopeEnd` MAY contain partial chunks accumulated before the terminal event. Consumers that replay streaming output MUST check `status` before treating `output` as a complete payload.

---

## 3. Event Kinds

### 3.1 `ScopeStartEvent`

Emitted when a scope is pushed onto the active scope stack.


| Field               | Type                  | Required | Description                                                                                                                                                                             |
| ------------------- | --------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`              | string                | Yes      | Literal `"ScopeStart"`.                                                                                                                                                                 |
| `schema_version`    | string                | Yes      | See §2.                                                                                                                                                                                 |
| `uuid`              | string (UUID)         | Yes      | See §2.                                                                                                                                                                                 |
| `parent_uuid`       | string (UUID) or null | No       | See §2. Null on the root scope.                                                                                                                                                         |
| `timestamp`         | string or integer     | Yes      | See §2.                                                                                                                                                                                 |
| `name`              | string                | Yes      | See §2.                                                                                                                                                                                 |
| `attributes`        | array of strings      | Yes      | See §2.1.                                                                                                                                                                               |
| `scope_type`        | string                | Yes      | Semantic category. See §4.                                                                                                                                                              |
| `subtype`           | string or null        | No       | Free-form vendor name. REQUIRED when `scope_type == "custom"`; SHOULD be absent otherwise. See §4.2.                                                                                    |
| `model_name`        | string or null        | No       | LLM model identifier. Populated when `scope_type == "llm"` and the producer knows the model name; null otherwise.                                                                       |
| `tool_call_id`      | string or null        | No       | Tool-call correlation ID. Populated when `scope_type == "tool"` and the tool was invoked via an LLM tool-use flow; null otherwise.                                                      |
| `codec`             | object or null        | No       | Codec identifier `{name: string, version: string}` declaring which codec shaped `annotated_request`. See `atof-codec-profiles.md`. Applicable when `scope_type` is `"llm"` or `"tool"`. |
| `input`             | any or null           | No       | Raw input payload (post-guardrail). Opaque to ATOF core.                                                                                                                                |
| `annotated_request` | object or null        | No       | Structured codec-decoded view of the input. Shape declared by `codec`. See `atof-codec-profiles.md`.                                                                                    |
| `data`              | object or null        | No       | See §2.                                                                                                                                                                                 |
| `metadata`          | object or null        | No       | See §2.                                                                                                                                                                                 |


### 3.2 `ScopeEndEvent`

Emitted when a scope is popped from the active scope stack. Paired 1:1 with `ScopeStart` by `uuid`.


| Field                | Type                  | Required | Description                                                                                              |
| -------------------- | --------------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| `kind`               | string                | Yes      | Literal `"ScopeEnd"`.                                                                                    |
| `schema_version`     | string                | Yes      | See §2.                                                                                                  |
| `uuid`               | string (UUID)         | Yes      | Same UUID as the paired `ScopeStart`.                                                                    |
| `parent_uuid`        | string (UUID) or null | No       | Same as the paired `ScopeStart`.                                                                         |
| `timestamp`          | string or integer     | Yes      | See §2. Differs from the `ScopeStart` timestamp (End occurs later).                                      |
| `name`               | string                | Yes      | Same as the paired `ScopeStart`.                                                                         |
| `attributes`         | array of strings      | Yes      | Same as the paired `ScopeStart`.                                                                         |
| `scope_type`         | string                | Yes      | Same as the paired `ScopeStart`.code                                                                     |
| `subtype`            | string or null        | No       | Same as the paired `ScopeStart`.                                                                         |
| `model_name`         | string or null        | No       | Same as the paired `ScopeStart`, or the actually-used model if different (e.g., after provider routing). |
| `tool_call_id`       | string or null        | No       | Same as the paired `ScopeStart`.                                                                         |
| `codec`              | object or null        | No       | Same as the paired `ScopeStart`, applied to `annotated_response` on this event.                          |
| `output`             | any or null           | No       | Raw output payload (post-guardrail). Opaque to ATOF core.                                                |
| `annotated_response` | object or null        | No       | Structured codec-decoded view of the output. Shape declared by `codec`. See `atof-codec-profiles.md`.    |
| `status`             | string (enum)         | Yes      | Terminal outcome. One of: `"ok"`, `"error"`, `"cancelled"`. See §5.                                      |
| `error`              | object or null        | No       | Structured error info when `status == "error"`. See §5.1.                                                |
| `data`               | object or null        | No       | See §2.                                                                                                  |
| `metadata`           | object or null        | No       | See §2.                                                                                                  |


### 3.3 `MarkEvent`

Emitted as a point-in-time checkpoint. Unpaired (no Start/End semantics).


| Field            | Type                  | Required | Description                                                               |
| ---------------- | --------------------- | -------- | ------------------------------------------------------------------------- |
| `kind`           | string                | Yes      | Literal `"Mark"`.                                                         |
| `schema_version` | string                | Yes      | See §2.                                                                   |
| `uuid`           | string (UUID)         | Yes      | See §2.                                                                   |
| `parent_uuid`    | string (UUID) or null | No       | See §2.                                                                   |
| `timestamp`      | string or integer     | Yes      | See §2.                                                                   |
| `name`           | string                | Yes      | Label for the checkpoint — e.g., `"workflow_start"`, `"retry_attempt_2"`. |
| `data`           | object or null        | No       | Optional checkpoint payload.                                              |
| `metadata`       | object or null        | No       | See §2.                                                                   |


`MarkEvent` does NOT carry `attributes`, `scope_type`, `status`, `input`, or `output`. It is deliberately minimal — a named timestamp with optional data.

### 3.4 `StreamHeaderEvent`

Optional structural event carrying stream-level metadata — specifically, the codec registry used by the 4-priority codec resolution chain (see `atof-codec-profiles.md` §6 for the full protocol). The `StreamHeader`, when present, MUST be the first event in the stream; exactly one `StreamHeader` is permitted per stream. If the first event is not a `StreamHeader`, no stream-level codec registry exists and events fall back to priority-3 (consumer-bundled) or priority-4 (opaque) resolution.


| Field            | Type                  | Required | Description                                                                                                    |
| ---------------- | --------------------- | -------- | -------------------------------------------------------------------------------------------------------------- |
| `kind`           | string                | Yes      | Literal `"StreamHeader"`.                                                                                      |
| `schema_version` | string                | Yes      | See §2.                                                                                                        |
| `uuid`           | string (UUID)         | Yes      | See §2.                                                                                                        |
| `parent_uuid`    | string (UUID) or null | No       | SHOULD be null — `StreamHeader` is not nested under any scope.                                                 |
| `timestamp`      | string or integer     | Yes      | See §2. Typically the stream's opening timestamp.                                                              |
| `name`           | string                | Yes      | Human-readable label — e.g., `"stream_header"`, `"exmp01_header"`.                                             |
| `codecs`         | object                | No       | Codec registry keyed by canonical `{name}.v{version}` string. Each value is a `CodecEntry` object (see below). |
| `data`           | object or null        | No       | See §2.                                                                                                        |
| `metadata`       | object or null        | No       | See §2.                                                                                                        |


`StreamHeaderEvent` does NOT carry `attributes`, `scope_type`, `subtype`, `status`, `error`, `input`, `output`, `model_name`, `tool_call_id`, `codec`, or `annotated_request` / `annotated_response`.

`**CodecEntry` shape:**

```json
{
  "$schema": { /* optional inline JSON Schema body */ }
}
```

An entry with an inline `$schema` makes the stream self-sufficient for that codec — consumers validate `annotated_request` / `annotated_response` against the inline body. An entry without `$schema` (empty `{}`) is a **manifest declaration**: the producer is announcing "this stream uses codec X"; consumers resolve the schema body from their own bundled registry (priority 3). Manifest declarations help consumers surface early warnings when a declared codec isn't in their local registry.

**Example `StreamHeader`:**

```json
{"kind":"StreamHeader","schema_version":"0.1","uuid":"hdr-001","parent_uuid":null,"timestamp":"2026-01-01T00:00:00Z","name":"stream_header","codecs":{"openai/chat-completions.v1":{},"nvidia/llm.v1":{"$schema":{"$id":"nvidia/llm.v1","type":"object","properties":{"model_name":{"type":["string","null"]}}}}},"data":null,"metadata":null}
```

---

## 4. `scope_type` Vocabulary

`scope_type` on `ScopeStart`/`ScopeEnd` classifies the kind of work the scope represents. The canonical vocabulary is a closed set of lowercase strings:


| `scope_type` value | Meaning                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------- |
| `"agent"`          | Top-level agent or workflow scope.                                                        |
| `"function"`       | Generic function or application step.                                                     |
| `"llm"`            | LLM call scope. Populates `model_name` and MAY populate `codec` + `annotated`_*.          |
| `"tool"`           | Tool invocation scope. Populates `tool_call_id` and MAY populate `codec` + `annotated`_*. |
| `"retriever"`      | Retrieval step (document search, index lookup).                                           |
| `"embedder"`       | Embedding-generation step.                                                                |
| `"reranker"`       | Result reranking step.                                                                    |
| `"guardrail"`      | Guardrail or validation step.                                                             |
| `"evaluator"`      | Evaluation or scoring step.                                                               |
| `"custom"`         | Vendor-defined custom category. REQUIRES `subtype` to name the vendor scope.              |
| `"unknown"`        | Producer does not know or cannot classify the scope.                                      |


### 4.1 `"unknown"` is the tier-1 escape hatch

Producers that have a payload but no classification (the tier-1 pass-through case from §1.1) emit `scope_type: "unknown"`. This is ALWAYS valid. Consumers SHOULD NOT reject events with `scope_type: "unknown"`.

### 4.2 `subtype` when `scope_type == "custom"`

When `scope_type == "custom"`, the event MUST carry a `subtype: string` field naming the vendor scope category. `subtype` SHOULD follow a dotted-namespace convention to avoid collisions — for example:

- `"nvidia.speculative_decode"`
- `"langchain.memory_retrieval"`
- `"internal.audit_gate"`

When `scope_type != "custom"`, `subtype` SHOULD be absent or null. Consumers SHOULD preserve `subtype` verbatim on re-emission.

### 4.3 Extensibility

The `scope_type` enum is closed but `"custom"` + `subtype` provides unbounded vendor expressiveness. ATOF reserves the right to promote frequently-used `subtype` values into first-class `scope_type` vocabulary entries in future versions (backward-compat MINOR bump).

---

## 5. Status and Error Semantics

### 5.1 Terminal status

Every `ScopeEnd` carries a required `status` field valued in:

- `"ok"` — the scope returned normally.
- `"error"` — the scope raised an exception or equivalent failure.
- `"cancelled"` — the scope was terminated before completion (timeout, parent cancel, explicit cancel).

`MarkEvent` does not carry `status` — marks are not lifecycle events.

When `status == "error"`, an optional `error` field MAY carry structured error info:


| Field       | Type           | Description                                                                             |
| ----------- | -------------- | --------------------------------------------------------------------------------------- |
| `message`   | string         | Human-readable error description.                                                       |
| `type`      | string or null | Error class or category — e.g., `"TimeoutError"`, `"ValidationError"`, `"RateLimited"`. |
| `traceback` | string or null | Optional stack trace or debug trace.                                                    |


When `status != "error"`, `error` SHOULD be absent or null.

### 5.2 Cascading cancellation

When a parent cancels its children (e.g., due to parent timeout or parent error recovery), each child emits its own `ScopeEnd` with `status == "cancelled"` before the parent emits its `ScopeEnd`. §6.3 still holds: all child events precede the parent's `ScopeEnd` in wall-clock order. The parent's own `status` reflects the parent's outcome:

- `"cancelled"` if the parent was itself cancelled from above.
- `"error"` if the parent raised (possibly because a child's failure propagated).
- `"ok"` if the parent chose to cancel its children as normal control flow and then completed.

### 5.3 Each scope reports its own terminal status

The `status` on a `ScopeEnd` describes the outcome of THAT scope, not its children or its parent. A parent whose child errored MAY itself report `status == "ok"` if the parent caught and handled the child's error.

### 5.4 Dangling scopes

If the runtime dies before emitting a paired `ScopeEnd`, no event appears in the stream. §6.3's pairing guarantee is contingent on orderly shutdown. Consumers that detect an unpaired `ScopeStart` after the stream ends MAY synthesize a `ScopeEnd` with `status == "cancelled"` for downstream processing; such synthetic events are out of scope for ATOF Core.

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

The root scope has `parent_uuid = null`. This is the only `ScopeStart` in a well-formed stream that may have a null `parent_uuid` (once the root scope is established). `MarkEvent`s MAY carry `parent_uuid = null` when emitted outside any scope.

### 6.3 Start/End Pairing

Every `ScopeStartEvent` is paired with exactly one `ScopeEndEvent` sharing the same `uuid`. The `ScopeEnd` is always emitted strictly after the `ScopeStart` (strict: `ts_micros(End) > ts_micros(Start)`).

`MarkEvent`s have no paired event — they are single-shot.

### 6.4 UUID Uniqueness

Each scope span receives a unique UUID at creation time. The `uuid` is stable across the Start and End events for the same scope. In the Rust reference implementation, UUIDs are v7 (time-ordered).

### 6.5 ID Relationships

Three distinct identifier namespaces appear in an ATOF stream:

- `**uuid` / `parent_uuid`** — runtime identifiers attached to every event. Form the scope graph.
- `**tool_call_id`** (on `ScopeStart`/`ScopeEnd` when `scope_type == "tool"`) — an LLM-provider identifier that bridges an LLM's tool-call response with the resulting tool execution. Null when the tool was not invoked via an LLM tool-use flow.
- **Codec-decoded response IDs** (e.g., `chatcmpl-`* inside a decoded LLM response body, under `annotated_response`) — provider tracking identifiers. Opaque to ATOF Core; see `atof-codec-profiles.md`.

### 6.6 Schema Version and Negotiation

Every event carries a required `schema_version` field, formatted `"MAJOR.MINOR"` — e.g., `"0.1"`. This section defines when producers bump the version and how consumers dispatch on it.

**Reading rules.** Consumers SHOULD accept any `0.Y` event as ATOF-v0-family. Major-version bumps (`1.0`, `2.0`) MAY introduce breaking changes; consumers that want forward compat MUST dispatch on the major version and fail fast on unknown majors.

**Mixed-version streams.** A single stream MAY contain events at different minor versions (`0.1` and `0.2`). Consumers MUST NOT reject a stream because it contains newer minor versions than expected; unknown fields are preserved per §2.

**When to bump.**

- Bump **MINOR** when adding new optional fields, new flag vocabulary, new `scope_type` values, new codec IDs, or new `attributes` flags. Backward-compatible.
- Bump **MAJOR** when renaming or removing required fields, changing `kind` discriminator values, changing `status` enum values, or altering pairing semantics. Breaking.

---

## 7. What ATOF Is Not

- **Not ATIF.** ATIF is a higher-level trajectory format with computed ancestry, merged observations, sequenced step_ids, and turn-based structure. ATOF events are the raw observations ATIF is built from. See `atof-to-atif-converter.md`.
- **Not a metrics format.** Token counts, latency budgets, cost attribution — those live in codec-annotated structures (`annotated_response.usage`) or in downstream aggregation. ATOF does not normalize or roll up metrics.
- **Not a trace format.** ATOF is compatible with distributed tracing (subscribers can export to OpenTelemetry via `metadata.trace_id`/`metadata.span_id`) but is not itself an OTLP-equivalent wire format.
- **Not a schema-validated contract.** ATOF's contract is "emit valid events" at the envelope level. Structured shapes (LLM messages, tool arguments) are codec-shaped, not ATOF-shaped. Codecs declare shapes; ATOF carries them.
- **Not a replay executor.** An ATOF stream lets you reconstruct what happened. It does not provide the mechanism to re-run it — that's a separate layer built on top.

---

## 8. Reference Implementations

- **Python (consumer + test-producer):** `src/nat/atof/` in `nvidia_nat_atif`. Pydantic models per event kind with `model_config = ConfigDict(extra="allow")` for lossless pass-through.
- **Producer runtimes:** Agent runtimes emitting ATOF MAY use more granular internal types (e.g., separate `LlmStartEvent`/`ToolStartEvent` structs in typed languages) for type-safe construction, but MUST serialize to ATOF's three-kind wire format on emission.
- **Language bindings:** Where a producer runtime exposes bindings to additional languages, those bindings SHOULD re-export the runtime's event types via language-idiomatic wrappers while preserving the wire format on serialization.

See `atof-codec-profiles.md` for codec registry and structured payload shapes, and `atof-to-atif-converter.md` for the normative ATOF → ATIF conversion.