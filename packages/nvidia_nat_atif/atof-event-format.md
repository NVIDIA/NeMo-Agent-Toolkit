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

# Agentic Trajectory Observability Format (ATOF) Specification — Core

**Version:** 0.1  
**NeMo Agent Toolkit Reference Implementation:** `src/nat/atof/`

**Companion documents:**

- [`examples/atof_to_atif/README.md`](./examples/atof_to_atif/README.md) — ATOF → ATIF conversion reference, mapping table, and runnable examples.

---

## 1. Overview

ATOF (Agentic Trajectory Observability Format) is the wire format for agent runtime subscriber callbacks. Events represent the lifecycle of scopes — composable units of agent work — within the runtime. Subscribers receive events in real time as the runtime executes agent workflows.

**Primary purpose:** lossless replay for inspection and evaluation. An ATOF event stream MUST carry enough information to reconstruct what happened in an agent run — identity, call graph, LLM messages in/out, tool calls and results — so that humans and tools can debug, audit, and evaluate the run post-hoc.

Transport is JSON Lines: one JSON object per line. The `kind` field at the top of every event is the primary discriminator. ATOF v0.1 defines **two event kinds**:

- `"scope"` — a scope lifecycle event (start or end, distinguished by `scope_category`)
- `"mark"` — a point-in-time checkpoint was recorded

A `scope` event carries a required `scope_category` field valued in `"start"` or `"end"`. A start/end pair shares the same `uuid` (§5.3).

What *kind of work* an event represents — an LLM call, a tool invocation, an agent turn, a retriever lookup, a vendor extension — is carried by the `category` field. Kind-specific typed fields (`model_name` for `llm`, `tool_call_id` for `tool`, `subtype` for `custom`, future fields for other categories) are packaged into a single optional `category_profile` object. The `category_profile` is `null` for tier-1 opaque events and for categories with no kind-specific fields; tier-2 producers populate the keys appropriate to the `category`. Keeping the profile as a sub-object keeps the envelope flat and extensible — adding a retriever profile shape in the future does not bloat the top-level JSON.

`category` is REQUIRED on `scope` events and OPTIONAL on `mark` events. A `mark` event MAY carry a `category` to indicate that the checkpoint relates to a particular kind of work (e.g., an `"llm"` mark); when absent, the mark is a generic checkpoint.

**Wire envelope example:**

```json
{"kind":"scope","scope_category":"start","atof_version":"0.1","uuid":"...","parent_uuid":"...","timestamp":"...","name":"agent001","attributes":["streaming"],"category":"llm","category_profile":{"model_name":"gpt-4.1"},"data":{...},"data_schema":null,"metadata":null}
```

### 1.1 Two Producer Enrichment Tiers

ATOF is designed for progressive enrichment at the producer's discretion. A producer emits what it knows; absent fields are legal everywhere except where noted.


| Tier                    | Producer knows                                  | Wire shape                                                                                                                                                | Use case                                                                                               |
| ----------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **1. Raw pass-through** | nothing semantic — just a payload               | event kind + envelope + opaque `data` JSON; `category: "unknown"` (scope) or absent (mark); `category_profile: null`                                      | runtime wrapping third-party frameworks where the callback provides a blob, not a classification           |
| **2. Semantic-tagged**  | the kind of work (LLM, tool, specific category) | typed event kind + populated `category` + kind-appropriate `category_profile` keys (`model_name`, `tool_call_id`, `subtype`, …) + `attributes` (on scope) | native agent runtimes emitting their own events; framework wrappers that can classify at the hook site |


**Design principle:** Tier 1 must always work. A consumer that doesn't understand tier-2 enrichment MUST still preserve the event verbatim. Consumers SHOULD NOT reject events whose `category` they don't recognize — unknown values are forward-compat extensions, not errors.

### 1.2 The Structured Fields at a Glance

Beyond the base envelope (`kind`, `uuid`, `parent_uuid`, `timestamp`, `name`, `atof_version`), ATOF events carry these structured fields:


| Spec-governed shape                                                                                        | Opaque to ATOF                    |
| ---------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `scope_category` (scope), `attributes` (scope), `category` (scope, mark), `category_profile` (scope, mark) | `data`, `data_schema`, `metadata` |


- `scope_category` — lifecycle phase of a `scope` event. Closed enum: `"start"` or `"end"`.
- `attributes` — behavioral flag array. Vocabulary is shared across categories (see §2.1); per-flag applicability is documented with each flag. Carried by `scope` events only.
- `category` — semantic category of the work. Closed enum (see §4). Required on `scope`, optional on `mark`.
- `category_profile` — category-specific typed fields packaged as a sub-object. Keys vary by `category` — `subtype` for `custom`, `model_name` for `llm`, `tool_call_id` for `tool`, additional keys reserved for future categories (see §4.4). Null for tier-1 opaque events and for categories with no kind-specific fields.
- `data` — application-defined payload. Opaque to ATOF. On `scope` events, typically carries the scope's input on `scope_category: "start"` and the scope's output on `scope_category: "end"`. Consumers MUST NOT dispatch on `data` contents.
- `data_schema` — optional identifier `{name: string, version: string}` describing the shape of `data`. Opaque to ATOF core; the producer declares it, and validation of `data` against the named schema is the consumer's responsibility. The reference ATOF→ATIF converter provides two registries keyed on this identifier: `nat.atof.schemas` for JSON Schema validators and `nat.atof.extractors` for payload parsers. See [examples/atof_to_atif/README.md](examples/atof_to_atif/README.md#extending-the-converter) for registration guidance.
- `metadata` — tracing/correlation envelope (`trace_id`, `span_id`, etc.).

---

## 2. Base Event Envelope

Every event carries the envelope fields below. The first six (`kind`, `atof_version`, `uuid`, `parent_uuid`, `timestamp`, `name`) are the structural identity of the event; `data`, `data_schema`, and `metadata` are common optional fields that MAY appear on any event. `scope` events add scope fields on top; `mark` events MAY carry `category` + `category_profile` (§3.2) and nothing else beyond this envelope.


| Field          | Type                                    | Required | Description                                                                                                                                           |
| -------------- | --------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`         | string                                  | Yes      | Event kind discriminator. One of: `"scope"`, `"mark"`.                                                                                                |
| `atof_version` | string                                  | Yes      | ATOF protocol version, `"MAJOR.MINOR"` (e.g., `"0.1"`). See §5.6.                                                                                     |
| `uuid`         | string (UUID)                           | Yes      | Unique identifier for this event or span. For `scope` start/end pairs, the two events share a `uuid`.                                                 |
| `parent_uuid`  | string (UUID) or null                   | No       | UUID of the containing scope when this event was emitted. Null only for root scope events and unparented `mark` events.                               |
| `timestamp`    | string (RFC 3339) or integer (epoch µs) | Yes      | Wall-clock time the event was emitted. See §5.1.                                                                                                      |
| `name`         | string                                  | Yes      | Human-readable label — e.g., `"my_agent"`, `"calculator__add"`, `"gpt-4.1"`.                                                                          |
| `data`         | object or null                          | No       | Application-defined payload. Opaque to ATOF.                                                                                                          |
| `data_schema`  | object or null                          | No       | Schema identifier `{name: string, version: string}` describing the shape of `data`. Opaque to ATOF core; validation is the consumer's responsibility. |
| `metadata`     | object or null                          | No       | Tracing/correlation envelope — e.g., `{"trace_id": "...", "span_id": "..."}`.                                                                         |


### 2.1 `attributes` — behavioral flag array

`attributes` is a cross-cutting field on `scope` events. `mark` does NOT carry `attributes`.


| Field        | Type             | Required | Description                                                                                    |
| ------------ | ---------------- | -------- | ---------------------------------------------------------------------------------------------- |
| `attributes` | array of strings | Yes      | Canonical lowercase flag names (sorted, deduplicated). Empty array `[]` when no flags are set. |


Producers MUST emit `attributes` in lexicographic order with no duplicates. Consumers SHOULD treat the array as an unordered set and MUST preserve unknown flag names when re-emitting. Unknown flags SHOULD NOT be treated as errors.

**Canonical flag vocabulary** (shared across all categories; individual flag applicability noted):


| Flag            | Applies when                                      | Meaning (when present)                                                                                         |
| --------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `"parallel"`    | any `category`                                    | Scope executes concurrently with sibling scopes under the same parent.                                         |
| `"relocatable"` | any `category`                                    | Scope may be moved across async task boundaries (e.g., between threads or event loops) without losing context. |
| `"stateful"`    | `category == "llm"` primarily, but not exclusive  | Scope maintains state between invocations — server-side memory, session history, or accumulated scratchpad.    |
| `"streaming"`   | `category == "llm"` primarily, but not exclusive  | Scope produces its output incrementally as chunks, rather than as a single payload at exit.                    |
| `"remote"`      | `category == "tool"` primarily, but not exclusive | Tool executes out-of-process — dispatched to a remote service (HTTP, MCP server, subprocess), not in-process.  |


**Why defaults are "absence":** Each flag describes the exceptional case. Absence means the default applies — serial (not parallel), pinned (not relocatable), stateless (not stateful), single-payload (not streaming), local (not remote).

**Flag extensibility.** Implementations MAY emit additional flag names for vendor extensions; non-canonical flags SHOULD be namespaced with a dotted prefix — for example, `"nvidia.speculative"`. Consumers MUST preserve unknown flag strings and MUST NOT reject events carrying them.

---

## 3. Event Kinds

### 3.1 `scope` event

Emitted at scope lifecycle transitions. A single scope span produces two `scope` events sharing the same `uuid`: one with `scope_category: "start"` when the scope is pushed onto the active scope stack, and one with `scope_category: "end"` when the scope is popped.


| Field              | Type                  | Required | Description                                                                                                                                                                                                                                                |
| ------------------ | --------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`             | string                | Yes      | Literal `"scope"`.                                                                                                                                                                                                                                         |
| `scope_category`   | string (enum)         | Yes      | Lifecycle phase. One of: `"start"`, `"end"`.                                                                                                                                                                                                               |
| `atof_version`     | string                | Yes      | See §2.                                                                                                                                                                                                                                                    |
| `uuid`             | string (UUID)         | Yes      | Shared between the start and end events for the same scope span.                                                                                                                                                                                           |
| `parent_uuid`      | string (UUID) or null | No       | See §2. Null on the root scope. Same on both start and end.                                                                                                                                                                                                |
| `timestamp`        | string or integer     | Yes      | See §2. The end event's timestamp is always strictly later than the start event's (see §5.3).                                                                                                                                                              |
| `name`             | string                | Yes      | See §2. Same on both start and end.                                                                                                                                                                                                                        |
| `attributes`       | array of strings      | Yes      | See §2.1. Same on both start and end.                                                                                                                                                                                                                      |
| `category`         | string                | Yes      | Semantic category. See §4. Same on both start and end.                                                                                                                                                                                                     |
| `category_profile` | object or null        | No       | Category-specific typed fields. Keys depend on `category`. See §4.4. On `scope_category: "end"`, `model_name` MAY reflect the actually-used model if different from the requested one (e.g., after provider routing).                                      |
| `data`             | object or null        | No       | See §2. Typically carries the scope's input on `scope_category: "start"` and the scope's output on `scope_category: "end"`, but producers MAY populate it on either phase.                                                                                 |
| `data_schema`      | object or null        | No       | See §2.                                                                                                                                                                                                                                                    |
| `metadata`         | object or null        | No       | See §2.                                                                                                                                                                                                                                                    |


### 3.2 `mark` event

Emitted as a point-in-time checkpoint. Unpaired (no start/end semantics). A `mark` MAY carry `category` + `category_profile` to indicate the kind of work the checkpoint relates to; when both are absent, the mark is a generic named timestamp.


| Field              | Type                  | Required | Description                                                                                                                                           |
| ------------------ | --------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`             | string                | Yes      | Literal `"mark"`.                                                                                                                                     |
| `atof_version`     | string                | Yes      | See §2.                                                                                                                                               |
| `uuid`             | string (UUID)         | Yes      | See §2.                                                                                                                                               |
| `parent_uuid`      | string (UUID) or null | No       | See §2.                                                                                                                                               |
| `timestamp`        | string or integer     | Yes      | See §2.                                                                                                                                               |
| `name`             | string                | Yes      | Label for the checkpoint — e.g., `"workflow_start"`, `"retry_attempt_2"`.                                                                             |
| `category`         | string or null        | No       | Semantic category. See §4. Null or absent means the mark is a generic checkpoint.                                                                     |
| `category_profile` | object or null        | No       | Category-specific typed fields. Keys depend on `category`. See §4.4. REQUIRED when `category == "custom"` (must carry `category_profile.subtype`).    |
| `data`             | object or null        | No       | Optional checkpoint payload.                                                                                                                          |
| `data_schema`      | object or null        | No       | Schema identifier `{name: string, version: string}` describing the shape of `data`. Opaque to ATOF core; validation is the consumer's responsibility. |
| `metadata`         | object or null        | No       | See §2.                                                                                                                                               |


`mark` does NOT carry `scope_category` or `attributes`.

---

## 4. `category` Vocabulary

`category` classifies the kind of work an event represents. The canonical vocabulary is a closed set of lowercase strings:


| `category` value | Meaning                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| `"agent"`        | Top-level agent or workflow scope.                                                               |
| `"function"`     | Generic function or application step.                                                            |
| `"llm"`          | LLM call. Populates `category_profile.model_name`.                                               |
| `"tool"`         | Tool invocation. Populates `category_profile.tool_call_id`.                                      |
| `"retriever"`    | Retrieval step (document search, index lookup).                                                  |
| `"embedder"`     | Embedding-generation step.                                                                       |
| `"reranker"`     | Result reranking step.                                                                           |
| `"guardrail"`    | Guardrail or validation step.                                                                    |
| `"evaluator"`    | Evaluation or scoring step.                                                                      |
| `"custom"`       | Vendor-defined custom category. REQUIRES `category_profile.subtype` to name the vendor category. |
| `"unknown"`      | Producer does not know or cannot classify the work.                                              |


`category` is REQUIRED on `scope` events. On `mark` events it is OPTIONAL — producers MAY omit it to emit a generic checkpoint, or populate it to tag the mark with the kind of work it relates to.

### 4.1 `"unknown"` is the tier-1 escape hatch

On `scope` events, producers that have a payload but no classification (the tier-1 pass-through case from §1.1) emit `category: "unknown"`. This is ALWAYS valid. Consumers SHOULD NOT reject events with `category: "unknown"`.

On `mark` events, the tier-1 equivalent is simply omitting `category` (since it is optional). Producers MAY still emit `category: "unknown"` explicitly to signal "I know about the category field but cannot classify this mark."

### 4.2 `category_profile.subtype` when `category == "custom"`

When `category == "custom"`, the event MUST carry `category_profile.subtype: string` naming the vendor category. The `subtype` string SHOULD follow a dotted-namespace convention to avoid collisions — for example:

- `"nvidia.speculative_decode"`
- `"langchain.memory_retrieval"`
- `"internal.audit_gate"`

This rule applies to both `scope` and `mark` events.

When `category != "custom"`, `category_profile.subtype` SHOULD be absent. Consumers SHOULD preserve the `category_profile` object verbatim on re-emission.

### 4.3 Extensibility

The `category` enum is closed but `"custom"` + `category_profile.subtype` provides unbounded vendor expressiveness. ATOF reserves the right to promote frequently-used `subtype` values into first-class `category` vocabulary entries in future versions (backward-compat MINOR bump).

### 4.4 The `category_profile` Object

`category_profile` packages category-specific typed fields into a sub-object. It is optional: `null` is legal for tier-1 opaque events and for categories with no defined profile keys in this version.

Per-category keys defined in v0.1:


| `category`                                                                                  | `category_profile` shape                                                                                                 |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `"llm"`                                                                                     | `{"model_name": "gpt-4.1"}` — LLM model identifier; null if not known.                                                   |
| `"tool"`                                                                                    | `{"tool_call_id": "call_abc"}` — LLM-provider correlation ID; null if the tool was not invoked via an LLM tool-use flow. |
| `"custom"`                                                                                  | `{"subtype": "nvidia.speculative_decode"}` — REQUIRED per §4.2.                                                          |
| `"unknown"`                                                                                 | `null` — tier-1 pass-through carries no profile information.                                                             |
| others (`agent`, `function`, `retriever`, `embedder`, `reranker`, `guardrail`, `evaluator`) | Reserved. No keys defined in v0.1; producers MAY emit `null` or `{}`. Future MINOR versions MAY define keys.             |


Unknown `category_profile` keys MUST be preserved verbatim by consumers. Adding new keys to an existing profile shape is a backward-compatible MINOR bump per §5.6.

---

## 5. Event Stream Semantics

### 5.1 Timestamp Format and Ordering

**Accepted forms.** Every event's `timestamp` carries one of two interchangeable forms:

- **RFC 3339 string** (e.g., `"2026-01-01T00:00:00.123456Z"`) — human-readable, interoperable with general-purpose date libraries, default choice for debug and log-tailing contexts. MUST end with `Z` or an explicit UTC offset.
- **Integer epoch microseconds UTC** (e.g., `1767225600123456`) — fast to parse (~15× faster than RFC 3339), ~50% smaller on the wire, safe in JSON numbers through year 2255. Chosen for high-throughput streams and columnar-storage pipelines.

Emitters choose per event. A single stream MAY contain events in both forms.

**Why microseconds and not nanoseconds.** JSON numbers are IEEE 754 doubles with 53 bits of integer precision (~9 × 10¹⁵). Nanoseconds since epoch for 2026 is ~1.76 × 10¹⁸ — exceeds safe integer range. Microseconds fits safely and remains precise enough for agent-scope event correlation.

**Ordering.** Events are emitted in wall-clock order. Delivery from subscriber callbacks MAY arrive out-of-order for concurrent operations. Consumers MUST sort by `timestamp` before processing. When sorting a mixed-format stream, consumers MUST normalize both forms to a common representation (typically integer microseconds) before comparison — lexicographic string vs integer comparison is undefined.

**ATIF compatibility.** ATIF requires timestamps as ISO 8601 strings. RFC 3339 is a strict subset of ISO 8601, so the ATOF → ATIF converter forwards the RFC 3339 string form unchanged as a zero-cost pass-through; only the integer microsecond form is serialized to an RFC 3339 string before emitting ATIF.

### 5.2 Scope Nesting and `parent_uuid`

The runtime maintains a scope stack per async task. The `parent_uuid` of any event is the UUID of the scope that was on top of the stack when the handle was created. Following `parent_uuid` links upward reconstructs the full call graph.

The root scope has `parent_uuid = null`. The root scope's events (both `scope_category: "start"` and `scope_category: "end"`) are the only `scope` events in a well-formed stream that may carry a null `parent_uuid` (once the root scope is established). `mark` events MAY carry `parent_uuid = null` when emitted outside any scope.

### 5.3 Start/End Pairing

Every `scope` event with `scope_category: "start"` is paired with exactly one `scope` event with `scope_category: "end"` sharing the same `uuid`. The end event is always emitted strictly after the start event (strict: `ts_micros(end) > ts_micros(start)`).

`mark` events have no paired event — they are single-shot.

If the runtime dies before emitting a paired end event, no event appears in the stream. The pairing guarantee is contingent on orderly shutdown. Consumers that detect an unpaired start event after the stream ends MAY synthesize an end event for downstream processing; such synthetic events are out of scope for ATOF Core.

### 5.4 UUID Uniqueness

Each scope span receives a unique UUID at creation time. The `uuid` is stable across the start and end events for the same scope. In the Rust reference implementation, UUIDs are v7 (time-ordered).

### 5.5 ID Relationships

Two distinct identifier namespaces appear in an ATOF stream:

- **`uuid` / `parent_uuid`** — runtime identifiers attached to every event. Form the scope graph.
- **`category_profile.tool_call_id`** (on `scope` or `mark` events when `category == "tool"`) — an LLM-provider identifier that bridges an LLM's tool-call response with the resulting tool execution. Null when the tool was not invoked via an LLM tool-use flow.

### 5.6 ATOF Version and Negotiation

Every event carries a required `atof_version` field, formatted `"MAJOR.MINOR"` — e.g., `"0.1"`. This section defines when producers bump the version and how consumers dispatch on it.

**Reading rules.** Consumers SHOULD accept any `0.Y` event as ATOF-v0-family. Major-version bumps (`1.0`, `2.0`) MAY introduce breaking changes; consumers that want forward compat MUST dispatch on the major version and fail fast on unknown majors.

**Mixed-version streams.** A single stream MAY contain events at different minor versions (`0.1` and `0.2`). Consumers MUST NOT reject a stream because it contains newer minor versions than expected; unknown fields are preserved per §2.

**When to bump.**

- Bump **MINOR** when adding new optional fields, new `category_profile` keys, new flag vocabulary, new `category` values, or new `attributes` flags. Backward-compatible.
- Bump **MAJOR** when renaming or removing required fields, changing `kind` or `scope_category` discriminator values, or altering pairing semantics. Breaking.

---

## 6. What ATOF Is Not

- **Not ATIF.** ATIF is a higher-level trajectory format with computed ancestry, merged observations, sequenced step_ids, and turn-based structure. ATOF events are the raw observations ATIF is built from. See `examples/atof_to_atif/README.md` for the conversion reference.
- **Not a metrics format.** Token counts, latency budgets, cost attribution — those live in `data` payloads or in downstream aggregation. ATOF does not normalize or roll up metrics.
- **Not a trace format.** ATOF is compatible with distributed tracing (subscribers can export to OpenTelemetry via `metadata.trace_id`/`metadata.span_id`) but is not itself an OTLP-equivalent wire format.
- **Not a replay executor.** An ATOF stream lets you reconstruct what happened. It does not provide the mechanism to re-run it — that's a separate layer built on top.

---

## 7. Reference Implementations

- **Python (consumer + test-producer):** `src/nat/atof/` in `nvidia_nat_atif`. Pydantic models per event kind with `model_config = ConfigDict(extra="allow")` for lossless pass-through.
- **Producer runtimes:** Agent runtimes emitting ATOF MAY use more granular internal types (e.g., separate `LlmStartEvent`/`ToolStartEvent` structs in typed languages) for type-safe construction, but MUST serialize to ATOF's two-kind wire format on emission.
- **Language bindings:** Where a producer runtime exposes bindings to additional languages, those bindings SHOULD re-export the runtime's event types via language-idiomatic wrappers while preserving the wire format on serialization.

See `examples/atof_to_atif/README.md` for the normative ATOF → ATIF conversion reference.

---

## 8. Roadmap / Under Consideration

The following capabilities have been deliberately deferred from v0.1. They may be added in a future version if concrete use cases demonstrate value.

- **Terminal status on scope end.** A `status` field on `scope` events with `scope_category: "end"` — valued in `"ok"` / `"error"` / `"cancelled"` — to carry the scope's terminal outcome on the wire. Consumers currently infer outcome (when needed) from `data` contents defined by the producer.
- **Structured error payload.** An `error` field pairing with `status == "error"`, carrying `{message, type, traceback}` for structured error reporting.
- **Cascading cancellation semantics.** Normative guidance for how parent/child cancellation flows through the scope stack — contingent on `status` being adopted.

Producers and consumers experimenting with these fields ahead of standardization SHOULD namespace them (e.g., under `data` with a vendor-prefixed `data_schema` name) so that a future promotion into ATOF core remains backward-compatible.

---

*Last updated: 2026-04-21 alongside ATOF v0.1.*