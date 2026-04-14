# Agentic Trajectory Observability Format (ATOF) Specification — Core

**Status:** Active
**Version:** 0.2
**Date:** 2026-04-14
**NeMo Agent Toolkit Reference Implementation:** `src/nat/atof/`

**Companion documents:**

- `[atof-codec-profiles.md](./atof-codec-profiles.md)` — provider-specific structured payload schemas (OpenAI Chat, Anthropic Messages, …) that may appear in `input`/`output` when a codec is registered.
- `[atof-to-atif-converter.md](./atof-to-atif-converter.md)` — normative mapping from an ATOF event stream to an ATIF trajectory.

---

## 1. Overview

ATOF (Agentic Trajectory Observability Format) is the wire format for agent runtime subscriber callbacks. Events represent the lifecycle of scopes — composable units of agent work — within the runtime. Subscribers receive events in real time as the runtime executes agent workflows.

ATOF is a **profile-contract protocol**: it defines the wire format and validation rules for the *contract* carried on every event (the `$schema` / `$version` / `$mode` meta-fields on `profile`), not the field shape for any particular category of scope work. Scope-specific payload shape is delegated to JSON Schemas published by vendors (and to the two reference profiles `default/llm.v1` and `default/tool.v1` defined in §6). New scope types extend the ecosystem by publishing a new schema ID, not by modifying this spec.

Transport is JSON-Lines: each event is one JSON object per line. The `kind` field is the outer discriminator. Valid `kind` values are:

- `"ScopeStart"` — a scope was opened
- `"ScopeEnd"` — a scope was closed
- `"Mark"` — a named checkpoint was emitted
- `"StreamHeaderEvent"` — stream-wide schema registry and default profile-mode declaration (§3.4, §5)

What *kind of work* a scope represents — an LLM call, a tool invocation, an agent turn, a retriever lookup — is carried by the `scope_type` string field on `ScopeStart`/`ScopeEnd` and by the companion `profile` contract (§4). `scope_type` is an open vocabulary (§3.1); `profile.$schema` declares the structured shape of the profile payload.

**Wire envelope shape:**

```json
{"kind": "ScopeStart", "schema_version": "0.1", "uuid": "...", "parent_uuid": "...", "timestamp": "...", "name": "...", "scope_type": "llm", "flags": [], "profile": {...}, "input": {...}, "data": null, "metadata": null}
```

ATOF events are the raw, un-merged observations from the runtime. Downstream layers (ATIF conversion, observability export) consume them separately.

### 1.1 The Four Structured Fields at a Glance

Beyond the core envelope (`kind`, `uuid`, `parent_uuid`, `timestamp`, `name`), ATOF events carry four structured "bag" fields. Two are spec-governed and describe the scope itself; two are opaque to ATOF and describe the surrounding context:


|                       | Spec-governed shape | Opaque to ATOF     |
| --------------------- | ------------------- | ------------------ |
| **About the scope**   | `flags`, `profile`  | —                  |
| **About the context** | —                   | `data`, `metadata` |


- `flags` — cross-cutting behavioral flag set on the scope. Vocabulary is documented in §2.1. Consumers MUST preserve unknown flag strings. Type: `string[]`.
- `profile` — structured profile payload for the scope, following the profile-contract protocol (§4). Carries `$schema` (required), `$version` (required), optional `$mode`, and vendor-defined fields whose shape is specified by the declared `$schema`. Type: object or null.
- `data` — application-defined payload. Opaque to ATOF. Consumers MUST NOT dispatch on `data` contents.
- `metadata` — tracing and correlation envelope. Conventionally carries `trace_id`, `span_id`, and similar plumbing.

**Choosing between them.** When attaching information to an event:

1. Is it a boolean-ish *characteristic* of the scope? → `flags`
2. Is it a typed field defined by the scope's declared `profile.$schema`? → `profile`
3. Is it tracing or correlation plumbing? → `metadata`
4. Is it an application payload that the spec does not need to reason about? → `data`

Emitters SHOULD NOT duplicate information already carried by `profile`, `input`, or `output` into `data`. `flags` and `profile` appear only on `ScopeStart` / `ScopeEnd`; `data` and `metadata` appear on all four event kinds (including `StreamHeaderEvent`).

---

## 2. Common Event Fields

All four event kinds share these seven envelope fields. The `flags` field is documented in §2.1 because it is cross-cutting across scope events (not a profile-level concern).


| Field            | Type                  | Required | Description                                                                                                                                                                                                                                      |
| ---------------- | --------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `schema_version` | string                | Yes      | ATOF protocol version this event conforms to, formatted as `"MAJOR.MINOR"` (e.g., `"0.2"`). This is the ATOF *wire-format* version; it is NOT the version of any particular profile schema (see §4 — profile version lives in `profile.$version`). Consumers dispatch on this value per the negotiation rules in §7.7. Mixed-version streams are permitted. |
| `parent_uuid`    | string (UUID) or null | No       | UUID of the scope that was on top of the stack when this handle was created. Null only on the root scope. Following `parent_uuid` links upward reconstructs the call graph.                                                                      |
| `uuid`           | string (UUID)         | Yes      | Unique identifier for this handle. The matching Start and End events for the same handle carry the same `uuid`.                                                                                                                                  |
| `timestamp`      | string (RFC 3339) or integer (epoch µs) | Yes | Wall-clock time when this event was emitted. Accepts two interchangeable forms — RFC 3339 string (e.g., `"2026-01-01T00:00:00Z"`) or integer epoch microseconds UTC (e.g., `1767225600000000`). See §7.1 for format choice guidance. Start and End events for the same handle have different timestamps. |
| `name`           | string                | Yes      | Human-readable label for this handle — e.g., `"my_agent"`, `"calculator__add"`, `"nvidia/nemotron-3-super-v3"`.                                                                                                                                  |
| `data`           | object or null        | No       | Application-specific JSON payload attached by the caller.                                                                                                                                                                                        |
| `metadata`       | object or null        | No       | Tracing and correlation metadata — e.g., `{"trace_id": "...", "span_id": "..."}`.                                                                                                                                                                |


### 2.1 `flags` — behavioral flag set (ScopeStart / ScopeEnd only)

`flags` is a cross-cutting field on `ScopeStartEvent` and `ScopeEndEvent` (it is NOT present on `MarkEvent` or `StreamHeaderEvent`). It carries a set of boolean indicators describing runtime properties of the scope — orthogonal to the scope's `$schema`-governed profile payload.

| Field   | Type             | Required on ScopeStart / ScopeEnd | Description                                                                                                                                                                                                                                      |
| ------- | ---------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `flags` | array of strings | Yes                               | Behavioral flag names in canonical form (lowercase, sorted, deduplicated). Empty array when no flags are set. Producers MUST emit `flags` in lexicographic order with no duplicates. Consumers SHOULD treat the array as an unordered set and MUST preserve unknown flag names when re-emitting. |

The core flag vocabulary is:

| Flag            | Meaning                                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------------------------------- |
| `"parallel"`    | Scope may execute concurrently with sibling scopes under the same parent.                                            |
| `"relocatable"` | Scope may be moved across async task boundaries (e.g., between threads or event loops) without losing context.       |
| `"stateless"`   | Scope does not maintain state between invocations — the same inputs produce the same behavior regardless of history. |
| `"local"`       | Scope executes in the same process as the runtime, as opposed to dispatching to a remote service.                    |
| `"streaming"`   | Scope produces its output incrementally as a sequence of chunks, rather than as a single payload at exit.            |

No flag is mandatory. Emitters SHOULD emit a flag only when the corresponding property is affirmatively true; absence is the default.

**Flag extensibility.** Implementations MAY emit additional flag names for vendor extensions; to avoid collisions, non-canonical flags SHOULD be namespaced with a dotted prefix — for example, `"nvidia.speculative"`. Consumers MUST preserve unknown flag strings when re-emitting events and MUST NOT treat unknown flags as an error.

**Streaming + terminal status.** For scopes carrying the `"streaming"` flag: if the scope terminates with `status == "error"` or `status == "cancelled"` (§3.2), the `output` on `ScopeEnd` MAY contain the partial chunks accumulated before the terminal event. Consumers that replay streaming output MUST check `status` before treating `output` as a complete payload.

---

## 3. Event Types

ATOF v0.2 defines four event kinds: `ScopeStartEvent`, `ScopeEndEvent`, `MarkEvent`, and `StreamHeaderEvent`. The `kind` field is the outer discriminator. `ScopeStart` and `ScopeEnd` describe the lifecycle of typed scopes; `Mark` records named checkpoints; `StreamHeaderEvent` declares stream-wide profile schemas and the default profile-mode (§5).

### 3.1 ScopeStartEvent

Emitted when a new scope is pushed onto the scope stack.


| Field            | Type                  | Required | Description                                                                                                                                                                                                                                                                                                                               |
| ---------------- | --------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`           | string                | Yes      | Discriminator literal `"ScopeStart"`.                                                                                                                                                                                                                                                                                                     |
| `schema_version` | string                | Yes      | See §2.                                                                                                                                                                                                                                                                                                                                   |
| `parent_uuid`    | string (UUID) or null | No       | See §2.                                                                                                                                                                                                                                                                                                                                   |
| `uuid`           | string (UUID)         | Yes      | See §2.                                                                                                                                                                                                                                                                                                                                   |
| `timestamp`      | string (RFC 3339) or integer (epoch µs) | Yes | See §2.                                                                                                                                                                                                                                                                                                                                   |
| `name`           | string                | Yes      | See §2.                                                                                                                                                                                                                                                                                                                                   |
| `data`           | object or null        | No       | See §2.                                                                                                                                                                                                                                                                                                                                   |
| `metadata`       | object or null        | No       | See §2.                                                                                                                                                                                                                                                                                                                                   |
| `scope_type`     | string                | Yes      | Non-empty string identifying the kind of work this scope represents. Open vocabulary — see the conventions note below. Producers MUST emit a non-empty string; consumers MUST NOT reject unknown values.                                                                                                                                  |
| `flags`          | array of strings      | Yes      | Behavioral flag names in canonical form (lowercase, sorted, deduplicated). See §2.1 for the shared vocabulary. Empty array when no flags are set.                                                                                                                                                                                         |
| `profile`        | Profile (§4) or null  | No       | Profile-contract payload carrying `$schema` (required), `$version` (required), optional `$mode`, and vendor-defined fields whose shape is governed by the declared `$schema`. Null when the scope has no typed profile (e.g., a structural `"agent"` or `"function"` scope). Validation rules live in §4; the two spec-defined reference profiles live in §6. MAY be null. |
| `input`          | any or null           | No       | Sanitized input payload handed to the scope at entry (post request-sanitize guardrails). Opaque by default. When a codec is registered for this scope, `input` holds the structured form defined by the codec profile (see `[atof-codec-profiles.md](./atof-codec-profiles.md)`). Omitted or null when the scope has no meaningful input. |

**`scope_type` conventions (informational, not normative).** Common values encountered in practice include `"agent"`, `"function"`, `"tool"`, `"llm"`, `"retriever"`, `"embedder"`, `"reranker"`, `"guardrail"`, `"evaluator"`, `"custom"`, and `"unknown"`. Producers SHOULD use these conventional values when they accurately describe the scope; when none fit, producers MAY coin new values (optionally vendor-namespaced, e.g., `"nvidia.tool-bundle"`). Consumers MUST NOT treat any value as normatively restricted — in particular, consumers MUST NOT drop events with unrecognized `scope_type` values. The structural shape of the scope's `profile` is governed entirely by `profile.$schema` (§4), not by `scope_type`.


### 3.2 ScopeEndEvent

Emitted when a scope is popped from the scope stack. Mirrors `ScopeStartEvent` except that `input` is replaced by `output` and the terminal `status` / `error` fields are added.


| Field            | Type                  | Required | Description                                                                                                                                                                                                                                                                                                                                          |
| ---------------- | --------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`           | string                | Yes      | Discriminator literal `"ScopeEnd"`.                                                                                                                                                                                                                                                                                                                  |
| `schema_version` | string                | Yes      | See §2.                                                                                                                                                                                                                                                                                                                                              |
| `parent_uuid`    | string (UUID) or null | No       | See §2.                                                                                                                                                                                                                                                                                                                                              |
| `uuid`           | string (UUID)         | Yes      | Same value as the matching `ScopeStartEvent`.                                                                                                                                                                                                                                                                                                        |
| `timestamp`   | string (RFC 3339) or integer (epoch µs) | Yes | See §2.                                                                                                                                                                                                                                                                                                                                              |
| `name`        | string                | Yes      | Same value as the matching `ScopeStartEvent`.                                                                                                                                                                                                                                                                                                        |
| `data`        | object or null        | No       | See §2.                                                                                                                                                                                                                                                                                                                                              |
| `metadata`    | object or null        | No       | See §2.                                                                                                                                                                                                                                                                                                                                              |
| `scope_type`  | string                | Yes      | Same value as the matching `ScopeStartEvent` (see §3.1).                                                                                                                                                                                                                                                                                             |
| `flags`       | array of strings      | Yes      | Same value as the matching `ScopeStartEvent`. See §2.1.                                                                                                                                                                                                                                                                                              |
| `profile`     | Profile (§4) or null  | No       | Profile-contract payload for scope exit. MAY differ from the `ScopeStart` `profile` for the same handle (e.g., the publisher may populate additional vendor fields at end time). Cross-event invariance rules apply — `ScopeEnd.profile.$schema` MUST equal `ScopeStart.profile.$schema` and `ScopeEnd.profile.$version` MUST equal `ScopeStart.profile.$version` (§4.7). |
| `output`      | any or null           | No       | Sanitized output payload produced by the scope at exit (post response-sanitize guardrails). Opaque by default. When a codec is registered for this scope, `output` holds the structured form defined by the codec profile (see `[atof-codec-profiles.md](./atof-codec-profiles.md)`). Omitted or null when the scope has no meaningful return value. |
| `status`      | string (enum)         | Yes      | Terminal outcome of the scope. One of: `"ok"` (scope returned normally), `"error"` (scope raised an exception), `"cancelled"` (scope was terminated before completion, e.g., timeout, parent cancel, explicit cancel). See status semantics below.                                                                                                   |
| `error`       | object or null        | No       | Structured exception/cancellation context. MUST be present when `status == "error"`; MAY be present when `status == "cancelled"`; MUST be null or absent when `status == "ok"`. See error object shape below.                                                                                                                                         |


**Status semantics.** The `status` field is required on every `ScopeEnd` and distinguishes three outcomes that would otherwise all present as `output=null`:

| `status`      | `output` behavior                                                                   | `error` behavior                                                                 |
| ------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `"ok"`        | MAY be any value; null means "scope returned but produced nothing".                 | MUST be null or absent.                                                          |
| `"error"`     | SHOULD be null, but MAY carry partial output (e.g., streamed tokens before failure). | MUST be present. `type` and `message` are required.                             |
| `"cancelled"` | SHOULD be null, but MAY carry partial output accumulated before cancellation.        | MAY be present (e.g., a `CancelledError` with location context); MAY be null.    |

**Error object shape.** When `error` is present, it is a JSON object with the following fields:

| Field     | Type            | Required | Description                                                                                                                                 |
| --------- | --------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`    | string          | Yes      | Native exception or error class name — e.g., `"TimeoutError"`, `"anthropic.RateLimitError"`, `"tokio::time::error::Elapsed"`. Free-form; no canonical vocabulary. |
| `message` | string          | Yes      | Human-readable error message.                                                                                                               |
| `stack`   | string or null  | No       | Stack trace as a single multi-line string. Implementations MAY redact for security. Null when unavailable or withheld.                      |

Vendor- or domain-specific error context (HTTP status codes, provider error codes, retry hints, cause chains) belongs in `data`, not in `error`. Keeping `error` minimal lets any emitter satisfy the spec without imposing a taxonomy that does not generalize.


### 3.3 MarkEvent

Emitted when the application records a named checkpoint in the event stream. Mark is the simplest event type — it has no `scope_type`, `flags`, `profile`, `input`, or `output`.


| Field            | Type                  | Required | Description                                      |
| ---------------- | --------------------- | -------- | ------------------------------------------------ |
| `kind`           | string                | Yes      | Discriminator literal `"Mark"`.                  |
| `schema_version` | string                | Yes      | See §2.                                          |
| `parent_uuid`    | string (UUID) or null | No       | See §2.                                          |
| `uuid`           | string (UUID)         | Yes      | See §2.                                          |
| `timestamp`      | string (RFC 3339) or integer (epoch µs) | Yes | See §2.                                          |
| `name`           | string                | Yes      | Name of the marker checkpoint.                   |
| `data`           | object or null        | No       | Application-specific payload for this milestone. |
| `metadata`       | object or null        | No       | See §2.                                          |


### 3.4 StreamHeaderEvent

Emitted to declare stream-wide profile schemas and the default profile-mode for all subsequent events. `StreamHeaderEvent` is the fourth event kind in ATOF v0.2; it does NOT describe a scope's lifecycle. It carries no `scope_type`, no `flags`, no `profile`, no `status`, and no `error` — the full declarative reference for its fields, placement rules, and mode semantics lives in §5.

| Field                  | Type                  | Required | Description                                                                                                                                                                                                                                       |
| ---------------------- | --------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`                 | string                | Yes      | Discriminator literal `"StreamHeaderEvent"`. Note: unlike the other three event kinds whose discriminators drop the "Event" suffix, this one retains it so that the wire discriminator matches the Python class name.                            |
| `schema_version`       | string                | Yes      | See §2. ATOF protocol version (e.g., `"0.2"`) — NOT the version of any profile schema declared in `schemas`.                                                                                                                                      |
| `parent_uuid`          | string (UUID) or null | No       | See §2. Typically null for header events.                                                                                                                                                                                                         |
| `uuid`                 | string (UUID)         | Yes      | See §2. Unique identifier for the header event itself.                                                                                                                                                                                            |
| `timestamp`            | string (RFC 3339) or integer (epoch µs) | Yes | See §2.                                                                                                                                                                                                                                           |
| `name`                 | string                | Yes      | Human-readable label for the header — e.g., `"exmp01_header"`, `"run_initial_header"`.                                                                                                                                                            |
| `data`                 | object or null        | No       | See §2.                                                                                                                                                                                                                                           |
| `metadata`             | object or null        | No       | See §2.                                                                                                                                                                                                                                           |
| `profile_mode_default` | string (enum)         | Yes      | Stream-wide default mode for profile events. One of: `"header"` (profiles reference schemas by string `$schema` ID, resolved via this event's `schemas` registry), `"inline"` (profiles carry their full JSON Schema as a dict in `$schema`), `"opaque"` (consumers preserve profiles but do NOT validate). A profile's own `$mode` field, when present, overrides this default for that one event (§4.4). |
| `schemas`              | object                | Yes      | Schema registry keyed by schema ID (e.g., `"default/llm.v1"`). Each value is a complete JSON Schema document (Draft 2020-12). When a value contains an `$id` field, it MUST equal the dict key. MAY be empty (`{}`) — common when `profile_mode_default` is `"inline"` or `"opaque"`. See §5.4. |

**Note:** `StreamHeaderEvent` does NOT carry `scope_type`, `flags`, `profile`, `status`, or `error` — those fields belong to scope-lifecycle events only.


---

## 4. Profile Contract Protocol

The `profile` field on `ScopeStartEvent` and `ScopeEndEvent` carries a **profile contract** — a structured payload whose shape is governed by a declared JSON Schema (the *contract*). ATOF v0.2 standardizes the wire format of that contract (the `$schema` / `$version` / `$mode` meta-fields) and the validation rules that producers and consumers MUST follow; it does NOT fix the shape of the vendor fields inside the profile. That shape is published as a JSON Schema and referenced by `$schema`.

This section defines the contract in seven subsections. Two spec-defined reference implementations (`default/llm.v1`, `default/tool.v1`) live in §6. Vendors publishing richer profiles (`openai/llm.v1`, `nvidia/guardrail-content-safety.v2`, …) follow the same wire format.

### 4.1 Profile object structure

A profile is a JSON object with three reserved meta-fields (all prefixed with `$`) plus zero or more vendor-defined fields:

| Field      | Type                              | Required | Description                                                                                                                                                                                                                                           |
| ---------- | --------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `$schema`  | string or object (JSON Schema)    | Yes      | Schema declaration. EITHER a string ID (e.g., `"default/llm.v1"`) resolved against a `StreamHeaderEvent` registry, OR an inline JSON Schema object (which MUST include its own `$id`). See §4.3 for the inline form.                                  |
| `$version` | string                            | Yes      | Schema publisher's version identifier. This is NOT the ATOF protocol version (that lives in `schema_version` on every event — §2). Schema publishers SHOULD follow their own versioning convention; semantic versioning is recommended.              |
| `$mode`    | string (enum) or absent           | No       | Per-event override of `StreamHeaderEvent.profile_mode_default`. One of `"header"`, `"inline"`, `"opaque"`. Absent means inherit the stream-wide default (§5).                                                                                         |
| *(vendor-defined fields)* | any                | —        | Additional fields defined by the declared `$schema`. Shape, required/optional, and validation are governed entirely by the JSON Schema, not by this spec.                                                                                             |

**Reserved field prefix.** All `$`-prefixed fields are reserved by the ATOF spec. Vendor profiles MUST NOT introduce additional `$`-prefixed fields. Non-`$` field names are available to vendor schemas without restriction, subject to JSON object naming.

### 4.2 Schema ID format

Schema IDs follow the convention `vendor/<id>` where `<id>` ends in `.v<N>` for integer major version `N`. Examples:

- `default/llm.v1` — reference implementation for LLM scopes (§6.1)
- `default/tool.v1` — reference implementation for tool scopes (§6.2)
- `openai/llm.v1` — hypothetical vendor-published richer LLM profile
- `nvidia/guardrail-content-safety.v2` — hypothetical vendor-published guardrail profile

The `default/*` namespace is reserved for spec-defined reference implementations; vendors MUST NOT publish schemas in the `default/*` namespace. Each schema ID identifies one major-version line. Breaking changes require a new major-version suffix (`.v1` → `.v2`); backward-compatible additions MAY be folded into the same major version with an updated `$version` string on the profile.

### 4.3 Inline schema mode

When `$schema` is a JSON Schema object (not a string ID), the object MUST include an `$id` field identifying the schema it describes. Producers emitting inline schemas MUST embed the full JSON Schema body — partial references (e.g., just the `$id`) are not valid inline schemas; use the string-ID form with a `StreamHeaderEvent` registry instead.

Inline schemas let a producer emit a self-contained event that a consumer can validate without external lookup. This is the `"inline"` mode (§5.3); it is useful for short streams, ad-hoc debugging output, and streams where no `StreamHeaderEvent` is present.

### 4.4 Stream-level mode default vs per-event override

`StreamHeaderEvent.profile_mode_default` (§5) sets the stream-wide default mode for interpreting profile payloads. A profile's own `$mode` field, when present, overrides the default for that single event.

**Resolution order:**

1. If `profile.$mode` is present, use it.
2. Else, if a prior `StreamHeaderEvent` has declared `profile_mode_default`, use that.
3. Else, default to `"opaque"` — consumers preserve unknown profile fields and do NOT validate.

The three modes are defined in §5.3.

### 4.5 Producer validation contract (MUST)

Producers MUST validate every profile payload against its declared `$schema` before emitting the containing event. Validation uses JSON Schema Draft 2020-12.

When the mode is `"header"` (string `$schema` ID), the producer MUST have published the corresponding schema in a `StreamHeaderEvent` registry before emitting the profile event; the producer validates against the registry entry. When the mode is `"inline"`, the producer validates against the embedded schema. When the mode is `"opaque"`, the producer MAY skip validation.

Validation failures MUST raise an error — emitting a profile payload that violates its declared schema is a producer bug. Implementations MAY choose the exception type (the Python reference implementation raises `pydantic.ValidationError` at profile construction). Producers MUST NOT silently truncate, coerce, or drop fields to make a payload validate.

### 4.6 Consumer validation contract (MAY)

Consumers MAY validate profiles against the declared `$schema`; validation is OPTIONAL. Validation rules:

- Consumers that do NOT validate MUST preserve unknown fields verbatim when re-emitting events. Pass-through tools (filters, pretty-printers, samplers) are consumers for this purpose.
- Consumers MUST NOT drop events whose `$schema` ID is unknown to the consumer's local schema registry. Unknown schemas imply an opaque passthrough, not a rejection.
- Consumers that DO validate and encounter a validation failure MUST log a warning (logger, not stream) and MUST NOT drop the event. The event continues to propagate downstream. This asymmetry — producer MUST raise, consumer MUST NOT drop — reflects the different tradeoffs at each side of the pipe: producers are the last point where a payload can be corrected, while consumers are last-mile and can do less damage by letting the event through than by discarding it.

### 4.7 Cross-event invariance

For every matched `(ScopeStartEvent, ScopeEndEvent)` pair sharing the same `uuid`:

- `ScopeStart.profile.$schema` MUST equal `ScopeEnd.profile.$schema`. Producers MUST NOT switch schemas mid-scope; if a paired `ScopeEnd` needs a different schema body, producers MUST emit a new `StreamHeaderEvent` that registers the supplemental schema and use a new `$schema` ID for both events (not half-upgrade an existing pair).
- `ScopeStart.profile.$version` MUST equal `ScopeEnd.profile.$version`. A paired scope is versioned once.

The purpose of these invariants is to let consumers treat a `(Start, End)` pair as a single logical unit with a single contract — essential for metrics aggregation, ATIF conversion, and trajectory replay.

---

## 5. Stream Header Event

`StreamHeaderEvent` is the fourth ATOF event kind (§3.4). It carries no scope data; its purpose is purely declarative — to publish a stream-wide registry of profile schemas and to declare the default mode for resolving them.

### 5.1 Wire shape

`StreamHeaderEvent` inherits the `_EventBase` envelope fields (§2: `schema_version`, `uuid`, `parent_uuid`, `timestamp`, `name`, `data`, `metadata`). It adds two event-specific fields documented in §3.4:

- `kind: "StreamHeaderEvent"` — the discriminator literal.
- `profile_mode_default: Literal["header", "inline", "opaque"]` — default mode for subsequent profile events.
- `schemas: dict[str, JSONSchemaObject]` — schema registry, keyed by schema ID (e.g., `"default/llm.v1"`).

`StreamHeaderEvent` does NOT carry `scope_type`, `flags`, `profile`, `input`, `output`, `status`, or `error`. Those fields describe scope lifecycle; the header is not part of any scope's lifecycle.

### 5.2 Placement rules

`StreamHeaderEvent` MAY appear anywhere in the stream before the first non-opaque profile event — it does NOT need to be the first event in the stream. This permits:

- Logs that concatenate multiple sessions with distinct schema registries
- Streams that emit structural scopes (with `profile == null`) before a schema is declared
- Runtime flows where the schema registry is built incrementally

**Multiple headers are allowed.** A stream MAY contain multiple `StreamHeaderEvent`s. For the same `$schema` ID appearing in more than one header, later schema definitions SUPERSEDE earlier ones (last-wins merge). Consumers building an effective registry walk the stream in timestamp order and merge headers as they encounter them.

When NO `StreamHeaderEvent` appears in the stream, the effective `profile_mode_default` is `"opaque"` and the effective schema registry is empty.

### 5.3 Mode semantics

Three modes govern how consumers interpret a profile:

- **`"header"`** — profiles reference schemas by string `$schema` ID. Consumers resolve the ID via the effective `StreamHeaderEvent` registry. Consumers that choose to validate look up the schema in the registry and validate against the body there.
- **`"inline"`** — profiles carry their schema directly: `$schema` is a JSON Schema object (with an `$id`). Consumers that choose to validate validate against the embedded body. No registry lookup needed.
- **`"opaque"`** — consumers preserve profile fields verbatim but do NOT validate. Used for streams where schema registration is not available or where downstream consumers are schema-unaware.

Per-event `$mode` overrides (§4.4) apply uniformly to all three modes — a profile MAY declare `$mode: "inline"` and carry an inline schema even when the stream default is `"header"`, or vice versa.

### 5.4 Schema registration rules

Each entry in `schemas[schema_id]` is a complete JSON Schema document (Draft 2020-12). Constraints:

- If `schemas[schema_id]["$id"]` is present, it MUST equal the dict key `schema_id`. Producers SHOULD include `$id` for self-description; consumers that encounter a mismatched `$id` SHOULD log a warning and treat the dict key as authoritative.
- ATOF v0.2 uses JSON Schema Draft 2020-12 as the canonical dialect. Schema bodies MAY reference other dialects via `$schema`; consumers MAY choose whether to honor non-canonical dialects.
- Entries MAY use any JSON Schema keywords permitted by Draft 2020-12 (`type`, `properties`, `required`, `additionalProperties`, `patternProperties`, `$ref`, etc.).

---

## 6. Reference Profile Implementations

ATOF v0.2 defines exactly two reference profile implementations, one for LLM scopes and one for tool scopes. These schemas preserve the single profile field each carried in v0.1 (`model_name` and `tool_call_id`) as the minimum useful payload. Vendors publishing richer LLM or tool profiles (e.g., `openai/llm.v1` with provider/usage/finish-reason fields) SHOULD use these as structural starting points.

Both reference profiles set `additionalProperties: true` — vendors MAY extend without subclassing or without republishing a new schema ID, and non-validating consumers preserve unknown fields verbatim.

### 6.1 default/llm.v1

**Purpose.** The canonical reference profile for scopes carrying `scope_type: "llm"`. Preserves the v0.1 `LLMProfile.model_name` field as the sole optional vendor field. Vendors publishing richer LLM profiles (e.g., `openai/llm.v1`, `anthropic/messages.v1`) SHOULD use this as a structural reference.

**Python reference:** `nat.atof.profiles.DefaultLlmV1` — the Pydantic model in the NeMo Agent Toolkit reference implementation mirrors this schema body as a `JSON_SCHEMA: ClassVar[dict]` class attribute.

**JSON Schema body:**

```json
{
  "$id": "default/llm.v1",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "model_name": {"type": ["string", "null"]}
  },
  "required": [],
  "additionalProperties": true
}
```

### 6.2 default/tool.v1

**Purpose.** The canonical reference profile for scopes carrying `scope_type: "tool"`. Preserves the v0.1 `ToolProfile.tool_call_id` field as the sole optional vendor field. Vendors publishing richer tool profiles SHOULD use this as a structural reference.

**Python reference:** `nat.atof.profiles.DefaultToolV1` — the Pydantic model in the NeMo Agent Toolkit reference implementation mirrors this schema body as a `JSON_SCHEMA: ClassVar[dict]` class attribute.

**JSON Schema body:**

```json
{
  "$id": "default/tool.v1",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "tool_call_id": {"type": ["string", "null"]}
  },
  "required": [],
  "additionalProperties": true
}
```

**Non-default reference profiles are out of scope for this spec.** Vendor profiles (`openai/llm.v1`, `nvidia/guardrail-content-safety.v2`, `anthropic/messages.v1`, …) are published by their respective vendors following the §4.2 schema ID format. ATOF v0.2 reserves the `default/*` namespace for spec-defined reference implementations and makes no normative statement about other namespaces.

---

## 5. Event Stream Semantics

### 5.1 Timestamp Format and Ordering

**Accepted forms.** Every event's `timestamp` (Section 2) carries one of two interchangeable forms:

- **RFC 3339 string** (e.g., `"2026-01-01T00:00:00.123456Z"`) — human-readable, interoperable with general-purpose date-handling libraries, default choice for debug and log-tailing contexts. MUST end with `Z` or an explicit UTC offset.
- **Integer epoch microseconds UTC** (e.g., `1767225600123456`) — fast to parse (~15× faster than RFC 3339 in most runtimes), ~50% smaller on the wire, safe in JSON numbers through year 2255 (fits in IEEE 754 double integer precision). Chosen for high-throughput streams and columnar-storage pipelines.

Emitters choose per event. A single stream MAY contain events in both forms (mixed-format streams are legal for the same reasons mixed-version streams are legal — see §5.7).

**Why microseconds and not nanoseconds.** JSON numbers are IEEE 754 doubles with 53 bits of integer precision (~9 × 10¹⁵). Nanoseconds since epoch for 2026 is ~1.76 × 10¹⁸ — exceeding safe integer range and causing silent precision loss in most parsers. Microseconds fits safely and remains precise enough for agent-scope event correlation. If nanosecond precision is required for a specific use case, emitters SHOULD use the RFC 3339 string form with a nanosecond fractional second.

**Ordering.** Events are emitted in wall-clock order. Delivery order from subscriber callbacks MAY differ for concurrent operations. Consumers MUST sort by `timestamp` before processing. When sorting a mixed-format stream, consumers MUST normalize both forms to a common representation (typically integer microseconds) before comparison — lexicographic comparison of a string against an integer is undefined.

**ATIF compatibility.** ATIF (see `[atof-to-atif-converter.md](./atof-to-atif-converter.md)`) requires timestamps as ISO 8601 strings on its optional `step.timestamp` field. The ATOF → ATIF converter serializes either ATOF form to an ISO 8601 string before emitting ATIF. No emitter-side action is required for ATIF compatibility regardless of which ATOF timestamp form is chosen.

### 5.2 Scope Nesting and parent_uuid

The runtime maintains a scope stack per async task. The `parent_uuid` of any event is the UUID of the scope that was on top of the stack when the handle was created. Following `parent_uuid` links upward reconstructs the full call graph.

The root scope has `parent_uuid = null`. This is the only event in a well-formed stream that may have a null `parent_uuid` (once the root scope is established).

### 5.3 Start/End Pairing

Every `ScopeStart` event is paired with exactly one `ScopeEnd` event sharing the same `uuid`. `ScopeEnd` events always arrive after their matching `ScopeStart` in wall-clock order. All child events (events whose `parent_uuid` equals this scope's `uuid`) will have been emitted before the parent's `ScopeEnd` fires.

`Mark` events have no paired event — they are single-shot.

### 5.4 UUID Uniqueness

Each handle receives a unique UUID at creation time. The `uuid` is stable across the Start and End events for the same handle, enabling correlation. In the Rust reference implementation, UUIDs are v7 (time-ordered).

### 5.5 ID Relationships

Three distinct identifier namespaces appear in an ATOF stream:

- **`uuid` / `parent_uuid`** — agent runtime identifiers attached to every event. Form the scope graph.
- **`profile.tool_call_id`** (on `scope_type: "tool"`) — an LLM-provider identifier that bridges an LLM's tool-call response with the resulting tool execution. Null when the tool was not invoked via an LLM tool-use flow.
- **Codec-decoded response IDs** (e.g., `chatcmpl-`* inside a decoded LLM response body) — provider tracking identifiers. Opaque to ATOF Core; see `[atof-codec-profiles.md](./atof-codec-profiles.md)`.

```text
┌─ ScopeStart (scope_type=agent) ─────────────── ScopeEnd ────────────────┐
│  uuid: "scope-001"                                                      │
│  parent_uuid: null                                                      │
│                                                                         │
│  ┌─ ScopeStart (scope_type=llm) ─────────────── ScopeEnd ─┐             │
│  │  uuid: "llm-001"                                       │             │
│  │  parent_uuid: "scope-001" ─────────────────────────────┼──► graph    │
│  │  profile.model_name: "..."                             │             │
│  │  output.tool_calls[0].id ──────────────────────────────┼──► "call_1" │
│  └────────────────────────────────────────────────────────┘             │
│                                                                         │
│  ┌─ ScopeStart (scope_type=tool) ────────────── ScopeEnd ─┐             │
│  │  uuid: "tool-001"  (≠ "llm-001")                       │ ← handle id │
│  │  parent_uuid: "scope-001" ─────────────────────────────┼──► graph    │
│  │  profile.tool_call_id: "call_1" ────────────────────---┼──► LLM corr │
│  └────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘

uuid                     → "Who am I?"          (handle identity, Start=End)
parent_uuid              → "Who created me?"    (scope stack lineage)
profile.tool_call_id     → "Which LLM request?" (LLM↔tool correlation)
codec response.id        → "Which API call?"    (provider tracking, see codec profiles)
```

### 5.6 Terminal Status and Cancellation Propagation

Every `ScopeEnd` carries a terminal `status` (Section 3.2). The following rules govern how terminal status relates across the scope graph.

**Each scope reports its own terminal status.** The `status` on a `ScopeEnd` describes the outcome of *that* scope, not its children or its parent. A parent whose child errored MAY itself report `status == "ok"` if the parent caught and handled the child's error; conversely, a parent MAY report `status == "error"` even if all its children completed cleanly.

**Cascading cancellation.** When a parent cancels its children (e.g., due to parent timeout or parent error recovery), each child emits its own `ScopeEnd` with `status == "cancelled"` before the parent emits its `ScopeEnd`. Section 5.3 still holds: all child events precede the parent's `ScopeEnd` in wall-clock order. The parent's own `status` reflects the parent's outcome:

- `"cancelled"` if the parent was itself cancelled from above.
- `"error"` if the parent raised (possibly because a child's failure propagated).
- `"ok"` if the parent chose to cancel its children as normal control flow and then completed.

**Dangling scopes.** If the runtime dies before emitting a paired `ScopeEnd`, no event appears in the stream. Section 5.3's pairing guarantee is contingent on orderly shutdown. Consumers that detect an unpaired `ScopeStart` after the stream ends MAY synthesize a `ScopeEnd` with `status == "cancelled"` for downstream processing; such synthetic events are out of scope for ATOF Core.

**Start/End pairing unchanged.** Section 5.3's invariants still hold: every `ScopeStart` has exactly one matching `ScopeEnd` sharing the same `uuid`, and all child events of a scope precede the parent's `ScopeEnd`. `status` is a property of the End event, not a modifier of the pairing rule.

### 5.7 Schema Version and Negotiation

Every ATOF event carries a required `schema_version` field (Section 2) formatted as `"MAJOR.MINOR"` — e.g., `"0.1"`. This section defines when producers bump the version, how consumers dispatch on it, and what guarantees exist across mixed-version streams.

**Version bump policy.**

- **Minor bump** (e.g., `0.1` → `0.2`) for additive, backward-compatible changes. Examples: adding a new optional field, adding a new flag name to an open vocabulary, adding a new enum value to an extensible enum, defining fields for a previously-empty scope profile, adding a new scope type.
- **Major bump** (e.g., `0.1` → `1.0`) for breaking changes. Examples: removing a field, renaming a field, changing a field's type, changing a required field's nullability, redefining the semantics of an existing field or enum value, making a previously optional field required.

Pre-release versions (`0.x`) are not subject to these rules — the spec may introduce breaking changes within the `0.x` series without a major bump. Major bump discipline begins with the first `1.0` release.

**Consumer dispatch on version mismatch.** Given an event's `schema_version` `SEEN` and the consumer's expected version `EXPECTED`, consumers SHOULD behave as follows:

| Comparison                                     | Consumer behavior                                                                                                                                     |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SEEN == EXPECTED`                             | Process normally.                                                                                                                                     |
| `SEEN.major == EXPECTED.major`, `SEEN.minor > EXPECTED.minor` | Process best-effort. Event shape is a forward-compatible superset. Unknown fields, flag names, and enum values MUST NOT cause errors and MUST be preserved when re-emitting. Consumers MAY log a soft warning. |
| `SEEN.major == EXPECTED.major`, `SEEN.minor < EXPECTED.minor` | Process normally. Event shape is a backward-compatible subset of what the consumer expects.                                                           |
| `SEEN.major != EXPECTED.major`                 | Reject or log a loud warning. Major-version mismatch indicates breaking changes the consumer cannot safely handle via forward-compat.                 |

Implementations MAY choose stricter or more lenient behavior (e.g., strict equality, loose best-effort for all mismatches). Implementations that deviate from the recommended dispatch SHOULD document their policy.

**Mixed-version streams.** A single stream MAY contain events with differing `schema_version` values. This supports log concatenation, multi-emitter aggregation, and ETL pipelines that join streams from different emitter versions. Consumers that require a uniform stream version MUST enforce that invariant themselves (e.g., by checking the first event and rejecting subsequent divergent events); the core spec imposes no uniformity requirement.

**Unknown-field preservation.** Pass-through tools (filters, pretty-printers, samplers, exporters that round-trip) MUST preserve any unknown fields encountered in an event when re-emitting it, regardless of `schema_version`. This preserves forward-compatibility for tools that sit between emitters and consumers with differing version expectations, and generalizes the existing preservation rules for unknown flag names (§4) and `"custom"` scope profiles (§4.7).

---

## 6. What ATOF Is Not

ATOF events are raw observations. They are not ATIF steps. See `[atof-to-atif-converter.md](./atof-to-atif-converter.md)` for the NeMo Agent Toolkit normative mapping from an ATOF stream to an ATIF trajectory.

Key structural differences from ATIF:


| Property          | ATOF events                                                     | ATIF steps                                                                  |
| ----------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Start/End pairing | Un-merged: `ScopeStart` and `ScopeEnd` are separate events      | Merged: a single step captures a full LLM call                              |
| Sequencing        | No `step_id`; ordered only by `timestamp`                       | Sequential 1-based `step_id`; no gaps allowed                               |
| Source field      | No `source` discriminator                                       | Required field: `"user"`, `"agent"`, or `"system"`                          |
| Tool ancestry     | Only `parent_uuid` for scope-graph navigation                   | `step.extra.tool_ancestry[]` aligns by index with `step.tool_calls[]`       |
| Observations      | `ScopeEnd` events on tool scopes carry raw output independently | Consecutive tool results merged into a single `system` step                 |
| Computed fields   | None                                                            | `step_id` assigned sequentially; `final_metrics` computed from step metrics |


**ATOF does not have:** `step_id`, `source` field, merged observations, `tool_ancestry` per step, `schema_version` (see §9), sequential guarantees beyond timestamp ordering.

---

## 7. EXMP-01: Simple Tool Call

A minimal 6-event stream illustrating one complete tool call cycle. Each line is one JSON object.

```jsonl
{"kind":"ScopeStart","schema_version":"0.1","uuid":"scope-agent-001","parent_uuid":null,"timestamp":"2026-01-01T00:00:00Z","name":"simple_calculator_agent","scope_type":"agent","flags":[],"profile":null,"input":null,"data":null,"metadata":null}
{"kind":"ScopeStart","schema_version":"0.1","uuid":"llm-001","parent_uuid":"scope-agent-001","timestamp":"2026-01-01T00:00:01Z","name":"nvidia/nemotron-3-super-v3","scope_type":"llm","flags":[],"profile":{"model_name":"nvidia/nemotron-3-super-v3"},"input":{"messages":[{"role":"user","content":"What is 3 + 4?"}],"model":"nvidia/nemotron-3-super-v3","tools":[{"type":"function","function":{"name":"calculator__add","description":"Add two numbers","parameters":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}}}}}]},"data":null,"metadata":null}
{"kind":"ScopeEnd","schema_version":"0.1","uuid":"llm-001","parent_uuid":"scope-agent-001","timestamp":"2026-01-01T00:00:02Z","name":"nvidia/nemotron-3-super-v3","scope_type":"llm","flags":[],"profile":{"model_name":"nvidia/nemotron-3-super-v3"},"output":{"content":"The result of 3 + 4 is 7.","tool_calls":[{"id":"call_calc_001","type":"function","function":{"name":"calculator__add","arguments":"{\"a\": 3, \"b\": 4}"}}]},"status":"ok","error":null,"data":null,"metadata":null}
{"kind":"ScopeStart","schema_version":"0.1","uuid":"tool-001","parent_uuid":"scope-agent-001","timestamp":"2026-01-01T00:00:03Z","name":"calculator__add","scope_type":"tool","flags":[],"profile":{"tool_call_id":"call_calc_001"},"input":{"a":3,"b":4},"data":null,"metadata":null}
{"kind":"ScopeEnd","schema_version":"0.1","uuid":"tool-001","parent_uuid":"scope-agent-001","timestamp":"2026-01-01T00:00:04Z","name":"calculator__add","scope_type":"tool","flags":[],"profile":{"tool_call_id":"call_calc_001"},"output":7,"status":"ok","error":null,"data":null,"metadata":null}
{"kind":"ScopeEnd","schema_version":"0.1","uuid":"scope-agent-001","parent_uuid":null,"timestamp":"2026-01-01T00:00:05Z","name":"simple_calculator_agent","scope_type":"agent","flags":[],"profile":null,"output":null,"status":"ok","error":null,"data":null,"metadata":null}
```

**Note on event ordering:** The LLM `ScopeEnd` arrives at `t=02` before the tool `ScopeStart` at `t=03`. This is the correct order: the LLM decides to call the tool (emitting `ScopeEnd` with `tool_calls` in `output`), then the runtime dispatches the tool call. The exporter sorts by timestamp, so the ordering `LLM-ScopeEnd → Tool-ScopeStart → Tool-ScopeEnd` is required for correct correlation.

## 7.1 EXMP-02: Tool Error with Parent Recovery

A 4-event stream illustrating a tool that raises an exception, while the parent agent handles the error and completes normally. Demonstrates `status: "error"` on the failed scope and `status: "ok"` on the parent that recovered.

```jsonl
{"kind":"ScopeStart","schema_version":"0.1","uuid":"scope-agent-002","parent_uuid":null,"timestamp":"2026-01-01T00:00:00Z","name":"resilient_fetch_agent","scope_type":"agent","flags":[],"profile":null,"input":"Fetch https://example.invalid/data","data":null,"metadata":null}
{"kind":"ScopeStart","schema_version":"0.1","uuid":"tool-002","parent_uuid":"scope-agent-002","timestamp":"2026-01-01T00:00:01Z","name":"http_get","scope_type":"tool","flags":[],"profile":{"tool_call_id":"call_fetch_001"},"input":{"url":"https://example.invalid/data","timeout_s":5},"data":null,"metadata":null}
{"kind":"ScopeEnd","schema_version":"0.1","uuid":"tool-002","parent_uuid":"scope-agent-002","timestamp":"2026-01-01T00:00:06Z","name":"http_get","scope_type":"tool","flags":[],"profile":{"tool_call_id":"call_fetch_001"},"output":null,"status":"error","error":{"type":"TimeoutError","message":"Request to https://example.invalid/data exceeded 5s timeout","stack":"Traceback (most recent call last):\n  File \"tools/http.py\", line 42, in http_get\n    resp = await client.get(url, timeout=timeout_s)\nTimeoutError: ..."},"data":null,"metadata":null}
{"kind":"ScopeEnd","schema_version":"0.1","uuid":"scope-agent-002","parent_uuid":null,"timestamp":"2026-01-01T00:00:07Z","name":"resilient_fetch_agent","scope_type":"agent","flags":[],"profile":null,"output":"Unable to fetch the requested URL; the endpoint timed out.","status":"ok","error":null,"data":null,"metadata":null}
```

**Note on status propagation:** The tool scope reports `status: "error"` with a populated `error` object carrying the exception type, message, and a stack trace. The parent agent scope caught the tool error, synthesized a user-facing response, and reports `status: "ok"` with its own `output`. Each scope reports its own terminal status (Section 5.6) — error does not auto-propagate up the scope graph.

---

## 8. Design Rationale

This section records design decisions made during the spec's pre-release development. References to "earlier iterations" describe shapes explored before landing on the current design; the spec is pre-release at version 0.1 and has not yet committed to a stable format.

**Why collapse `LLMStart`/`LLMEnd`/`ToolStart`/`ToolEnd` into `ScopeStart`/`ScopeEnd`?**
An earlier iteration gave each scope subject its own event kind, duplicating ~80% of the envelope fields across four types and requiring a new event kind for every new scope subject (retriever, guardrail, evaluator, …). The current design uses `scope_type` as the sole discriminator and delegates subject-specific shape to `profile` sub-schemas (Section 4). New scope types extend the format via a new profile entry, not a new event kind. This mirrors OpenTelemetry's single-span-shape-with-kind approach.

**Tradeoff — grep distinctness.** A raw `grep LLMStart` no longer works; consumers must filter on `"scope_type":"llm"` instead. Tooling that consumes ATOF streams (pretty-printers, filters) is expected to offset this. Documented here so the tradeoff is explicit.

**Why a single `input` / `output` pair and not separate raw + structured payloads?**
An earlier iteration carried both `input`/`output` (raw, any) and `annotated_request`/`annotated_response` (structured, codec-decoded). This doubled the payload fields on LLM events and bound the wire format to a specific codec pipeline. The current design collapses to a single payload field per side: opaque by default, structured (per a codec profile) when a codec is registered. Emitters that wish to preserve both raw and structured forms place the structured form in `input`/`output` and stash the raw bytes in `data` or `metadata`.

**Why is the per-scope structured field named `profile` and not `scope_data`?**
An earlier iteration used `scope_data`, which was nearly identical to the free-form `data` field and consistently led readers to confuse them — one is spec-governed per `scope_type`, the other is an opaque caller payload. The name `profile` ties the field directly to the Section 4 terminology (*scope profiles*) and removes the collision.

**Why is the behavioral flag field named `flags` and not `attributes`?**
An earlier iteration used `attributes`, which collided with the OpenTelemetry convention of an arbitrary key/value attribute map. In ATOF the field is strictly a set of boolean flag names. `flags` matches the shape (`string[]`), matches the prose vocabulary ("flag names", "flag vocabulary"), and sidesteps the false cognate.

**Why does `ScopeEnd` repeat `name`, `scope_type`, and `flags` from `ScopeStart`?**
So that each event is independently interpretable. Stream filters, sampling consumers, and partial-stream analyzers can classify a `ScopeEnd` without joining to its matching `ScopeStart`. A tail-and-filter pipeline such as `atof-stream | jq 'select(.scope_type == "llm" and .kind == "ScopeEnd")'` works directly; no in-memory `uuid → scope_state` table is required. The wire-size cost is small (and further reduced by transport compression); the consumer-side cost of "absent means inherit from Start" semantics — state-tracking per open scope, plus ambiguity when `profile` legitimately differs between Start and End — is disproportionately larger. This mirrors OpenTelemetry, where span-close carries the full attribute set, not a delta.

Note that `data` and `metadata` are *per-event* payloads, not per-scope. They are not "duplicated" on `ScopeEnd` — each event carries its own, and emitters MAY attach different values at Start and End (e.g., span-open vs. span-close tracing).

**Why a three-state `status` enum (`"ok"`, `"error"`, `"cancelled"`) on `ScopeEnd`?**
Without an explicit status, `output=null` is ambiguous between "scope returned nothing", "scope raised an exception", and "scope was cancelled" — three outcomes that consumers (metrics, error reporting, ATIF conversion) need to distinguish. Making `status` a required, first-class field separates the question "how did this scope terminate?" from "what did it return?", so consumers do not have to sniff `output` shape or look for sentinel values.

`"cancelled"` is distinct from `"error"` because cancellation is a normal control-flow outcome (timeout, parent cancel, explicit cancel) that most consumers should handle differently from an exception. For example, ATIF conversion should treat cancelled scopes as non-failures when computing aggregate success metrics, and trace viewers typically surface cancelled scopes with different styling from errored ones.

**Why keep the `error` object minimal (`type`, `message`, optional `stack`)?**
These three fields cover the debuggability essentials without imposing a taxonomy. Additional context — HTTP status codes, provider error codes, retry hints, nested cause chains — is inherently vendor- or domain-specific and does not generalize across emitters. Such context belongs in `data`, where it is opaque to ATOF and safely ignored by consumers that do not care about the specific emitter's shape. The error object avoids bundling a flat list of ten rarely-populated fields and keeps the core contract small.

**Why carry `schema_version` on every event rather than once per stream?**
Two reasons, both rooted in prior design decisions. First, it mirrors the principle laid out in "Why does `ScopeEnd` repeat `name`, `scope_type`, and `flags`?" above: every event is independently interpretable without joining to its siblings. Sampling consumers, stream filters, partial-stream analyzers, and log-tailing pipelines all benefit. A stream-header event would force any consumer that processes a single event in isolation to either maintain a uuid-to-version dispatch table or fail opaquely when the header was dropped, sampled out, or never present. Second, it makes mixed-version streams a first-class use case — concatenated logs, multi-emitter aggregation, and ETL joins across emitter generations produce streams with per-event version divergence, and `schema_version` lets consumers dispatch correctly without out-of-band coordination. The wire cost (≈22 bytes per event, near-zero after transport compression) is small compared to the consumer-side cost of working around a stream-header design.

**Why string `"MAJOR.MINOR"` and not a structured version object or integer?**
String `"MAJOR.MINOR"` is the minimum machine-parseable format that supports the dispatch rules in Section 5.7. A structured object (`{"major": 0, "minor": 1}`) costs more bytes per event and requires consumers to write an accessor on every access. An integer (e.g., `1` for v0.1, `2` for v0.2) loses the major/minor distinction and forces arbitrary mappings. String equality handles the common same-version case in a single comparison; consumers that need major/minor comparison split on `.` once.

**Why polymorphic `timestamp` (string or integer) instead of one canonical form?**
The two timestamp uses — human debugging and high-throughput machine ingestion — have directly opposed performance and readability tradeoffs. RFC 3339 strings are ~15× slower to parse than integer casts in most runtimes and ~50% larger on the wire, but are self-documenting when a human reads raw log output. Forcing every emitter to pay the string cost penalizes throughput-sensitive pipelines; forcing every emitter to use integers penalizes ad-hoc debuggability. Polymorphism lets each emitter pick per its use case, at the cost of a one-line type dispatch on the consumer side. The cost is contained because the dispatch is trivial (isinstance/typeof) and the normalized form (integer microseconds) is the universal sort key.

**Why epoch microseconds and not epoch nanoseconds as the integer form?**
JSON numbers are IEEE 754 doubles with 53 bits of integer precision. Nanoseconds since epoch exceed that range today and would silently lose precision in any JSON parser that follows the spec. Microseconds fits safely and preserves enough precision for agent-scope event correlation (the tightest observed agent schedulers resolve events at tens of microseconds, not nanoseconds). Use cases requiring nanosecond precision — e.g., OpenTelemetry-compatible distributed tracing at hardware-level granularity — can use the RFC 3339 string form, which preserves arbitrary sub-second precision via its decimal fraction.

**Why split codec profiles and the ATIF converter into companion documents?**
Codec profiles (OpenAI Chat, Anthropic Messages, …) are provider-specific overlays, not core ATOF. The ATOF→ATIF converter is a consumer-specific downstream layer. Keeping both out of the core spec lets non-NAT emitters adopt ATOF without reading converter or codec content, and lets NAT evolve those layers independently of the wire format.

---

## 9. Roadmapped Issues

The following structural gaps are acknowledged but deferred to keep the current pre-release scope bounded. They are non-normative and will be addressed in a subsequent revision.

1. **Emitter / resource identity.** No analog to OpenTelemetry's Resource — useful when multiple services feed into one event store. Consider a `resource` block or a dedicated stream-header event carrying `service.name`, runtime version, host identity, etc.
2. **Streaming chunk event type (conditional).** A dedicated `Chunk` event for incremental streaming output is deferred pending a concrete use case. The primary streaming concern — partial output on termination — is already handled: when a `"streaming"`-flagged scope terminates with `status == "error"` or `status == "cancelled"`, the `output` field MAY carry the chunks accumulated before the terminal event, representing a partial response (§3.2, §4.1). A `Chunk` event would only be warranted if a consumer required real-time visibility into in-flight chunks before `ScopeEnd` — e.g., a UI that renders tokens as they arrive rather than at scope close. Absent such a use case, the `"streaming"` flag plus `status`-gated partial output in `ScopeEnd` are sufficient.
3. **Common Flags section placement.** §4.1 "Common Flags" is currently nested inside §4 "Scope Profiles", but flags are not a profile — they are a cross-cutting field on `ScopeStartEvent`/`ScopeEndEvent` that applies across all scope types regardless of profile. The current placement reflects that profile subsections reference flags inline, not that flags semantically belong to profiles. A subsequent revision should either (a) promote Common Flags to its own top-level section or (b) move it under §2 "Common Event Fields" alongside other shared fields. This is a documentation-only restructure; field semantics, wire format, and the `flags` enum vocabulary are unchanged. Downstream consumers that cite "spec §4.1" (including the `nvidia_nat_atif` reference implementation's `flags.py` docstring) will need to track the new section number.

---

## 10. Reference Implementation

The Toolkit-native ATOF emitter and consumer live under `src/nat/atof/`. The ATOF→ATIF converter is specified in `[atof-to-atif-converter.md](./atof-to-atif-converter.md)` and implemented in `src/nat/atof/scripts/atof_to_atif_converter.py`.


| Module                                    | Entry Point                       | Description                                 |
| ----------------------------------------- | --------------------------------- | ------------------------------------------- |
| `nat.atof.scripts.atof_to_atif_converter` | `convert(events) → Trajectory`    | Convert typed Event list to ATIF Trajectory |
| `nat.atof.scripts.atof_to_atif_converter` | `convert_file(path) → Trajectory` | Read JSONL file and convert                 |
| `nat.atof.io`                             | `read_jsonl(path) → list[Event]`  | Parse ATOF JSONL to typed events            |
| `nat.atof.io`                             | `write_jsonl(events, path)`       | Serialize events to JSONL                   |
