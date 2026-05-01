# ATOF → ATIF Conversion Guide

A specification for translating Agentic Trajectory Observability Format  
(ATOF) v0.1 event streams into Agent Trajectory Interchange Format (ATIF) v1.7  
trajectories.

This document is aimed at being implementation-neutral. It captures the **rules** and
**philosophy** that any ATOF→ATIF mapper must follow, regardless of
language, provider, or framework. The final section (§7) sketches how a
specific implementation — the `nat.atof` Python package shipped with the
NeMo Agent Toolkit — realizes these rules and how to extend it for new
**consumer-side** schemas. **Producer-side** schema delivery is left as
a placeholder (§8) pending a future ATOF revision.

The intent: a coding assistant or engineer reading this guide should be
able to write a correct ATOF→ATIF mapper for any new provider in any
language, given only the spec links in §9.

---

## 1. Background

### 1.1 What ATOF is

ATOF is a wire format for **runtime observation** of agent execution. It
captures events as they happen — scopes opening and closing, marks being
placed — serialized as JSON Lines. Producers (instrumented agent
runtimes, observability SDKs) emit ATOF; consumers (replay systems,
validators, eval harnesses) ingest it.

ATOF makes few assumptions about the agent. Each event carries:

- A common envelope (`uuid`, `parent_uuid`, `timestamp`, `name`, optional
`metadata`)
- A `data` payload — **opaque**, **producer-defined**
- An optional `data_schema = {name, version}` identifier declaring the
payload's shape

ATOF defines two event kinds:

- `**ScopeEvent`** — paired start/end events sharing a `uuid`. Represents
a span of work: an agent turn, an LLM call, a tool invocation, a
retrieval. Each scope carries a `category` (`agent`, `llm`, `tool`,
`function`, `retriever`, `embedder`, `reranker`, `guardrail`,
`evaluator`, `custom`, `unknown`) and an optional `category_profile`
with category-specific typed fields (`model_name` for `llm`,
`tool_call_id` for `tool`, `subtype` for `custom`).
- `**MarkEvent`** — unpaired, point-in-time. Represents a checkpoint, a
session boundary, a user notification.

### 1.2 What ATIF is

ATIF is a **static interchange format** for completed trajectories.
Where ATOF captures motion, ATIF captures result. A `Trajectory`
contains an ordered list of `Step`s; each `Step` represents a single
sourced action.

`Step` structure:

- `source` ∈ {`user`, `system`, `agent`}
- `message` — string or multimodal content
- `tool_calls` — list of issued tool calls (assistant-initiated)
- `observation.results[]` — tool results, each linking back to a
`tool_call` by `source_call_id`
- Ancestry, timing, and per-step metadata

ATIF is the format consumed by trajectory validators, eval frameworks,
and replay tools. It is **higher-level** than ATOF — many low-level
events collapse into a single ATIF step.

### 1.3 Why the conversion matters

ATOF is what producers naturally emit. ATIF is what consumers want. The
conversion is the seam between live observation and offline analysis. A
faithful conversion must:

- Preserve every user-visible turn
- Preserve every assistant action and every tool result
- Reconcile `tool_call_id` ↔ tool result across event streams
- Filter out wire-level redundancy (echoed tool results, prior assistant
turns re-sent on subsequent LLM calls)
- Tolerate producer-specific payload shapes without losing content

---

## 2. Conceptual Model

### 2.1 The mapping problem

ATOF carries low-level **events**; ATIF carries high-level **steps**. The
conversion is N-to-M: many events collapse into a few steps.
Specifically:

- A user's question + the LLM's response + any tool round-trips collapse
into ~2-3 steps (user, agent-with-tool-call-and-observation, agent
final).
- Tool scope events between LLM calls don't produce steps directly —
they produce **observations** that attach to the agent step that
issued the matching `tool_call`.
- Mark events optionally lift to sourced steps when their payload
carries a recognizable role hint.
- Opaque (tier-1) scopes fall through to system steps.

### 2.2 The role of `data_schema`

ATOF's `data_schema` field is the bridge between producer-defined
payload shapes and consumer-side parsing. The wire envelope is
producer-agnostic, but the **contents** of `data` are not — different
LLM providers carry messages, tool calls, and tool results in different
layouts.

**The conversion rule:**

- The consumer maintains a registry mapping `(name, version) → extractor`.
- Each event is routed to its extractor via `event.data_schema`.
- Events without a `data_schema`, or with an unregistered one, fall
back to a built-in default extractor.

This is a **per-event** decision. A single trajectory MAY declare
multiple schemas — one event per LLM provider, all in the same stream.
**Per-event dispatch** is the architectural commitment.

### 2.3 Three extractor concerns

LLM events, tool events, and mark events have different payload shapes
and need separate extraction logic. A complete mapping framework defines
three extractor types, each backed by its own registry:


| Extractor type     | Pulls from event `data`                 | Used at                     |
| ------------------ | --------------------------------------- | --------------------------- |
| **LLM extractor**  | input messages, output text, tool_calls | every `llm` scope start/end |
| **Tool extractor** | serialized result string                | every `tool` scope-end      |
| **Mark extractor** | optional `(role, content)` lift         | every mark event            |


Each extractor MUST be a pure function over `data` — no side effects, no
network, no filesystem access. Empty results are returned as empty
collections/strings; the converter distinguishes "legitimately empty"
from "shape mismatch" at the dispatch layer.

---

## 3. Event Mapping Rules

This section gives the conversion rule for each ATOF event type.
Rule IDs use the form `M-NN`. Conformant mappers MUST satisfy every
rule.

### 3.1 Quick reference: which events emit which steps


| ATOF event                   | Step emission                                                          |
| ---------------------------- | ---------------------------------------------------------------------- |
| Agent scope-start            | None (informational only)                                              |
| Agent scope-end              | None (informational only)                                              |
| LLM scope-start              | One `user` or `system` step per **new** role=user/system input message |
| LLM scope-end                | Exactly one `agent` step (with text, tool_calls, or both)              |
| Tool scope-start             | None (cached for ancestry/args)                                        |
| Tool scope-end               | An observation result (attached later, not its own step)               |
| Mark event with role lift    | One sourced step (role from extractor)                                 |
| Mark event without role lift | One `system` step (opaque)                                             |
| Unknown/opaque scope-end     | One `system` step (tier-1 fall-through)                                |
| Unknown/opaque scope-start   | None (ignored)                                                         |


### 3.2 Time ordering

**Rule M-01.** All events MUST be sorted by timestamp (or its
microsecond normalization, `ts_micros`) before processing. The
conversion is order-deterministic. Events with equal timestamps MUST
use a stable secondary sort (typically arrival order or UUID).

### 3.3 Agent scope events

An `agent` scope marks the boundary of the trajectory. Its `data` MAY
carry an `input` (user query) on start and a `response` on end. In
well-formed trajectories the user input also appears as the first
message of the first LLM scope-start under this agent — the LLM scope
event is the canonical source.

**Rule M-02.** Treat the agent scope-start `data` as informational
only. Do NOT directly emit user steps from agent scope-starts. The LLM
scope extracts canonical user content.

**Rule M-03.** Treat the agent scope-end `data` as informational only.
Do NOT directly emit a final agent step from agent scope-ends. The last
LLM scope-end under this agent emits the canonical final agent step.

### 3.4 LLM scope-start

When an LLM scope-start fires, the consumer:

1. Resolves the LLM extractor for `event.data_schema`.
2. Calls `extract_input_messages(data)` — yields a list of
  `{role, content}` dicts.
3. For each message with `role ∈ {user, system}`: emits a sourced step
  IFF the `(parent_uuid, role, content_hash)` tuple is **new** under
   the current agent.

**Rule M-04 (dedup).** Steps are deduplicated per
`(parent_uuid, role, content_hash)`. On a multi-turn LLM call, the prior
user turn appears again in the input — the dedup ensures it doesn't
re-emit.

**Rule M-05 (role filter).** Only `role ∈ {user, system}` emits steps
from LLM input. Assistant turns are skipped (the assistant message is
re-emitted by the LLM scope-end). Tool-role turns and any
provider-specific role values not in the canonical set are skipped.

**Rule M-06 (multimodal pass-through).** When `content` is a list of
content parts (multimodal), pass it through unchanged. The dedup key is
the canonical JSON serialization of the list.

**Rule M-07 (parent reset on new sourced step).** Emitting a new user or
system step resets the "current agent step" pointer (any subsequent
buffered observations attach to the next agent step, not a previous
one).

### 3.5 LLM scope-end

When an LLM scope-end fires, the consumer:

1. Calls `flush_observations()` to attach any buffered tool results to
  the current agent step.
2. Resolves the LLM extractor.
3. Calls `extract_output_text(data)` — yields a string.
4. Calls `extract_tool_calls(data)` — yields a list of
  `{tool_call_id, function_name, arguments}` dicts.
5. Emits exactly ONE `agent` step with the text and tool_calls.

**Rule M-08 (output uniqueness).** Each LLM scope-end emits exactly one
agent step. A response with both text and tool_calls produces ONE agent
step carrying both. A response with only tool_calls emits an agent step
with empty `message` and the tool_calls. A response with only text emits
an agent step with the text and no tool_calls.

**Rule M-09 (shape mismatch).** If `data` is non-empty but BOTH
extracted text and extracted tool_calls are empty, the converter MUST
raise `ShapeMismatchError`. This catches schema mismatches at the
dispatch layer rather than silently dropping content.

### 3.6 Tool scope events

Tool scope events are paired and carry
`category_profile.tool_call_id`. The ID matches the `tool_call_id` of a
tool_call extracted from the parent agent's LLM scope-end.

When a tool scope-start fires:

- The converter MAY cache the args from `data` for later ancestry
reconciliation. No step is emitted.

When a tool scope-end fires:

1. Resolves the tool extractor.
2. Calls `extract_tool_result(data)` — yields a string (the serialized
  result).
3. Buffers an observation: `{source_call_id: tool_call_id, content: result}`.

**Rule M-10 (observation attachment).** Buffered observations attach to
the **most recent** agent step under the same parent. Attachment happens
at `flush_observations()` time, which is invoked by:

- The next LLM scope-start (before emitting any new sourced steps)
- The next LLM scope-end
- The trajectory's terminal flush

**Rule M-11 (ID consistency).** The `tool_call_id` on the tool scope's
`category_profile` MUST match a `tool_call_id` in the issuing
assistant's `tool_calls`. If the producer doesn't supply an ID natively,
the LLM extractor MUST synthesize a stable ID (e.g. `name__index`) and
the producer MUST use the same synthesis when emitting the tool scope.
Mismatches cause ATIF validation to reject the trajectory.

**Rule M-12 (orphan tool results).** If buffered observations have no
preceding agent step under the current agent (e.g. a tool fires before
any LLM call), emit a synthetic `system` step carrying the observations.
This preserves content rather than dropping it.

### 3.7 Mark events

Mark events have no scope semantics. They lift to ATIF steps via the
mark extractor.

When a mark event fires:

1. Resolves the mark extractor for `event.data_schema`.
2. Calls `extract_role_and_content(data)`.
3. If the result is `None`, emits a `system` step with the mark's `data`
  serialized as the message.
4. If the result is `(role, content)` where `role ∈ {user, system, agent}`,
  emits a step with that source and content.

**Rule M-13 (mark independence).** Marks are unpaired and independent.
A mark event that classifies as a sourced step does NOT participate in
LLM-derived dedup. The same content can appear as both a mark-lifted
step and an LLM-derived step without collision.

### 3.8 Unknown / tier-1 categories

Producers that can't classify their scopes emit `category: "unknown"`
with no `category_profile` and raw payloads in `data`. The conversion
falls back to:

**Rule M-14 (opaque fall-through).** Any scope-end event with
`category: "unknown"` (or any unrecognized category lacking a registered
extractor) emits a single `system` step with the JSON-serialized `data`
as the message. Scope-start events for unknown categories are ignored
(their data is informational only).

This guarantees that even zero-instrumentation producers produce a valid
ATIF trajectory — just one without rich agent/user/tool decomposition.

### 3.9 Other categories

ATOF defines additional categories (`function`, `retriever`, `embedder`,
`reranker`, `guardrail`, `evaluator`, `custom`). The mapping treats them
as follows:

- `**function`** — similar to `tool`. Buffered observations may attach.
Function scope-end's `data` is JSON-serialized into observation
content.
- `**retriever`, `embedder`, `reranker`, `guardrail`, `evaluator`** —
Tier-1 fall-through (Rule M-14) by default. Producers MAY register
custom extractors to lift them as observations or sourced steps.
- `**custom**` — REQUIRES `category_profile.subtype`. Treated as Tier-1
unless a custom extractor is registered for the `(custom, subtype)`
pair.

The mapping is extensible — the framework MUST support new categories
without changing the core dispatch rules.

---

## 4. Field-Level Mapping Philosophy

This section describes **how** extractors should be designed,
independent of any particular provider.

### 4.1 The schema-map approach

Provider payloads vary widely in shape but share a common skeleton:
input messages, output text, output tool calls. The mapping is mostly
**positional** — "the messages live at this path" — with a small
irreducible set of transforms that can't be expressed as paths alone.

The **schema-map architecture** captures both:

- **Declarative paths** — dotted paths (with array indices) telling the
engine where to find messages, text, tool calls, and per-tool-call
fields (ID, name, arguments).
- **Escape-hatch hooks** — three named functions that handle the
irreducible per-provider transforms.

Pure-paths providers (e.g. simple JSON-RPC-style payloads) require zero
hooks. Richer providers (block-list content, parts arrays, polymorphic
fields) use one or two hooks. No provider should require more than three
hooks; if it does, the schema is a poor fit for the schema-map
architecture and a sibling Protocol implementation is cleaner.

### 4.2 Paths vs. hooks

Use **paths** when:

- The field has a fixed location in the payload
- The field is a primitive or a homogeneous list

Use **hooks** when:

- Content is polymorphic at the same position (string OR list-of-blocks)
- Multiple ATIF fields are derived from a single payload field (text +
tool_calls from a single content block list)
- Per-call shape requires non-trivial logic (ID synthesis, JSON parsing,
multi-step field assembly)

### 4.3 The three irreducible hooks

These three transforms can NOT be expressed as field paths and must be
hooks:

#### Hook 1 — `normalize_input_messages(data) → list[{role, content}]`

Use when input content is polymorphic (string OR typed-block list) or
when role normalization is non-trivial. Returns a flattened ATIF-shaped
message list.

**Common responsibilities:**

- Walk a polymorphic content field, extract text blocks, drop wire-level
artifacts (tool_use markers, tool_result echoes — see §4.5)
- Normalize role names (e.g. `model` → `assistant`)
- Skip messages that have no surface text after extraction (avoids
duplicate user steps from echoed tool results)

#### Hook 2 — `normalize_output_message(data) → (text, tool_calls)`

Use when assistant text and tool calls coexist in a single structure
(e.g. a list of typed blocks). The hook walks the structure once and
returns both pieces. Without this hook, two separate path extractions
would scan the same array twice and need shared filtering logic.

**Common responsibilities:**

- Concatenate text-block text values into a single output string
- Collect tool-use-block fields into ATIF tool_call dicts
- Synthesize tool_call_ids when the provider doesn't supply them

#### Hook 3 — `transform_tool_call(raw_call, index) → ATIF tool_call`

Use for per-call adaptation when paths aren't enough. Useful for:

- Synthesizing `tool_call_id` from name + ordinal index
- Parsing `arguments` from a non-standard form (e.g. URL-encoded)
- Pulling fields from non-standard nesting

When set, this hook replaces the per-call path resolution entirely.

### 4.4 Role naming

Different providers use different role names for the assistant turn
(e.g. `assistant`, `model`). The mapping framework SHOULD support a
declarative `role_aliases` field that normalizes provider-specific role
values to a canonical vocabulary (`{user, assistant, system, tool}`)
before the converter sees them.

This normalization is necessary even though assistant turns are skipped
by the converter — downstream consumers may want consistent role labels,
and dedup keys benefit from canonical role values.

### 4.5 Tool result transport (provider-specific echoes)

Each LLM provider has its own way of representing tool results in the
**next** LLM call's input. Examples (without naming providers):

- A dedicated `tool`-role message with a result string and a
back-reference to the tool_call_id
- A typed `tool_result` block embedded in a `user`-role message's
content list
- A typed `function_response` part in a `user`-role parts list

In all cases, the converter MUST NOT emit a user step for the echoed
tool result — the result is already captured by the tool scope-end
event (Rule M-10). The extractor's `normalize_input_messages` hook is
responsible for skipping these echoed turns. A common heuristic that
covers most providers:

> Drop input messages whose content yields no plain text after block
> extraction.

Tool-use markers (the assistant-side echo of a prior tool call) are
similarly skipped — they're informational redundancy, not new content.

### 4.6 Error contracts

The conversion has two fail-fast checks at the dispatch layer:

1. **Schema validation.** If a JSON Schema is registered for the
  `data_schema`, the consumer validates `data` against it before
   extraction. Failure raises `DataSchemaViolationError`.
2. **Shape mismatch.** If `data` is non-empty but the resolved extractor
  yields no content, the converter raises `ShapeMismatchError`.

These two errors catch the failure modes that would otherwise silently
lose producer content. Conformant mappers MUST surface both as typed
exceptions, not warnings.

---

## 5. Conversion Invariants

A correct ATOF→ATIF mapper MUST guarantee these properties.


| ID   | Invariant                                                                                                                     |
| ---- | ----------------------------------------------------------------------------------------------------------------------------- |
| I-01 | Every user-visible turn appears as exactly one ATIF step with `source: "user"`.                                               |
| I-02 | Every assistant response appears as exactly one ATIF step with `source: "agent"`.                                             |
| I-03 | Every tool result appears as exactly one observation result, attached to the agent step that issued the matching `tool_call`. |
| I-04 | `tool_call_id` is consistent across the issuing `tool_call` and the receiving observation `source_call_id`.                   |
| I-05 | Multimodal content (lists of typed parts) is preserved end-to-end, not flattened to strings.                                  |
| I-06 | Tier-1 (opaque) producers produce valid ATIF, even if every step is `source: "system"`.                                       |
| I-07 | Schema validation, when enabled for a `(name, version)`, fires before any extraction.                                         |
| I-08 | An LLM event with non-empty `data` that yields no extractable content raises `ShapeMismatchError` (never silently empty).     |
| I-09 | Multi-schema streams are dispatched per-event; no per-stream schema lock.                                                     |
| I-10 | Conversion is deterministic given a sorted event sequence and a stable extractor registry.                                    |
| I-11 | `parent_uuid` ancestry is preserved in `function_ancestry` of each step.                                                      |
| I-12 | Mark events that don't classify as sourced steps still preserve their `data` as a system step's message.                      |


---

## 6. Multi-Schema Handling

The conversion architecture is designed to handle multiple producer
schemas in a single stream without producer-side coordination. Three
principles:

### 6.1 Per-event dispatch

Each event is independently routed by `event.data_schema`. The same
trajectory MAY declare different schemas on different events. A
heterogeneous stream — e.g. an orchestrator routing requests to LLM
specialists from three providers — is a first-class case, not a special
mode.

### 6.2 Opt-in registration

Non-default extractors are opt-in. The consumer registers them before
invoking the converter. Default extractors handle the common case (the
implementation chooses what's "common"). A stream that uses no other
providers requires no extra setup.

### 6.3 Graceful fallback

If an event declares a `data_schema` for which no extractor is
registered, the dispatch falls back to the default extractor. If the
default yields a shape mismatch, the converter raises `ShapeMismatchError`
so the consumer can fix the registration. **There is no silent loss of
content.**

---

## 7. Extending the Framework (consumer side)

> *This section is implementation-specific to the `nat.atof` Python
> package shipped with the NeMo Agent Toolkit. The principles above
> apply to any implementation; this section shows how **this**
> implementation realizes them, and how to extend it for a new consumer
> schema.*

### 7.1 The four registries

The `nat.atof` package maintains four module-level registries:


| Registry                  | Type                                          | Purpose                     |
| ------------------------- | --------------------------------------------- | --------------------------- |
| `LLM_EXTRACTOR_REGISTRY`  | `dict[(name, version), LlmPayloadExtractor]`  | LLM payload parsers         |
| `TOOL_EXTRACTOR_REGISTRY` | `dict[(name, version), ToolPayloadExtractor]` | Tool result parsers         |
| `MARK_EXTRACTOR_REGISTRY` | `dict[(name, version), MarkPayloadExtractor]` | Mark role-lift parsers      |
| `SCHEMA_REGISTRY`         | `dict[(name, version), dict]`                 | JSON Schemas for validation |


Registration is via `register_*()` helpers; lookup is via `resolve_*()`
resolvers. The default OpenAI chat-completions extractor is auto-registered
at import time; all other built-in extractors (Anthropic Messages, Gemini
generateContent) are opt-in.

### 7.2 Adding a new LLM consumer schema (by example)

Suppose a new LLM provider, `myco`, uses this payload shape:

```json
{
  "input": {
    "history": [{"role": "user", "text": "hello"}]
  },
  "output": {
    "answer": "Hi!",
    "actions": [
      {"action_id": "a1", "action_name": "lookup", "args": {"q": "x"}}
    ]
  }
}
```

To add it as a consumer-side extractor:

#### Step 1 — Define a SchemaMap with paths

```python
from nat.atof.extractors import SchemaMap, SchemaMapLlmExtractor

MYCO_LLM_V1_MAP = SchemaMap(
    name="myco/llm",
    version="1",
    input_messages_paths=("input.history",),
    output_text_paths=("output.answer",),
    output_tool_calls_paths=("output.actions",),
    tool_call_id_paths=("action_id",),
    tool_call_name_paths=("action_name",),
    tool_call_args_paths=("args",),
    tool_call_args_parse_json=False,  # args are already dicts
)
```

If the input messages use a non-canonical role name (e.g. `text` instead
of `content`), or content is polymorphic, add a `normalize_input_messages`
hook. If output text and tool calls coexist in a single structure, add a
`normalize_output_message` hook. If tool calls need ID synthesis, add a
`transform_tool_call` hook.

#### Step 2 — Define a JSON Schema for validation (optional but recommended)

```python
MYCO_LLM_V1: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "myco/llm@1",
    "type": "object",
    "anyOf": [
        {"type": "object", "required": ["input"]},
        {"type": "object", "required": ["output"]},
    ],
}
```

Keep the schema permissive — it's a **shape boundary** check, not a
field-by-field validation. Strict validation belongs at the producer.

#### Step 3 — Register both before invoking the converter

```python
from nat.atof import register_schema, register_llm_extractor

register_schema("myco/llm", "1", MYCO_LLM_V1)
register_llm_extractor(
    "myco/llm", "1", SchemaMapLlmExtractor(MYCO_LLM_V1_MAP)
)
```

#### Step 4 — (Optional) Bundle into a convenience helper

```python
def register_myco_llm_v1() -> None:
    register_schema("myco/llm", "1", MYCO_LLM_V1)
    register_llm_extractor(
        "myco/llm", "1", SchemaMapLlmExtractor(MYCO_LLM_V1_MAP)
    )
```

This mirrors the built-in `register_anthropic_messages_v1()` and
`register_gemini_generate_content_v1()` helpers.

### 7.3 Adding tool or mark extractors

The pattern is identical with the corresponding Protocol and registry:

- **Tool**: implement `ToolPayloadExtractor` (single method
`extract_tool_result(data) -> str | None`), register via
`register_tool_extractor(name, version, instance)`.
- **Mark**: implement `MarkPayloadExtractor` (single method
`extract_role_and_content(data) -> tuple[str, Any] | None`), register
via `register_mark_extractor(name, version, instance)`.

Tool and mark extractors don't use the schema-map architecture — their
contracts are too narrow to benefit from declarative paths. A direct
class implementing the Protocol is the right shape.

### 7.4 When to write a hook

Default to declarative paths. Reach for a hook only when:

- Content is polymorphic at one position → `normalize_input_messages`
- Output text and tool_calls share a structure → `normalize_output_message`
- Per-call processing requires synthesis → `transform_tool_call`

A hook should be small (typically 5-20 lines). If your hook is
approaching 50 lines, the schema may not be a good fit for the
schema-map architecture — consider a sibling class implementing
`LlmPayloadExtractor` directly.

### 7.5 Reference: built-in providers

The `nat.atof` package ships three built-in LLM schema maps as reference
implementations. Read these as templates when adding a new provider:


| Provider                | Schema map                       | Hooks used                                                              | Notes                                                                              |
| ----------------------- | -------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| OpenAI chat-completions | `OPENAI_CHAT_COMPLETIONS_V1_MAP` | none                                                                    | Pure paths — the simplest case.                                                    |
| Anthropic Messages      | `ANTHROPIC_MESSAGES_V1_MAP`      | `normalize_input_messages`, `normalize_output_message`                  | Polymorphic `content` (string OR block list); text and tool_use coexist in output. |
| Gemini generateContent  | `GEMINI_GENERATE_CONTENT_V1_MAP` | `normalize_input_messages`, `normalize_output_message` + `role_aliases` | Polymorphic `parts[]`; `model` → `assistant`; synthesized tool_call_ids.           |


Example trajectories exercising each are under
`packages/nvidia_nat_atif/examples/atof_to_atif/` (`exmp04` Anthropic,
`exmp05` Gemini, `exmp06` heterogeneous router using all three in one
stream).

### 7.6 Testing a new schema

The matrix-style test harness at
`packages/nvidia_nat_atif/tests/test_schema_validation.py` defines a
factory pattern with three scenario builders (`simple`, `nested`,
`multi_turn`). Adding a new provider is a one-step extension: implement
a `_PayloadFactory` subclass for the provider, add it to the
`_FACTORIES` dict, and the existing parametrized tests cover it.

---

## 8. Producer-Side Schema Declaration (Future)

> ⚠️ **Placeholder — to be specified once the producer story is built
> out.**

The current architecture is **consumer-driven**: producers declare their
schema per-event via `data_schema`, but the schema definition and
extractor logic must be pre-installed on the consumer side. This is fine
when producer and consumer are co-developed, but creates friction when:

- A consumer ingests trajectories from multiple producers without prior
coordination
- A new producer ships in production before the consumer has been
updated
- A trajectory is replayed from an archive whose original producer is no
longer running

A future ATOF revision should specify how producers can ship their
schema **along with** the trajectory. Three design options have been
sketched:


| Option                       | Mechanism                                                                             | Pros                                        | Cons                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------- |
| **A. Stream-level manifest** | Reserve the first JSONL line for `{"type": "atof_schema_manifest", "schemas": [...]}` | Backward-compat; explicit; easy to ship     | Requires wire-format reservation; opaque `extractor_plugin` references      |
| **B. Scope-start metadata**  | Embed in `metadata._atof_schemas` on the root agent ScopeStart                        | No wire-format change; zero-overhead opt-in | Late discovery (consumer can't pre-register); convention adherence required |
| **C. Out-of-band sidecar**   | Ship a manifest file alongside the JSONL                                              | Clean separation; signable independently    | Two-file coupling fragile; transports often drop sidecars                   |


**Recommendation when this work is taken up:** prototype Option A first
— least invasive, self-documenting in the stream itself. Decline Option
C unless storage transports demand it. Option B is a cheap fallback if
A hits backward-compat blockers.

This section will be populated when the design is locked in. Until then,
producers and consumers MUST coordinate out-of-band.

---

## 9. References

- **ATOF wire-format spec**: `[atof-event-format.md](../../../../atif-alignment/rfc/atof-event-format.md)` (in the
`atif-alignment` repo)
- **ATIF v1.7 trajectory model**: see [Harbor RFC 0001: Trajectory Format](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md)
and NeMo Agent Toolkit ATIF docs; canonical models in `nat.atif`
(Trajectory, Step, ToolCall, Observation)
- **Reference implementation**: `nat.atof` Python package
(`packages/nvidia_nat_atif` in the NeMo Agent Toolkit monorepo)
- **Example trajectories**: `packages/nvidia_nat_atif/examples/atof_to_atif/`

---

## Appendix A — Vocabulary Index

For consistency, mappers SHOULD use these terms with the meanings given.


| Term           | Meaning                                                                                                                                   |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Producer**   | The system emitting ATOF events (instrumented agent runtime, observability SDK).                                                          |
| **Consumer**   | The system ingesting ATOF events and producing ATIF (replay tool, validator, eval harness).                                               |
| **Event**      | A single ATOF JSON-Lines record.                                                                                                          |
| **Step**       | A single ATIF action with `source`, `message`, optional `tool_calls`, optional `observation`.                                             |
| **Schema**     | A `(name, version)` pair declaring the shape of `event.data`. Optional per-event.                                                         |
| **Schema map** | A declarative description of where ATIF-relevant fields live within a producer's payload, plus optional hooks for irreducible transforms. |
| **Extractor**  | A function/object that pulls ATIF fields from an event's `data`. Three types: LLM, tool, mark.                                            |
| **Hook**       | An imperative escape hatch in a schema map that handles a transform paths can't express.                                                  |
| **Dispatch**   | The act of resolving the right extractor for an event based on `event.data_schema`.                                                       |
| **Tier-1**     | An ATOF stream where producers can't classify scopes (everything is `category: "unknown"`). Falls through to system steps.                |
| **Tier-2**     | An ATOF stream with semantic categories and category profiles. Decomposes to rich ATIF.                                                   |


---

*This document is implementation-neutral except where explicitly marked
(§7). The conversion rules (§3-§5) and architectural philosophy (§4,
§6) apply to any ATOF→ATIF mapper regardless of language or framework.*