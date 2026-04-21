# ATOF-to-ATIF Examples

End-to-end examples exercising the ATOF v0.1 reference implementation. The
three main scenarios walk through the producer enrichment tiers in order —
EXMP-01 is tier-1, EXMP-02 is tier-2, EXMP-03 is tier-3 (spec §1.1). EXMP-03b
is a tier-2 side example showing error-recovery semantics.

This README doubles as the ATOF → ATIF conversion reference: the mapping
table, dispatch conventions, and known limitations live in the
[Conversion reference](#conversion-reference) section at the bottom.

## Scripts

- `generate_examples.py` — produces `output/exmpNN_atof.jsonl` for each
  scenario using the v0.1 public API (`StreamHeaderEvent`, `ScopeStartEvent`,
  `ScopeEndEvent`, `ErrorInfo`, `write_jsonl`).
- `convert_to_atif.py` — reads each regenerated JSONL, runs the ATOF→ATIF
  converter (`nat.atof.scripts.atof_to_atif_converter.convert_file`), and
  writes `output/exmpNN_atif.json` as a formatted ATIF `Trajectory`.

## The scenarios

### EXMP-01 — tier-1 raw pass-through

A calculator-shaped workflow where the producer can't classify any scope.
Every scope carries `scope_type: "unknown"`, `profile: null`, `schema: null`,
and opaque raw JSON in `input` / `output`. No `StreamHeader` is emitted —
tier-1 producers have nothing to declare. Demonstrates the floor: a valid
ATOF stream capturing only timing + raw payloads, no semantic tagging.

Converts to a 4-step ATIF trajectory of opaque `system` steps via the
reference converter's generic `ScopeEnd` fall-through. `Trajectory.agent.name`
uses the outermost root scope's `name` since no `scope_type: "agent"` event
is present.

**When to use:** runtime wrapping a third-party framework whose callback
fires a raw blob the wrapper can't classify.

### EXMP-02 — tier-2 semantic-tagged

Same calculator workflow as EXMP-01 but with every scope classified
(`scope_type: "agent"` / `"llm"` / `"tool"`) and `profile` populated
(`profile.model_name` for llm events, `profile.tool_call_id` for tool
events — see spec §4.4). `schema` remains `null` — no tier-3 decoding.

Converts to a 5-step rich ATIF trajectory (user → agent → system → user →
agent) with `Trajectory.agent.name` derived from the `scope_type: "agent"`
scope's name.

**When to use:** native producers that classify events at the hook site
but don't decode provider-specific request/response shapes.

### EXMP-03 — tier-3 schema-annotated

Same calculator workflow as EXMP-02 but every LLM event declares a schema
(`openai/chat-completions.v1`) and attaches structured `annotated_request` /
`annotated_response` payloads following OpenAI's native wire shape. The
`StreamHeader.schemas` registry carries an inline `$schema` body — priority-2
fallback for consumers without a local bundled schema.

Tool events stay tier-2 (no schema) to show that the schema layer is
per-event, not per-stream.

**When to use:** producers wrapping a known provider API; consumers that
want structured access to messages, params, tool defs, usage metrics
without bespoke per-provider parsing.

See `../../atof-schema-profiles.md` §7.1 for the full 4-priority schema
resolution protocol.

### EXMP-03b — tier-2 with error recovery (variant)

A web-search tool times out (`status: "error"` + `ErrorInfo`); the parent
agent catches the failure and reports `status: "ok"` with a graceful
output message. Demonstrates spec §5.2-5.3 — each scope reports its own
terminal status; parents may catch child errors.

Shares the tier-2 shape with EXMP-02; sits alongside EXMP-03 as a side
example of error-recovery semantics rather than a tier progression step.

**When to use:** showcase status semantics, error propagation, and
parent-side recovery patterns.

## Running

```bash
cd NeMo-Agent-Toolkit/packages/nvidia_nat_atif/examples/atof_to_atif
python generate_examples.py
python convert_to_atif.py
# Outputs in output/
```

## Event counts

| Scenario | Events | ATIF steps | Tier | Workflow                                        |
| -------- | ------ | ---------- | ---- | ----------------------------------------------- |
| EXMP-01  | 8      | 4          | 1    | Opaque wrapper: 3 unclassified inner callbacks  |
| EXMP-02  | 9      | 5          | 2    | Calculator: agent → llm → tool → llm → agent    |
| EXMP-03  | 9      | 5          | 3    | Calculator with OpenAI schema annotations       |
| EXMP-03b | 7      | 3          | 2    | Search: agent → llm → tool (timeout) → agent    |

EXMP-02, EXMP-03, and EXMP-03b each open with a `StreamHeaderEvent` at
position 0 (spec §3.4 — MUST be first when present). EXMP-01 omits the
header because tier-1 producers have nothing to declare — the header is
optional per the spec.

---

## Conversion reference

This section is the canonical mapping from ATOF event streams to ATIF
trajectories. The reference implementation lives at
`../../src/nat/atof/scripts/atof_to_atif_converter.py`; the code is the
source of truth for edge cases. This section documents the conventions any
consumer should follow to round-trip cleanly.

### Source mapping

ATIF requires every `Step` to declare a `source ∈ {"user", "agent", "system"}`.
ATOF events carry no `source` field — the converter derives it from the
event's `kind` and `scope_type`:


| ATOF event                        | Condition                                                 | ATIF `source` | Step content                                                                                                               |
| --------------------------------- | --------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `ScopeStartEvent`                 | `scope_type == "llm"`                                     | `user`        | `message` = serialized messages array from `event.input`                                                                   |
| `ScopeEndEvent`                   | `scope_type == "llm"`                                     | `agent`       | `message` = LLM response content; `tool_calls` extracted from `output`                                                     |
| `ScopeEndEvent`                   | `scope_type == "tool"`                                    | `system`      | merged into `observation.results[]`; consecutive tool ends flush as a single step                                          |
| `MarkEvent`                       | `data != null`                                             | `system`      | `message` = serialized `data` (null-data marks are skipped)                                                                |
| `ScopeStartEvent`                 | `scope_type == "agent"`                                   | (none)        | call-graph shaping only — `name` captured for `Trajectory.agent.name`                                                      |
| `ScopeEndEvent`                   | `scope_type ∉ {"llm", "tool", "agent"}`                   | `system`      | `message` = serialized `event.output`; ancestry + invocation timing preserved. Covers tier-1 opaque and unclassified types. |
| `ScopeStartEvent`                 | `scope_type ∉ {"llm", "agent"}`                           | (none)        | call-graph shaping only — included in `extra.ancestry` chains                                                              |
| `StreamHeaderEvent`               | (any)                                                     | (none)        | optional metadata carrier; never materializes as a step                                                                    |


**Tier-1 pass-through guarantee.** A strict tier-1 stream — every scope with
`scope_type == "unknown"`, `profile: null`, `schema: null` — converts to a
non-empty trajectory: each opaque `ScopeEndEvent` becomes a `source: "system"`
step whose `message` is the serialized raw `event.output`.
`Trajectory.agent.name` falls back to the outermost (root) `ScopeStart`'s
`name` when no `scope_type == "agent"` event is present.

### Why `(kind, scope_type)` as dispatch key

The three string literals `"llm"`, `"tool"`, `"agent"` are the conventions
the reference converter recognizes for **specialised** ATIF-step materialisation:

- **`llm`** scopes become paired user/agent steps with messages and
  tool-call extraction.
- **`tool`** scopes become merged observation results buffered between LLM
  turns.
- **`agent`** scopes populate `Trajectory.agent.name` only (no step emitted).

All **other** `ScopeEnd` events (`function`, `retriever`, `embedder`,
`reranker`, `guardrail`, `evaluator`, `custom`, `unknown`) fall into the
generic opaque-system-step arm — each contributes a `source: "system"` step
whose `message` is the serialised raw `event.output`. This guarantees that
**every tier produces a non-empty ATIF trajectory**: tier-1 streams yield a
sequence of opaque system steps; tier-2+ streams enrich that structure with
user/agent/observation steps where scopes are classified.

### Tool-result merging

Consecutive `tool` `ScopeEnd` events between two LLM turns produce **one**
ATIF system step with multiple `observation.results[]`, not one step per
tool result. Per Harbor's ATIF RFC, observations belong to the agent turn
that produced them — a single `system` step with N results models "the
system returning the results of the N tools the agent just called."
Per-tool steps would inflate `step_id` counts and confuse downstream
metrics.

A flush happens at three triggers:

1. **Next LLM turn begins** (`ScopeStartEvent` with `scope_type == "llm"`)
   — flushes pending observations into a single `system` step before the
   new `user` step is appended.
2. **MarkEvent with data** — flushes pending observations before the mark's
   `system` step.
3. **End of stream** — flushes any remaining observations.

### ID mappings

| ATOF field                                            | ATIF field                                  | Mapping rule                                                            |
| ----------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------- |
| `event.uuid`                                          | `extra.ancestry.function_id`                | Direct                                                                  |
| `event.parent_uuid`                                   | `extra.ancestry.parent_id`                  | Direct (empty string if `null`)                                         |
| `event.name`                                          | `extra.ancestry.function_name`              | Direct                                                                  |
| `name_map[parent_uuid]`                               | `extra.ancestry.parent_name`                | Looked up via pre-pass `uuid → name` map; `"unknown"` if unresolved     |
| `event.profile.tool_call_id` (`scope_type == "tool"`) | `tool_calls[*].tool_call_id`                | Read from the `profile` sub-object (spec §4.4)                          |
| `event.profile.tool_call_id` (`scope_type == "tool"`) | `observation.results[*].source_call_id`     | Same value                                                              |
| `event.profile.model_name` (`scope_type == "llm"`)    | `Trajectory.agent.model_name`               | First LLM `ScopeEnd`'s `profile.model_name` wins                        |
| `ScopeStartEvent.name` (`scope_type == "agent"`)      | `Trajectory.agent.name`                     | First `agent` scope wins; falls back to root `ScopeStart.name` if absent |

### Producers that need different mappings

If your runtime emits a `scope_type` not in the specialised list
(`llm`/`tool`/`agent`) and you want richer ATIF output than the generic
system-step fallback provides, you have three options:

1. **Wrap a known type** — emit your custom scope as `scope_type == "llm"`
   or `scope_type == "tool"` and use `attributes` + `data` fields to carry
   the distinguishing semantics.
2. **Implement a custom converter** — fork the reference converter's
   dispatch loop and add your `scope_type` arm.
3. **Use `MarkEvent` with structured `data`** — for non-lifecycle
   observations, a `Mark` with non-null `data` produces a `system` step
   with the data serialized into `message`.

Option 3 is the fastest path for one-off events. Option 2 is correct when
your scope type has lifecycle semantics (start + end + status).

### Known limitations

- **Tools without `profile.tool_call_id`.** Tool events emitted without a
  `profile.tool_call_id` (tier-1 producers that don't have
  provider-assigned correlation IDs) produce
  `observation.results[*].source_call_id == None`. The call graph is still
  reconstructable via `parent_uuid` / `extra.ancestry`, but invocation-level
  correlation is lost.
- **Naive RFC 3339 timestamps.** `datetime.fromisoformat()` accepts naive
  ISO 8601 strings (no timezone), which spec §6.1 forbids. Naive strings
  reinterpret in the consumer's local timezone and can silently shift
  `ts_micros` by hours between environments. Producers MUST emit `Z` or an
  explicit UTC offset.
- **Null-data Marks.** A `MarkEvent` whose `data` field is `null` is
  skipped — no step is emitted. If you need a marker step with empty
  content, emit `data: {}` instead — the converter produces a `system`
  step with `message == "{}"`.
- **StreamHeader not consulted for validation.** The reference converter
  skips `StreamHeaderEvent`s in the main dispatch loop. It does not
  validate `annotated_*` payloads against the `schemas` registry — schema
  resolution is a consumer-side concern (see
  `../../atof-schema-profiles.md` §7.1).

### Public API

Two entry points in `../../src/nat/atof/scripts/atof_to_atif_converter.py`:

```python
def convert(events: list[Event]) -> Trajectory:
    """In-memory: typed ATOF events → validated ATIF Trajectory."""


def convert_file(input_path: str | Path, output_path: str | Path | None = None) -> Trajectory:
    """File-based: read .jsonl → convert → optionally write ATIF JSON."""
```

Both return a Pydantic-validated `nat.atif.trajectory.Trajectory` (the
NAT-side ATIF model that mirrors Harbor's `0001-trajectory-format.md` RFC
v1.7).

---

## See also

- `../../atof-event-format.md` — canonical v0.1 spec (wire format, scope types, status semantics)
- `../../atof-schema-profiles.md` — schema identifiers, namespace conventions, 4-priority resolution (§7.1)
- `../../src/nat/atof/scripts/atof_to_atif_converter.py` — reference converter implementation
- `../../tests/test_tier1_conversion.py` — unit tests for tier-1 conversion behaviour
