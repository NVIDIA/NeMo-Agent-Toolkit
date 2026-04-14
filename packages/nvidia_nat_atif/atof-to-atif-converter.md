<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
-->

# ATOF → ATIF Converter

**Version:** 0.2
**Date:** 2026-04-14
**Status:** Active
**Companion to:** [`atof-event-format.md`](./atof-event-format.md) (ATOF wire format)

---

## 1. Purpose

This document specifies the canonical mapping from an ATOF event stream to an ATIF (Agent Trajectory Interchange Format) trajectory. It is the **normative reference** for any ATOF consumer that needs to produce ATIF output.

ATOF is a wire format for raw runtime observations (start/end events, marks, schema headers). ATIF is a higher-level structure: an ordered sequence of conversation turns (`source`, `message`, `tool_calls`, `observation`) with computed metadata (`step_id`, ancestry, timing). The converter is the layer that bridges the two.

The reference implementation lives at `src/nat/atof/scripts/atof_to_atif_converter.py`. Non-NAT consumers MAY implement different conversion strategies — this document specifies the convention the reference converter follows so that producers using the documented `scope_type` vocabulary will round-trip cleanly.

## 2. Scope

This document covers:

- The mapping from ATOF event kinds + `scope_type` values to ATIF `step.source` values
- The accumulator state machine that merges multi-event observations into single ATIF steps
- ID and timing field mappings
- Ancestry reconstruction from `parent_uuid` chains
- Handling of `StreamHeaderEvent`, opaque profiles, and unknown `scope_type` values
- Known limitations and extension points

This document does NOT cover:

- The ATOF wire format itself — see `atof-event-format.md`
- The IntermediateStep → ATIF pipeline used by NAT's evaluation tooling — see `intermediate-step-to-atif-mapping.md`
- The ATIF schema — see Harbor's `0001-trajectory-format.md` RFC
- ATIF `step.extra` field conventions — see `atif-step-extra-guide.md`

## 3. ATIF Source Mapping

ATIF requires every `Step` to declare a `source ∈ {"user", "agent", "system"}`. ATOF events carry no `source` field — the converter derives it from the event's `kind` and `scope_type`.

### 3.1 Mapping Table

| ATOF event           | Condition                       | ATIF `source` | Step content                                                            |
| -------------------- | ------------------------------- | ------------- | ----------------------------------------------------------------------- |
| `ScopeStartEvent`    | `scope_type == "llm"`           | `user`        | `message` = serialized messages array from `event.input`                |
| `ScopeEndEvent`      | `scope_type == "llm"`           | `agent`       | `message` = LLM response content; `tool_calls` extracted from `output`  |
| `ScopeEndEvent`      | `scope_type == "tool"`          | `system`      | merged into `observation.results[]`; flushed as one step (see §4)       |
| `MarkEvent`          | `data != null`                  | `system`      | `message` = serialized `data`                                           |
| `ScopeStartEvent`    | `scope_type == "agent"`         | (none)        | call-graph shaping only — `name` captured for `Trajectory.agent.name`   |
| `ScopeStartEvent`/`ScopeEndEvent` | any other `scope_type` | (none)        | call-graph shaping only — included in `extra.ancestry` chains           |
| `StreamHeaderEvent`  | (any)                           | (none)        | consumed by the schema-registry pre-pass; never materializes as a step  |

### 3.2 Why `scope_type` and not `kind`

`kind` is a closed enum (`ScopeStart` | `ScopeEnd` | `Mark` | `StreamHeader`); `scope_type` is an open-vocabulary string (ATOF spec §3.1, D-11). The `(kind, scope_type)` pair is the dispatch key. This separation lets producers introduce new scope categories (`"retriever"`, `"reranker"`, `"router"`) without forking the wire format — the converter SKIPS unknown `scope_type` values rather than rejecting them, preserving forward compatibility.

The three string literals `"llm"`, `"tool"`, `"agent"` are **conventions** the reference converter recognizes. Producers that emit these scope types and follow the reference profile schemas (`default/llm.v1`, `default/tool.v1`) will produce well-formed ATIF.

### 3.3 Producers that need different mappings

If your runtime emits a `scope_type` not in the table above and you want it to materialize as an ATIF step, you have three options:

1. **Wrap a known type** — emit your custom scope as `scope_type == "llm"` or `scope_type == "tool"` and use the `flags` and `profile` fields to carry the distinguishing semantics.
2. **Implement a custom converter** — fork the reference converter's dispatch loop and add your `scope_type` arm.
3. **Use `MarkEvent` with structured `data`** — for non-lifecycle observations, a `Mark` produces a `system` step with the data serialized into `message`.

Option 3 is the fastest path for one-off events. Option 2 is correct when your scope type has lifecycle semantics (start + end + status).

## 4. The Accumulator State Machine

The converter is a single-pass accumulator over events sorted by `ts_micros`. It maintains:

- `step_dicts: list[dict]` — the output ATIF steps, built incrementally
- `pending_observations: list[dict]` — tool results buffered between LLM turns
- `pending_obs_timestamp: str | int | None` — timestamp of the first buffered tool result (used as the system step's `timestamp`)
- `last_tool_call_map: dict[str, str]` — `function_name → tool_call_id` (fallback for opaque profiles)
- `last_tool_call_order: list[str]` — declaration order of tool calls in the most recent agent step (used for stable observation ordering)
- `current_agent_step_idx: int | None` — index of the most recent agent step (used to attach `tool_ancestry` and `tool_invocations` after all child tool events arrive)
- `pending_tool_ancestry`, `pending_tool_invocations` — buffered ancestry/timing rows attached to the current agent step at flush

### 4.1 Flush Semantics

A flush happens at three triggers:

1. **Next LLM turn begins** (`ScopeStartEvent` with `scope_type == "llm"`) — flushes pending observations into a single `system` step BEFORE the new `user` step is appended; finalizes the current agent step's `tool_ancestry`/`tool_invocations` extras.
2. **MarkEvent with data** — flushes pending observations BEFORE the mark's `system` step.
3. **End of stream** — flushes any remaining observations + finalizes the last agent step.

This means consecutive `tool` ScopeEnd events between two LLM turns produce **one** ATIF system step with multiple `observation.results[]`, NOT one step per tool result. The flush order is: `flush_observations()` → `finalize_agent_extra()` → append next step.

### 4.2 Why merge tool results

Per Harbor's ATIF RFC, observations belong to the agent turn that produced them — a single `system` step with N results models "the system returning the results of the N tools the agent just called." Per-tool steps would inflate `step_id` counts and confuse downstream metrics (e.g., turn-count, tool-call density).

## 5. ID Mappings

| ATOF field                                    | ATIF field                                   | Mapping rule                                              |
| --------------------------------------------- | -------------------------------------------- | --------------------------------------------------------- |
| `event.uuid` (any event)                      | `extra.ancestry.function_id`                 | Direct                                                    |
| `event.parent_uuid`                           | `extra.ancestry.parent_id`                   | Direct (empty string if `null`)                           |
| `event.name`                                  | `extra.ancestry.function_name`               | Direct                                                    |
| `name_map[parent_uuid]`                       | `extra.ancestry.parent_name`                 | Looked up via the pre-pass `uuid → name` map; `"unknown"` if unresolved |
| `ScopeStartEvent.uuid` (`scope_type == "llm"`) | `extra.tool_invocations[*].invocation_id`   | For tool ScopeEnds whose `parent_uuid` matches            |
| `event.profile.tool_call_id` (`default/tool.v1`) | `tool_calls[*].tool_call_id`              | Read via `model_dump(by_alias=True).get("tool_call_id")` (Pitfall 7) |
| `event.profile.tool_call_id` (`default/tool.v1`) | `observation.results[*].source_call_id`   | Same value                                                |
| `(none)`                                      | `step_id`                                    | Generated 1-indexed sequence after all steps are built    |
| `ScopeStartEvent.uuid` (`scope_type == "agent"`) | `Trajectory.agent.name`                   | Read from the matching `name` field (first agent scope)   |

### 5.1 Profile vendor-field extraction

Wire-deserialized profiles arrive as base `ProfileContract` instances (D-22 anchor) with vendor fields in `model_extra`. **Attribute access silently returns `None`** for vendor fields — the canonical extraction pattern is:

```python
if event.profile is not None:
    schema_id = _schema_id_string(event.profile.schema_id)
    if schema_id == "default/tool.v1":
        raw = event.profile.model_dump(by_alias=True).get("tool_call_id")
        if isinstance(raw, str):
            tool_call_id = raw
```

This is documented in the reference converter as Pitfall 7 (`atof_to_atif_converter.py` lines 339-344). The same idiom is used for `default/llm.v1` `model_name` extraction.

### 5.2 Tool call correlation fallback

When a tool `ScopeEndEvent` carries no `default/tool.v1` profile (or the profile has no `tool_call_id`), the converter falls back to a name-based map populated from the most recent LLM agent step's `tool_calls[]`:

```python
if not tool_call_id:
    tool_call_id = last_tool_call_map.get(event.name)
```

**Limitation (WR-03):** This fallback collides on repeated tool names within a single LLM turn. If the agent calls `calculator__add` twice in one turn, the second invocation's `tool_call_id` overwrites the first in `last_tool_call_map` and both observations pair with the second (later) ID. The primary v0.2 reference path (`default/tool.v1` with explicit `tool_call_id`) is unaffected — this only applies to opaque or vendor profiles. See §8.1 for mitigations.

## 6. Timing Mappings

| ATOF field                                     | ATIF field                                              | Mapping rule                                                    |
| ---------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------- |
| `event.timestamp` (string OR int)              | `step.timestamp`                                        | RFC 3339 string passes through unchanged; integer microseconds serialize to RFC 3339 (see §7.1 of the wire-format spec) |
| `event.ts_micros` (computed)                   | (sort key only)                                         | Used for stable cross-format ordering before any other dispatch |
| `start_ts_map[event.uuid]` (microseconds)      | `extra.invocation.start_timestamp`                      | Converted to seconds-as-float at millisecond precision (`round(micros / 1_000_000, 3)`) |
| `event.ts_micros` (ScopeEnd)                   | `extra.invocation.end_timestamp`                        | Same conversion                                                 |
| Tool ScopeStart `ts_micros`                    | `extra.tool_invocations[*].start_timestamp`             | Same conversion                                                 |
| Tool ScopeEnd `ts_micros`                      | `extra.tool_invocations[*].end_timestamp`               | Same conversion                                                 |

### 6.1 Why seconds-as-float for `extra.invocation`

ATIF's `extra.invocation` timing fields use seconds-as-float at millisecond precision per Harbor's ATIF RFC convention. The converter rounds at construction (`round(value, 3)`) to avoid IEEE 754 floating-point drift accumulating across long traces.

`step.timestamp` itself remains a string (ISO 8601) per the ATIF schema; only the `extra.invocation` rows use the seconds-as-float form.

## 7. Ancestry Reconstruction

Every ATIF step gets an `extra.ancestry` block built from the source event's identity:

```python
{
  "function_id":   event.uuid,
  "function_name": event.name,
  "parent_id":     event.parent_uuid or "",
  "parent_name":   name_map.get(event.parent_uuid or "", "unknown"),
}
```

The pre-pass walks the full event list once to build `name_map: dict[uuid, name]` so parent names are available regardless of event ordering. Following `parent_id` links upward through the steps reconstructs the full call graph.

### 7.1 Tool ancestry attachment

Tool ScopeEnd events do NOT directly produce ATIF steps (they merge into the system observation). Their ancestry is attached to the **preceding agent step** under `extra.tool_ancestry[]` at the next flush. Each tool's ancestry row also carries an `_sort_id` (the tool's `tool_call_id`) used to sort the array in declaration order matching `last_tool_call_order`.

## 8. Limitations

### 8.1 Repeated tool names within a single turn (WR-03)

See §5.2. The name-keyed `last_tool_call_map` collides on duplicate `function_name` within a single LLM turn for opaque/vendor profiles. Mitigations, in order of preference:

1. **Use `default/tool.v1` profile with explicit `tool_call_id`** — primary v0.2 path; not affected.
2. **Avoid repeated tool names per turn** — emit distinct names (e.g., `calculator__add__1`, `calculator__add__2`) when calling the same tool multiple times.
3. **Future: FIFO queue keyed on `function_name`** — preserve correlation by call order when explicit IDs are unavailable. Not currently implemented; flag for a future revision if opaque-profile multi-call-same-tool scenarios become common.

### 8.2 OpenAI-style `choices` extraction

The reference `_extract_tool_calls()` reads `llm_output["tool_calls"]` directly. LLM outputs that wrap tool calls under `choices[0].message.tool_calls` (OpenAI Chat Completions shape) currently fall through with empty `tool_calls`. Tracked as CR-01 in `08-VERIFICATION.md`. Producers SHOULD pre-flatten OpenAI-style outputs before emitting `ScopeEndEvent.output`, OR consumers SHOULD enhance `_extract_tool_calls()` to inspect `choices[0].message`.

### 8.3 Naive RFC 3339 timestamps

`datetime.fromisoformat()` accepts naive ISO 8601 strings (no timezone), which the wire-format spec §7.1 forbids ("MUST end with `Z` or an explicit UTC offset"). Naive strings reinterpret in the consumer's local timezone and silently shift `ts_micros` by hours when CI runs in UTC and laptops don't. Tracked as HI-01. Future strict-parser flag in the converter is the planned mitigation.

### 8.4 `MarkEvent` without `data`

A `MarkEvent` whose `data` field is `null` is currently SKIPPED (no step emitted). Marks function as logical checkpoints in the wire stream; null-data marks carry no ATIF-meaningful payload. If you need a marker step with empty content, emit `data == {}` instead — the converter will produce a `system` step with `message == "{}"`.

## 9. StreamHeaderEvent Handling

`StreamHeaderEvent`s are consumed by a pre-pass that builds:

- `effective_mode: str` — the latest `profile_mode_default` value (later headers supersede; default `"opaque"` if no header present)
- `schema_registry: dict[str, dict]` — merged registry of all `schemas` entries across all headers (later schema definitions for the same `$schema` ID supersede earlier ones, per ATOF spec D-08)

The current converter does NOT validate profiles against the registry — the registry is built but unused. This is a forward-compatibility hook: a future enhancement can validate each profile against its declared `$schema` without requiring another pre-pass.

After the pre-pass, the main dispatch loop SKIPS `StreamHeaderEvent`s — they never produce ATIF steps.

## 10. Public API

Two entry points in `src/nat/atof/scripts/atof_to_atif_converter.py`:

```python
def convert(events: list[Event]) -> Trajectory:
    """In-memory: typed ATOF events → validated ATIF Trajectory."""

def convert_file(input_path: str | Path, output_path: str | Path | None = None) -> Trajectory:
    """File-based: read .jsonl → convert → optionally write ATIF JSON."""
```

Both return a Pydantic-validated `nat.atif.trajectory.Trajectory` (the NAT-side ATIF model that mirrors Harbor's `0001-trajectory-format.md` RFC v1.7).

## 11. Reference Examples

The `examples/atof_to_atif/` directory contains three end-to-end scenarios demonstrating the conversion:

| Example   | Mode     | Events | Pattern                                              |
| --------- | -------- | ------ | ---------------------------------------------------- |
| EXMP-01   | header   | 9      | Single tool call (calculator)                        |
| EXMP-02   | inline   | 13     | Nested tool chain (weather lookup, 2-level nesting)  |
| EXMP-03   | mixed    | 15     | Branching parallel tools + per-event mode override   |

Each example produces a 5-step ATIF trajectory: `user` (LLM input) → `agent` (LLM output with tool calls) → `system` (merged tool observations) → `user` (next LLM input) → `agent` (final response).

Run end-to-end:

```bash
python examples/atof_to_atif/generate_examples.py    # writes 3 .jsonl streams
python examples/atof_to_atif/convert_to_atif.py      # converts each to ATIF JSON
```

## 12. Versioning

This document tracks the ATOF spec version. Conversion behavior MAY evolve within a `MAJOR.MINOR` ATOF version — for example, fixing CR-01 (OpenAI choices unwrap) or HI-01 (naive timestamp guard) does not require a version bump because the public API and event contract are unchanged. Behavior changes that alter the ATIF output for a given input stream (e.g., changing `source` mapping conventions) DO require a documented version bump and migration note.

---

*Last updated: 2026-04-14 alongside ATOF spec v0.2.*
