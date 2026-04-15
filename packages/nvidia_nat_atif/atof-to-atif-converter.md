<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
-->

# ATOF → ATIF Converter

**Version:** 0.1
**Date:** 2026-04-15
**Status:** Active
**Companion to:** [`atof-event-format.md`](./atof-event-format.md) (ATOF wire format)

---

## 1. Purpose

This document specifies the canonical mapping from an ATOF event stream to an ATIF (Agent Trajectory Interchange Format) trajectory. It is the **normative reference** for any ATOF consumer that needs to produce ATIF output.

ATOF is a wire format for raw runtime observations (start/end events, marks, and an optional stream header for codec metadata). ATIF is a higher-level structure: an ordered sequence of conversation turns (`source`, `message`, `tool_calls`, `observation`) with computed metadata (`step_id`, ancestry, timing). The converter is the layer that bridges the two.

The reference implementation lives at `src/nat/atof/scripts/atof_to_atif_converter.py`. Non-NAT consumers MAY implement different conversion strategies — this document specifies the convention the reference converter follows so that producers using the documented `scope_type` vocabulary will round-trip cleanly.

## 2. Scope

This document covers:

- The mapping from ATOF event kinds + `scope_type` values to ATIF `step.source` values
- The accumulator state machine that merges multi-event observations into single ATIF steps
- ID and timing field mappings
- Ancestry reconstruction from `parent_uuid` chains
- Handling of `StreamHeaderEvent` and unknown `scope_type` values
- Known limitations and extension points

This document does NOT cover:

- The ATOF wire format itself — see `atof-event-format.md`
- The codec resolution protocol — see `atof-codec-profiles.md` §6
- The IntermediateStep → ATIF pipeline used by NAT's evaluation tooling — see `intermediate-step-to-atif-mapping.md`
- The ATIF schema — see Harbor's `0001-trajectory-format.md` RFC
- ATIF `step.extra` field conventions — see `atif-step-extra-guide.md`

## 3. ATIF Source Mapping

ATIF requires every `Step` to declare a `source ∈ {"user", "agent", "system"}`. ATOF events carry no `source` field — the converter derives it from the event's `kind` and `scope_type`.

### 3.1 Mapping Table

| ATOF event           | Condition                       | ATIF `source` | Step content                                                                        |
| -------------------- | ------------------------------- | ------------- | ----------------------------------------------------------------------------------- |
| `ScopeStartEvent`    | `scope_type == "llm"`           | `user`        | `message` = serialized messages array from `event.input`                            |
| `ScopeEndEvent`      | `scope_type == "llm"`           | `agent`       | `message` = LLM response content; `tool_calls` extracted from `output`              |
| `ScopeEndEvent`      | `scope_type == "tool"`          | `system`      | merged into `observation.results[]`; flushed as one step (see §4)                   |
| `MarkEvent`          | `data != null`                  | `system`      | `message` = serialized `data`                                                       |
| `ScopeStartEvent`    | `scope_type == "agent"`         | (none)        | call-graph shaping only — `name` captured for `Trajectory.agent.name`               |
| `ScopeStartEvent`/`ScopeEndEvent` | any other `scope_type` | (none)        | call-graph shaping only — included in `extra.ancestry` chains                       |
| `StreamHeaderEvent`  | (any)                           | (none)        | optional metadata carrier; never materializes as a step (see §9)                    |

### 3.2 Why `(kind, scope_type)` and not just `kind`

`kind` is a closed set of four values (`ScopeStart`, `ScopeEnd`, `Mark`, `StreamHeader`); `scope_type` is the closed enum from spec §4 (`agent`, `function`, `llm`, `tool`, `retriever`, `embedder`, `reranker`, `guardrail`, `evaluator`, `custom`, `unknown`). The `(kind, scope_type)` pair is the dispatch key. This separation lets producers introduce `custom` + `subtype` vendor scopes (spec §4.2) or fall through with `scope_type: "unknown"` (tier-1 pass-through, spec §4.1) without forking the wire format. The converter SKIPS unrecognized `scope_type` values rather than rejecting them, preserving forward compatibility.

The three string literals `"llm"`, `"tool"`, `"agent"` are **conventions** the reference converter recognizes for materializing ATIF steps. Producers emitting these scope types and following the typed-field conventions (`model_name` on llm, `tool_call_id` on tool) will produce well-formed ATIF.

### 3.3 Producers that need different mappings

If your runtime emits a `scope_type` not in the table above and you want it to materialize as an ATIF step, you have three options:

1. **Wrap a known type** — emit your custom scope as `scope_type == "llm"` or `scope_type == "tool"` and use `attributes` + `data` fields to carry the distinguishing semantics.
2. **Implement a custom converter** — fork the reference converter's dispatch loop and add your `scope_type` arm.
3. **Use `MarkEvent` with structured `data`** — for non-lifecycle observations, a `Mark` produces a `system` step with the data serialized into `message`.

Option 3 is the fastest path for one-off events. Option 2 is correct when your scope type has lifecycle semantics (start + end + status).

## 4. The Accumulator State Machine

The converter is a single-pass accumulator over events sorted by `ts_micros`. It maintains:

- `step_dicts: list[dict]` — the output ATIF steps, built incrementally
- `pending_observations: list[dict]` — tool results buffered between LLM turns
- `pending_obs_timestamp: str | int | None` — timestamp of the first buffered tool result (used as the system step's `timestamp`)
- `last_tool_call_order: list[str]` — declaration order of `tool_call_id` values from the most recent agent step (used for stable observation ordering)
- `current_agent_step_idx: int | None` — index of the most recent agent step (used to attach `tool_ancestry` and `tool_invocations` after all child tool events arrive)
- `pending_tool_ancestry`, `pending_tool_invocations` — buffered ancestry/timing rows attached to the current agent step at flush

### 4.1 Flush Semantics

A flush happens at three triggers:

1. **Next LLM turn begins** (`ScopeStartEvent` with `scope_type == "llm"`) — flushes pending observations into a single `system` step BEFORE the new `user` step is appended; finalizes the current agent step's `tool_ancestry` / `tool_invocations` extras.
2. **MarkEvent with data** — flushes pending observations BEFORE the mark's `system` step.
3. **End of stream** — flushes any remaining observations + finalizes the last agent step.

Consecutive `tool` ScopeEnd events between two LLM turns produce **one** ATIF system step with multiple `observation.results[]`, NOT one step per tool result. Flush order: `flush_observations()` → `finalize_agent_extra()` → append next step.

### 4.2 Why merge tool results

Per Harbor's ATIF RFC, observations belong to the agent turn that produced them — a single `system` step with N results models "the system returning the results of the N tools the agent just called." Per-tool steps would inflate `step_id` counts and confuse downstream metrics (e.g., turn count, tool-call density).

## 5. ID Mappings

| ATOF field                                       | ATIF field                                  | Mapping rule                                                            |
| ------------------------------------------------ | ------------------------------------------- | ----------------------------------------------------------------------- |
| `event.uuid` (any event)                         | `extra.ancestry.function_id`                | Direct                                                                  |
| `event.parent_uuid`                              | `extra.ancestry.parent_id`                  | Direct (empty string if `null`)                                         |
| `event.name`                                     | `extra.ancestry.function_name`              | Direct                                                                  |
| `name_map[parent_uuid]`                          | `extra.ancestry.parent_name`                | Looked up via the pre-pass `uuid → name` map; `"unknown"` if unresolved |
| `ScopeStartEvent.uuid` (`scope_type == "llm"`)   | `extra.tool_invocations[*].invocation_id`   | For tool ScopeEnds whose `parent_uuid` matches                          |
| `event.tool_call_id` (`scope_type == "tool"`)    | `tool_calls[*].tool_call_id`                | Read directly from the typed event field                                |
| `event.tool_call_id` (`scope_type == "tool"`)    | `observation.results[*].source_call_id`     | Same value                                                              |
| `event.model_name` (`scope_type == "llm"`)       | `Trajectory.agent.model_name`               | Read directly from the typed event field                                |
| `(none)`                                         | `step_id`                                   | Generated 1-indexed sequence after all steps are built                  |
| `ScopeStartEvent.name` (`scope_type == "agent"`) | `Trajectory.agent.name`                     | Read from the first `agent` scope start                                 |

### 5.1 Tool call extraction from LLM output

The converter's `_extract_tool_calls()` handles two output shapes:

```python
# 1. Flat shape (NAT-native, simple producers)
output = {"tool_calls": [{"id": "call_abc", "name": "calc__add", "arguments": {"a": 3, "b": 4}}]}

# 2. OpenAI Chat Completions shape (output.choices[0].message.tool_calls)
output = {
    "id": "chatcmpl-...",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "tool_calls": [
            {"id": "call_abc", "type": "function",
             "function": {"name": "calc__add", "arguments": '{"a":3,"b":4}'}}
        ]}, "finish_reason": "tool_calls"}
    ],
}
```

The flat shape is checked first; OpenAI shape is the fallback. Tool calls under `function.name` / `function.arguments` are normalized to the flat `name` / `arguments` shape; string-encoded JSON arguments are parsed (with `{"raw": <string>}` fallback if parsing fails).

### 5.2 Tool correlation

`tool_call_id` is read directly from the typed `tool_call_id` field on `ScopeStartEvent`/`ScopeEndEvent` when `scope_type == "tool"`. The v0.1 wire format makes `tool_call_id` a first-class typed field on tool events, so no name-based fallback or schema-discriminator dispatch is needed.

When a tool event is missing `tool_call_id` entirely (tier-1 producer that doesn't have the correlation ID), the converter still produces an observation but with `source_call_id == None` — the tool result is preserved but not linked to a specific LLM tool-call invocation.

## 6. Timing Mappings

| ATOF field                                | ATIF field                                  | Mapping rule                                                                                            |
| ----------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `event.timestamp` (string OR int)         | `step.timestamp`                            | RFC 3339 string passes through unchanged; integer microseconds serialize to RFC 3339 (spec §6.1)        |
| `event.ts_micros` (computed)              | (sort key only)                             | Used for stable cross-format ordering before any other dispatch                                          |
| `start_ts_map[event.uuid]` (microseconds) | `extra.invocation.start_timestamp`          | Converted to seconds-as-float at millisecond precision (`round(micros / 1_000_000, 3)`)                 |
| `event.ts_micros` (ScopeEnd)              | `extra.invocation.end_timestamp`            | Same conversion                                                                                         |
| Tool ScopeStart `ts_micros`               | `extra.tool_invocations[*].start_timestamp` | Same conversion                                                                                         |
| Tool ScopeEnd `ts_micros`                 | `extra.tool_invocations[*].end_timestamp`   | Same conversion                                                                                         |

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

### 8.1 Tools without `tool_call_id`

Tool events emitted without a `tool_call_id` typed field (tier-1 producers that don't have provider-assigned correlation IDs) produce `observation.results[*].source_call_id == None`. Downstream tools that join observations to specific tool invocations by ID will not be able to correlate these — the call graph is still reconstructable via `parent_uuid` / `extra.ancestry`, but invocation-level correlation is lost.

Mitigation: producers SHOULD populate `tool_call_id` whenever the tool was dispatched via an LLM tool-use flow (the LLM provider's response carries the ID). For tools invoked outside an LLM flow (e.g., scheduled tasks), `tool_call_id` is genuinely absent and the limitation is intrinsic.

### 8.2 Naive RFC 3339 timestamps (HI-01)

`datetime.fromisoformat()` accepts naive ISO 8601 strings (no timezone), which the wire-format spec §6.1 forbids ("MUST end with `Z` or an explicit UTC offset"). Naive strings reinterpret in the consumer's local timezone and silently shift `ts_micros` by hours when CI runs in UTC and laptops don't. A future strict-parser flag in the converter is the planned mitigation.

### 8.3 `MarkEvent` without `data`

A `MarkEvent` whose `data` field is `null` is currently SKIPPED (no step emitted). Marks function as logical checkpoints in the wire stream; null-data marks carry no ATIF-meaningful payload. If you need a marker step with empty content, emit `data == {}` instead — the converter will produce a `system` step with `message == "{}"`.

## 9. StreamHeaderEvent Handling

The reference converter SKIPS `StreamHeaderEvent`s in the main dispatch loop — they never produce ATIF steps. Their `name` is added to the pre-pass `name_map` (harmless), but no other processing occurs.

`StreamHeaderEvent` is part of the optional **codec resolution layer** (companion doc `atof-codec-profiles.md` §6) — a separate concern from ATIF conversion. The reference converter does not consult the StreamHeader's `codecs` registry because it does not perform codec-driven validation. A future enhancement could:

- Resolve each LLM event's codec via the 4-priority chain (per-event inline → header registry → consumer-bundled → opaque).
- Use the resolved schema to validate `annotated_request` / `annotated_response` payloads at conversion time, surfacing schema violations as warnings.
- Use `annotated_response` (when present) as a richer source for `tool_calls` extraction than `output` directly.

These are forward-compatible enhancements that don't change the v0.1 conversion contract.

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

| Example   | Tier | Events | ATIF steps | Pattern                                                      |
| --------- | ---- | ------ | ---------- | ------------------------------------------------------------ |
| EXMP-01   | 2    | 9      | 5          | Calculator: agent → llm → tool → llm → agent                 |
| EXMP-02   | 2    | 7      | 3          | Search: agent → llm → tool (timeout) → agent (recovery)      |
| EXMP-03   | 3    | 9      | 5          | Calculator with `openai/chat-completions.v1` codec annotations |

Each example opens with a `StreamHeaderEvent` at position 0 (spec §3.4 — MUST be first when present).

EXMP-01 produces a 5-step trajectory: `user` (LLM input) → `agent` (LLM output with tool calls) → `system` (merged tool observations) → `user` (next LLM input) → `agent` (final response).

EXMP-02 demonstrates the cascading-status semantics (spec §5.2-5.3) — tool reports `status: "error"`; parent agent catches and reports `status: "ok"`. Trajectory is shorter (3 steps) because the LLM never gets to formulate a final answer with tool results.

EXMP-03 shows the same workflow as EXMP-01 but with `codec` declarations and `annotated_request` / `annotated_response` payloads on every LLM event. The converter's `_extract_tool_calls()` handles the OpenAI Chat Completions output shape (§5.1), so EXMP-03 still converts to a clean 5-step trajectory.

Run end-to-end:

```bash
python examples/atof_to_atif/generate_examples.py    # writes 3 .jsonl streams
python examples/atof_to_atif/convert_to_atif.py      # converts each to ATIF JSON
```

## 12. Versioning

This document tracks the ATOF spec version. Conversion behavior MAY evolve within a `MAJOR.MINOR` ATOF version — for example, adding a strict-timestamp-parser option (HI-01 mitigation) does not require a version bump because the public API and event contract are unchanged. Behavior changes that alter the ATIF output for a given input stream (e.g., changing `source` mapping conventions) DO require a documented version bump and migration note.

---

*Last updated: 2026-04-15 alongside ATOF spec v0.1.*
