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

# ATOF-to-ATIF Examples

End-to-end examples exercising the ATOF v0.1 reference implementation. EXMP-01 is tier-1 (raw pass-through), EXMP-02 is tier-2 (semantic-tagged), and EXMP-03 demonstrates `mark` events. See spec §1.1 in [`../../atof-event-format.md`](../../atof-event-format.md) for tier definitions and §3 for event kinds.

This README doubles as the ATOF → ATIF conversion reference: the mapping table, dispatch conventions, and known limitations live in the [Conversion reference](#conversion-reference) section at the bottom.

## Scripts

- `generate_atof_examples.py` — produces `output/exmpNN_atof.jsonl` for each scenario using the v0.1 public API (`scope` / `mark` event models, `write_jsonl`).
- `convert_atof_examples_to_atif.py` — reads each regenerated JSONL, runs the ATOF→ATIF converter (`nat.atof.scripts.atof_to_atif_converter.convert_file`), and writes `output/exmpNN_atif.json` as a formatted ATIF `Trajectory`.

## The scenarios

### EXMP-01 — tier-1 raw pass-through

A calculator-shaped workflow where the producer can't classify any scope. Every `scope` event carries `category: "unknown"`, `category_profile: null`, and opaque raw JSON in `data`. Demonstrates the floor: a valid ATOF stream capturing only timing + raw payloads, with no semantic tagging.

Converts to a 4-step ATIF trajectory of opaque `system` steps via the reference converter's generic scope-end fall-through. `Trajectory.agent.name` uses the outermost root scope's `name` since no `category: "agent"` event is present.

**When to use:** a runtime wrapping a third-party framework whose callback fires a raw blob the wrapper can't classify.

### EXMP-02 — tier-2 semantic-tagged

Same calculator workflow as EXMP-01 but with every scope classified (`category: "agent"` / `"llm"` / `"tool"`) and `category_profile` populated (`category_profile.model_name` for llm events, `category_profile.tool_call_id` for tool events — see spec §4.4). Additionally demonstrates `attributes: ["remote"]` on the tool scope (the tool is dispatched out-of-process, spec §2.1) and `data_schema` on the llm scopes pointing at `openai/chat-completions.v1` (spec §2).

Converts to a 5-step rich ATIF trajectory (user → agent → system → user → agent) with `Trajectory.agent.name` derived from the `category: "agent"` scope's `name`.

**When to use:** native producers that classify events at the hook site.

### EXMP-03 — mark events

A short chat agent bracketed by two `mark` events — a `session_start` mark before the agent opens and a `session_end` mark after it closes. Both marks are generic checkpoints (`category` absent, `category_profile` absent), carrying `data` that records session-level metadata (session/user IDs, message count).

Converts to a 4-step ATIF trajectory (system → user → agent → system): each mark with non-null `data` materialises as a `source: "system"` step whose `message` is the serialized `data`; the single LLM turn produces the user/agent pair.

**When to use:** demonstrating the `mark` event kind — point-in-time checkpoints that sit outside the start/end scope-pairing semantics.

## Running

```bash
cd NeMo-Agent-Toolkit/packages/nvidia_nat_atif/examples/atof_to_atif
python generate_atof_examples.py
python convert_atof_examples_to_atif.py
# Outputs in output/
```

## Event counts

| Scenario | Events | ATIF steps | Tier | Workflow                                         |
| -------- | ------ | ---------- | ---- | ------------------------------------------------ |
| EXMP-01  | 8      | 4          | 1    | Opaque wrapper: 3 unclassified inner callbacks   |
| EXMP-02  | 8      | 5          | 2    | Calculator: agent → llm → tool → llm → agent     |
| EXMP-03  | 6      | 4          | 2    | Chat agent bracketed by session-boundary marks   |

EXMP-01 and EXMP-02 each consist of 4 paired `scope` events (4 start + 4 end). EXMP-03 consists of 2 paired `scope` events plus 2 `mark` events. The ATOF v0.1 spec has no stream-level metadata event.

---

## Conversion reference

This section is the canonical mapping from ATOF event streams to ATIF trajectories. The reference implementation lives at [`../../src/nat/atof/scripts/atof_to_atif_converter.py`](../../src/nat/atof/scripts/atof_to_atif_converter.py); the code is the source of truth for edge cases. This section documents the conventions any consumer should follow to round-trip cleanly.

### Source mapping

ATIF requires every `Step` to declare a `source ∈ {"user", "agent", "system"}`. ATOF events carry no `source` field — the converter derives it from the event's `kind`, `scope_category`, and `category`:


| ATOF event                           | Condition                                              | ATIF `source` | Step content                                                                                                           |
| ------------------------------------ | ------------------------------------------------------ | ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `scope`, `scope_category: "start"`   | `category == "llm"`                                    | `user`        | `message` = serialized messages array from `event.data`                                                                |
| `scope`, `scope_category: "end"`     | `category == "llm"`                                    | `agent`       | `message` = LLM response content; `tool_calls` extracted from `event.data`                                             |
| `scope`, `scope_category: "end"`     | `category == "tool"`                                   | `system`      | merged into `observation.results[]`; consecutive tool ends flush as a single step                                      |
| `mark`                               | `data != null`                                         | `system`      | `message` = serialized `data` (null-data marks are skipped)                                                            |
| `scope`, `scope_category: "start"`   | `category == "agent"`                                  | (none)        | call-graph shaping only — `name` captured for `Trajectory.agent.name`                                                  |
| `scope`, `scope_category: "end"`     | `category ∉ {"llm", "tool", "agent"}`                  | `system`      | `message` = serialized `event.data`; ancestry + invocation timing preserved. Covers tier-1 opaque and unclassified categories. |
| `scope`, `scope_category: "start"`   | `category ∉ {"llm", "agent"}`                          | (none)        | call-graph shaping only — included in `extra.ancestry` chains                                                          |


**Tier-1 pass-through guarantee.** A strict tier-1 stream — every scope with `category == "unknown"` and `category_profile: null` — converts to a non-empty trajectory: each opaque `scope_category: "end"` event becomes a `source: "system"` step whose `message` is the serialized raw `event.data`. `Trajectory.agent.name` falls back to the outermost (root) start event's `name` when no `category == "agent"` event is present.

### Why `(kind, scope_category, category)` as dispatch key

The three string literals `"llm"`, `"tool"`, `"agent"` are the `category` values the reference converter recognizes for **specialised** ATIF-step materialisation:

- **`llm`** scopes become paired user/agent steps with messages and tool-call extraction.
- **`tool`** scopes become merged observation results buffered between LLM turns.
- **`agent`** scopes populate `Trajectory.agent.name` only (no step emitted).

All **other** scope-end events (`function`, `retriever`, `embedder`, `reranker`, `guardrail`, `evaluator`, `custom`, `unknown`) fall into the generic opaque-system-step arm — each contributes a `source: "system"` step whose `message` is the serialised raw `event.data`. This guarantees that **every tier produces a non-empty ATIF trajectory**: tier-1 streams yield a sequence of opaque system steps; tier-2 streams enrich that structure with user/agent/observation steps where scopes are classified.

### Tool-result merging

Consecutive `tool` scope-end events between two LLM turns produce **one** ATIF system step with multiple `observation.results[]`, not one step per tool result. Per Harbor's ATIF RFC, observations belong to the agent turn that produced them — a single `system` step with N results models "the system returning the results of the N tools the agent just called." Per-tool steps would inflate `step_id` counts and confuse downstream metrics.

A flush happens at three triggers:

1. **Next LLM turn begins** (`scope` event with `scope_category: "start"` and `category == "llm"`) — flushes pending observations into a single `system` step before the new `user` step is appended.
2. **`mark` event with non-null `data`** — flushes pending observations before the mark's `system` step.
3. **End of stream** — flushes any remaining observations.

### ID mappings

| ATOF field                                                    | ATIF field                              | Mapping rule                                                             |
| ------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------ |
| `event.uuid`                                                  | `extra.ancestry.function_id`            | Direct                                                                   |
| `event.parent_uuid`                                           | `extra.ancestry.parent_id`              | Direct (empty string if `null`)                                          |
| `event.name`                                                  | `extra.ancestry.function_name`          | Direct                                                                   |
| `name_map[parent_uuid]`                                       | `extra.ancestry.parent_name`            | Looked up via pre-pass `uuid → name` map; `"unknown"` if unresolved      |
| `event.category_profile.tool_call_id` (`category == "tool"`)  | `tool_calls[*].tool_call_id`            | Read from the `category_profile` sub-object (spec §4.4)                  |
| `event.category_profile.tool_call_id` (`category == "tool"`)  | `observation.results[*].source_call_id` | Same value                                                               |
| `event.category_profile.model_name` (`category == "llm"`)     | `Trajectory.agent.model_name`           | First LLM scope-end's `category_profile.model_name` wins                 |
| `scope` start event `name` (`category == "agent"`)            | `Trajectory.agent.name`                 | First `agent` scope wins; falls back to root scope start's `name` if absent |

### Producers that need different mappings

If your runtime emits a `category` not in the specialised list (`llm`/`tool`/`agent`) and you want richer ATIF output than the generic system-step fallback provides, you have three options:

1. **Wrap a known category** — emit your custom scope as `category == "llm"` or `category == "tool"` and use `attributes` + `data` fields to carry the distinguishing semantics.
2. **Implement a custom converter** — fork the reference converter's dispatch loop and add your `category` arm.
3. **Use a `mark` event with structured `data`** — for non-lifecycle observations, a `mark` with non-null `data` produces a `system` step with the data serialized into `message`.

Option 3 is the fastest path for one-off events. Option 2 is correct when your category has lifecycle semantics (paired start + end).

### Known limitations

- **Tools without `category_profile.tool_call_id`.** Tool events emitted without a `category_profile.tool_call_id` (tier-1 producers that don't have provider-assigned correlation IDs) produce `observation.results[*].source_call_id == None`. The call graph is still reconstructable via `parent_uuid` / `extra.ancestry`, but invocation-level correlation is lost.
- **Naive RFC 3339 timestamps.** `datetime.fromisoformat()` accepts naive ISO 8601 strings (no timezone), which spec §5.1 forbids. Naive strings reinterpret in the consumer's local timezone and can silently shift `ts_micros` by hours between environments. Producers MUST emit `Z` or an explicit UTC offset.
- **Null-data marks.** A `mark` event whose `data` field is `null` is skipped — no step is emitted. If you need a marker step with empty content, emit `data: {}` instead — the converter produces a `system` step with `message == "{}"`.

### Public API

Two entry points in [`../../src/nat/atof/scripts/atof_to_atif_converter.py`](../../src/nat/atof/scripts/atof_to_atif_converter.py):

```python
def convert(events: list[Event]) -> Trajectory:
    """In-memory: typed ATOF events → validated ATIF Trajectory."""


def convert_file(input_path: str | Path, output_path: str | Path | None = None) -> Trajectory:
    """File-based: read .jsonl → convert → optionally write ATIF JSON."""
```

Both return a Pydantic-validated `nat.atif.trajectory.Trajectory` (the NAT-side ATIF model that mirrors Harbor's `0001-trajectory-format.md` RFC v1.7).

---

## See also

- [`../../atof-event-format.md`](../../atof-event-format.md) — canonical ATOF v0.1 spec (wire format, categories, event kinds)
- [`../../src/nat/atof/scripts/atof_to_atif_converter.py`](../../src/nat/atof/scripts/atof_to_atif_converter.py) — reference converter implementation
- [`../../tests/test_tier1_conversion.py`](../../tests/test_tier1_conversion.py) — unit tests for tier-1 conversion behaviour
