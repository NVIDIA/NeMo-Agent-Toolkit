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

# ATIF `Step.extra` and `ToolCall.extra` Contract

## Purpose

This document defines the canonical contract for NVIDIA NeMo Agent Toolkit
metadata embedded inside ATIF `Step.extra` and `ToolCall.extra`.

It is intended as a shareable reference for:

1. Publishers (for example, runtime trajectory exporters)
2. Consumers (for example, evaluator and profiler maintainers)
3. Upstream ATIF schema reviewers

## What changed in ATIF v1.7

ATIF v1.7 deliberately leaves ancestry out of the typed schema. The previous
Toolkit-internal layout exposed `Step.function_ancestry` and `ToolCall.tool_ancestry`
as typed top-level fields; those fields are no longer part of the ATIF schema.
Per-record `extra` objects are the spec-blessed location for producer-specific
metadata, and the v1.7 spec adds an `extra` field to `ToolCall` specifically
to make per-tool-call metadata first-class.

The Toolkit convention places its lineage and timing payloads inside `extra` dictionaries:

- **`Step.extra["ancestry"]`** — step-level callable lineage (formerly
  `Step.function_ancestry`).
- **`Step.extra["invocation"]`** — step-level invocation timing.
- **`ToolCall.extra["ancestry"]`** — per-tool callable lineage (formerly
  `Step.extra.tool_ancestry[i]`, an aligned-by-index list; now co-located
  with the `tool_call`).
- **`ToolCall.extra["invocation"]`** — per-tool invocation timing
  (formerly `Step.extra.tool_invocations[i]`).

The aligned-by-index `tool_ancestry` and `tool_invocations` lists on
`Step.extra` are removed. Per-tool data lives next to the `tool_call` it
describes.

## Data Model

The data model is defined in
`packages/nvidia_nat_atif/src/nat/atif/atif_step_extra.py`.

### `AtifAncestry`

`AtifAncestry` represents one callable lineage node and is used in both
`Step.extra["ancestry"]` and `ToolCall.extra["ancestry"]`.

- `function_id: str` (required)
  - Unique identifier for the callable node, stable across invocations.
- `function_name: str` (required)
  - Human-readable callable name.
- `parent_id: str | None` (optional)
  - Parent callable identifier; null at the root.
- `parent_name: str | None` (optional)
  - Parent callable name; null when `parent_id` is null.

The model enforces a parent-pair invariant: `parent_name` MAY only be set
when `parent_id` is also set.

### `AtifInvocationInfo`

`AtifInvocationInfo` represents one invocation occurrence with timing
metadata. Used in both `Step.extra["invocation"]` and
`ToolCall.extra["invocation"]`.

- `start_timestamp: float | None` (optional)
  - Invocation start timestamp in epoch seconds.
- `end_timestamp: float | None` (optional)
  - Invocation end timestamp in epoch seconds.
- `invocation_id: str | None` (optional)
  - Stable invocation identifier for correlation. For tool invocations
    the Toolkit sets this equal to `tool_call_id`.
- `status: str | None` (optional)
  - Terminal status (for example, `completed`, `error`).
- `framework: str | None` (optional)
  - Runtime or framework label (for example, `langchain`).

If one of `start_timestamp` / `end_timestamp` is set, the other MUST also
be set (validated).

### `AtifStepExtra`

`AtifStepExtra` is the validated structure for Toolkit-owned `Step.extra`
content.

- `ancestry: AtifAncestry` (required)
  - Step-level callable lineage — which callable produced this step.
- `invocation: AtifInvocationInfo | None` (optional)
  - Step-level invocation timing.

`AtifStepExtra` uses `extra="allow"` so additional keys (for example,
producer-supplied `data_schema`) may coexist without breaking readers.

### `AtifToolCallExtra`

`AtifToolCallExtra` is the validated structure for Toolkit-owned
`ToolCall.extra` content.

- `ancestry: AtifAncestry | None` (optional)
  - Per-tool-call callable lineage — which callable issued this tool
    invocation.
- `invocation: AtifInvocationInfo | None` (optional)
  - Per-tool-call invocation timing.

Both fields are optional. `AtifToolCallExtra` uses `extra="allow"` so a
`ToolCall.extra` lacking either key still validates and additional keys
may coexist.

## Canonical Contract

Canonical lineage and timing context should be reconstructed from:

- `tool_calls` (all observed invocation occurrences)
- For step-level lineage: `step.extra.ancestry`
- For per-tool lineage: `tool_calls[i].extra.ancestry`
- For step-level timing: `step.extra.invocation`
- For per-tool timing: `tool_calls[i].extra.invocation`
- `observation.results[i]` (each linked to a `tool_call` by
  `source_call_id` matching the corresponding `tool_call_id`)

Identity semantics are split:

- Call instance identity: `tool_call_id` (invocation occurrence)
- Callable node identity: `function_id` (function/workflow lineage node)

## Core Invariants

- For each `tool_calls[i]`, when ancestry is emitted, it lives at
  `tool_calls[i].extra.ancestry`. Ancestry is per-record, not aligned by
  position.
- For each `tool_calls[i]`, when timing is emitted, it lives at
  `tool_calls[i].extra.invocation`.
- If `start_timestamp` / `end_timestamp` is set on an invocation, both
  MUST be set.
- Observation linkage is stable: each `observation.results[*].source_call_id`
  matches a `tool_call_id` in the parent step's `tool_calls`.
- `parent_name` MAY only be set when `parent_id` is also set on any
  `AtifAncestry`-shaped dict (validated by `AtifAncestry`).

The pre-v1.7 invariants `len(tool_ancestry) == len(tool_calls)` and
`len(tool_invocations) == len(tool_calls)` no longer apply — there is no
aligned-by-index list to validate. Per-tool metadata is keyed by record,
not by position.

## Producer Requirements

Producers should satisfy these requirements:

1. `tool_calls` is a flat list that includes all observed tool and
   function invocations, not only top-level model-selected calls.
2. When emitting per-tool ancestry, write it to
   `tool_calls[i].extra["ancestry"]` (`AtifAncestry` shape). Do NOT use
   the deprecated `Step.extra.tool_ancestry[i]` aligned-by-index list.
3. When emitting per-tool timing, write it to
   `tool_calls[i].extra["invocation"]` (`AtifInvocationInfo` shape).
   Do NOT use the deprecated `Step.extra.tool_invocations[i]` list.
4. Step-level ancestry goes to `step.extra["ancestry"]`. Step-level
   timing goes to `step.extra["invocation"]`.
5. Invocation order is deterministic and based on start time. When
   start timestamps are equal, producers should preserve stable source
   event order.
6. Repeated calls to the same function are emitted as distinct
   invocation entries with distinct `tool_call_id`s.
7. `tool_calls` should include only callable execution invocation
   occurrences. Non-execution lifecycle or wrapper records should not be
   emitted as `tool_calls`.

### Publisher ID Guidance

Publishers should maintain two complementary identity layers:

- Callable lineage identity (`function_id`, `parent_id`)
- Invocation instance identity (`tool_call_id`, `source_call_id`,
  `invocation_id`)

Callable lineage identity guidance:

1. `function_id` should identify one callable occurrence in the lineage
   tree.
2. `function_id` should remain stable across events emitted for that
   same callable occurrence.
3. Repeated calls to the same callable should use distinct `function_id`
   values.
4. `parent_id` should reference the parent callable occurrence's
   `function_id`.

Invocation instance identity guidance:

1. `tool_call_id` should be unique per emitted invocation row.
2. For each invocation, `observation.results[*].source_call_id` should
   equal the corresponding `tool_call_id`.
3. For each invocation, `tool_calls[i].extra.invocation.invocation_id`
   should equal `tool_calls[i].tool_call_id` when set.

Deep and parallel chain example:

- Branch A: `agent1 -> fn-a -> fn-b -> fn-c`
- Branch B: `agent1 -> fn-d -> fn-a -> fn-b`

One valid lineage assignment:

- Branch A:
  - `fn-a`: `function_id=A1`, `parent_id=ROOT`
  - `fn-b`: `function_id=B1`, `parent_id=A1`
  - `fn-c`: `function_id=C1`, `parent_id=B1`
- Branch B:
  - `fn-d`: `function_id=D1`, `parent_id=ROOT`
  - `fn-a`: `function_id=A2`, `parent_id=D1`
  - `fn-b`: `function_id=B2`, `parent_id=A2`

In this example, callable names repeat across branches, but callable
occurrence IDs remain distinct and lineage stays unambiguous.

## Consumer Requirements

Consumers should implement lineage reads in this order:

1. Iterate `tool_calls`. For each` tool_call`, read `tool_call.extra.ancestry`
   for callable lineage and `tool_call.extra.invocation` for timing.
   (No alignment by index — each record carries its own metadata.)
2. Read `step.extra.ancestry` for step-level callable context and
   `step.extra.invocation` for step-level timing.
3. Tolerate missing observation rows and treat absent observation output
   as unavailable, not linkage failure.
4. Tolerate missing `extra` keys: `extra` is loosely-typed per ATIF spec
   §3, and consumers MUST treat any subset of Toolkit keys as optional.

### Evaluator guidance for nested trajectories

Evaluators parsing nested trajectories should:

- Treat each `tool_calls[i]` as one invocation occurrence and resolve
  lineage from `tool_calls[i].extra.ancestry`.
- Preserve separation between invocation identity (`tool_call_id`) and
  callable identity (`function_id`) during scoring.
- Use observation linkage (`source_call_id`) to associate tool outputs
  with the correct invocation instance.
- Apply only structural normalization in consumer parsing (for example,
  removing structurally empty rows or adjacent exact duplicates) and
  avoid semantic rewrites of invocation order or hierarchy.

### Observability consumer requirements

Observability consumers should reconstruct execution as per-invocation
records with:

- `step_index`
- `tool_call_id`
- `function_id` / `function_name` (from `tool_call.extra.ancestry`)
- `parent_id` / `parent_name` (from `tool_call.extra.ancestry`)
- `source_call_id` (from `observation.results[*]`)
- `start_timestamp` / `end_timestamp` (from `tool_call.extra.invocation`)
  when timing is available

Identity and lineage interpretation should follow:

- Call instance identity: `tool_calls[i].tool_call_id`
- Callable node identity: `tool_calls[i].extra.ancestry.function_id`
- Parent-child lineage: `tool_calls[i].extra.ancestry.parent_id` /
  `parent_name`
- Observation linkage: `observation.results[*].source_call_id` equals
  `tool_call_id`

Interpretation notes:

- Invocation occurrences without timestamps are valid when timestamps
  are unavailable.
- Invocation identity and callable identity are separate (`tool_call_id`
  vs `function_id`). For example, two calls to `calculator__multiply`
  may have `tool_call_id=call_abc` and `tool_call_id=call_def`, while
  both map to the same callable node `function_id=fn_multiply`.
- Consumer output may group by ATIF step for readability, while trace
  dashboards may center on span timelines.

## Reference Artifacts

Use the following config-specific artifact pairs as reference visual
output (Phoenix dashboard screenshot + generated `workflow_output_atif.json`).
The sequence is intentionally ordered from simpler to progressively
richer nested trajectories.

1. One-level tool calls
   - config:
     `examples/evaluation_and_profiling/simple_calculator_eval/configs/config-trajectory-eval.yml`
   - Phoenix PNG:
     `examples/evaluation_and_profiling/simple_calculator_eval/src/nat_simple_calculator_eval/data/output_samples/trajectory_eval/simple_calculator_trajectory_phoenix_trace.png`
   - ATIF output:
     `examples/evaluation_and_profiling/simple_calculator_eval/data/output_samples/trajectory_eval/workflow_output_atif.json`

2. Nested tool call chain
   - config:
     `examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-trajectory-eval.yml`
   - Phoenix PNG:
     `examples/evaluation_and_profiling/simple_calculator_eval/src/nat_simple_calculator_eval/data/output_samples/nested_trajectory_eval/simple_calculator_nested_phoenix_trace.png`
   - ATIF output:
     `examples/evaluation_and_profiling/simple_calculator_eval/data/output_samples/nested_trajectory_eval/workflow_output_atif.json`

3. Branching nested tool calls
   - config:
     `examples/evaluation_and_profiling/simple_calculator_eval/configs/config-branching-nested-trajectory-eval.yml`
   - Phoenix PNG:
     `examples/evaluation_and_profiling/simple_calculator_eval/src/nat_simple_calculator_eval/data/output_samples/branching_nested_trajectory_eval/simple_calculator_branching_phoenix_trace.png`
   - ATIF output:
     `examples/evaluation_and_profiling/simple_calculator_eval/data/output_samples/branching_nested_trajectory_eval/workflow_output_atif.json`

> **Note:** the reference artifacts above were generated with the
> pre-v1.7 layout (aligned-by-index `tool_ancestry` / `tool_invocations`
> on `Step.extra`). They will be regenerated to the v1.7-aligned layout
> in a follow-up pass.
