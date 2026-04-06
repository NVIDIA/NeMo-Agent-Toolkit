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

# ATIF `Step.extra` Contract

## Purpose

This document defines the canonical contract for NVIDIA NeMo Agent Toolkit metadata in
ATIF `Step.extra`.

This is intended to be a shareable reference for:

1. Publishers (for example, runtime trajectory exporters)
2. Consumers (for example, evaluator and profiler maintainers)
3. Upstream ATIF schema reviewers

## Data Model

The data model is defined in
`packages/nvidia_nat_atif/src/nat/atif/atif_step_extra.py`.

The converter emits this model into ATIF `Step.extra`, and downstream tools
consume it for lineage reconstruction, invocation identity mapping, and timing.

### `AtifAncestry`

`AtifAncestry` represents one callable lineage node.

- `function_id: str` (required)
  - Unique identifier for the callable node.
- `function_name: str` (required)
  - Callable name.
- `parent_id: str | None` (optional)
  - Optional parent callable identifier.
- `parent_name: str | None` (optional)
  - Optional parent callable name.

### `AtifInvocationInfo`

`AtifInvocationInfo` represents one invocation occurrence with timing metadata.

- `start_timestamp: float | None` (optional)
  - Invocation start timestamp in epoch seconds when timed.
- `end_timestamp: float | None` (optional)
  - Invocation end timestamp in epoch seconds when timed.
- `invocation_id: str | None` (optional)
  - Optional stable invocation identifier for correlation.
- `status: str | None` (optional)
  - Optional terminal status for the invocation.
- `framework: str | None` (optional)
  - Runtime or framework label (for example, `langchain`).

### `AtifStepExtra`

`AtifStepExtra` is the lineage payload inside `Step.extra`.

- `ancestry: AtifAncestry` (required, canonical)
  - Step-level lineage context.
- `invocation: AtifInvocationInfo | None` (canonical)
  - Step-level invocation timing metadata.
- `tool_ancestry: list[AtifAncestry]` (canonical)
  - Per-invocation lineage entries aligned by index with `tool_calls`.
- `tool_invocations: list[AtifInvocationInfo] | None` (canonical)
  - Per-tool invocation timing entries aligned by index with `tool_calls`.

`AtifStepExtra` uses `extra="allow"` so new metadata can coexist without
breaking readers.

Extension fields should be additive and optional.
Publishers should avoid redefining canonical keys
(`ancestry`, `invocation`, `tool_ancestry`, `tool_invocations`).
Consumer implementations should ignore unknown extension keys by default.

## Canonical Contract

Canonical lineage and invocation context should be reconstructed from:

- `tool_calls` (all observed invocation occurrences)
- `extra.ancestry`
- `extra.tool_ancestry`
- `extra.invocation` and `extra.tool_invocations` for timing metadata

Identity semantics are split:

- call instance identity: `tool_call_id` (invocation occurrence)
- callable node identity: `function_id` (function/workflow lineage node)

## Core Invariants

- `len(tool_ancestry)` matches `len(tool_calls)` for emitted invocation lineage.
- Index alignment is stable (`tool_calls[i]` maps to `tool_ancestry[i]`).
- If `tool_invocations` is emitted, `len(tool_invocations)` equals `len(tool_calls)`.
- If one of `start_timestamp` / `end_timestamp` is set, the other must also be set.
- When observation results are present for a call occurrence, linkage is stable:
  the matching observation `source_call_id` equals `tool_calls[i].tool_call_id`.

## Producer Requirements

Producers should satisfy these requirements:

1. `tool_calls` is a flat list that includes all observed tool and function
   invocations, not only top-level model-selected calls.
2. `tool_ancestry[i]` corresponds to `tool_calls[i]`.
3. `tool_invocations[i]` corresponds to `tool_calls[i]` when timing metadata is emitted.
4. Invocation order is deterministic and based on start time.
   When start timestamps are equal, producers should preserve stable source event order.
5. Repeated calls to the same function are emitted as distinct invocation
   entries.
6. `tool_calls` should include only callable execution invocation occurrences.
   Non-execution lifecycle or wrapper records should not be emitted as `tool_calls`.

## Consumer Requirements

Consumers should implement lineage reads in this order:

1. Use `tool_calls` + `tool_ancestry` as the primary source.
2. Use `invocation` and `tool_invocations` for timing.
3. Use `ancestry` for step-level lineage context.
4. Tolerate missing observation rows and treat absent observation output as unavailable, not linkage failure.

### Evaluator guidance for nested trajectories

Evaluators parsing nested trajectories should:

- Treat each `tool_calls[i]` as one invocation occurrence and resolve lineage from
  `tool_ancestry[i]`.
- Preserve separation between invocation identity (`tool_call_id`) and callable
  identity (`function_id`) during scoring.
- Use observation linkage (`source_call_id`) to associate tool outputs with the
  correct invocation instance.
- Apply only structural normalization in consumer parsing (for example, removing
  structurally empty rows or adjacent exact duplicates) and avoid semantic
  rewrites of invocation order or hierarchy.

### Observability consumer requirements

Observability consumers should reconstruct execution as per-invocation records with:

- `step_index`
- `tool_call_id`
- `function_id` / `function_name`
- `parent_id` / `parent_name`
- `source_call_id`
- `start_timestamp` / `end_timestamp` when timing is available

Identity and lineage interpretation should follow:

- Call instance identity: `tool_calls[i].tool_call_id`
- Callable node identity: `tool_ancestry[i].function_id`
- Parent-child lineage: `parent_id` / `parent_name`
- Observation linkage: `observation.results[*].source_call_id` equals `tool_call_id`

Interpretation notes:

- Invocation occurrences without timestamps are valid when timestamps are unavailable.
- Invocation identity and callable identity are separate (`tool_call_id` vs `function_id`).
  For example, two calls to `calculator__multiply` may have
  `tool_call_id=call_abc` and `tool_call_id=call_def`, while both map to the
  same callable node `function_id=fn_multiply`.
- Consumer output may group by ATIF step for readability, while trace dashboards may center on span timelines.

## Reference Artifacts

Use the following config-specific artifact pairs as reference visual output
(Phoenix dashboard screenshot + generated `workflow_output_atif.json`):
The sequence is intentionally ordered from simpler to progressively richer nested trajectories.

1. One-level tool calls
   - config:
     `examples/evaluation_and_profiling/simple_calculator_eval/configs/config-trajectory-eval.yml`
   - Phoenix PNG:
     `docs/source/_static/simple_calculator_trajectory_phoenix_trace.png`
   - ATIF output:
     `examples/evaluation_and_profiling/simple_calculator_eval/src/nat_simple_calculator_eval/data/output_samples/trajectory_eval/workflow_output_atif.json`

2. Nested tool call chain
   - config:
     `examples/evaluation_and_profiling/simple_calculator_eval/configs/config-nested-trajectory-eval.yml`
   - Phoenix PNG:
     `docs/source/_static/simple_calculator_nested_phoenix_trace.png`
   - ATIF output:
     `examples/evaluation_and_profiling/simple_calculator_eval/src/nat_simple_calculator_eval/data/output_samples/nested_trajectory_eval/workflow_output_atif.json`

3. Branching nested tool calls
   - config:
     `examples/evaluation_and_profiling/simple_calculator_eval/configs/config-branching-nested-trajectory-eval.yml`
   - Phoenix PNG:
     `docs/source/_static/simple_calculator_branching_phoenix_trace.png`
   - ATIF output:
     `examples/evaluation_and_profiling/simple_calculator_eval/src/nat_simple_calculator_eval/data/output_samples/branching_nested_trajectory_eval/workflow_output_atif.json`

