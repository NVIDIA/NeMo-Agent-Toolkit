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


# IntermediateStep to ATIF Mapping

This document explains how IntermediateStep event streams are mapped to ATIF
trajectories.

It is intentionally generic and applies to the current conversion path used by
the toolkit. Mappings reflect the **ATIF v1.7** layout where ancestry and
per-invocation timing live inside `extra` rather than as typed top-level
fields. See `atif-step-extra-guide.md` for the full `Step.extra` /
`ToolCall.extra` contract.

## ID Mappings

| IntermediateStep | ATIF | Mapping Rule | Notes |
|---|---|---|---|
| `payload.UUID` | `tool_calls[*].tool_call_id` | `tool_call_id = "call_" + payload.UUID` | Invocation occurrence identity |
| `payload.UUID` | `observation.results[*].source_call_id` | `source_call_id = tool_call_id` | Observation links to invocation |
| `payload.UUID` | `tool_calls[*].extra.invocation.invocation_id` | `invocation_id = tool_call_id` | Per-tool timing row identity |
| `function_ancestry.function_id` (tool context) | `tool_calls[*].extra.ancestry.function_id` | Direct match | Per-tool callable lineage node identity |
| `function_ancestry.parent_id` (tool context) | `tool_calls[*].extra.ancestry.parent_id` | Direct match | Per-tool parent callable identity |
| `function_ancestry.function_id` (step context) | `step.extra.ancestry.function_id` | Direct match | Step-level callable context |
| `function_ancestry.parent_id` (step context) | `step.extra.ancestry.parent_id` | Direct match | Step-level parent callable context |
| Not applicable | `step_id` | Generated ATIF sequence counter | Not derived from IntermediateStep UUID |

## Name Mappings

| IntermediateStep | ATIF | Mapping Rule | Notes |
|---|---|---|---|
| `payload.name` (tool, function, or LLM by event type) | `tool_calls[*].function_name` or `model_name` | Context dependent | IntermediateStep name is polymorphic by event type |
| `function_ancestry.function_name` (tool context) | `tool_calls[*].extra.ancestry.function_name` | Direct match | Per-tool callable lineage name |
| `function_ancestry.parent_name` (tool context) | `tool_calls[*].extra.ancestry.parent_name` | Direct match | Per-tool parent callable name |
| `function_ancestry.function_name` (step context) | `step.extra.ancestry.function_name` | Direct match | Step-level lineage name |
| `function_ancestry.parent_name` (step context) | `step.extra.ancestry.parent_name` | Direct match | Step-level parent name |

## Event-to-Step Mapping

- `WORKFLOW_START` maps to an ATIF user step (`source = "user"`).
- `LLM_END` starts an ATIF agent turn candidate step.
- `TOOL_END` and `FUNCTION_END` are accumulated into the pending ATIF step as
  observed invocations. Their ancestry / timing attaches to the matching
  `tool_call.extra` (not to the parent step's `extra`).
- `WORKFLOW_END` may emit a terminal ATIF agent step when final output is
  present and not redundant.
- `LLM_NEW_TOKEN` and other non-terminal chunk events are not directly emitted
  as standalone ATIF steps.

## Identity Semantics

- Invocation identity and callable identity are intentionally different:
  - Invocation identity: `tool_call_id`, `source_call_id`, `invocation_id`
  - Callable identity: `function_id`, `parent_id`
- Correct lineage interpretation requires both:
  - use invocation IDs for per-call correlation,
  - use callable IDs for hierarchy and repeated-call disambiguation.

## Timing Mappings

- IntermediateStep end events commonly use:
  - `event_timestamp` as end timestamp,
  - `span_event_timestamp` as start timestamp.
- ATIF v1.7 timing is represented per record:
  - **Step-level**: `step.extra.invocation.start_timestamp` and
    `step.extra.invocation.end_timestamp`.
  - **Per-tool-call**: `tool_calls[i].extra.invocation.start_timestamp` and
    `tool_calls[i].extra.invocation.end_timestamp`.

  Per-tool timing was an aligned-by-index list at `step.extra.tool_invocations[i]`
  in the pre-v1.7 layout; v1.7 co-locates each invocation with its tool_call
  via `tool_call.extra.invocation`.

## Practical Validation Checklist

- Verify `tool_call_id == source_call_id == tool_calls[i].extra.invocation.invocation_id`
  for each invocation row.
- Verify `tool_call_id == "call_" + payload.UUID` for mapped tool or function
  end events.
- Verify per-tool callable lineage consistency:
  - `function_ancestry.function_id <-> tool_calls[i].extra.ancestry.function_id`
  - `function_ancestry.parent_id <-> tool_calls[i].extra.ancestry.parent_id`
- Verify per-tool name consistency:
  - `function_ancestry.function_name <-> tool_calls[i].extra.ancestry.function_name`
  - `function_ancestry.parent_name <-> tool_calls[i].extra.ancestry.parent_name`
- Verify step-level lineage consistency:
  - `function_ancestry.function_id <-> step.extra.ancestry.function_id`
  - `function_ancestry.parent_id <-> step.extra.ancestry.parent_id`

## Additional Identifiers Worth Tracking

- IntermediateStep structural parent linkage: `parent_id`
- Event semantics: `payload.event_type`
- Time surfaces: `event_timestamp`, `span_event_timestamp`, ATIF `timestamp`
- Session scope: ATIF `session_id` (run-scoped) and `trajectory_id`
  (per-document, ATIF v1.7+)
- Framework or provider run IDs when present in metadata (for example, model
  framework trace IDs)
