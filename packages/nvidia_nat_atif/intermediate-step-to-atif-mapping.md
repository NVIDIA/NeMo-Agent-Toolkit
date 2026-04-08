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

This document explains how IntermediateStep event streams are mapped to ATIF trajectories.

It is intentionally generic and applies to the current conversion path used by the toolkit.

## ID Mappings

| IntermediateStep | ATIF | Mapping Rule | Notes |
|---|---|---|---|
| `payload.UUID` | `tool_calls[*].tool_call_id` | `tool_call_id = "call_" + payload.UUID` | Invocation occurrence identity |
| `payload.UUID` | `observation.results[*].source_call_id` | `source_call_id = tool_call_id` | Observation links to invocation |
| `payload.UUID` | `extra.tool_invocations[*].invocation_id` | `invocation_id = tool_call_id` | Timing row identity for same invocation |
| `function_ancestry.function_id` | `extra.tool_ancestry[*].function_id` | Direct match | Callable lineage node identity |
| `function_ancestry.parent_id` | `extra.tool_ancestry[*].parent_id` | Direct match | Parent callable lineage node identity |
| `function_ancestry.function_id` (step context) | `extra.ancestry.function_id` | Direct match | Step-level callable context |
| Not applicable | `step_id` | Generated ATIF sequence counter | Not derived from IntermediateStep UUID |

## Name Mappings

| IntermediateStep | ATIF | Mapping Rule | Notes |
|---|---|---|---|
| `payload.name` (tool, function, or LLM by event type) | `tool_calls[*].function_name` or `model_name` | Context dependent | IntermediateStep name is polymorphic by event type |
| `function_ancestry.function_name` | `extra.tool_ancestry[*].function_name` | Direct match | Callable lineage node name |
| `function_ancestry.parent_name` | `extra.tool_ancestry[*].parent_name` | Direct match | Parent callable lineage node name |
| `function_ancestry.function_name` (step context) | `extra.ancestry.function_name` | Direct match | Step-level lineage name |

## Event-to-Step Mapping

- `WORKFLOW_START` maps to an ATIF user step (`source = "user"`).
- `LLM_END` starts an ATIF agent turn candidate step.
- `TOOL_END` and `FUNCTION_END` are accumulated into the pending ATIF step as observed invocations.
- `WORKFLOW_END` may emit a terminal ATIF agent step when final output is present and not redundant.
- `LLM_NEW_TOKEN` and other non-terminal chunk events are not directly emitted as standalone ATIF steps.

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
- ATIF timing is represented in:
  - `extra.invocation.start_timestamp` and `extra.invocation.end_timestamp` (step-level),
  - `extra.tool_invocations[*].start_timestamp` and `extra.tool_invocations[*].end_timestamp` (per invocation).

## Practical Validation Checklist

- Verify `tool_call_id == source_call_id == invocation_id` for each ATIF invocation row.
- Verify `tool_call_id == "call_" + payload.UUID` for mapped tool or function end events.
- Verify callable lineage consistency:
  - `function_ancestry.function_id <-> extra.tool_ancestry[*].function_id`
  - `function_ancestry.parent_id <-> extra.tool_ancestry[*].parent_id`
- Verify name consistency:
  - `function_ancestry.function_name <-> extra.tool_ancestry[*].function_name`
  - `function_ancestry.parent_name <-> extra.tool_ancestry[*].parent_name`

## Additional Identifiers Worth Tracking

- IntermediateStep structural parent linkage: `parent_id`
- Event semantics: `payload.event_type`
- Time surfaces: `event_timestamp`, `span_event_timestamp`, ATIF `timestamp`
- Session scope: ATIF `session_id`
- Framework or provider run IDs when present in metadata (for example, model framework trace IDs)
