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

This document defines the current conversion contract from NAT `IntermediateStep`
events to ATIF trajectories.

The conversion is intentionally split into two passes:

1. **Pass 1: build execution structure**
   - Determine root context ownership and delegated child context ownership.
   - Build subagent linkage metadata from explicit delegation markers.
2. **Pass 2: project contexts to ATIF**
   - Convert each context into ATIF steps using deterministic step window rules.
   - Emit flat tool call and observation rows with lineage metadata for nesting reconstruction.

The converter code implementing this model lives in:
`packages/nvidia_nat_core/src/nat/utils/atif_converter.py`, with key entry points:
`_pass1_build_execution_structure()` and `_pass2_project_context_to_steps()`.

## Pass 1: Execution Structure Rules

- Events are ordered by `event_timestamp`.
- Context ownership is computed from callable lineage and explicit delegation markers.
- Delegation boundaries are explicit only:
  - marker source: `metadata.provided_metadata["is_subagent_delegation"] == true`
  - no implicit delegation inference in the clean mapping path.
- Pass 1 outputs:
  - `root_events`
  - `child_events_by_session`
  - `subagent_ref_by_call_id`

## Pass 2: Context Projection Rules

- Each context is projected independently into ATIF steps.
- `WORKFLOW_START` can emit a `source="user"` step for the root context.
- Step windows are anchored by terminal events:
  - `LLM_END`
  - `TOOL_END`
  - `FUNCTION_END`
  - `WORKFLOW_END` (terminal output only when not redundant)
- Tool rows are flat by design:
  - nested tool calls are kept in `tool_calls`, `observation.results`,
    `extra.tool_ancestry`, and `extra.tool_invocations`
  - subscribers reconstruct nesting using ancestry `function_id` and `parent_id`.
- Chunk events (`LLM_NEW_TOKEN`, `SPAN_CHUNK`) do not emit standalone ATIF steps.

## Subagent Trajectory Rules

- Parent observation rows include `subagent_trajectory_ref` only for explicitly
  marked delegating calls.
- Child trajectories are built from delegated context events only.
- Embedded children are stored under:
  - `trajectory.extra["subagent_trajectories"]`
- Reference resolution precedence:
  1. `subagent_trajectory_ref.trajectory_path`
  2. `trajectory.extra.subagent_trajectories[session_id]`

## ID and Lineage Mapping

| IntermediateStep | ATIF | Mapping Rule | Notes |
|---|---|---|---|
| `payload.UUID` | `tool_calls[*].tool_call_id` | `tool_call_id = "call_" + payload.UUID` | Invocation occurrence identity |
| `payload.UUID` | `observation.results[*].source_call_id` | `source_call_id = tool_call_id` | Observation-to-call link |
| `payload.UUID` | `extra.tool_invocations[*].invocation_id` | `invocation_id = tool_call_id` | Invocation timing row identity |
| `function_ancestry.function_id` | `extra.tool_ancestry[*].function_id` | Direct match | Callable lineage node identity |
| `function_ancestry.parent_id` | `extra.tool_ancestry[*].parent_id` | Direct match | Lineage parent node identity |
| `function_ancestry.function_id` (step anchor) | `extra.ancestry.function_id` | Direct match | Step context identity |
| Not applicable | `step_id` | Generated sequentially from 1 | Not derived from UUID |

## Name Mapping

| IntermediateStep | ATIF | Mapping Rule | Notes |
|---|---|---|---|
| `payload.name` on `LLM_END` | `model_name` | Direct match | Model name for agent step |
| `payload.name` on `TOOL_END` or `FUNCTION_END` | `tool_calls[*].function_name` | Direct match | Invocation display name |
| `function_ancestry.function_name` | `extra.tool_ancestry[*].function_name` | Direct match | Callable lineage name |
| `function_ancestry.parent_name` | `extra.tool_ancestry[*].parent_name` | Direct match | Parent lineage name |

## Timing Mapping

- `span_event_timestamp` maps to start timestamp surfaces when present.
- `event_timestamp` maps to end timestamp surfaces.
- ATIF timing fields:
  - `extra.invocation.start_timestamp`
  - `extra.invocation.end_timestamp`
  - `extra.tool_invocations[*].start_timestamp`
  - `extra.tool_invocations[*].end_timestamp`

## Validation Checklist

- `tool_call_id == source_call_id == invocation_id` for each projected invocation row.
- `tool_call_id == "call_" + payload.UUID` for mapped tool and function end events.
- Lineage consistency:
  - `function_ancestry.function_id` equals `extra.tool_ancestry[*].function_id`
  - `function_ancestry.parent_id` equals `extra.tool_ancestry[*].parent_id`
- Step IDs are sequential and start from 1.
- Every emitted `subagent_trajectory_ref.session_id` resolves by precedence rules.
