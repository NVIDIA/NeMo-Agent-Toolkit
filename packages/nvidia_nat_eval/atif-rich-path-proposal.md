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
limitations und
-->

# ATIF Rich Path Proposal

## Context

For nested workflow behavior analysis, the current ATIF payload can require implementation-specific metadata to recover a full call tree (for example, internal nested calls such as `power_of_two -> calculator__multiply`).

The goal is to provide a richer, upstream-friendly structure that allows deterministic tree reconstruction from ATIF alone.

## Problem Statement

Current ATIF `Step.extra` has:

- `ancestry`: one step-level ancestry object
- `tool_ancestry`: one ancestry object per top-level tool call

This is often sufficient for top-level call structure, but can be insufficient for internal nested call structure when:

- multiple nested internal calls occur inside one tool execution
- nested internals do not map one-to-one to `tool_calls`
- downstream consumers need a canonical parent-child chain without runtime-specific event logs

## Goals

- Represent full nested ancestry in a vendor-neutral way
- Keep ATIF payload self-contained for offline analysis
- Minimize schema complexity for upstream acceptance

## Trajectory Use Requirements

The canonical trajectory must support all of the following uses without requiring
runtime-specific side channels:

1. Tree reconstruction for debugging
   - Render `workflow -> llm/tool -> nested tool/function` for one item.
2. Evaluator inputs
   - Provide enough structure for trajectory quality evaluators and step-level reasoning.
3. Profiler lineage and timing
   - Support call attribution, parent-child lineage, and timing-based analysis.
4. Training trajectory builders
   - Allow filtering/selecting subsets by function path and role.
5. Security and policy filters
   - Allow extraction/filtering of relevant calls and outputs by function path.
6. Deterministic offline analysis
   - Given one trajectory artifact, reconstruct execution path consistently.
7. Dual-publish migration
   - Serve as ATIF-native projection from IntermediateStep stream during migration.
8. ATIF-native observability export
   - Enable span tree reconstruction and exporter output directly from ATIF trajectory data.

## Final Trajectory Schema

Canonical lineage should be represented by parent-linked, time-ordered invocation
records in `tool_calls` (all observed invocations), with optional path helpers in
`Step.extra` for migration convenience.

Optional helper fields in `Step.extra`:

- `step_ancestry_path: list[InvocationNode] | None`
- `tool_ancestry_paths: list[list[InvocationNode]]`

### Semantics

- `step_ancestry_path`: ordered root-to-leaf path for the step context
- `tool_ancestry_paths[i]`: ordered root-to-leaf path for `tool_calls[i]`
- leaf should represent the deepest known callable context for that tool call

### Tool Invocation Semantics

To preserve runtime behavior comparable to `IntermediateStep`, `tool_calls` should
represent **all observed tool/function invocations** in a step, not only
LLM-selected top-level tool calls.

- Each observed invocation occurrence should produce one `tool_calls` entry.
- Repeated calls to the same tool/function should appear multiple times.
- `tool_ancestry_paths[i]` must align with `tool_calls[i]` and represent that
  invocation occurrence's root-to-leaf lineage.
- Ordering should be deterministic and based on invocation start time
  (`span_event_timestamp` preferred, `event_timestamp` fallback).

### Canonical Lineage Source

- Canonical source of lineage truth is `tool_calls` invocation records with:
  - stable invocation identity (`tool_call_id`)
  - parent-linked ancestry metadata
  - deterministic start-time ordering
- Path fields are optional, derived convenience views for consumers that prefer
  precomputed root-to-leaf chains.

### Canonicality rules

- `tool_calls` (all observed invocations) is the canonical tool-lineage surface.
- `tool_ancestry_paths` and `step_ancestry_path` are optional derived helpers.
- Legacy summary fields are optional and non-canonical.

### Why this shape is final

- Uses existing ancestry primitive (`InvocationNode`)
- Generic across runtimes/frameworks
- Easy for tree renderers and profilers to consume
- Avoids adding runtime-specific event objects to core schema

## Example

```json
{
  "step_ancestry_path": [
    {
      "function_id": "root",
      "function_name": "root"
    },
    {
      "function_id": "wf-1",
      "function_name": "<workflow>",
      "parent_id": "root",
      "parent_name": "root"
    },
    {
      "function_id": "llm-1",
      "function_name": "nvidia/nemotron-3-nano-30b-a3b",
      "parent_id": "wf-1",
      "parent_name": "<workflow>"
    }
  ],
  "tool_ancestry_paths": [
    [
      {
        "function_id": "root",
        "function_name": "root"
      },
      {
        "function_id": "wf-1",
        "function_name": "<workflow>",
        "parent_id": "root",
        "parent_name": "root"
      },
      {
        "function_id": "tool-1",
        "function_name": "power_of_two",
        "parent_id": "wf-1",
        "parent_name": "<workflow>"
      },
      {
        "function_id": "fn-1",
        "function_name": "calculator__multiply",
        "parent_id": "tool-1",
        "parent_name": "power_of_two"
      }
    ]
  ]
}
```

## Validation Invariants

If `tool_ancestry_paths` is present:

- Length must match `tool_calls` length
- Every path must be non-empty
- Parent-child consistency should hold within each path where IDs are provided
- Path ordering is root to leaf

If `step_ancestry_path` is present:

- Path must be non-empty
- Path ordering must be root to leaf

## Release Position

This release treats all-observed, parent-linked invocation records in
`tool_calls` as the target lineage model. Path fields may remain as transitional
helpers, but consumers should be able to reconstruct lineage directly from
invocation records.

## Consumer Impact

Tree/analysis tools can implement deterministic reconstruction:

1. Use `tool_calls` + aligned ancestry metadata as primary lineage input
2. Optionally use `tool_ancestry_paths` / `step_ancestry_path` when present
3. Derive legacy summary fields from invocation records or paths

This supports stable behavior for:

- execution tree scripts
- profiler lineage views
- trajectory debugging workflows

## Open Questions

- Should path nodes include optional `kind` (workflow/llm/tool/function) for easier rendering?
- Should leaf alignment with `tool_calls[i].function_name` be strict or best-effort?
- Is `step_ancestry_path` needed initially, or only `tool_ancestry_paths`?

## Next Steps

1. Socialize schema proposal upstream for feedback
2. Implement and validate all-observed invocation semantics for `tool_calls`
3. Keep path fields as optional derived outputs during migration
4. Update scripts/profilers/evaluators to consume invocation records first
5. Add tests for nested lineage reconstruction from invocation records (with and without paths)

