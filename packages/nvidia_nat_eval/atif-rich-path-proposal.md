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

# ATIF `Step.extra` Lineage Contract

## Purpose

This document explains the lineage fields implemented in
`packages/nvidia_nat_core/src/nat/data_models/atif/atif_step_extra.py` and
defines a concrete path to upstreaming this contract into `AtifStep`.

This is intended to be a shareable reference for:

- runtime and converter maintainers
- profiler and evaluator maintainers
- upstream ATIF schema reviewers

## Source of Truth

The data model is defined in:

- `AtifAncestry` in `atif_step_extra.py`
- `AtifStepExtra` in `atif_step_extra.py`

The converter currently emits this model into ATIF `Step.extra`, and downstream
tools consume it for tree reconstruction and lineage analysis.

## Sample Trace

For a visual example of nested lineage rendered via the NAT observability module, see:
`docs/source/_static/nat_nested_trace.png`.

This trace complements the sample artifact below and illustrates how a flat
`tool_calls` sequence can still represent nested ancestry via aligned lineage
metadata.

## Data Model

### `AtifAncestry`

`AtifAncestry` represents one lineage record.

- `function_ancestry: InvocationNode` (required)
  - Callable ancestry node for this record.

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
  - Runtime/framework label (for example, `langchain`).

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

## Canonical Contract

Canonical lineage should be reconstructed from:

- `tool_calls` (all observed invocation occurrences)
- `extra.ancestry`
- `extra.tool_ancestry`
- `extra.invocation` and `extra.tool_invocations` for timing metadata

## Producer Requirements

Producers should satisfy these requirements:

1. `tool_calls` is a flat list that includes all observed tool/function
   invocations, not only top-level model-selected calls.
2. `tool_ancestry[i]` corresponds to `tool_calls[i]`.
3. `tool_invocations[i]` corresponds to `tool_calls[i]` when timing metadata is emitted.
4. Invocation order is deterministic and based on start time.
5. Repeated calls to the same function are emitted as distinct invocation
   entries.

### Call Identity Mapping

For each invocation occurrence at index `i`:

- `tool_calls[i].tool_call_id` is the call instance identifier in `tool_calls`.
- `observation.results[*].source_call_id` should equal that `tool_call_id` for
  the corresponding observation payload.
- `tool_ancestry[i].function_ancestry.function_id` is the callable node identity
  for lineage (function/workflow node), not the call instance ID.

Practical relationship:

- `tool_call_id` and `source_call_id` identify one execution instance.
- `function_id` identifies which callable node executed.
- Multiple tool calls can share the same `function_id` while each keeps a unique
  `tool_call_id`/`source_call_id` pair.

Reference sample ATIF artifact:
`examples/evaluation_and_profiling/simple_calculator_eval/src/nat_simple_calculator_eval/data/workflow_output_atif.json`.

## Consumer Requirements

Consumers should implement lineage reads in this order:

1. Use `tool_calls` + `tool_ancestry` as the primary source.
2. Use `invocation` and `tool_invocations` for timing.
3. Use `ancestry` for step-level lineage context.

## Validation Invariants

### Canonical invariants

- `len(tool_ancestry)` matches `len(tool_calls)` for emitted invocation lineage.
- Index alignment is stable (`tool_calls[i]` maps to `tool_ancestry[i]`).
- If `tool_invocations` is emitted, `len(tool_invocations)` equals `len(tool_calls)`.
- If one of `start_timestamp` / `end_timestamp` is set, the other must also be set.

## Upstreaming Path to `AtifStep`

## Phase 1: Standardize lineage semantics

Upstream the canonical semantics first:

- step-level lineage record
- per-invocation lineage aligned with `tool_calls`
- deterministic ordering expectations

Do not add non-canonical convenience fields to the upstream contract.

## Phase 2: Transition compatibility

During transition:

- producers continue writing current `extra` fields
- consumers prioritize canonical reads during rollout

This phase avoids breaking existing artifacts and tools.

## Phase 3: Promote lineage to first-class `AtifStep` fields

Add typed lineage field(s) directly to `AtifStep` with wire-compatible mapping
from current `extra` payload.

## Phase 4: Stabilize canonical schema

After consumers are canonical:

- keep canonical lineage and invocation behavior unchanged

## Acceptance Criteria for Upstreaming

Upstreaming is complete when all of the following are true:

- deterministic lineage tree reconstruction from one ATIF artifact
- no runtime-specific side-channel dependency
- stable invocation mapping (`tool_calls[i]` to lineage record `i`)
- profiler and evaluator scenarios work with canonical invocation metadata only

## Recommended Implementation Guidance

- New consumers should implement canonical lineage reads immediately.
- Existing consumers should migrate to canonical invocation metadata reads.
- Producers should avoid introducing non-canonical fields.
