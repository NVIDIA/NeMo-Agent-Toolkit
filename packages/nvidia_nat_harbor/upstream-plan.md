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

# Harbor Upstream Plan for `nvidia-nat-harbor`

This document separates generic Harbor improvements from NAT-specific integration code.

## Scope and principles

- Upstream framework-agnostic improvements to Harbor core.
- Keep NAT runtime/evaluator-specific behavior in `nvidia-nat-harbor`.
- Avoid long-lived Harbor fork drift.

If a change is useful for any Harbor agent/benchmark, upstream it.
If a change depends on NAT runtime contracts, keep it in NAT.

---

## 1) Local env

### Upstream to Harbor

- Add first-class `--env local` support in Harbor CLI/type validation.
- Ensure local env behavior is aligned with Harbor environment abstractions (paths, lifecycle, preflight).
- Add a generic local install policy primitive (`skip`/`allow`) usable by installed agents.
- Optionally standardize policy metadata emission hooks.

### Why

- Local mode is generic Harbor capability, not NAT-specific.
- Improves usability and avoids custom env workarounds in integrations.

---

## 2) ATIF verifier mode

### Upstream to Harbor (generic parts)

- Define/document an optional ATIF-style verifier mode contract:
  - consume trajectory artifact
  - emit `reward.txt` / `reward.json` / `details.json`
- Provide reusable verifier utilities for:
  - reward/details writing
  - fallback handling
  - artifact path resolution across local/container layouts
- Add extension hooks for external verifier bridges/adapters.
- Add Harbor docs/template verifier script for ATIF-style integrations.

### Keep out of Harbor core

- NAT evaluator dispatch implementation itself (`evaluate_atif_fn`, NAT config loading, NAT evaluator selection logic).

### Why

- Harbor should provide bridge-friendly primitives.
- NAT-specific evaluator logic should stay in NAT package boundaries.

---

## 3) NAT agent runner updates

### Candidate Harbor upstreams

- Upstream inline runner support directly into Harbor `nemo-agent`.
- Default to library-mode execution for `nemo-agent` when `env=local`.
- Keep shell mode available behind explicit opt-out (`library_mode=false`) during migration.
- Generic `NemoAgent` quality-of-life improvements (policy wiring, run ergonomics, wrapper/runtime polish).

### Keep in NAT package

- NAT evaluator and bridge-specific behavior.
- NAT-specific docs/examples and integration tests outside Harbor core scope.

### Why

- `nemo-agent` is already Harbor-owned; local default behavior should live with that agent.
- Local-only default minimizes behavior changes in non-local environments.
- Temporary shell fallback reduces migration risk and supports parity debugging.

---

## 4) Parts that will remain in NAT

- ATIF bridge evaluator dispatch tied to NAT evaluator stack:
  - `evaluate_atif_fn` dispatch
  - NAT config/evaluator-name resolution
  - NAT sample coercion conventions
- NAT-specific adapters, examples, and runbooks.

These remain in `nvidia-nat-harbor` to preserve clear ownership and avoid Harbor core coupling to NAT internals.

---

## Execution plan (PR sequence)

### Phase A: Harbor

1. `--env local` support and local env polish.
2. Generic local install policy primitives.
3. Generic ATIF verifier-mode utilities and hooks.
4. Docs/template for ATIF-style verifier integrations.

### Phase B: `nvidia-nat-harbor`

1. Rebase NAT integration onto new Harbor primitives.
2. Remove duplicated generic glue now provided by Harbor.
3. Keep NAT-specific ATIF bridge logic in package.

### Phase C: stabilization

1. Validate matrix:
   - shell mode
   - local mode
   - library mode
   - ATIF bridge lanes
2. Lock contract boundaries in docs.

---

## Acceptance criteria

- Harbor supports `--env local` natively.
- Harbor `nemo-agent` defaults to library mode when `env=local`.
- Harbor exposes generic verifier utilities/hooks for ATIF-style adapters.
- NAT package no longer carries generic Harbor workarounds.
- NAT package still supports:
  - ATIF bridge evaluator dispatch
  - custom callable evaluator lane
- No regression in existing Harbor shell-mode workflows.

