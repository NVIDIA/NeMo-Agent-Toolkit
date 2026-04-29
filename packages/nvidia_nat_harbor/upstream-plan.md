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

This plan separates generic Harbor framework improvements from NeMo Agent Toolkit-specific
runtime and evaluator integration. The goal is to avoid long-lived Harbor fork
drift while keeping NeMo Agent Toolkit behavior inside `nvidia-nat-harbor` unless Harbor
already owns the abstraction.

## Principles

- Upstream behavior that is useful for any Harbor agent, verifier, or benchmark.
- Keep NeMo Agent Toolkit runtime contracts, evaluator dispatch, and sample coercion in
  `nvidia-nat-harbor`.
- Make local execution easy to use, but describe it as developer convenience,
  not benchmark isolation.
- Preserve shell-mode Harbor workflows while adding inline and library-mode paths.

## Terms

| Term | Meaning |
|---|---|
| Local environment | Host-local Harbor environment backend selected by `--env local` once accepted upstream. |
| Shell compatibility mode | Existing Harbor `nemo-agent` wrapper process path plus task verifier script. |
| Library mode | NeMo Agent Toolkit workflow execution in-process through the active Harbor Python. |
| Inline verifier | Harbor verifier object that scores in-process through a verifier import hook. |
| Script bridge | Compatibility path where the task verifier script invokes `nat_harbor.verifier.bridge_runner`. |

## Upstream Priority Table

| Priority | Upstream item | Why it matters | NeMo Agent Toolkit impact while missing |
|---|---|---|---|
| P0 | Generic verifier extension hook (`VerifierFactory`, `VerifierConfig.import_path`, `VerifierConfig.kwargs`, `--verifier-import-path`, `--verifier-kwarg`) | Allows NeMo Agent Toolkit to provide `ATIFInlineVerifier` without modifying Harbor verifier internals. | We must carry a Harbor side branch or patch Harbor installs to run inline verifier examples. |
| P1 | First-class `--env local` and generic local environment behavior | Removes the temporary `--env docker` enumeration workaround and makes host-local Harbor execution explicit. | README commands need `--env docker` plus `--environment-import-path ...LocalEnvironment`, which is confusing and easy to misread as Docker isolation. |
| P1 | Local install policy primitive for installed agents | Gives local runs a safe default that skips host mutation unless explicitly allowed. | `nvidia-nat-harbor` has to carry local install policy handling in its Harbor agent wrapper. |
| P2 | Generic agent run-mode extension point for installed agents | Lets Harbor support library and in-process execution without baking NeMo Agent Toolkit evaluator details into core. | `nvidia-nat-harbor` keeps overriding `nemo-agent` behavior to select shell mode vs library mode. |
| P2 | Harbor-owned `nemo-agent` library mode, if maintainers accept NeMo Agent Toolkit runtime ownership | Could make library mode a first-class Harbor `nemo-agent` feature instead of a package-local extension. | `DefaultNemoInlineRunner` remains in `nvidia-nat-harbor`, and Harbor `nemo-agent` stays shell-first. |
| P3 | Shared docs and examples for local environment and imported verifiers | Helps integrations outside NeMo Agent Toolkit use the same extension points. | `nvidia-nat-harbor` docs have to explain generic Harbor concepts alongside toolkit-specific setup. |

## Upstream Area 1: Local Environment

### Upstream To Harbor

- Add first-class `--env local` support in Harbor CLI type validation.
- Provide a generic local environment backend aligned with Harbor environment
  abstractions:
  - path mapping for `/app`, `/workspace`, `/logs`, and `/tests`
  - lifecycle setup and cleanup
  - upload and download behavior
  - preflight checks
  - clear non-sandbox semantics
- Add a generic local install policy primitive usable by installed agents:
  - `skip`: assume dependencies are already available on the host
  - `allow`: permit local installation during agent setup
- Optionally standardize policy metadata emission so integrations can record
  whether setup was skipped or executed.

### Keep In `nvidia-nat-harbor`

- NeMo Agent Toolkit-specific local run examples and README commands.
- Any temporary path-translation workaround needed before Harbor owns local environment behavior.

### Acceptance Criteria

- `harbor run --env local` works without enumeration workarounds.
- Local environment artifacts land under the trial directory with the same logical
  layout expected by existing agents and verifiers.
- Local environment docs state that host processes are executed directly and are
  not isolated by a sandbox.
- Existing shell-mode Harbor runs still work.

## Upstream Area 2: Verifier Extension Hook

### Upstream To Harbor

- Keep verifier creation behind a generic factory.
- Expose verifier extension wiring in config and CLI:
  - `VerifierConfig.import_path`
  - `VerifierConfig.kwargs`
  - `--verifier-import-path`
  - `--verifier-kwarg`
- Preserve the Harbor verifier contract for all verifier implementations:
  - verifier output directory is provided consistently
  - `reward.txt` is consumed consistently
  - structured reward and details files are allowed but optional
  - errors propagate as trial verifier failures
  - single-step and multi-step verification use the same factory path

### Landed Minimal Harbor Hook

- `VerifierFactory` in Harbor verifier runtime.
- `VerifierConfig.import_path` and `VerifierConfig.kwargs`.
- CLI flags:
  - `--verifier-import-path`
  - `--verifier-kwarg`
- Trial verification constructs verifiers through `VerifierFactory` in both
  single-step and multi-step paths.

### Keep In `nvidia-nat-harbor`

- `ATIFInlineVerifier`.
- NeMo Agent Toolkit evaluator dispatch:
  - `evaluate_atif_fn` resolution
  - NeMo Agent Toolkit config loading
  - evaluator-name selection
  - custom callable evaluator lane
  - ATIF sample coercion conventions

### Acceptance Criteria

- A non-Harbor verifier class can be loaded by import path and receive keyword arguments.
- Imported verifiers can write the same reward artifacts as script verifiers.
- Harbor result aggregation is identical for built-in, script, and imported
  verifiers.
- NeMo Agent Toolkit inline verifier examples do not require patching Harbor inside `.venv` or
  `site-packages`.

## Upstream Area 3: Agent Run Modes

The concrete NeMo Agent Toolkit inline runner is the boundary that needs the most care. Harbor
already owns `nemo-agent`, but in-process NeMo Agent Toolkit workflow execution depends on toolkit
runtime contracts. There are two acceptable upstream shapes:

1. Harbor owns a generic run-mode extension point, while `nvidia-nat-harbor`
   provides the concrete NeMo Agent Toolkit library-mode runner.
2. Harbor's `nemo-agent` owns library mode directly, if Harbor maintainers accept
   the NeMo Agent Toolkit runtime dependency and maintenance burden.

### Candidate Harbor Changes

- Generic agent run-mode selection and keyword argument plumbing.
- Local environment defaults for installed agents.
- Shell-mode fallback behavior for parity debugging.
- Quality-of-life improvements in Harbor `nemo-agent` that are not NeMo Agent Toolkit
  evaluator-specific:
  - local install policy wiring
  - clearer setup metadata
  - wrapper and runtime error reporting

### Keep In `nvidia-nat-harbor`

- `DefaultNemoInlineRunner` unless Harbor accepts direct ownership.
- NeMo Agent Toolkit workflow loading and invocation details.
- ATIF trajectory construction from NeMo Agent Toolkit intermediate steps.
- Inline environment overlay serialization for process-global `os.environ`.
- NeMo Agent Toolkit-specific integration tests and examples.

### Acceptance Criteria

- Shell mode remains available and backward compatible.
- Library mode works for local Harbor runs without invoking the wrapper
  child process.
- Concurrent inline trials cannot read another trial's temporary environment
  overlay.
- Missing trajectory or intermediate-step output still produces a verifier-readable
  artifact or an explicit failure.

## What Remains In `nvidia-nat-harbor`

- NeMo Agent Toolkit-backed Harbor agent wrapper and optional library-mode runner.
- Host-local NeMo Agent Toolkit examples and run books.
- ATIF inline verifier and script bridge compatibility path.
- NeMo Agent Toolkit evaluator dispatch and custom evaluator callable support.
- Simple calculator Harbor adapters and example E2E coverage.

These pieces stay out of Harbor core to preserve ownership boundaries and avoid
coupling Harbor to NeMo Agent Toolkit evaluator internals.

## PR Sequence

### Phase A: Harbor Core

1. Add first-class `--env local` and local environment implementation.
2. Add generic local install policy primitives and setup metadata.
3. Land generic verifier extension hooks. Current side-branch status:
   `VerifierFactory`, import-path wiring, and verifier keyword arguments are implemented.
4. Decide agent run-mode ownership:
   - generic run-mode extension point, or
   - direct Harbor `nemo-agent` library-mode support.

### Phase B: `nvidia-nat-harbor`

1. Consume Harbor local environment behavior directly once `--env local` lands.
2. Consume Harbor verifier import hook with `ATIFInlineVerifier`.
3. Keep `bridge_runner` as a script compatibility path only.
4. Keep NeMo Agent Toolkit evaluator dispatch and custom evaluator support in the package.
5. Document local environment behavior, shell mode, library mode, inline verifier, and script
   bridge as distinct modes.

### Phase C: Stabilization

Validate the full matrix before removing temporary workarounds:

- Shell compatibility mode with host Python wiring.
- Local environment path mapping and artifact layout.
- Library-mode inline agent execution.
- Inline verifier with builtin trajectory evaluator.
- Inline verifier with builtin tunable-rag evaluator.
- Inline verifier with custom callable evaluator.
- Script bridge compatibility through the task verifier script.
- Concurrent inline trials with serialized environment overlays.
- No regression in existing Harbor shell-mode workflows.

## Final Acceptance Criteria

- Harbor supports `--env local` natively.
- Harbor exposes generic verifier import hooks and stable verifier contracts.
- Harbor has either a generic agent run-mode extension point or direct
  `nemo-agent` library-mode ownership.
- `nvidia-nat-harbor` no longer carries generic Harbor workarounds.
- `nvidia-nat-harbor` still owns NeMo Agent Toolkit evaluator dispatch, ATIF sample coercion,
  custom callable evaluator support, and toolkit-specific examples.
- All stabilization matrix lanes pass without patching installed packages inside
  `.venv` or `site-packages`.
